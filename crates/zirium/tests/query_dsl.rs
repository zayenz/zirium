use std::collections::BTreeMap;
use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    query::{
        EvaluationLimits, EvaluationOptions, Query, QueryOutput, always, has_attr, input, op, ops,
    },
    semantic::{Document, LoweringMode, lower_with_dialect_registry},
};

const SOURCE: &str = r#"
"test.scope"() ({
  %c = "test.seed"() {tag = "start"} : () -> i32
  %a = "test.add"(%c, %c) : (i32, i32) -> i32
  "test.end"(%a) : (i32) -> ()
}) {sym_name = "first"} : () -> ()
"test.scope"() ({
  "test.idle"() : () -> ()
}) {sym_name = "second"} : () -> ()
"#;

fn document() -> Document {
    document_from(SOURCE)
}

fn document_from(source: &str) -> Document {
    let parsed =
        ParsedFile::parse_with_registry(source.as_bytes(), DialectRegistry::baseline()).unwrap();
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::baseline());
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    lowered.document.unwrap()
}

#[test]
fn reachable_treats_unknown_operations_as_leaves_unless_strict() {
    let source = r#"%seed = arith.constant 1 : i32
%value = "vendor.unknown"(%seed) : (i32) -> i32"#;
    let parsed =
        ParsedFile::parse_with_registry(source.as_bytes(), DialectRegistry::baseline()).unwrap();
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::baseline());
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let query = ops().filter(op("vendor.unknown")).reachable().names();
    assert_eq!(
        query
            .evaluate(
                &document,
                DialectRegistry::baseline(),
                EvaluationLimits::default()
            )
            .unwrap(),
        vec!["vendor.unknown"]
    );
    let error = query
        .evaluate_with_options(
            &document,
            DialectRegistry::baseline(),
            EvaluationOptions {
                strict_unknown_references: true,
            },
            EvaluationLimits::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("`vendor.unknown`"));
}

#[test]
fn structured_and_textual_queries_agree_on_stream_and_set_semantics() {
    let mut document = document();
    let seed = ops().filter(op("test.seed"));
    let cases = [
        (seed.users(), r#"filter(op("test.seed")) | users"#),
        (
            seed.users().unique(),
            r#"filter(op("test.seed")) | users | unique"#,
        ),
        (
            seed.users().union(&seed),
            r#"filter(op("test.seed")) | (users union filter(true))"#,
        ),
        (
            seed.users().difference(&seed.users()).union(&seed).users(),
            r#"filter(op("test.seed")) | ((users except users) union filter(true)) | users"#,
        ),
        (
            seed.fixpoint(&input().union(&input().users())),
            r#"filter(op("test.seed")) | fixpoint(filter(true) union users)"#,
        ),
    ];
    for (expression, source) in cases {
        let expected = document.query(&expression).unwrap();
        let mut actual = None;
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, output| {
                actual = Some(output);
                Ok(())
            })
            .unwrap();
        assert_eq!(actual, Some(QueryOutput::Operations(expected)), "{source}");
    }
    assert_eq!(document.query(&seed.users().count()).unwrap(), 2);
    assert_eq!(document.query(&seed.users().unique().count()).unwrap(), 1);
}

#[test]
fn document_context_reports_when_evaluation_does_not_supply_a_name() {
    for source in [r#"print("{document}")"#, r#"{"file": document}"#] {
        let query = Query::parse_with_document_context(source).unwrap();
        let mut document = document();
        let error = query
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains("unavailable"));
    }
}

#[test]
fn literal_interpolation_rejects_known_kinds_before_emitting() {
    for literal in [r#"{"x": "{a}"}"#, r#"{"{a}": 1}"#] {
        let source = format!(r#"a = filter(true); print("earlier"); {literal} | json"#);
        let query = Query::parse(&source).unwrap();
        let mut document = document();
        let mut emitted = Vec::new();
        let error = query
            .evaluate(&mut document, DialectRegistry::baseline(), |_, output| {
                emitted.push(output);
                Ok(())
            })
            .unwrap_err();
        assert!(emitted.is_empty(), "{source}: {emitted:?}");
        assert!(
            error
                .to_string()
                .contains("string interpolation requires a count or exactly one string")
        );
        let start = source.find(literal).unwrap();
        assert_eq!(error.range(), Some(start..start + literal.len()));
    }
}

#[test]
fn nested_queries_distinguish_relative_input_from_the_document() {
    let document = document();
    let scopes = ops().filter(op("test.scope"));
    let key = input().string_attr("sym_name").one();
    let counts: BTreeMap<String, usize> = document
        .query(&scopes.map_by(&key, &input().children().subtree().count()))
        .unwrap();
    assert_eq!(
        counts,
        BTreeMap::from([("first".into(), 3), ("second".into(), 1)])
    );
    let global = document
        .query(&scopes.map_by(&key, &ops().count()))
        .unwrap();
    assert_eq!(
        global,
        BTreeMap::from([("first".into(), 6), ("second".into(), 6)])
    );
    let histograms = document
        .query(&scopes.map_by(&key, &input().children().names().tally()))
        .unwrap();
    assert_eq!(histograms["first"]["test.add"], 1);
    let selections = document
        .query(&scopes.map_by(&key, &input().children()))
        .unwrap();
    assert_eq!(
        document.operation_name(selections["first"][0]),
        Some("test.seed")
    );
    let selected = document
        .query(
            &ops()
                .where_exists(&input().users().filter(op("test.add")))
                .names(),
        )
        .unwrap();
    assert_eq!(selected, ["test.seed"]);
    assert_eq!(
        document
            .query(
                &scopes
                    .sort_by(&input().children().count())
                    .string_attr("sym_name")
            )
            .unwrap(),
        ["second", "first"]
    );
}

#[test]
fn native_projections_and_reusable_expressions_follow_document_edits() {
    let mut document = document();
    let tagged = ops().filter(has_attr("tag"));
    let selection = document.query(&tagged).unwrap();
    let attributes = document.query(&tagged.attributes("tag")).unwrap();
    assert_eq!(
        document.attribute_spelling_value(attributes[0]),
        Some("\"start\"")
    );
    let types = document.query(&tagged.result_types()).unwrap();
    assert_eq!(document.type_spelling(types[0]), Some("i32"));
    assert_eq!(
        document
            .query(&tagged.users().operand_types().spellings())
            .unwrap(),
        ["i32", "i32", "i32", "i32"]
    );
    let mut edit = document.edit(DialectRegistry::baseline()).unwrap();
    edit.remove_attribute(selection[0], "tag").unwrap();
    edit.commit().unwrap();
    assert_eq!(document.query(&tagged.count()).unwrap(), 0);
    assert_eq!(document.operation_name(selection[0]), Some("test.seed"));
    assert_eq!(
        document
            .query(
                &ops()
                    .filter((op("test.seed") | op("test.add")) & !has_attr("tag"))
                    .count()
            )
            .unwrap(),
        2
    );
}

#[test]
fn cardinality_duplicate_keys_and_limits_report_errors() {
    let document = document();
    assert!(
        document
            .query(&ops().names().one())
            .unwrap_err()
            .to_string()
            .contains("got 6")
    );
    assert!(
        document
            .query(&ops().filter(always(false)).names().one())
            .is_err()
    );
    let duplicate = ops()
        .filter(op("test.seed"))
        .users()
        .map_by(&input().names().one(), &input().count());
    assert!(
        document
            .query(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate key")
    );
    let limits = EvaluationLimits {
        max_work: 20,
        max_items: 100,
    };
    assert!(
        ops()
            .fixpoint(&input().subtree())
            .evaluate(&document, DialectRegistry::baseline(), limits)
            .is_err()
    );
    let mut nested = input();
    for _ in 0..1000 {
        nested = input().where_exists(&nested);
    }
    assert!(
        document
            .query(&nested)
            .unwrap_err()
            .to_string()
            .contains("nesting limit")
    );
}

#[test]
fn flat_composition_does_not_consume_recursive_nesting() {
    let document = document();
    let mut predicate = always(false);
    let mut selection = ops().filter(always(false));
    let seed = ops().filter(op("test.seed"));
    let mut pipeline = ops();
    for _ in 0..200 {
        predicate = predicate | op("test.seed");
        selection = selection.union(&seed);
        pipeline = pipeline.filter(always(true));
    }
    assert_eq!(document.query(&ops().filter(predicate).count()).unwrap(), 1);
    assert_eq!(document.query(&selection.count()).unwrap(), 1);
    assert_eq!(document.query(&pipeline.count()).unwrap(), 6);
    assert_eq!(
        document.query(&selection.users().names()).unwrap(),
        ["test.add", "test.add"]
    );
}

#[test]
fn predicate_work_is_charged_per_operation_and_short_circuits() {
    let mut document = document();
    let mut all_false = always(false);
    let mut first_true = always(true);
    let mut first_false = always(false);
    for _ in 0..200 {
        all_false = all_false | always(false);
        first_true = first_true | always(false);
        first_false = first_false & always(true);
    }
    let limits = EvaluationLimits {
        max_work: 800,
        max_items: 100,
    };
    for query in [ops().filter(all_false.clone()), ops().root(all_false)] {
        let error = query
            .evaluate(&document, DialectRegistry::baseline(), limits)
            .unwrap_err();
        assert!(error.to_string().contains("work limit exceeded"));
    }
    for query in [ops().filter(first_true.clone()), ops().root(first_true)] {
        assert_eq!(
            query
                .evaluate(&document, DialectRegistry::baseline(), limits)
                .unwrap()
                .len(),
            6
        );
    }
    assert!(
        ops()
            .filter(first_false)
            .evaluate(&document, DialectRegistry::baseline(), limits)
            .unwrap()
            .is_empty()
    );

    // The CLI uses the same budget-aware predicate evaluator.
    let predicate = vec!["false"; 201].join(" or ");
    for stage in ["filter", "root"] {
        let query = Query::parse(&format!("{stage}({predicate})")).unwrap();
        let error = query
            .evaluate_with_limits(
                &mut document,
                DialectRegistry::baseline(),
                limits,
                |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("work limit exceeded"));
    }
}

#[test]
fn stream_stages_and_nested_queries_preserve_exact_work_allowances() {
    let mut document = document();
    let scopes = ops().filter(op("test.scope"));
    let nested = scopes.map_by(
        &input().string_attr("sym_name").one(),
        &input().children().count(),
    );
    for (query, allowance, expected) in [
        (
            ops().names().unique().tally(),
            27,
            BTreeMap::from([
                ("test.scope".to_owned(), 1),
                ("test.seed".to_owned(), 1),
                ("test.add".to_owned(), 1),
                ("test.end".to_owned(), 1),
                ("test.idle".to_owned(), 1),
            ]),
        ),
        (
            nested,
            40,
            BTreeMap::from([("first".to_owned(), 3), ("second".to_owned(), 1)]),
        ),
    ] {
        let limits = EvaluationLimits {
            max_work: allowance,
            max_items: 100,
        };
        assert_eq!(
            query
                .evaluate(&document, DialectRegistry::baseline(), limits)
                .unwrap(),
            expected
        );
        let error = query
            .evaluate(
                &document,
                DialectRegistry::baseline(),
                EvaluationLimits {
                    max_work: allowance - 1,
                    ..limits
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("work limit exceeded"));
    }
    for (source, allowance) in [
        ("names | unique | tally", 17),
        (
            r#"filter(op("test.scope")) | map_by(attr("sym_name"), children | count)"#,
            24,
        ),
    ] {
        let query = Query::parse(source).unwrap();
        let limits = EvaluationLimits {
            max_work: allowance,
            max_items: 100,
        };
        query
            .evaluate_with_limits(
                &mut document,
                DialectRegistry::baseline(),
                limits,
                |_, _| Ok(()),
            )
            .unwrap();
        let error = query
            .evaluate_with_limits(
                &mut document,
                DialectRegistry::baseline(),
                EvaluationLimits {
                    max_work: allowance - 1,
                    ..limits
                },
                |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("work limit exceeded"));
    }
}

#[test]
fn nested_root_queries_keep_the_frontend_nesting_boundary() {
    // The admitted maximum needs the main-sized stack also used by the baseline evaluator.
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let mut document = document_from(r#""test.op"() : () -> ()"#);
            let mut text = "root(true)".to_owned();
            let mut builder = input().root(always(true));
            for _ in 0..62 {
                text = format!("({text})");
                builder = input().where_exists(&builder);
            }
            assert_eq!(document.query(&builder).unwrap().len(), 1);
            Query::parse(&text)
                .unwrap()
                .evaluate(&mut document, DialectRegistry::baseline(), |_, output| {
                    let QueryOutput::Operations(selected) = output else {
                        panic!("root must retain operations")
                    };
                    assert_eq!(selected.len(), 1);
                    Ok(())
                })
                .unwrap();
            assert!(
                Query::parse(&format!("({text})"))
                    .unwrap_err()
                    .message
                    .contains("nesting limit")
            );
            assert!(
                document
                    .query(&input().where_exists(&builder))
                    .unwrap_err()
                    .to_string()
                    .contains("nesting limit")
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn explicit_program_allowance_bounds_binding_copies_with_located_errors() {
    let mut document = document();
    let query = Query::parse("selected = input; selected; selected").unwrap();
    let error = query
        .evaluate_with_program_limits(
            &mut document,
            DialectRegistry::baseline(),
            None,
            zirium::query::EvaluationOptions::default(),
            zirium::query::ProgramLimits {
                work: 10_000,
                results: 20,
                nesting: 64,
                stream_items: 100,
            },
            |_, _| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("result limit"));
    assert!(error.range().is_some());
}

#[test]
fn contextual_comprehensions_keep_order_and_native_typed_values() {
    use zirium::query::NativeValue;
    use zirql::ConstructedValue as V;
    let mut document = document_from("\"alpha\"() : () -> ()\n\"beta\"() : () -> ()");
    let query = Query::parse(r#"labels = names; [{"name": n, "label": "data:" ++ s} for n in labels for s in [n, n ++ "!"] if s != "beta!"]"#).unwrap();
    let mut result = None;
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            result = Some(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(report)) = result.unwrap() else {
        panic!("expected native constructed report")
    };
    assert_eq!(
        report.value(),
        &V::Array(vec![
            V::Record(vec![
                ("name".into(), V::String("alpha".into())),
                ("label".into(), V::String("data:alpha".into()))
            ]),
            V::Record(vec![
                ("name".into(), V::String("alpha".into())),
                ("label".into(), V::String("data:alpha!".into()))
            ]),
            V::Record(vec![
                ("name".into(), V::String("beta".into())),
                ("label".into(), V::String("data:beta".into()))
            ]),
        ])
    );
    assert_eq!(report.clone(), report);

    // Contextual words stay bindable, and quoted labels are static strings.
    let query = Query::parse(
        r#"for = names; if = names; [for]; [if]; [{"{n}": n, "text": "{n}", "first": for[1]} for n in if] | json"#,
    )
    .unwrap();
    let mut results = Vec::new();
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            results.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        results[0],
        QueryOutput::Array(vec![serde_json::json!(["alpha", "beta"])])
    );
    assert_eq!(results[0], results[1]);
    let QueryOutput::Json(json) = &results[2] else {
        panic!("expected JSON")
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(json).unwrap(),
        serde_json::json!([
            {"{n}": "alpha", "text": "{n}", "first": "alpha"},
            {"{n}": "beta", "text": "{n}", "first": "alpha"}
        ])
    );

    for (source, expected) in [
        (
            "if = count; [if-1 for n in [1]] | json",
            serde_json::json!([1]),
        ),
        (
            r#"[{"binder": if-1, "spaced": if - 1, "field": r.if-1, "operator_field": r.div-1} for if in [2] for r in [{"if": 2, "div": 2}]] | json"#,
            serde_json::json!([{"binder": 1, "spaced": 1, "field": 1, "operator_field": 1}]),
        ),
    ] {
        let query = Query::parse(source).unwrap();
        query
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                let QueryOutput::Json(json) = value else {
                    panic!("expected JSON")
                };
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&json).unwrap(),
                    expected
                );
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn comprehension_refusals_and_skipped_runtime_errors_keep_query_locations() {
    use zirium::query::NativeValue;
    use zirql::ConstructedValue as V;
    let mut document = document_from("\"alpha\"() : () -> ()");
    let query = Query::parse(
        r#"[[false and (1 div 0 == 1), true or {}.missing, [x * 2 for x in 1..n], - -1, -(-1), -9223372036854775808] for n in [2]]"#,
    )
    .unwrap();
    let mut result = None;
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            result = Some(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(report)) = result.unwrap() else {
        panic!("expected native value")
    };
    assert_eq!(
        report.value(),
        &V::Array(vec![V::Array(vec![
            V::Boolean(false),
            V::Boolean(true),
            V::Array(vec![V::Integer(2), V::Integer(4)]),
            V::Integer(1),
            V::Integer(1),
            V::Integer(i64::MIN)
        ])])
    );

    let source = r#"do set_attr("tag", "changed"); labels = names; [n for n in labels if false and (1 + "bad" == 0)]"#;
    let query = Query::parse(source).unwrap();
    let error = query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    let range = error.range().unwrap();
    assert_eq!(&source[range], r#"1 + "bad""#);
    assert_eq!(document.query(&ops().attributes("tag")).unwrap().len(), 0);

    for (source, responsible, message) in [
        ("[missing for n in [1]]", "missing", "undefined binding"),
        (
            "labels = names; [labels for labels in labels]",
            "labels for labels",
            "shadows",
        ),
        (
            "[9223372036854775807 + 1 for n in [1]]",
            "9223372036854775807 + 1",
            "overflow",
        ),
        (
            "[- -9223372036854775808 for n in [1]]",
            "- -9223372036854775808",
            "overflow",
        ),
        (
            "native = input; [n for n in native]",
            "native",
            "explicit scalar projection",
        ),
        ("a = [null]; [n for n in a]", "a", "null"),
    ] {
        let error = Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains(message), "{source}: {error}");
        let range = error.range().unwrap();
        assert!(
            source[range].contains(responsible.split(' ').next().unwrap()),
            "{source}"
        );
    }
    for source in [
        "[1e999 for n in [1]]",
        "[1e-999 for n in [1]]",
        "[unknown(n) for n in [1] if false]",
    ] {
        assert!(Query::parse(source).is_err(), "{source}");
    }
}

#[test]
fn comprehension_requests_share_parse_expansion_snapshot_and_render_allowances() {
    use zirium::query::{EvaluationOptions, ProgramLimits};
    let mut document = document_from("\"alpha\"() : () -> ()");
    let source = "[n for n in 1..100000 if false]";
    let query = Query::parse(source).unwrap();
    let limits = ProgramLimits {
        work: 10_000,
        results: 2_000,
        nesting: 64,
        stream_items: 1_000,
    };
    let error = query
        .evaluate_with_program_limits(
            &mut document,
            DialectRegistry::baseline(),
            None,
            EvaluationOptions::default(),
            limits,
            |_, _| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("limit exceeded"));
    assert_eq!(&source[error.range().unwrap()], "1..100000");

    let query = Query::parse("labels = names; [n for n in labels]").unwrap();
    let mut budget = zirql::Budget::for_program(ProgramLimits {
        work: 50_000,
        results: 10_000,
        ..limits
    });
    for _ in 0..2 {
        query
            .evaluate_with_budget(
                &mut document,
                DialectRegistry::baseline(),
                None,
                EvaluationOptions::default(),
                &mut budget,
                |document, output, budget| {
                    let value = output.to_json_with_budget(document, budget)?;
                    assert_eq!(value, serde_json::json!(["alpha"]));
                    Ok(())
                },
            )
            .unwrap();
    }
    let before = budget.remaining_work();
    assert!(before < 50_000);
    let mut small = zirql::Budget::for_program(ProgramLimits {
        work: 1,
        results: 10_000,
        ..limits
    });
    let error = query
        .evaluate_with_budget(
            &mut document,
            DialectRegistry::baseline(),
            None,
            EvaluationOptions::default(),
            &mut small,
            |_, _, _| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("work limit exceeded"));
    assert!(
        Query::parse_with_program_limits("[n for n in [1]]", ProgramLimits { work: 1, ..limits })
            .is_err()
    );
}

#[test]
fn library_emissions_and_committed_edits_survive_a_later_failure() {
    let mut document = document();
    let retained = document.query(&ops().filter(op("test.seed"))).unwrap();
    let source = r#"kept = filter(op("test.seed")); saved = kept | attr("tag"); do kept | set_attr("tag", "edited"); saved; kept | attr("tag") | emit | check(2)"#;
    let mut emitted = Vec::new();
    let error = Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap_err();
    assert!(error.to_string().contains("check failed"));
    assert_eq!(
        emitted,
        [
            QueryOutput::Values(vec!["start".to_owned()]),
            QueryOutput::Values(vec!["edited".to_owned()])
        ]
    );
    assert_eq!(document.operation_name(retained[0]), Some("test.seed"));
    assert_eq!(
        document
            .query(&ops().filter(op("test.seed")).string_attr("tag"))
            .unwrap(),
        ["edited"]
    );
}

const FUNCTION_REPORT_SOURCE: &str = include_str!("../../../examples/cli/function-report.mlir");
const FUNCTION_REPORT_PROGRAM: &str = include_str!("../../../examples/cli/function-report.zirql");
#[test]
fn correlated_function_report_agrees_through_text_and_typed_native_data() {
    use zirium::query::{
        ConstructedValue as V, NativeValue, ReportExpr as E, ReportGenerator as G,
        ValueBinaryOperator as B, bound_ops, op_expr, report,
    };
    let mut document = document_from(FUNCTION_REPORT_SOURCE);
    let query = Query::parse(FUNCTION_REPORT_PROGRAM).unwrap();
    let mut outputs = vec![];
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parsed)) = &outputs[0] else {
        panic!("native report data")
    };
    assert_eq!(outputs.len(), 1);
    let f = bound_ops("f");
    let rows = ops().filter(op("func.func")).project(
        "f",
        [
            ("name", E::from(f.attr("sym_name").one())),
            ("operations", E::from(f.children().subtree().count())),
            (
                "matches",
                E::from(
                    f.children()
                        .subtree()
                        .filter(op_expr(E::name("operation_name")))
                        .count(),
                ),
            ),
            ("kinds", E::from(f.children().subtree().names().unique())),
        ],
    );
    let selected = E::comprehension(
        E::name("r"),
        [G::in_values(
            "r",
            E::name("rows"),
            Some(
                E::name("r")
                    .field("operations")
                    .binary(B::GreaterEqual, E::name("threshold")),
            ),
        )],
    );
    let total = |field: &str| {
        E::comprehension(
            E::name("r").field(field),
            [G::in_values("r", E::name("selected"), None)],
        )
        .sum()
    };
    let typed = report(E::scope(
        [
            ("operation_name", E::string("arith.addi")),
            ("minimum_ops", E::integer(2)),
            (
                "threshold",
                E::require(
                    E::name("minimum_ops").binary(B::GreaterEqual, E::integer(0)),
                    E::name("minimum_ops"),
                    "minimum_ops must be nonnegative",
                ),
            ),
            ("rows", rows.into()),
            ("selected", selected),
        ],
        E::record([
            ("functions", E::name("selected")),
            ("total_operations", total("operations")),
            ("total_matches", total("matches")),
        ]),
    ));
    let constructed = document.query(&typed).unwrap();
    assert_eq!(constructed.value(), parsed.value());
    let row = |name: &str, operations, matches, kinds: Vec<&str>| {
        V::Record(vec![
            ("name".into(), V::String(name.into())),
            ("operations".into(), V::Integer(operations)),
            ("matches".into(), V::Integer(matches)),
            (
                "kinds".into(),
                V::Array(
                    kinds
                        .into_iter()
                        .map(|name| V::String(name.into()))
                        .collect(),
                ),
            ),
        ])
    };
    assert_eq!(
        parsed.value(),
        &V::Record(vec![
            (
                "functions".into(),
                V::Array(vec![
                    row("double", 2, 1, vec!["arith.addi", "func.return"]),
                    row("mix", 3, 1, vec!["arith.muli", "arith.addi", "func.return"])
                ])
            ),
            ("total_operations".into(), V::Integer(5)),
            ("total_matches".into(), V::Integer(2)),
        ])
    );
    let mut empty = vec![];
    Query::parse(&FUNCTION_REPORT_PROGRAM.replace("minimum_ops = 2", "minimum_ops = 4"))
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            empty.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(empty)) = &empty[0] else {
        panic!("empty native report")
    };
    assert_eq!(
        empty.value(),
        &V::Record(vec![
            ("functions".into(), V::Array(vec![])),
            ("total_operations".into(), V::Integer(0)),
            ("total_matches".into(), V::Integer(0))
        ])
    );
    let printed = |document: &Document| {
        let mut output = String::new();
        document
            .print(&mut output, zirium::printer::PrintLayout::Pretty)
            .unwrap();
        output
    };
    for source in [
        r#"filter(false) | project(f, {"x": missing})"#,
        r#"filter(false) | project(f, {"x": sum(f | names)})"#,
        r#"filter(false) | project(f, {"x": f | set_attr("tag", "x") | names})"#,
        r#"filter(false) | project(f, {"x": f | names | json})"#,
    ] {
        let before = printed(&document);
        let error = Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .unwrap_err();
        assert!(error.range().is_some());
        assert_eq!(printed(&document), before);
    }
    let error = Query::parse(
        r#"report require(false, filter(op("func.func")) | attr("absent") | one, "stop");"#,
    )
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
    .unwrap_err();
    assert_eq!(error.to_string(), "stop");
    for source in [
        r#"let = names; let"#,
        r#"report = names; report"#,
        r#"project = names; project"#,
        r#"sum = names; sum"#,
        r#"one = names; one"#,
    ] {
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .unwrap();
    }
    let reversed = ops()
        .filter(op("func.func"))
        .reverse()
        .project("f", [("name", E::from(f.attr("sym_name").one()))]);
    assert_eq!(
        document.query(&reversed).unwrap().value(),
        &V::Array(vec![
            V::Record(vec![("name".into(), V::String("plain".into()))]),
            V::Record(vec![("name".into(), V::String("mix".into()))]),
            V::Record(vec![("name".into(), V::String("double".into()))])
        ])
    );
    let mut scopes = document_from(
        r#"
"builtin.module"() ({ func.func @same() { func.return } }) {sym_name = "left"} : () -> ()
"builtin.module"() ({ func.func @same() { func.return } }) {sym_name = "right"} : () -> ()
"#,
    );
    let repeated = ops()
        .filter(op("func.func"))
        .project("f", [("name", E::from(f.attr("sym_name").one()))]);
    assert_eq!(
        scopes.query(&repeated).unwrap().value(),
        &V::Array(vec![
            V::Record(vec![("name".into(), V::String("same".into()))]),
            V::Record(vec![("name".into(), V::String("same".into()))]),
        ])
    );
    for (source, expected) in [
        (
            "let flag = false; report require(not flag, true, \"stop\");",
            V::Boolean(true),
        ),
        (
            "report [n # π\n for n in [1]];",
            V::Array(vec![V::Integer(1)]),
        ),
        (
            "[names for names in [1] if names == 1]",
            V::Array(vec![V::Integer(1)]),
        ),
        (
            "let operation_name = \"func.func\"; report filter(op(\"arith.addi\")) | root(op(operation_name)) | count;",
            V::Integer(2),
        ),
    ] {
        let mut result = vec![];
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                result.push(value);
                Ok(())
            })
            .unwrap();
        let QueryOutput::Native(NativeValue::Constructed(value)) = &result[0] else {
            panic!("native scalar result")
        };
        assert_eq!(value.value(), &expected);
    }
    let commented =
        "filter(op(\"func.func\")) | project # π\n(f, {\"name\": f | attr(\"sym_name\") | one});";
    let mut values = vec![];
    Query::parse(commented)
        .unwrap()
        .evaluate(&mut scopes, DialectRegistry::baseline(), |_, value| {
            values.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(value)) = &values[0] else {
        panic!("native projection")
    };
    assert_eq!(value.value(), scopes.query(&repeated).unwrap().value());
    let selector = input().attr("a".repeat(4_000)).one();
    let bounded = ops()
        .filter(op("func.func"))
        .project("f", [("n", E::from(f.sort_by(&selector).count()))]);
    let error = bounded
        .evaluate_with_program_limits(
            &document,
            DialectRegistry::baseline(),
            zirium::query::ProgramLimits {
                work: 1_000,
                results: 10_000,
                nesting: 64,
                stream_items: 100,
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("work limit"));
}
