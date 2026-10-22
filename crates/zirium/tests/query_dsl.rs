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
            seed.users().intersect(&seed.users()),
            r#"filter(op("test.seed")) | (users intersect users)"#,
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
    let names = seed.users().names();
    for (expression, source, expected) in [
        (
            names.union(&seed.names()),
            r#"filter(op("test.seed")) | (users | names union names)"#,
            vec!["test.add", "test.seed"],
        ),
        (
            names.intersect(&names),
            r#"filter(op("test.seed")) | (users | names intersect users | names)"#,
            vec!["test.add"],
        ),
        (
            names.difference(&seed.names()),
            r#"filter(op("test.seed")) | (users | names except names)"#,
            vec!["test.add"],
        ),
    ] {
        assert_eq!(document.query(&expression).unwrap(), expected);
        let mut actual = None;
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, output| {
                actual = Some(output);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            actual,
            Some(QueryOutput::Values(
                expected.into_iter().map(str::to_owned).collect()
            ))
        );
    }
    let source =
        r#"filter(op("test.seed")) union filter(op("test.add")) intersect filter(op("test.add"))"#;
    let mut actual = None;
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, output| {
            actual = Some(output);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        actual,
        Some(QueryOutput::Operations(
            document.query(&seed.users().unique()).unwrap()
        ))
    );
    let source = "(filter(false)) union (filter(false) | names)";
    let error = Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("same kind"));
    assert!(error.range().is_some());
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
    let counts: Vec<(String, usize)> = document
        .query(&scopes.map_by(&key, &input().children().subtree().count()))
        .unwrap();
    assert_eq!(
        counts,
        Vec::from([("first".into(), 3), ("second".into(), 1)])
    );
    let global = document
        .query(&scopes.map_by(&key, &ops().count()))
        .unwrap();
    assert_eq!(
        global,
        Vec::from([("first".into(), 6), ("second".into(), 6)])
    );
    let histograms = document
        .query(&scopes.map_by(&key, &input().children().names().tally()))
        .unwrap();
    assert_eq!(
        histograms[0]
            .1
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["test.seed", "test.add", "test.end"]
    );
    assert_eq!(
        histograms[0]
            .1
            .iter()
            .find(|(key, _)| key == "test.add")
            .unwrap()
            .1,
        1
    );
    let tied = scopes.reverse().map_by(&key, &ops().count());
    assert_eq!(
        document.query(&tied).unwrap(),
        [("second".into(), 6), ("first".into(), 6)]
    );
    assert_eq!(
        document
            .query(&tied.sort_by_value().reverse().sort_by_value())
            .unwrap(),
        [("first".into(), 6), ("second".into(), 6)]
    );
    assert_eq!(
        document.query(&tied.sort().head(1)).unwrap(),
        [("first".into(), 6)]
    );
    assert!(document.query(&tied.head(0)).unwrap().is_empty());
    assert_eq!(
        document
            .query(&tied.tail_expr(zirium::query::ReportExpr::integer(1)))
            .unwrap(),
        [("first".into(), 6)]
    );
    let native = document.query(&tied.erased()).unwrap();
    let mut budget = tied.new_budget(EvaluationLimits::default());
    let json = native.to_json_with_budget(&document, &mut budget).unwrap();
    assert_eq!(
        json.as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["second", "first"]
    );
    let selections = document
        .query(&scopes.map_by(&key, &input().children()))
        .unwrap();
    assert_eq!(
        document.operation_name(selections[0].1[0]),
        Some("test.seed")
    );
    let types = document
        .query(&scopes.map_by(&key, &input().children().result_types()))
        .unwrap();
    assert_eq!(document.type_spelling(types[0].1[0]), Some("i32"));
    let attributes = document
        .query(&scopes.map_by(&key, &input().attributes("sym_name")))
        .unwrap();
    assert_eq!(
        document.attribute_spelling_value(attributes[1].1[0]),
        Some("\"second\"")
    );
    let selected = document
        .query(
            &ops()
                .where_exists(&input().users().filter(op("test.add")))
                .names(),
        )
        .unwrap();
    assert_eq!(selected, ["test.seed"]);
    assert!(
        document
            .query(&ops().where_exists(&input().filter(always(false))))
            .unwrap()
            .is_empty()
    );
    let error = document
        .query(&ops().where_exists(&input().sort_by(&input().string_attr("missing").one())))
        .unwrap_err();
    assert!(error.to_string().contains("exactly one"));

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
fn predicate_boolean_chains_preserve_short_circuit_results() {
    let mut document = document();
    let mut all_false = always(false);
    let mut first_true = always(true);
    let mut first_false = always(false);
    for _ in 0..200 {
        all_false = all_false | always(false);
        first_true = first_true | always(false);
        first_false = first_false & always(true);
    }
    for query in [
        ops().filter(all_false.clone()),
        ops().root(all_false),
        ops().filter(first_false),
    ] {
        assert!(document.query(&query).unwrap().is_empty());
    }
    for query in [ops().filter(first_true.clone()), ops().root(first_true)] {
        assert_eq!(document.query(&query).unwrap().len(), 6);
    }
    let predicate = vec!["false"; 201].join(" or ");
    for stage in ["filter", "root"] {
        Query::parse(&format!("{stage}({predicate})"))
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, output| {
                assert!(matches!(output,QueryOutput::Operations(ids) if ids.is_empty()));
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn stream_stages_and_nested_queries_share_work_limits() {
    let mut document = document();
    let scopes = ops().filter(op("test.scope"));
    let nested = scopes.map_by(
        &input().string_attr("sym_name").one(),
        &input().children().count(),
    );
    for (query, expected) in [
        (
            ops().names().unique().tally(),
            Vec::from([
                ("test.scope".to_owned(), 1),
                ("test.seed".to_owned(), 1),
                ("test.add".to_owned(), 1),
                ("test.end".to_owned(), 1),
                ("test.idle".to_owned(), 1),
            ]),
        ),
        (
            nested,
            Vec::from([("first".to_owned(), 3), ("second".to_owned(), 1)]),
        ),
    ] {
        assert_eq!(
            query
                .evaluate(
                    &document,
                    DialectRegistry::baseline(),
                    EvaluationLimits::default(),
                )
                .unwrap(),
            expected
        );
        let error = query
            .evaluate(
                &document,
                DialectRegistry::baseline(),
                EvaluationLimits {
                    max_work: 10,
                    max_items: 100,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("work limit exceeded"));
    }
    for source in [
        "names | unique | tally",
        r#"filter(op("test.scope")) | map_by(attr("sym_name"), children | count)"#,
    ] {
        let query = Query::parse(source).unwrap();
        let limits = EvaluationLimits {
            max_work: 1_000,
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
                    max_work: 10,
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
            "if = count; [if, if, if] | json",
            serde_json::json!([2, 2, 2]),
        ),
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
fn native_reports_retain_operations_with_or_without_unused_functions() {
    use zirium::query::NativeValue;
    let mut document = document_from("\"alpha\"() : () -> ()\n\"beta\"() : () -> ()");
    let expected = document
        .root_operations()
        .iter()
        .map(|&operation| QueryOutput::Operations(vec![operation]))
        .collect::<Vec<_>>();
    let report = "report [n for n in input]; report [n for n in input] | json; report [n for n in (input | tail(1)) union (input | head(1))];";
    for prelude in ["", "fn id(x:int)->int{x} "] {
        let mut outputs = Vec::new();
        Query::parse(&format!("{prelude}{report}"))
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        assert_eq!(outputs.len(), 3);
        let QueryOutput::Native(NativeValue::Sequence(rows)) = &outputs[0] else {
            panic!("native operation report")
        };
        assert_eq!(rows.values(), expected);
        let QueryOutput::Json(text) = &outputs[1] else {
            panic!("explicit JSON report")
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(text).unwrap(),
            serde_json::json!([
                [{"name": "alpha", "attributes": {}, "operand_types": [], "result_types": []}],
                [{"name": "beta", "attributes": {}, "operand_types": [], "result_types": []}]
            ])
        );
        let QueryOutput::Native(NativeValue::Sequence(rows)) = &outputs[2] else {
            panic!("native union report")
        };
        assert_eq!(rows.values(), [expected[1].clone(), expected[0].clone()]);
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
        ("a = [null]; [n + 1 for n in a]", "n + 1", "numeric"),
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
    for source in ["[1e999 for n in [1]]", "[1e-999 for n in [1]]"] {
        assert!(Query::parse(source).is_err(), "{source}");
    }
    let source = r#"do set_attr("tag", "changed"); [unknown(n) for n in [1] if false]"#;
    let error = Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("undefined function: unknown"));
    assert_eq!(&source[error.range().unwrap()], "unknown");
    assert_eq!(document.query(&ops().attributes("tag")).unwrap().len(), 0);
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

#[test]
fn json_literal_snapshots_remain_saved_after_an_edit() {
    let mut document = document();
    let source = r#"kept = filter(op("test.seed")); saved = {"nodes": kept};
        do kept | set_attr("tag", "edited"); {"saved": saved, "current": kept}"#;
    let mut outputs = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(outputs.len(), 1);
    let QueryOutput::Map(entries) = &outputs[0] else {
        panic!("JSON object snapshot")
    };
    assert_eq!(entries[0].1["nodes"][0]["attributes"]["tag"], "\"start\"");
    assert_eq!(entries[1].1[0]["attributes"]["tag"], "\"edited\"");
}

#[test]
fn legacy_json_containers_use_the_active_located_nesting_limit() {
    let source = r#"{"nested": [0]}"#;
    let query = Query::parse(source).unwrap();
    let error = query
        .evaluate_with_program_limits(
            &mut document(),
            DialectRegistry::baseline(),
            None,
            EvaluationOptions::default(),
            zirium::query::ProgramLimits {
                work: 10_000,
                results: 10_000,
                stream_items: 1_000,
                nesting: 1,
            },
            |_, _| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("nesting limit"));
    assert_eq!(error.range(), Some(0..source.len()));
}

const FUNCTION_REPORT_SOURCE: &str = include_str!("../../../examples/cli/function-report.mlir");
const FUNCTION_REPORT_PROGRAM: &str = include_str!("../../../examples/cli/function-report.zirql");
#[test]
fn function_groups_agree_through_text_and_rust_native_values() {
    use zirium::query::{ConstructedValue as V, NativeValue, ReportExpr as E, bound_ops, report};
    let mut document = document_from(FUNCTION_REPORT_SOURCE);
    let program = include_str!("../../../examples/cli/function-groups.zirql");
    // Inspect the native value before the saved program's explicit JSON boundary.
    let native_program = format!("{};", program.trim_end().strip_suffix(" | json;").unwrap());
    let query = Query::parse(&native_program).unwrap();
    let mut outputs = vec![];
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parsed)) = &outputs[0] else {
        panic!("native groups")
    };
    let row = |name: &str, matches| {
        V::Record(vec![
            ("name".into(), V::String(name.into())),
            ("matches".into(), V::Integer(matches)),
        ])
    };
    assert_eq!(
        parsed.value(),
        &V::Array(vec![
            V::Record(vec![
                ("key".into(), V::Integer(1)),
                (
                    "items".into(),
                    V::Array(vec![row("double", 1), row("mix", 1)])
                )
            ]),
            V::Record(vec![
                ("key".into(), V::Integer(0)),
                ("items".into(), V::Array(vec![row("plain", 0)]))
            ]),
        ])
    );
    let f = bound_ops("f");
    let rows = ops().filter(op("func.func")).project(
        "f",
        [
            ("name", E::from(f.attr("sym_name").one())),
            (
                "matches",
                E::from(f.children().subtree().filter(op("arith.addi")).count()),
            ),
        ],
    );
    let typed = document
        .query(&report(E::from(rows).group_by("matches")))
        .unwrap();
    assert_eq!(typed.value(), parsed.value());
}
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
    let bindings = vec![
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
    ];
    let result = E::record([
        ("functions", E::name("selected")),
        ("total_operations", total("operations")),
        ("total_matches", total("matches")),
    ]);
    let typed = report(E::scope(bindings.clone(), result.clone()));

    let constructed = document.query(&typed).unwrap();
    assert_eq!(constructed.value(), parsed.value());
    let parameter_query = Query::parse(include_str!(
        "../../../examples/cli/function-report-parameters.zirql"
    ))
    .unwrap();
    let parameters = vec![
        ("operation_name".into(), V::String("arith.addi".into())),
        ("minimum_ops".into(), V::Integer(2)),
    ];
    let mut parameter_outputs = vec![];
    parameter_query
        .evaluate_with_parameters(
            &mut document,
            DialectRegistry::baseline(),
            &parameters,
            |_, value| {
                parameter_outputs.push(value);
                Ok(())
            },
        )
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parameter_value)) = &parameter_outputs[0]
    else {
        panic!("native parameter report")
    };
    assert_eq!(parameter_outputs.len(), 1);
    assert_eq!(parameter_value.value(), parsed.value());
    let parameter_builder = report(E::scope(bindings.into_iter().skip(2), result))
        .with_parameters(parameter_query.parameters().to_vec());
    let parameter_result = parameter_builder
        .evaluate_with_parameters(
            &document,
            DialectRegistry::baseline(),
            &parameters,
            EvaluationLimits::default(),
        )
        .unwrap();
    assert_eq!(parameter_result.value(), parsed.value());
    let erased = parameter_builder.erased();
    assert_eq!(erased.parameters(), parameter_query.parameters());
    let QueryOutput::Native(NativeValue::Constructed(erased_value)) = erased
        .evaluate_with_parameters(
            &document,
            DialectRegistry::baseline(),
            &parameters,
            EvaluationLimits::default(),
        )
        .unwrap()
    else {
        panic!("erased native report")
    };
    assert_eq!(erased_value.value(), parsed.value());
    let nested = report(E::from(parameter_builder.clone()));
    assert_eq!(nested.parameters(), parameter_query.parameters());
    assert_eq!(
        nested
            .evaluate_with_parameters(
                &document,
                DialectRegistry::baseline(),
                &parameters,
                EvaluationLimits::default()
            )
            .unwrap()
            .value(),
        parsed.value()
    );

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
    assert!(error.to_string().contains("work limit"), "{error}");
}

#[test]
fn parameter_arguments_keep_ordered_scope_and_preflight() {
    use zirium::query::{ConstructedValue as V, NativeValue};
    let mut document = document();
    let parameters = vec![("factor".into(), V::Integer(2))];
    let source = "report [factor for factor in [1]]; param factor: int; report factor;";
    let mut outputs = vec![];
    Query::parse(source)
        .unwrap()
        .evaluate_with_parameters(
            &mut document,
            DialectRegistry::baseline(),
            &parameters,
            |_, value| {
                outputs.push(value);
                Ok(())
            },
        )
        .unwrap();
    assert!(
        matches!(&outputs[0], QueryOutput::Native(NativeValue::Constructed(v)) if v.value()==&V::Array(vec![V::Integer(1)]))
    );
    assert!(
        matches!(&outputs[1], QueryOutput::Native(NativeValue::Constructed(v)) if v.value()==&V::Integer(2))
    );
    for source in [
        "report factor; param factor: int; report factor;",
        "param factor: int; report [factor for factor in [1]];",
        "param document: string; names",
        "param factor: int; factor = count; names",
    ] {
        assert!(
            Query::parse(source)
                .map(|query| query
                    .evaluate_with_parameters(
                        &mut document,
                        DialectRegistry::baseline(),
                        &parameters,
                        |_, _| Ok(())
                    )
                    .is_err())
                .unwrap_or(true),
            "{source}"
        );
    }
    let mut output = vec![];
    Query::parse("param kind: string; param n: int; filter(op(kind)) | head(n) | names")
        .unwrap()
        .evaluate_with_parameters(
            &mut document,
            DialectRegistry::baseline(),
            &[
                ("kind".into(), V::String("test.seed".into())),
                ("n".into(), V::Integer(1)),
            ],
            |_, value| {
                output.push(value);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(output, vec![QueryOutput::Values(vec!["test.seed".into()])]);
    let invalid = "param n: int; filter(false) | filter(op(n)) | names";
    let error = Query::parse(invalid)
        .unwrap()
        .evaluate_with_parameters(
            &mut document,
            DialectRegistry::baseline(),
            &[("n".into(), V::Integer(1))],
            |_, _| panic!("preflight before output"),
        )
        .unwrap_err();
    assert_eq!(&invalid[error.range().unwrap()], "n");
}

const NATIVE_CAPTURE_PROGRAM: &str = include_str!("../../../examples/cli/native-captures.zirql");
const NATIVE_CAPTURE_MATCH: &str = r#"match { from add in (input | filter(op("arith.addi"))); bind lhs = operand(add, 0); bind rhs = operand(add, 1); where same_value(lhs, rhs); }"#;
fn same_ssa_matches() -> zirium::query::CaptureQuery {
    use zirium::query::{bound_ops, bound_ssa, matches, same_value};
    matches()
        .from("add", &ops().filter(op("arith.addi")))
        .bind("lhs", &bound_ops("add").operand(0))
        .bind("rhs", &bound_ops("add").operand(1))
        .where_(&same_value(&bound_ssa("lhs"), &bound_ssa("rhs")))
        .build()
}

#[test]
fn exact_ssa_captures_agree_in_parsed_typed_nested_and_erased_reports() {
    use zirium::{
        query::{ConstructedValue as V, NativeValue, ReportExpr as E, capture_ops, report},
        semantic::{ValueId, ValueReference},
    };
    let mut document = document_from(FUNCTION_REPORT_SOURCE);
    let captures = same_ssa_matches();
    let native = document.query(&captures).unwrap();
    assert_eq!(native.len(), 1);
    let record = &native.rows()[0];
    assert_eq!(
        record
            .fields
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["add", "lhs", "rhs"]
    );
    let QueryOutput::Operations(add) = &record.fields[0].1 else {
        panic!("native operation")
    };
    assert_eq!(add.len(), 1);
    let QueryOutput::Native(NativeValue::Ssa(lhs)) = record.fields[1].1 else {
        panic!("exact SSA")
    };
    let QueryOutput::Native(NativeValue::Ssa(rhs)) = record.fields[2].1 else {
        panic!("exact SSA")
    };
    assert_eq!(lhs, rhs);
    assert!(matches!(lhs, ValueId::BlockArgument { argument: 0, .. }));
    assert_eq!(
        document.operands(add[0]).unwrap(),
        [ValueReference::Resolved(lhs), ValueReference::Resolved(rhs)]
    );
    document.check_value(lhs).unwrap();
    let mut outputs = vec![];
    Query::parse(NATIVE_CAPTURE_MATCH)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Captures(parsed)) = &outputs[0] else {
        panic!("native parsed rows")
    };
    assert_eq!(parsed.rows(), native.rows());
    assert_eq!(
        document.query(&captures.erased()).unwrap(),
        QueryOutput::Native(NativeValue::Captures(native.clone()))
    );
    let add = capture_ops("m", "add");
    let rows = captures.project(
        "m",
        [
            (
                "function",
                E::from(add.root(op("func.func")).attr("sym_name").one()),
            ),
            ("operation", E::from(add.names().one())),
        ],
    );
    let typed = report(E::scope(
        [("rows", rows.into())],
        E::record([
            ("matches", E::name("rows")),
            ("count", E::name("rows").length()),
        ]),
    ));
    let expected = V::Record(vec![
        (
            "matches".into(),
            V::Array(vec![V::Record(vec![
                ("function".into(), V::String("double".into())),
                ("operation".into(), V::String("arith.addi".into())),
            ])]),
        ),
        ("count".into(), V::Integer(1)),
    ]);
    assert_eq!(document.query(&typed).unwrap().value(), &expected);
    assert_eq!(
        document.query(&typed.clone().erased()).unwrap(),
        document.query(&typed.erased()).unwrap()
    );
    outputs.clear();
    Query::parse(NATIVE_CAPTURE_PROGRAM)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parsed)) = &outputs[0] else {
        panic!("native scalar report")
    };
    assert_eq!(parsed.value(), &expected);
    // The new field wrapper also retains parameter declarations through projection.
    let parameterized = captures.with_parameters(vec![zirium::query::ParameterDeclaration {
        name: zirium::query::Located::new("label".into(), 0..0),
        ty: zirium::query::Located::new(zirium::query::ParameterType::String, 0..0),
    }]);
    let projected = parameterized.project(
        "m",
        [("name", E::from(capture_ops("m", "add").names().one()))],
    );
    assert_eq!(projected.parameters(), parameterized.parameters());
    assert!(
        projected
            .erased()
            .evaluate_with_parameters(
                &document,
                DialectRegistry::baseline(),
                &[("label".into(), V::String("test".into()))],
                EvaluationLimits::default()
            )
            .is_ok()
    );
}

#[test]
fn same_value_distinguishes_two_results_with_one_owning_operation() {
    use zirium::semantic::{ValueId, ValueReference};
    let source = r#"func.func @different() -> i32 {
      %left, %right = "test.pair"() : () -> (i32, i32)
      %sum = arith.addi %left, %right : i32
      func.return %sum : i32
    }"#;
    let document = document_from(source);
    let add = document.query(&ops().filter(op("arith.addi"))).unwrap()[0];
    let [
        ValueReference::Resolved(ValueId::OperationResult {
            operation: left,
            result: 0,
        }),
        ValueReference::Resolved(ValueId::OperationResult {
            operation: right,
            result: 1,
        }),
    ] = document.operands(add).unwrap()
    else {
        panic!("two native result positions")
    };
    assert_eq!(left, right);
    assert_ne!(
        document
            .query(&ops().filter(op("arith.addi")).operand(0))
            .unwrap(),
        document
            .query(&ops().filter(op("arith.addi")).operand(1))
            .unwrap()
    );
    assert!(document.query(&same_ssa_matches()).unwrap().is_empty());
}

#[test]
fn native_capture_scope_empty_preflight_and_required_operands_share_limits() {
    use zirium::query::{
        NativeValue, ProgramLimits, ReportExpr as E, bound_ops, bound_ssa, capture_ops, matches,
        same_value,
    };
    let mut document = document_from(FUNCTION_REPORT_SOURCE);
    let ordered = matches()
        .from("outer", &ops().filter(op("arith.addi")))
        .from("inner", &bound_ops("outer"))
        .build();
    let rows = document.query(&ordered).unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows.rows() {
        assert_eq!(row.fields[0].1, row.fields[1].1);
    }
    for source in [
        "match { from x in x; }",
        "match { from x in input; bind x = operand(x, 0); }",
        "match { from x in (input | filter(false)); where same_value(x,x); }",
        "match { from x in (input | filter(false)); bind value = operand((input | count),0); }",
        "match { from x in (input | filter(false)); where false; bind value = operand((input | set_attr(\"tag\",\"x\")),0); }",
        "match { from x in (input | filter(false)); } | project(m, {\"x\": m.missing | names})",
        "match { from x in (input | filter(false)); bind lhs = operand(x,0); } | project(m, {\"x\": m.lhs | count})",
        "let rows = match { from x in (input | filter(false)); }; rows.x | names",
    ] {
        let query = Query::parse(source).unwrap();
        let mut outputs = vec![];
        assert!(
            query
                .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                    outputs.push(value);
                    Ok(())
                })
                .is_err(),
            "{source}"
        );
        assert!(outputs.is_empty());
    }
    let unavailable = ops().filter(op("func.return")).head(1).operand(1);
    assert!(
        document
            .query(&unavailable)
            .unwrap_err()
            .to_string()
            .contains("unavailable")
    );
    let empty = matches()
        .from("add", &ops().filter(op("absent")))
        .bind("lhs", &bound_ops("add").operand(0))
        .build();
    assert!(
        document
            .query(&empty.project(
                "m",
                [("name", E::from(capture_ops("m", "add").names().one()))]
            ))
            .unwrap()
            .value()
            == &zirium::query::ConstructedValue::Array(vec![])
    );
    assert!(
        document
            .query(&empty.project(
                "m",
                [("wrong", E::from(capture_ops("m", "lhs").names().one()))]
            ))
            .is_err()
    );
    let filtered = matches()
        .from("add", &ops().filter(op("arith.addi")))
        .from("other", &input())
        .bind("lhs", &bound_ops("add").operand(0))
        .where_(&!same_value(&bound_ssa("lhs"), &bound_ssa("lhs")))
        .build();
    assert!(document.query(&filtered).unwrap().is_empty());
    assert!(
        filtered
            .evaluate_with_program_limits(
                &document,
                DialectRegistry::baseline(),
                ProgramLimits {
                    work: 500,
                    results: 100_000,
                    nesting: 64,
                    stream_items: 1_000
                }
            )
            .unwrap_err()
            .to_string()
            .contains("work limit")
    );
    let mut legacy = zirql::Budget::new(zirql::Limits {
        work: 100_000,
        items: 1_000,
        nesting: 64,
    });
    assert!(
        same_ssa_matches()
            .evaluate_output_with_budget(
                &document,
                DialectRegistry::baseline(),
                EvaluationOptions::default(),
                &mut legacy
            )
            .is_err()
    );
    let native = document.query(&same_ssa_matches()).unwrap();
    let output = QueryOutput::Native(NativeValue::Captures(native));
    let mut scarce = zirql::Budget::for_program(ProgramLimits {
        work: 100_000,
        results: 2,
        nesting: 64,
        stream_items: 1_000,
    });
    assert!(output.to_json_with_budget(&document, &mut scarce).is_err());
}

#[test]
fn bounded_users_paths_keep_native_steps_and_share_projection_parameters_and_limits() {
    use zirium::query::{NativeValue, PathEdge, ReportExpr as E, capture_ops, op_expr};
    let document = document_from(FUNCTION_REPORT_SOURCE);
    let query = ops()
        .filter(op("arith.muli"))
        .paths(PathEdge::Users, op("func.return"), 2);
    let result = document.query(&query).unwrap();
    assert_eq!(result.len(), 1);
    let fields = &result.rows()[0].fields;
    let QueryOutput::Operations(source) = &fields[0].1 else {
        panic!("source")
    };
    let QueryOutput::Operations(sink) = &fields[1].1 else {
        panic!("sink")
    };
    let QueryOutput::Native(NativeValue::OperationSequence(steps)) = &fields[2].1 else {
        panic!("native sequence")
    };
    assert_eq!(source, &steps[..1]);
    assert_eq!(sink, &steps[2..]);
    assert_eq!(
        steps
            .iter()
            .map(|&id| document.operation_name(id).unwrap())
            .collect::<Vec<_>>(),
        ["arith.muli", "arith.addi", "func.return"]
    );
    for &id in steps {
        document.check_operation(id).unwrap();
        let range = document.operation_source_range(id).unwrap().as_range();
        assert!(FUNCTION_REPORT_SOURCE[range].contains(document.operation_name(id).unwrap()));
    }
    let zero = ops()
        .filter(op("arith.muli"))
        .paths(PathEdge::Users, op("arith.muli"), 0);
    let zero = document.query(&zero).unwrap();
    assert!(
        matches!(&zero.rows()[0].fields[2].1, QueryOutput::Native(NativeValue::OperationSequence(ids)) if ids==source)
    );
    assert!(
        document
            .query(
                &ops()
                    .filter(op("arith.muli"))
                    .paths(PathEdge::Users, op("func.return"), 1)
            )
            .unwrap()
            .is_empty()
    );
    let projected = query.project("p", [("steps", E::from(capture_ops("p", "steps").names()))]);
    let report = document.query(&projected).unwrap();
    assert_eq!(
        report.value(),
        &zirql::ConstructedValue::Array(vec![zirql::ConstructedValue::Record(vec![(
            "steps".into(),
            zirql::ConstructedValue::Array(
                ["arith.muli", "arith.addi", "func.return"]
                    .into_iter()
                    .map(|s| zirql::ConstructedValue::String(s.into()))
                    .collect()
            )
        )])])
    );
    let parameterized = ops()
        .filter(op("arith.muli"))
        .paths(PathEdge::Users, op_expr(E::name("target")), 2)
        .with_parameters(vec![zirql::ParameterDeclaration {
            name: zirql::Located::new("target".into(), 0..0),
            ty: zirql::Located::new(zirql::ParameterType::String, 0..0),
        }]);
    assert_eq!(
        parameterized
            .evaluate_with_parameters(
                &document,
                DialectRegistry::baseline(),
                &[(
                    "target".into(),
                    zirql::ConstructedValue::String("func.return".into())
                )],
                EvaluationLimits::default()
            )
            .unwrap()
            .len(),
        1
    );
    let mut limited = zirql::Budget::for_program(zirql::ProgramLimits {
        work: 100_000,
        results: 20,
        nesting: 64,
        stream_items: 1000,
    });
    assert!(
        query
            .evaluate_output_with_budget(
                &document,
                DialectRegistry::baseline(),
                EvaluationOptions::default(),
                &mut limited
            )
            .is_err()
    );
    assert!(
        document
            .query(
                &ops()
                    .filter(op("missing"))
                    .paths(PathEdge::Users, op("func.return"), 2)
                    .project("p", [("bad", E::from(capture_ops("p", "missing").names()))])
            )
            .is_err()
    );
}

#[test]
fn optional_reports_agree_in_parsed_and_typed_native_values() {
    use zirium::query::{
        ConstructedValue as V, NativeValue, ReportExpr as E, ValueBinaryOperator as B, report,
    };
    let mut document = document_from(FUNCTION_REPORT_SOURCE);
    let source = include_str!("../../../examples/cli/optional-report.zirql");
    let mut outputs = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parsed)) = &outputs[0] else {
        panic!("native report")
    };
    let info = E::record([("id", E::integer(4)), ("stored", E::null())]);
    let cells = E::array([E::integer(10), E::null(), E::integer(30)]);
    let typed = report(E::record([
        (
            "id",
            info.clone().lookup(E::string("id")).default(E::integer(0)),
        ),
        (
            "omitted",
            info.clone().lookup(E::string("missing")).absent(),
        ),
        ("stored", info.clone().has_key(E::string("stored"))),
        ("missing", info.clone().has_key(E::string("missing"))),
        ("stored_cell", cells.clone().has_key(E::integer(2))),
        ("missing_cell", cells.clone().has_key(E::integer(4))),
        (
            "fallback",
            cells.clone().lookup(E::integer(4)).default(E::integer(8)),
        ),
        ("zero", E::integer(0).default(E::integer(99))),
        ("flag", E::boolean(false).default(E::boolean(true))),
        ("empty", E::array([]).default(E::array([E::integer(1)]))),
        (
            "skip",
            E::integer(7).default(E::integer(1).binary(B::IntegerDivide, E::integer(0))),
        ),
        ("native", E::from(ops().filter(op("arith.addi")).count())),
        ("info", info),
        ("cells", cells),
    ]));
    let actual = document.query(&typed).unwrap();
    assert_eq!(actual.value(), parsed.value());
    let V::Record(fields) = actual.value() else {
        panic!("record")
    };
    assert_eq!(fields[1].1, V::Boolean(true));
    assert_eq!(fields[3].1, V::Boolean(false));
    assert_eq!(fields[6].1, V::Integer(8));
    assert_eq!(fields[11].1, V::Integer(2));
    assert_eq!(
        document.query(&report(E::null().absent())).unwrap().value(),
        &V::Boolean(true)
    );
    assert_eq!(
        document
            .query(&report(E::boolean(false).occurs()))
            .unwrap()
            .value(),
        &V::Boolean(true)
    );
    for source in [
        r#"a={"id":4}; report 7 default a;"#,
        "a=[1]; report 7 default a;",
    ] {
        assert!(
            Query::parse(source)
                .unwrap()
                .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
                .is_err()
        );
    }
    // Call names retain their legacy binding meaning outside call syntax.
    assert!(
        Query::parse("lookup=names; default=lookup; default")
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .is_ok()
    );
    assert!(
        Query::parse("[] default [1]")
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .is_ok()
    );
    // Parameter metadata on a skipped fallback remains part of the request.
    let parameter = report(E::integer(7).default(E::name("fallback"))).with_parameters(vec![
        zirium::query::ParameterDeclaration {
            name: zirium::query::Located::new("fallback".into(), 0..0),
            ty: zirium::query::Located::new(zirium::query::ParameterType::Int, 0..0),
        },
    ]);
    assert!(document.query(&parameter).is_err());
}

#[test]
fn prepared_native_arguments_keep_row_context_and_skipped_errors() {
    use zirium::query::{ConstructedValue as V, NativeValue};
    let query = Query::parse(
        r#"let labels=["first"]; report input | project(o, {
        "same": o | filter(op(one(o | names))) | head(length(labels)) | count,
        "skipped": o | filter(true or op(require(false, "bad", "stop"))) | count
    });"#,
    )
    .unwrap();
    for source in [
        "\"alpha\"() : () -> ()\n\"beta\"() : () -> ()\n",
        "\"gamma\"() : () -> ()\n",
    ] {
        let mut document = document_from(source);
        let mut outputs = Vec::new();
        query
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        let QueryOutput::Native(NativeValue::Constructed(value)) = &outputs[0] else {
            panic!("constructed rows");
        };
        let V::Array(rows) = value.value() else {
            panic!("array");
        };
        assert_eq!(rows.len(), source.lines().count());
        assert!(rows.iter().all(|row| row
            == &V::Record(vec![
                ("same".into(), V::Integer(1)),
                ("skipped".into(), V::Integer(1))
            ])));
    }
}

#[test]
fn typed_functions_reuse_native_tree_and_tail_continuations() {
    use zirium::query::{ConstructedValue, FunctionLimits, NativeValue};
    let mut document = document_from(include_str!("../../../examples/cli/subtree-cost.mlir"));
    let query = Query::parse(include_str!("../../../examples/cli/subtree-cost.zirql")).unwrap();
    let mut emitted = Vec::new();
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(emitted, vec![QueryOutput::Json("8\n".to_owned())]);
    for (source, expected) in [
        (
            "fn read_cost(if: Operation) -> opt int { cost_annotation(if true { if } else { if }) } let selected = filter(op(\"example.root\")); report read_cost(selected) default 8 | json",
            "8\n",
        ),
        (
            "fn read_cost(op: Operation) -> opt int { let if = op; cost_annotation(if) } let selected = filter(op(\"example.root\")); report read_cost(selected) default 8 | json",
            "8\n",
        ),
        (
            "fn count_named(op: Operation, if: string) -> int { op | filter(op(if)) | count } let selected = filter(op(\"example.root\")); report count_named(selected, \"example.root\") | json",
            "1\n",
        ),
    ] {
        emitted.clear();
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                emitted.push(value);
                Ok(())
            })
            .unwrap();
        assert_eq!(emitted, vec![QueryOutput::Json(expected.to_owned())]);
    }
    let query = Query::parse(include_str!("../../../examples/cli/tail-accumulator.zirql"))
        .unwrap()
        .with_function_limits(FunctionLimits { active_depth: 1 });
    emitted.clear();
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(emitted, vec![QueryOutput::Json("500500\n".to_owned())]);
    let query=Query::parse("fn identity(x: Operation) -> Operation { x } let selected = filter(op(\"example.root\")); identity(selected)").unwrap();
    emitted.clear();
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap();
    assert!(matches!(&emitted[0],QueryOutput::Operations(ids) if ids.len()==1));
    let query = Query::parse("fn plus(x: int) -> int { x+1 } report plus(4)").unwrap();
    emitted.clear();
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&emitted[0],QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data()==&ConstructedValue::Integer(5)),
        "{emitted:?}"
    );
}

#[test]
fn function_native_contracts_preserve_ids_sequences_and_current_owners() {
    use zirium::query::{
        FunctionDefinition as F, FunctionNativeType as N, FunctionParameter as P,
        FunctionType as T, Located, NativeValue, ReportExpr as E, function_call, report,
    };
    use zirium::semantic::OperationId;
    let mut document = document_from(include_str!("../../../examples/cli/subtree-cost.mlir"));
    let parameter = |name: &str, declaration| P {
        name: Located::new(name.to_owned(), 0..0),
        declaration,
    };
    let identity = F::new(
        "identity",
        vec![parameter("x", T::Native(N::Operation))],
        T::Native(N::Operation),
        vec![],
        E::name("x"),
    )
    .unwrap();
    let selected = ops()
        .filter(op("example.root"))
        .evaluate(
            &document,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
        .unwrap()[0];
    let call = |value| {
        function_call::<Vec<OperationId>>("identity", [value])
            .with_functions(vec![identity.clone()])
    };
    assert_eq!(
        document.query(&call(E::operation(selected))).unwrap(),
        vec![selected]
    );
    assert_eq!(
        document
            .query(&call(E::native(&ops().filter(op("example.root")))))
            .unwrap(),
        vec![selected]
    );
    let foreign = document_from("\"foreign\"() : () -> ()").root_operations()[0];
    let stale = ops()
        .filter(op("example.right"))
        .evaluate(
            &document,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
        .unwrap()[0];
    let ssa = ops()
        .filter(op("example.right"))
        .operand(0)
        .evaluate(
            &document,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
        .unwrap();
    let ssa_identity = F::new(
        "ssa_identity",
        vec![parameter("x", T::Native(N::SSAValue))],
        T::Native(N::SSAValue),
        vec![],
        E::name("x"),
    )
    .unwrap();
    assert_eq!(
        document
            .query(
                &function_call::<zirium::semantic::ValueId>("ssa_identity", [E::ssa_value(ssa)])
                    .with_functions(vec![ssa_identity])
            )
            .unwrap(),
        ssa
    );
    let mut edit = document.edit(DialectRegistry::baseline()).unwrap();
    edit.erase(stale).unwrap();
    edit.commit().unwrap();
    for invalid in [foreign, stale] {
        let error = document.query(&call(E::operation(invalid))).unwrap_err();
        assert!(error.range().is_some());
    }
    let sequence = F::new(
        "sequence_id",
        vec![parameter(
            "x",
            T::Sequence(Box::new(T::Native(N::Operation))),
        )],
        T::Sequence(Box::new(T::Native(N::Operation))),
        vec![],
        E::name("x"),
    )
    .unwrap();
    for values in [vec![], vec![selected]] {
        let value = document
            .query(
                &function_call::<QueryOutput>(
                    "sequence_id",
                    [E::operation_sequence(values.clone())],
                )
                .with_functions(vec![sequence.clone()]),
            )
            .unwrap();
        assert_eq!(
            value,
            QueryOutput::Native(NativeValue::OperationSequence(values))
        );
    }
    let wrong = F::new("wrong", vec![], T::Integer, vec![], E::boolean(true)).unwrap();
    assert!(
        document
            .query(&report(E::call("wrong", [])).with_functions(vec![wrong]))
            .unwrap_err()
            .range()
            .is_some()
    );
    assert!(
        document
            .query(&call(E::integer(1)))
            .unwrap_err()
            .range()
            .is_some()
    );
}

#[test]
fn projections_compose_with_prepared_values_and_function_results() {
    use zirium::query::{ConstructedValue as V, NativeValue};
    let mut document = document_from(r#""test.a"() : () -> ()"#);
    {
        use zirium::query::{ReportExpr as E, bound_ops, input, report};
        let row = bound_ops("row");
        let mapping = row.map_by(&input().names().one(), &report(E::from(row.count())));
        let nested = ops().project(
            "row",
            [
                ("mapped", E::from(mapping.count())),
                ("existing", E::from(row.where_exists(&input()).count())),
                ("fixed", E::from(row.fixpoint(&input()).count())),
            ],
        );
        let actual = document.query(&nested).unwrap();
        assert_eq!(
            actual.value().data(),
            &V::Array(vec![V::Record(vec![
                ("mapped".into(), V::Integer(1)),
                ("existing".into(), V::Integer(1)),
                ("fixed".into(), V::Integer(1)),
            ])])
        );
    }
    for (source, expected) in [
        (
            r#"project(row,{"z":one(row | names)})"#,
            V::String("test.a".into()),
        ),
        (
            r#"fn n()->int{1} project(row,{"z":one(row | names)})"#,
            V::String("test.a".into()),
        ),
        (
            r#"fn make(root:Operation)->sequence<record {"z":int}> {root | project(row,{"z":1})} make(input | head(1))"#,
            V::Integer(1),
        ),
    ] {
        let mut outputs = Vec::new();
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        let [QueryOutput::Native(NativeValue::Constructed(report))] = outputs.as_slice() else {
            panic!("projected report")
        };
        assert_eq!(
            report.value().data(),
            &V::Array(vec![V::Record(vec![("z".into(), expected)])])
        );
    }
    for (source, expected) in [
        (
            r#"fn make(root:Operation,n:int)->sequence<record {"nested":sequence<record {"z":int}>}> {let offset=n+1; root | project(row,{"nested":row | project(inner,{"z":offset+count})})} make(input | head(1),7) | json"#,
            serde_json::json!([{"nested":[{"z":9}]}]),
        ),
        (
            r#"fn make(root:Operation)->sequence<record {"z":int}> {root | filter(false) | project(row,{"z":1})} let rows=make(input | head(1)); report [r.z for r in rows] | json"#,
            serde_json::json!([]),
        ),
    ] {
        let mut outputs = Vec::new();
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        let [QueryOutput::Json(json)] = outputs.as_slice() else {
            panic!("rendered report")
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(json).unwrap(),
            expected
        );
    }
    let invalid = r#"fn n()->int{1} filter(false) | project(row,{"z":one(row | names),"bad":row})"#;
    assert!(
        Query::parse(invalid)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .is_err()
    );
    for (source, subject) in [
        (
            r#"do set_attr("tag","changed"); input | project(_, {"x":1}) | json"#,
            "_",
        ),
        (
            r#"fn unused(root:Operation)->sequence<record {"rows":sequence<record {"x":int}>}> {root | project(row,{"rows":row | filter(false) | project(count,{"x":1})})} count"#,
            "count",
        ),
    ] {
        let mut emitted = false;
        let error = Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| {
                emitted = true;
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("reserved name"), "{error}");
        assert_eq!(&source[error.range().unwrap()], subject);
        assert!(!emitted);
    }
    assert!(document.query(&ops().attributes("tag")).unwrap().is_empty());
    let error = document
        .query(
            &ops()
                .filter(always(false))
                .project("count", [("x", zirium::query::ReportExpr::integer(1))]),
        )
        .unwrap_err();
    assert!(error.to_string().contains("reserved name"), "{error}");
    assert_eq!(error.range(), Some(0..0));
    let failure = r#"fn make(root:Operation)->sequence<record {"z":int}> {root | project(row,{"z":1 div (count-1)})} make(input | head(1))"#;
    let error = Query::parse(failure)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("division by zero"), "{error}");
    assert!(error.to_string().contains("make"), "{error}");
    let start = failure.find("1 div").unwrap();
    assert_eq!(error.range(), Some(start..start + "1 div (count-1)".len()));
}

#[test]
fn functions_prepare_fields_arguments_and_all_written_bodies_before_execution() {
    use zirium::query::{
        FunctionDefinition as F, FunctionLimits, FunctionType as T, PathEdge, ReportExpr as E,
        function_call, matches, op_expr,
    };
    let mut document = document_from(include_str!("../../../examples/cli/subtree-cost.mlir"));
    let source = r#"fn bump(n: int) -> int { n+1 }
        fn label() -> string { "example.root" }
        filter(op(label())) | head(bump(0)) | project(row, {"cost":bump(row | children | count)}) | json"#;
    let mut outputs = Vec::new();
    for source in [
        source,
        r#"filter(op("example.root")) | head(1) | project(row, {"cost":1+(row | children | count)}) | json"#,
    ] {
        outputs.clear();
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            outputs,
            vec![QueryOutput::Json(
                "[\n  {\n    \"cost\": 3\n  }\n]\n".into()
            )]
        );
    }
    let mut repeated = self::document();
    outputs.clear();
    Query::parse(r#"let outer=7; let original=input; fn bump(n:int)->int{n+1} filter(op("test.seed")) | users | project(row, {"z":one(row | names), "names":row | names, "a":bump(outer), "relative":count, "original":original | count}) | emit"#)
        .unwrap()
        .evaluate(&mut repeated, DialectRegistry::baseline(), |_, value| { outputs.push(value); Ok(()) })
        .unwrap();
    use zirium::query::{ConstructedValue as V, NativeValue};
    let QueryOutput::Native(NativeValue::Constructed(report)) = &outputs[0] else {
        panic!("projection report")
    };
    let V::Array(rows) = report.value().data() else {
        panic!("projection rows")
    };
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], rows[1]);
    let V::Record(fields) = rows[0].data() else {
        panic!("projection record")
    };
    assert_eq!(
        fields
            .iter()
            .map(|(name, value)| (name.as_str(), value.data()))
            .collect::<Vec<_>>(),
        [
            ("z", &V::String("test.add".into())),
            ("names", &V::Array(vec![V::String("test.add".into())])),
            ("a", &V::Integer(8)),
            ("relative", &V::Integer(1)),
            ("original", &V::Integer(6)),
        ]
    );
    outputs.clear();
    Query::parse(r#"fn bump(n:int)->int{n+1} let rows=filter(false) | project(row, {"cost":bump(row | children | count)}); report [r.cost for r in rows] | json"#)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| { outputs.push(value); Ok(()) })
        .unwrap();
    assert_eq!(outputs, vec![QueryOutput::Json("[]\n".into())]);
    let failure = r#"fn broken(n:int)->int{10 div n} filter(op("example.root")) | project(row, {"cost":broken(row | parent | count)}) | json"#;
    let error = Query::parse(failure)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("division by zero"), "{error}");
    assert!(error.range().is_some());
    let mut path_outputs = Vec::new();
    for source in [
        r#"filter(op("example.root")) | paths(users,to=op("example.root"),max_depth=0) | json"#,
        r#"fn label()->string {"example.root"} filter(op("example.root")) | paths(users,to=op(label()),max_depth=0) | json"#,
    ] {
        let mut output = Vec::new();
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                output.push(value);
                Ok(())
            })
            .unwrap();
        path_outputs.push(output);
    }
    assert_eq!(path_outputs[0], path_outputs[1]);
    let label = F::new(
        "label",
        vec![],
        T::String,
        vec![],
        E::string("example.root"),
    )
    .unwrap();
    let typed = ops().filter(op("example.root")).paths(
        PathEdge::Users,
        op_expr(E::from(
            function_call::<String>("label", []).with_functions(vec![label]),
        )),
        0,
    );
    let literal = ops()
        .filter(op("example.root"))
        .paths(PathEdge::Users, op("example.root"), 0);
    assert_eq!(
        document.query(&typed).unwrap().rows(),
        document.query(&literal).unwrap().rows()
    );
    let body = r#"fn label()->string {"example.root"} fn witnesses(op:Operation)->int { op | paths(users,to=op(label()),max_depth=0) | count } let selected=filter(op("example.root")); report witnesses(selected) | json"#;
    let query = Query::parse(body)
        .unwrap()
        .with_function_limits(FunctionLimits { active_depth: 1 });
    let error = query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("active function depth limit"));
    assert!(error.range().is_some());
    outputs.clear();
    query
        .with_function_limits(FunctionLimits { active_depth: 2 })
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(outputs, vec![QueryOutput::Json("1\n".into())]);
    for pipeline in [
        "op | paths(users,to=false and op(label()),max_depth=0)",
        "op | filter(false) | paths(users,to=op(label()),max_depth=0)",
    ] {
        let source = format!(
            r#"fn label()->string {{"example.root"}} fn witnesses(op:Operation)->int {{ {pipeline} | count }} let selected=filter(op("example.root")); report witnesses(selected) | json"#
        );
        outputs.clear();
        Query::parse(&source)
            .unwrap()
            .with_function_limits(FunctionLimits { active_depth: 1 })
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        assert_eq!(outputs, vec![QueryOutput::Json("0\n".into())]);
    }
    for source in [
        "fn bad() -> int { missing(0) } emit; report 1",
        "fn bad() -> int { if true {1} else {missing(0)} } emit; report 1",
        "fn same() -> int {1} fn same() -> int {2} emit; report 1",
        "fn unary(n:int) -> int {n} emit; report unary()",
        "let outer = 2; fn capture() -> int {outer} emit; report 1",
        "fn shadow(n:int) -> int {let n=2; n} emit; report 1",
        "fn impure(op:Operation) -> int { op | set_attr(\"x\",\"y\") | count } emit; report 1",
        "fn label()->string {\"example.root\"} emit; filter(false) | paths(users,to=op(missing()),max_depth=0)",
        "fn bump(n:int)->int{n+1} emit; filter(false) | project(row, {\"x\":if true {bump(0)} else {missing()}})",
        "fn identity(op:Operation)->Operation{op} emit; filter(false) | project(row, {\"x\":identity(row)})",
        "fn bump(n:int)->int{n+1} emit; filter(false) | project(row, {\"x\":bump(0), \"x\":bump(1)})",
    ] {
        let mut emitted = 0;
        let query = Query::parse(source);
        if let Ok(query) = query {
            let error = query
                .evaluate(&mut document, DialectRegistry::baseline(), |_, _| {
                    emitted += 1;
                    Ok(())
                })
                .unwrap_err();
            assert!(error.range().is_some(), "{source}: {error}");
        }
        assert_eq!(emitted, 0, "{source}");
    }
    let bad = F::new("unused", vec![], T::Integer, vec![], E::call("missing", [])).unwrap();
    let source = ops().with_functions(vec![bad]);
    let captures = matches().from("row", &source).build();
    assert!(document.query(&captures).unwrap_err().range().is_some());
    let malformed =
        include_str!("../../../examples/cli/subtree-cost.mlir").replace("3 : i64", "\"bad\"");
    let mut malformed = document_from(&malformed);
    let error = Query::parse(include_str!("../../../examples/cli/subtree-cost.zirql"))
        .unwrap()
        .evaluate(&mut malformed, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("cost annotation"));
    assert!(error.range().is_some());
}

#[test]
fn tail_functions_keep_active_calls_and_cumulative_allowances_bounded() {
    use zirium::query::{FunctionLimits, ProgramLimits};
    let mut document = document();
    let tail = Query::parse(include_str!("../../../examples/cli/tail-accumulator.zirql"))
        .unwrap()
        .with_function_limits(FunctionLimits { active_depth: 1 });
    let limits = ProgramLimits {
        work: 2_000_000,
        results: 1_000_000,
        stream_items: 1000,
        nesting: 64,
    };
    for scarce in [
        ProgramLimits {
            work: 1000,
            ..limits
        },
        ProgramLimits {
            results: 1000,
            ..limits
        },
    ] {
        assert!(
            tail.evaluate_with_program_limits(
                &mut document,
                DialectRegistry::baseline(),
                None,
                EvaluationOptions::default(),
                scarce,
                |_, _| Ok(())
            )
            .is_err()
        );
    }
    for source in [
        "fn forever(n:int)->int {forever(n+1)} report forever(0)",
        "fn grow(n:int)->int {if n==0{0}else{1+grow(n-1)}} report grow(100)",
    ] {
        let query = Query::parse(source)
            .unwrap()
            .with_function_limits(FunctionLimits { active_depth: 1 });
        assert!(
            query
                .evaluate_with_program_limits(
                    &mut document,
                    DialectRegistry::baseline(),
                    None,
                    EvaluationOptions::default(),
                    limits,
                    |_, _| Ok(())
                )
                .unwrap_err()
                .range()
                .is_some()
        );
    }
    let mutual=Query::parse("fn even(n:int)->bool {if n==0{true}else{odd(n-1)}} fn odd(n:int)->bool {if n==0{false}else{even(n-1)}} report even(1000) | json").unwrap().with_function_limits(FunctionLimits{active_depth:1});
    let mut emitted = Vec::new();
    mutual
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(emitted, vec![QueryOutput::Json("true\n".to_owned())]);
    tail.evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap();
}

#[test]
fn function_runtime_failures_retain_explicit_library_emission_events() {
    let mut document = document();
    let mut emitted = Vec::new();
    let error =
        Query::parse("fn fail(n:int)->int {10 div n} report 1 | json; report fail(0) | json")
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                emitted.push(value);
                Ok(())
            })
            .unwrap_err();
    assert_eq!(emitted, vec![QueryOutput::Json("1\n".to_owned())]);
    assert!(error.range().is_some());
}

#[test]
fn exact_string_matching_agrees_in_text_and_typed_projected_names() {
    use zirium::query::{
        ConstructedValue as V, NativeValue, ReportExpr as E, ReportGenerator as G,
        ValueBinaryOperator as B, report,
    };
    let mut document = document_from(FUNCTION_REPORT_SOURCE);
    let source = include_str!("../../../examples/cli/string-matching.zirql");
    let native_source = source.trim_end().strip_suffix(" | json;").unwrap();
    let mut outputs = Vec::new();
    Query::parse(native_source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parsed)) = &outputs[0] else {
        panic!("native report")
    };
    let n = E::name("n");
    let typed = report(E::record([
        (
            "names",
            E::comprehension(
                n.clone(),
                [G::in_values(
                    "n",
                    E::from(ops().filter(op("func.func")).attr("sym_name")),
                    Some(
                        n.clone()
                            .contains(E::string("i"))
                            .binary(B::And, n.clone().starts_with(E::string("m")))
                            .binary(B::And, n.ends_with(E::string("x"))),
                    ),
                )],
            ),
        ),
        (
            "controls",
            E::array([
                E::string("Ångström").contains(E::string("ström")),
                E::string("Ångström").starts_with(E::string("ång")),
                E::string("Ångström").ends_with(E::string("")),
                E::string("é").contains(E::string("é")),
                E::string("").starts_with(E::string("")),
            ]),
        ),
    ]));
    let actual = document.query(&typed).unwrap();
    assert_eq!(actual.value(), parsed.value());
    assert_eq!(
        actual.value(),
        &V::Record(vec![
            ("names".into(), V::Array(vec![V::String("mix".into())])),
            (
                "controls".into(),
                V::Array(
                    vec![true, false, true, false, true]
                        .into_iter()
                        .map(V::Boolean)
                        .collect()
                )
            ),
        ])
    );
    assert!(
        document
            .query(&report(E::string("x").contains(E::integer(1))))
            .is_err()
    );
    let source = r#"report if false {contains("x",1)} else {false};"#;
    let refusal = Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert_eq!(&source[refusal.range().unwrap()], r#"contains("x",1)"#);
    let mut function_outputs = Vec::new();
    Query::parse(
        r#"fn has_part(text:string)->bool { contains(text,"str") } report has_part("Ångström");"#,
    )
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
        function_outputs.push(value);
        Ok(())
    })
    .unwrap();
    assert!(
        matches!(&function_outputs[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Boolean(true))
    );
    assert!(
        Query::parse("fn contains(text:string)->bool {true} report true;")
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .is_err()
    );
    let long = "a".repeat(4_000);
    let limits = zirium::query::EvaluationLimits {
        max_work: 6_000,
        ..Default::default()
    };
    report(E::string(long.clone()).starts_with(E::string("a")))
        .evaluate(&document, DialectRegistry::baseline(), limits)
        .unwrap();
    let refusal = report(E::string(long).contains(E::string("z")))
        .evaluate(&document, DialectRegistry::baseline(), limits)
        .unwrap_err();
    assert!(refusal.to_string().contains("work limit"));
}

#[test]
fn report_ordering_agrees_with_typed_helpers_and_keeps_native_selector_stages() {
    use zirium::query::{ConstructedValue as V, NativeValue, ReportExpr as E, report};
    let mut document = document();
    let source = include_str!("../../../examples/cli/report-ordering.zirql");
    let mut output = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            output.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parsed)) = &output[0] else {
        panic!("constructed report")
    };
    let rows = E::array(
        [("A", 2), ("B", 1), ("C", 2), ("D", 1)]
            .map(|(id, score)| E::record([("id", E::string(id)), ("score", E::integer(score))])),
    );
    let scalars = E::array([2, 1, 2, 1].map(E::integer));
    let typed = report(E::record([
        ("ascending", rows.clone().sort_by("score", false)),
        ("descending", rows.clone().sort_by("score", true)),
        ("minimum", rows.clone().min_by("score")),
        ("maximum", rows.max_by("score")),
        ("scalar", scalars.clone().sort(false)),
        ("scalar_min", scalars.clone().min()),
        ("scalar_max", scalars.max()),
        ("empty_min", E::array([]).min()),
        ("empty_max", E::array([]).max()),
        ("fallback", E::array([]).max().default(E::integer(7))),
    ]));
    let dictionary = report(
        E::record([
            ("z", E::integer(1)),
            ("a", E::integer(1)),
            ("m", E::integer(0)),
        ])
        .sort_by_value(false),
    );
    let sorted = document.query(&dictionary).unwrap();
    let V::Record(entries) = sorted.value().data() else {
        panic!("dictionary");
    };
    assert_eq!(
        entries
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["m", "z", "a"]
    );
    let actual = document.query(&typed).unwrap();
    assert_eq!(actual.value(), parsed.value());
    let V::Record(fields) = actual.value() else {
        panic!("record")
    };
    fn ids(value: &V) -> Vec<&str> {
        let V::Array(rows) = value else {
            panic!("rows")
        };
        rows.iter()
            .map(|row| {
                let V::Record(fields) = row else {
                    panic!("row")
                };
                let V::String(id) = &fields[0].1 else {
                    panic!("id")
                };
                id.as_str()
            })
            .collect::<Vec<_>>()
    }
    assert_eq!(ids(&fields[0].1), ["B", "D", "A", "C"]);
    assert_eq!(ids(&fields[1].1), ["A", "C", "B", "D"]);
    assert_eq!(fields[7].1, V::Absent);
    assert_eq!(fields[9].1, V::Integer(7));
    for source in [
        "names | sort",
        "sort_by(names) | names",
        "sort_by((names # ignored comma, and closing delimiter )\n)) | names",
        "min_by(names) | names",
    ] {
        Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .unwrap();
    }
    let mut outputs = Vec::new();
    Query::parse(
        r#"fn key(n:Operation)->sequence<int> { [1, 2] }
        fn empty(scope:Operation)->sequence<Operation> { direct_children(scope) | filter(false) }
        fn pass(scopes:sequence<Operation>)->sequence<Operation> { scopes }
        report sort([[1], "a", -2.0, 1, true, null]);
        sort_by(key(input)) | names;
        pass(empty(head(1)) | min_by(names));"#,
    )
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
        outputs.push(value);
        Ok(())
    })
    .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(value)) = &outputs[0] else {
        panic!("structural report")
    };
    assert_eq!(
        value.value().data(),
        &V::Array(vec![
            V::Absent,
            V::Boolean(true),
            V::Integer(1),
            V::Float(-2.0),
            V::String("a".into()),
            V::Array(vec![V::Integer(1)])
        ])
    );
    assert!(
        matches!(&outputs[2], QueryOutput::Native(NativeValue::OperationSequence(values)) if values.is_empty())
    );
    let prepared = r#"fn bound(n:int)->int {max(sort([n,1],descending=true)) default 0} filter(op(one(sort(["test.seed"])))) | head(bound(1)) | count"#;
    let mut output = Vec::new();
    Query::parse(prepared)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            output.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(output, [QueryOutput::Count(1)]);
    let projected = r#"let ordered = sort_by([{"score":lookup({"id":1},"score") default 0},{"score":2}],"score"); report min_by(ordered,"score");"#;
    let mut output = Vec::new();
    Query::parse(projected)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            output.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&output[0],QueryOutput::Native(NativeValue::Constructed(value)) if value.value()==&V::Record(vec![("score".into(),V::Integer(0))]))
    );
    use zirium::query::{
        ConstructedReport, FunctionDefinition as F, FunctionParameter as P, FunctionType as T,
        Located, ParameterDeclaration, ParameterType, function_call,
    };
    let pair = F::new(
        "pair",
        vec![P {
            name: Located::new("n".into(), 0..0),
            declaration: T::Integer,
        }],
        T::Sequence(Box::new(T::Integer)),
        vec![],
        E::array([E::name("n"), E::integer(1)]),
    )
    .unwrap();
    let call = function_call::<ConstructedReport>("pair", [E::name("n")])
        .with_functions(vec![pair])
        .with_parameters(vec![ParameterDeclaration {
            name: Located::new("n".into(), 0..0),
            ty: Located::new(ParameterType::Int, 0..0),
        }]);
    let ordered = report(E::from(call).sort(false));
    assert_eq!(ordered.parameters().len(), 1);
    let value = ordered
        .evaluate_with_parameters(
            &document,
            DialectRegistry::baseline(),
            &[("n".into(), V::Integer(2))],
            EvaluationLimits::default(),
        )
        .unwrap();
    let V::Array(values) = value.value().data() else {
        panic!("array")
    };
    assert_eq!(
        [values[0].data(), values[1].data()],
        [&V::Integer(1), &V::Integer(2)]
    );
    {
        let source = r#"report min_by([{"score":1},{"other":2}],"score");"#;
        let position = match Query::parse(source) {
            Err(error) => error.position,
            Ok(query) => {
                query
                    .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
                    .unwrap_err()
                    .range()
                    .unwrap()
                    .start
            }
        };
        assert!(
            source[position..].starts_with("\"score\""),
            "{position}: {source}"
        );
    }
}

#[test]
fn destructured_reports_and_guarded_matches_agree_with_typed_builders() {
    use zirium::query::{
        FunctionDefinition as F, FunctionType as T, Located, NativeValue, PatternKind as P,
        PatternParameter, ReportExpr as E, ReportMatchArm as A, ValueBinaryOperator as B, report,
    };
    let p = |value| Located::new(value, 0..0);
    let name = |name: &str| p(P::Name(Located::new(name.into(), 0..0)));
    let record_pattern = |fields, ignore_rest| {
        p(P::Record {
            fields,
            ignore_rest,
        })
    };
    let parameter = p(P::Tuple(vec![
        name("label"),
        record_pattern(
            vec![(
                Located::new("cost".into(), 0..0),
                p(P::Tuple(vec![p(P::Wildcard), name("cost")])),
            )],
            true,
        ),
    ]));
    let function = F::with_patterns(
        "add_cost",
        vec![PatternParameter {
            pattern: parameter,
            declaration: T::Tuple(vec![
                T::String,
                T::Record(vec![
                    ("cost".into(), T::Tuple(vec![T::String, T::Integer])),
                    ("ignored".into(), T::Integer),
                ]),
            ]),
        }],
        T::Record(vec![
            ("label".into(), T::String),
            ("total".into(), T::Integer),
        ]),
        E::conditional(
            E::name("cost").binary(B::GreaterEqual, E::integer(0)),
            E::block(
                vec![(
                    name("answer"),
                    E::name("cost").binary(B::Add, E::integer(1)),
                )],
                E::record([("label", E::name("label")), ("total", E::name("answer"))]),
            )
            .unwrap(),
            E::record([("label", E::name("label")), ("total", E::integer(0))]),
        ),
    )
    .unwrap();
    let typed = report(
        E::block(
            vec![
                (
                    record_pattern(
                        vec![
                            (Located::new("name".into(), 0..0), name("title")),
                            (
                                Located::new("counts".into(), 0..0),
                                p(P::Tuple(vec![p(P::Wildcard), name("amount")])),
                            ),
                        ],
                        true,
                    ),
                    E::record([
                        ("name", E::string("operations")),
                        ("counts", E::tuple([E::integer(0), E::integer(3)])),
                        ("ignored", E::string("unused")),
                    ]),
                ),
                (
                    p(P::Tuple(vec![name("label"), name("details")])),
                    E::tuple([
                        E::string("alpha"),
                        E::record([
                            ("cost", E::tuple([E::string("estimate"), E::integer(4)])),
                            ("ignored", E::integer(99)),
                        ]),
                    ]),
                ),
            ],
            E::record([
                (
                    "summary",
                    E::call(
                        "add_cost",
                        [E::tuple([E::name("label"), E::name("details")])],
                    ),
                ),
                ("title", E::name("title")),
                ("count", E::name("amount")),
            ]),
        )
        .unwrap(),
    )
    .with_functions(vec![function]);
    let mut document = document();
    let expected = document.query(&typed).unwrap();
    let source = include_str!("../../../examples/cli/destructured-report.zirql");
    let source = source.strip_suffix(" | json;\n").unwrap();
    let mut emitted = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&emitted[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value()==expected.value())
    );
    let cost = E::name("cost");
    let selected = E::block(
        vec![(name("answer"), cost.clone().binary(B::Add, E::integer(1)))],
        E::record([("kind", E::string("small")), ("cost", E::name("answer"))]),
    )
    .unwrap();
    let typed = report(
        E::block(
            vec![(name("row"), E::record([("cost", E::integer(4))]))],
            E::value_match(
                E::name("row").lookup(E::string("cost")),
                vec![
                    A {
                        pattern: p(P::Absent),
                        guard: None,
                        value: E::record([("kind", E::string("absent")), ("cost", E::integer(0))]),
                    },
                    A {
                        pattern: p(P::Present(Box::new(name("cost")))),
                        guard: Some(cost.clone().binary(B::Less, E::integer(0))),
                        value: E::require(
                            E::boolean(false),
                            E::record([("kind", E::string("negative")), ("cost", E::integer(0))]),
                            "negative cost",
                        ),
                    },
                    A {
                        pattern: p(P::Present(Box::new(name("cost")))),
                        guard: Some(cost.clone().binary(B::Greater, E::integer(10))),
                        value: E::record([("kind", E::string("large")), ("cost", cost)]),
                    },
                    A {
                        pattern: p(P::Present(Box::new(name("cost")))),
                        guard: None,
                        value: selected,
                    },
                    A {
                        pattern: p(P::Wildcard),
                        guard: None,
                        value: E::record([
                            ("kind", E::string("fallback")),
                            ("cost", E::integer(0)),
                        ]),
                    },
                ],
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let expected = document.query(&typed).unwrap();
    emitted.clear();
    let source = include_str!("../../../examples/cli/guarded-match.zirql")
        .strip_suffix(" | json;\n")
        .unwrap();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            emitted.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&emitted[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value()==expected.value())
    );
}

#[test]
fn pattern_native_views_tail_calls_and_refusals_keep_host_contracts() {
    use zirium::query::{
        FunctionDefinition as F, FunctionLimits, FunctionNativeType as N, FunctionType as T,
        Located, NativeValue, PatternKind as P, PatternParameter, ReportExpr as E,
        ReportMatchArm as A, function_call,
    };
    let p = |value| Located::new(value, 41..48);
    let name = || p(P::Name(Located::new("selected".into(), 41..48)));
    let function = F::with_patterns(
        "select",
        vec![PatternParameter {
            pattern: p(P::Record {
                fields: vec![(Located::new("operation".into(), 41..48), name())],
                ignore_rest: true,
            }),
            declaration: T::Record(vec![
                ("operation".into(), T::Native(N::Operation)),
                ("ignored".into(), T::Native(N::Operation)),
            ]),
        }],
        T::Native(N::Operation),
        E::name("selected"),
    )
    .unwrap();
    let mut document = document();
    let operation = ops()
        .head(1)
        .evaluate(
            &document,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
        .unwrap()[0];
    let call = |operation| {
        function_call::<Vec<zirium::semantic::OperationId>>(
            "select",
            [E::capture(vec![
                ("operation".into(), QueryOutput::Operations(vec![operation])),
                ("ignored".into(), QueryOutput::Operations(vec![operation])),
            ])],
        )
        .with_functions(vec![function.clone()])
    };
    assert_eq!(document.query(&call(operation)).unwrap(), vec![operation]);
    let other = document_from(SOURCE);
    let foreign = ops()
        .head(1)
        .evaluate(
            &other,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
        .unwrap()[0];
    assert!(document.query(&call(foreign)).is_err());
    let matched = E::value_match(
        E::capture(vec![(
            "operation".into(),
            QueryOutput::Operations(vec![operation]),
        )]),
        vec![
            A {
                pattern: p(P::Record {
                    fields: vec![(Located::new("operation".into(), 41..48), name())],
                    ignore_rest: false,
                }),
                guard: None,
                value: E::name("selected"),
            },
            A {
                pattern: p(P::Wildcard),
                guard: None,
                value: E::operation(operation),
            },
        ],
    )
    .unwrap();
    let identity = F::with_patterns(
        "identity",
        vec![PatternParameter {
            pattern: p(P::Wildcard),
            declaration: T::Integer,
        }],
        T::Native(N::Operation),
        matched,
    )
    .unwrap();
    assert_eq!(
        document
            .query(
                &function_call::<Vec<zirium::semantic::OperationId>>("identity", [E::integer(0)])
                    .with_functions(vec![identity])
            )
            .unwrap(),
        vec![operation]
    );
    let source = "fn sum_to((n, total): tuple<int,int>) -> int { match (n) { 0 => total, _ => { let next = (n - 1, total + n); if true { sum_to(next) } else { 0 } } } } report sum_to((100,0))";
    let mut values = Vec::new();
    Query::parse(source)
        .unwrap()
        .with_function_limits(FunctionLimits { active_depth: 1 })
        .evaluate(&mut document, DialectRegistry::baseline(), |_, v| {
            values.push(v);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&values[0],QueryOutput::Native(NativeValue::Constructed(v)) if v.value().data()==&zirium::query::ConstructedValue::Integer(5050))
    );
    for source in [
        "let (a,b) = (1,); report a",
        "let {a} = {\"a\":1,\"b\":2}; report a",
        "let (a,a) = (1,2); report a",
        "let a = 1; report if true { 1 } else { let a = 2; a }",
        "report match (1) { x if 1 => x, _ => 0 }",
        "report match (1) { x if (1 div 0) == 0 => x, _ => 0 }",
        "report match (1) { x => x }",
        "report if true { let local = 1; local } else { local }",
        "report match (0) { 0 => 1, _ => input | set_attr(\"tag\",\"x\") }",
    ] {
        let result = Query::parse(source).and_then(|query| {
            query
                .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
                .map_err(|error| zirium::query::QueryError {
                    position: error.range().map_or(0, |range| range.start),
                    message: error.to_string(),
                })
        });
        let error = result.unwrap_err();
        assert!(error.position < source.len(), "{source}: {error}");
    }
}

#[test]
fn native_worklists_and_named_reducers_use_the_shared_runner() {
    use zirium::query::{ConstructedValue as V, NativeValue, ReportExpr as E, report};
    let mut document = document_from(include_str!(
        "../../../examples/cli/worklist-reachability.mlir"
    ));
    let mut outputs = Vec::new();
    Query::parse(include_str!(
        "../../../examples/cli/worklist-reachability.zirql"
    ))
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
        outputs.push(value);
        Ok(())
    })
    .unwrap();
    let QueryOutput::Json(json) = &outputs[0] else {
        panic!("JSON report")
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(json).unwrap(),
        serde_json::json!({"zero_edges":true,"reachable":true,"disconnected":false})
    );
    let mut outputs = Vec::new();
    Query::parse(include_str!("../../../examples/cli/fold-scan.zirql"))
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Json(json) = &outputs[0] else {
        panic!("JSON report")
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(json).unwrap(),
        serde_json::json!({"fold":[3,10],"scan":[[1,2],[2,5],[3,10]],"empty_fold":[0,0],"empty_scan":[]})
    );

    let nodes = document.query(&ops().filter(op("graph.node"))).unwrap();
    let same = report(E::same_operation(
        E::operation(nodes[0]),
        E::operation(nodes[1]),
    ));
    assert_eq!(
        document.query(&same).unwrap().value().data(),
        &V::Boolean(false)
    );
    let membership = report(E::record([
        (
            "same",
            E::operation(nodes[0]).is_in(E::insert(
                E::identity_set(zirium::query::FunctionNativeType::Operation),
                E::operation(nodes[0]),
            )),
        ),
        (
            "different",
            E::operation(nodes[1]).is_in(E::insert(
                E::identity_set(zirium::query::FunctionNativeType::Operation),
                E::operation(nodes[0]),
            )),
        ),
    ]));
    assert_eq!(
        document
            .query(&membership)
            .unwrap()
            .to_json_with_budget(&mut zirium::query::Budget::for_program(
                zirium::query::ProgramLimits {
                    work: 100000,
                    results: 100000,
                    nesting: 64,
                    stream_items: 1000
                }
            ))
            .unwrap(),
        serde_json::json!({"same":true,"different":false})
    );
    let mut retained = Vec::new();
    for value in [
        E::queue(E::operation_sequence(vec![])),
        E::identity_set(zirium::query::FunctionNativeType::Operation),
        E::insert(
            E::identity_set(zirium::query::FunctionNativeType::Operation),
            E::operation(nodes[0]),
        ),
    ] {
        let query = report(value);
        let mut budget = query.new_budget(EvaluationLimits::default());
        retained.push(
            query
                .evaluate_output_with_budget(
                    &document,
                    DialectRegistry::baseline(),
                    EvaluationOptions::default(),
                    &mut budget,
                )
                .unwrap(),
        );
    }
    let queue = report(E::queue(E::operation_sequence(nodes.clone())));
    let mut budget = queue.new_budget(EvaluationLimits::default());
    let queue = queue
        .evaluate_output_with_budget(
            &document,
            DialectRegistry::baseline(),
            EvaluationOptions::default(),
            &mut budget,
        )
        .unwrap();
    assert!(matches!(
        queue,
        QueryOutput::Native(NativeValue::Computational(_))
    ));
    let pop = report(
        E::value_match(
            E::pop_front(E::native_input(queue.clone())),
            vec![
                zirium::query::ReportMatchArm {
                    pattern: zirium::query::Located::new(
                        zirium::query::PatternKind::Present(Box::new(zirium::query::Located::new(
                            zirium::query::PatternKind::Wildcard,
                            0..0,
                        ))),
                        0..0,
                    ),
                    guard: None,
                    value: E::boolean(true),
                },
                zirium::query::ReportMatchArm {
                    pattern: zirium::query::Located::new(
                        zirium::query::PatternKind::Wildcard,
                        0..0,
                    ),
                    guard: None,
                    value: E::boolean(false),
                },
            ],
        )
        .unwrap(),
    );
    assert_eq!(
        document.query(&pop).unwrap().value().data(),
        &V::Boolean(true)
    );
    assert!(
        queue
            .to_json_with_budget(&document, &mut budget)
            .unwrap_err()
            .range()
            .is_some()
    );
    let nested = report(E::tuple([E::native_input(queue.clone()), E::integer(1)]));
    let mut nested_budget = nested.new_budget(EvaluationLimits::default());
    retained.push(
        nested
            .evaluate_output_with_budget(
                &document,
                DialectRegistry::baseline(),
                EvaluationOptions::default(),
                &mut nested_budget,
            )
            .unwrap(),
    );
    let foreign = document_from(include_str!(
        "../../../examples/cli/worklist-reachability.mlir"
    ));
    assert!(foreign.query(&pop).is_err());
    for value in &retained {
        let query = report(E::native_input(value.clone()));
        let mut budget = query.new_budget(EvaluationLimits::default());
        assert!(
            query
                .evaluate_output_with_budget(
                    &foreign,
                    DialectRegistry::baseline(),
                    EvaluationOptions::default(),
                    &mut budget
                )
                .is_err()
        );
    }

    let disconnected = document
        .query(&ops().filter(op("graph.disconnected")))
        .unwrap()[0];
    let mut edit = document.edit(DialectRegistry::baseline()).unwrap();
    edit.erase(disconnected).unwrap();
    edit.commit().unwrap();
    assert!(document.query(&pop).is_err());
    for value in retained {
        let query = report(E::native_input(value));
        let mut budget = query.new_budget(EvaluationLimits::default());
        assert!(
            query
                .evaluate_output_with_budget(
                    &document,
                    DialectRegistry::baseline(),
                    EvaluationOptions::default(),
                    &mut budget
                )
                .is_err()
        );
    }
}

#[test]
fn worklist_refusals_are_located_and_preflight_unused_paths() {
    let mut document = document();
    for source in [
        "report loop (n = 0) { continue (n + 1, n); }",
        "report loop (n = 0) { if true { break 1; } else { 2 } }",
        "report loop (n = 0) { continue (\"wrong\"); }",
        "report queue([])",
        "report break 1",
        "report continue (0)",
        "fn transfer() -> int { break 1 } report loop (n=0) { break transfer(); }",
        "fn bad(state:int,item:int)->int { input | set_attr(\"x\",\"y\") } report fold([],0,bad)",
        "fn bad(state: int, item: int) -> string { \"bad\" } report fold([], 0, bad)",
        "fn bad(state: int) -> int { state } report scan([], 0, bad)",
        "report if true { 1 } else { loop (n = 0) { input | set_attr(\"x\",\"y\") } }",
        "report queue<Operation>([]) | json",
        "report {\"nested\": identity_set<Operation>()} | json",
    ] {
        let result = Query::parse(source)
            .map_err(|e| e.to_string())
            .and_then(|query| {
                query
                    .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
                    .map_err(|e| {
                        assert!(e.range().is_some(), "{source}: {e}");
                        e.to_string()
                    })
            });
        assert!(result.is_err(), "{source}");
    }
    let query = Query::parse("report loop (n = 0) { continue (n + 1); }").unwrap();
    assert!(
        query
            .evaluate_with_limits(
                &mut document,
                DialectRegistry::baseline(),
                EvaluationLimits {
                    max_work: 1000,
                    ..EvaluationLimits::default()
                },
                |_, _| Ok(())
            )
            .is_err()
    );
}

#[test]
fn native_queue_aliases_and_scan_snapshots_remain_immutable() {
    let mut document = document_from(include_str!(
        "../../../examples/cli/worklist-reachability.mlir"
    ));
    let source = r#"
fn size(initial: queue<Operation>) -> int {
  loop (pending = initial, count = 0) {
    match (pop_front(pending)) {
      present((_, remaining)) => continue (remaining, count + 1),
      _ => break count
    }
  }
}
fn append(state: queue<Operation>, item: Operation) -> queue<Operation> {
  push_back_all(state, [item])
}
let nodes = filter(op("graph.node"));
let original = queue(nodes);
let advanced = push_back_all(original, nodes);
let snapshots = scan(nodes, queue<Operation>([]), append);
report {"aliases": (size(original), size(advanced)), "snapshots": [size(snapshot) for snapshot in snapshots]} | json
"#;
    let mut values = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            values.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Json(json) = &values[0] else {
        panic!("JSON report")
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(json).unwrap(),
        serde_json::json!({"aliases":[2,4],"snapshots":[1,2]})
    );
}

#[test]
fn user_info_refuses_incomplete_adjacency_and_keeps_successor_use_order() {
    use zirium::query::{ReportExpr as E, report};
    let parsed = ParsedFile::parse(b"vendor.outer @entry { vendor.inner }".as_slice()).unwrap();
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &DialectRegistry::EMPTY);
    assert!(!lowered.semantically_complete);
    let document = lowered.document.unwrap();
    let query = report(E::user_info(E::operation(document.root_operations()[0])));
    let mut budget = query.new_budget(EvaluationLimits::default());
    let error = query
        .evaluate_output_with_budget(
            &document,
            &DialectRegistry::EMPTY,
            EvaluationOptions::default(),
            &mut budget,
        )
        .unwrap_err();
    assert!(error.to_string().contains("complete semantic adjacency"));
    assert!(error.range().is_some());
    let document = document_from(
        r#"module {
      func.func @f() -> i32 {
        %seed = arith.constant 1 : i32
        cf.br ^next(%seed : i32)
      ^next(%arg: i32):
        %twice = arith.addi %seed, %seed : i32
        func.return %arg : i32
      }
    }"#,
    );
    let seed = document.query(&ops().filter(op("arith.constant"))).unwrap()[0];
    let expected = document
        .query(&ops().filter(op("arith.constant")).users())
        .unwrap();
    let query = report(E::user_info(E::operation(seed)));
    let mut budget = query.new_budget(EvaluationLimits::default());
    let output = query
        .evaluate_output_with_budget(
            &document,
            DialectRegistry::baseline(),
            EvaluationOptions::default(),
            &mut budget,
        )
        .unwrap()
        .into_owned_with_budget(&mut budget)
        .unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Capture(record)) = output else {
        panic!("native view")
    };
    let QueryOutput::Native(zirium::query::NativeValue::OperationSequence(targets)) =
        &record.fields[0].1
    else {
        panic!("native targets")
    };
    assert_eq!(targets, &expected);
    assert_eq!(targets.len(), 3);
    assert_eq!(targets[1], targets[2]);
}

#[test]
fn typed_native_reducers_retain_queue_aliases_and_scan_states() {
    use zirium::query::{
        FunctionDefinition as F, FunctionNativeType as N, FunctionParameter, FunctionType as T,
        Located, NativeValue, ReportExpr as E, SignatureType, report,
    };
    let document = document_from(include_str!(
        "../../../examples/cli/worklist-reachability.mlir"
    ));
    let nodes = document.query(&ops().filter(op("graph.node"))).unwrap();
    let append = F::new(
        "append",
        vec![
            FunctionParameter {
                name: Located::new("state".into(), 0..0),
                declaration: T::Queue(Box::new(T::Native(N::Operation))),
            },
            FunctionParameter {
                name: Located::new("item".into(), 0..0),
                declaration: T::Native(N::Operation),
            },
        ],
        T::Queue(Box::new(T::Native(N::Operation))),
        vec![],
        E::push_back_all(E::name("state"), E::array([E::name("item")])),
    )
    .unwrap();
    let empty = report(E::typed_queue(
        E::array([]),
        SignatureType::Named(Located::new("Operation".into(), 0..0)),
    ));
    let mut budget = empty.new_budget(EvaluationLimits::default());
    let empty = empty
        .evaluate_output_with_budget(
            &document,
            DialectRegistry::baseline(),
            EvaluationOptions::default(),
            &mut budget,
        )
        .unwrap();
    let scan = report(E::scan(
        E::operation_sequence(nodes.clone()),
        E::native_input(empty.clone()),
        "append",
    ))
    .with_functions(vec![append.clone()]);
    let mut budget = scan.new_budget(EvaluationLimits::default());
    let result = scan
        .evaluate_output_with_budget(
            &document,
            DialectRegistry::baseline(),
            EvaluationOptions::default(),
            &mut budget,
        )
        .unwrap()
        .into_owned_with_budget(&mut budget)
        .unwrap();
    let QueryOutput::Native(NativeValue::Sequence(states)) = result else {
        panic!("native snapshots")
    };
    let lengths = states
        .values()
        .iter()
        .map(|value| {
            let value = value.clone().into_owned_with_budget(&mut budget).unwrap();
            let QueryOutput::Native(NativeValue::Computational(queue)) = value else {
                panic!("queue state")
            };
            queue.len()
        })
        .collect::<Vec<_>>();
    assert_eq!(lengths, [1, 2]);
    let fold = report(E::fold(
        E::operation_sequence(nodes),
        E::native_input(empty.clone()),
        "append",
    ))
    .with_functions(vec![append]);
    let mut budget = fold.new_budget(EvaluationLimits::default());
    let result = fold
        .evaluate_output_with_budget(
            &document,
            DialectRegistry::baseline(),
            EvaluationOptions::default(),
            &mut budget,
        )
        .unwrap()
        .into_owned_with_budget(&mut budget)
        .unwrap();
    let QueryOutput::Native(NativeValue::Computational(queue)) = result else {
        panic!("fold queue")
    };
    assert_eq!(queue.len(), 2);
    let QueryOutput::Native(NativeValue::Computational(initial)) = empty else {
        panic!("empty queue")
    };
    assert!(initial.is_empty());
}

#[test]
fn common_index_sets_and_tuple_positions_read_projected_names() {
    let mut document = document_from(FUNCTION_REPORT_SOURCE);
    let source = include_str!("../../../examples/cli/index-sets-and-tuples.zirql");
    let mut outputs = Vec::new();
    Query::parse(source.trim_end().strip_suffix(" | json;").unwrap())
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(value)) = &outputs[0] else {
        panic!("constructed report")
    };
    use zirium::query::ConstructedValue as V;
    assert_eq!(
        value.value(),
        &V::Array(
            ["double", "mix", "plain"]
                .into_iter()
                .enumerate()
                .map(|(i, name)| {
                    V::Record(vec![
                        ("position".into(), V::Integer(i64::try_from(i + 1).unwrap())),
                        ("name".into(), V::String(name.into())),
                    ])
                })
                .collect()
        )
    );
}

#[test]
fn prepared_ordering_selector_keeps_parameter_and_saved_outer_input() {
    let mut document = document_from("module { \"test.z\"() : () -> () \"test.a\"() : () -> () }");
    let source = r#"fn labels(root: Operation, needle: string) -> sequence<string> {
        let saved = root;
        direct_children(root) | sort_by((filter(op(needle)) union saved) | count) | names
    }
    let selected_root = filter(op("builtin.module")); report labels(selected_root, "test.z") | json"#;
    let mut outputs = vec![];
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        outputs,
        [QueryOutput::Json(
            "[\n  \"test.a\",\n  \"test.z\"\n]\n".into()
        )]
    );
}

#[test]
fn prepared_map_selectors_keep_parameters_saved_inputs_and_native_values() {
    let mut document = document_from(
        r#"module {
        "test.z"() {sym_name = "z"} : () -> ()
        "test.a"() {sym_name = "a"} : () -> ()
    }"#,
    );
    let source = r#"fn summary(root: Operation, needle: string) -> record {a: int, z: int} {
        let saved = root;
        direct_children(root) | map_by(attr("sym_name"), (filter(op(needle)) union saved) | count)
    }
    let selected_root = filter(op("builtin.module")); report summary(selected_root, "test.z") | json"#;
    let mut outputs = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        outputs,
        [QueryOutput::Json("{\n  \"z\": 2,\n  \"a\": 1\n}\n".into())]
    );

    let source = r#"fn selected(root: Operation) -> record {a: Operation, z: Operation} {
        direct_children(root) | map_by(attr("sym_name"), input)
    }
    let selected_root = filter(op("builtin.module")); selected(selected_root)"#;
    let mut result = None;
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            result = Some(value);
            Ok(())
        })
        .unwrap();
    let result = result.unwrap();
    let mut value = &result;
    while let QueryOutput::Native(zirium::query::NativeValue::Retained(retained)) = value {
        value = retained;
    }
    let QueryOutput::Native(zirium::query::NativeValue::Map(entries)) = value else {
        panic!("native map")
    };
    let QueryOutput::Operations(operations) =
        &entries.iter().find(|(key, _)| key == "z").unwrap().1
    else {
        panic!("native operations")
    };
    assert_eq!(document.operation_name(operations[0]), Some("test.z"));

    let source = r#"fn duplicate(root: Operation, needle: string) -> int {
        direct_children(root) | map_by(parent | names, filter(op(needle)) | check(1)) | count
    }
    let selected_root = filter(op("builtin.module")); duplicate(selected_root, "test.z")"#;
    let error = Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("duplicate key"), "{error}");
    assert!(error.range().is_some());

    let source = r#"fn invalid(root: Operation) -> int {
        direct_children(root) | filter(false) | map_by(children, count) | count
    }
    let selected_root = filter(op("builtin.module")); invalid(selected_root)"#;
    let error = Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("exactly one string"), "{error}");
    assert!(error.range().is_some());
}

#[test]
fn prepared_fixed_point_keeps_function_locals_saved_bindings_and_native_kind() {
    let mut document = document_from("module { \"test.z\"() : () -> () \"test.a\"() : () -> () }");
    let query = Query::parse(
        r#"
        fn labels(root: Operation, needle: string) -> sequence<string> {
            let saved = root;
            let local = needle;
            direct_children(root) | fixpoint(filter(op(local)) union saved) | names
        }
        let selected_root = filter(op("builtin.module"));
        report labels(selected_root, "test.z") | json
    "#,
    )
    .unwrap();
    let mut outputs = vec![];
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        outputs,
        [QueryOutput::Json(
            "[\n  \"test.z\",\n  \"builtin.module\"\n]\n".into()
        )]
    );

    let query = Query::parse(
        r#"
        fn names_stable(root: Operation) -> sequence<string> {
            direct_children(root) | names | fixpoint(input)
        }
        let selected = filter(op("builtin.module"));
        report names_stable(selected) | json
    "#,
    )
    .unwrap();
    let mut outputs = vec![];
    query
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        outputs,
        [QueryOutput::Json(
            "[\n  \"test.z\",\n  \"test.a\"\n]\n".into()
        )]
    );

    let mut outputs = vec![];
    Query::parse("N = names; fixpoint(N)")
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&outputs[..], [QueryOutput::Values(values)] if values == &["builtin.module", "test.z", "test.a"])
    );
}

#[test]
fn textual_map_ranking_keeps_saved_order_with_prepared_bounds() {
    let mut document = document();
    let source = r#"fn bound(n:int)->int {n}
        ranked = filter(op("test.scope")) | map_by(attr("sym_name"), children | count) | sort_by(value);
        ranked | reverse | head(bound(2)) | tail(bound(1)) | json;
        ranked | json"#;
    let mut outputs = vec![];
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        outputs,
        [
            QueryOutput::Json("{\n  \"second\": 1\n}\n".into()),
            QueryOutput::Json("{\n  \"second\": 1,\n  \"first\": 3\n}\n".into()),
        ]
    );

    let mut document = document_from(r#"module { "test.z"() : () -> () "test.a"() : () -> () }"#);
    let source = r#"fn ordered(root:Operation)->int {
        direct_children(root) | map_by(names, count) | sort_by(value) | count
    } let selected_root = filter(op("builtin.module")); ordered(selected_root)"#;
    let mut outputs = vec![];
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(outputs.as_slice(), [QueryOutput::Native(zirium::query::NativeValue::Constructed(value))] if value.value().data() == &zirium::query::ConstructedValue::Integer(2))
    );
}

#[test]
fn text_existence_keeps_typed_identities_duplicates_and_prepared_scope() {
    let mut document = document();
    let selected = ops()
        .filter(op("test.add") | op("test.end"))
        .reverse()
        .defs();
    let expected = document
        .query(&selected.where_exists(&input().users()))
        .unwrap();
    assert_eq!(expected.len(), 3);
    assert_eq!(expected[1], expected[2]);
    let mut actual = vec![];
    Query::parse(
        r#"filter(op("test.add") or op("test.end")) | reverse | defs | where_exists(users)"#,
    )
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
        actual.push(value);
        Ok(())
    })
    .unwrap();
    assert_eq!(actual, [QueryOutput::Operations(expected)]);

    let source = r#"fn select(scope: Operation, needle: string) -> sequence<Operation> {
        let saved = scope;
        let local = needle;
        direct_children(scope) | where_exists((users | filter(op(local))) intersect (saved | children))
    }
    let scope = filter(op("test.scope")) | head(1);
    select(scope, "test.add") | names"#;
    let mut actual = vec![];
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            actual.push(value);
            Ok(())
        })
        .unwrap();
    let expected = document.query(&ops().filter(op("test.seed"))).unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(value)) = &actual[0] else {
        panic!("native names")
    };
    assert_eq!(
        value.value(),
        &zirium::query::ConstructedValue::Array(vec![zirium::query::ConstructedValue::String(
            "test.seed".into()
        )])
    );
    assert_eq!(expected.len(), 1);
    // The ordinary saved-input path also retains a relation without projecting it.
    let mut empty = vec![];
    Query::parse("let saved = filter(false); where_exists(saved)")
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            empty.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(empty, [QueryOutput::Operations(vec![])]);
    for source in [
        "filter(false) | where_exists(names)",
        "filter(false) | where_exists(unknown)",
        "filter(false) | where_exists(emit)",
        "where_exists(users | check(1))",
    ] {
        if let Ok(query) = Query::parse(source) {
            let error = query
                .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
                .unwrap_err();
            assert!(error.range().is_some(), "{source}: {error}");
        }
    }
}

#[test]
fn brace_collections_keep_native_data_order_and_function_locals() {
    let mut document = document();
    let mut outputs = Vec::new();
    Query::parse(include_str!(
        "../../../examples/cli/brace-collections.zirql"
    ))
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
        outputs.push(value);
        Ok(())
    })
    .unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(report)) = &outputs[0] else {
        panic!("constructed report")
    };
    use zirium::query::ConstructedValue as V;
    let V::Record(fields) = report.value().data() else {
        panic!("report dictionary")
    };
    assert_eq!(
        fields
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["set", "dictionary", "access", "empty", "typed", "sorted"]
    );
    let V::Record(entries) = fields[1].1.data() else {
        panic!("computed dictionary")
    };
    assert_eq!(
        entries
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["pear!", "apple!"]
    );
    assert!(
        matches!(fields[0].1.data(), V::Set { elements, .. } if **elements == [V::Integer(3),V::Integer(1)])
    );
    assert!(
        matches!(fields[2].1.data(), V::Set { elements, .. } if **elements == [V::Integer(1),V::Integer(2),V::Integer(0)])
    );
    outputs.clear();
    Query::parse(r#"[{"pear":1 for _ in [1]}]"#)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(value)) = &outputs[0] else {
        panic!("nested dictionary comprehension")
    };
    assert!(
        matches!(value.value().data(), V::Array(values) if matches!(values[0].data(), V::Record(fields) if fields[0].0 == "pear"))
    );
    outputs.clear();
    Query::parse(r#"fn make(k:string) -> record {"pear":set<int>} { let key=k; {key:{3,1,3}} } let saved=make("pear"); report saved["pear"];"#).unwrap()
        .evaluate(&mut document,DialectRegistry::baseline(),|_,value| { outputs.push(value); Ok(()) }).unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(value)) = &outputs[0] else {
        panic!("function result")
    };
    assert!(
        matches!(value.value().data(), V::Set { elements, .. } if **elements == [V::Integer(3),V::Integer(1)])
    );
}

#[test]
fn indexed_arrays_retain_declared_domains_and_sparse_native_cells() {
    use zirium::query::{ConstructedValue as V, NativeValue};
    use zirql::{KeyAxis, KeyedEntry, ValueKey as K, ValueKeyType};

    let mut document = document();
    let evaluate = |document: &mut Document, source: &str| {
        let mut outputs = Vec::new();
        Query::parse(source)
            .unwrap()
            .evaluate(document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        outputs
    };
    let literal = evaluate(
        &mut document,
        "[(2,1):21, (5,8):58, (2,8):28] axes([5,2,9], [8,1,4])",
    );
    let generated = evaluate(
        &mut document,
        "let rows=[5,2,9]; let a=[(i,j):i*10+j for i in [5,2] for j in [8,1] if i==2 or j==8] axes(rows,[8,1,4]); a",
    );
    let QueryOutput::Native(NativeValue::Constructed(report)) = &literal[0] else {
        panic!("native keyed report")
    };
    let QueryOutput::Native(NativeValue::Constructed(generated)) = &generated[0] else {
        panic!("native keyed comprehension")
    };
    assert_eq!(report.value(), generated.value());
    assert_eq!(
        report.value().data(),
        &V::KeyedArray {
            axes: vec![
                KeyAxis {
                    key_type: ValueKeyType::Integer,
                    keys: vec![K::Integer(5), K::Integer(2), K::Integer(9)],
                },
                KeyAxis {
                    key_type: ValueKeyType::Integer,
                    keys: vec![K::Integer(8), K::Integer(1), K::Integer(4)],
                },
            ],
            entries: vec![
                KeyedEntry {
                    key: vec![K::Integer(5), K::Integer(8)],
                    value: V::Integer(58)
                },
                KeyedEntry {
                    key: vec![K::Integer(2), K::Integer(8)],
                    value: V::Integer(28)
                },
                KeyedEntry {
                    key: vec![K::Integer(2), K::Integer(1)],
                    value: V::Integer(21)
                },
            ],
        }
    );
    let empty = evaluate(
        &mut document,
        "fn empty_keys()->sequence<int>{ [] } [] axes(empty_keys(),[8,1])",
    );
    let QueryOutput::Native(NativeValue::Constructed(report)) = &empty[0] else {
        panic!("typed empty keyed report")
    };
    assert!(
        matches!(report.value().data(), V::KeyedArray { axes, entries }
        if axes.len()==2 && axes[0].keys.is_empty()
        && axes[0].key_type==ValueKeyType::Integer
        && axes[1].keys==[K::Integer(8),K::Integer(1)] && entries.is_empty())
    );
    let native_domain = evaluate(
        &mut document,
        "let labels=input | names; [] axes(index_set(labels))",
    );
    let QueryOutput::Native(NativeValue::Constructed(report)) = &native_domain[0] else {
        panic!("native source domain")
    };
    assert!(
        matches!(report.value().data(), V::KeyedArray { axes, entries }
        if axes[0].keys.len()==document.query(&ops().names()).unwrap().len()
        && !axes[0].keys.is_empty() && entries.is_empty())
    );
    for source in [
        "[i:1 for i in [1]] axes([i])",
        "[2:1] axes([1])",
        "[] axes([])",
    ] {
        let query = Query::parse(source).unwrap();
        let error = query
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .unwrap_err();
        assert!(error.range().is_some(), "{source}: {error}");
    }
}

#[test]
fn shared_value_calls_and_saved_suffixes_keep_constructed_and_native_sources() {
    use zirium::query::{ConstructedValue as V, NativeValue};
    let mut document = document();
    let evaluate = |document: &mut Document, source: &str| {
        let mut outputs = Vec::new();
        Query::parse(source)
            .unwrap()
            .evaluate(document, DialectRegistry::baseline(), |_, value| {
                outputs.push(value);
                Ok(())
            })
            .unwrap();
        outputs
    };
    let final_call = evaluate(&mut document, "let numbers=[3,1,3]; sum(numbers)");
    let initializer = evaluate(
        &mut document,
        "let numbers=[3,1,3]; let total=sum(numbers); total",
    );
    for output in [&final_call[0], &initializer[0]] {
        assert!(
            matches!(output, QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Integer(7))
        );
    }
    for (expression, expected) in [
        ("n+2", V::Integer(3)),
        ("1+2", V::Integer(3)),
        ("(n+2)*3-1", V::Integer(8)),
        ("(n+2)", V::Integer(3)),
        (r#""a" ++ "b""#, V::String("ab".into())),
        ("not false and n+2 == 3", V::Boolean(true)),
        ("[n,2][1]+2", V::Integer(3)),
        (r#"{"n":n}.n+2"#, V::Integer(3)),
    ] {
        let direct = evaluate(&mut document, &format!("let n=1; {expression}"));
        let saved = evaluate(
            &mut document,
            &format!("let n=1; let result={expression}; result"),
        );
        for output in [&direct[0], &saved[0]] {
            assert!(
                matches!(output, QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &expected),
                "{expression}: {output:?}"
            );
        }
    }
    for (expression, expected) in [
        ("1 == 1", true),
        ("true==true", true),
        ("n == 1", true),
        ("n==2", false),
    ] {
        let direct = evaluate(&mut document, &format!("let n=1; {expression}"));
        let reported = evaluate(&mut document, &format!("let n=1; report {expression}"));
        for output in [&direct[0], &reported[0]] {
            assert!(
                matches!(output, QueryOutput::Native(NativeValue::Constructed(value))
                    if value.value().data() == &V::Boolean(expected)),
                "{expression}: {output:?}"
            );
        }
    }
    for expression in ["d.n", "(d.n)", "t.1"] {
        let bindings = r#"let d={"n":1}; let t=(1,2);"#;
        let direct = evaluate(&mut document, &format!("{bindings} {expression}"));
        let saved = evaluate(
            &mut document,
            &format!("{bindings} let result={expression}; result"),
        );
        for output in [&direct[0], &saved[0]] {
            assert!(
                matches!(output, QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Integer(1)),
                "{expression}: {output:?}"
            );
        }
    }
    let captured = evaluate(
        &mut document,
        "let selected=input | head(1); let row=user_info(selected); row.targets; (row.targets); legacy=user_info(selected); legacy.targets",
    );
    let expected = document.query(&ops().head(1).users()).unwrap();
    for output in captured {
        assert!(
            matches!(output, QueryOutput::Native(NativeValue::OperationSequence(operations)) if operations == expected)
        );
    }
    let native_fields = evaluate(
        &mut document,
        r#"let selected=input | filter(op("test.seed")) | head(1);
        let row=user_info(selected);
        row.targets | names; (row.targets) | names; (row.targets | names);
        let answer=row.targets; answer | names; (answer | names)"#,
    );
    let expected = document
        .query(&ops().filter(op("test.seed")).head(1).users().names())
        .unwrap();
    for output in native_fields {
        assert_eq!(output, QueryOutput::Values(expected.clone()));
    }
    for (expression, message) in [
        ("1 div 0", "zero"),
        ("9223372036854775807+1", "overflow"),
        (r#"1+"x""#, "numeric"),
    ] {
        let error = Query::parse(expression)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(&expression[error.range().unwrap()], expression);
    }
    let saved = evaluate(
        &mut document,
        "let numbers=[3,1,3]; numbers | sort | unique | count",
    );
    assert!(
        matches!(&saved[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Integer(2))
    );
    let functions = evaluate(
        &mut document,
        r#"fn numbers()->sequence<int> { [3,1,3] }
        fn distinct()->int { let local=[3,1,3]; local | sort | unique | count }
        fn filtered()->int { let local=[3,1,3]; local | filter(true) | unique | count }
        numbers() | sort | unique | count; distinct(); filtered();
        let words=["ash","birch","apple","ash"]; words | filter(starts_with("a")) | sort | unique | count"#,
    );
    for output in functions {
        assert!(
            matches!(&output, QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Integer(2)),
            "{output:?}"
        );
    }
    let siblings = evaluate(
        &mut document,
        r#"let numbers=[3,1,3]; report {"distinct":numbers | sort | unique | count, "input":input | count}"#,
    );
    assert!(
        matches!(&siblings[0], QueryOutput::Native(NativeValue::Constructed(value)) if matches!(value.value().data(), V::Record(fields) if fields[0].1.data()==&V::Integer(2) && fields[1].1.data()==&V::Integer(6)))
    );
    let outputs = evaluate(&mut document, "let numbers=[true]; numbers | sort");
    assert!(
        matches!(&outputs[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Array(vec![V::Boolean(true)]))
    );
    let request = r#"fn histogram()->dict<string,int> { let words=["ash"]; words | tally | reverse | head(1) } histogram()"#;
    assert_eq!(evaluate(&mut document, request).len(), 1);
    let native = evaluate(
        &mut document,
        r#"let saved=filter(op("test.scope")) | head(1); saved | children | sort_by(names)"#,
    );
    let QueryOutput::Operations(operations) = &native[0] else {
        panic!("native operations: {native:?}");
    };
    let expected = document
        .query(
            &ops()
                .filter(op("test.scope"))
                .head(1)
                .children()
                .sort_by(&input().names().one()),
        )
        .unwrap();
    assert_eq!(operations, &expected);
}

#[test]
fn shared_predicates_keep_constructed_elements_and_native_candidates() {
    use zirium::query::{ConstructedValue as V, NativeValue};
    let mut document = document();
    let mut run = |source: &str| {
        let mut values = Vec::new();
        Query::parse(source)
            .unwrap_or_else(|error| panic!("{source}: {error}"))
            .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
                values.push(value);
                Ok(())
            })
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        values
    };
    for source in [
        "let rows=[(1,2),(3,4)]; rows | filter((a,b) => a+b > 4)",
        "let rows=[(1,2),(3,4)]; [row for row in rows if row.1+row.2 > 4]",
        "let rows=[(1,2),(3,4)]; [row | row in rows where row.1+row.2 > 4]",
    ] {
        let values = run(source);
        assert!(
            matches!(&values[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Array(vec![V::Tuple(vec![V::Integer(3), V::Integer(4)])])),
            "{source}: {values:?}"
        );
    }
    let values = run(
        "fn positive(n:int)->bool { n>0 } let [first,second,..]=[1,2,3]; [[a,b] for (a,b) in [(first,second)] if positive(a)] | filter([a,b] => positive(a) and _[2]==b)",
    );
    assert!(
        matches!(&values[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Array(vec![V::Array(vec![V::Integer(1), V::Integer(2)])]))
    );
    let values = run(
        "let rows=[(items:[1,2],limit:1),(items:[1],limit:3)]; rows | filter(outer => length(outer.items | filter(_ > outer.limit)) > 0)",
    );
    assert!(
        matches!(&values[0], QueryOutput::Native(NativeValue::Constructed(value)) if matches!(value.value().data(), V::Array(rows) if rows.len()==1))
    );
    let values =
        run(r#"filter(op("test.seed")) | root(p => one(p | names)=="test.scope") | names"#);
    assert_eq!(values, [QueryOutput::Values(vec!["test.scope".into()])]);
    let values = run(
        r#"fn wanted(p:Operation)->bool { one(p | names)=="test.scope" } filter(op("test.seed")) | root(candidate => op(one(candidate | names)) and wanted(_) and wanted(candidate)) | names"#,
    );
    assert_eq!(values, [QueryOutput::Values(vec!["test.scope".into()])]);
    let values = run(
        r#"fn wanted(p:Operation)->bool { one(p | names)=="test.scope" } fn owner(p:Operation)->Operation { p | root(candidate => wanted(_) and op(one(candidate | names))) } let selected=filter(op("test.seed")); owner(selected) | names"#,
    );
    assert!(
        matches!(&values[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Array(vec![V::String("test.scope".into())]))
    );
    let values = run(
        "fn positive(n:int)->bool {n>0} fn keep(rows:sequence<int>)->sequence<int> {rows | filter(x => positive(_) and x>1)} keep([1,2,3])",
    );
    assert!(
        matches!(&values[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Array(vec![V::Integer(2), V::Integer(3)]))
    );
    let values = run(
        r#"filter(op("test.seed")) | paths(users,to=target => one(target | names)=="test.end",max_depth=2) | count"#,
    );
    assert_eq!(values, [QueryOutput::Count(1)]);
    let values = run(
        r#"fn nonempty(n:int)->bool {n>0} match { from seed in filter(op("test.seed")); where original => nonempty(original | count) and one(seed | names)=="test.seed"; } | count"#,
    );
    assert_eq!(values, [QueryOutput::Count(1)]);
    let values = run(
        r#"fn label()->string {"test.seed"} match { from seed in filter(op(label())); where original => one(seed | names)==label() and original | count > 0; } | count"#,
    );
    assert_eq!(values, [QueryOutput::Count(1)]);
    for source in [
        r#"match { from add in filter(op("test.add")); where original => same_value(operand(original | filter(op("test.add")),0),operand(add,1)); } | count"#,
        r#"match { from add in filter(op("test.add")); where same_value(operand(_ | filter(op("test.add")),0),operand(add,1)); } | count"#,
    ] {
        assert_eq!(run(source), [QueryOutput::Count(1)], "{source}");
    }
    let values = run(
        r#"fn same(p:Operation,label:string)->int { p | match { from add in filter(op(label)); where original => same_value(operand(original | filter(op(label)),0),operand(add,1)); } | count } same(filter(op("test.add")),"test.add")"#,
    );
    assert!(
        matches!(&values[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Integer(1))
    );
    for source in [
        "let [a,b]=[1]; a",
        "let rows=[[1]]; rows | filter([a,b] => true)",
        "fn select(rows:sequence<int>)->sequence<int> { rows | filter(_+1) } select([])",
        r#"filter((a,b) => true)"#,
    ] {
        let mut emitted = false;
        let error = Query::parse(source)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| {
                emitted = true;
                Ok(())
            })
            .err()
            .unwrap_or_else(|| panic!("expected refusal: {source}"));
        assert!(error.range().is_some(), "{source}: {error}");
        assert!(!emitted);
    }
}

#[test]
fn general_collection_keys_and_tally_render_at_the_json_boundary() {
    let mut document = document();
    let query = r#"fn histogram(xs:sequence<set<string>>) -> dict<set<string>,int> { xs | tally }
        report histogram([{"ash","apple"}, {"apple","birch"}, {"birch","apple"}]);"#;
    let mut outputs = Vec::new();
    Query::parse(query)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(report)) = &outputs[0] else {
        panic!("constructed histogram")
    };
    let mut budget = zirium::query::Budget::for_program(zirium::query::ProgramLimits {
        work: 10_000,
        results: 10_000,
        nesting: 64,
        stream_items: 1_000,
    });
    assert_eq!(
        report.to_json_with_budget(&mut budget).unwrap(),
        serde_json::json!({
            "kind": "dictionary", "entries": [
                {"key": ["ash", "apple"], "value": 1},
                {"key": ["apple", "birch"], "value": 2}
            ]
        })
    );
    outputs.clear();
    Query::parse(r#"report {[1,2]: {{"a"}, {"b"}}, ("x",1): {"a":2}};"#)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(report)) = &outputs[0] else {
        panic!("nested collection dictionary")
    };
    assert_eq!(
        report.to_json_with_budget(&mut budget).unwrap(),
        serde_json::json!({
            "kind": "dictionary", "entries": [
                {"key": [1, 2], "value": [["a"], ["b"]]},
                {"key": ["x", 1], "value": {"a": 2}}
            ]
        })
    );
}

#[test]
fn adopted_report_extensions_agree_between_text_and_builders() {
    use zirium::query::{
        ConstructedValue as V, Located, NativeValue, PatternKind, ReportExpr as E,
        ReportGenerator as G, report,
    };
    let pattern =
        |name: &str| Located::new(PatternKind::Name(Located::new(name.into(), 0..0)), 0..0);
    let ints = |values: &[i64]| E::array(values.iter().copied().map(E::integer));
    let row = |id| E::record([("id", E::integer(id))]);
    let expression = E::record([
        (
            "mapped",
            ints(&[1, 2]).map(
                pattern("n"),
                E::name("n").binary(zirium::query::ValueBinaryOperator::Multiply, E::integer(2)),
            ),
        ),
        (
            "flat",
            ints(&[1, 2]).flat_map(
                pattern("n"),
                E::array([
                    E::name("n"),
                    E::name("n").binary(zirium::query::ValueBinaryOperator::Add, E::integer(10)),
                ]),
            ),
        ),
        (
            "selected",
            E::comprehension(
                E::name("n"),
                [G::in_pattern(
                    Located::new(PatternKind::Present(Box::new(pattern("n"))), 0..0),
                    E::array([E::null(), E::integer(2)]),
                    None,
                    true,
                )],
            ),
        ),
        ("windows", ints(&[1, 2, 3]).windows(E::integer(2))),
        (
            "runs",
            E::array([row(1), row(1), row(2)])
                .runs_by("id")
                .map(pattern("run"), E::name("run").field("items").length()),
        ),
        (
            "updated",
            E::record([("id", E::integer(1)), ("n", E::integer(2))])
                .with_fields([("n".into(), E::integer(3))]),
        ),
        (
            "aligned",
            E::array([row(1)])
                .align(E::array([row(2)]), "id")
                .map(pattern("entry"), E::name("entry").field("key")),
        ),
    ]);
    let mut doc = document();
    let expected = doc.query(&report(expression)).unwrap();
    let source = r#"{
        "mapped": [1,2] | map(n => n * 2),
        "flat": [1,2] | flat_map(n => [n,n+10]),
        "selected": [n for case present(n) in [null,2]],
        "windows": [1,2,3] | windows(2),
        "runs": [{"id":1},{"id":1},{"id":2}] | runs_by("id") | map(length(_.items)),
        "updated": {"id":1,"n":2} with {n:3},
        "aligned": align([{"id":1}], [{"id":2}], by="id") | map(_.key)
    }"#;
    let mut actual = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut doc, DialectRegistry::baseline(), |_, value| {
            actual.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(value)) = &actual[0] else {
        panic!("constructed report")
    };
    assert_eq!(value.value().data(), expected.value().data());
    let V::Record(fields) = value.value().data() else {
        panic!("record")
    };
    assert_eq!(
        fields[0].1.data(),
        &V::Array(vec![V::Integer(2), V::Integer(4)])
    );
    assert_eq!(
        fields[4].1.data(),
        &V::Array(vec![V::Integer(2), V::Integer(1)])
    );

    let seed = doc.query(&ops().filter(op("test.seed"))).unwrap()[0];
    let rows = E::array([E::record([
        ("operation", E::native(&ops().filter(op("test.seed")))),
        ("n", E::integer(1)),
    ])])
    .map(
        pattern("row"),
        E::name("row").with_fields([("n".into(), E::integer(2))]),
    );
    let query = report(rows);
    let mut budget = query.new_budget(EvaluationLimits::default());
    let output = query
        .evaluate_output_with_budget(
            &doc,
            DialectRegistry::baseline(),
            EvaluationOptions::default(),
            &mut budget,
        )
        .unwrap();
    let QueryOutput::Native(NativeValue::Sequence(rows)) = output else {
        panic!("native positional report")
    };
    let QueryOutput::Native(NativeValue::Capture(row)) = &rows.values()[0] else {
        panic!("native row")
    };
    assert_eq!(row.fields[0].1, QueryOutput::Operations(vec![seed]));
    let QueryOutput::Native(NativeValue::Constructed(count)) = &row.fields[1].1 else {
        panic!("scalar field")
    };
    assert_eq!(count.value().data(), &V::Integer(2));
}

#[test]
fn markdown_native_capture_rows_render_through_the_library() {
    let mut document = document_from(r#""test.op"() : () -> ()"#);
    let mut outputs = Vec::new();
    Query::parse(include_str!(
        "../../../examples/cli/markdown-subjects.zirql"
    ))
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
        outputs.push(value);
        Ok(())
    })
    .unwrap();
    let QueryOutput::Text(text) = &outputs[0] else {
        panic!("Markdown text")
    };
    assert!(text.starts_with("\n| subject |\n| --- |\n"));
    assert!(text.contains("test.op"));
}

#[test]
fn scalar_conversion_reports_share_native_projections_and_callable_builders() {
    use zirium::query::{ConstructedValue as V, NativeValue, ReportExpr as E, report};
    let mut document = document_from(r#""test.amount"() {quantity = "002"} : () -> ()"#);
    let typed = report(E::record([
        (
            "quantity",
            E::from(ops().string_attr("quantity").one()).parse_int(),
        ),
        ("price", E::string("1.25e1").parse_float()),
        ("label", E::integer(2).to_string()),
    ]));
    let actual = document.query(&typed).unwrap();
    let mut outputs = Vec::new();
    Query::parse(r#"fn quantity(text:string)->int { parse_int(text) } report {"quantity":quantity(one(input | attr("quantity"))),"price":parse_float("1.25e1"),"label":to_string(2)};"#).unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| { outputs.push(value); Ok(()) }).unwrap();
    let QueryOutput::Native(NativeValue::Constructed(parsed)) = &outputs[0] else {
        panic!("constructed")
    };
    let V::Record(parsed_fields) = parsed.value().data() else {
        panic!("record")
    };
    assert_eq!(parsed_fields[0].1.data(), &V::Integer(2));
    assert_eq!(parsed_fields[1].1.data(), &V::Float(12.5));
    assert_eq!(parsed_fields[2].1.data(), &V::String("2".into()));
    assert_eq!(
        actual.value(),
        &V::Record(vec![
            ("quantity".into(), V::Integer(2)),
            ("price".into(), V::Float(12.5)),
            ("label".into(), V::String("2".into()))
        ])
    );
    assert!(
        document
            .query(&report(E::native(&ops()).to_string()))
            .is_err()
    );
}

#[test]
fn forward_functions_ignore_headers_in_strings_and_comments() {
    let mut document = document();
    let source = r#"
        # fn commented(value: int) -> int { 99 }
        let note = "fn quoted(value: int) -> int { 99 }";
        report later(4);
        fn later(value: int) -> int { value + 1 }
        report note | json;
    "#;
    let mut outputs = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(outputs.len(), 2);
    let QueryOutput::Native(zirium::query::NativeValue::Constructed(value)) = &outputs[0] else {
        panic!("constructed function result")
    };
    assert_eq!(
        value.value().data(),
        &zirium::query::ConstructedValue::Integer(5)
    );
    let QueryOutput::Json(text) = &outputs[1] else {
        panic!("explicit JSON string")
    };
    assert_eq!(text, "\"fn quoted(value: int) -> int { 99 }\"\n");
}

#[test]
fn collection_report_builders_share_runtime_bounds_and_dictionary_values() {
    use zirium::query::{
        ConstructedValue as V, Located, NativeValue, PatternKind, ReportExpr as E, report,
    };
    let mut document = document_from(r#""test.seed"() : () -> ()"#);
    let pattern = Located::new(PatternKind::Name(Located::new("n".into(), 0..0)), 0..0);
    let typed = report(
        E::array([E::integer(3), E::integer(1), E::integer(3)])
            .filter(
                pattern,
                E::name("n").binary(zirql::ValueBinaryOperator::Greater, E::integer(1)),
            )
            .head(E::integer(2))
            .tail(E::integer(9))
            .reverse()
            .check(Some(E::integer(2)), Some(E::string("kept rows"))),
    );
    let expected = document.query(&typed).unwrap();
    let source = r#"let limit=2; let message="kept" ++ " rows"; report [3,1,3] | filter(_>1) | head(limit) | tail(9) | reverse | check(limit,message); report {"a":1,"b":3,"c":2} | filter(_>1) | reverse | head(1); report length({1,2}); report {1:2} | filter(false) | check(0) | count;"#;
    let mut outputs = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(NativeValue::Constructed(actual)) = &outputs[0] else {
        panic!("constructed")
    };
    assert_eq!(actual.value().data(), expected.value().data());
    let QueryOutput::Native(NativeValue::Constructed(dictionary)) = &outputs[1] else {
        panic!("dictionary")
    };
    assert_eq!(
        dictionary.value().data(),
        &V::Record(vec![("c".into(), V::Integer(2))])
    );
    let mut native = Vec::new();
    Query::parse(
        r#"legacy={"a":1,"b":3,"c":2}; legacy | filter(_>1) | reverse | check(2); legacy | count"#,
    )
    .unwrap()
    .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
        native.push(value);
        Ok(())
    })
    .unwrap();
    assert_eq!(
        native[0],
        QueryOutput::Map(vec![
            ("c".into(), serde_json::json!(2)),
            ("b".into(), serde_json::json!(3))
        ])
    );
    assert_eq!(native[1], QueryOutput::Count(3));
    for source in ["report [] | head(true);", "report [] | check(0,7);"] {
        let mut emitted = false;
        let result = Query::parse(source).unwrap().evaluate(
            &mut document,
            DialectRegistry::baseline(),
            |_, _| {
                emitted = true;
                Ok(())
            },
        );
        assert!(result.is_err(), "{source}");
        assert!(!emitted);
    }
}

#[test]
fn report_unique_and_tally_retain_request_metadata_and_match_text() {
    use zirium::query::{
        ConstructedValue as V, FunctionDefinition as F, FunctionParameter, FunctionType as T,
        Located, NativeValue, ParameterDeclaration, ParameterType, ReportExpr as E, report,
    };
    let mut document = document();
    let echo = F::new(
        "echo",
        vec![FunctionParameter {
            name: Located::new("value".into(), 0..0),
            declaration: T::String,
        }],
        T::String,
        vec![],
        E::name("value"),
    )
    .unwrap();
    let words = E::from(
        report(E::array([
            E::call("echo", [E::name("word")]),
            E::string("add"),
            E::call("echo", [E::name("word")]),
        ]))
        .with_functions(vec![echo])
        .with_parameters(vec![ParameterDeclaration {
            name: Located::new("word".into(), 0..0),
            ty: Located::new(ParameterType::String, 0..0),
        }]),
    );
    let typed = report(E::record([
        ("unique", words.clone().unique()),
        ("tally", words.tally()),
    ]));
    let result = typed
        .evaluate_with_parameters(
            &document,
            DialectRegistry::baseline(),
            &[("word".into(), V::String("load".into()))],
            EvaluationLimits::default(),
        )
        .unwrap();
    let mut budget = typed.new_budget(EvaluationLimits::default());
    assert_eq!(
        result.to_json_with_budget(&mut budget).unwrap(),
        serde_json::json!({"unique": ["load", "add"], "tally": {"load": 2, "add": 1}})
    );
    Query::parse(r#"fn echo(value:string)->string { value } let words=[echo("load"),"add",echo("load")]; report {"unique":words | unique,"tally":words | tally};"#)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, output| {
            let QueryOutput::Native(NativeValue::Constructed(actual)) = output else {
                panic!("constructed report")
            };
            assert_eq!(actual.value().data(), result.value().data());
            Ok(())
        })
        .unwrap();
}

#[test]
fn native_tally_retains_equal_named_operations_and_composes_in_text_and_builders() {
    use zirium::query::NativeValue;
    let mut document = document();
    let selected = ops().filter(op("test.scope"));
    let expected = document.query(&selected).unwrap();
    let value = document.query(&selected.tally()).unwrap();
    let QueryOutput::Native(NativeValue::Dictionary(dictionary)) = value else {
        panic!("dictionary")
    };
    assert_eq!(dictionary.entries().len(), 2);
    for ((key, _), expected) in dictionary.entries().iter().zip(&expected) {
        assert_eq!(key, &QueryOutput::Operations(vec![*expected]));
    }
    let typed = document
        .query(&selected.tally().reverse().head(1).sort_by_value())
        .unwrap();
    let mut textual = Vec::new();
    Query::parse(r#"let keys=filter(op("test.scope")); let hist=keys | tally; let saved=hist; saved | reverse | head(1) | check(1) | sort_by(value); lookup(hist,keys | head(1))"#).unwrap().evaluate(&mut document, DialectRegistry::baseline(), |_, value| { textual.push(value); Ok(()) }).unwrap();
    let mut budget = zirql::Budget::for_program(zirql::ProgramLimits {
        work: 1_000_000,
        results: 100_000,
        nesting: 64,
        stream_items: 100_000,
    });
    assert_eq!(
        textual[0]
            .to_json_with_budget(&document, &mut budget)
            .unwrap(),
        typed.to_json_with_budget(&document, &mut budget).unwrap()
    );
    assert_eq!(textual[1], QueryOutput::Count(1));
}

#[test]
fn empty_native_tally_admits_its_retained_dictionary_schema() {
    let mut document = document();
    let valid = r#"fn empty(root:Operation)->dict<string,int> { root | head(0) | names | tally } empty(input | head(1))"#;
    let mut outputs = Vec::new();
    Query::parse(valid)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&outputs[0], QueryOutput::Native(zirium::query::NativeValue::Dictionary(value)) if value.entries().is_empty())
    );

    let nonempty = r#"fn counts(root:Operation)->dict<string,int> { root | names | tally } let saved=counts(input | head(1)); lookup(saved,one(input | head(1) | names))"#;
    Query::parse(nonempty)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(outputs[1], QueryOutput::Count(1));

    let invalid = r#"fn empty(root:Operation)->dict<int,int> { root | head(0) | names | tally } empty(input | head(1))"#;
    let mut emitted = false;
    let error = Query::parse(invalid)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, _| {
            emitted = true;
            Ok(())
        })
        .unwrap_err();
    assert!(error.range().is_some());
    assert!(!emitted);
}

#[test]
fn nested_declared_dictionaries_retain_native_operations() {
    let mut document = document();
    let expected = document.query(&ops().head(1)).unwrap()[0];
    let source = "fn nested(root:Operation)->dict<Operation,opt dict<Operation,Operation>> { {root:{root:root}} } nested(input | head(1))";
    let mut outputs = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let QueryOutput::Native(zirium::query::NativeValue::Dictionary(outer)) = &outputs[0] else {
        panic!("outer dictionary")
    };
    assert_eq!(
        outer.entries()[0].0,
        QueryOutput::Operations(vec![expected])
    );
    let QueryOutput::Native(zirium::query::NativeValue::Dictionary(inner)) = &outer.entries()[0].1
    else {
        panic!("inner dictionary")
    };
    assert_eq!(
        inner.entries()[0].0,
        QueryOutput::Operations(vec![expected])
    );
    assert_eq!(
        inner.entries()[0].1,
        QueryOutput::Operations(vec![expected])
    );
}

#[test]
fn optional_native_sequences_admit_present_operations_in_functions_and_queues() {
    use zirium::query::{ConstructedValue as V, NativeValue};
    let mut document = document_from(include_str!("../../../examples/cli/arithmetic.mlir"));
    let source = r#"
        fn size(d:sequence<opt Operation>)->int {1}
        fn first_present(nodes:sequence<opt Operation>)->bool {
            let present((first,_))=pop_front(queue<opt Operation>(nodes));
            occurs(first)
        }
        report size([n for n in input]);
        let selected=input | head(1);
        report first_present([selected]);
        report size(direct_children(selected));
        report occurs(pop_front(queue<opt Operation>(direct_children(selected))));
    "#;
    let mut outputs = Vec::new();
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    assert!(
        matches!(&outputs[0], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Integer(1))
    );

    assert!(
        matches!(&outputs[1], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Boolean(true))
    );

    assert!(
        matches!(&outputs[2], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Integer(1))
    );
    assert!(
        matches!(&outputs[3], QueryOutput::Native(NativeValue::Constructed(value)) if value.value().data() == &V::Boolean(true))
    );

    let incompatible = r#"
        fn empty(scope:Operation)->sequence<Operation> { direct_children(scope) | head(0) }
        fn accept(nodes:sequence<opt SSAValue>)->int { 1 }
        report accept(empty(input | head(1)));
    "#;
    assert!(
        Query::parse(incompatible)
            .unwrap()
            .evaluate(&mut document, DialectRegistry::baseline(), |_, _| Ok(()))
            .is_err()
    );
}

#[test]
fn report_field_ordering_and_grouping_retain_operation_subjects() {
    use zirium::query::NativeValue;
    use zirql::ConstructedValue as V;
    let mut document =
        document_from("\"alpha\"() : () -> ()\n\"beta\"() : () -> ()\n\"gamma\"() : () -> ()");
    let original = document.root_operations().to_vec();
    let source = r#"
        let rows = [{"key": if (n | names | one) == "beta" {"a"} else {"b"}, "subject": n} for n in input];
        report sort_by(rows,"key");
        report sort_by(rows,"key",descending=true);
        report group_by(rows,"key");
    "#;
    let mut outputs = vec![];
    Query::parse(source)
        .unwrap()
        .evaluate(&mut document, DialectRegistry::baseline(), |_, value| {
            outputs.push(value);
            Ok(())
        })
        .unwrap();
    let check_row = |row: &QueryOutput, index: usize| {
        let QueryOutput::Native(NativeValue::Capture(row)) = row else {
            panic!("native row")
        };
        let QueryOutput::Native(NativeValue::Constructed(key)) = &row.fields[0].1 else {
            panic!("key")
        };
        assert_eq!(
            key.value().data(),
            &V::String(if index == 1 { "a" } else { "b" }.into())
        );
        assert_eq!(
            row.fields[1].1,
            QueryOutput::Operations(vec![original[index]])
        );
    };
    for (output, expected) in outputs[..2].iter().zip([[1, 0, 2], [0, 2, 1]]) {
        let QueryOutput::Native(NativeValue::Sequence(rows)) = output else {
            panic!("ordered rows")
        };
        assert_eq!(rows.values().len(), 3);
        for (row, index) in rows.values().iter().zip(expected) {
            check_row(row, index);
        }
    }
    let QueryOutput::Native(NativeValue::Sequence(groups)) = &outputs[2] else {
        panic!("groups")
    };
    assert_eq!(groups.values().len(), 2);
    for (group, expected) in groups.values().iter().zip([vec![0, 2], vec![1]]) {
        let QueryOutput::Native(NativeValue::Capture(group)) = group else {
            panic!("group")
        };
        let QueryOutput::Native(NativeValue::Sequence(items)) = &group.fields[1].1 else {
            panic!("group items")
        };
        assert_eq!(items.values().len(), expected.len());
        for (row, index) in items.values().iter().zip(expected) {
            check_row(row, index);
        }
    }
}
