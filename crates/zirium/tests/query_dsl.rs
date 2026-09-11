use std::collections::BTreeMap;
use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    query::{EvaluationLimits, Query, QueryOutput, always, has_attr, input, op, ops},
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
    let parsed =
        ParsedFile::parse_with_registry(SOURCE.as_bytes(), DialectRegistry::baseline()).unwrap();
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::baseline());
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    lowered.document.unwrap()
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
