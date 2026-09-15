use zirium::{
    dialect::DialectRegistry,
    diff::{ChangeField, DiffLimits, DiffOptions, compare},
    parser::ParsedFile,
    query::{changed, changes, dialect, diff_op_input, op},
    semantic::{Document, LoweringMode, lower_with_dialect_registry},
};

fn document(source: &str) -> Document {
    let registry = DialectRegistry::baseline();
    let parsed = ParsedFile::parse_with_registry(source.as_bytes(), registry).unwrap();
    lower_with_dialect_registry(&parsed, LoweringMode::Strict, registry)
        .document
        .unwrap()
}

#[test]
fn typed_change_filters_project_and_navigate_on_the_after_document() {
    let before =
        document("module { %value = arith.constant 4 : i32 \"test.use\"(%value) : (i32) -> () }");
    let after = document(
        "module { %renamed = arith.constant 8 : i32 \"test.use\"(%renamed) : (i32) -> () }",
    );
    let comparison = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();

    let rewired = changes().filter(changed(ChangeField::Attributes) & dialect("arith"));
    assert_eq!(comparison.query(&rewired).unwrap().len(), 1);

    let consumers = rewired.after().users().unique().names();
    assert_eq!(comparison.query(&consumers).unwrap(), ["test.use"]);
    assert_eq!(comparison.query(&rewired.before().count()).unwrap(), 1);
    assert_eq!(comparison.query(&rewired.attr("value")).unwrap(), ["8"]);
    assert_eq!(comparison.query(&rewired.result_types()).unwrap(), ["i32"]);
    assert_eq!(
        comparison
            .query(&rewired.after().users().operand_types())
            .unwrap(),
        ["i32"]
    );
    assert_eq!(
        comparison.query(&changes().names().sort().min()).unwrap(),
        ["arith.constant"]
    );

    let roots = rewired.after().root(op("builtin.module")).unique();
    assert_eq!(comparison.query(&roots).unwrap().len(), 1);
    assert_eq!(
        comparison
            .query(
                &rewired
                    .after()
                    .root(op("builtin.module"))
                    .subtree()
                    .filter(op("test.use"))
                    .names(),
            )
            .unwrap(),
        ["test.use"]
    );
    assert_eq!(
        comparison
            .query(&changes().reverse().head(1).tail(1).count())
            .unwrap(),
        1
    );
}

#[test]
fn typed_projected_graph_traversals_use_the_selected_document() {
    let before = document(
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %x, %y : i32 }",
    );
    let after = document(
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %y, %x : i32 }",
    );
    let comparison = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let changed_add = changes().filter(changed(ChangeField::Operands)).after();
    let expected = ["arith.constant", "arith.constant", "arith.addi"];
    assert_eq!(
        comparison.query(&changed_add.slice().names()).unwrap(),
        expected
    );
    assert_eq!(
        comparison.query(&changed_add.closure().names()).unwrap(),
        expected
    );
    assert_eq!(
        comparison.query(&changed_add.reachable().names()).unwrap(),
        expected
    );
    assert_eq!(
        comparison
            .query(&changed_add.fixpoint(&diff_op_input().closure()).names())
            .unwrap(),
        expected
    );
}
