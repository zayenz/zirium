use zirium::{
    dialect::DialectRegistry,
    diff::{ChangeField, ChangeKind, DiffLimits, DiffOptions, compare},
    parser::ParsedFile,
    semantic::{
        Document, LoweringMode, RetentionProfile, lower_with_dialect_registry,
        lower_with_dialect_registry_and_retention,
    },
};

fn document(source: &str) -> Document {
    let parsed =
        ParsedFile::parse_with_registry(source.as_bytes(), DialectRegistry::baseline()).unwrap();
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::baseline());
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    lowered.document.unwrap()
}

#[test]
fn rejects_unrepresented_resources_under_semantic_retention() {
    let parsed = ParsedFile::parse(
        b"\"payload\"() {value = dense_resource<handle> : tensor<4xi32>} : () -> ()".as_slice(),
    )
    .unwrap();
    let resource =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
            .document
            .unwrap();
    assert!(
        compare(
            &resource,
            &resource,
            &DialectRegistry::EMPTY,
            DiffOptions::default(),
            DiffLimits::default()
        )
        .is_err()
    );

    let parsed = ParsedFile::parse(
        b"\"work\"() : () -> ()\n{-# dialect_resources: { value: 1 } #-}".as_slice(),
    )
    .unwrap();
    let metadata = lower_with_dialect_registry_and_retention(
        &parsed,
        LoweringMode::Strict,
        RetentionProfile::SemanticOnly,
        &DialectRegistry::EMPTY,
    )
    .document
    .unwrap();
    assert!(
        metadata
            .comparison_coverage()
            .unrepresented_file_metadata
            .is_some()
    );
    assert!(
        compare(
            &metadata,
            &metadata,
            &DialectRegistry::EMPTY,
            DiffOptions::default(),
            DiffLimits::default()
        )
        .is_err()
    );
}

fn compare_documents<'a>(before: &'a Document, after: &'a Document) -> zirium::diff::Diff<'a> {
    compare(
        before,
        after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap()
}

#[test]
fn ignores_formatting_comments_and_ssa_names() {
    let before = document(
        r#"module { func.func @main(%left: i32, %right: i32) -> i32 {
          %result = arith.addi %left, %right : i32
          func.return %result : i32
        } }"#,
    );
    let after = document(
        r#"// formatting is not structure
        module { func.func @main(%a: i32, %b: i32) -> i32 {
          %sum = arith.addi %a, %b : i32 // renamed
          func.return %sum : i32
        } }"#,
    );
    assert!(compare_documents(&before, &after).is_empty());
}

#[test]
fn reports_changed_attribute_without_consumer_noise() {
    let before = document(
        r#"module { func.func @main() {
          %value = arith.constant 4 : i32
          func.return %value : i32
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() {
          %renamed = arith.constant 8 : i32
          func.return %renamed : i32
        } }"#,
    );
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 1, "{}", diff.to_text());
    assert_eq!(diff.changes()[0].kind(), ChangeKind::Modified);
    assert_eq!(diff.changes()[0].fields(), &[ChangeField::Attributes]);
    assert!(!diff.to_json().unwrap().contains("zirium"));
}

#[test]
fn insertion_preserves_the_unchanged_suffix() {
    let before = document(
        r#"module { func.func @main() {
          "test.a"() : () -> ()
          "test.b"() : () -> ()
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() {
          "test.inserted"() : () -> ()
          "test.a"() : () -> ()
          "test.b"() : () -> ()
        } }"#,
    );
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 1, "{}", diff.to_text());
    assert_eq!(diff.changes()[0].kind(), ChangeKind::Added);
}

#[test]
fn reordered_independent_operations_are_moves() {
    let before = document(
        r#"module { func.func @main() {
          "test.a"() : () -> ()
          "test.b"() : () -> ()
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() {
          "test.b"() : () -> ()
          "test.a"() : () -> ()
        } }"#,
    );
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 2, "{}", diff.to_text());
    assert!(diff.changes().iter().all(|change| {
        change.kind() == ChangeKind::Moved
            && change.moved()
            && change.fields() == [ChangeField::Position]
    }));
}

#[test]
fn moved_and_modified_operation_has_one_record() {
    let before = document(
        r#"module { func.func @main() {
          "test.a"() {value = 1 : i32} : () -> ()
          "test.b"() : () -> ()
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() {
          "test.b"() : () -> ()
          "test.a"() {value = 2 : i32} : () -> ()
        } }"#,
    );
    let diff = compare_documents(&before, &after);
    let changed = diff
        .changes()
        .iter()
        .find(|change| change.kind() == ChangeKind::Modified)
        .expect("modified operation");
    assert!(changed.moved());
    assert_eq!(
        changed.fields(),
        [ChangeField::Attributes, ChangeField::Position]
    );
    assert_eq!(diff.len(), 2, "{}", diff.to_text());
}

#[test]
fn operand_swap_changes_only_the_consumer_edges() {
    let before = document(
        r#"module { func.func @main() -> i32 {
          %a = arith.constant 1 : i32
          %b = arith.constant 2 : i32
          %r = "test.sub"(%a, %b) : (i32, i32) -> i32
          func.return %r : i32
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() -> i32 {
          %left = arith.constant 1 : i32
          %right = arith.constant 2 : i32
          %r = "test.sub"(%right, %left) : (i32, i32) -> i32
          func.return %r : i32
        } }"#,
    );
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 1, "{}", diff.to_text());
    assert_eq!(diff.changes()[0].fields(), [ChangeField::Operands]);
    assert_eq!(
        diff.changes()[0]
            .details()
            .iter()
            .map(|detail| detail.path.as_str())
            .collect::<Vec<_>>(),
        ["/operands/0", "/operands/1"]
    );
    let json: serde_json::Value = serde_json::from_str(&diff.to_json().unwrap()).unwrap();
    assert_eq!(json[0]["details"][0]["before"]["value"]["result"], 0);
    assert!(json[0]["details"][0]["before"]["value"]["definition"].is_string());
}

#[test]
fn reordered_blocks_change_the_region_shell_without_child_noise() {
    let before = document(
        r#"module { func.func @main() {
        ^entry:
          cf.br ^left
        ^left:
          "test.left"() : () -> ()
          cf.br ^right
        ^right:
          "test.right"() : () -> ()
          func.return
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() {
        ^entry:
          cf.br ^left
        ^right:
          "test.right"() : () -> ()
          func.return
        ^left:
          "test.left"() : () -> ()
          cf.br ^right
        } }"#,
    );
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 1, "{}", diff.to_text());
    assert_eq!(diff.changes()[0].fields(), [ChangeField::Regions]);
}

#[test]
fn renamed_registered_symbol_is_removal_and_addition() {
    let before = document("module { func.func @before() { func.return } }");
    let after = document("module { func.func @after() { func.return } }");
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 4, "{}", diff.to_text());
    assert_eq!(
        diff.changes()
            .iter()
            .filter(|change| change.kind() == ChangeKind::Added)
            .count(),
        2
    );
    assert_eq!(
        diff.changes()
            .iter()
            .filter(|change| change.kind() == ChangeKind::Removed)
            .count(),
        2
    );
}

#[test]
fn locations_are_optional_comparison_data() {
    let before = document(r#""test.op"() : () -> () loc("before":1:1)"#);
    let after = document(r#""test.op"() : () -> () loc("after":2:3)"#);
    assert!(compare_documents(&before, &after).is_empty());
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions {
            compare_locations: true,
        },
        DiffLimits::default(),
    )
    .unwrap();
    assert_eq!(diff.len(), 1);
    assert_eq!(diff.changes()[0].fields(), [ChangeField::Location]);
}

#[test]
fn attribute_details_use_json_pointer_paths_and_explicit_absence() {
    let before = document(r#""test.op"() {"path/name" = 1 : i32} : () -> ()"#);
    let after = document(r#""test.op"() {added = true} : () -> ()"#);
    let diff = compare_documents(&before, &after);
    let details = diff.changes()[0].details();
    assert_eq!(
        details
            .iter()
            .map(|detail| detail.path.as_str())
            .collect::<Vec<_>>(),
        ["/attributes/added", "/attributes/path~1name"]
    );
    assert!(!details[0].before.present);
    assert!(!details[1].after.present);
}

#[test]
fn opaque_value_statistics_count_distinct_payloads() {
    let source =
        r#""test.op"() {first = #vendor.data<abc>, second = #vendor.data<abc>} : () -> ()"#;
    let before = document(source);
    let after = document(source);
    let diff = compare_documents(&before, &after);
    assert!(diff.is_empty());
    assert_eq!(diff.statistics().opaque_before, 1);
    assert_eq!(diff.statistics().opaque_after, 1);
}

#[test]
fn matched_consumer_slots_disambiguate_changed_producers() {
    let before = document(
        r#"module {
          %a = arith.constant 1 : i32
          %b = arith.constant 2 : i32
          %r = "test.combine"(%a, %b) : (i32, i32) -> i32
        }"#,
    );
    let after = document(
        r#"module {
          %a = arith.constant 3 : i32
          %b = arith.constant 4 : i32
          %r = "test.combine"(%a, %b) : (i32, i32) -> i32
        }"#,
    );
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 2, "{}", diff.to_text());
    assert!(diff.changes().iter().all(|change| {
        change.kind() == ChangeKind::Modified && change.fields() == [ChangeField::Attributes]
    }));
}

#[test]
fn changed_successor_reports_structural_block_endpoints() {
    let before = document(
        r#"module { func.func @main() {
        ^entry:
          cf.br ^left
        ^left:
          "test.left"() : () -> ()
          func.return
        ^right:
          "test.right"() : () -> ()
          func.return
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() {
        ^entry:
          cf.br ^right
        ^left:
          "test.left"() : () -> ()
          func.return
        ^right:
          "test.right"() : () -> ()
          func.return
        } }"#,
    );
    let diff = compare_documents(&before, &after);
    assert_eq!(diff.len(), 1, "{}", diff.to_text());
    let change = &diff.changes()[0];
    assert_eq!(change.fields(), [ChangeField::Successors]);
    assert_eq!(change.details()[0].path, "/successors/0");
    let before = &change.details()[0].before.value;
    let after = &change.details()[0].after.value;
    assert_ne!(before, after);
}
