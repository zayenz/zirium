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
        r#"module { func.func @main() -> i32 {
          %value = arith.constant 4 : i32
          func.return %value : i32
        } }"#,
    );
    let after = document(
        r#"module { func.func @main() -> i32 {
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
