use zirium::{
    SyntaxKind,
    dialect::{DialectRegistry, OperationShape},
    parser::{ParseLimits, ParsedFile},
};

fn registry() -> DialectRegistry {
    DialectRegistry::with_operation_shapes(&[
        ("test.operands", OperationShape::OperandClauses),
        ("test.region", OperationShape::RegionClauses),
    ])
    .unwrap()
}

#[test]
fn named_delimited_rhs_captures_ssa_operands_for_clause_shapes() {
    let registry = registry();
    let source = br#"%packed = test.operands %source inner_dims_pos = [0] inner_tiles = [%tile, %other] into %dest : (tensor<8xf32>, index, index, tensor<4x2xf32>) -> tensor<4x2xf32>
test.region values = [%first, %second] {
  "test.done"() : () -> ()
}"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();

    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let operations = parsed.syntax().file().operations().collect::<Vec<_>>();
    assert_eq!(operations[0].operands().count(), 4);
    assert_eq!(operations[1].operands().count(), 2);
    assert_eq!(operations[1].regions().count(), 1);
    assert_eq!(
        parsed
            .syntax()
            .file()
            .nodes(SyntaxKind::ArrayAttribute)
            .count(),
        1
    );
}

#[test]
fn named_delimited_attributes_without_ssa_keep_attribute_syntax() {
    let registry = registry();
    let source = br#"%result = test.operands %source dims = [] sizes = [2, 4] config = {rank = 2} data = dense<[1, 2]> : tensor<2xi32> : tensor<2xi32>"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();

    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let operation = parsed.syntax().file().operations().next().unwrap();
    assert_eq!(operation.operands().count(), 1);
    assert_eq!(
        parsed
            .syntax()
            .file()
            .nodes(SyntaxKind::ArrayAttribute)
            .count(),
        2
    );
    assert_eq!(
        parsed
            .syntax()
            .file()
            .nodes(SyntaxKind::DictionaryAttribute)
            .count(),
        1
    );
    assert_eq!(
        parsed
            .syntax()
            .file()
            .nodes(SyntaxKind::DenseElementsAttribute)
            .count(),
        1
    );
}

#[test]
fn malformed_deep_named_rhs_keeps_recovery_bounded() {
    let registry = registry();
    let source = b"test.operands values = [[[%value]]] : index\n\"after\"() : () -> ()";
    let parsed = ParsedFile::parse_with_limits_and_registry(
        source.as_slice(),
        ParseLimits {
            max_delimiter_depth: 2,
            ..ParseLimits::default()
        },
        &registry,
    )
    .unwrap();

    assert!(!parsed.syntax().diagnostics().is_empty());
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    parsed.syntax().tree().verify().unwrap();
}
