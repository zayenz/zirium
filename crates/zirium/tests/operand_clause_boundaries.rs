use zirium::{
    SyntaxKind,
    dialect::{DialectRegistry, OperationShape},
    parser::{ParseDiagnosticKind, ParsedFile},
};

fn parse(source: &str) -> ParsedFile {
    let registry =
        DialectRegistry::with_operation_shapes(&[("test.multi", OperationShape::OperandClauses)])
            .unwrap();
    ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap()
}

#[test]
fn shared_type_list_preserves_the_following_operation_boundary() {
    for (types, separator) in [
        ("i32, i64", "\n"),
        ("i32 , i64", "  // result types end here\n  "),
    ] {
        let source = format!("%first, %second = test.multi %arg : {types}{separator}func.return");
        let parsed = parse(&source);

        assert!(
            parsed.syntax().diagnostics().is_empty(),
            "unexpected diagnostics for {types:?} and {separator:?}: {:?}",
            parsed.syntax().diagnostics()
        );
        let operations = parsed.syntax().file().operations().collect::<Vec<_>>();
        assert_eq!(operations.len(), 2, "{types:?} and {separator:?}");
        assert_eq!(
            operations[0].tree().kind(operations[0].id()),
            Some(SyntaxKind::DialectOperation),
            "{types:?} and {separator:?}"
        );
    }
}

#[test]
fn shared_type_list_still_accepts_a_same_line_location() {
    let parsed = parse("%first, %second = test.multi %arg : i32, i64  loc(unknown)");

    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    assert_eq!(parsed.syntax().file().operations().count(), 1);
}

#[test]
fn shared_type_list_does_not_turn_same_line_bare_syntax_into_a_boundary() {
    let parsed = parse("%first, %second = test.multi %arg : i32, i64 next.op");

    assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
        diagnostic.kind() == ParseDiagnosticKind::ShapeMismatch(OperationShape::OperandClauses)
    }));
}

#[test]
fn variadic_shared_type_preserves_the_following_operation_boundary() {
    let registry = DialectRegistry::with_operation_shapes(&[(
        "test.results",
        OperationShape::VariadicOperands,
    )])
    .unwrap();
    let parsed = ParsedFile::parse_with_registry(
        b"%result = test.results : index\nfunc.return".as_slice(),
        &registry,
    )
    .unwrap();

    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let operations = parsed.syntax().file().operations().collect::<Vec<_>>();
    assert_eq!(operations.len(), 2);
    assert_eq!(
        operations[0].tree().kind(operations[0].id()),
        Some(SyntaxKind::DialectOperation)
    );
}
