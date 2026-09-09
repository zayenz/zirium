use zirium::{
    SyntaxKind,
    dialect::DialectRegistry,
    parser::ParsedFile,
    printer::PrintLayout,
    semantic::{LoweringMode, TypeValue, lower_with_dialect_registry},
};

#[test]
fn complex_types_parse_lower_intern_and_print_canonically() {
    let parsed = ParsedFile::parse(
        b"!component = type f32\n%results:2 = \"uses\"() : (complex<!component>, tensor<2xcomplex<i32>>) -> (complex<f32>, complex<f32>)"
            .to_vec(),
    )
    .unwrap();
    assert!(parsed.lexer_diagnostics().is_empty());
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    parsed.syntax().tree().verify().unwrap();
    assert_eq!(
        parsed
            .syntax()
            .file()
            .nodes(SyntaxKind::ComplexType)
            .count(),
        4
    );

    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let operation = document.root_operations()[0];
    let results = document.result_types(operation).unwrap();
    assert_eq!(results[0], results[1]);
    assert!(matches!(
        document.type_value(results[0]),
        Some(TypeValue::Complex(element))
            if matches!(element.as_ref(), TypeValue::Float(name) if name == "f32")
    ));

    let mut printed = String::new();
    document.print(&mut printed, PrintLayout::Compact).unwrap();
    assert!(printed.contains("complex<f32 >"), "{printed}");
    assert!(printed.contains("tensor<2xcomplex<i32 > >"), "{printed}");
    let reparsed = ParsedFile::parse(printed.as_bytes()).unwrap();
    let reparsed =
        lower_with_dialect_registry(&reparsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
            .document
            .unwrap();
    assert!(document.structurally_eq(&reparsed));
}

#[test]
fn malformed_complex_types_are_lossless_and_recover_the_next_operation() {
    for source in [
        b"\"bad\"() : (complex<>) -> ()\n\"after\"() : () -> ()".as_slice(),
        b"\"bad\"() : (complex<i32, f32>) -> ()\n\"after\"() : () -> ()".as_slice(),
    ] {
        let parsed = ParsedFile::parse(source.to_vec()).unwrap();
        assert_eq!(parsed.original_bytes(), source);
        assert!(!parsed.syntax().diagnostics().is_empty());
        parsed.syntax().tree().verify().unwrap();
        assert_eq!(parsed.syntax().file().operations().count(), 2);
    }

    let nested = format!(
        "%result = \"deep\"() : () -> {}f32{}\n\"after\"() : () -> ()",
        "complex<".repeat(65),
        ">".repeat(65)
    );
    let parsed = ParsedFile::parse(nested.as_bytes()).unwrap();
    assert_eq!(parsed.original_bytes(), nested.as_bytes());
    assert!(!parsed.syntax().diagnostics().is_empty());
    parsed.syntax().tree().verify().unwrap();
    assert_eq!(parsed.syntax().file().operations().count(), 2);
}

#[test]
fn complex_rejects_non_integer_and_non_float_elements_semantically() {
    let parsed = ParsedFile::parse(b"%result = \"bad\"() : () -> complex<index>".to_vec()).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY);
    assert!(lowered.document.is_none());
    assert!(
        lowered
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message == "invalid element type for complex"),
        "{:?}",
        lowered.diagnostics
    );
}
