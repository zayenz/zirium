use zirium::{
    dialect::{OperationShape, RegistryConfig},
    parser::{ParseDiagnosticKind, ParsedFile},
    semantic::{LoweringMode, lower_with_dialect_registry},
};

fn registry() -> zirium::dialect::DialectRegistry {
    RegistryConfig::from_json(
        r#"{
          "builtins": [],
          "operation_shapes": [
            {"name":"test.empty","shape":"attr_first_optional_typed_operands"},
            {"name":"test.one","shape":"attr_first_optional_typed_operands"},
            {"name":"test.many","shape":"attr_first_optional_typed_operands"}
          ]
        }"#,
    )
    .unwrap()
    .build()
    .unwrap()
}

#[test]
fn attr_first_optional_typed_operands_preserve_empty_and_typed_forms() {
    let registry = registry();
    assert_eq!(
        registry.operation_shape("test.one"),
        Some(OperationShape::AttrFirstOptionalTypedOperands)
    );
    let source = br#""builtin.module"() ({
^bb0:
  %integer = "test.source"() : () -> i32
  %tuple = "test.source"() : () -> tuple<f32, i64>
  test.empty
  "test.boundary"() : () -> ()
  test.empty {tag = "empty", typed = 7 : i32}
  "test.boundary"() : () -> ()
  test.empty {} : ()
  test.one {tag = "one"} %integer : i32
  test.many {tag = "many"} %integer, %tuple : (i32, tuple<f32, i64>)
}) : () -> ()"#;

    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();

    let operations = document
        .operations()
        .filter(|operation| {
            document
                .operation_name(*operation)
                .is_some_and(|name| registry.operation_shape(name).is_some())
        })
        .collect::<Vec<_>>();
    assert_eq!(operations.len(), 5);
    let signatures = operations
        .iter()
        .map(|operation| document.type_spelling(document.function_type(*operation).unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        signatures,
        [
            Some("() -> ()"),
            Some("() -> ()"),
            Some("() -> ()"),
            Some("(i32) -> ()"),
            Some("(i32, tuple<f32, i64>) -> ()"),
        ]
    );
    for operation in operations {
        assert!(document.result_types(operation).unwrap().is_empty());
    }
    let attributed = document
        .operations()
        .find(|operation| {
            document.operation_name(*operation) == Some("test.empty")
                && document
                    .attributes(*operation)
                    .is_some_and(|mut attributes| attributes.any(|(name, _)| name == "typed"))
        })
        .unwrap();
    assert!(
        document
            .attributes(attributed)
            .unwrap()
            .any(|(name, value)| { name == "typed" && value == "7 : i32" })
    );
}

#[test]
fn attr_first_shape_rejects_reordered_or_incomplete_typed_forms_at_boundaries() {
    let registry = registry();
    for malformed in [
        "test.one %value {tag = true} : i32",
        "test.many %value, %value : (i32)",
    ] {
        let source = format!(
            "%value = \"test.source\"() : () -> i32\n{malformed}\n\"test.after\"() : () -> ()"
        );
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
            diagnostic.kind()
                == ParseDiagnosticKind::ShapeMismatch(
                    OperationShape::AttrFirstOptionalTypedOperands,
                )
        }));
        let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
        let document = lowered.document.unwrap();
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some("test.after"))
        );
    }

    let parsed = ParsedFile::parse_with_registry(
        b"%value = \"test.source\"() : () -> i32\ntest.one %value : (i32".as_slice(),
        &registry,
    )
    .unwrap();
    assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
        diagnostic.kind()
            == ParseDiagnosticKind::ShapeMismatch(OperationShape::AttrFirstOptionalTypedOperands)
    }));
}

#[test]
fn attr_first_shape_never_infers_results_from_result_bindings() {
    let registry = registry();
    let parsed = ParsedFile::parse_with_registry(
        b"%result = test.empty {tag = true}\n\"test.after\"() : () -> ()".as_slice(),
        &registry,
    )
    .unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.document.is_none());
    assert!(
        lowered
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("result definition count 1"))
    );
}

#[test]
fn consumed_newline_still_ends_a_shaped_operation() {
    let registry = registry();
    let parsed = ParsedFile::parse_with_registry(
        b"test.empty {tag = true}\ntest.empty\n".as_slice(),
        &registry,
    )
    .unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
}
