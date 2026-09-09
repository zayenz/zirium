use zirium::{
    dialect::{DeclarativeRegistryError, RegistryConfig},
    parser::{ParseDiagnosticKind, ParsedFile},
    semantic::{LoweringMode, lower_with_dialect_registry},
};

const INTO_FORMAT: &str = "$operands attr-dict `:` type($operands) `into` type($results)";

fn registry(format: &str) -> RegistryConfig {
    RegistryConfig::from_json(&format!(
        r#"{{"builtins":[],"operation_shapes":[],"operation_formats":[
          {{"name":"test.widen","format":"{format}"}}
        ]}}"#
    ))
    .unwrap()
}

#[test]
fn into_result_separator_parses_and_lowers_widening_operands() {
    let registry = registry(INTO_FORMAT).build().unwrap();
    let source = br#""builtin.module"() ({
^bb0:
  %lhs = "test.source"() : () -> vector<[8]xf16>
  %rhs = "test.source"() : () -> vector<[8]xf16>
  %result = test.widen %lhs, %rhs {tag = true} : vector<[8]xf16> into vector<[4]x[4]xf32>
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.widen"))
        .unwrap();

    assert_eq!(document.operands(operation).unwrap().len(), 2);
    assert_eq!(
        document
            .result_types(operation)
            .unwrap()
            .iter()
            .map(|ty| document.type_spelling(*ty).unwrap())
            .collect::<Vec<_>>(),
        ["vector<[4]x[4]xf32>"]
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );
}

#[test]
fn conversion_separator_follows_nested_affine_arrow_in_operand_type() {
    let registry = registry("$operands attr-dict `:` type($operands) `to` type($results)")
        .build()
        .unwrap();
    let source = br#""builtin.module"() ({
^bb0:
  %input = "test.source"() : () -> !test.opaque<affine_map<(d0) -> (d0)>>
  %result = test.widen %input : !test.opaque<affine_map<(d0) -> (d0)>> to !test.opaque<affine_map<(d0) -> (d0 + 1)>>
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.widen"))
        .unwrap();

    assert_eq!(
        document
            .result_types(operation)
            .unwrap()
            .iter()
            .map(|ty| document.type_spelling(*ty).unwrap())
            .collect::<Vec<_>>(),
        ["!test.opaque<affine_map<(d0) -> (d0 + 1)>>"]
    );
}

#[test]
fn into_and_to_are_distinct_format_literals() {
    let to = registry("$operands attr-dict `:` type($operands) `to` type($results)");
    let into = registry(INTO_FORMAT);
    assert!(matches!(
        RegistryConfig::build_many(&[to, into]),
        Err(DeclarativeRegistryError::ConflictingFormat(name))
            if name == "test.widen"
    ));

    let registry = registry(INTO_FORMAT).build().unwrap();
    let parsed = ParsedFile::parse_with_registry(
        b"%result = test.widen %lhs, %rhs : i16 to i32".as_slice(),
        &registry,
    )
    .unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::FormatMismatch)
    );
}

#[test]
fn trailing_synthetic_format_steps_preserve_a_newline_boundary() {
    let registry = RegistryConfig::from_json(
        r#"{
          "builtins":["func.return"],
          "operation_shapes":[],
          "operation_formats":[{
            "name":"test.widen",
            "format":"$operands attr-dict `:` type($operands) `to` type($results)"
          }]
        }"#,
    )
    .unwrap()
    .build()
    .unwrap();
    let source = br#"%input = "test.source"() : () -> i16
%result = test.widen %input : i16 to i32
func.return"#;

    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
}
