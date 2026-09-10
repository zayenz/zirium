use zirium::{
    dialect::{DeclarativeRegistryError, RegistryConfig},
    parser::{ParseDiagnosticKind, ParsedFile},
    query::{Query, QueryOutput},
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

#[test]
fn arrow_format_with_indexed_operands_enforces_arity() {
    let format = "$operands[0] `,` $operands[1] attr-dict `:` type($operands) `->` type($results)";
    let registry = registry(format).build().unwrap();
    let source = br#"%lhs = "test.source"() : () -> f32
%rhs = "test.source"() : () -> f32
%result = test.widen %lhs, %rhs : f32 -> i1"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.widen"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.function_type(operation).unwrap()),
        Some("(f32, f32) -> i1")
    );

    let malformed = ParsedFile::parse_with_registry(
        b"%result = test.widen %lhs, %rhs, %extra : f32 -> i1".as_slice(),
        &registry,
    )
    .unwrap();
    assert!(
        malformed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| { diagnostic.kind() == ParseDiagnosticKind::FormatMismatch })
    );
}

#[test]
fn composed_type_targets_describe_nonuniform_operand_types() {
    let format = "$operands[0] `,` $operands[1] `,` $operands[2] attr-dict `:` type($operands[0]) `,` type($operands[1], $operands[2]) `->` type($results)";
    let registry = registry(format).build().unwrap();
    let source = br#"%condition = "test.source"() : () -> i1
%left = "test.source"() : () -> f32
%right = "test.source"() : () -> f32
%result = test.widen %condition, %left, %right : i1, f32 -> f32"#;
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
        document.type_spelling(document.function_type(operation).unwrap()),
        Some("(i1, f32, f32) -> f32")
    );
}

#[test]
fn per_operand_type_list_requires_one_type_per_operand() {
    let format = "$operands attr-dict `:` types($operands) `->` type($results)";
    let registry = registry(format).build().unwrap();
    let parsed = ParsedFile::parse_with_registry(
        br#"%lhs = "test.source"() : () -> i1
%rhs = "test.source"() : () -> f32
%result = test.widen %lhs, %rhs : i1, f32 -> i32"#
            .as_slice(),
        &registry,
    )
    .unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.widen"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.function_type(operation).unwrap()),
        Some("(i1, f32) -> i32")
    );
}

#[test]
fn callee_only_format_lowers_symbol_attribute() {
    let registry = registry("$callee `as` attr-dict").build().unwrap();
    let parsed = ParsedFile::parse_with_registry(
        b"test.widen @outer::@inner as {tag = true}".as_slice(),
        &registry,
    )
    .unwrap();
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
            .attributes(operation)
            .unwrap()
            .find(|(name, _)| *name == "callee"),
        Some(("callee", "@outer::@inner"))
    );
}

#[test]
fn one_type_can_be_shared_by_operands_and_results() {
    let registry = registry("$operands attr-dict `:` type($operands, $results)")
        .build()
        .unwrap();
    let source = br#"%input = "test.source"() : () -> f32
%result = test.widen %input : f32"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.widen"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.function_type(operation).unwrap()),
        Some("(f32) -> f32")
    );
}

#[test]
fn parenthesized_result_list_maps_each_ssa_result() {
    let registry = registry("$operands attr-dict `:` type($operands) `to` type($results)")
        .build()
        .unwrap();
    let source = br#"%input = "test.source"() : () -> f32
%first, %second = test.widen %input : f32 to (i32, i64)"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
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
        ["i32", "i64"]
    );
}

#[test]
fn missing_result_type_assignment_fails_during_lowering() {
    let registry = registry("$operands attr-dict `:` type($operands)")
        .build()
        .unwrap();
    let source = br#"%input = "test.source"() : () -> f32
%result = test.widen %input : f32"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.document.is_none());
    assert!(!lowered.diagnostics.is_empty());
}

#[test]
fn explicit_alternatives_parse_and_lower_both_spellings() {
    let registry = RegistryConfig::from_json(
        r#"{
          "builtins": [],
          "operation_shapes": [],
          "operation_alternatives": [{
            "name": "test.widen",
            "alternatives": [
              {"format": "$value `:` type($value) attr-dict `:` type($result)"},
              {"shape": "operand_clauses"}
            ]
          }]
        }"#,
    )
    .unwrap()
    .build()
    .unwrap();
    let source = br#"%literal = test.widen 0.0 : f64 : f32
%dimension = test.widen dim(#test.dimension<3>) : i32"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let mut document = lowered.document.unwrap();
    let operations = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("test.widen"))
        .collect::<Vec<_>>();
    assert_eq!(operations.len(), 2);
    assert!(document.attribute_id(operations[0], "value").is_some());
    assert!(document.attribute_id(operations[1], "value").is_none());
    assert_eq!(
        document
            .result_types(operations[1])
            .unwrap()
            .iter()
            .map(|ty| document.type_spelling(*ty).unwrap())
            .collect::<Vec<_>>(),
        ["i32"]
    );

    let query = Query::parse(r#"filter(op("test.widen")) | reachable | count"#).unwrap();
    let mut outputs = Vec::new();
    query
        .evaluate(&mut document, &registry, |_, output| {
            outputs.push(output);
            Ok(())
        })
        .unwrap();
    assert_eq!(outputs, [QueryOutput::Count(2)]);
}

#[test]
fn alternatives_require_consistent_symbol_behavior() {
    let inconsistent = RegistryConfig::from_json(
        r#"{"builtins":[],"operation_shapes":[],"operation_alternatives":[{
          "name":"test.choice","alternatives":[{"shape":"call_like"},{"shape":"operand_clauses"}]
        }]}"#,
    )
    .unwrap();
    assert!(matches!(
        inconsistent.build(),
        Err(DeclarativeRegistryError::InvalidOperationAlternatives(name)) if name == "test.choice"
    ));
}

#[test]
fn alternatives_are_explicit_and_order_is_part_of_the_definition() {
    let one = RegistryConfig::from_json(
        r#"{"builtins":[],"operation_shapes":[],"operation_alternatives":[{
          "name":"test.widen","alternatives":[{"shape":"literal_attribute"},{"shape":"operand_clauses"}]
        }]}"#,
    ).unwrap();
    let reversed = RegistryConfig::from_json(
        r#"{"builtins":[],"operation_shapes":[],"operation_alternatives":[{
          "name":"test.widen","alternatives":[{"shape":"operand_clauses"},{"shape":"literal_attribute"}]
        }]}"#,
    ).unwrap();
    assert!(matches!(
        RegistryConfig::build_many(&[one, reversed]),
        Err(DeclarativeRegistryError::ConflictingAlternatives(name)) if name == "test.widen"
    ));

    let duplicate = RegistryConfig::from_json(
        r#"{"builtins":[],"operation_shapes":[{"name":"test.widen","shape":"literal_attribute"}],
        "operation_alternatives":[{"name":"test.widen","alternatives":[{"shape":"literal_attribute"},{"shape":"operand_clauses"}]}]}"#,
    ).unwrap();
    assert!(matches!(
        duplicate.build(),
        Err(DeclarativeRegistryError::DuplicateOperation(name)) if name == "test.widen"
    ));
}

#[test]
fn invalid_format_error_names_the_entry_and_rule() {
    let error = match registry("$operands type($missing)").build() {
        Ok(_) => panic!("invalid format was accepted"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("test.widen"), "{message}");
    assert!(message.contains("unknown type target"), "{message}");
}

#[test]
fn formats_with_untyped_ssa_operands_fail_during_registry_construction() {
    for format in [
        "$operands attr-dict",
        "$operands attr-dict `:` type($results)",
        "$operands attr-dict `:` type($result)",
    ] {
        let error = match registry(format).build() {
            Ok(_) => panic!("accepted {format}"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("every SSA operand capture needs a type assignment"),
            "{format}: {error}"
        );
    }
}

#[test]
fn failed_alternatives_recover_once_and_preserve_the_next_operation() {
    let registry = RegistryConfig::from_json(
        r#"{"builtins":[],"operation_shapes":[],"operation_alternatives":[{
          "name":"test.widen","alternatives":[{"shape":"literal_attribute"},{"shape":"binary_operands"}]
        }]}"#,
    )
    .unwrap()
    .build()
    .unwrap();
    let parsed = ParsedFile::parse_with_registry(
        b"test.widen\n\"test.after\"() : () -> ()".as_slice(),
        &registry,
    )
    .unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::FormatMismatch)
            .count(),
        1
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );
}
