use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use zirium::{
    SyntaxKind,
    dialect::{
        AssemblyProgram, AttributeDescriptor, DialectRegistry, OperandCount, OperationDescriptor,
        OperationSchema, RegionDescriptor, RegionKind, ResultCount, SymbolDescriptor,
        TypeDescriptor,
    },
    parser::{ParseDiagnosticKind, ParsedFile},
    printer::{DialectPrintMode, PrintLayout},
    semantic::{
        ArithAddiOp, ArithConstantOp, AttributeValue, BuiltinModuleOp, CfBrOp, CfCondBrOp,
        FuncCallOp, FuncFuncOp, FuncReturnOp, LoweringMode, SemanticVerificationError, TypeSpec,
        TypeValue, lower_proving_fixture, lower_with_dialect_registry,
    },
};

static TYPE_VERIFICATIONS: AtomicUsize = AtomicUsize::new(0);
static ATTRIBUTE_VERIFICATIONS: AtomicUsize = AtomicUsize::new(0);

fn verify_test_type(spelling: &str) -> Result<(), &'static str> {
    if spelling.contains("reject") {
        Err("test type rejected")
    } else {
        Ok(())
    }
}

#[test]
fn declarative_registry_owns_a_selected_builtin_subset() {
    let registry = DialectRegistry::declarative(&["arith.constant", "func.return"]).unwrap();
    assert_eq!(
        registry.operation_names().collect::<Vec<_>>(),
        ["func.return", "arith.constant"]
    );
    assert!(DialectRegistry::declarative(&["unknown.operation"]).is_err());
    assert!(DialectRegistry::declarative(&["cf.br", "cf.br"]).is_err());
}

fn verify_test_attribute(spelling: &str) -> Result<(), &'static str> {
    if spelling.contains("reject") {
        Err("test attribute rejected")
    } else {
        Ok(())
    }
}

fn count_test_type(_: &str) -> Result<(), &'static str> {
    TYPE_VERIFICATIONS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn count_test_attribute(_: &str) -> Result<(), &'static str> {
    ATTRIBUTE_VERIFICATIONS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

static TEST_TYPES: [TypeDescriptor; 1] = [TypeDescriptor {
    name: "!test.value",
    parse: None,
    lower: None,
    verify: Some(verify_test_type),
    print: None,
}];
static TEST_ATTRIBUTES: [AttributeDescriptor; 1] = [AttributeDescriptor {
    name: "#test.value",
    parse: None,
    lower: None,
    verify: Some(verify_test_attribute),
    print: None,
}];
static VALUE_REGISTRY: DialectRegistry = DialectRegistry::new(&[], &TEST_TYPES, &TEST_ATTRIBUTES);
static COUNTED_TYPES: [TypeDescriptor; 1] = [TypeDescriptor {
    name: "!test.value",
    parse: None,
    lower: None,
    verify: Some(count_test_type),
    print: None,
}];
static COUNTED_ATTRIBUTES: [AttributeDescriptor; 1] = [AttributeDescriptor {
    name: "#test.value",
    parse: None,
    lower: None,
    verify: Some(count_test_attribute),
    print: None,
}];
static COUNTING_VALUE_REGISTRY: DialectRegistry =
    DialectRegistry::new(&[], &COUNTED_TYPES, &COUNTED_ATTRIBUTES);

fn parse_registered(source: &str) -> ParsedFile {
    ParsedFile::parse_with_registry(
        Arc::<[u8]>::from(source.as_bytes()),
        DialectRegistry::proving(),
    )
    .unwrap()
}

fn lower_registered(source: &str) -> zirium::semantic::Document {
    let parsed = parse_registered(source);
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving());
    lowered
        .document
        .unwrap_or_else(|| panic!("registered lowering failed: {:?}", lowered.diagnostics))
}

fn lower_generic(source: &str) -> zirium::semantic::Document {
    let parsed = ParsedFile::parse(source.as_bytes()).unwrap();
    let lowered = lower_proving_fixture(
        &parsed,
        LoweringMode::Strict,
        &zirium::semantic::SharedRegistry,
    );
    lowered
        .document
        .unwrap_or_else(|| panic!("generic lowering failed: {:?}", lowered.diagnostics))
}

#[test]
fn registered_value_verifiers_reject_opaque_values() {
    let type_document = lower_generic("%0 = \"use\"() : () -> !test.value<reject>");
    assert!(matches!(
        type_document.verify_semantics(&VALUE_REGISTRY),
        Err(SemanticVerificationError::Type { spelling, message })
            if spelling == "!test.value<reject>" && message == "test type rejected"
    ));

    let attribute_document = lower_generic("\"use\"() {tag = #test.value<reject>} : () -> ()");
    assert!(matches!(
        attribute_document.verify_semantics(&VALUE_REGISTRY),
        Err(SemanticVerificationError::Attribute { spelling, message })
            if spelling == "#test.value<reject>" && message == "test attribute rejected"
    ));
}

#[test]
fn registered_value_verification_reaches_nested_values_once() {
    TYPE_VERIFICATIONS.store(0, Ordering::Relaxed);
    ATTRIBUTE_VERIFICATIONS.store(0, Ordering::Relaxed);
    let mut document =
        lower_generic("%0 = \"use\"() {tags = [#test.value<ok>, #test.value<ok>]} : () -> i32");
    let operation = document.root_operations()[0];
    let opaque = TypeValue::Opaque(Arc::from(b"!test.value<ok>".as_slice()));
    let mut editor = document.edit(&DialectRegistry::EMPTY).unwrap();
    editor
        .replace_result_types(
            operation,
            &[TypeSpec {
                spelling: "tuple<!test.value<ok>, !test.value<ok>>".into(),
                value: TypeValue::Tuple(vec![opaque.clone(), opaque]),
            }],
        )
        .unwrap();
    editor.commit().unwrap();
    document.verify_semantics(&COUNTING_VALUE_REGISTRY).unwrap();
    assert_eq!(TYPE_VERIFICATIONS.load(Ordering::Relaxed), 1);
    assert_eq!(ATTRIBUTE_VERIFICATIONS.load(Ordering::Relaxed), 1);
}

#[test]
fn handwritten_constant_has_dialect_cst_and_typed_semantics() {
    let parsed = parse_registered("%c = arith.constant 7 : i32");
    let operation = parsed.syntax().file().operations().next().unwrap();
    assert_eq!(
        operation.tree().kind(operation.id()),
        Some(SyntaxKind::DialectOperation)
    );
    assert!(
        operation
            .tree()
            .children(operation.id())
            .unwrap()
            .any(|child| operation.tree().kind(child) == Some(SyntaxKind::ArithConstantValue))
    );

    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving())
            .document
            .unwrap();
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    let id = document.root_operations()[0];
    let constant = ArithConstantOp::cast(&document, id).unwrap();
    assert!(matches!(constant.value(), Some(AttributeValue::Integer(value)) if value == "7"));
    assert_eq!(document.operation_name(id), Some("arith.constant"));
}

#[test]
fn floating_constant_lowers_verifies_and_round_trips_in_both_print_modes() {
    let document = lower_registered("%c = arith.constant -1.25e+2 : f64");
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    let constant = ArithConstantOp::cast(&document, document.root_operations()[0]).unwrap();
    assert!(matches!(constant.value(), Some(AttributeValue::Float(value)) if value == "-1.25e+2"));

    for mode in [
        DialectPrintMode::PreferCustom,
        DialectPrintMode::GenericOnly,
    ] {
        let mut text = String::new();
        document
            .print_with_registry(
                &mut text,
                PrintLayout::Compact,
                mode,
                DialectRegistry::proving(),
            )
            .unwrap();
        let reparsed = if mode == DialectPrintMode::PreferCustom {
            lower_registered(&text)
        } else {
            let parsed = ParsedFile::parse(Arc::<[u8]>::from(text.as_bytes())).unwrap();
            lower_proving_fixture(
                &parsed,
                LoweringMode::Strict,
                &zirium::semantic::SharedRegistry,
            )
            .document
            .unwrap()
        };
        assert!(document.structurally_eq(&reparsed), "{text}");
    }
}

#[test]
fn malformed_custom_syntax_recovers_to_the_next_operation() {
    let parsed = parse_registered("%bad = arith.constant : i32\n\"next\"() : () -> ()");
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| { diagnostic.kind() == ParseDiagnosticKind::Syntax })
    );
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    assert!(
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving(),)
            .document
            .is_none()
    );
}

#[test]
fn malformed_declarative_fixture_keeps_cst_errors_and_following_operations() {
    let parsed = ParsedFile::parse_with_registry(
        include_bytes!("../../../tests/corpus/mlir-22.1/declarative-core/malformed.mlir")
            .as_slice(),
        DialectRegistry::proving(),
    )
    .unwrap();
    let syntax = parsed.syntax();
    let diagnostics = syntax.diagnostics();
    let operations = syntax.file().operations().collect::<Vec<_>>();
    assert_eq!(operations.len(), 4);
    for operation in operations {
        let range = operation.tree().text_range(operation.id()).unwrap();
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic.kind() == ParseDiagnosticKind::Syntax
                    && diagnostic.range().start() >= range.start()
                    && diagnostic.range().start() <= range.end()
            }),
            "malformed operation was not diagnosed"
        );
    }
}

#[test]
fn registered_verifier_rejects_wrong_constant_value_kind() {
    let document = lower_registered("%c = arith.constant \"not an integer\" : i32");
    assert!(matches!(
        document.verify_semantics(DialectRegistry::proving()),
        Err(SemanticVerificationError::Operation { .. })
    ));
}

#[test]
fn constant_wrapper_requires_the_registered_value_attribute() {
    let parsed = ParsedFile::parse(b"%c = \"arith.constant\"() : () -> i32".as_slice()).unwrap();
    let document = lower_proving_fixture(
        &parsed,
        LoweringMode::Strict,
        &zirium::semantic::SharedRegistry,
    )
    .document
    .unwrap();
    assert!(ArithConstantOp::cast(&document, document.root_operations()[0]).is_none());
}

#[test]
fn custom_constant_lowering_rejects_value_and_result_kind_mismatches() {
    for source in [
        "%c = arith.constant 1.0 : i32",
        "%c = arith.constant 1 : f32",
    ] {
        let parsed = parse_registered(source);
        assert!(
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving())
                .document
                .is_none(),
            "accepted {source}"
        );
    }
}

#[test]
fn generic_constants_lower_but_strict_verification_rejects_kind_mismatches() {
    for source in [
        "%c = \"arith.constant\"() {value = 1.0} : () -> i32",
        "%c = \"arith.constant\"() {value = 1} : () -> f32",
    ] {
        let parsed = ParsedFile::parse(Arc::<[u8]>::from(source.as_bytes())).unwrap();
        let document = lower_proving_fixture(
            &parsed,
            LoweringMode::Strict,
            &zirium::semantic::SharedRegistry,
        )
        .document
        .unwrap();
        assert!(matches!(
            document.verify_semantics(DialectRegistry::proving()),
            Err(SemanticVerificationError::Operation { .. })
        ));
    }
}

#[test]
fn generic_only_and_prefer_custom_round_trip() {
    let original = lower_registered("%c = arith.constant 7 : i32");

    let mut generic = String::new();
    original
        .print_with_registry(
            &mut generic,
            PrintLayout::Compact,
            DialectPrintMode::GenericOnly,
            DialectRegistry::proving(),
        )
        .unwrap();
    assert!(generic.contains("\"arith.constant\"()"));
    let generic_parsed = ParsedFile::parse(Arc::<[u8]>::from(generic.as_bytes())).unwrap();
    let generic_document = lower_proving_fixture(
        &generic_parsed,
        LoweringMode::Strict,
        &zirium::semantic::SharedRegistry,
    )
    .document
    .unwrap();
    assert!(original.structurally_eq(&generic_document));

    let mut custom = String::new();
    original
        .print_with_registry(
            &mut custom,
            PrintLayout::Compact,
            DialectPrintMode::PreferCustom,
            DialectRegistry::proving(),
        )
        .unwrap();
    assert_eq!(custom, "%v0 = arith.constant 7 : i32");
    let custom_document = lower_registered(&custom);
    assert!(original.structurally_eq(&custom_document));

    let fallback = ParsedFile::parse(b"\"unknown.op\"() : () -> ()".as_slice()).unwrap();
    let fallback = lower_proving_fixture(
        &fallback,
        LoweringMode::Strict,
        &zirium::semantic::SharedRegistry,
    )
    .document
    .unwrap();
    let mut fallback_text = String::new();
    fallback
        .print_with_registry(
            &mut fallback_text,
            PrintLayout::Compact,
            DialectPrintMode::PreferCustom,
            DialectRegistry::proving(),
        )
        .unwrap();
    assert_eq!(fallback_text, "\"unknown.op\"() : () -> ()");
}

#[test]
fn unregistered_metadata_defaults_are_conservative() {
    let registry = DialectRegistry::proving();
    assert_eq!(registry.region("unknown.op", 0).kind, RegionKind::Ssacfg);
    assert!(!registry.region("unknown.op", 0).isolated_from_above);
    assert_eq!(registry.symbols("unknown.op"), Default::default());
}

#[test]
fn registration_rejects_a_program_with_an_inconsistent_schema() {
    let operation = OperationDescriptor {
        name: "test.bad",
        syntax_kind: SyntaxKind::DialectOperation,
        parse: None,
        lower: None,
        verify: None,
        print: None,
        assembly: Some(AssemblyProgram::BinaryOperands),
        schema: OperationSchema {
            operands: OperandCount::Exact(1),
            results: ResultCount::Exact(1),
            required_attributes: &[],
        },
        regions: &[],
        symbols: SymbolDescriptor::default(),
    };
    let operations = Box::leak(Box::new([operation]));
    assert!(std::panic::catch_unwind(|| DialectRegistry::new(operations, &[], &[])).is_err());

    let wrong_identity = OperationDescriptor {
        name: "test.addi",
        syntax_kind: SyntaxKind::DialectOperation,
        parse: None,
        lower: None,
        verify: None,
        print: None,
        assembly: Some(AssemblyProgram::BinaryOperands),
        schema: OperationSchema {
            operands: OperandCount::Exact(2),
            results: ResultCount::Exact(1),
            required_attributes: &[],
        },
        regions: &[],
        symbols: SymbolDescriptor::default(),
    };
    let operations = Box::leak(Box::new([wrong_identity]));
    assert!(std::panic::catch_unwind(|| DialectRegistry::new(operations, &[], &[])).is_err());
}

#[test]
fn registration_rejects_inconsistent_required_attribute_lists() {
    let cases = [
        (
            "arith.constant",
            AssemblyProgram::TypedAttribute,
            OperandCount::Exact(0),
            1,
            &["wrong"] as &'static [&'static str],
        ),
        (
            "arith.constant",
            AssemblyProgram::TypedAttribute,
            OperandCount::Exact(0),
            1,
            &["value", "extra"],
        ),
        (
            "arith.addi",
            AssemblyProgram::BinaryOperands,
            OperandCount::Exact(2),
            1,
            &["overflowFlags"],
        ),
        (
            "func.return",
            AssemblyProgram::OptionalTypedOperands,
            OperandCount::Variadic,
            0,
            &["value"],
        ),
        (
            "cf.br",
            AssemblyProgram::TypedSuccessor,
            OperandCount::Exact(0),
            0,
            &["successor"],
        ),
    ];

    for (name, assembly, operands, results, required_attributes) in cases {
        let operations = Box::leak(Box::new([OperationDescriptor {
            name,
            syntax_kind: SyntaxKind::DialectOperation,
            parse: None,
            lower: None,
            verify: None,
            print: None,
            assembly: Some(assembly),
            schema: OperationSchema {
                operands,
                results: ResultCount::Exact(results),
                required_attributes,
            },
            regions: &[],
            symbols: SymbolDescriptor::default(),
        }]));
        assert!(
            std::panic::catch_unwind(|| DialectRegistry::new(operations, &[], &[])).is_err(),
            "accepted inconsistent required attributes for {name}"
        );
    }
}

#[test]
fn declarative_program_rejects_duplicate_inherent_attributes() {
    for source in [
        "%c = arith.constant 1 {value = 2} : i32",
        "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b overflow<nsw> {overflowFlags = #arith.overflow<nuw>} : i32",
    ] {
        let parsed = parse_registered(source);
        let lowered =
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving());
        assert!(lowered.document.is_none());
        assert!(
            lowered
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.message.contains("duplicate inherent attribute") })
        );
    }
}

#[test]
fn declarative_arithmetic_program_lowers_verifies_and_prints() {
    let document = lower_registered(
        "%a = arith.constant 1 {tag = \"a\"} : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b overflow<nsw> : i32",
    );
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    let mut text = String::new();
    document
        .print_with_registry(
            &mut text,
            PrintLayout::Compact,
            DialectPrintMode::PreferCustom,
            DialectRegistry::proving(),
        )
        .unwrap();
    assert!(text.contains("arith.addi %v0, %v1 overflow<nsw> : i32"));
    assert!(document.structurally_eq(&lower_registered(&text)));
}

#[test]
fn declarative_program_rejects_out_of_schema_material_and_bad_overflow() {
    for source in [
        "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b nope : i32",
        "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b overflow<foo> : i32",
    ] {
        let parsed = parse_registered(source);
        assert!(
            parsed
                .syntax()
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::Syntax)
        );
        let lowered =
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving());
        assert!(lowered.document.is_none());
    }
}

#[test]
fn overflow_flags_accept_trivia_and_lower_the_complete_exact_list() {
    for flags in [
        "none",
        "nsw",
        "nuw",
        "nsw, nuw",
        "nuw , nsw",
        "nsw, // second flag\n nuw",
    ] {
        let source = format!(
            "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b overflow< {flags} > : i32"
        );
        let document = lower_registered(&source);
        document
            .verify_semantics(DialectRegistry::proving())
            .unwrap();
        let addi = document
            .operations()
            .find_map(|id| ArithAddiOp::cast(&document, id))
            .unwrap();
        assert_eq!(addi.operands().unwrap().len(), 2);
        assert!(matches!(
            addi.result_type(),
            Some(zirium::semantic::TypeValue::Integer {
                width: 32,
                signedness: None
            })
        ));
    }
}

#[test]
fn overflow_flags_reject_every_non_schema_form() {
    for flags in [
        "",
        "nsw,nsw",
        "nuw,nuw",
        "none,nsw",
        "nsw,none",
        "foo",
        "nsw,foo",
        "nsw,nuw,nsw",
        "nsw,",
        ",nsw",
    ] {
        let source = format!(
            "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b overflow<{flags}> : i32"
        );
        let parsed = parse_registered(&source);
        assert!(
            parsed
                .syntax()
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::Syntax),
            "accepted overflow<{flags}>"
        );
    }
    for suffix in [
        "overflow<nsw",
        "overflow<nsw>>",
        "overflow<nsw> nuw",
        "nuw overflow<nsw>",
    ] {
        let source = format!(
            "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b {suffix} : i32"
        );
        let parsed = parse_registered(&source);
        assert!(
            parsed
                .syntax()
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::Syntax),
            "accepted misplaced or malformed `{suffix}`"
        );
    }
}

#[test]
fn declarative_program_rejects_return_and_successor_mismatches() {
    let bad_return = r#"%function = "func.func"() ({
^entry(%arg: i32):
  func.return %arg : (i32, i32)
}) : () -> i32"#;
    let parsed = parse_registered(bad_return);
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::Syntax)
    );
    assert!(
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving())
            .document
            .is_none()
    );

    for branch in [
        "cf.br ^exit(%arg : i32, %arg : i32)",
        "cf.br ^exit(%other : i64)",
    ] {
        let source = format!(
            "%function = \"func.func\"() ({{\n^entry(%arg: i32):\n  {branch}\n^exit(%result: i32):\n  func.return %result : i32\n}}) : () -> i32"
        );
        let parsed = parse_registered(&source);
        let lowered =
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving());
        assert!(lowered.document.is_none());
    }
}

#[test]
fn declarative_strict_and_best_effort_paths_are_distinct() {
    let parsed = ParsedFile::parse_with_registry(
        include_bytes!("../../../tests/corpus/mlir-22.1/declarative-core/malformed.mlir")
            .as_slice(),
        DialectRegistry::proving(),
    )
    .unwrap();
    let strict =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving());
    assert!(strict.document.is_none());
    let best = lower_with_dialect_registry(
        &parsed,
        LoweringMode::BestEffort,
        DialectRegistry::proving(),
    );
    assert!(best.document.is_some());
    assert!(!best.semantically_complete);
}

#[test]
fn generic_fallback_remains_available_for_each_declarative_operation() {
    let source = r#"%a = "arith.constant"() {value = 1} : () -> i32
%b = "arith.constant"() {value = 2} : () -> i32
%sum = "arith.addi"(%a, %b) : (i32, i32) -> i32
"func.return"() : () -> ()
"cf.br"() : () -> ()"#;
    let parsed = ParsedFile::parse(Arc::<[u8]>::from(source.as_bytes())).unwrap();
    let document = lower_proving_fixture(
        &parsed,
        LoweringMode::Strict,
        &zirium::semantic::SharedRegistry,
    )
    .document
    .unwrap();
    let mut text = String::new();
    document
        .print_with_registry(
            &mut text,
            PrintLayout::Compact,
            DialectPrintMode::GenericOnly,
            DialectRegistry::proving(),
        )
        .unwrap();
    let reparsed = ParsedFile::parse(Arc::<[u8]>::from(text.as_bytes())).unwrap();
    let redocument = lower_proving_fixture(
        &reparsed,
        LoweringMode::Strict,
        &zirium::semantic::SharedRegistry,
    )
    .document
    .unwrap();
    assert!(document.structurally_eq(&redocument));
}

#[test]
fn declarative_return_and_branch_check_enclosing_types() {
    let source = r#"%function = "func.func"() ({
^entry(%arg: i32):
  cf.br ^exit(%arg : i32)
^exit(%result: i32):
  func.return %result : i32
}) : () -> i32"#;
    let document = lower_registered(source);
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    let mut text = String::new();
    document
        .print_with_registry(
            &mut text,
            PrintLayout::Compact,
            DialectPrintMode::PreferCustom,
            DialectRegistry::proving(),
        )
        .unwrap();
    assert!(document.structurally_eq(&lower_registered(&text)));

    let mut generic = String::new();
    document
        .print_with_registry(
            &mut generic,
            PrintLayout::Compact,
            DialectPrintMode::GenericOnly,
            DialectRegistry::proving(),
        )
        .unwrap();
    assert!(
        generic.contains("\"func.return\"(%v2) : i32 -> ()"),
        "{generic}"
    );
    let generic_document = lower_registered(&generic);
    generic_document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    assert!(document.structurally_eq(&generic_document));
}

#[test]
fn registered_wrappers_expose_add_return_and_branch_structure() {
    let source = r#"%a = arith.constant 1 : i32
%b = arith.constant 2 : i32
%function = "func.func"() ({
^entry:
  %sum = arith.addi %a, %b : i32
  cf.br ^exit(%sum : i32)
^exit(%result: i32):
  func.return %result : i32
}) : () -> i32"#;
    let document = lower_registered(source);
    let addi = document
        .operations()
        .find_map(|id| ArithAddiOp::cast(&document, id))
        .unwrap();
    assert_eq!(addi.operands().unwrap().len(), 2);
    let branch = document
        .operations()
        .find_map(|id| CfBrOp::cast(&document, id))
        .unwrap();
    assert_eq!(
        document
            .successor_arguments(branch.successor().unwrap())
            .unwrap()
            .len(),
        1
    );
    let returned = document
        .operations()
        .find_map(|id| FuncReturnOp::cast(&document, id))
        .unwrap();
    assert_eq!(returned.operands().unwrap().len(), 1);
}

#[test]
fn declarative_core_fixture_round_trips_custom_attribute_dictionaries() {
    let source = include_str!("../../../tests/corpus/mlir-22.1/declarative-core/valid.mlir");
    let document = lower_registered(source);
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    let mut text = String::new();
    document
        .print_with_registry(
            &mut text,
            PrintLayout::Compact,
            DialectPrintMode::PreferCustom,
            DialectRegistry::proving(),
        )
        .unwrap();
    assert!(text.contains("cf.br ^bb1 {tag = \"edge\"}"));
    assert!(text.contains("func.return {tag = \"return\"}"));
    assert!(document.structurally_eq(&lower_registered(&text)));
}

#[test]
fn complete_proving_dialect_fixture_verifies_and_round_trips_both_modes() {
    let source = include_str!("../../../tests/corpus/mlir-22.1/proving-dialects/valid.mlir");
    let document = lower_registered(source);
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    assert_eq!(
        DialectRegistry::proving()
            .operation_names()
            .collect::<Vec<_>>(),
        [
            "builtin.module",
            "func.func",
            "func.return",
            "func.call",
            "arith.constant",
            "arith.addi",
            "cf.br",
            "cf.cond_br",
        ]
    );
    assert!(
        document
            .operations()
            .any(|id| BuiltinModuleOp::cast(&document, id).is_some())
    );
    assert!(
        document
            .operations()
            .any(|id| FuncFuncOp::cast(&document, id).is_some())
    );
    assert!(
        document
            .operations()
            .any(|id| FuncCallOp::cast(&document, id).is_some())
    );
    assert!(
        document
            .operations()
            .any(|id| CfCondBrOp::cast(&document, id).is_some())
    );

    for mode in [
        DialectPrintMode::PreferCustom,
        DialectPrintMode::GenericOnly,
    ] {
        let mut text = String::new();
        document
            .print_with_registry(
                &mut text,
                PrintLayout::Compact,
                mode,
                DialectRegistry::proving(),
            )
            .unwrap();
        let parsed = if mode == DialectPrintMode::PreferCustom {
            parse_registered(&text)
        } else {
            ParsedFile::parse(Arc::<[u8]>::from(text.as_bytes())).unwrap()
        };
        let lowered = if mode == DialectPrintMode::PreferCustom {
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving())
        } else {
            lower_proving_fixture(
                &parsed,
                LoweringMode::Strict,
                &zirium::semantic::SharedRegistry,
            )
        };
        let round_trip = lowered
            .document
            .unwrap_or_else(|| panic!("{text}\n{:?}", lowered.diagnostics));
        assert!(document.structurally_eq(&round_trip), "{text}");
    }
}

#[test]
fn complete_proving_dialect_malformed_fixture_recovers() {
    let parsed = parse_registered(include_str!(
        "../../../tests/corpus/mlir-22.1/proving-dialects/malformed.mlir"
    ));
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::Syntax)
    );
    assert_eq!(parsed.syntax().file().operations().count(), 4);
}

#[test]
fn functions_check_signature_attributes_and_scoped_call_targets() {
    let valid = r#"builtin.module @outer {
  func.func @id(%arg: i32) -> (i32) attributes {arg_attrs = [{arg = 1}], res_attrs = [{result = 2}]} {
  ^entry(%value: i32):
    func.return %value : i32
  }
  func.func @caller() -> i32 {
    %input = arith.constant 1 : i32
    %output = func.call @id(%input) : (i32) -> i32
    func.return %output : i32
  }
}"#;
    let document = lower_registered(valid);
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();

    for source in [
        valid.replace(
            "@id(%input) : (i32) -> i32",
            "@missing(%input) : (i32) -> i32",
        ),
        valid.replace("@id(%input) : (i32) -> i32", "@id(%input) : (i64) -> i32"),
        valid.replace(
            "arg_attrs = [{arg = 1}]",
            "arg_attrs = [{arg = 1}, {extra = 3}]",
        ),
    ] {
        let parsed = parse_registered(&source);
        let lowered =
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving());
        let document = lowered
            .document
            .unwrap_or_else(|| panic!("{:?}", lowered.diagnostics));
        assert!(matches!(
            document.verify_semantics(DialectRegistry::proving()),
            Err(SemanticVerificationError::Operation { .. })
        ));
    }
}

#[test]
fn zero_result_functions_and_unit_no_inline_use_the_registered_forms() {
    let document =
        lower_registered("builtin.module { func.func @decl() func.func @body() { func.return } }");
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();

    let unit = lower_registered("builtin.module { func.func @f() attributes {no_inline} }");
    unit.verify_semantics(DialectRegistry::proving()).unwrap();
    let mut printed = String::new();
    unit.print_with_registry(
        &mut printed,
        PrintLayout::Compact,
        DialectPrintMode::PreferCustom,
        DialectRegistry::proving(),
    )
    .unwrap();
    assert!(printed.contains("no_inline"));

    let integer = "builtin.module { func.func @f() attributes {no_inline = 1} }";
    let document = lower_registered(integer);
    assert!(matches!(
        document.verify_semantics(DialectRegistry::proving()),
        Err(SemanticVerificationError::Operation { .. })
    ));
}

#[test]
fn conditional_branch_weights_require_two_nonnegative_i32_values() {
    let source = include_str!("../../../tests/corpus/mlir-22.1/proving-dialects/valid.mlir");
    for weights in [
        "[1, 2]",
        "dense<[1, -2]> : vector<2xi32>",
        "dense<[1, 2147483648]> : vector<2xi32>",
        "dense<[1, 2]> : vector<3xi32>",
        "\"one,two\"",
    ] {
        let source = source.replace("dense<[1, 2]> : vector<2xi32>", weights);
        let document = lower_registered(&source);
        assert!(
            matches!(
                document.verify_semantics(DialectRegistry::proving()),
                Err(SemanticVerificationError::Operation { message, .. })
                    if message.contains("branch_weights")
            ),
            "accepted {weights}"
        );
    }
}

#[test]
fn dominance_is_checked_across_cfg_blocks() {
    let source = r#"builtin.module {
  func.func @bad() -> i32 {
    %condition = arith.constant 1 : i1
    cf.cond_br %condition, ^left, ^right
  ^left:
    %only_left = arith.constant 1 : i32
    cf.br ^join
  ^right:
    cf.br ^join
  ^join:
    func.return %only_left : i32
  }
}"#;
    let document = lower_registered(source);
    assert!(matches!(
        document.verify_semantics(DialectRegistry::proving()),
        Err(SemanticVerificationError::Operation { message, .. })
            if message == "SSA definition does not dominate its use"
    ));
    let definition = document
        .operations()
        .find(|operation| {
            document.operation_name(*operation) == Some("arith.constant")
                && document.result_types(*operation).is_some_and(|types| {
                    types.len() == 1 && document.type_spelling(types[0]) == Some("i32")
                })
        })
        .unwrap();
    let value = document
        .operation(definition)
        .unwrap()
        .result(definition, 0)
        .unwrap();
    let use_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.return"))
        .unwrap();
    assert!(!document.dominates(value, use_operation, DialectRegistry::proving()));
}

#[test]
fn hierarchical_dominance_accepts_outer_cfg_definitions_and_rejects_nested_escape() {
    let valid = r#"builtin.module {
  func.func @good() -> i32 {
    %outer = arith.constant 7 : i32
    %condition = arith.constant 1 : i1
    cf.cond_br %condition, ^left, ^right
  ^left:
    cf.br ^join
  ^right:
    cf.br ^join
  ^join:
    func.return %outer : i32
  }
}"#;
    let document = lower_registered(valid);
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    let definition = document
        .operations()
        .find(|operation| {
            document.operation_name(*operation) == Some("arith.constant")
                && document.result_types(*operation).is_some_and(|types| {
                    types.len() == 1 && document.type_spelling(types[0]) == Some("i32")
                })
        })
        .unwrap();
    let value = document
        .operation(definition)
        .unwrap()
        .result(definition, 0)
        .unwrap();
    let use_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.return"))
        .unwrap();
    assert!(document.dominates(value, use_operation, DialectRegistry::proving()));

    let escaped = r#"builtin.module {
  builtin.module @nested {
    %inner = arith.constant 1 : i32
  }
  func.func @outer() -> i32 {
    func.return %inner : i32
  }
}"#;
    let parsed = parse_registered(escaped);
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving());
    assert!(lowered.document.is_none());
}

#[test]
fn module_and_function_registration_checks_all_metadata() {
    let operation = OperationDescriptor {
        name: "builtin.module",
        syntax_kind: SyntaxKind::DialectOperation,
        parse: None,
        lower: None,
        verify: None,
        print: None,
        assembly: Some(AssemblyProgram::Module),
        schema: OperationSchema {
            operands: OperandCount::Exact(0),
            results: ResultCount::Exact(0),
            required_attributes: &[],
        },
        regions: &[zirium::dialect::RegionDescriptor {
            kind: RegionKind::Graph,
            isolated_from_above: false,
        }],
        symbols: SymbolDescriptor {
            defines_symbol: true,
            symbol_table: false,
            uses_symbols: true,
        },
    };
    let operations = Box::leak(Box::new([operation]));
    assert!(std::panic::catch_unwind(|| DialectRegistry::new(operations, &[], &[])).is_err());
}

#[test]
fn registration_rejects_wrong_fixed_descriptor_metadata() {
    static VALID_REGION: &[RegionDescriptor] = &[RegionDescriptor {
        kind: RegionKind::Ssacfg,
        isolated_from_above: true,
    }];
    static WRONG_REGION: &[RegionDescriptor] = &[RegionDescriptor {
        kind: RegionKind::Graph,
        isolated_from_above: false,
    }];

    let cases = [
        (
            "builtin.module",
            AssemblyProgram::Module,
            OperationSchema {
                operands: OperandCount::Exact(0),
                results: ResultCount::Exact(0),
                required_attributes: &[],
            },
            VALID_REGION,
            SymbolDescriptor {
                defines_symbol: false,
                symbol_table: true,
                uses_symbols: false,
            },
        ),
        (
            "func.func",
            AssemblyProgram::Function,
            OperationSchema {
                operands: OperandCount::Exact(0),
                results: ResultCount::Exact(0),
                required_attributes: &["sym_name", "function_type"],
            },
            VALID_REGION,
            SymbolDescriptor {
                defines_symbol: true,
                symbol_table: true,
                uses_symbols: false,
            },
        ),
        (
            "func.call",
            AssemblyProgram::Call,
            OperationSchema {
                operands: OperandCount::Variadic,
                results: ResultCount::Variadic,
                required_attributes: &["callee"],
            },
            WRONG_REGION,
            SymbolDescriptor {
                defines_symbol: false,
                symbol_table: false,
                uses_symbols: true,
            },
        ),
        (
            "cf.cond_br",
            AssemblyProgram::ConditionalBranch,
            OperationSchema {
                operands: OperandCount::Variadic,
                results: ResultCount::Exact(0),
                required_attributes: &[],
            },
            &[],
            SymbolDescriptor {
                defines_symbol: true,
                symbol_table: false,
                uses_symbols: false,
            },
        ),
        (
            "arith.constant",
            AssemblyProgram::TypedAttribute,
            OperationSchema {
                operands: OperandCount::Exact(0),
                results: ResultCount::Exact(1),
                required_attributes: &["value"],
            },
            WRONG_REGION,
            SymbolDescriptor::default(),
        ),
        (
            "arith.addi",
            AssemblyProgram::BinaryOperands,
            OperationSchema {
                operands: OperandCount::Exact(2),
                results: ResultCount::Exact(1),
                required_attributes: &[],
            },
            &[],
            SymbolDescriptor {
                defines_symbol: false,
                symbol_table: false,
                uses_symbols: true,
            },
        ),
        (
            "func.return",
            AssemblyProgram::OptionalTypedOperands,
            OperationSchema {
                operands: OperandCount::Variadic,
                results: ResultCount::Exact(0),
                required_attributes: &[],
            },
            &[],
            SymbolDescriptor {
                defines_symbol: false,
                symbol_table: false,
                uses_symbols: true,
            },
        ),
        (
            "cf.br",
            AssemblyProgram::TypedSuccessor,
            OperationSchema {
                operands: OperandCount::Exact(0),
                results: ResultCount::Exact(0),
                required_attributes: &[],
            },
            &[],
            SymbolDescriptor {
                defines_symbol: true,
                symbol_table: false,
                uses_symbols: false,
            },
        ),
    ];

    for (name, assembly, schema, regions, symbols) in cases {
        let operations = Box::leak(Box::new([OperationDescriptor {
            name,
            syntax_kind: SyntaxKind::DialectOperation,
            parse: None,
            lower: None,
            verify: None,
            print: None,
            assembly: Some(assembly),
            schema,
            regions,
            symbols,
        }]));
        assert!(
            std::panic::catch_unwind(|| DialectRegistry::new(operations, &[], &[])).is_err(),
            "inconsistent metadata for {name} was accepted"
        );
    }
}
