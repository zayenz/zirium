use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use zirium::{
    SyntaxKind,
    dialect::{
        AssemblyProgram, AttributeDescriptor, DialectRegistry, OperandCount, OperationDescriptor,
        OperationSchema, OperationShape, RegionDescriptor, RegionKind, ResultCount,
        SymbolDescriptor, TypeDescriptor,
    },
    parser::{ParseDiagnosticKind, ParsedFile},
    printer::{DialectPrintMode, PrintLayout},
    semantic::{
        ArithAddiOp, ArithConstantOp, AttributeValue, BuiltinModuleOp, CfBrOp, CfCondBrOp,
        FuncCallOp, FuncFuncOp, FuncReturnOp, LoweringMode, SemanticVerificationError,
        lower_with_dialect_registry,
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

#[test]
fn operation_shapes_extend_existing_registries() {
    let registry = DialectRegistry::declarative(&["arith.constant"])
        .unwrap()
        .extend_operation_shapes(&[("vendor.function", OperationShape::FuncLike)])
        .unwrap();
    assert!(registry.operation("arith.constant").is_some());
    assert_eq!(
        registry.operation_shape("vendor.function"),
        Some(OperationShape::FuncLike)
    );
    assert!(
        registry
            .extend_operation_shapes(&[("arith.constant", OperationShape::CallLike)])
            .is_err()
    );
}

#[test]
fn named_stablehlo_registry_parses_and_lowers_its_supported_custom_forms() {
    let registry = DialectRegistry::from_name("stablehlo").unwrap();
    for name in [
        "stablehlo.add",
        "stablehlo.and",
        "stablehlo.atan2",
        "stablehlo.divide",
        "stablehlo.maximum",
        "stablehlo.minimum",
        "stablehlo.multiply",
        "stablehlo.or",
        "stablehlo.power",
        "stablehlo.remainder",
        "stablehlo.shift_left",
        "stablehlo.shift_right_arithmetic",
        "stablehlo.shift_right_logical",
        "stablehlo.subtract",
        "stablehlo.xor",
    ] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::BinaryOperands)
        );
    }
    let source = br#"module {
      func.func @add(%lhs: tensor<2xf32>, %rhs: tensor<2xf32>) -> tensor<2xf32> {
        %sum = stablehlo.add %lhs, %rhs : tensor<2xf32> loc(unknown)
        %unused = stablehlo.add %lhs, %rhs : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        func.return %sum : tensor<2xf32>
      }
      "test.container"() ({
      ^bb0(%arg: tensor<f32>):
        stablehlo.return %arg : tensor<f32>
      }) : () -> ()
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty());
    let document = lowered.document.unwrap();
    let add = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("stablehlo.add"))
        .unwrap();
    assert_eq!(document.operands(add).unwrap().len(), 2);
    assert_eq!(document.result_types(add).unwrap().len(), 1);
    assert_eq!(
        document
            .operations()
            .filter(|operation| document.operation_name(*operation) == Some("stablehlo.add"))
            .count(),
        2
    );
    let stablehlo_return = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("stablehlo.return"))
        .unwrap();
    assert_eq!(document.operands(stablehlo_return).unwrap().len(), 1);
    assert!(document.result_types(stablehlo_return).unwrap().is_empty());
}

#[test]
fn stablehlo_preset_structures_common_attribute_heavy_forms() {
    let registry = DialectRegistry::from_name("stablehlo").unwrap();
    let source = br#"module {
      func.func @main(%input: tensor<4x4xf32>, %start: tensor<i32>) -> tensor<2x4xf32> {
        %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %slice = stablehlo.dynamic_slice %input, %start, %start, sizes = [2, 4] : (tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<2x4xf32>
        %broadcast = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
        %result = stablehlo.add %slice, %broadcast : tensor<2x4xf32>
        func.return %result : tensor<2x4xf32>
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();

    let dynamic_slice = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("stablehlo.dynamic_slice"))
        .unwrap();
    assert_eq!(document.operands(dynamic_slice).unwrap().len(), 3);
    assert_eq!(document.result_types(dynamic_slice).unwrap().len(), 1);
    assert!(document.attribute_id(dynamic_slice, "sizes").is_some());

    let broadcast = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("stablehlo.broadcast_in_dim"))
        .unwrap();
    assert_eq!(document.operands(broadcast).unwrap().len(), 1);
    assert!(document.attribute_id(broadcast, "dims").is_some());

    let constant = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("stablehlo.constant"))
        .unwrap();
    let value = document.attribute_id(constant, "value").unwrap();
    assert_eq!(
        document.attribute_spelling_value(value),
        Some("dense<0.000000e+00> : tensor<f32>")
    );
}

#[test]
fn stablehlo_preset_structures_reducer_regions_and_arguments() {
    let registry = DialectRegistry::from_name("stablehlo").unwrap();
    let source = br#"%input = "test.source"() : () -> tensor<f32>
    %init = "test.source"() : () -> tensor<f32>
    stablehlo.reduce(%input init: %init) across dimensions = [0] reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      stablehlo.return %lhs : tensor<f32>
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let reduce = parsed.syntax().file().operations().nth(2).unwrap();
    assert_eq!(reduce.operands().count(), 2);
    assert_eq!(reduce.arguments().count(), 2);
    let block = reduce.regions().next().unwrap().blocks().next().unwrap();
    assert_eq!(block.arguments().count(), 0);

    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let reduce = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("stablehlo.reduce"))
        .unwrap();
    let region = document.operation_regions(reduce).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
    assert_eq!(document.block_argument_types(block).unwrap().len(), 2);
}

#[test]
fn tosa_preset_structures_operands_attributes_and_constants() {
    let registry = DialectRegistry::from_name("tosa").unwrap();
    let source = br#"module {
      func.func @main(%lhs: tensor<2xf32>, %rhs: tensor<2xf32>) -> tensor<2xf32> {
        %zero = tosa.const dense<0.000000e+00> : tensor<2xf32>
        %sum = tosa.add %lhs, %rhs {shift = 0 : i32} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        func.return %sum : tensor<2xf32>
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let add = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("tosa.add"))
        .unwrap();
    assert_eq!(document.operands(add).unwrap().len(), 2);
    assert!(document.attribute_id(add, "shift").is_some());
    assert_eq!(document.result_types(add).unwrap().len(), 1);
}

#[test]
fn scf_preset_exposes_regions_and_header_block_arguments() {
    let registry = DialectRegistry::from_name("scf").unwrap();
    let source = br#"module {
      func.func @loop(%lower: index, %upper: index, %step: index) {
        scf.for %iv = %lower to %upper step %step {
          "test.use"(%iv) : (index) -> ()
          scf.yield
        }
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let loop_op = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("scf.for"))
        .unwrap();
    assert_eq!(document.operands(loop_op).unwrap().len(), 3);
    let region = document.operation_regions(loop_op).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
    assert_eq!(document.block_argument_types(block).unwrap().len(), 1);
}

#[test]
fn linalg_preset_keeps_generic_regions_and_explicit_block_arguments() {
    let registry = DialectRegistry::from_name("linalg").unwrap();
    let source = br#"module {
      func.func @kernel(%input: memref<4xf32>, %output: memref<4xf32>) {
        linalg.generic indexing_maps = [], iterator_types = [] ins(%input : memref<4xf32>) outs(%output : memref<4xf32>) {
        ^bb0(%element: f32, %accumulator: f32):
          linalg.yield %element : f32
        }
        func.return
      }
      func.func @tensor_copy(%input: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
        %result = linalg.copy ins(%input : tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
        func.return %result : tensor<4xf32>
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let generic = parsed
        .syntax()
        .file()
        .operations()
        .find(|operation| {
            operation.mnemonic_range().is_some_and(|range| {
                &source[range.start() as usize..range.end() as usize] == b"linalg.generic"
            })
        })
        .unwrap();
    assert_eq!(generic.operands().count(), 2);
    let region = generic.regions().next().unwrap();
    assert_eq!(region.blocks().next().unwrap().arguments().count(), 2);

    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let generic = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("linalg.generic"))
        .unwrap();
    assert!(document.attribute_id(generic, "indexing_maps").is_some());
    assert!(document.attribute_id(generic, "iterator_types").is_some());
    let region = document.operation_regions(generic).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
    assert_eq!(document.block_argument_types(block).unwrap().len(), 2);
    let copy = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("linalg.copy"))
        .unwrap();
    assert_eq!(document.operands(copy).unwrap().len(), 2);
    assert_eq!(document.result_types(copy).unwrap().len(), 1);
}

#[test]
fn acc_preset_structures_mapping_accessors_and_regions() {
    let registry = DialectRegistry::from_name("acc").unwrap();
    let source = br#"module {
      acc.private.recipe @private_memref : memref<4xf32> init {
      ^bb0(%original: memref<4xf32>):
        acc.yield %original : memref<4xf32>
      }
      func.func @kernel(%host: memref<4xf32>, %queue: i32, %bounds: !acc.data_bounds_ty) {
        %device = acc.copyin varPtr(%host : memref<4xf32>) async(%queue : i32) -> memref<4xf32>
        %extent = acc.get_extent %bounds : (!acc.data_bounds_ty) -> index
        acc.parallel async(%queue : i32) private(%device : memref<4xf32>) {
          "test.use"(%extent) : (index) -> ()
          acc.yield
        }
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    // The OpenACC data-bounds type is intentionally opaque to a declarative
    // operation preset, so retain the structurally lowered document here.
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();

    let copyin = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("acc.copyin"))
        .unwrap();
    assert_eq!(document.operands(copyin).unwrap().len(), 2);
    assert_eq!(document.result_types(copyin).unwrap().len(), 1);

    let extent = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("acc.get_extent"))
        .unwrap();
    assert_eq!(document.operands(extent).unwrap().len(), 1);
    assert_eq!(document.result_types(extent).unwrap().len(), 1);

    for name in ["acc.private.recipe", "acc.parallel"] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.operation_regions(operation).unwrap().len(), 1);
    }
}

#[test]
fn affine_preset_lowers_supported_forms_without_inventing_results() {
    let registry = DialectRegistry::from_name("affine").unwrap();
    let source = br#"module {
      func.func @kernel(%n: index, %x: index, %y: index, %basis: index, %seed: index) -> index {
        %linear = affine.linearize_index disjoint [%x, %y] by (4, %basis) : index
        %result = affine.for %iv = 0 to %n iter_args(%carried = %seed) -> index {
          %selected = affine.if affine_set<(d0) : (d0 >= 0)> (%iv) -> index {
            affine.yield %carried : index
          } else {
            affine.yield %linear : index
          }
          affine.yield %selected : index
        }
        func.return %result : index
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();

    let linearize = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("affine.linearize_index"))
        .unwrap();
    assert_eq!(document.operands(linearize).unwrap().len(), 3);
    assert_eq!(document.result_types(linearize).unwrap().len(), 1);
    assert_eq!(
        document.type_spelling(document.result_types(linearize).unwrap()[0]),
        Some("index")
    );

    let for_op = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("affine.for"))
        .unwrap();
    assert_eq!(document.operands(for_op).unwrap().len(), 2);
    assert_eq!(document.result_types(for_op).unwrap().len(), 1);
    let for_region = document.operation_regions(for_op).unwrap()[0];
    let for_block = document
        .region(for_region)
        .unwrap()
        .blocks(&document)
        .unwrap()[0];
    assert_eq!(document.block_argument_types(for_block).unwrap().len(), 2);

    let if_op = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("affine.if"))
        .unwrap();
    assert_eq!(document.operands(if_op).unwrap().len(), 1);
    assert_eq!(document.result_types(if_op).unwrap().len(), 1);
    assert_eq!(document.operation_regions(if_op).unwrap().len(), 2);

    let yields = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("affine.yield"))
        .collect::<Vec<_>>();
    assert_eq!(yields.len(), 3);
    assert!(
        yields
            .iter()
            .all(|operation| document.result_types(*operation).unwrap().is_empty())
    );

    // Typed memory and DMA forms have no SSA results. Keeping them off the
    // broad clause shape prevents their trailing memref types becoming results.
    for name in [
        "affine.store",
        "affine.vector_store",
        "affine.prefetch",
        "affine.dma_start",
        "affine.dma_wait",
    ] {
        assert_eq!(registry.operation_shape(name), None);
    }
}

#[test]
fn binary_operand_shape_recovers_from_arity_mismatches() {
    let registry =
        DialectRegistry::with_operation_shapes(&[("a.Op", OperationShape::BinaryOperands)])
            .unwrap();
    for (operation, expect_diagnostics) in [
        ("%r = a.Op : bf16", true),
        ("%r = a.Op %x0 : bf16", true),
        ("%r = a.Op %x0, %x1 : bf16", false),
        ("%r = a.Op %x0, %x1, %x2 : bf16", true),
        ("a.Op %x0 : bf16", true),
        ("a.Op %x0, %x1 : bf16", true),
    ] {
        let source = format!(
            r#""builtin.module"() ({{
^bb0:
  %x0 = "test.source"() : () -> bf16
  %x1 = "test.source"() : () -> bf16
  %x2 = "test.source"() : () -> bf16
  {operation}
  "test.after"() : () -> ()
}}) : () -> ()"#
        );
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
        assert_eq!(
            parsed.syntax().diagnostics().iter().any(|diagnostic| {
                diagnostic.kind()
                    == ParseDiagnosticKind::ShapeMismatch(OperationShape::BinaryOperands)
            }) || !lowered.diagnostics.is_empty(),
            expect_diagnostics,
            "unexpected diagnostics for {operation:?}: {:?}",
            parsed.syntax().diagnostics()
        );
        let document = lowered.document.unwrap();
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some("test.after")),
            "following operation was lost after {operation:?}"
        );
        if expect_diagnostics {
            assert!(!document.is_semantically_complete());
        } else {
            document.verify_semantics(&registry).unwrap();
        }
    }
}

#[test]
fn binary_operand_shape_accepts_result_and_function_type_trailers() {
    let registry =
        DialectRegistry::with_operation_shapes(&[("a.Op", OperationShape::BinaryOperands)])
            .unwrap();
    for (trailer, expected_result) in [
        ("i32", "i32"),
        ("i32 -> i1", "i1"),
        ("i32, i32 -> i1", "i1"),
        ("(i32, i32) -> i1", "i1"),
        ("i32 to i1", "i1"),
        ("i32 loc(unknown)", "i32"),
        ("i32 -> i1 loc(unknown)", "i1"),
        ("i32 to i1 loc(unknown)", "i1"),
    ] {
        let source = format!(
            r#""builtin.module"() ({{
^bb0:
  %x0 = "test.source"() : () -> i32
  %x1 = "test.source"() : () -> i32
  %r = a.Op %x0, %x1 : {trailer}
}}) : () -> ()"#
        );
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        assert!(
            parsed.syntax().diagnostics().is_empty(),
            "unexpected syntax diagnostics for {trailer:?}: {:?}",
            parsed.syntax().diagnostics()
        );
        let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
        assert!(
            lowered.diagnostics.is_empty(),
            "unexpected lowering diagnostics for {trailer:?}: {:?}",
            lowered.diagnostics
        );
        let document = lowered.document.unwrap();
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some("a.Op"))
            .unwrap();
        assert_eq!(document.operands(operation).unwrap().len(), 2);
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(document.type_spelling(results[0]), Some(expected_result));
    }
}

#[test]
fn operation_shapes_accept_conversion_type_trailers_without_fabricated_siblings() {
    for (name, shape, spelling, operand_count, result) in [
        (
            "stir.bitwise_and",
            OperationShape::BinaryOperands,
            "%r = stir.bitwise_and %a, %b : i16 to i32",
            2,
            "i32",
        ),
        (
            "arith.index_cast",
            OperationShape::UnaryOperand,
            "%r = arith.index_cast %a : i16 to i64",
            1,
            "i64",
        ),
        (
            "stir.bitwise_or",
            OperationShape::VariadicOperands,
            "%r = stir.bitwise_or %a, %b : i16 to i16",
            2,
            "i16",
        ),
        (
            "memref.cast",
            OperationShape::UnaryOperand,
            "%r = memref.cast %m : memref<4xf32> to memref<?xf32>",
            1,
            "memref<?xf32>",
        ),
    ] {
        let registry = DialectRegistry::core()
            .extend_operation_shapes(&[(name, shape)])
            .unwrap();
        let source = format!(
            r#""builtin.module"() ({{
^bb0:
  %a = "test.source"() : () -> i16
  %b = "test.source"() : () -> i16
  %m = "test.source"() : () -> memref<4xf32>
  {spelling}
}}) : () -> ()"#
        );
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        assert!(
            parsed.syntax().diagnostics().is_empty(),
            "unexpected syntax diagnostics for {name}: {:?}",
            parsed.syntax().diagnostics()
        );
        assert!(parsed.syntax().file().operations().all(|operation| {
            operation
                .mnemonic_range()
                .and_then(|range| source.get(range.start() as usize..range.end() as usize))
                != Some("to")
        }));

        let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
        assert!(
            lowered.diagnostics.is_empty(),
            "unexpected lowering diagnostics for {name}: {:?}",
            lowered.diagnostics
        );
        let document = lowered.document.unwrap();
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.operands(operation).unwrap().len(), operand_count);
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(document.type_spelling(results[0]), Some(result));
    }
}

#[test]
fn binary_operand_shape_rejects_parenthesized_single_input_function_type() {
    let registry =
        DialectRegistry::with_operation_shapes(&[("a.Op", OperationShape::BinaryOperands)])
            .unwrap();
    let source = br#""builtin.module"() ({
^bb0:
  %x0 = "test.source"() : () -> i32
  %x1 = "test.source"() : () -> i32
  %r = a.Op %x0, %x1 : (i32) -> i1
  "test.after"() : () -> ()
}) : () -> ()"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(!parsed.syntax().diagnostics().is_empty());
    assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
        diagnostic.kind() == ParseDiagnosticKind::ShapeMismatch(OperationShape::BinaryOperands)
    }));
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );
}

#[test]
fn operation_shapes_cover_representative_spellings() {
    for (name, spelling, shape, operands, result, attribute) in [
        (
            "stir.mul",
            "%r = stir.mul %a, %b : bf16",
            OperationShape::BinaryOperands,
            2,
            Some("bf16"),
            None,
        ),
        (
            "stir.select",
            "%r = stir.select %c, %a, %b : i16, bf16 -> bf16",
            OperationShape::VariadicOperands,
            3,
            Some("bf16"),
            None,
        ),
        (
            "stir.reciprocal",
            "%r = stir.reciprocal %a : bf16",
            OperationShape::UnaryOperand,
            1,
            Some("bf16"),
            None,
        ),
        (
            "stir.return",
            "stir.return %a : bf16",
            OperationShape::OptionalTypedOperands,
            1,
            None,
            None,
        ),
        (
            "stir.return",
            "stir.return %a, %b : bf16, bf16",
            OperationShape::VariadicOperands,
            2,
            None,
            None,
        ),
        (
            "stir.iter_index",
            r#"%r = stir.iter_index "default_1" : i32"#,
            OperationShape::LiteralAttribute,
            0,
            Some("i32"),
            Some(r#""default_1""#),
        ),
        (
            "stir.imm",
            "%r = stir.imm 0 : i64 : i32",
            OperationShape::LiteralAttribute,
            0,
            Some("i32"),
            Some("0 : i64"),
        ),
        (
            "stir.arg_in",
            "%r = stir.arg_in -1.09e+12 : bf16 : bf16",
            OperationShape::LiteralAttribute,
            0,
            Some("bf16"),
            Some("-1.09e+12 : bf16"),
        ),
    ] {
        let registry = DialectRegistry::with_operation_shapes(&[(name, shape)]).unwrap();
        let source = format!(
            r#""builtin.module"() ({{
^bb0:
  %a = "test.source"() : () -> bf16
  %b = "test.source"() : () -> bf16
  %c = "test.source"() : () -> i16
  {spelling}
}}) : () -> ()"#
        );
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        assert!(
            parsed.syntax().diagnostics().is_empty(),
            "unexpected syntax diagnostics for {spelling:?}: {:?}",
            parsed.syntax().diagnostics()
        );
        let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
        assert!(
            lowered.diagnostics.is_empty(),
            "unexpected lowering diagnostics for {spelling:?}: {:?}",
            lowered.diagnostics
        );
        let document = lowered.document.unwrap();
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.operation_is_unparsed(operation), Some(false));
        assert_eq!(document.operands(operation).unwrap().len(), operands);
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), usize::from(result.is_some()));
        if let Some(expected) = result {
            assert_eq!(document.type_spelling(results[0]), Some(expected));
        }
        if let Some(expected) = attribute {
            let value = document.attribute_id(operation, "value").unwrap();
            assert_eq!(document.attribute_spelling_value(value), Some(expected));
        }
    }
}

#[test]
fn literal_attribute_shape_accepts_attribute_dictionaries() {
    let registry = DialectRegistry::with_operation_shapes(&[(
        "stir.arg_in",
        OperationShape::LiteralAttribute,
    )])
    .unwrap();
    for (spelling, value) in [
        (
            "%r = stir.arg_in 1 : i64 {k = 2 : i64} : i32 loc(unknown)",
            "1 : i64",
        ),
        ("%r = stir.arg_in 1 {k = 2 : i64} : i32", "1"),
    ] {
        let source = format!(
            r#""builtin.module"() ({{
^bb0:
  {spelling}
}}) : () -> ()"#
        );
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        assert!(
            parsed.syntax().diagnostics().is_empty(),
            "unexpected syntax diagnostics for {spelling:?}: {:?}",
            parsed.syntax().diagnostics()
        );
        let syntax = parsed
            .syntax()
            .file()
            .operations()
            .find(|operation| {
                operation.mnemonic_range().and_then(|range| {
                    source
                        .as_bytes()
                        .get(range.start() as usize..range.end() as usize)
                }) == Some(b"stir.arg_in")
            })
            .unwrap();
        let range = syntax.tree().text_range(syntax.id()).unwrap();
        assert_eq!(
            &source.as_bytes()[range.start() as usize..range.end() as usize],
            format!("{spelling}\n").as_bytes()
        );
        assert_eq!(
            syntax
                .attributes()
                .and_then(|dictionary| dictionary.tree().text_range(dictionary.id()))
                .map(|range| &source.as_bytes()[range.start() as usize..range.end() as usize]),
            Some(b"{k = 2 : i64}".as_slice())
        );
        assert_eq!(
            syntax.trailing_location().is_some(),
            spelling.contains("loc(")
        );

        let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
        assert!(
            lowered.diagnostics.is_empty(),
            "unexpected lowering diagnostics for {spelling:?}: {:?}",
            lowered.diagnostics
        );
        let document = lowered.document.unwrap();
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some("stir.arg_in"))
            .unwrap();
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(document.type_spelling(results[0]), Some("i32"));
        let literal = document.attribute_id(operation, "value").unwrap();
        assert_eq!(document.attribute_spelling_value(literal), Some(value));
        let dictionary_entry = document.attribute_id(operation, "k").unwrap();
        assert_eq!(
            document.attribute_spelling_value(dictionary_entry),
            Some("2 : i64")
        );
        assert!(
            document
                .operations()
                .all(|operation| document.operation_name(operation) != Some("k"))
        );
    }
}

#[test]
fn configured_format_programs_parse_and_lower_captured_roles() {
    use zirium::dialect::RegistryConfig;

    let config = RegistryConfig::from_json(
        r#"{
          "builtins": [],
          "operation_shapes": [],
          "operation_formats": [
            {"name":"a.Op","format":"$operands attr-dict `:` type($operands) `to` type($results)"},
            {"name":"a.Imm","format":"$value `:` type($value) attr-dict `:` type($result)"}
          ]
        }"#,
    )
    .unwrap();
    let registry = config.build().unwrap();
    assert_eq!(registry.operation_shape("a.Op"), None);

    let source = br#""builtin.module"() ({
^bb0:
  %a = "test.source"() : () -> i16
  %b = "test.source"() : () -> i16
  %r = a.Op %a, %b : i16 to i16
  %0 = a.Imm 1 : i64 {k = 2 : i64} : i32
}) : () -> ()"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();

    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("a.Op"))
        .unwrap();
    assert_eq!(document.operands(operation).unwrap().len(), 2);
    let results = document.result_types(operation).unwrap();
    assert_eq!(document.type_spelling(results[0]), Some("i16"));

    let literal = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("a.Imm"))
        .unwrap();
    let results = document.result_types(literal).unwrap();
    assert_eq!(document.type_spelling(results[0]), Some("i32"));
    let value = document.attribute_id(literal, "value").unwrap();
    assert_eq!(document.attribute_spelling_value(value), Some("1 : i64"));
    assert!(document.attribute_id(literal, "k").is_some());
}

#[test]
fn configured_format_programs_fail_during_registry_construction() {
    use zirium::dialect::RegistryConfig;

    let config = RegistryConfig::from_json(
        r#"{"builtins":[],"operation_shapes":[],"operation_formats":[
          {"name":"a.Broken","format":"$value `:` type($operands)"}
        ]}"#,
    )
    .unwrap();
    let error = match config.build() {
        Ok(_) => panic!("invalid format was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("a.Broken"), "{error}");
}

#[test]
fn configured_format_mismatch_recovers_the_whole_operation() {
    use zirium::dialect::RegistryConfig;

    let registry = RegistryConfig::from_json(
        r#"{"builtins":[],"operation_shapes":[],"operation_formats":[
          {"name":"a.Op","format":"$operands attr-dict `:` type($operands) `to` type($results)"}
        ]}"#,
    )
    .unwrap()
    .build()
    .unwrap();
    let source = br#""builtin.module"() ({
^bb0:
  %a = "test.source"() : () -> i16
  %r = a.Op %a : i16 -> i16
  "test.after"() : () -> ()
}) : () -> ()"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::FormatMismatch)
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );
    assert!(
        !document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("to"))
    );
}

#[test]
fn registry_presets_validate_names_and_compose_with_explicit_entries() {
    use zirium::dialect::RegistryConfig;

    let config = RegistryConfig::from_json(
        r#"{
          "presets": ["stablehlo"],
          "builtins": ["arith.constant"],
          "operation_shapes": [
            {"name": "vendor.function", "shape": "func_like"}
          ]
        }"#,
    )
    .unwrap();
    let registry = config.build().unwrap();
    assert!(registry.operation("arith.constant").is_some());
    assert_eq!(
        registry.operation_shape("stablehlo.add"),
        Some(OperationShape::BinaryOperands)
    );
    assert_eq!(
        registry.operation_shape("vendor.function"),
        Some(OperationShape::FuncLike)
    );

    assert!(DialectRegistry::from_name("unknown").is_err());
    let duplicate = RegistryConfig::from_json(
        r#"{"presets":["stablehlo","stablehlo"],"builtins":[],"operation_shapes":[]}"#,
    )
    .unwrap();
    assert!(duplicate.build().is_err());
}

#[test]
fn operation_shapes_preserve_the_core_module_alias() {
    let source = br#"module @outer {
      module @inner {
        "a.Op"() : () -> ()
      }
    }"#;
    let empty_extension = DialectRegistry::core()
        .extend_operation_shapes(&[])
        .unwrap();
    let shaped_extension = DialectRegistry::core()
        .extend_operation_shapes(&[("vendor.function", OperationShape::FuncLike)])
        .unwrap();

    for registry in [&empty_extension, &shaped_extension] {
        let parsed = ParsedFile::parse_with_registry(source.as_slice(), registry).unwrap();
        assert!(parsed.syntax().diagnostics().is_empty());
        assert_eq!(parsed.syntax().file().operations().count(), 3);
    }
}

#[test]
fn owned_operation_shapes_lower_neutral_func_and_call_forms() {
    assert!(std::mem::needs_drop::<DialectRegistry>());
    for index in 0..32 {
        let name = format!("vendor.temporary_{index}");
        let registry =
            DialectRegistry::with_operation_shapes(&[(name.as_str(), OperationShape::FuncLike)])
                .unwrap();
        drop(name);
        assert_eq!(
            registry.operation_shape(&format!("vendor.temporary_{index}")),
            Some(OperationShape::FuncLike)
        );
    }
    let registry = DialectRegistry::with_operation_shapes(&[
        ("vendor.function", OperationShape::FuncLike),
        ("vendor.invoke", OperationShape::CallLike),
    ])
    .unwrap();
    let source = br#"module {
      vendor.function @"quoted symbol"(%arg: i32 loc(unknown)) -> i32 attributes {tag = "body"} {
        %result = vendor.invoke @"quoted symbol"(%arg) {tag = "call"} : (i32) -> i32
        vendor.unregistered @other()
      }
      vendor.function @declaration()
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(parsed.syntax().diagnostics().len(), 1);
    assert_eq!(
        parsed.syntax().diagnostics()[0].kind(),
        ParseDiagnosticKind::UnknownCustomOperation
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("vendor.function"))
        .unwrap();
    let symbol = document.attribute_id(function, "sym_name").unwrap();
    assert_eq!(
        document.attribute_spelling_value(symbol),
        Some("@\"quoted symbol\"")
    );
    assert_eq!(
        document.operation_symbol_name(function).as_deref(),
        Some("quoted symbol")
    );
    assert_eq!(
        document.operation_signature(function).as_deref(),
        Some("(i32) -> i32")
    );
    let call = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("vendor.invoke"))
        .unwrap();
    assert_eq!(document.operands(call).unwrap().len(), 1);
    assert_eq!(document.result_types(call).unwrap().len(), 1);
    assert_eq!(
        document.operation_callee(call).as_deref(),
        Some("quoted symbol")
    );
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("vendor.unregistered"))
    );
}

#[test]
fn shaped_call_like_operations_accept_nested_symbol_callees() {
    let registry =
        DialectRegistry::with_operation_shapes(&[("vendor.invoke", OperationShape::CallLike)])
            .unwrap();

    for (spelling, expected) in [
        ("@root::@callee", "root::callee"),
        ("@root::@\"<lambda>_1\"", "root::<lambda>_1"),
    ] {
        let source = format!("%result = vendor.invoke {spelling}() : () -> i32");
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        assert!(parsed.syntax().diagnostics().is_empty(), "{spelling}");
        let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
        assert!(lowered.diagnostics.is_empty(), "{spelling}");
        let document = lowered.document.unwrap();
        let call = document.operations().next().unwrap();
        assert_eq!(document.operation_callee(call).as_deref(), Some(expected));
    }
}

#[test]
fn shaped_func_like_operations_own_trailing_locations() {
    let registry =
        DialectRegistry::with_operation_shapes(&[("vendor.function", OperationShape::FuncLike)])
            .unwrap();
    let source = br#"vendor.function @f(%arg: i32) -> i32 {
      "a.Nop"() : () -> ()
    } loc(#loc1)
    #loc1 = loc(unknown)"#;

    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let operation = parsed.syntax().file().operations().next().unwrap();
    assert!(operation.trailing_location().is_some());

    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    assert!(lowered.diagnostics.is_empty());
    let document = lowered.document.unwrap();
    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("vendor.function"))
        .unwrap();
    assert_eq!(
        document.operation_location(function),
        Some(Some("loc(#loc1)"))
    );
}

#[test]
fn declarative_operations_own_trailing_locations() {
    let registry = DialectRegistry::proving();
    for (name, source) in [
        ("builtin.module", "module @m {} loc(unknown)"),
        ("func.func", "func.func @f() { func.return } loc(unknown)"),
        ("func.call", "func.call @f() : () -> () loc(unknown)"),
        ("arith.constant", "%x = arith.constant 1 : i32 loc(unknown)"),
        ("arith.addi", "%x = arith.addi %a, %b : i32 loc(unknown)"),
        ("func.return", "func.return loc(unknown)"),
        ("cf.br", "cf.br ^next loc(unknown)"),
        (
            "cf.cond_br",
            "cf.cond_br %condition, ^yes, ^no loc(unknown)",
        ),
    ] {
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), registry).unwrap();
        assert!(parsed.syntax().diagnostics().is_empty(), "{name}");
        let operation = parsed.syntax().file().operations().next().unwrap();
        assert!(operation.trailing_location().is_some(), "{name}");
    }

    let source = b"module @outer { module @inner {} loc(#loc1) }\n#loc1 = loc(unknown)";
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), registry).unwrap();
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, registry);
    assert!(lowered.diagnostics.is_empty());
    let document = lowered.document.unwrap();
    let inner = document
        .operations()
        .find(|operation| document.operation_symbol_name(*operation).as_deref() == Some("inner"))
        .unwrap();
    assert_eq!(document.operation_location(inner), Some(Some("loc(#loc1)")));
}

#[test]
fn shaped_operations_use_header_syntax_boundaries() {
    let registry = DialectRegistry::with_operation_shapes(&[
        ("vendor.function", OperationShape::FuncLike),
        ("vendor.invoke", OperationShape::CallLike),
    ])
    .unwrap();
    let source = br#"module {
      vendor.function @callee(%arg: i32 {tag = {nested = "quoted } value"}, unit = true}) -> (i32 {tag = {nested = "quoted } comma, value"}}, tensor<2xi32> {other = "x,y"})
      %result.0 = vendor.invoke @callee() : () -> i32
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    let document = lowered
        .document
        .unwrap_or_else(|| panic!("lowering failed: {:?}", lowered.diagnostics));
    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("vendor.function"))
        .unwrap();
    let argument_attributes = document.attribute_id(function, "arg_attrs").unwrap();
    assert_eq!(
        document.attribute_spelling_value(argument_attributes),
        Some(r#"[{tag = {nested = "quoted } value"}, unit = true}]"#)
    );
    let function_type = document.attribute_id(function, "function_type").unwrap();
    assert_eq!(
        document.attribute_spelling_value(function_type),
        Some("(i32) -> (i32, tensor<2xi32>)")
    );
    let result_attributes = document.attribute_id(function, "res_attrs").unwrap();
    assert_eq!(
        document.attribute_spelling_value(result_attributes),
        Some(r#"[{tag = {nested = "quoted } comma, value"}}, {other = "x,y"}]"#)
    );
    let call = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("vendor.invoke"))
        .unwrap();
    assert_eq!(document.result_types(call).unwrap().len(), 1);
}

#[test]
fn unnamed_module_does_not_adopt_a_nested_function_symbol() {
    let parsed = ParsedFile::parse_with_registry(
        b"module { func.func @nested() }".as_slice(),
        DialectRegistry::core(),
    )
    .unwrap();
    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::core())
            .document
            .unwrap();
    let module = document.root_operations()[0];
    assert_eq!(document.operation_name(module), Some("builtin.module"));
    assert!(document.attribute_id(module, "sym_name").is_none());
}

#[test]
fn builtin_registries_accept_module_alias() {
    assert_eq!(
        DialectRegistry::core()
            .operation_names()
            .collect::<Vec<_>>(),
        ["builtin.module", "func.func", "func.return", "func.call"]
    );
    let source = b"module { module { } }".as_slice();
    let core = ParsedFile::parse_with_registry(source, DialectRegistry::core()).unwrap();
    assert!(core.syntax().diagnostics().is_empty());
    let document =
        lower_with_dialect_registry(&core, LoweringMode::Strict, DialectRegistry::core())
            .document
            .unwrap();
    let outer = document.root_operations()[0];
    assert_eq!(document.operation_name(outer), Some("builtin.module"));
    assert_eq!(document.statistics().operations, 2);

    let declarative = DialectRegistry::declarative(&["builtin.module"]).unwrap();
    for registry in [DialectRegistry::proving(), &declarative] {
        let parsed = ParsedFile::parse_with_registry(source, registry).unwrap();
        assert!(parsed.syntax().diagnostics().is_empty());
        assert!(
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, registry)
                .document
                .is_some()
        );
    }
    let empty = ParsedFile::parse_with_registry(source, &DialectRegistry::EMPTY).unwrap();
    assert!(!empty.syntax().diagnostics().is_empty());
}

#[test]
fn core_function_header_arguments_bind_to_the_entry_block() {
    let source = br#"module {
      func.func @add(%lhs: tensor<2xf32>, %rhs: tensor<2xf32>) -> tensor<2xf32> {
        %sum = "stablehlo.add"(%lhs, %rhs) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        func.return %sum : tensor<2xf32>
      }
    }"#;
    let parsed =
        ParsedFile::parse_with_registry(source.as_slice(), DialectRegistry::core()).unwrap();
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::core());
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    document.verify_semantics(DialectRegistry::core()).unwrap();
    assert_eq!(document.statistics().operations, 4);
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
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY);
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
    let document = lower_generic(
        "%0 = \"use\"() {tags = [#test.value<ok>, #test.value<ok>]} : () -> tuple<!test.value<ok>, !test.value<ok>>",
    );
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
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
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
fn mismatched_operation_shapes_recover_the_whole_operation() {
    let cases = [
        (
            "a.Op",
            OperationShape::VariadicOperands,
            "module { %a = a.Src : i16\n  %b = a.Src : i16\n  %r = a.Op %a, %b : i16 toward i16 }",
            4,
            "%r = a.Op %a, %b : i16 toward i16 ",
            "toward",
        ),
        (
            "a.Op",
            OperationShape::VariadicOperands,
            "module { %r = a.Op : i16 x.y }",
            2,
            "%r = a.Op : i16 x.y ",
            "x.y",
        ),
    ];

    for (name, shape, source, operation_count, recovered_text, fabricated_name) in cases {
        let registry = DialectRegistry::with_operation_shapes(&[(name, shape)]).unwrap();
        let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();
        let operations = parsed.syntax().file().operations().collect::<Vec<_>>();

        assert_eq!(operations.len(), operation_count, "{name}");
        assert_eq!(
            parsed
                .syntax()
                .diagnostics()
                .iter()
                .filter(|diagnostic| {
                    diagnostic.kind() == ParseDiagnosticKind::ShapeMismatch(shape)
                })
                .count(),
            1,
            "{name}"
        );
        assert_eq!(
            operations[0]
                .tree()
                .text_range(operations[0].id())
                .unwrap()
                .end(),
            source.len() as u32,
            "{name}"
        );

        let recovered = operations
            .iter()
            .copied()
            .find(|operation| {
                operation
                    .mnemonic_range()
                    .and_then(|range| source.get(range.start() as usize..range.end() as usize))
                    == Some(name)
            })
            .unwrap();
        assert_eq!(
            recovered.tree().kind(recovered.id()),
            Some(SyntaxKind::UnparsedCustomOperation),
            "{name}"
        );
        let range = recovered.tree().text_range(recovered.id()).unwrap();
        assert_eq!(
            source.get(range.start() as usize..range.end() as usize),
            Some(recovered_text),
            "{name}"
        );
        assert!(
            operations.iter().all(|operation| {
                operation
                    .mnemonic_range()
                    .and_then(|range| source.get(range.start() as usize..range.end() as usize))
                    != Some(fabricated_name)
            }),
            "{name}"
        );
    }
}

#[test]
fn matching_operation_shape_keeps_structured_syntax() {
    let registry =
        DialectRegistry::with_operation_shapes(&[("a.Op", OperationShape::VariadicOperands)])
            .unwrap();
    let source = "%r = a.Op %a, %b : i16";
    let parsed = ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap();

    assert!(parsed.syntax().diagnostics().is_empty());
    let operation = parsed.syntax().file().operations().next().unwrap();
    assert_eq!(
        operation.tree().kind(operation.id()),
        Some(SyntaxKind::DialectOperation)
    );
    assert_eq!(operation.operands().count(), 2);
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
    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
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
        let document =
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
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
    let generic_document = lower_with_dialect_registry(
        &generic_parsed,
        LoweringMode::Strict,
        &DialectRegistry::EMPTY,
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
    let fallback =
        lower_with_dialect_registry(&fallback, LoweringMode::Strict, &DialectRegistry::EMPTY)
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
    assert!(!registry.region("unknown.op", 0).requires_terminator);
    assert_eq!(registry.symbols("unknown.op"), Default::default());
}

#[test]
fn registered_function_termination_metadata_is_explicit() {
    let registry = DialectRegistry::proving();
    assert!(registry.region("func.func", 0).requires_terminator);
    assert!(!registry.region("builtin.module", 0).requires_terminator);
    for name in ["func.return", "cf.br", "cf.cond_br"] {
        assert!(registry.operation(name).unwrap().is_terminator, "{name}");
    }
    assert!(!registry.operation("func.call").unwrap().is_terminator);
}

#[test]
fn registered_function_blocks_require_a_final_terminator() {
    for (source, expected) in [
        (
            "builtin.module { func.func @unfinished(%flag: i1) { \"test.observe\"(%flag) : (i1) -> () \"test.record\"() : () -> () } }",
            "function block must end with a registered terminator",
        ),
        (
            "builtin.module { func.func @misordered() { cf.br ^done \"test.after_branch\"() : () -> () ^done: func.return } }",
            "terminator must be the final operation in its block",
        ),
        (
            "builtin.module { func.func @vacant() { } }",
            "function block must end with a registered terminator",
        ),
    ] {
        let document = lower_registered(source);
        assert!(matches!(
            document.verify_semantics(DialectRegistry::proving()),
            Err(SemanticVerificationError::Operation { message, .. }) if message == expected
        ));
    }

    let valid = lower_registered(
        r#"builtin.module {
  func.func @valid(%condition: i1) {
    cf.cond_br %condition, ^left, ^right
  ^left:
    cf.br ^exit
  ^right:
    cf.br ^exit
  ^exit:
    func.return
  }
}"#,
    );
    valid.verify_semantics(DialectRegistry::proving()).unwrap();
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
        is_terminator: false,
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
        is_terminator: false,
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
            is_terminator: false,
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
    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
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
    let redocument =
        lower_with_dialect_registry(&reparsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
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
        generic.contains("\"func.return\"(%v2) : (i32) -> ()"),
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
            lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
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
fn function_with_argument_prints_and_reparses_with_its_binding() {
    let document = lower_registered(
        "builtin.module { func.func @identity(%arg: i32) -> i32 { func.return %arg : i32 } }",
    );
    let mut text = String::new();
    document
        .print_with_registry(
            &mut text,
            PrintLayout::Compact,
            DialectPrintMode::PreferCustom,
            DialectRegistry::proving(),
        )
        .unwrap();

    let reparsed = lower_registered(&text);
    reparsed
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
    let function = reparsed
        .operations()
        .find(|&operation| FuncFuncOp::cast(&reparsed, operation).is_some())
        .unwrap();
    let signature = reparsed.attribute_id(function, "function_type").unwrap();
    assert_eq!(
        reparsed.attribute_spelling_value(signature),
        Some("(i32) -> i32"),
        "{text}"
    );
    let entry = reparsed
        .operation_regions(function)
        .and_then(|regions| regions.first())
        .and_then(|region| reparsed.region(*region))
        .and_then(|region| region.blocks(&reparsed))
        .and_then(|blocks| blocks.first())
        .copied()
        .unwrap();
    let returned = reparsed
        .operations()
        .find_map(|operation| FuncReturnOp::cast(&reparsed, operation))
        .unwrap();
    assert_eq!(
        returned.operands().unwrap(),
        &[zirium::semantic::ValueReference::Resolved(
            zirium::semantic::ValueId::BlockArgument {
                block: entry,
                argument: 0,
            },
        )],
        "{text}"
    );
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
            requires_terminator: false,
        }],
        symbols: SymbolDescriptor {
            defines_symbol: true,
            symbol_table: false,
            uses_symbols: true,
        },
        is_terminator: false,
    };
    let operations = Box::leak(Box::new([operation]));
    assert!(std::panic::catch_unwind(|| DialectRegistry::new(operations, &[], &[])).is_err());
}

#[test]
fn registration_rejects_wrong_fixed_descriptor_metadata() {
    static VALID_REGION: &[RegionDescriptor] = &[RegionDescriptor {
        kind: RegionKind::Ssacfg,
        isolated_from_above: true,
        requires_terminator: false,
    }];
    static WRONG_REGION: &[RegionDescriptor] = &[RegionDescriptor {
        kind: RegionKind::Graph,
        isolated_from_above: false,
        requires_terminator: false,
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
            is_terminator: false,
        }]));
        assert!(
            std::panic::catch_unwind(|| DialectRegistry::new(operations, &[], &[])).is_err(),
            "inconsistent metadata for {name} was accepted"
        );
    }
}

#[test]
fn registry_json_validates_records_and_registrations() {
    use zirium::dialect::RegistryConfig;
    let config =
        RegistryConfig::from_json(include_str!("../../../examples/cli/registry.json")).unwrap();
    let registry = config.build().unwrap();
    assert_eq!(
        registry.operation_shape("vendor.function"),
        Some(OperationShape::FuncLike)
    );
    assert!(registry.operation("arith.constant").is_some());
    assert!(registry.operation("cf.br").is_none());
    for json in [
        r#"{}"#,
        r#"{"builtins": [], "operation_shapes": [], "typo": true}"#,
        r#"{"builtins": [], "operation_shapes": [{"name": "a.b", "shape": "other"}]}"#,
        r#"{"builtins": [], "operation_shapes": [{"name": "a.b", "shape": "func_like", "typo": 1}]}"#,
        r#"{"builtins": null, "operation_shapes": []}"#,
    ] {
        assert!(RegistryConfig::from_json(json).is_err(), "{json}");
    }
    for json in [
        r#"{"builtins": ["unknown"], "operation_shapes": []}"#,
        r#"{"builtins": ["func.func", "func.func"], "operation_shapes": []}"#,
        r#"{"builtins": ["func.func"], "operation_shapes": [{"name":"func.func", "shape":"func_like"}]}"#,
        r#"{"builtins": [], "operation_shapes": [{"name":"a.b", "shape":"func_like"}, {"name":"a.b", "shape":"call_like"}]}"#,
    ] {
        assert!(
            RegistryConfig::from_json(json).unwrap().build().is_err(),
            "{json}"
        );
    }
    for name in [
        " a.b",
        "a.b ",
        "a b",
        "a.b()",
        "\"a.b\"",
        "i32",
        "a.b\n",
        "a.b//comment",
    ] {
        assert!(
            DialectRegistry::with_operation_shapes(&[(name, OperationShape::FuncLike)]).is_err(),
            "{name:?}"
        );
    }
}
