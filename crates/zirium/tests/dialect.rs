use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use zirium::{
    SyntaxKind,
    dialect::{
        AssemblyProgram, AttributeDescriptor, DialectRegistry, OperandCount, OperationDescriptor,
        OperationSchema, OperationShape, RegionDescriptor, RegionKind, RegistryConfig, ResultCount,
        SymbolDescriptor, TypeDescriptor,
    },
    parser::{ParseDiagnosticKind, ParsedFile},
    printer::{DialectPrintMode, PrintLayout},
    semantic::{
        ArithAddiOp, ArithConstantOp, AttributeValue, BuiltinModuleOp, CfBrOp, CfCondBrOp,
        FuncCallOp, FuncFuncOp, FuncReturnOp, LoweringMode, SemanticVerificationError, TypeValue,
        ValueId, ValueReference, lower_with_dialect_registry,
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
fn scf_preset_preserves_operation_arity_regions_and_header_bindings() {
    let registry = DialectRegistry::from_name("scf").unwrap();
    let source = br#"module {
      func.func @structured(%idx: index, %condition: i1, %value: i32, %output: tensor<4xi32>) {
        %executed = scf.execute_region -> i32 {
          scf.yield %value : i32
        }
        %looped = scf.for %iv = %idx to %idx step %idx iter_args(%iter = %value) -> i32 {
          scf.yield %iter : i32
        }
        %distributed = scf.forall (%thread, %static_thread) in (%idx, 4) shared_outs(%out = %output) -> tensor<4xi32> {
          scf.forall.in_parallel {
          }
        }
        %selected = scf.if %condition -> (i32) {
          scf.yield %value : i32
        } else {
          scf.yield %value : i32
        }
        %switched = scf.index_switch %idx -> i32
        case 0 {
          scf.yield %value : i32
        }
        case 1 {
          scf.yield %value : i32
        }
        default {
          scf.yield %value : i32
        }
        scf.index_switch %idx
        default {
          scf.yield
        }
        %parallel = scf.parallel (%p) = (%idx) to (%idx) step (%idx) init (%value) -> i32 {
          scf.reduce(%value, %value : i32, i32) {
          ^bb0(%lhs: i32, %rhs: i32):
            scf.reduce.return %lhs : i32
          }, {
          ^bb0(%lhs2: i32, %rhs2: i32):
            scf.reduce.return %rhs2 : i32
          }
        }
        %continued = scf.while (%before = %value) : (i32) -> i32 {
          "test.condition"(%condition, %before) : (i1, i32) -> ()
        } do {
        ^bb0(%after: i32):
          scf.yield %after : i32
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

    let operation = |name: &str| {
        document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    for (name, operands, results, regions) in [
        ("scf.execute_region", 0, 1, 1),
        ("scf.for", 4, 1, 1),
        ("scf.forall", 2, 1, 1),
        ("scf.forall.in_parallel", 0, 0, 1),
        ("scf.if", 1, 1, 2),
        ("scf.parallel", 4, 1, 1),
        ("scf.reduce", 2, 0, 2),
        ("scf.while", 1, 1, 2),
    ] {
        let operation = operation(name);
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert_eq!(
            document.operation_regions(operation).unwrap().len(),
            regions,
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }

    let switches = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("scf.index_switch"))
        .collect::<Vec<_>>();
    assert_eq!(switches.len(), 2);
    for (operation, results, regions) in [(switches[0], 1, 3), (switches[1], 0, 1)] {
        assert_eq!(document.operands(operation).unwrap().len(), 1);
        assert_eq!(document.result_types(operation).unwrap().len(), results);
        assert_eq!(
            document.operation_regions(operation).unwrap().len(),
            regions
        );
        assert!(document.successors(operation).unwrap().is_empty());
    }

    for operation in document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("scf.reduce.return"))
    {
        assert_eq!(document.operands(operation).unwrap().len(), 1);
        assert_eq!(document.result_types(operation).unwrap().len(), 0);
        assert!(document.successors(operation).unwrap().is_empty());
    }
    for operation in document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("scf.yield"))
    {
        assert_eq!(document.result_types(operation).unwrap().len(), 0);
        assert!(document.successors(operation).unwrap().is_empty());
    }
    for (name, expected_arguments) in [("scf.for", 2), ("scf.forall", 3), ("scf.parallel", 1)] {
        let region = document.operation_regions(operation(name)).unwrap()[0];
        let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
        assert_eq!(
            document.block_argument_types(block).unwrap().len(),
            expected_arguments,
            "{name}"
        );
    }

    assert_eq!(
        registry.operation_shape("scf.reduce.return"),
        Some(OperationShape::OptionalTypedOperands)
    );
    assert_eq!(registry.operation_shape("scf.condition"), None);
}

#[test]
fn scf_condition_recovers_without_inventing_a_result() {
    let registry = DialectRegistry::from_name("scf").unwrap();
    let source = br#"func.func @condition(%condition: i1, %value: i32) {
      scf.condition(%condition) %value : i32
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();

    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    assert!(!lowered.semantically_complete);
    let document = lowered.document.unwrap();
    let condition = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("scf.condition"))
        .unwrap();
    assert_eq!(document.result_types(condition).unwrap().len(), 0);
}

#[test]
fn linalg_preset_keeps_regions_block_arguments_and_relayout_operands() {
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
      func.func @relayout(%input: tensor<7x16xf32>, %packed: tensor<4x16x2xf32>, %tile: index, %pad: f32) -> tensor<7x16xf32> {
        %dynamic_pack = linalg.pack %input padding_value(%pad : f32) inner_dims_pos = [0] inner_tiles = [%tile] into %packed : tensor<7x16xf32> -> tensor<4x16x2xf32>
        %static_pack = linalg.pack %input outer_dims_perm = [0, 1] inner_dims_pos = [0] inner_tiles = [2] into %packed : tensor<7x16xf32> -> tensor<4x16x2xf32>
        %static_unpack = linalg.unpack %dynamic_pack outer_dims_perm = [0, 1] inner_dims_pos = [0] inner_tiles = [2] into %input : tensor<4x16x2xf32> -> tensor<7x16xf32>
        %dynamic_unpack = linalg.unpack %static_pack inner_dims_pos = [0] inner_tiles = [%tile] into %input : tensor<4x16x2xf32> -> tensor<7x16xf32>
        func.return %dynamic_unpack : tensor<7x16xf32>
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
    let packs = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("linalg.pack"))
        .collect::<Vec<_>>();
    assert_eq!(packs.len(), 2);
    assert_eq!(document.operands(packs[0]).unwrap().len(), 4);
    assert_eq!(document.operands(packs[1]).unwrap().len(), 2);
    let unpacks = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("linalg.unpack"))
        .collect::<Vec<_>>();
    assert_eq!(unpacks.len(), 2);
    assert_eq!(document.operands(unpacks[0]).unwrap().len(), 2);
    assert_eq!(document.operands(unpacks[1]).unwrap().len(), 3);
    for operation in packs.into_iter().chain(unpacks) {
        assert!(document.attribute_id(operation, "inner_dims_pos").is_some());
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            document.type_spelling(results[0]),
            Some(
                if document.operation_name(operation) == Some("linalg.pack") {
                    "tensor<4x16x2xf32>"
                } else {
                    "tensor<7x16xf32>"
                }
            )
        );
    }
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
fn amdgpu_preset_lowers_only_structurally_honest_forms() {
    let registry = DialectRegistry::from_name("amdgpu").unwrap();
    let source = br#"module {
      func.func @kernel(%packed: vector<4xf8E4M3FNUZ>, %matrix: vector<8xf4E2M1FN>, %scale: vector<4xf8E8M0FNU>) {
        %extended = amdgpu.ext_packed_fp8 {tag = true} %packed[0] : vector<4xf8E4M3FNUZ> to f32
        %scaled = amdgpu.scaled_ext_packed_matrix %matrix scale(%scale) blockSize(32) firstScaleLane(0) firstScaleByte(0) : vector<8xf4E2M1FN>, vector<4xf8E8M0FNU> -> vector<8xf32>
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

    for (name, operands, results) in [
        ("amdgpu.ext_packed_fp8", 1, 1),
        ("amdgpu.scaled_ext_packed_matrix", 2, 1),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
    }

    let extended = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("amdgpu.ext_packed_fp8"))
        .unwrap();
    assert!(
        document
            .attributes(extended)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );

    let opaque_source = br#"module {
      func.func @tensor(%desc: !amdgpu.tdm_descriptor) {
        amdgpu.tensor_load_to_lds %desc : !amdgpu.tdm_descriptor
        amdgpu.tensor_store_from_lds %desc : !amdgpu.tdm_descriptor
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(opaque_source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().is_empty());
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let opaque_document = lowered.document.unwrap();
    for name in ["amdgpu.tensor_load_to_lds", "amdgpu.tensor_store_from_lds"] {
        let operation = opaque_document
            .operations()
            .find(|operation| opaque_document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(opaque_document.operands(operation).unwrap().len(), 1);
        assert!(opaque_document.result_types(operation).unwrap().is_empty());
        assert!(
            opaque_document
                .operation_regions(operation)
                .unwrap()
                .is_empty()
        );
    }

    for name in [
        "amdgpu.dpp",
        "amdgpu.packed_trunc_2xfp8",
        "amdgpu.permlane_swap",
        "amdgpu.swizzle_bitmode",
        "amdgpu.lds_barrier",
        "amdgpu.raw_buffer_store",
        "amdgpu.mfma",
        "amdgpu.gather_to_lds",
        "amdgpu.make_dma_descriptor",
        "amdgpu.memory_counter_wait",
    ] {
        assert_eq!(registry.operation_shape(name), None, "{name}");
    }
}

#[test]
fn amx_preset_lowers_only_the_structurally_honest_zero_form() {
    let registry = DialectRegistry::from_name("amx").unwrap();
    let source = br#"module {
      func.func @zero() {
        %tile = amx.tile_zero {tag = true} : !amx.tile<16x16xbf16>
        "func.return"() : () -> ()
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
    let zero = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("amx.tile_zero"))
        .unwrap();
    assert!(document.operands(zero).unwrap().is_empty());
    assert_eq!(
        document
            .result_types(zero)
            .unwrap()
            .iter()
            .map(|ty| document.type_spelling(*ty).unwrap())
            .collect::<Vec<_>>(),
        ["!amx.tile<16x16xbf16>"]
    );
    assert!(
        document
            .attributes(zero)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );
    assert!(document.operation_regions(zero).unwrap().is_empty());
    assert!(document.successors(zero).unwrap().is_empty());

    // These trailers describe operand roles or use syntax that the current
    // reusable shapes cannot lower without inventing semantic results.
    for name in [
        "amx.tile_load",
        "amx.tile_store",
        "amx.tile_mulf",
        "amx.tile_muli",
    ] {
        assert_eq!(registry.operation_shape(name), None, "{name}");
    }
}

#[test]
fn arith_preset_exposes_default_unary_binary_and_cast_structure() {
    let registry = DialectRegistry::from_name("arith").unwrap();
    let source = br#"module {
      func.func @calculate(%lhs: i32, %rhs: i32, %float: f32) -> i64 {
        %one = arith.constant 1 : i32
        %sum = arith.addi %lhs, %one overflow<nsw> : i32
        %difference = arith.subi %sum, %rhs {tag = true} : i32
        %negated = arith.negf %float : f32
        %wide = arith.extsi %difference : i32 to i64
        func.return %wide : i64
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

    for (name, operands, result) in [
        ("arith.constant", 0, "i32"),
        ("arith.addi", 2, "i32"),
        ("arith.subi", 2, "i32"),
        ("arith.negf", 1, "f32"),
        ("arith.extsi", 1, "i64"),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1, "{name}");
        assert_eq!(document.type_spelling(results[0]), Some(result), "{name}");
        assert!(document.operation_regions(operation).unwrap().is_empty());
        assert!(document.successors(operation).unwrap().is_empty());
    }

    let add = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arith.addi"))
        .unwrap();
    assert!(
        document
            .attributes(add)
            .unwrap()
            .any(|(name, value)| { name == "overflowFlags" && value == "#arith.overflow<nsw>" })
    );
    let sub = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arith.subi"))
        .unwrap();
    assert!(
        document
            .attributes(sub)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );
}

#[test]
fn arith_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("arith").unwrap();
    let binary = [
        "arith.subi",
        "arith.muli",
        "arith.divui",
        "arith.divsi",
        "arith.ceildivui",
        "arith.ceildivsi",
        "arith.floordivsi",
        "arith.remui",
        "arith.remsi",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shli",
        "arith.shrui",
        "arith.shrsi",
        "arith.addf",
        "arith.subf",
        "arith.maximumf",
        "arith.maxnumf",
        "arith.maxsi",
        "arith.maxui",
        "arith.minimumf",
        "arith.minnumf",
        "arith.minsi",
        "arith.minui",
        "arith.mulf",
        "arith.divf",
        "arith.remf",
    ];
    let unary = [
        "arith.negf",
        "arith.extui",
        "arith.extsi",
        "arith.extf",
        "arith.trunci",
        "arith.truncf",
        "arith.uitofp",
        "arith.sitofp",
        "arith.fptoui",
        "arith.fptosi",
        "arith.index_cast",
        "arith.index_castui",
        "arith.bitcast",
    ];
    for name in binary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::BinaryOperands),
            "{name}"
        );
    }
    for name in unary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::UnaryOperand),
            "{name}"
        );
    }
    assert!(registry.operation("arith.constant").is_some());
    assert!(registry.operation("arith.addi").is_some());
    assert_eq!(binary.len() + unary.len() + 2, 43);

    for name in [
        "arith.addui_extended",
        "arith.mulsi_extended",
        "arith.mului_extended",
        "arith.scaling_extf",
        "arith.scaling_truncf",
        "arith.cmpi",
        "arith.cmpf",
        "arith.select",
    ] {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn arm_neon_preset_exposes_the_widening_multiply_structure() {
    let registry = DialectRegistry::from_name("arm_neon").unwrap();
    let source = br#"module {
      func.func @widen(%lhs: vector<8xi8>, %rhs: vector<8xi8>) -> vector<8xi16> {
        %result = arm_neon.intr.smull %lhs, %rhs {tag = true} : vector<8xi8> to vector<8xi16>
        func.return %result : vector<8xi16>
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
    let smull = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arm_neon.intr.smull"))
        .unwrap();

    assert_eq!(document.operands(smull).unwrap().len(), 2);
    let result_types = document.result_types(smull).unwrap();
    assert_eq!(result_types.len(), 1);
    assert_eq!(
        document.type_spelling(result_types[0]),
        Some("vector<8xi16>")
    );
    assert!(
        document
            .attributes(smull)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );
    assert!(document.operation_regions(smull).unwrap().is_empty());
    assert!(document.successors(smull).unwrap().is_empty());
}

#[test]
fn arm_neon_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("arm_neon").unwrap();
    assert_eq!(
        registry.operation_shape("arm_neon.intr.smull"),
        Some(OperationShape::BinaryOperands)
    );

    for name in [
        "arm_neon.intr.sdot",
        "arm_neon.intr.smmla",
        "arm_neon.intr.ummla",
        "arm_neon.intr.usmmla",
        "arm_neon.intr.bfmmla",
        "arm_neon.2d.sdot",
    ] {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn arm_sme_preset_exposes_result_only_and_same_typed_tile_forms() {
    let registry = DialectRegistry::from_name("arm_sme").unwrap();
    let source = br#"module {
      func.func @tiles(%input: vector<[4]x[4]xf32>) -> vector<[4]x[4]xf32> {
        %fresh = arm_sme.get_tile {origin = "fresh"} : vector<[4]x[4]xf32>
        %zero = arm_sme.zero : vector<[4]x[4]xf32>
        %copy = arm_sme.copy_tile %zero {tag = true} : vector<[4]x[4]xf32>
        func.return %copy : vector<[4]x[4]xf32>
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

    for name in ["arm_sme.get_tile", "arm_sme.zero", "arm_sme.copy_tile"] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.result_types(operation).unwrap().len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(document.result_types(operation).unwrap()[0]),
            Some("vector<[4]x[4]xf32>"),
            "{name}"
        );
        assert!(document.operation_regions(operation).unwrap().is_empty());
        assert!(document.successors(operation).unwrap().is_empty());
    }

    let copy = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arm_sme.copy_tile"))
        .unwrap();
    assert_eq!(document.operands(copy).unwrap().len(), 1);
    assert!(
        document
            .attributes(copy)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );
}

#[test]
fn arm_sme_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("arm_sme").unwrap();
    for name in ["arm_sme.get_tile", "arm_sme.zero"] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::VariadicOperands),
            "{name}"
        );
    }
    assert_eq!(
        registry.operation_shape("arm_sme.copy_tile"),
        Some(OperationShape::UnaryOperand)
    );

    let unsupported = [
        "arm_sme.tile_load",
        "arm_sme.tile_store",
        "arm_sme.load_tile_slice",
        "arm_sme.store_tile_slice",
        "arm_sme.insert_tile_slice",
        "arm_sme.extract_tile_slice",
        "arm_sme.outerproduct",
        "arm_sme.fmopa_2way",
        "arm_sme.fmops_2way",
        "arm_sme.smopa_2way",
        "arm_sme.smops_2way",
        "arm_sme.umopa_2way",
        "arm_sme.umops_2way",
        "arm_sme.smopa_4way",
        "arm_sme.smops_4way",
        "arm_sme.umopa_4way",
        "arm_sme.umops_4way",
        "arm_sme.sumopa_4way",
        "arm_sme.sumops_4way",
        "arm_sme.usmopa_4way",
        "arm_sme.usmops_4way",
        "arm_sme.streaming_vl",
        "arm_sme.intr.zero",
        "arm_sme.intr.mopa",
        "arm_sme.intr.mops",
        "arm_sme.intr.mopa.wide",
        "arm_sme.intr.mops.wide",
        "arm_sme.intr.smopa.wide",
        "arm_sme.intr.smops.wide",
        "arm_sme.intr.umopa.wide",
        "arm_sme.intr.umops.wide",
        "arm_sme.intr.sumopa.wide",
        "arm_sme.intr.sumops.wide",
        "arm_sme.intr.usmopa.wide",
        "arm_sme.intr.usmops.wide",
        "arm_sme.intr.smopa.za32",
        "arm_sme.intr.umopa.za32",
        "arm_sme.intr.smops.za32",
        "arm_sme.intr.umops.za32",
        "arm_sme.intr.ld1b.horiz",
        "arm_sme.intr.ld1h.horiz",
        "arm_sme.intr.ld1w.horiz",
        "arm_sme.intr.ld1d.horiz",
        "arm_sme.intr.ld1q.horiz",
        "arm_sme.intr.ld1b.vert",
        "arm_sme.intr.ld1h.vert",
        "arm_sme.intr.ld1w.vert",
        "arm_sme.intr.ld1d.vert",
        "arm_sme.intr.ld1q.vert",
        "arm_sme.intr.st1b.horiz",
        "arm_sme.intr.st1h.horiz",
        "arm_sme.intr.st1w.horiz",
        "arm_sme.intr.st1d.horiz",
        "arm_sme.intr.st1q.horiz",
        "arm_sme.intr.st1b.vert",
        "arm_sme.intr.st1h.vert",
        "arm_sme.intr.st1w.vert",
        "arm_sme.intr.st1d.vert",
        "arm_sme.intr.st1q.vert",
        "arm_sme.intr.str",
        "arm_sme.intr.write.horiz",
        "arm_sme.intr.write.vert",
        "arm_sme.intr.read.horiz",
        "arm_sme.intr.read.vert",
        "arm_sme.intr.cntsd",
    ];
    assert_eq!(unsupported.len() + 3, 68);
    for name in unsupported {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn arm_sve_preset_exposes_mask_and_source_type_relationships() {
    let registry = DialectRegistry::from_name("arm_sve").unwrap();
    let source = br#"module {
      func.func @masked(
          %mask: vector<2x[4]xi1>,
          %lhs: vector<2x[4]xf32>,
          %rhs: vector<2x[4]xf32>) -> vector<2x[4]xf32> {
        %result = arm_sve.masked.addf %mask, %lhs, %rhs {tag = true} :
            vector<2x[4]xi1>, vector<2x[4]xf32>
        func.return %result : vector<2x[4]xf32>
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
    document.verify_semantics(&registry).unwrap();
    let addf = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arm_sve.masked.addf"))
        .unwrap();

    assert_eq!(document.operands(addf).unwrap().len(), 3);
    let result_types = document.result_types(addf).unwrap();
    assert_eq!(result_types.len(), 1);
    assert_eq!(
        document.type_spelling(result_types[0]),
        Some("vector<2x[4]xf32>")
    );
    assert_eq!(
        document.type_spelling(document.function_type(addf).unwrap()),
        Some("(vector<2x[4]xi1>, vector<2x[4]xf32>, vector<2x[4]xf32>) -> vector<2x[4]xf32>")
    );
    assert!(
        document
            .attributes(addf)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );
    assert!(document.operation_regions(addf).unwrap().is_empty());
    assert!(document.successors(addf).unwrap().is_empty());
}

#[test]
fn arm_sve_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("arm_sve").unwrap();
    let masked = [
        "arm_sve.masked.addi",
        "arm_sve.masked.addf",
        "arm_sve.masked.subi",
        "arm_sve.masked.subf",
        "arm_sve.masked.muli",
        "arm_sve.masked.mulf",
        "arm_sve.masked.divi_signed",
        "arm_sve.masked.divi_unsigned",
        "arm_sve.masked.divf",
    ];
    for name in masked {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::OperandClauses),
            "{name}"
        );
    }

    let unsupported_custom = [
        "arm_sve.sdot",
        "arm_sve.smmla",
        "arm_sve.udot",
        "arm_sve.ummla",
        "arm_sve.usmmla",
        "arm_sve.intr.bfmmla",
        "arm_sve.convert_from_svbool",
        "arm_sve.convert_to_svbool",
        "arm_sve.zip.x2",
        "arm_sve.zip.x4",
        "arm_sve.psel",
        "arm_sve.dupq_lane",
    ];
    for name in unsupported_custom {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }

    let generic_intrinsics = [
        "arm_sve.intr.ummla",
        "arm_sve.intr.smmla",
        "arm_sve.intr.usmmla",
        "arm_sve.intr.sdot",
        "arm_sve.intr.udot",
        "arm_sve.intr.add",
        "arm_sve.intr.fadd",
        "arm_sve.intr.mul",
        "arm_sve.intr.fmul",
        "arm_sve.intr.sub",
        "arm_sve.intr.fsub",
        "arm_sve.intr.sdiv",
        "arm_sve.intr.udiv",
        "arm_sve.intr.fdiv",
        "arm_sve.intr.convert.from.svbool",
        "arm_sve.intr.convert.to.svbool",
        "arm_sve.intr.zip.x2",
        "arm_sve.intr.zip.x4",
        "arm_sve.intr.psel",
        "arm_sve.intr.whilelt",
        "arm_sve.intr.dupq_lane",
    ];
    assert_eq!(masked.len() + unsupported_custom.len(), 21);
    assert_eq!(generic_intrinsics.len(), 21);
    for name in generic_intrinsics {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn async_preset_exposes_function_call_and_runtime_structure() {
    let registry = DialectRegistry::from_name("async").unwrap();
    let source = br#"module {
      async.func @produce(%arg: f32) -> (!async.token, !async.value<f32>) {
        async.return %arg : f32
      }
      func.func @drive(%arg: f32) {
        %token, %value = async.call @produce(%arg) : (f32) -> (!async.token, !async.value<f32>)
        %created = async.runtime.create {tag = true} : !async.token
        %threads = async.runtime.num_worker_threads : index
        async.runtime.set_available %created : !async.token
        async.runtime.set_error %created : !async.token
        async.runtime.await %value : !async.value<f32>
        async.runtime.add_ref %value {count = 1 : i64} : !async.value<f32>
        async.runtime.drop_ref %value {count = 1 : i64} : !async.value<f32>
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

    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("async.func"))
        .unwrap();
    assert_eq!(
        document.operation_symbol_name(function).as_deref(),
        Some("produce")
    );
    assert_eq!(
        document.operation_signature(function).as_deref(),
        Some("(f32) -> (!async.token, !async.value<f32>)")
    );
    assert_eq!(document.operation_regions(function).unwrap().len(), 1);

    let call = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("async.call"))
        .unwrap();
    assert_eq!(document.operation_callee(call).as_deref(), Some("produce"));
    assert_eq!(document.operands(call).unwrap().len(), 1);
    let call_results = document.result_types(call).unwrap();
    assert_eq!(call_results.len(), 2);
    assert_eq!(
        document.type_spelling(call_results[0]),
        Some("!async.token")
    );
    assert_eq!(
        document.type_spelling(call_results[1]),
        Some("!async.value<f32>")
    );

    for (name, operands, result) in [
        ("async.runtime.create", 0, Some("!async.token")),
        ("async.runtime.num_worker_threads", 0, Some("index")),
        ("async.runtime.set_available", 1, None),
        ("async.runtime.set_error", 1, None),
        ("async.runtime.await", 1, None),
        ("async.runtime.add_ref", 1, None),
        ("async.runtime.drop_ref", 1, None),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), usize::from(result.is_some()), "{name}");
        if let Some(expected) = result {
            assert_eq!(document.type_spelling(results[0]), Some(expected), "{name}");
        }
    }

    let add_ref = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("async.runtime.add_ref"))
        .unwrap();
    assert!(document.attribute_id(add_ref, "count").is_some());
}

#[test]
fn async_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("async").unwrap();
    for (name, shape) in [
        ("async.func", OperationShape::FuncLike),
        ("async.call", OperationShape::CallLike),
        ("async.return", OperationShape::OptionalTypedOperands),
        ("async.yield", OperationShape::OptionalTypedOperands),
        ("async.runtime.create", OperationShape::VariadicOperands),
        (
            "async.runtime.set_available",
            OperationShape::OptionalTypedOperands,
        ),
        (
            "async.runtime.set_error",
            OperationShape::OptionalTypedOperands,
        ),
        ("async.runtime.await", OperationShape::OptionalTypedOperands),
        (
            "async.runtime.add_ref",
            OperationShape::OptionalTypedOperands,
        ),
        (
            "async.runtime.drop_ref",
            OperationShape::OptionalTypedOperands,
        ),
        (
            "async.runtime.num_worker_threads",
            OperationShape::VariadicOperands,
        ),
    ] {
        assert_eq!(registry.operation_shape(name), Some(shape), "{name}");
    }

    let unsupported = [
        "async.execute",
        "async.await",
        "async.create_group",
        "async.add_to_group",
        "async.await_all",
        "async.coro.id",
        "async.coro.begin",
        "async.coro.free",
        "async.coro.end",
        "async.coro.save",
        "async.coro.suspend",
        "async.runtime.create_group",
        "async.runtime.is_error",
        "async.runtime.resume",
        "async.runtime.await_and_resume",
        "async.runtime.store",
        "async.runtime.load",
        "async.runtime.add_to_group",
    ];
    assert_eq!(11 + unsupported.len(), 29);
    for name in unsupported {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn bufferization_preset_exposes_only_spelled_operand_and_result_types() {
    let registry = DialectRegistry::from_name("bufferization").unwrap();
    let source = br#"module {
      func.func @bufferize(%tensor: tensor<4xf32>, %buffer: memref<4xf32>) {
        %clone = bufferization.clone %buffer {tag = "clone"} : memref<4xf32> to memref<4xf32>
        %from_buffer = bufferization.to_tensor %buffer restrict writable {tag = "tensor"} : memref<4xf32> to tensor<4xf32>
        %to_buffer = bufferization.to_buffer %tensor read_only : tensor<4xf32> to memref<4xf32>
        %materialized = bufferization.materialize_in_destination %tensor in %from_buffer : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        bufferization.materialize_in_destination %tensor in restrict writable %buffer : (tensor<4xf32>, memref<4xf32>) -> ()
        bufferization.dealloc_tensor %materialized : tensor<4xf32>
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

    for (name, operands, results) in [
        ("bufferization.clone", 1, &["memref<4xf32>"][..]),
        ("bufferization.to_tensor", 1, &["tensor<4xf32>"][..]),
        ("bufferization.to_buffer", 1, &["memref<4xf32>"][..]),
        (
            "bufferization.materialize_in_destination",
            2,
            &["tensor<4xf32>"][..],
        ),
        ("bufferization.dealloc_tensor", 1, &[][..]),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document
                .result_types(operation)
                .unwrap()
                .iter()
                .map(|ty| document.type_spelling(*ty).unwrap())
                .collect::<Vec<_>>(),
            results,
            "{name}"
        );
    }

    let materializations = document
        .operations()
        .filter(|operation| {
            document.operation_name(*operation) == Some("bufferization.materialize_in_destination")
        })
        .collect::<Vec<_>>();
    assert_eq!(materializations.len(), 2);
    assert_eq!(document.result_types(materializations[1]).unwrap().len(), 0);

    let clone = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("bufferization.clone"))
        .unwrap();
    assert!(
        document
            .attributes(clone)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "\"clone\"")
    );
}

#[test]
fn bufferization_preset_inventory_matches_llvm_22_1_custom_forms() {
    let registry = DialectRegistry::from_name("bufferization").unwrap();
    for (name, shape) in [
        ("bufferization.clone", OperationShape::UnaryOperand),
        (
            "bufferization.materialize_in_destination",
            OperationShape::OperandClauses,
        ),
        (
            "bufferization.dealloc_tensor",
            OperationShape::OptionalTypedOperands,
        ),
        ("bufferization.to_tensor", OperationShape::OperandClauses),
        ("bufferization.to_buffer", OperationShape::OperandClauses),
    ] {
        assert_eq!(registry.operation_shape(name), Some(shape), "{name}");
    }

    let unsupported = ["bufferization.alloc_tensor", "bufferization.dealloc"];
    assert_eq!(5 + unsupported.len(), 7);
    for name in unsupported {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn cf_preset_preserves_and_resolves_successors() {
    let registry = DialectRegistry::from_name("cf").unwrap();
    let source = br#"module {
      func.func @route(%condition: i1, %value: i32) {
        cf.cond_br %condition, ^left(%value : i32), ^right(%value : i32) {branch_weights = dense<[3, 2]> : vector<2xi32>, tag = "condition"}
      ^left(%left_value: i32):
        cf.br ^join(%left_value : i32) {tag = "left"}
      ^right(%right_value: i32):
        cf.br ^join(%right_value : i32)
      ^join(%joined: i32):
        "test.consume"(%joined) : (i32) -> ()
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
    document.verify_semantics(&registry).unwrap();

    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.func"))
        .unwrap();
    let region = document.operation_regions(function).unwrap()[0];
    let blocks = document.region(region).unwrap().blocks(&document).unwrap();
    let condition = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("cf.cond_br"))
        .unwrap();
    let condition_successors = document.successors(condition).unwrap();
    assert_eq!(condition_successors.len(), 2);
    assert!(document.attribute_id(condition, "branch_weights").is_some());
    assert!(document.attribute_id(condition, "tag").is_some());
    assert_eq!(condition_successors[0].block(), blocks[1]);
    assert_eq!(condition_successors[1].block(), blocks[2]);
    for successor in condition_successors {
        assert_eq!(
            document.successor_arguments(*successor),
            Some(
                &[ValueReference::Resolved(ValueId::BlockArgument {
                    block: blocks[0],
                    argument: 1,
                })][..]
            )
        );
    }

    let branches = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("cf.br"))
        .collect::<Vec<_>>();
    assert_eq!(branches.len(), 2);
    for (branch, source_block) in branches.iter().copied().zip([blocks[1], blocks[2]]) {
        assert!(document.result_types(branch).unwrap().is_empty());
        assert!(document.operation_regions(branch).unwrap().is_empty());
        let successor = document.successors(branch).unwrap()[0];
        assert_eq!(successor.block(), blocks[3]);
        assert_eq!(
            document.successor_arguments(successor),
            Some(
                &[ValueReference::Resolved(ValueId::BlockArgument {
                    block: source_block,
                    argument: 0,
                })][..]
            )
        );
    }
    assert!(
        document
            .attributes(branches[0])
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "\"left\"")
    );
    assert!(document.result_types(condition).unwrap().is_empty());
    assert!(document.operation_regions(condition).unwrap().is_empty());
}

#[test]
fn cf_preset_inventory_and_recovery_match_llvm_22_1() {
    let registry = DialectRegistry::from_name("cf").unwrap();
    let supported = ["cf.br", "cf.cond_br"];
    for name in supported {
        assert!(registry.operation(name).is_some(), "{name}");
    }
    let unsupported = ["cf.assert", "cf.switch"];
    assert_eq!(supported.len() + unsupported.len(), 4);
    for name in unsupported {
        assert!(registry.operation(name).is_none(), "{name}");
        assert_eq!(registry.operation_shape(name), None, "{name}");
    }

    let source = br#"module {
      func.func @gaps(%condition: i1, %flag: i32) {
        cf.assert %condition, "condition failed" {tag = true}
        cf.switch %flag : i32, [
          default: ^exit,
          7: ^exit
        ] {tag = "switch"}
      ^exit:
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        2
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );
}

#[test]
fn complex_preset_exposes_default_unary_binary_and_bitcast_structure() {
    let registry = DialectRegistry::from_name("complex").unwrap();
    let source = br#"module {
      func.func @calculate(%value: complex<f32>) -> i64 {
        %cosine = complex.cos %value {tag = "unary"} : complex<f32>
        %sum = complex.add %cosine, %value {tag = "binary"} : complex<f32>
        %bits = complex.bitcast %sum {tag = "cast"} : complex<f32> to i64
        %constant = complex.constant [0.1, -1.0] {tag = "constant"} : complex<f32>
        "test.attribute"() {value = #complex.number<:f32 1.0, 2.0>} : () -> ()
        func.return %bits : i64
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

    for (name, operands, result_type, tag) in [
        ("complex.cos", 1, "complex<f32>", "\"unary\""),
        ("complex.add", 2, "complex<f32>", "\"binary\""),
        ("complex.bitcast", 1, "i64", "\"cast\""),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(results[0]),
            Some(result_type),
            "{name}"
        );
        assert!(
            document
                .attributes(operation)
                .unwrap()
                .any(|(attribute, value)| attribute == "tag" && value == tag),
            "{name}"
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }

    let constant = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("complex.constant"))
        .unwrap();
    assert!(document.operands(constant).unwrap().is_empty());
    let results = document.result_types(constant).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(document.type_spelling(results[0]), Some("complex<f32>"));
    assert!(
        document
            .attributes(constant)
            .unwrap()
            .any(|(name, value)| name == "value" && value == "[0.1, -1.0]")
    );
    assert!(
        document
            .attributes(constant)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "\"constant\"")
    );

    let attribute = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.attribute"))
        .unwrap();
    assert!(
        document
            .attributes(attribute)
            .unwrap()
            .any(|(name, value)| { name == "value" && value == "#complex.number<:f32 1.0, 2.0>" })
    );
}

#[test]
fn complex_preset_inventory_and_recovery_match_llvm_22_1() {
    let registry = DialectRegistry::from_name("complex").unwrap();
    let unary = [
        "complex.cos",
        "complex.exp",
        "complex.expm1",
        "complex.log",
        "complex.log1p",
        "complex.neg",
        "complex.rsqrt",
        "complex.sign",
        "complex.sin",
        "complex.sqrt",
        "complex.tanh",
        "complex.tan",
        "complex.conj",
        "complex.bitcast",
    ];
    let binary = [
        "complex.add",
        "complex.atan2",
        "complex.div",
        "complex.mul",
        "complex.pow",
        "complex.sub",
    ];
    for name in unary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::UnaryOperand),
            "{name}"
        );
    }
    for name in binary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::BinaryOperands),
            "{name}"
        );
    }
    assert_eq!(
        registry.operation_shape("complex.constant"),
        Some(OperationShape::LiteralAttribute)
    );

    let unsupported = [
        "complex.abs",
        "complex.create",
        "complex.eq",
        "complex.im",
        "complex.neq",
        "complex.powi",
        "complex.re",
        "complex.angle",
    ];
    assert_eq!(unary.len() + binary.len() + 1, 21);
    assert_eq!(unary.len() + binary.len() + 1 + unsupported.len(), 29);
    for name in unsupported {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }

    let source = br#"module {
      func.func @gaps(%part: f32, %value: complex<f32>, %power: i32) {
        %created = complex.create %part, %part : complex<f32>
        %absolute = complex.abs %value : complex<f32>
        %imaginary = complex.im %value : complex<f32>
        %real = complex.re %value : complex<f32>
        %angle = complex.angle %value : complex<f32>
        %equal = complex.eq %value, %value : complex<f32>
        %unequal = complex.neq %value, %value : complex<f32>
        %raised = complex.powi %value, %power : complex<f32>, i32
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        8
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );

    let positional = br#"module {
      func.func @fast(%value: complex<f32>) {
        %sum = complex.add %value, %value fastmath<fast> : complex<f32>
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(positional.as_slice(), &registry).unwrap();
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
fn emitc_preset_exposes_spelled_values_types_symbols_and_function_regions() {
    let registry = DialectRegistry::from_name("emitc").unwrap();
    let source = br#"module {
      emitc.func @calculate(%lhs: i32, %rhs: i32, %array: !emitc.array<4xf32>, %index: index) -> i64 attributes {tag = "function"} {
        %sum = emitc.add %lhs, %rhs {tag = "binary"} : (i32, i32) -> i32
        %wide = emitc.cast %sum {tag = "cast"} : i32 to i64
        %literal = emitc.literal "M_PI" {tag = "literal"} : f32
        %element = emitc.subscript %array[%index] {tag = "subscript"} : (!emitc.array<4xf32>, index) -> !emitc.lvalue<f32>
        %called = emitc.call @callee(%wide) {tag = "call"} : (i64) -> i64
        %constant = "emitc.constant"() {value = #emitc.opaque<"VALUE">} : () -> !emitc.opaque<"T">
        emitc.return %called : i64
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

    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("emitc.func"))
        .unwrap();
    assert_eq!(
        document.operation_symbol_name(function).as_deref(),
        Some("calculate")
    );
    assert_eq!(document.operation_regions(function).unwrap().len(), 1);

    for (name, operand_count, result_type, tag) in [
        ("emitc.add", 2, "i32", "\"binary\""),
        ("emitc.cast", 1, "i64", "\"cast\""),
        ("emitc.call", 1, "i64", "\"call\""),
        ("emitc.subscript", 2, "!emitc.lvalue<f32>", "\"subscript\""),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operand_count,
            "{name}"
        );
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(results[0]),
            Some(result_type),
            "{name}"
        );
        assert!(
            document
                .attributes(operation)
                .unwrap()
                .any(|(attribute, value)| { attribute == "tag" && value == tag })
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }

    let call = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("emitc.call"))
        .unwrap();
    assert_eq!(document.operation_callee(call).as_deref(), Some("callee"));

    let literal = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("emitc.literal"))
        .unwrap();
    assert!(
        document
            .attributes(literal)
            .unwrap()
            .any(|(name, value)| { name == "value" && value == "\"M_PI\"" })
    );
    let results = document.result_types(literal).unwrap();
    assert_eq!(document.type_spelling(results[0]), Some("f32"));

    let return_op = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("emitc.return"))
        .unwrap();
    assert_eq!(document.operands(return_op).unwrap().len(), 1);
    assert!(document.result_types(return_op).unwrap().is_empty());
}

#[test]
fn emitc_preset_inventory_and_recovery_match_llvm_22_1() {
    assert!(DialectRegistry::preset_names().contains(&"emitc"));
    let registry = DialectRegistry::from_name("emitc").unwrap();
    let binary = [
        "emitc.add",
        "emitc.bitwise_and",
        "emitc.bitwise_left_shift",
        "emitc.bitwise_or",
        "emitc.bitwise_right_shift",
        "emitc.bitwise_xor",
        "emitc.div",
        "emitc.mul",
        "emitc.rem",
        "emitc.sub",
    ];
    let unary = [
        "emitc.bitwise_not",
        "emitc.cast",
        "emitc.unary_minus",
        "emitc.unary_plus",
    ];
    for name in binary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::BinaryOperands),
            "{name}"
        );
    }
    for name in unary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::UnaryOperand),
            "{name}"
        );
    }
    for (name, shape) in [
        ("emitc.call", OperationShape::CallLike),
        ("emitc.func", OperationShape::FuncLike),
        ("emitc.return", OperationShape::OptionalTypedOperands),
        ("emitc.literal", OperationShape::LiteralAttribute),
        ("emitc.yield", OperationShape::OptionalTypedOperands),
        ("emitc.subscript", OperationShape::OperandClauses),
    ] {
        assert_eq!(registry.operation_shape(name), Some(shape), "{name}");
    }

    let unsupported_custom = [
        "emitc.file",
        "emitc.address_of",
        "emitc.apply",
        "emitc.call_opaque",
        "emitc.cmp",
        "emitc.dereference",
        "emitc.expression",
        "emitc.for",
        "emitc.declare_func",
        "emitc.include",
        "emitc.logical_and",
        "emitc.logical_not",
        "emitc.logical_or",
        "emitc.load",
        "emitc.conditional",
        "emitc.global",
        "emitc.get_global",
        "emitc.verbatim",
        "emitc.assign",
        "emitc.if",
        "emitc.switch",
        "emitc.class",
        "emitc.field",
        "emitc.get_field",
        "emitc.do",
    ];
    let generic_only = [
        "emitc.constant",
        "emitc.variable",
        "emitc.member",
        "emitc.member_of_ptr",
    ];
    assert_eq!(binary.len() + unary.len() + 6, 20);
    assert_eq!(
        binary.len() + unary.len() + 6 + unsupported_custom.len(),
        45
    );
    assert_eq!(45 + generic_only.len(), 49);
    for name in unsupported_custom.into_iter().chain(generic_only) {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }

    let source = br#"module {
      func.func @gaps(%reference: !emitc.lvalue<i32>, %lhs: i32, %rhs: i32) {
        %address = emitc.address_of %reference : !emitc.lvalue<i32>
        %logical = emitc.logical_and %lhs, %rhs : i32, i32
        %call = emitc.call_opaque "callee"(%lhs) : (i32) -> i32
        emitc.if %logical {
          emitc.yield
        }
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| { diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation })
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| { document.operation_name(operation) == Some("test.after") })
    );
}

#[test]
fn func_preset_preserves_exact_semantics_and_recovers_the_two_remaining_ops() {
    assert!(DialectRegistry::preset_names().contains(&"func"));
    let registry = DialectRegistry::from_name("func").unwrap();
    assert_eq!(
        registry.operation_names().collect::<Vec<_>>(),
        ["builtin.module", "func.func", "func.return", "func.call"]
    );

    let supported = ["func.func", "func.call", "func.return"];
    let unsupported = ["func.constant", "func.call_indirect"];
    assert_eq!(supported.len() + unsupported.len(), 5);
    for name in supported {
        assert!(registry.operation(name).is_some(), "{name}");
    }
    for name in unsupported {
        assert!(registry.operation(name).is_none(), "{name}");
        assert_eq!(registry.operation_shape(name), None, "{name}");
    }

    let exact = br#"module {
      func.func private @identity(%arg: i32 {test.argument = true}) -> (i32 {test.result = true})
      func.func @caller(%arg: i32) -> i32 attributes {test.kind = "caller"} {
        %result = func.call @identity(%arg) {test.kind = "call"} : (i32) -> i32
        func.return %result : i32
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(exact.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    document.verify_semantics(&registry).unwrap();

    let identity = document
        .operations()
        .find(|operation| document.operation_symbol_name(*operation).as_deref() == Some("identity"))
        .unwrap();
    assert_eq!(
        document.operation_signature(identity).as_deref(),
        Some("(i32) -> i32")
    );
    assert!(document.attribute_id(identity, "arg_attrs").is_some());
    assert!(document.attribute_id(identity, "res_attrs").is_some());

    let call = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.call"))
        .unwrap();
    assert_eq!(document.operation_callee(call).as_deref(), Some("identity"));
    assert_eq!(document.operands(call).unwrap().len(), 1);
    assert_eq!(document.result_types(call).unwrap().len(), 1);
    assert_eq!(
        document.type_spelling(document.result_types(call).unwrap()[0]),
        Some("i32")
    );

    let return_op = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.return"))
        .unwrap();
    assert_eq!(document.operands(return_op).unwrap().len(), 1);
    assert!(document.result_types(return_op).unwrap().is_empty());

    let gaps = br#"module {
      func.func @gaps(%arg: i32) {
        %callee = func.constant {test.kind = "constant"} @identity : (i32) -> i32
        %result = func.call_indirect %callee(%arg) {test.kind = "indirect"} : (i32) -> i32
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(gaps.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        2
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    for name in [
        "func.constant",
        "func.call_indirect",
        "test.after",
        "func.return",
    ] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn gpu_preset_exposes_only_explicit_operand_and_result_types() {
    assert!(DialectRegistry::preset_names().contains(&"gpu"));
    let registry = DialectRegistry::from_name("gpu").unwrap();
    let source = br#"module {
      func.func @host(%buffer: memref<*xf32>, %value: i32) {
        gpu.barrier {tag = "barrier"}
        %subgroup = gpu.subgroup_id {tag = "id"} : index
        %count = gpu.num_subgroups : index
        %size = gpu.subgroup_size : index
        %shared = gpu.dynamic_shared_memory {tag = "shared"} : memref<?xi8, #gpu.address_space<workgroup>>
        gpu.host_register %buffer {tag = "register"} : memref<*xf32>
        gpu.host_unregister %buffer : memref<*xf32>
        gpu.yield %value : i32
        gpu.return
      }
      func.func @terminator() {
        gpu.terminator
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

    for (name, expected_type) in [
        ("gpu.subgroup_id", "index"),
        ("gpu.num_subgroups", "index"),
        ("gpu.subgroup_size", "index"),
        (
            "gpu.dynamic_shared_memory",
            "memref<?xi8, #gpu.address_space<workgroup>>",
        ),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert!(document.operands(operation).unwrap().is_empty(), "{name}");
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(results[0]),
            Some(expected_type),
            "{name}"
        );
    }

    for name in ["gpu.host_register", "gpu.host_unregister"] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.operands(operation).unwrap().len(), 1, "{name}");
        assert!(
            document.result_types(operation).unwrap().is_empty(),
            "{name}"
        );
    }
    for name in ["gpu.return", "gpu.terminator", "gpu.yield", "gpu.barrier"] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert!(
            document.result_types(operation).unwrap().is_empty(),
            "{name}"
        );
    }

    let yield_op = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("gpu.yield"))
        .unwrap();
    assert_eq!(document.operands(yield_op).unwrap().len(), 1);
    let shared = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("gpu.dynamic_shared_memory"))
        .unwrap();
    assert!(document.attribute_id(shared, "tag").is_some());
}

#[test]
fn gpu_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("gpu").unwrap();
    let supported = [
        ("gpu.subgroup_id", OperationShape::VariadicOperands),
        ("gpu.num_subgroups", OperationShape::VariadicOperands),
        ("gpu.subgroup_size", OperationShape::VariadicOperands),
        (
            "gpu.dynamic_shared_memory",
            OperationShape::VariadicOperands,
        ),
        ("gpu.return", OperationShape::OptionalTypedOperands),
        ("gpu.terminator", OperationShape::OptionalTypedOperands),
        ("gpu.yield", OperationShape::OptionalTypedOperands),
        ("gpu.barrier", OperationShape::OptionalTypedOperands),
        ("gpu.host_register", OperationShape::OptionalTypedOperands),
        ("gpu.host_unregister", OperationShape::OptionalTypedOperands),
    ];
    for (name, shape) in supported {
        assert_eq!(registry.operation_shape(name), Some(shape), "{name}");
    }

    let unsupported = [
        "gpu.cluster_dim",
        "gpu.cluster_dim_blocks",
        "gpu.cluster_id",
        "gpu.cluster_block_id",
        "gpu.block_dim",
        "gpu.block_id",
        "gpu.grid_dim",
        "gpu.thread_id",
        "gpu.lane_id",
        "gpu.global_id",
        "gpu.func",
        "gpu.launch_func",
        "gpu.launch",
        "gpu.printf",
        "gpu.all_reduce",
        "gpu.subgroup_reduce",
        "gpu.shuffle",
        "gpu.rotate",
        "gpu.module",
        "gpu.binary",
        "gpu.wait",
        "gpu.alloc",
        "gpu.dealloc",
        "gpu.memcpy",
        "gpu.memset",
        "gpu.set_default_device",
        "gpu.subgroup_mma_load_matrix",
        "gpu.subgroup_mma_store_matrix",
        "gpu.subgroup_mma_compute",
        "gpu.subgroup_mma_constant_matrix",
        "gpu.subgroup_mma_extract_thread_local",
        "gpu.subgroup_mma_insert_thread_local",
        "gpu.subgroup_mma_elementwise",
        "gpu.create_dn_tensor",
        "gpu.destroy_dn_tensor",
        "gpu.create_coo",
        "gpu.create_coo_aos",
        "gpu.create_csr",
        "gpu.create_csc",
        "gpu.create_bsr",
        "gpu.create_2to4_spmat",
        "gpu.destroy_sp_mat",
        "gpu.spmv_buffer_size",
        "gpu.spmv",
        "gpu.spmm_buffer_size",
        "gpu.spmm",
        "gpu.sddmm_buffer_size",
        "gpu.sddmm",
        "gpu.spgemm_create_descr",
        "gpu.spgemm_destroy_descr",
        "gpu.spgemm_work_estimation_or_compute",
        "gpu.spgemm_copy",
        "gpu.spmat_get_size",
        "gpu.set_csr_pointers",
        "gpu.warp_execute_on_lane_0",
        "gpu.subgroup_broadcast",
    ];
    assert_eq!(supported.len() + unsupported.len(), 66);
    for name in unsupported {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn gpu_optional_upper_bound_variant_recovers_without_swallowing_the_next_operation() {
    let registry = DialectRegistry::from_name("gpu").unwrap();
    let source = br#"module {
      func.func @query() {
        %id = gpu.subgroup_id upper_bound 32 : index
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
        diagnostic.kind() == ParseDiagnosticKind::ShapeMismatch(OperationShape::VariadicOperands)
    }));
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("func.return"))
    );
}

#[test]
fn index_preset_exposes_exact_cast_source_and_destination_types() {
    assert!(DialectRegistry::preset_names().contains(&"index"));
    let registry = DialectRegistry::from_name("index").unwrap();
    let source = br#"module {
      func.func @casts(%idx: index) -> index {
        %integer = index.casts %idx {tag = "signed"} : index to i32
        %result = index.castu %integer {tag = "unsigned"} : i32 to index
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
    document.verify_semantics(&registry).unwrap();

    for (name, signature, result_type, tag) in [
        ("index.casts", "(index) -> i32", "i32", "\"signed\""),
        ("index.castu", "(i32) -> index", "index", "\"unsigned\""),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.operands(operation).unwrap().len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
        let results = document.result_types(operation).unwrap();
        assert_eq!(results.len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(results[0]),
            Some(result_type),
            "{name}"
        );
        assert!(
            document
                .attributes(operation)
                .unwrap()
                .any(|(attribute, value)| attribute == "tag" && value == tag),
            "{name}"
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }
}

#[test]
fn index_preset_inventory_and_inferred_forms_match_llvm_22_1() {
    let registry = DialectRegistry::from_name("index").unwrap();
    for name in ["index.casts", "index.castu"] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::UnaryOperand),
            "{name}"
        );
    }

    let unsupported = [
        "index.add",
        "index.sub",
        "index.mul",
        "index.divs",
        "index.divu",
        "index.ceildivs",
        "index.ceildivu",
        "index.floordivs",
        "index.rems",
        "index.remu",
        "index.maxs",
        "index.maxu",
        "index.mins",
        "index.minu",
        "index.shl",
        "index.shrs",
        "index.shru",
        "index.and",
        "index.or",
        "index.xor",
        "index.cmp",
        "index.sizeof",
        "index.constant",
        "index.bool.constant",
    ];
    assert_eq!(2 + unsupported.len(), 26);
    for name in unsupported {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }

    let source = br#"module {
      func.func @gaps(%lhs: index, %rhs: index) {
        %sum = index.add %lhs, %rhs {tag = "binary"}
        %comparison = index.cmp eq(%lhs, %rhs) {tag = "predicate"}
        %width = index.sizeof {tag = "sizeof"}
        %number = index.constant {tag = "integer"} 42
        %boolean = index.bool.constant {tag = "boolean"} true
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        5
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("func.return"))
    );
}

#[test]
fn llvm_preset_exposes_core_and_intrinsic_function_types() {
    assert!(DialectRegistry::preset_names().contains(&"llvm"));
    let registry = DialectRegistry::from_name("llvm").unwrap();
    let source = br#"module {
      func.func @families(%lhs: i32, %rhs: i32, %ptr: !llvm.ptr, %vector: vector<4xf32>) {
        %sum = llvm.add %lhs, %rhs : i32
        %wide = llvm.zext %sum : i32 to i64
        %slot = llvm.alloca %wide x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr
        %selected = llvm.select %lhs, %sum, %rhs : i32, i32
        %none = llvm.mlir.none : !llvm.token
        %sine = llvm.intr.sin(%vector) : (vector<4xf32>) -> vector<4xf32>
        llvm.intr.lifetime.start %ptr : !llvm.ptr
        llvm.return
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

    for (name, operands, results, signature) in [
        ("llvm.add", 2, 1, "(i32, i32) -> i32"),
        ("llvm.zext", 1, 1, "(i32) -> i64"),
        ("llvm.alloca", 1, 1, "(i64) -> !llvm.ptr"),
        ("llvm.select", 3, 1, "(i32, i32, i32) -> i32"),
        ("llvm.mlir.none", 0, 1, "() -> !llvm.token"),
        ("llvm.intr.sin", 1, 1, "(vector<4xf32>) -> vector<4xf32>"),
        ("llvm.intr.lifetime.start", 1, 0, "(!llvm.ptr) -> ()"),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
    }
}

#[test]
fn llvm_preset_inventory_matches_llvm_22_1_assembly_families() {
    let registry = DialectRegistry::from_name("llvm").unwrap();
    // LLVMOps.td contains 80 concrete operations and LLVMIntrinsicOps.td 204.
    // The preset covers 43 core and 95 intrinsic custom forms plus four core
    // container operations inherited by every bundled dialect preset.
    assert_eq!(registry.operation_names().count(), 4);
    assert_eq!(
        registry.operation_shape("llvm.add"),
        Some(OperationShape::BinaryOperands)
    );
    assert_eq!(
        registry.operation_shape("llvm.intr.sincos"),
        Some(OperationShape::OperandClauses)
    );

    for name in [
        "llvm.getelementptr",
        "llvm.load",
        "llvm.store",
        "llvm.call",
        "llvm.invoke",
        "llvm.switch",
        "llvm.mlir.global",
        "llvm.func",
        "llvm.atomicrmw",
        "llvm.cmpxchg",
        "llvm.call_intrinsic",
        "llvm.intr.dbg.value",
        "llvm.intr.vp.add",
        "llvm.intr.vector.insert",
        "llvm.intr.matrix.transpose",
        "llvm.intr.get.active.lane.mask",
    ] {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn llvm_unsupported_indexed_and_atomic_forms_recover_at_operation_boundaries() {
    let registry = DialectRegistry::from_name("llvm").unwrap();
    let source = br#"module {
      func.func @gaps(%ptr: !llvm.ptr, %index: i64) {
        %element = llvm.getelementptr %ptr[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %value = llvm.load %ptr atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        2
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn math_preset_exposes_unary_binary_and_ternary_structure() {
    assert!(DialectRegistry::preset_names().contains(&"math"));
    let registry = DialectRegistry::from_name("math").unwrap();
    let source = br#"module {
      func.func @families(%float: f32, %integer: i32) -> f32 {
        %root = math.sqrt %float {tag = "unary"} : f32
        %power = math.ipowi %integer, %integer : i32
        %raised = math.powf %root, %float {tag = "binary"} : f32
        %clamped = math.clampf %raised to [%float, %root] {tag = "clamp"} : f32
        %fused = math.fma %clamped, %float, %root {tag = "ternary"} : f32
        func.return %fused : f32
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

    for (name, operands, signature) in [
        ("math.sqrt", 1, "(f32) -> f32"),
        ("math.ipowi", 2, "(i32, i32) -> i32"),
        ("math.powf", 2, "(f32, f32) -> f32"),
        ("math.clampf", 3, "(f32, f32, f32) -> f32"),
        ("math.fma", 3, "(f32, f32, f32) -> f32"),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(document.result_types(operation).unwrap().len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
        assert!(document.operation_regions(operation).unwrap().is_empty());
        assert!(document.successors(operation).unwrap().is_empty());
    }

    for (name, tag) in [
        ("math.sqrt", "\"unary\""),
        ("math.powf", "\"binary\""),
        ("math.clampf", "\"clamp\""),
        ("math.fma", "\"ternary\""),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert!(
            document
                .attributes(operation)
                .unwrap()
                .any(|(attribute, value)| attribute == "tag" && value == tag),
            "{name}"
        );
    }
}

#[test]
fn math_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("math").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/math.json")).unwrap();
    assert_eq!(config.operation_shapes.len(), 40);
    let unary = [
        "math.absf",
        "math.absi",
        "math.acosh",
        "math.asin",
        "math.asinh",
        "math.atan",
        "math.atanh",
        "math.cbrt",
        "math.ceil",
        "math.cos",
        "math.acos",
        "math.cosh",
        "math.sin",
        "math.sinh",
        "math.ctlz",
        "math.cttz",
        "math.ctpop",
        "math.erf",
        "math.erfc",
        "math.exp",
        "math.exp2",
        "math.expm1",
        "math.floor",
        "math.log",
        "math.log10",
        "math.log1p",
        "math.log2",
        "math.rsqrt",
        "math.sqrt",
        "math.tan",
        "math.tanh",
        "math.roundeven",
        "math.round",
        "math.trunc",
    ];
    let binary = ["math.atan2", "math.copysign", "math.ipowi", "math.powf"];
    let ternary = ["math.clampf", "math.fma"];
    let recovery = [
        "math.isfinite",
        "math.isinf",
        "math.isnan",
        "math.isnormal",
        "math.sincos",
        "math.fpowi",
    ];
    assert_eq!(
        unary.len() + binary.len() + ternary.len() + recovery.len(),
        46
    );
    for name in unary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::UnaryOperand),
            "{name}"
        );
    }
    for name in binary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::BinaryOperands),
            "{name}"
        );
    }
    for name in ternary {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::OperandClauses),
            "{name}"
        );
    }
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn math_inferred_and_mixed_type_forms_and_positional_modifiers_recover() {
    let registry = DialectRegistry::from_name("math").unwrap();
    let source = br#"module {
      func.func @gaps(%scalar: f32, %tensor: tensor<2xf32>, %power: i32) {
        %finite = math.isfinite %tensor : tensor<2xf32>
        %sin, %cos = math.sincos %scalar : f32
        %raised = math.fpowi %scalar, %power : f32, i32
        %root = math.sqrt %scalar fastmath<fast> : f32
        %sum = math.powf %scalar, %scalar fastmath<contract> : f32
        %clamped = math.clampf %scalar to [%scalar, %scalar] fastmath<fast> : f32
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        3
    );
    assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
        diagnostic.kind() == ParseDiagnosticKind::ShapeMismatch(OperationShape::UnaryOperand)
    }));
    assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
        diagnostic.kind() == ParseDiagnosticKind::ShapeMismatch(OperationShape::BinaryOperands)
    }));
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    let clamped = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("math.clampf"))
        .unwrap();
    assert_eq!(document.operands(clamped).unwrap().len(), 3);
    assert_eq!(document.result_types(clamped).unwrap().len(), 1);
    assert_eq!(
        document.type_spelling(document.function_type(clamped).unwrap()),
        Some("(f32, f32, f32) -> f32")
    );
    assert!(
        document
            .attributes(clamped)
            .unwrap()
            .all(|(name, _)| name != "fastmath")
    );
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn irdl_preset_inventory_and_custom_forms_match_llvm_22_1() {
    assert!(DialectRegistry::preset_names().contains(&"irdl"));
    let registry = DialectRegistry::from_name("irdl").unwrap();
    let unsupported = [
        "irdl.dialect",
        "irdl.type",
        "irdl.attribute",
        "irdl.parameters",
        "irdl.operation",
        "irdl.operands",
        "irdl.results",
        "irdl.attributes",
        "irdl.region",
        "irdl.regions",
        "irdl.is",
        "irdl.base",
        "irdl.parametric",
        "irdl.any",
        "irdl.any_of",
        "irdl.all_of",
        "irdl.c_pred",
    ];
    assert_eq!(unsupported.len(), 17);
    assert_eq!(registry.operation_names().count(), 4);
    for name in unsupported {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }

    let source = br#"module {
      irdl.dialect @example attributes {tag = "definition"} {
        irdl.operation @variadic {
          %constraint = irdl.any
          irdl.operands(input: optional %constraint)
          irdl.results(output: variadic %constraint)
        }
      }
      "test.after"() : () -> ()
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| { diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation })
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("irdl.dialect"))
    );
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );
}

#[test]
fn irdl_preset_preserves_handle_types_and_variadicity_attributes_as_opaque_values() {
    let registry = DialectRegistry::from_name("irdl").unwrap();
    let source = br#"%attribute, %region = "test.irdl_values"() {
      variadicity = #irdl<variadicity_array[single, optional, variadic]>
    } : () -> (!irdl.attribute, !irdl.region)"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let operation = document.operations().next().unwrap();
    let types = document.result_types(operation).unwrap();
    assert_eq!(document.type_spelling(types[0]), Some("!irdl.attribute"));
    assert_eq!(document.type_spelling(types[1]), Some("!irdl.region"));
    assert!(matches!(
        document.type_value(types[0]),
        Some(TypeValue::Opaque(_))
    ));
    assert!(matches!(
        document.type_value(types[1]),
        Some(TypeValue::Opaque(_))
    ));
    let attribute = document.attribute_id(operation, "variadicity").unwrap();
    let spelling = "#irdl<variadicity_array[single, optional, variadic]>";
    assert_eq!(document.attribute_spelling_value(attribute), Some(spelling));
    assert!(matches!(
        document.attribute_value(attribute),
        Some(AttributeValue::Opaque(value)) if value.as_ref() == spelling.as_bytes()
    ));
}

#[test]
fn dlti_preset_preserves_llvm_22_1_attributes_without_claiming_operations() {
    let registry = DialectRegistry::from_name("dlti").unwrap();
    assert_eq!(registry.operation_names().count(), 4);
    assert!(
        registry
            .operation_names()
            .all(|name| !name.starts_with("dlti."))
    );
    assert!(registry.operation("transform.dlti.query").is_none());

    let source = br#"module {
      "test.dlti"() {
        entry = #dlti.dl_entry<"test.identifier", 42 : i64>,
        spec = #dlti.dl_spec<"test.id" = 42 : i32>,
        map = #dlti.map<"bitwidth" = 32 : i32>,
        system = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"bits" = 64 : i32>>,
        device = #dlti.target_device_spec<"bits" = 64 : i32>,
        alignment = #dlti.function_pointer_alignment<64, function_dependent = false>
      } : () -> ()
    }"#;
    let expected = [
        ("entry", r#"#dlti.dl_entry<"test.identifier", 42 : i64>"#),
        ("spec", r#"#dlti.dl_spec<"test.id" = 42 : i32>"#),
        ("map", r#"#dlti.map<"bitwidth" = 32 : i32>"#),
        (
            "system",
            r#"#dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"bits" = 64 : i32>>"#,
        ),
        ("device", r#"#dlti.target_device_spec<"bits" = 64 : i32>"#),
        (
            "alignment",
            "#dlti.function_pointer_alignment<64, function_dependent = false>",
        ),
    ];

    for mode in [LoweringMode::Strict, LoweringMode::BestEffort] {
        let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
        assert!(
            parsed.syntax().diagnostics().is_empty(),
            "{:?}",
            parsed.syntax().diagnostics()
        );
        let lowered = lower_with_dialect_registry(&parsed, mode, &registry);
        assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
        let document = lowered.document.unwrap();
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some("test.dlti"))
            .unwrap();
        for (name, spelling) in expected {
            let attribute = document.attribute_id(operation, name).unwrap();
            assert_eq!(document.attribute_spelling_value(attribute), Some(spelling));
            assert!(matches!(
                document.attribute_value(attribute),
                Some(AttributeValue::Opaque(value)) if value.as_ref() == spelling.as_bytes()
            ));
        }
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

#[test]
fn memref_preset_exposes_exact_conversion_reshape_and_region_structure() {
    assert!(DialectRegistry::preset_names().contains(&"memref"));
    let registry = DialectRegistry::from_name("memref").unwrap();
    let source = br#"module {
      func.func @memref_forms(
          %source: memref<?x?xf32>, %static_source: memref<4x4xf32>,
          %shape: memref<1xi32>, %lhs: memref<?xf32>, %rhs: memref<?xf32>,
          %index: index) {
        %aligned = memref.assume_alignment %source, 16 : memref<?x?xf32>
        %distinct_lhs, %distinct_rhs = memref.distinct_objects %lhs, %rhs : memref<?xf32>, memref<?xf32>
        %cast = memref.cast %static_source : memref<4x4xf32> to memref<?x?xf32>
        %space = memref.memory_space_cast %lhs : memref<?xf32> to memref<?xf32, 1>
        %pointer = memref.extract_aligned_pointer_as_index %lhs : memref<?xf32> -> index
        %reshaped = memref.reshape %source(%shape) : (memref<?x?xf32>, memref<1xi32>) -> memref<*xf32>
        %atomic = memref.generic_atomic_rmw %lhs[%index] : memref<?xf32> {
        ^bb0(%current: f32):
          memref.atomic_yield %current : f32
        }
        %scoped = memref.alloca_scope -> (index) {
          memref.alloca_scope.return %index : index
        }
        memref.dealloc %lhs : memref<?xf32>
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

    for (name, operands, results, signature) in [
        (
            "memref.assume_alignment",
            1,
            1,
            "(memref<?x?xf32>) -> memref<?x?xf32>",
        ),
        (
            "memref.distinct_objects",
            2,
            2,
            "(memref<?xf32>, memref<?xf32>) -> (memref<?xf32>, memref<?xf32>)",
        ),
        ("memref.cast", 1, 1, "(memref<4x4xf32>) -> memref<?x?xf32>"),
        (
            "memref.memory_space_cast",
            1,
            1,
            "(memref<?xf32>) -> memref<?xf32, 1>",
        ),
        (
            "memref.extract_aligned_pointer_as_index",
            1,
            1,
            "(memref<?xf32>) -> index",
        ),
        (
            "memref.reshape",
            2,
            1,
            "(memref<?x?xf32>, memref<1xi32>) -> memref<*xf32>",
        ),
        ("memref.atomic_yield", 1, 0, "(f32) -> ()"),
        ("memref.alloca_scope.return", 1, 0, "(index) -> ()"),
        ("memref.dealloc", 1, 0, "(memref<?xf32>) -> ()"),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
    }

    for (name, operands, results) in [
        ("memref.alloca_scope", 0, 1),
        ("memref.generic_atomic_rmw", 2, 1),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert_eq!(
            document.operation_regions(operation).unwrap().len(),
            1,
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }
}

#[test]
fn memref_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("memref").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/memref.json")).unwrap();
    assert_eq!(config.operation_shapes.len(), 11);
    let recovery = [
        "memref.alloc",
        "memref.realloc",
        "memref.alloca",
        "memref.copy",
        "memref.dim",
        "memref.dma_start",
        "memref.dma_wait",
        "memref.extract_strided_metadata",
        "memref.get_global",
        "memref.global",
        "memref.load",
        "memref.prefetch",
        "memref.reinterpret_cast",
        "memref.rank",
        "memref.expand_shape",
        "memref.collapse_shape",
        "memref.store",
        "memref.subview",
        "memref.transpose",
        "memref.view",
        "memref.atomic_rmw",
    ];
    assert_eq!(config.operation_shapes.len() + recovery.len(), 32);
    for operation in &config.operation_shapes {
        assert_eq!(
            registry.operation_shape(&operation.name),
            Some(operation.shape)
        );
    }
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn memref_inferred_index_and_mixed_index_list_forms_recover() {
    let registry = DialectRegistry::from_name("memref").unwrap();
    let source = br#"module {
      func.func @gaps(%source: memref<?xf32>, %index: index, %value: f32) {
        %dimension = memref.dim %source, %index : memref<?xf32>
        %loaded = memref.load %source[%index] : memref<?xf32>
        memref.store %value, %source[%index] : memref<?xf32>
        %slice = memref.subview %source[%index][4][1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
        memref.copy %source, %source : memref<?xf32> to memref<?xf32>
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        5
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    assert!(
        document
            .operations()
            .any(|operation| { document.operation_name(operation) == Some("test.after") })
    );
}

#[test]
fn ml_program_preset_inventory_matches_llvm_22_1_recovery_coverage() {
    assert!(DialectRegistry::preset_names().contains(&"ml_program"));
    let registry = DialectRegistry::from_name("ml_program").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/ml_program.json")).unwrap();
    assert!(config.operation_shapes.is_empty());
    assert_eq!(registry.operation_names().count(), 4);

    let recovery = [
        "ml_program.func",
        "ml_program.global",
        "ml_program.global_load",
        "ml_program.global_load_const",
        "ml_program.global_load_graph",
        "ml_program.global_store",
        "ml_program.global_store_graph",
        "ml_program.output",
        "ml_program.return",
        "ml_program.subgraph",
        "ml_program.token",
    ];
    assert_eq!(recovery.len(), 11);
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn ml_program_symbol_inference_region_and_dictionary_forms_recover() {
    let registry = DialectRegistry::from_name("ml_program").unwrap();
    let source = br#"module {
      ml_program.func private @external(i32) -> i32
      ml_program.subgraph private @external_graph(i32) -> i32
      ml_program.global private mutable @state : tensor<?xi32>
      func.func @gaps(%value: tensor<?xi32>, %token: !ml_program.token) {
        %loaded = ml_program.global_load @state : tensor<?xi32>
        %constant = ml_program.global_load_const @state : tensor<?xi32>
        ml_program.global_store @state = %value : tensor<?xi32>
        %graph_loaded, %ordered = ml_program.global_load_graph @state ordering(%token -> !ml_program.token) : tensor<?xi32>
        %stored = ml_program.global_store_graph @state = %value ordering(%token -> !ml_program.token) : tensor<?xi32>
        %fresh = ml_program.token
        ml_program.return {tag = "before-operands"} %value : tensor<?xi32>
        ml_program.output {tag = "before-operands"} %value : tensor<?xi32>
      }
      "test.after"() : () -> ()
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        11,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    assert!(
        document
            .operations()
            .any(|operation| { document.operation_name(operation) == Some("test.after") })
    );
}

#[test]
fn ml_program_opaque_type_and_attribute_need_no_dialect_descriptors() {
    let registry = DialectRegistry::from_name("ml_program").unwrap();
    let source = br#"module {
      %token = "test.source"() {value = #ml_program.extern<tensor<4xi32>>}
        : () -> !ml_program.token
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.source"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.result_types(operation).unwrap()[0]),
        Some("!ml_program.token")
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| {
                name == "value" && value == "#ml_program.extern<tensor<4xi32>>"
            })
    );
}

#[test]
fn mpi_preset_exposes_only_fully_typed_result_and_same_type_forms() {
    assert!(DialectRegistry::preset_names().contains(&"mpi"));
    let registry = DialectRegistry::from_name("mpi").unwrap();
    let source = br#"module {
      func.func @mpi() {
        %init = mpi.init {tag = "init"} : !mpi.retval
        %comm = mpi.comm_world {tag = "world"} : !mpi.comm
        %final = mpi.finalize {tag = "final"} : !mpi.retval
        %class = mpi.error_class %init {tag = "class"} : !mpi.retval
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
    for (name, operands, signature) in [
        ("mpi.init", 0, "() -> !mpi.retval"),
        ("mpi.comm_world", 0, "() -> !mpi.comm"),
        ("mpi.finalize", 0, "() -> !mpi.retval"),
        ("mpi.error_class", 1, "(!mpi.retval) -> !mpi.retval"),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(document.result_types(operation).unwrap().len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
        assert!(document.operation_regions(operation).unwrap().is_empty());
        assert!(document.successors(operation).unwrap().is_empty());
        assert!(
            document
                .attributes(operation)
                .unwrap()
                .any(|(attribute, _)| attribute == "tag"),
            "{name}"
        );
    }
}

#[test]
fn mpi_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("mpi").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/mpi.json")).unwrap();
    assert_eq!(config.operation_shapes.len(), 4);
    for name in ["mpi.init", "mpi.comm_world", "mpi.finalize"] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::VariadicOperands),
            "{name}"
        );
    }
    assert_eq!(
        registry.operation_shape("mpi.error_class"),
        Some(OperationShape::UnaryOperand)
    );

    let recovery = [
        "mpi.comm_rank",
        "mpi.comm_size",
        "mpi.comm_split",
        "mpi.send",
        "mpi.isend",
        "mpi.recv",
        "mpi.irecv",
        "mpi.allreduce",
        "mpi.barrier",
        "mpi.wait",
        "mpi.retval_check",
    ];
    assert_eq!(config.operation_shapes.len() + recovery.len(), 15);
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn mpi_inferred_mixed_and_positional_forms_recover_to_the_next_operation() {
    let registry = DialectRegistry::from_name("mpi").unwrap();
    let source = br#"module {
      func.func @gaps(%ref: memref<100xf32>, %i: i32, %comm: !mpi.comm, %req: !mpi.request, %retval: !mpi.retval) {
        mpi.init
        mpi.finalize
        %rank = mpi.comm_rank(%comm) : i32
        %size = mpi.comm_size(%comm) : i32
        %split = mpi.comm_split(%comm, %i, %i) : !mpi.comm
        mpi.send(%ref, %i, %i, %comm) : memref<100xf32>, i32, i32
        %sent = mpi.isend(%ref, %i, %i, %comm) : memref<100xf32>, i32, i32 -> !mpi.request
        mpi.recv(%ref, %i, %i, %comm) : memref<100xf32>, i32, i32
        %received = mpi.irecv(%ref, %i, %i, %comm) : memref<100xf32>, i32, i32 -> !mpi.request
        mpi.allreduce(%ref, %ref, MPI_SUM, %comm) : memref<100xf32>, memref<100xf32>
        mpi.barrier(%comm)
        mpi.wait(%req) : !mpi.request
        %ok = mpi.retval_check %retval = <MPI_SUCCESS> : i1
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        11,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| {
                diagnostic.kind()
                    == ParseDiagnosticKind::ShapeMismatch(OperationShape::VariadicOperands)
            })
            .count(),
        2,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn mpi_opaque_types_and_attributes_need_no_dialect_descriptors() {
    let registry = DialectRegistry::from_name("mpi").unwrap();
    let source = br#"module {
      %status = "test.source"() {error = #mpi.errclass<MPI_ERR_COMM>}
        : () -> !mpi.status
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.source"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.result_types(operation).unwrap()[0]),
        Some("!mpi.status")
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| name == "error" && value == "#mpi.errclass<MPI_ERR_COMM>")
    );
}

#[test]
fn shard_preset_exposes_the_explicit_get_sharding_signature() {
    assert!(DialectRegistry::preset_names().contains(&"shard"));
    let registry = DialectRegistry::from_name("shard").unwrap();
    let source = br#"module {
      func.func @get(%input: tensor<4x8xf32>) -> !shard.sharding {
        %sharding = shard.get_sharding %input {tag = "queryable"}
          : tensor<4x8xf32> -> !shard.sharding
        func.return %sharding : !shard.sharding
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("shard.get_sharding"))
        .unwrap();
    assert_eq!(document.operands(operation).unwrap().len(), 1);
    assert_eq!(document.result_types(operation).unwrap().len(), 1);
    assert_eq!(
        document.type_spelling(document.function_type(operation).unwrap()),
        Some("(tensor<4x8xf32>) -> !shard.sharding")
    );
    assert!(document.operation_regions(operation).unwrap().is_empty());
    assert!(document.successors(operation).unwrap().is_empty());
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| { name == "tag" && value == "\"queryable\"" })
    );
}

#[test]
fn shard_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("shard").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/shard.json")).unwrap();
    assert_eq!(config.operation_shapes.len(), 1);
    assert_eq!(
        registry.operation_shape("shard.get_sharding"),
        Some(OperationShape::UnaryOperand)
    );
    let recovery = [
        "shard.grid",
        "shard.grid_shape",
        "shard.process_multi_index",
        "shard.process_linear_index",
        "shard.neighbors_linear_indices",
        "shard.sharding",
        "shard.shard_shape",
        "shard.shard",
        "shard.all_gather",
        "shard.all_reduce",
        "shard.all_slice",
        "shard.all_to_all",
        "shard.broadcast",
        "shard.gather",
        "shard.recv",
        "shard.reduce",
        "shard.reduce_scatter",
        "shard.scatter",
        "shard.send",
        "shard.shift",
        "shard.update_halo",
    ];
    assert_eq!(config.operation_shapes.len() + recovery.len(), 22);
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn shard_symbol_index_collective_and_destination_forms_recover() {
    let registry = DialectRegistry::from_name("shard").unwrap();
    let source = br#"module {
      shard.grid @grid(shape = 2x2)
      func.func @gaps(%input: tensor<4x8xf32>, %index: index, %sharding: !shard.sharding) {
        %shape:2 = shard.grid_shape @grid axes = [0, 1] : index, index
        %made = shard.sharding @grid split_axes = [[0]] : !shard.sharding
        %dims:2 = shard.shard_shape dims = [8, %index] sharding = %sharding device = [%index] : index, index
        %annotated = shard.shard %input to %sharding annotate_for_users : tensor<4x8xf32>
        %reduced = shard.all_reduce %input on @grid grid_axes = [0] reduction = max : tensor<4x8xf32> -> tensor<4x8xf64>
        %halo = shard.update_halo %input on @grid split_axes = [[0]] halo_sizes = [1, %index] : tensor<4x8xf32>
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    // Each unsupported Shard operation diagnoses recovery. The unregistered
    // shard.shard conversion also diagnoses its `to` continuation separately.
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        8,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn nvgpu_preset_exposes_only_fully_spelled_operand_and_result_roles() {
    assert!(DialectRegistry::preset_names().contains(&"nvgpu"));
    let registry = DialectRegistry::from_name("nvgpu").unwrap();
    let source = br#"module {
      func.func @families(
          %a: vector<4x2xf16>, %b: vector<2x2xf16>, %c: vector<2x2xf16>,
          %tensor: memref<16x16xf16, 3>,
          %map: !nvgpu.tensormap.descriptor,
          %desc_a: !nvgpu.warpgroup.descriptor,
          %desc_b: !nvgpu.warpgroup.descriptor) {
        %mma = nvgpu.mma.sync(%a, %b, %c) {mmaShape = [16, 8, 16]} :
          (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
        %barriers = nvgpu.mbarrier.create {tag = "barrier"} -> !nvgpu.mbarrier.group
        nvgpu.tma.fence.descriptor %map {tag = "fence"} : !nvgpu.tensormap.descriptor
        nvgpu.tma.prefetch.descriptor %map : !nvgpu.tensormap.descriptor
        %desc = nvgpu.warpgroup.generate.descriptor %tensor, %map :
          memref<16x16xf16, 3>, !nvgpu.tensormap.descriptor -> !nvgpu.warpgroup.descriptor
        %zero = nvgpu.warpgroup.mma.init.accumulator -> !nvgpu.warpgroup.accumulator
        %product = nvgpu.warpgroup.mma %desc_a, %desc_b, %zero {waitGroup = 1 : i64} : !nvgpu.warpgroup.descriptor, !nvgpu.warpgroup.descriptor, !nvgpu.warpgroup.accumulator -> !nvgpu.warpgroup.accumulator
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
    document.verify_semantics(&registry).unwrap();

    for (name, operands, results, signature) in [
        (
            "nvgpu.mma.sync",
            3,
            1,
            "(vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>",
        ),
        ("nvgpu.mbarrier.create", 0, 1, "() -> !nvgpu.mbarrier.group"),
        (
            "nvgpu.tma.fence.descriptor",
            1,
            0,
            "(!nvgpu.tensormap.descriptor) -> ()",
        ),
        (
            "nvgpu.tma.prefetch.descriptor",
            1,
            0,
            "(!nvgpu.tensormap.descriptor) -> ()",
        ),
        (
            "nvgpu.warpgroup.generate.descriptor",
            2,
            1,
            "memref<16x16xf16, 3>, !nvgpu.tensormap.descriptor -> !nvgpu.warpgroup.descriptor",
        ),
        (
            "nvgpu.warpgroup.mma.init.accumulator",
            0,
            1,
            "() -> !nvgpu.warpgroup.accumulator",
        ),
        (
            "nvgpu.warpgroup.mma",
            3,
            1,
            "!nvgpu.warpgroup.descriptor, !nvgpu.warpgroup.descriptor, !nvgpu.warpgroup.accumulator -> !nvgpu.warpgroup.accumulator",
        ),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }
}

#[test]
fn nvgpu_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("nvgpu").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/nvgpu.json")).unwrap();
    let supported = [
        ("nvgpu.mma.sync", OperationShape::OperandClauses),
        ("nvgpu.mbarrier.create", OperationShape::OperandClauses),
        (
            "nvgpu.tma.fence.descriptor",
            OperationShape::OptionalTypedOperands,
        ),
        (
            "nvgpu.tma.prefetch.descriptor",
            OperationShape::OptionalTypedOperands,
        ),
        (
            "nvgpu.warpgroup.generate.descriptor",
            OperationShape::OperandClauses,
        ),
        ("nvgpu.warpgroup.mma", OperationShape::OperandClauses),
        (
            "nvgpu.warpgroup.mma.init.accumulator",
            OperationShape::OperandClauses,
        ),
    ];
    let recovery = [
        "nvgpu.ldmatrix",
        "nvgpu.mma.sp.sync",
        "nvgpu.device_async_copy",
        "nvgpu.device_async_create_group",
        "nvgpu.device_async_wait",
        "nvgpu.mbarrier.get",
        "nvgpu.mbarrier.init",
        "nvgpu.mbarrier.test.wait",
        "nvgpu.mbarrier.arrive",
        "nvgpu.mbarrier.arrive.nocomplete",
        "nvgpu.mbarrier.arrive.expect_tx",
        "nvgpu.mbarrier.try_wait.parity",
        "nvgpu.tma.async.load",
        "nvgpu.tma.async.store",
        "nvgpu.tma.create.descriptor",
        "nvgpu.warpgroup.mma.store",
        "nvgpu.rcp",
    ];
    assert_eq!(supported.len() + recovery.len(), 24);
    assert_eq!(config.operation_shapes.len(), supported.len());
    for (name, shape) in supported {
        assert_eq!(registry.operation_shape(name), Some(shape), "{name}");
    }
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn nvgpu_inferred_index_token_and_destination_forms_recover() {
    let registry = DialectRegistry::from_name("nvgpu").unwrap();
    let source = br#"module {
      func.func @gaps(
          %barriers: !nvgpu.mbarrier.group, %index: index,
          %token: !nvgpu.device.async.token,
          %map: !nvgpu.tensormap.descriptor, %predicate: i1,
          %buffer: memref<16xf32, 3>) {
        %pointer = nvgpu.mbarrier.get %barriers[%index] : !nvgpu.mbarrier.group -> i64
        nvgpu.mbarrier.init %barriers[%index], %index : !nvgpu.mbarrier.group
        %group = nvgpu.device_async_create_group %token
        nvgpu.tma.async.store %buffer to %map[%index] : memref<16xf32, 3> -> !nvgpu.tensormap.descriptor
        nvgpu.tma.prefetch.descriptor %map, predicate = %predicate : !nvgpu.tensormap.descriptor
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        5,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    assert!(parsed.syntax().diagnostics().iter().any(|diagnostic| {
        diagnostic.kind()
            == ParseDiagnosticKind::ShapeMismatch(OperationShape::OptionalTypedOperands)
    }));
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn nvvm_preset_exposes_explicit_register_intrinsic_and_function_type_roles() {
    assert!(DialectRegistry::preset_names().contains(&"nvvm"));
    let registry = DialectRegistry::from_name("nvvm").unwrap();
    let source = br#"module {
      func.func @families(%x: f32, %mask: i32, %barrier: !llvm.ptr<3>, %count: i32) {
        %clock = nvvm.read.ptx.sreg.clock64 {tag = "clock"} : i64
        nvvm.barrier0 {tag = "barrier"}
        %reciprocal = nvvm.rcp.approx.ftz.f %x {tag = "unary"} : f32
        nvvm.bar.warp.sync %mask {tag = "warp"} : i32
        nvvm.mbarrier.inval %barrier {tag = "invalidate"} : !llvm.ptr<3>
        %state = nvvm.mbarrier.arrive.nocomplete %barrier, %count {tag = "arrive"}
          : !llvm.ptr<3>, i32 -> i64
        %matrix = nvvm.ldmatrix %barrier
          {num = 1 : i32, layout = #nvvm.mma_layout<row>}
          : (!llvm.ptr<3>) -> i32
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

    for (name, operands, results, signature) in [
        ("nvvm.read.ptx.sreg.clock64", 0, 1, "() -> i64"),
        ("nvvm.barrier0", 0, 0, "() -> ()"),
        ("nvvm.rcp.approx.ftz.f", 1, 1, "(f32) -> f32"),
        ("nvvm.bar.warp.sync", 1, 0, "(i32) -> ()"),
        ("nvvm.mbarrier.inval", 1, 0, "(!llvm.ptr<3>) -> ()"),
        (
            "nvvm.mbarrier.arrive.nocomplete",
            2,
            1,
            "!llvm.ptr<3>, i32 -> i64",
        ),
        ("nvvm.ldmatrix", 1, 1, "(!llvm.ptr<3>) -> i32"),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }
}

#[test]
fn omp_preset_exposes_plain_regions_control_points_and_threadprivate_types() {
    assert!(DialectRegistry::preset_names().contains(&"omp"));
    let registry = DialectRegistry::from_name("omp").unwrap();
    let source = br#"module {
      func.func @families(%address: !llvm.ptr) {
        %tls = omp.threadprivate %address : !llvm.ptr -> !llvm.ptr
        omp.master {
          omp.barrier {tag = "control"}
        }
        omp.section {
          omp.taskyield
        }
        omp.workshare.loop_wrapper {
          omp.terminator
        }
        omp.workdistribute {
          omp.terminator
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

    let threadprivate = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("omp.threadprivate"))
        .unwrap();
    assert_eq!(document.operands(threadprivate).unwrap().len(), 1);
    assert_eq!(document.result_types(threadprivate).unwrap().len(), 1);
    assert_eq!(
        document.type_spelling(document.function_type(threadprivate).unwrap()),
        Some("(!llvm.ptr) -> !llvm.ptr")
    );
    assert!(
        document
            .operation_regions(threadprivate)
            .unwrap()
            .is_empty()
    );
    assert!(document.successors(threadprivate).unwrap().is_empty());

    for name in [
        "omp.master",
        "omp.section",
        "omp.workshare.loop_wrapper",
        "omp.workdistribute",
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert!(document.operands(operation).unwrap().is_empty(), "{name}");
        assert!(
            document.result_types(operation).unwrap().is_empty(),
            "{name}"
        );
        assert_eq!(
            document.operation_regions(operation).unwrap().len(),
            1,
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }
}

#[test]
fn omp_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("omp").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/omp.json")).unwrap();
    let supported = [
        ("omp.terminator", OperationShape::OptionalTypedOperands),
        ("omp.section", OperationShape::RegionClauses),
        ("omp.workshare.loop_wrapper", OperationShape::RegionClauses),
        ("omp.taskyield", OperationShape::OptionalTypedOperands),
        ("omp.master", OperationShape::RegionClauses),
        ("omp.barrier", OperationShape::OptionalTypedOperands),
        ("omp.threadprivate", OperationShape::UnaryOperand),
        ("omp.workdistribute", OperationShape::RegionClauses),
    ];
    let recovery = [
        "omp.private",
        "omp.parallel",
        "omp.teams",
        "omp.sections",
        "omp.single",
        "omp.new_cli",
        "omp.canonical_loop",
        "omp.unroll_heuristic",
        "omp.tile",
        "omp.workshare",
        "omp.loop_nest",
        "omp.loop",
        "omp.wsloop",
        "omp.simd",
        "omp.yield",
        "omp.distribute",
        "omp.task",
        "omp.taskloop",
        "omp.taskgroup",
        "omp.flush",
        "omp.map.bounds",
        "omp.map.info",
        "omp.target_data",
        "omp.target_enter_data",
        "omp.target_exit_data",
        "omp.target_update",
        "omp.target",
        "omp.critical.declare",
        "omp.critical",
        "omp.ordered",
        "omp.ordered.region",
        "omp.taskwait",
        "omp.atomic.read",
        "omp.atomic.write",
        "omp.atomic.update",
        "omp.atomic.capture",
        "omp.cancel",
        "omp.cancellation_point",
        "omp.scan",
        "omp.declare_mapper",
        "omp.declare_mapper.info",
        "omp.declare_reduction",
        "omp.masked",
        "omp.allocate_dir",
        "omp.target_allocmem",
        "omp.target_freemem",
    ];
    assert_eq!(supported.len(), 8);
    assert_eq!(supported.len() + recovery.len(), 54);
    assert_eq!(config.operation_shapes.len(), supported.len());
    for (name, shape) in supported {
        assert_eq!(registry.operation_shape(name), Some(shape), "{name}");
    }
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn omp_clause_map_and_inferred_result_forms_recover_to_the_next_operation() {
    let registry = DialectRegistry::from_name("omp").unwrap();
    let source = br#"module {
      func.func @gaps(%value: i32, %address: !llvm.ptr) {
        %cli = omp.new_cli
        omp.flush(%value : i32)
        omp.atomic.write %address = %value : !llvm.ptr, i32
        omp.target_freemem %value, %value : i32, i32
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count()
            >= 4,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after"))
    );
}

#[test]
fn omp_opaque_types_and_attributes_need_no_dialect_descriptors() {
    let registry = DialectRegistry::from_name("omp").unwrap();
    let source = br#"module {
      %cli = "test.source"() {kind = #omp<private>} : () -> !omp.cli
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.source"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.result_types(operation).unwrap()[0]),
        Some("!omp.cli")
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| name == "kind" && value == "#omp<private>")
    );
}

#[test]
fn nvvm_preset_inventory_matches_expanded_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("nvvm").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/nvvm.json")).unwrap();
    assert_eq!(config.operation_shapes.len(), 71);

    for name in [
        "nvvm.read.ptx.sreg.clock",
        "nvvm.read.ptx.sreg.envreg0",
        "nvvm.read.ptx.sreg.envreg31",
        "nvvm.read.ptx.sreg.lanemask.eq",
    ] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::VariadicOperands),
            "{name}"
        );
    }
    for name in [
        "nvvm.barrier0",
        "nvvm.cp.async.bulk.commit.group",
        "nvvm.fence.proxy",
        "nvvm.tcgen05.relinquish_alloc_permit",
        "nvvm.bar.warp.sync",
        "nvvm.mbarrier.inval",
        "nvvm.tcgen05.shift",
    ] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::OptionalTypedOperands),
            "{name}"
        );
    }
    assert_eq!(
        registry.operation_shape("nvvm.rcp.approx.ftz.f"),
        Some(OperationShape::UnaryOperand)
    );
    for name in [
        "nvvm.ldmatrix",
        "nvvm.mbarrier.arrive.nocomplete",
        "nvvm.mbarrier.test.wait",
        "nvvm.tcgen05.mma_smem_desc",
        "nvvm.wmma.mma",
    ] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::OperandClauses),
            "{name}"
        );
    }

    let recovery = [
        "nvvm.barrier",
        "nvvm.barrier.arrive",
        "nvvm.clusterlaunchcontrol.query.cancel",
        "nvvm.clusterlaunchcontrol.try.cancel",
        "nvvm.convert.bf16x2.to.f8x2",
        "nvvm.convert.f16x2.to.f8x2",
        "nvvm.convert.f32x2.to.bf16x2",
        "nvvm.convert.f32x2.to.f16x2",
        "nvvm.convert.f32x2.to.f4x2",
        "nvvm.convert.f32x2.to.f6x2",
        "nvvm.convert.f32x2.to.f8x2",
        "nvvm.convert.f32x4.to.f4x4",
        "nvvm.convert.f32x4.to.f6x4",
        "nvvm.convert.f32x4.to.f8x4",
        "nvvm.convert.f4x2.to.f16x2",
        "nvvm.convert.f6x2.to.f16x2",
        "nvvm.convert.f8x2.to.bf16x2",
        "nvvm.convert.f8x2.to.f16x2",
        "nvvm.convert.float.to.tf32",
        "nvvm.cp.async.bulk.global.shared.cta",
        "nvvm.cp.async.bulk.prefetch",
        "nvvm.cp.async.bulk.shared.cluster.global",
        "nvvm.cp.async.bulk.shared.cluster.shared.cta",
        "nvvm.cp.async.bulk.tensor.global.shared.cta",
        "nvvm.cp.async.bulk.tensor.prefetch",
        "nvvm.cp.async.bulk.tensor.reduce",
        "nvvm.cp.async.bulk.tensor.shared.cluster.global",
        "nvvm.cp.async.bulk.wait_group",
        "nvvm.cp.async.shared.global",
        "nvvm.cp.async.wait.group",
        "nvvm.dot.accumulate.2way",
        "nvvm.dot.accumulate.4way",
        "nvvm.elect.sync",
        "nvvm.fence.proxy.acquire",
        "nvvm.fence.proxy.release",
        "nvvm.griddepcontrol",
        "nvvm.inline_ptx",
        "nvvm.mapa",
        "nvvm.match.sync",
        "nvvm.mbarrier.arrive",
        "nvvm.mbarrier.arrive.expect_tx",
        "nvvm.mbarrier.arrive_drop",
        "nvvm.mbarrier.arrive_drop.expect_tx",
        "nvvm.mbarrier.complete_tx",
        "nvvm.mbarrier.expect_tx",
        "nvvm.mbarrier.init",
        "nvvm.mbarrier.try_wait.parity",
        "nvvm.mbarrier.try_wait",
        "nvvm.memory.barrier",
        "nvvm.mma.block_scale",
        "nvvm.mma.sp.block_scale",
        "nvvm.mma.sp.sync",
        "nvvm.mma.sync",
        "nvvm.nanosleep",
        "nvvm.pmevent",
        "nvvm.prefetch",
        "nvvm.prmt",
        "nvvm.read.ptx.sreg.cluster.ctaid.x",
        "nvvm.read.ptx.sreg.cluster.ctaid.y",
        "nvvm.read.ptx.sreg.cluster.ctaid.z",
        "nvvm.read.ptx.sreg.cluster.ctarank",
        "nvvm.read.ptx.sreg.cluster.nctaid.x",
        "nvvm.read.ptx.sreg.cluster.nctaid.y",
        "nvvm.read.ptx.sreg.cluster.nctaid.z",
        "nvvm.read.ptx.sreg.cluster.nctarank",
        "nvvm.read.ptx.sreg.clusterid.x",
        "nvvm.read.ptx.sreg.clusterid.y",
        "nvvm.read.ptx.sreg.clusterid.z",
        "nvvm.read.ptx.sreg.ctaid.x",
        "nvvm.read.ptx.sreg.ctaid.y",
        "nvvm.read.ptx.sreg.ctaid.z",
        "nvvm.read.ptx.sreg.gridid",
        "nvvm.read.ptx.sreg.laneid",
        "nvvm.read.ptx.sreg.nclusterid.x",
        "nvvm.read.ptx.sreg.nclusterid.y",
        "nvvm.read.ptx.sreg.nclusterid.z",
        "nvvm.read.ptx.sreg.nctaid.x",
        "nvvm.read.ptx.sreg.nctaid.y",
        "nvvm.read.ptx.sreg.nctaid.z",
        "nvvm.read.ptx.sreg.nsmid",
        "nvvm.read.ptx.sreg.ntid.x",
        "nvvm.read.ptx.sreg.ntid.y",
        "nvvm.read.ptx.sreg.ntid.z",
        "nvvm.read.ptx.sreg.nwarpid",
        "nvvm.read.ptx.sreg.smid",
        "nvvm.read.ptx.sreg.tid.x",
        "nvvm.read.ptx.sreg.tid.y",
        "nvvm.read.ptx.sreg.tid.z",
        "nvvm.read.ptx.sreg.warpid",
        "nvvm.read.ptx.sreg.warpsize",
        "nvvm.redux.sync",
        "nvvm.setmaxregister",
        "nvvm.shfl.sync",
        "nvvm.st.bulk",
        "nvvm.stmatrix",
        "nvvm.tcgen05.alloc",
        "nvvm.tcgen05.commit",
        "nvvm.tcgen05.cp",
        "nvvm.tcgen05.dealloc",
        "nvvm.tcgen05.fence",
        "nvvm.tcgen05.ld",
        "nvvm.tcgen05.mma",
        "nvvm.tcgen05.mma.block_scale",
        "nvvm.tcgen05.mma.sp",
        "nvvm.tcgen05.mma.sp.block_scale",
        "nvvm.tcgen05.mma.ws",
        "nvvm.tcgen05.mma.ws.sp",
        "nvvm.tcgen05.st",
        "nvvm.tcgen05.wait",
        "nvvm.vote.sync",
        "nvvm.wgmma.mma_async",
        "nvvm.wgmma.wait.group.sync.aligned",
        "nvvm.wmma.load",
        "nvvm.wmma.store",
    ];
    assert_eq!(config.operation_shapes.len() + recovery.len(), 185);
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn nvvm_optional_positional_and_inferred_forms_recover_to_the_next_operation() {
    let registry = DialectRegistry::from_name("nvvm").unwrap();
    let source = br#"module {
      func.func @gaps(%x: i32, %ptr: !llvm.ptr<3>, %predicate: i1) {
        %tid = nvvm.read.ptx.sreg.tid.x range <0, 1024> : i32
        nvvm.memory.barrier #nvvm.mem_scope<cta>
        nvvm.setmaxregister #nvvm.action<increase> 64
        %state = nvvm.mbarrier.arrive %ptr : !llvm.ptr<3> -> i64
        nvvm.mbarrier.init %ptr, %x, predicate = %predicate : !llvm.ptr<3>, i32, i1
        nvvm.cp.async.shared.global %ptr, %ptr, %x, cache = #nvvm.load_cache_modifier<ca>
          : !llvm.ptr<3>, !llvm.ptr, i32
        %converted = nvvm.convert.float.to.tf32 %x
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count()
            >= 7,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn pdl_preset_inventory_matches_llvm_22_1_recovery_coverage() {
    assert!(DialectRegistry::preset_names().contains(&"pdl"));
    let registry = DialectRegistry::from_name("pdl").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/pdl.json")).unwrap();
    assert!(config.operation_shapes.is_empty());
    assert!(config.operation_formats.is_empty());
    assert_eq!(registry.operation_names().count(), 4);

    let recovery = [
        "pdl.apply_native_constraint",
        "pdl.apply_native_rewrite",
        "pdl.attribute",
        "pdl.erase",
        "pdl.operand",
        "pdl.operands",
        "pdl.operation",
        "pdl.pattern",
        "pdl.range",
        "pdl.replace",
        "pdl.result",
        "pdl.results",
        "pdl.rewrite",
        "pdl.type",
        "pdl.types",
    ];
    assert_eq!(recovery.len(), 15);
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn pdl_inferred_handle_and_rewrite_forms_recover_to_following_operations() {
    let registry = DialectRegistry::from_name("pdl").unwrap();
    let source = br#"module {
      %type = pdl.type : i32
      %types = pdl.types : [i32, i64]
      %attribute = pdl.attribute : %type
      %operand = pdl.operand : %type
      %operands = pdl.operands : %types
      %operation = pdl.operation "foo.op"(%operand : !pdl.value) {"value" = %attribute} -> (%type : !pdl.type)
      %result = pdl.result 0 of %operation
      %results = pdl.results of %operation
      pdl.apply_native_constraint "constraint"(%operation : !pdl.operation)
      %rewritten = pdl.apply_native_rewrite "rewrite"(%attribute : !pdl.attribute) : !pdl.attribute
      %range = pdl.range %operand, %operands : !pdl.value, !pdl.range<value>
      pdl.erase %operation
      pdl.replace %operation with (%result : !pdl.value)
      "test.after_handles"() : () -> ()
      pdl.pattern @named : benefit(1) attributes {tag = "pattern"} { "test.match"() : () -> () }
      pdl.rewrite %operation with "external"(%operand : !pdl.value) attributes {tag = "rewrite"}
      "test.after_regions"() : () -> ()
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count()
            >= 15,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after_handles", "test.after_regions"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn pdl_opaque_types_and_attributes_need_no_dialect_descriptors() {
    let registry = DialectRegistry::from_name("pdl").unwrap();
    let source = br#"module {
      %handles = "test.source"() {marker = #pdl<opaque>} : () -> tuple<!pdl.attribute, !pdl.operation, !pdl.type, !pdl.value, !pdl.range<value>>
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.source"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.result_types(operation).unwrap()[0]),
        Some("tuple<!pdl.attribute, !pdl.operation, !pdl.type, !pdl.value, !pdl.range<value>>")
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| name == "marker" && value == "#pdl<opaque>")
    );
}

#[test]
fn pdl_interp_preset_inventory_matches_llvm_22_1_recovery_coverage() {
    assert!(DialectRegistry::preset_names().contains(&"pdl_interp"));
    let registry = DialectRegistry::from_name("pdl_interp").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/pdl_interp.json")).unwrap();
    assert!(config.operation_shapes.is_empty());
    assert!(config.operation_formats.is_empty());
    assert_eq!(registry.operation_names().count(), 4);

    let recovery = [
        "pdl_interp.apply_constraint",
        "pdl_interp.apply_rewrite",
        "pdl_interp.are_equal",
        "pdl_interp.branch",
        "pdl_interp.check_attribute",
        "pdl_interp.check_operand_count",
        "pdl_interp.check_operation_name",
        "pdl_interp.check_result_count",
        "pdl_interp.check_type",
        "pdl_interp.check_types",
        "pdl_interp.continue",
        "pdl_interp.create_attribute",
        "pdl_interp.create_operation",
        "pdl_interp.create_range",
        "pdl_interp.create_type",
        "pdl_interp.create_types",
        "pdl_interp.erase",
        "pdl_interp.extract",
        "pdl_interp.finalize",
        "pdl_interp.foreach",
        "pdl_interp.func",
        "pdl_interp.get_attribute",
        "pdl_interp.get_attribute_type",
        "pdl_interp.get_defining_op",
        "pdl_interp.get_operand",
        "pdl_interp.get_operands",
        "pdl_interp.get_result",
        "pdl_interp.get_results",
        "pdl_interp.get_users",
        "pdl_interp.get_value_type",
        "pdl_interp.is_not_null",
        "pdl_interp.record_match",
        "pdl_interp.replace",
        "pdl_interp.switch_attribute",
        "pdl_interp.switch_operand_count",
        "pdl_interp.switch_operation_name",
        "pdl_interp.switch_result_count",
        "pdl_interp.switch_type",
        "pdl_interp.switch_types",
    ];
    assert_eq!(recovery.len(), 39);
    for name in recovery {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn pdl_interp_successors_regions_and_inferred_results_recover_at_cfg_boundaries() {
    let registry = DialectRegistry::from_name("pdl_interp").unwrap();
    let source = br#"module {
      pdl_interp.func @matcher(%root: !pdl.operation) {
        %type = pdl_interp.create_type i32
        %attribute = pdl_interp.create_attribute 1 : i32
        %operation = pdl_interp.create_operation "foo.op" -> <inferred>
        %result = pdl_interp.get_result 0 of %operation
        pdl_interp.apply_constraint "constraint"(%result : !pdl.value) -> ^match, ^failure
      ^match:
        pdl_interp.switch_operation_name of %operation to ["foo.op"](^case) -> ^failure
      ^case:
        pdl_interp.foreach %item : !pdl.operation in %items {
          pdl_interp.continue
        } -> ^after_loop
      ^after_loop:
        pdl_interp.record_match @rewriters::rewrite(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^failure
      ^failure:
        pdl_interp.finalize
      }
      "test.after_cfg"() : () -> ()
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count()
            >= 11,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    assert!(
        document
            .operations()
            .any(|operation| document.operation_name(operation) == Some("test.after_cfg")),
        "following generic operation should survive whole-operation recovery"
    );
}

#[test]
fn pdl_interp_opaque_values_need_no_dialect_descriptors() {
    let registry = DialectRegistry::from_name("pdl_interp").unwrap();
    let source = br#"module {
      %handles = "test.source"() {marker = #pdl_interp<opaque>} : () -> tuple<!pdl.attribute, !pdl.operation, !pdl.type, !pdl.value, !pdl.range<operation>>
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.source"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.result_types(operation).unwrap()[0]),
        Some("tuple<!pdl.attribute, !pdl.operation, !pdl.type, !pdl.value, !pdl.range<operation>>")
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| name == "marker" && value == "#pdl_interp<opaque>")
    );
}

#[test]
fn ptr_preset_exposes_explicit_default_signatures() {
    assert!(DialectRegistry::preset_names().contains(&"ptr"));
    let registry = DialectRegistry::from_name("ptr").unwrap();
    let source = br#"module {
      func.func @forms(%memref: memref<f32>, %ptr: !ptr.ptr<#ptr.generic_space>) {
        %to = ptr.to_ptr %memref {tag = "to"} : memref<f32> -> !ptr.ptr<#ptr.generic_space>
        %from = ptr.from_ptr %ptr {tag = "from"} : !ptr.ptr<#ptr.generic_space> -> memref<f32>
        %loaded = ptr.load %ptr {tag = "load"} : !ptr.ptr<#ptr.generic_space> -> i32
        %difference = ptr.ptr_diff %ptr, %ptr {tag = "diff"} : !ptr.ptr<#ptr.generic_space> -> i64
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

    for (name, operands, signature, tag) in [
        (
            "ptr.to_ptr",
            1,
            "(memref<f32>) -> !ptr.ptr<#ptr.generic_space>",
            "\"to\"",
        ),
        (
            "ptr.from_ptr",
            1,
            "(!ptr.ptr<#ptr.generic_space>) -> memref<f32>",
            "\"from\"",
        ),
        (
            "ptr.load",
            1,
            "(!ptr.ptr<#ptr.generic_space>) -> i32",
            "\"load\"",
        ),
        (
            "ptr.ptr_diff",
            2,
            "(!ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>) -> i64",
            "\"diff\"",
        ),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.operands(operation).unwrap().len(), operands);
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature)
        );
        assert!(document.operation_regions(operation).unwrap().is_empty());
        assert!(document.successors(operation).unwrap().is_empty());
        assert!(
            document
                .attributes(operation)
                .unwrap()
                .any(|(attribute, value)| attribute == "tag" && value == tag)
        );
    }
}

#[test]
fn ptr_preset_inventory_matches_llvm_22_1_structural_coverage() {
    let registry = DialectRegistry::from_name("ptr").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/ptr.json")).unwrap();
    assert_eq!(config.operation_shapes.len(), 4);

    for name in ["ptr.from_ptr", "ptr.load", "ptr.to_ptr"] {
        assert_eq!(
            registry.operation_shape(name),
            Some(OperationShape::UnaryOperand),
            "{name}"
        );
    }
    assert_eq!(
        registry.operation_shape("ptr.ptr_diff"),
        Some(OperationShape::BinaryOperands)
    );
    let recovery = [
        "ptr.constant",
        "ptr.gather",
        "ptr.get_metadata",
        "ptr.masked_load",
        "ptr.masked_store",
        "ptr.ptr_add",
        "ptr.scatter",
        "ptr.store",
        "ptr.type_offset",
    ];
    assert_eq!(config.operation_shapes.len() + recovery.len(), 13);
    for name in recovery {
        assert!(registry.operation(name).is_none(), "{name}");
    }
}

#[test]
fn ptr_modifier_inference_and_partial_type_forms_recover_to_the_next_operation() {
    let registry = DialectRegistry::from_name("ptr").unwrap();
    let source = br#"module {
      func.func @gaps(
          %ptr: !ptr.ptr<#ptr.generic_space>,
          %metadata: !ptr.ptr_metadata<memref<f32, #ptr.generic_space>>,
          %ptrs: vector<4x!ptr.ptr<#ptr.generic_space>>,
          %mask: vector<4xi1>, %values: vector<4xf32>, %value: i32,
          %offset: index) {
        %from = ptr.from_ptr %ptr metadata %metadata : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
        %load = ptr.load volatile %ptr : !ptr.ptr<#ptr.generic_space> -> i32
        %difference = ptr.ptr_diff nuw %ptr, %ptr : !ptr.ptr<#ptr.generic_space> -> i64
        %null = ptr.constant {tag = "leading"} #ptr.null : !ptr.ptr<#ptr.generic_space>
        %type_offset = ptr.type_offset f32 : index
        %gathered = ptr.gather %ptrs, %mask, %values : vector<4x!ptr.ptr<#ptr.generic_space>> -> vector<4xf32>
        %metadata_result = ptr.get_metadata %ptr : !ptr.ptr<#ptr.generic_space>
        %masked = ptr.masked_load %ptr, %mask, %values : !ptr.ptr<#ptr.generic_space> -> vector<4xf32>
        ptr.masked_store %values, %ptr, %mask : vector<4xf32>, !ptr.ptr<#ptr.generic_space>
        %added = ptr.ptr_add %ptr, %offset : !ptr.ptr<#ptr.generic_space>, index
        ptr.scatter %values, %ptrs, %mask : vector<4xf32>, vector<4x!ptr.ptr<#ptr.generic_space>>
        ptr.store %value, %ptr : i32, !ptr.ptr<#ptr.generic_space>
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count()
            >= 10,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| matches!(
                diagnostic.kind(),
                ParseDiagnosticKind::ShapeMismatch(
                    OperationShape::UnaryOperand
                        | OperationShape::BinaryOperands
                        | OperationShape::LiteralAttribute
                )
            ))
            .count(),
        3,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn ptr_opaque_types_and_attributes_need_no_dialect_descriptors() {
    let registry = DialectRegistry::from_name("ptr").unwrap();
    let source = br#"module {
      %pointer = "test.source"() {
        layout = #ptr.spec<size = 64, abi = 64, preferred = 64>
      } : () -> !ptr.ptr<#ptr.generic_space>
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
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.source"))
        .unwrap();
    assert_eq!(
        document.type_spelling(document.result_types(operation).unwrap()[0]),
        Some("!ptr.ptr<#ptr.generic_space>")
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| name == "layout"
                && value == "#ptr.spec<size = 64, abi = 64, preferred = 64>")
    );
}

#[test]
fn quant_preset_exposes_complete_explicit_cast_signatures() {
    assert!(DialectRegistry::preset_names().contains(&"quant"));
    let registry = DialectRegistry::from_name("quant").unwrap();
    let source = br#"module {
      func.func @forms(
          %quantized: !quant.uniform<i8:f32, 2.0>,
          %expressed: f32,
          %integer: i8) {
        %dequantized = quant.dcast %quantized {tag = "dcast", metadata = #quant<opaque>} : !quant.uniform<i8:f32, 2.0> to f32
        %quantized_result = quant.qcast %expressed {tag = "qcast"} : f32 to !quant.uniform<i8:f32, 2.0>
        %stored = quant.scast %quantized {tag = "storage"} : !quant.uniform<i8:f32, 2.0> to i8
        %requantized = quant.scast %integer {tag = "requantized"} : i8 to !quant.uniform<i8:f32, 2.0>
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

    for (name, signature, tag) in [
        (
            "quant.dcast",
            "(!quant.uniform<i8:f32, 2.0>) -> f32",
            "\"dcast\"",
        ),
        (
            "quant.qcast",
            "(f32) -> !quant.uniform<i8:f32, 2.0>",
            "\"qcast\"",
        ),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(document.operands(operation).unwrap().len(), 1, "{name}");
        assert_eq!(document.result_types(operation).unwrap().len(), 1, "{name}");
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
        assert!(
            document
                .attributes(operation)
                .unwrap()
                .any(|(attribute, value)| attribute == "tag" && value == tag),
            "{name}"
        );
    }

    let storage_casts = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("quant.scast"))
        .collect::<Vec<_>>();
    assert_eq!(storage_casts.len(), 2);
    assert_eq!(
        document.type_spelling(document.function_type(storage_casts[0]).unwrap()),
        Some("(!quant.uniform<i8:f32, 2.0>) -> i8")
    );
    assert_eq!(
        document.type_spelling(document.function_type(storage_casts[1]).unwrap()),
        Some("(i8) -> !quant.uniform<i8:f32, 2.0>")
    );

    let dequantize = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("quant.dcast"))
        .unwrap();
    assert!(
        document
            .attributes(dequantize)
            .unwrap()
            .any(|(name, value)| name == "metadata" && value == "#quant<opaque>")
    );
}

#[test]
fn quant_preset_inventory_matches_llvm_22_1_complete_coverage() {
    let registry = DialectRegistry::from_name("quant").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/quant.json")).unwrap();
    assert!(config.operation_shapes.is_empty());
    assert_eq!(config.operation_formats.len(), 3);
    assert_eq!(registry.operation_names().count(), 4);
    assert_eq!(
        config
            .operation_formats
            .iter()
            .map(|operation| operation.name.as_str())
            .collect::<Vec<_>>(),
        ["quant.dcast", "quant.qcast", "quant.scast"]
    );
    for operation in &config.operation_formats {
        assert_eq!(
            operation.format,
            "$operands attr-dict `:` type($operands) `to` type($results)"
        );
    }

    // These older spellings are deliberately not claimed by the LLVM 22.1 preset.
    for name in ["quant.stats", "quant.stats_ref"] {
        assert!(registry.operation(name).is_none(), "{name}");
        assert_eq!(registry.operation_shape(name), None, "{name}");
    }
}

#[test]
fn quant_partial_modifier_and_removed_statistics_forms_recover_at_boundaries() {
    let registry = DialectRegistry::from_name("quant").unwrap();
    let source = br#"module {
      func.func @gaps(%quantized: !quant.uniform<i8:f32, 2.0>, %expressed: f32, %storage: i8) {
        %bad_dcast = quant.dcast %quantized rounding nearest : !quant.uniform<i8:f32, 2.0> to f32
        %bad_qcast = quant.qcast %expressed saturating : f32 to !quant.uniform<i8:f32, 2.0>
        %bad_scast = quant.scast %storage signed : i8 to !quant.uniform<i8:f32, 2.0>
        %missing_input_type = quant.qcast %expressed to !quant.uniform<i8:f32, 2.0>
        quant.stats %expressed : f32
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::FormatMismatch)
            .count(),
        4,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    assert_eq!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        1,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}

#[test]
fn rocdl_preset_exposes_explicit_intrinsic_families() {
    assert!(DialectRegistry::preset_names().contains(&"rocdl"));
    let registry = DialectRegistry::from_name("rocdl").unwrap();
    let source = br#"module {
      func.func @families(%a: i32, %b: i32, %ptr: !llvm.ptr<3>, %x: f32, %y: f32, %acc: vector<4xf32>) {
        rocdl.barrier {tag = "barrier"}
        %count = rocdl.mbcnt.lo %a, %b {tag = "wave"} : (i32, i32) -> i32
        %read = rocdl.ds.read.tr4.b64 %ptr {tag = "read"} : !llvm.ptr<3> -> vector<2xi32>
        %lane = rocdl.readfirstlane %a {tag = "lane"} : i32
        %mfma = rocdl.mfma.f32.4x4x1f32 %x, %y, %acc, %a, %a, %a {tag = "mfma"} : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
        %wmma = rocdl.wmma.f16.16x16x16.f16 %a, %b, %acc {opsel = false} : (i32, i32, vector<4xf32>) -> vector<4xf32>
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

    for (name, operands, results, signature) in [
        ("rocdl.barrier", 0, 0, "() -> ()"),
        ("rocdl.mbcnt.lo", 2, 1, "(i32, i32) -> i32"),
        (
            "rocdl.ds.read.tr4.b64",
            1,
            1,
            "!llvm.ptr<3> -> vector<2xi32>",
        ),
        ("rocdl.readfirstlane", 1, 1, "(i32) -> i32"),
        (
            "rocdl.mfma.f32.4x4x1f32",
            6,
            1,
            "(f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>",
        ),
        (
            "rocdl.wmma.f16.16x16x16.f16",
            3,
            1,
            "(i32, i32, vector<4xf32>) -> vector<4xf32>",
        ),
    ] {
        let operation = document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some(name))
            .unwrap();
        assert_eq!(
            document.operands(operation).unwrap().len(),
            operands,
            "{name}"
        );
        assert_eq!(
            document.result_types(operation).unwrap().len(),
            results,
            "{name}"
        );
        assert_eq!(
            document.type_spelling(document.function_type(operation).unwrap()),
            Some(signature),
            "{name}"
        );
        assert!(
            document.operation_regions(operation).unwrap().is_empty(),
            "{name}"
        );
        assert!(document.successors(operation).unwrap().is_empty(), "{name}");
    }
}
#[test]
fn rocdl_preset_inventory_matches_expanded_llvm_22_1_coverage() {
    let registry = DialectRegistry::from_name("rocdl").unwrap();
    let config = RegistryConfig::from_json(include_str!("../registries/rocdl.json")).unwrap();
    assert_eq!(config.operation_shapes.len(), 125);
    assert!(config.operation_formats.is_empty());

    let supported = config
        .operation_shapes
        .iter()
        .map(|operation| operation.name.as_str())
        .collect::<Vec<_>>();
    for (prefix, expected) in [
        ("rocdl.mfma.", 47),
        ("rocdl.smfmac.", 28),
        ("rocdl.wmma.", 38),
    ] {
        assert_eq!(
            supported
                .iter()
                .filter(|name| name.starts_with(prefix))
                .count(),
            expected,
            "{prefix}"
        );
    }

    for (name, shape) in [
        ("rocdl.barrier", OperationShape::OptionalTypedOperands),
        ("rocdl.s.barrier", OperationShape::OptionalTypedOperands),
        ("rocdl.mbcnt.lo", OperationShape::OperandClauses),
        ("rocdl.mbcnt.hi", OperationShape::OperandClauses),
        ("rocdl.ds_swizzle", OperationShape::OperandClauses),
        ("rocdl.ds_bpermute", OperationShape::OperandClauses),
        ("rocdl.readlane", OperationShape::OperandClauses),
        ("rocdl.readfirstlane", OperationShape::UnaryOperand),
        ("rocdl.ds.read.tr4.b64", OperationShape::OperandClauses),
        ("rocdl.ds.read.tr6.b96", OperationShape::OperandClauses),
        ("rocdl.ds.read.tr8.b64", OperationShape::OperandClauses),
        ("rocdl.ds.read.tr16.b64", OperationShape::OperandClauses),
    ] {
        assert_eq!(registry.operation_shape(name), Some(shape), "{name}");
    }

    for name in [
        "rocdl.workitem.id.x",
        "rocdl.s.barrier.signal",
        "rocdl.s.barrier.init",
        "rocdl.raw.buffer.load",
        "rocdl.tensor.load.to.lds",
        "rocdl.cvt.scalef32.pk8.fp8.f32",
        "rocdl.update.dpp",
        "rocdl.ballot",
        "rocdl.cos",
    ] {
        assert_eq!(registry.operation_shape(name), None, "{name}");
        assert!(registry.operation(name).is_none(), "{name}");
    }
}
#[test]
fn rocdl_inferred_immediate_range_and_qualified_forms_recover_at_boundaries() {
    let registry = DialectRegistry::from_name("rocdl").unwrap();
    let source = br#"module {
      func.func @gaps(%ptr: !llvm.ptr<3>, %value: i32, %scale: i32) {
        %id = rocdl.workitem.id.x range #llvm.constant_range<0, 64> : i32
        rocdl.s.barrier.signal id = 1
        rocdl.s.barrier.init %ptr member_cnt = 4 : !llvm.ptr<3>
        %cos = rocdl.cos %value i32 -> i32
        %converted = rocdl.cvt.scalef32.pk8.fp8.f32 {round = 0 : i32} %value, %scale : i32
        "test.after"() : () -> ()
        func.return
      }
    }"#;
    let parsed = ParsedFile::parse_with_registry(source.as_slice(), &registry).unwrap();
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count()
            >= 5,
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &registry);
    let document = lowered.document.unwrap();
    assert!(!document.is_semantically_complete());
    for name in ["test.after", "func.return"] {
        assert!(
            document
                .operations()
                .any(|operation| document.operation_name(operation) == Some(name)),
            "{name}"
        );
    }
}
