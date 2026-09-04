use std::{fs, path::PathBuf};
use zirium::{
    parser::{ParseLimits, ParsedFile},
    semantic::{
        LargeAttributeValue, LoweringMode, RetentionProfile, SharedRegistry, ValueId,
        ValueReference, lower_proving_fixture, lower_proving_fixture_with_retention,
    },
};

use zirium::semantic::{
    AffineBinaryOperator, AffineExprValue, AttributeValue, IntegerSetRelation, LocationValue,
    MemRefLayout, TypeValue,
};

fn fixture(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/semantic-proving")
            .join(name),
    )
    .unwrap()
}

fn generic_complete_fixture(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/generic-complete")
            .join(name),
    )
    .unwrap()
}

fn shaped_affine_fixture(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/shaped-affine")
            .join(name),
    )
    .unwrap()
}

#[test]
fn unknown_custom_operations_lower_with_exact_text_and_nested_regions() {
    let source = b"vendor.outer @entry {\n  vendor.inner\n}".to_vec();
    let parsed = ParsedFile::parse(source.clone()).unwrap();
    let lowered = lower_proving_fixture_with_retention(
        &parsed,
        LoweringMode::BestEffort,
        RetentionProfile::SemanticOnly,
        &SharedRegistry,
    );
    assert!(!lowered.semantically_complete);
    assert!(
        lowered
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("unknown custom operation"))
    );
    let document = lowered.document.unwrap();
    let outer = document.root_operations()[0];
    assert_eq!(document.operation_name(outer), Some("vendor.outer"));
    assert!(
        document
            .attributes(outer)
            .unwrap()
            .any(|(name, spelling)| name == "sym_name" && spelling == "@entry")
    );
    assert_eq!(document.operation_is_unparsed(outer), Some(true));
    assert_eq!(
        document.operation_unparsed_text(outer),
        Some(source.as_slice())
    );
    let region = document.operation_regions(outer).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
    let inner = document.block_operations(block).unwrap()[0];
    assert_eq!(document.operation_name(inner), Some("vendor.inner"));
    assert_eq!(document.operation_is_unparsed(inner), Some(true));

    let ordinary = ParsedFile::parse(b"\"ordinary\"() : () -> ()".to_vec()).unwrap();
    let ordinary = lower_proving_fixture(&ordinary, LoweringMode::Strict, &SharedRegistry)
        .document
        .unwrap();
    let operation = ordinary.root_operations()[0];
    assert_eq!(ordinary.operation_is_unparsed(operation), Some(false));
    assert_eq!(ordinary.operation_unparsed_text(operation), None);

    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
}

#[test]
fn consecutive_unknown_custom_operations_preserve_result_prefix_text() {
    let source = b"%result = vendor.first\nvendor.second".to_vec();
    let parsed = ParsedFile::parse(source).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = lowered.document.unwrap();
    let operations = document.operations().collect::<Vec<_>>();
    assert_eq!(operations.len(), 2);
    assert_eq!(document.operation_name(operations[0]), Some("vendor.first"));
    assert!(
        document
            .operation_unparsed_text(operations[0])
            .unwrap()
            .starts_with(b"%result = vendor.first")
    );
    assert_eq!(
        document.operation_name(operations[1]),
        Some("vendor.second")
    );
}

#[test]
fn generic_mnemonic_range_survives_dotted_results_and_escapes() {
    let source = br#"%result.0 = "vendor.\22quoted"() : () -> i32"#;
    let parsed = ParsedFile::parse(source.as_slice()).unwrap();
    let document = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
        .document
        .unwrap();
    let operation = document.root_operations()[0];
    assert_eq!(
        document.operation_name(operation),
        Some(r"vendor.\22quoted")
    );
    assert_eq!(document.result_types(operation).unwrap().len(), 1);
}

#[test]
fn stablehlo_reduce_clauses_remain_part_of_the_recovered_operation() {
    let source = br#"stablehlo.reduce(%input init: %init) applies stablehlo.add across dimensions = [0]
%result = stablehlo.reduce(%input init: %init) across dimensions = [0] reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
  stablehlo.add %lhs, %rhs
}
"#;
    let parsed = ParsedFile::parse(source.to_vec()).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = lowered.document.unwrap();

    assert_eq!(document.root_operations().len(), 2);
    assert!(
        document
            .root_operations()
            .iter()
            .all(|&operation| document.operation_name(operation) == Some("stablehlo.reduce"))
    );
    assert_eq!(
        document.operation_regions(document.root_operations()[0]),
        Some(&[][..])
    );

    let second = document.root_operations()[1];
    let region = document.operation_regions(second).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
    let nested = document.block_operations(block).unwrap();
    assert_eq!(nested.len(), 1);
    assert_eq!(document.operation_name(nested[0]), Some("stablehlo.add"));
    for invented in ["applies", "across", "reducer"] {
        assert!(
            document
                .operations()
                .all(|operation| document.operation_name(operation) != Some(invented))
        );
    }
}

#[test]
fn unknown_custom_leading_symbol_stops_before_argument_list() {
    let parsed =
        ParsedFile::parse(b"vendor.func @entry(%x: i32) attributes {flag = true}".to_vec())
            .unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = lowered.document.unwrap();
    let operation = document.root_operations()[0];
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, spelling)| name == "sym_name" && spelling == "@entry")
    );
}

#[test]
fn boolean_attributes_lower_with_exact_spelling() {
    let parsed =
        ParsedFile::parse(b"\"flags\"() {enabled = true, disabled = false} : () -> ()".to_vec())
            .unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    let document = lowered.document.expect("strict boolean attributes");
    let operation = document.root_operations()[0];

    for (name, expected) in [("enabled", true), ("disabled", false)] {
        let attribute = document.attribute_id(operation, name).unwrap();
        assert_eq!(
            document.attribute_value(attribute),
            Some(&AttributeValue::Boolean(expected))
        );
        assert_eq!(
            document.attribute_spelling_value(attribute),
            Some(if expected { "true" } else { "false" })
        );
    }
}

#[test]
fn semantic_attribute_depth_limit_uses_invalid_sentinel() {
    let nested = (0..12).fold("1".to_owned(), |value, depth| {
        if depth % 2 == 0 {
            format!("[{value}]")
        } else {
            format!("{{k = {value}}}")
        }
    });
    let source = format!("\"deep\"() {{a = {nested}}} : () -> ()");
    let parsed = ParsedFile::parse_with_limits(
        source.into_bytes(),
        ParseLimits {
            max_attribute_depth: 3,
            ..ParseLimits::default()
        },
    )
    .unwrap();

    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("attribute nesting depth limit"))
    );

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.expect("best-effort document");
    document.validate().unwrap();
    let operation = document.operations().next().unwrap();
    let attribute = document.attribute_id(operation, "a").unwrap();
    assert!(matches!(
        document.attribute_value(attribute),
        Some(AttributeValue::Array(_) | AttributeValue::Dictionary(_))
    ));
    assert!(
        best.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("attribute nesting depth limit"))
    );
}

#[test]
fn affine_maps_sets_aliases_and_nested_values_lower_and_intern() {
    let parsed = ParsedFile::parse(shaped_affine_fixture("semantic-valid.mlir")).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    document.validate().unwrap();
    let operation = document.operations().next().unwrap();

    let map = match document.attribute_value(document.attribute_id(operation, "map_alias").unwrap())
    {
        Some(AttributeValue::AffineMap(map)) => *map,
        other => panic!("unexpected affine map: {other:?}"),
    };
    assert_eq!(
        document.attribute_value(document.attribute_id(operation, "same_map").unwrap()),
        Some(&AttributeValue::AffineMap(map))
    );
    let map_value = document.affine_map(map).unwrap();
    assert_eq!(
        (
            map_value.dimensions,
            map_value.symbols,
            map_value.results.len()
        ),
        (2, 1, 2)
    );
    assert!(matches!(
        document.affine_expression(map_value.results[0]),
        Some(AffineExprValue::Binary {
            operator: AffineBinaryOperator::Add,
            ..
        })
    ));

    let set = match document.attribute_value(document.attribute_id(operation, "set_alias").unwrap())
    {
        Some(AttributeValue::IntegerSet(set)) => *set,
        other => panic!("unexpected integer set: {other:?}"),
    };
    let set_value = document.integer_set(set).unwrap();
    assert_eq!(
        (
            set_value.dimensions,
            set_value.symbols,
            set_value.constraints.len()
        ),
        (1, 1, 2)
    );
    assert_eq!(
        set_value.constraints[0].relation,
        IntegerSetRelation::GreaterEqual
    );
    assert_eq!(set_value.constraints[1].relation, IntegerSetRelation::Equal);

    let nested = document
        .attribute_value(document.attribute_id(operation, "nested").unwrap())
        .unwrap();
    assert!(
        matches!(nested, AttributeValue::Array(values) if matches!(values.first(), Some(AttributeValue::AffineMap(id)) if *id == map) && matches!(values.get(1), Some(AttributeValue::Array(inner)) if matches!(inner.first(), Some(AttributeValue::IntegerSet(_)))))
    );
    let stats = document.statistics();
    assert!(stats.affine_expressions > 0);
    assert_eq!(
        stats.affine_maps, 2,
        "alias/direct structural duplicates must intern"
    );
    assert_eq!(stats.integer_sets, 2);
    assert!(matches!(
        document.type_value(document.result_types(operation).unwrap()[0]),
        Some(TypeValue::MemRef {
            layout: Some(MemRefLayout::AffineMap(id)),
            ..
        }) if *id == map
    ));
}

#[test]
fn malformed_affine_values_preserve_shape_with_nested_invalid_sentinels() {
    let parsed = ParsedFile::parse(shaped_affine_fixture("semantic-malformed.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    for expected in ["arity", "operator", "constraint", "operand"] {
        assert!(
            strict
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains(expected)),
            "missing {expected}: {:?}",
            strict.diagnostics
        );
    }

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    assert!(!document.is_semantically_complete());
    document.validate().unwrap();
    let operation = document
        .operations()
        .find(|&id| document.operation_name(id) == Some("affine.semantic.bad"))
        .unwrap();
    let arity = match document
        .attribute_value(document.attribute_id(operation, "arity").unwrap())
        .unwrap()
    {
        AttributeValue::AffineMap(map) => document.affine_map(*map).unwrap(),
        other => panic!("unexpected arity value: {other:?}"),
    };
    assert!(
        matches!(document.affine_expression(arity.results[0]), Some(AffineExprValue::Binary { left, .. }) if matches!(document.affine_expression(*left), Some(AffineExprValue::Invalid(_))))
    );
    let nested = document
        .attribute_value(document.attribute_id(operation, "nested").unwrap())
        .unwrap();
    assert!(
        matches!(nested, AttributeValue::Array(values) if matches!(values.first(), Some(AttributeValue::AffineMap(map)) if document.affine_map(*map).unwrap().results.iter().any(|id| matches!(document.affine_expression(*id), Some(AffineExprValue::Binary { right, .. }) if matches!(document.affine_expression(*right), Some(AffineExprValue::Invalid(_)))))))
    );
}

#[test]
fn affine_incomplete_literals_and_transitive_aliases_are_safe_and_diagnosed() {
    let parsed = ParsedFile::parse(shaped_affine_fixture("semantic-edge.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    for expected in [
        "malformed affine dimension arity",
        "out of range for i64",
        "cyclic attribute alias",
        "has type kind",
        "unresolved attribute alias",
    ] {
        assert!(
            strict
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains(expected)),
            "missing {expected}: {:?}",
            strict.diagnostics
        );
    }

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    let operation = document
        .operations()
        .find(|&id| document.operation_name(id) == Some("affine.semantic.edge"))
        .unwrap();
    assert!(matches!(
        document.attribute_value(document.attribute_id(operation, "map").unwrap()),
        Some(AttributeValue::AffineMap(map)) if document.affine_map(*map).unwrap().results.len() == 1
    ));
    for name in ["empty", "incomplete"] {
        assert!(matches!(
            document.attribute_value(document.attribute_id(operation, name).unwrap()),
            Some(AttributeValue::AffineMap(map))
                if document.affine_map(*map).unwrap().results.iter().any(|id| matches!(document.affine_expression(*id), Some(AffineExprValue::Invalid(_))))
        ));
    }
    assert!(matches!(
        document.attribute_value(document.attribute_id(operation, "empty_set").unwrap()),
        Some(AttributeValue::IntegerSet(set))
            if document.integer_set(*set).unwrap().constraints.iter().any(|constraint| matches!(constraint.relation, IntegerSetRelation::Invalid(_)))
    ));
    assert!(matches!(
        document.type_value(document.result_types(operation).unwrap()[0]),
        Some(TypeValue::MemRef {
            layout: Some(MemRefLayout::AffineMap(_)),
            ..
        })
    ));
    for name in ["cycle", "wrong", "unresolved"] {
        assert!(matches!(
            document.attribute_value(document.attribute_id(operation, name).unwrap()),
            Some(AttributeValue::Invalid(_))
        ));
    }
    let huge = match document
        .attribute_value(document.attribute_id(operation, "huge").unwrap())
        .unwrap()
    {
        AttributeValue::AffineMap(map) => document.affine_map(*map).unwrap(),
        other => panic!("unexpected huge value: {other:?}"),
    };
    assert!(matches!(
        document.affine_expression(huge.results[0]),
        Some(AffineExprValue::Invalid(_))
    ));
}

#[test]
fn valid_and_forward_references_lower_to_direct_result_identity() {
    for name in ["valid.mlir", "forward.mlir"] {
        let parsed = ParsedFile::parse(fixture(name)).unwrap();
        let result = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
        assert!(
            result.diagnostics.is_empty(),
            "{name}: {:?}",
            result.diagnostics
        );
        assert!(result.semantically_complete);
        let document = result.document.unwrap();
        document.validate().unwrap();
        let operations: Vec<_> = document.operations().collect();
        assert_eq!(operations.len(), 3);
        assert_eq!(document.root_operations(), &operations[..1]);
        let make = *operations
            .iter()
            .find(|&&id| document.operation_name(id) == Some("vendor.make"))
            .unwrap();
        let consume = *operations
            .iter()
            .find(|&&id| document.operation_name(id) == Some("vendor.consume"))
            .unwrap();
        assert_eq!(
            document.operands(consume),
            Some(
                &[ValueReference::Resolved(ValueId::OperationResult {
                    operation: make,
                    result: 0
                })][..]
            )
        );
        assert_eq!(document.operation_regions(operations[0]).unwrap().len(), 1);
        let region = document.operation_regions(operations[0]).unwrap()[0];
        let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
        assert_eq!(document.block_operations(block).unwrap().len(), 2);
        let stats = document.statistics();
        assert_eq!(
            (stats.retained_source_bytes, stats.retained_cst_nodes),
            (0, 0)
        );
        assert!(stats.local_strings > 0 && stats.local_types > 0 && stats.local_attributes > 0);
        assert_eq!(SharedRegistry.statistics(), Default::default());

        let other = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
            .document
            .unwrap();
        assert!(other.operation(operations[0]).is_none());
    }
}

#[test]
fn unresolved_reference_obeys_strict_and_best_effort_contract() {
    let parsed = ParsedFile::parse(fixture("unresolved.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert_eq!(strict.diagnostics.len(), 1);
    assert!(
        strict.diagnostics[0]
            .message
            .contains("unresolved SSA value `%missing`")
    );
    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    assert!(!best.semantically_complete);
    let document = best.document.unwrap();
    document.validate().unwrap();
    assert!(!document.is_semantically_complete());
    let consume = document
        .operations()
        .find(|&id| document.operation_name(id) == Some("vendor.consume"))
        .unwrap();
    assert!(matches!(
        document.operands(consume),
        Some([ValueReference::Invalid(_)])
    ));
}

#[test]
fn document_owns_semantic_storage_after_source_is_dropped() {
    let document = {
        let parsed = ParsedFile::parse(fixture("valid.mlir")).unwrap();
        lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
            .document
            .unwrap()
    };
    assert_eq!(document.statistics().retained_source_bytes, 0);
    assert_eq!(document.statistics().retained_cst_nodes, 0);
    assert_eq!(document.operations().count(), 3);
    document.validate().unwrap();
}

#[test]
fn checked_entity_apis_reject_foreign_document_and_operation_ids() {
    let first = {
        let parsed = ParsedFile::parse(fixture("valid.mlir")).unwrap();
        lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
            .document
            .unwrap()
    };
    let second = {
        let parsed = ParsedFile::parse(fixture("valid.mlir")).unwrap();
        lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
            .document
            .unwrap()
    };
    let operations: Vec<_> = first.operations().collect();
    let make = *operations
        .iter()
        .find(|&&id| first.operation_name(id) == Some("vendor.make"))
        .unwrap();
    let consume = *operations
        .iter()
        .find(|&&id| first.operation_name(id) == Some("vendor.consume"))
        .unwrap();
    assert!(first.operation(make).unwrap().result(consume, 0).is_none());
    assert!(first.operation(make).unwrap().result(make, 0).is_some());
    let region = first.operation_regions(operations[0]).unwrap()[0];
    assert!(first.region(region).unwrap().blocks(&second).is_none());
}

#[test]
fn complete_generic_surface_lowers_with_owned_identity() {
    let parsed = ParsedFile::parse(generic_complete_fixture("valid.mlir")).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    document.validate().unwrap();

    let operation = |name| {
        document
            .operations()
            .find(|&id| document.operation_name(id) == Some(name))
            .unwrap()
    };
    let results = operation("test.results");
    let uses = operation("test.uses");
    assert_eq!(document.result_types(results).unwrap().len(), 3);
    assert_eq!(
        document.operands(uses).unwrap(),
        &[
            ValueReference::Resolved(ValueId::OperationResult {
                operation: results,
                result: 0
            }),
            ValueReference::Resolved(ValueId::OperationResult {
                operation: results,
                result: 1
            }),
            ValueReference::Resolved(ValueId::OperationResult {
                operation: results,
                result: 2
            }),
        ]
    );

    let properties = operation("test.properties");
    assert_eq!(
        document.properties(properties).unwrap().collect::<Vec<_>>(),
        [("inherent", "7")]
    );
    assert_eq!(
        document.attributes(properties).unwrap().collect::<Vec<_>>(),
        [("discardable", "\"yes\"")]
    );

    let successors = operation("test.successors");
    let edges = document.successors(successors).unwrap();
    assert_eq!(edges.len(), 3);
    assert_eq!(document.block_label(edges[0].block), Some(Some("next")));
    assert_eq!(document.successor_arguments(edges[0]).unwrap().len(), 2);
    assert_eq!(document.block_label(edges[1].block), Some(Some("empty")));
    assert_eq!(document.block_label(edges[2].block), Some(Some("exit")));

    let regions = document
        .operation_regions(operation("test.regions"))
        .unwrap();
    assert_eq!(regions.len(), 2);
    let first_blocks = document
        .region(regions[0])
        .unwrap()
        .blocks(&document)
        .unwrap();
    assert_eq!(first_blocks.len(), 4);
    assert_eq!(
        document
            .block_argument_types(first_blocks[1])
            .unwrap()
            .len(),
        2
    );
    let in_block = operation("test.in_block");
    assert_eq!(
        document.operands(in_block),
        Some(
            &[ValueReference::Resolved(ValueId::BlockArgument {
                block: first_blocks[1],
                argument: 0
            })][..]
        )
    );
    let location = document.operation_location(in_block).unwrap().unwrap();
    assert_eq!(location, "loc(\"operation\")");
    assert!(
        document.operation_source_range(in_block).unwrap().end()
            > document.operation_source_range(in_block).unwrap().start()
    );
    assert_ne!(location, "operation source span");
}

#[test]
fn generic_complete_malformed_inputs_have_component_diagnostics() {
    let parsed = ParsedFile::parse(generic_complete_fixture("malformed.mlir")).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    assert!(lowered.document.is_some());
    let messages = lowered
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>();
    for component in ["result", "successor", "property", "block", "location"] {
        assert!(
            messages.iter().any(|message| message.contains(component)),
            "missing {component}: {messages:?}"
        );
    }

    let parsed = ParsedFile::parse(generic_complete_fixture("unresolved.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(strict.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("unresolved SSA value `%pair#2`")
    }));
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("unresolved block `^missing`"))
    );
    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    assert!(!document.is_semantically_complete());
    document.validate().unwrap();
}

#[test]
fn generic_complete_adversarial_semantics_are_diagnosed_without_losing_ownership() {
    for (name, expected) in [
        ("duplicates.mlir", "duplicate property key"),
        ("successor-shape.mlir", "expects"),
        ("successor-shape.mlir", "has type"),
        ("cross-region.mlir", "unresolved block `^to`"),
    ] {
        let parsed = ParsedFile::parse(generic_complete_fixture(name)).unwrap();
        let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
        assert!(
            strict.document.is_none(),
            "{name}: {:?}",
            strict.diagnostics
        );
        assert!(
            strict
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains(expected)),
            "{name}: {:?}",
            strict.diagnostics
        );
        let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
        let document = best.document.unwrap();
        assert!(!document.is_semantically_complete());
        document.validate().unwrap();
    }

    let parsed = ParsedFile::parse(generic_complete_fixture("cross-region.mlir")).unwrap();
    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    let jump = document
        .operations()
        .find(|&id| document.operation_name(id) == Some("test.jump"))
        .unwrap();
    let edges = document.successors(jump).unwrap();
    assert_eq!(edges.len(), 1, "invalid edges must not be dropped");
    assert!(document.block_label(edges[0].block).is_none());
    document.validate().unwrap();

    let parsed = ParsedFile::parse(generic_complete_fixture("repeated-scoped.mlir")).unwrap();
    let result = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    result.document.unwrap().validate().unwrap();

    let parsed = ParsedFile::parse(generic_complete_fixture("malformed.mlir")).unwrap();
    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    assert!(best.document.is_some());
    assert!(
        best.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message == "malformed trailing location")
    );
    best.document.unwrap().validate().unwrap();

    let parsed =
        ParsedFile::parse(generic_complete_fixture("duplicate-block-arguments.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(strict.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("duplicate block argument `%same`")
    }));
    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    assert!(best.document.is_some());
    best.document.unwrap().validate().unwrap();
}

#[test]
fn duplicate_block_labels_are_diagnosed_and_leave_successors_unresolved() {
    let source = br#""test.region"() ({
  ^entry:
    "test.jump"() [^same] : () -> ()
  ^same:
    "test.return"() : () -> ()
  ^same:
    "test.return"() : () -> ()
}) : () -> ()"#;
    let parsed = ParsedFile::parse(source.to_vec()).unwrap();

    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.message == "duplicate block label `^same` in region" })
    );

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    let jump = document
        .operations()
        .find(|&operation| document.operation_name(operation) == Some("test.jump"))
        .unwrap();
    let successor = document.successors(jump).unwrap()[0];
    assert!(document.block_label(successor.block).is_none());
    document.validate().unwrap();
}

#[test]
fn block_labels_are_scoped_to_their_region() {
    let source = br#""test.regions"() ({
  ^same:
    "test.return"() : () -> ()
}, {
  ^same:
    "test.return"() : () -> ()
}) : () -> ()"#;
    let parsed = ParsedFile::parse(source.to_vec()).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);

    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    lowered.document.unwrap().validate().unwrap();
}

#[test]
fn core_values_are_structural_interned_sorted_and_retained() {
    let document = {
        let parsed = ParsedFile::parse(fixture("core-values.mlir")).unwrap();
        let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
        assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
        lowered.document.unwrap()
    };
    document.validate().unwrap();
    let operation = document.operations().next().unwrap();
    let result_types = document.result_types(operation).unwrap();
    assert_eq!(result_types[0], result_types[1]);
    assert!(
        matches!(document.type_value(result_types[0]), Some(TypeValue::Tuple(values)) if values.len() == 2)
    );

    let first = document.attribute_id(operation, "first").unwrap();
    let second = document.attribute_id(operation, "second").unwrap();
    assert_eq!(first, second);
    assert!(matches!(
        document.attribute_value(first),
        Some(AttributeValue::Dictionary(entries)) if entries.iter().map(|(name, _)| name.as_str()).eq(["a", "z"])
    ));
    assert!(matches!(
        document.operation_location_value(operation),
        Some(Some(LocationValue::FileLineColumn {
            line: 7,
            column: 4,
            ..
        }))
    ));
    assert_eq!(document.statistics().retained_source_bytes, 0);
    assert_eq!(document.statistics().retained_cst_nodes, 0);
}

#[test]
fn aliases_diagnose_cycles_duplicates_wrong_kinds_and_unresolved_names() {
    for (source, expected) in [
        (
            "!a = type !b\n!b = type !a\n%r = \"x\"() : () -> !a",
            "cyclic type alias",
        ),
        (
            "!a = type i32\n!a = type i64\n%r = \"x\"() : () -> !a",
            "duplicate alias definition",
        ),
        ("#a = 1\n%r = \"x\"() : () -> !a", "has attribute kind"),
        ("%r = \"x\"() : () -> !missing", "unresolved type alias"),
    ] {
        let parsed = ParsedFile::parse(source.as_bytes()).unwrap();
        let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
        assert!(strict.document.is_none(), "{source}");
        assert!(
            strict
                .diagnostics
                .iter()
                .any(|d| d.message.contains(expected)),
            "{source}: {:?}",
            strict.diagnostics
        );
        let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
        let document = best.document.unwrap();
        assert!(!document.is_semantically_complete());
        document.validate().unwrap();
    }
}

#[test]
fn alias_expansion_limit_applies_to_each_alias_family() {
    let cases = [
        "!a = type !b\n!b = type i32\n%r = \"type.alias\"() : () -> !a",
        "#a = #b\n#b = 1\n\"attribute.alias\"() {value = #a} : () -> ()",
        "#a = #b\n#b = affine_map<(d0) -> (d0)>\n%r = \"affine.alias\"() : () -> memref<4xi32, #a>",
        "#a = #b\n#b = loc(\"file\":1:1)\n\"location.alias\"() : () -> () loc(#a)",
    ];
    for source in cases {
        let parsed = ParsedFile::parse_with_limits(
            source.as_bytes().to_vec(),
            ParseLimits {
                max_alias_expansion_depth: 1,
                ..ParseLimits::default()
            },
        )
        .unwrap();
        let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
        assert!(strict.document.is_none(), "{source}");
        assert!(
            strict
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message == "alias expansion depth exceeds limit of 1"),
            "{source}: {:?}",
            strict.diagnostics
        );
        let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
        let document = best.document.unwrap();
        assert!(!document.is_semantically_complete(), "{source}");
        document.validate().unwrap();
    }
}

#[test]
fn nested_alias_expansion_uses_the_shared_budget() {
    let source =
        b"!a = type tensor<1xi32, #b>\n#b = #c\n#c = 1\n%r = \"nested.alias\"() : () -> !a";
    let parsed = ParsedFile::parse_with_limits(
        source.as_slice(),
        ParseLimits {
            max_alias_expansion_depth: 2,
            ..ParseLimits::default()
        },
    )
    .unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(lowered.document.is_none());
    assert!(
        lowered
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message == "alias expansion depth exceeds limit of 2")
    );
}

#[test]
fn default_alias_expansion_limit_is_64() {
    let aliases = (0..65)
        .map(|index| {
            let target = if index == 64 {
                "i32".to_owned()
            } else {
                format!("!a{}", index + 1)
            };
            format!("!a{index} = type {target}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let source = format!("{aliases}\n%r = \"default.alias\"() : () -> !a0");
    let parsed = ParsedFile::parse(source.into_bytes()).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(lowered.document.is_none());
    assert!(
        lowered
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message == "alias expansion depth exceeds limit of 64")
    );
}

#[test]
fn direct_composite_type_alias_cycle_preserves_nested_shape() {
    let source = b"!a = type tuple<!a>\n%r = \"cycle.direct\"() : () -> !a";
    let parsed = ParsedFile::parse(source.as_slice()).unwrap();

    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("cyclic type alias `!a`"))
    );

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    let operation = document.operations().next().unwrap();
    let result = document.result_types(operation).unwrap()[0];
    assert!(matches!(
        document.type_value(result),
        Some(TypeValue::Tuple(values))
            if values.len() == 1 && matches!(values[0], TypeValue::Invalid(_))
    ));
}

#[test]
fn indirect_composite_type_alias_cycle_preserves_nested_shape() {
    let source = b"!a = type !b\n!b = type tuple<!c>\n!c = type tensor<1x!a>\n%r = \"cycle.indirect\"() : () -> !a";
    let parsed = ParsedFile::parse(source.as_slice()).unwrap();

    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("cyclic type alias `!a`"))
    );

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    let operation = document.operations().next().unwrap();
    let result = document.result_types(operation).unwrap()[0];
    assert!(matches!(
        document.type_value(result),
        Some(TypeValue::Tuple(values))
            if values.len() == 1
                && matches!(
                    values.first(),
                    Some(TypeValue::Tensor { dimensions, element, .. })
                        if dimensions.len() == 1 && matches!(element.as_ref(), TypeValue::Invalid(_))
                )
    ));
}

#[test]
fn direct_function_type_alias_cycle_preserves_function_shape() {
    let source = b"!a = type (!a) -> (!a)\n%r = \"cycle.function.direct\"() : () -> !a";
    let parsed = ParsedFile::parse(source.as_slice()).unwrap();

    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("cyclic type alias `!a`"))
    );

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    let operation = document.operations().next().unwrap();
    let result = document.result_types(operation).unwrap()[0];
    assert!(matches!(
        document.type_value(result),
        Some(TypeValue::Function { inputs, results })
            if inputs.len() == 1
                && results.len() == 1
                && matches!(inputs[0], TypeValue::Invalid(_))
                && matches!(results[0], TypeValue::Invalid(_))
    ));
}

#[test]
fn indirect_function_type_alias_cycle_preserves_function_shape() {
    let source =
        b"!a = type !b\n!b = type (!a) -> (!a)\n%r = \"cycle.function.indirect\"() : () -> !a";
    let parsed = ParsedFile::parse(source.as_slice()).unwrap();

    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("cyclic type alias `!a`"))
    );

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    let operation = document.operations().next().unwrap();
    let result = document.result_types(operation).unwrap()[0];
    assert!(matches!(
        document.type_value(result),
        Some(TypeValue::Function { inputs, results })
            if inputs.len() == 1
                && results.len() == 1
                && matches!(inputs[0], TypeValue::Invalid(_))
                && matches!(results[0], TypeValue::Invalid(_))
    ));
}

#[test]
fn owned_scalar_shapes_locations_and_nested_best_effort_values_are_structural() {
    let parsed = ParsedFile::parse(fixture("core-values-adversarial.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    assert!(!document.is_semantically_complete());

    let find = |name| {
        document
            .operations()
            .find(|&id| document.operation_name(id) == Some(name))
            .unwrap()
    };
    let scalars = document.result_types(find("test.scalars")).unwrap();
    assert!(matches!(
        document.type_value(scalars[0]),
        Some(TypeValue::Integer {
            width: 16,
            signedness: Some(true)
        })
    ));
    assert!(matches!(
        document.type_value(scalars[1]),
        Some(TypeValue::Integer {
            width: 32,
            signedness: Some(false)
        })
    ));
    assert!(
        matches!(document.type_value(scalars[2]), Some(TypeValue::Float(kind)) if kind == "bf16")
    );
    assert!(
        matches!(document.type_value(scalars[3]), Some(TypeValue::Float(kind)) if kind == "f8E4M3FN")
    );

    let shapes = document.result_types(find("test.shapes")).unwrap();
    assert!(matches!(
        document.type_value(shapes[0]),
        Some(TypeValue::Tensor { unranked: true, .. })
    ));
    assert!(matches!(
        document.type_value(shapes[1]),
        Some(TypeValue::Tensor {
            encoding: Some(_),
            ..
        })
    ));
    assert!(
        matches!(document.type_value(shapes[2]), Some(TypeValue::Vector { scalable, .. }) if scalable == &vec![true, true])
    );
    assert!(
        matches!(document.type_value(shapes[3]), Some(TypeValue::MemRef { layout: Some(MemRefLayout::Opaque { spelling, .. }), memory_space: Some(space), .. }) if spelling.contains("strided") && matches!(space.as_ref(), AttributeValue::Integer(value) if value == "3"))
    );

    assert!(matches!(
        document.operation_location_value(find("test.locations")),
        Some(Some(LocationValue::CallSite { .. }))
    ));
    assert!(matches!(
        document.operation_location_value(find("test.fused")),
        Some(Some(LocationValue::Fused { metadata: Some(metadata), locations })) if metadata == "\"pass\"" && matches!(locations.first(), Some(LocationValue::Name { child: Some(_), .. }))
    ));

    let invalid = document.result_types(find("test.invalid")).unwrap();
    assert!(
        matches!(document.type_value(invalid[0]), Some(TypeValue::Tuple(values)) if matches!(values.get(1), Some(TypeValue::Invalid(_))))
    );
    let function = document.function_type(find("test.invalid")).unwrap();
    assert!(
        matches!(document.type_value(function), Some(TypeValue::Function { results, .. }) if matches!(results.get(1), Some(TypeValue::Invalid(_))))
    );
    let items = document
        .attribute_id(find("test.invalid"), "items")
        .unwrap();
    assert!(
        matches!(document.attribute_value(items), Some(AttributeValue::Array(values)) if matches!(values.get(1), Some(AttributeValue::Type(TypeValue::Invalid(_)))))
    );
}

#[test]
fn location_aliases_encodings_fused_members_and_duplicate_values_recover_structurally() {
    let parsed = ParsedFile::parse(fixture("location-aggregate-rework.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    for expected in [
        "unresolved location alias",
        "cyclic location alias",
        "expected location",
        "unresolved attribute alias",
        "duplicate dictionary key",
    ] {
        assert!(
            strict
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains(expected)),
            "missing {expected}: {:?}",
            strict.diagnostics
        );
    }

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    let find = |name| {
        document
            .operations()
            .find(|&id| document.operation_name(id) == Some(name))
            .unwrap()
    };
    let ok = find("test.location_alias");
    assert!(
        matches!(document.operation_location_value(ok), Some(Some(LocationValue::Name { name, .. })) if name == "\"aliased\"")
    );
    let value = document
        .attribute_value(document.attribute_id(ok, "value").unwrap())
        .unwrap();
    assert!(
        matches!(value, AttributeValue::Location(LocationValue::Name { name, .. }) if name == "\"aliased\"")
    );

    let nested = find("test.location_nested_aliases");
    assert!(
        matches!(document.operation_location_value(nested), Some(Some(LocationValue::Fused { locations, .. })) if locations.len() == 2 && matches!(locations.first(), Some(LocationValue::Name { name, .. }) if name == "\"aliased\"") && matches!(locations.get(1), Some(LocationValue::CallSite { callee, .. }) if matches!(callee.as_ref(), LocationValue::Name { name, .. } if name == "\"aliased\""))),
        "{:?}",
        document.operation_location_value(nested)
    );
    let nested_value = document
        .attribute_value(document.attribute_id(nested, "value").unwrap())
        .unwrap();
    assert!(
        matches!(nested_value, AttributeValue::Location(LocationValue::Name { child: Some(child), .. }) if matches!(child.as_ref(), LocationValue::Name { name, .. } if name == "\"aliased\"")),
        "{:?}",
        (nested_value, document.diagnostics())
    );

    let bad = find("test.location_bad");
    assert!(
        matches!(document.operation_location_value(bad), Some(Some(LocationValue::Fused { locations, .. })) if locations.len() == 3 && matches!(locations.get(1), Some(LocationValue::Invalid(_))))
    );
    let encoding = match document
        .type_value(document.result_types(bad).unwrap()[0])
        .unwrap()
    {
        TypeValue::Tensor {
            encoding: Some(encoding),
            ..
        } => encoding,
        other => panic!("unexpected type: {other:?}"),
    };
    assert!(matches!(encoding.as_ref(), AttributeValue::Invalid(_)));
    for name in ["wrong", "missing", "wrong_type"] {
        assert!(matches!(
            document.attribute_value(document.attribute_id(bad, name).unwrap()),
            Some(AttributeValue::Location(LocationValue::Invalid(_)))
        ));
    }

    let nested_bad = find("test.location_nested_bad");
    assert!(
        matches!(document.operation_location_value(nested_bad), Some(Some(LocationValue::Fused { locations, .. })) if locations.len() == 3 && matches!(locations.first(), Some(LocationValue::Invalid(_))) && matches!(locations.get(1), Some(LocationValue::CallSite { callee, .. }) if matches!(callee.as_ref(), LocationValue::Invalid(_))) && matches!(locations.get(2), Some(LocationValue::Name { child: Some(child), .. }) if matches!(child.as_ref(), LocationValue::Invalid(_))))
    );

    let duplicate = document
        .attribute_value(
            document
                .attribute_id(find("test.duplicate_aggregate"), "value")
                .unwrap(),
        )
        .unwrap();
    assert!(
        matches!(duplicate, AttributeValue::Dictionary(entries) if entries.len() == 2 && matches!(entries.get(1), Some((_, AttributeValue::Invalid(_)))))
    );
}

#[test]
fn forward_aliases_resolve_through_nested_locations() {
    let bytes = br#"%0 = "test.forward"() {value = #later_attr} : () -> !later_type loc(fused[#later_loc, callsite(#later_loc at "caller"(#later_loc))])
!later_type = type i32
#later_attr = 7
#later_loc = loc("resolved")
"#;
    let parsed = ParsedFile::parse(bytes.as_slice()).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let operation = document.operations().next().unwrap();
    assert!(matches!(
        document
            .result_types(operation)
            .and_then(|values| values.first())
            .and_then(|id| document.type_value(*id)),
        Some(TypeValue::Integer { .. })
    ));
    let value = document.attribute_value(document.attribute_id(operation, "value").unwrap());
    assert!(
        matches!(value, Some(AttributeValue::Integer(_))),
        "{value:?}"
    );
    assert!(
        matches!(document.operation_location_value(operation), Some(Some(LocationValue::Fused { locations, .. })) if matches!(locations.first(), Some(LocationValue::Name { name, .. }) if name == "\"resolved\"") && matches!(locations.get(1), Some(LocationValue::CallSite { callee, caller }) if matches!(callee.as_ref(), LocationValue::Name { .. }) && matches!(caller.as_ref(), LocationValue::Name { child: Some(child), .. } if matches!(child.as_ref(), LocationValue::Name { .. }))))
    );
}

#[test]
fn unresolved_nested_forward_location_preserves_surrounding_operations() {
    let bytes = br#""before"() : () -> ()
"nested"() : () -> () loc(callsite(#missing at "caller"(#missing)))
"after"() : () -> ()
"#;
    let parsed = ParsedFile::parse(bytes.as_slice()).unwrap();
    let lowered = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    assert!(lowered.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("unresolved location alias `#missing`")
    }));
    let document = lowered.document.unwrap();
    assert_eq!(document.operations().count(), 3);
    let nested = document
        .operations()
        .find(|&id| document.operation_name(id) == Some("nested"))
        .unwrap();
    assert!(
        matches!(document.operation_location_value(nested), Some(Some(LocationValue::CallSite { callee, caller })) if matches!(callee.as_ref(), LocationValue::Invalid(_)) && matches!(caller.as_ref(), LocationValue::Name { child: Some(child), .. } if matches!(child.as_ref(), LocationValue::Invalid(_))))
    );
}

#[test]
fn memref_parameters_resolve_aliases_and_retain_invalid_nested_values() {
    let parsed = ParsedFile::parse(fixture("memref-parameters-rework.mlir")).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("memref layout"))
    );
    assert!(
        strict
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("memory space"))
    );

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    let document = best.document.unwrap();
    document.validate().unwrap();
    let find = |name| {
        document
            .operations()
            .find(|&id| document.operation_name(id) == Some(name))
            .unwrap()
    };
    let valid = match document
        .type_value(document.result_types(find("test.memref.valid")).unwrap()[0])
        .unwrap()
    {
        TypeValue::MemRef {
            layout: Some(MemRefLayout::Attribute(layout)),
            memory_space: Some(space),
            ..
        } => (layout, space),
        other => panic!("unexpected valid memref: {other:?}"),
    };
    assert!(matches!(valid.0.as_ref(), AttributeValue::Integer(value) if value == "7"));
    assert!(matches!(valid.1.as_ref(), AttributeValue::Integer(value) if value == "4"));

    let invalid = document.result_types(find("test.memref.invalid")).unwrap();
    assert!(
        matches!(document.type_value(invalid[0]), Some(TypeValue::MemRef { layout: Some(MemRefLayout::Invalid(_)), memory_space: Some(space), .. }) if matches!(space.as_ref(), AttributeValue::Invalid(_)))
    );
    assert!(
        matches!(document.type_value(invalid[1]), Some(TypeValue::MemRef { memory_space: Some(space), .. }) if matches!(space.as_ref(), AttributeValue::Invalid(_)))
    );
    assert!(
        matches!(document.type_value(invalid[2]), Some(TypeValue::MemRef { layout: Some(MemRefLayout::Opaque { parameters, .. }), .. }) if parameters.iter().any(|parameter| matches!(parameter, AttributeValue::Invalid(_))))
    );
}

#[test]
fn large_values_are_blob_backed_and_retention_profiles_are_exact() {
    let bytes = include_bytes!("../../../tests/corpus/mlir-22.1/payload-opaque/valid.mlir");
    let parsed = ParsedFile::parse(bytes.as_slice()).unwrap();
    for profile in [
        RetentionProfile::SyntaxOnly,
        RetentionProfile::SemanticOnly,
        RetentionProfile::Hybrid,
    ] {
        let lowered = lower_proving_fixture_with_retention(
            &parsed,
            LoweringMode::Strict,
            profile,
            &SharedRegistry,
        );
        assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
        let document = lowered.document.unwrap();
        document.validate().unwrap();
        let stats = document.statistics();
        assert!(stats.payload_blobs >= 5);
        assert!(stats.payload_blob_bytes > 200);
        match profile {
            RetentionProfile::SyntaxOnly => {
                assert_eq!(stats.retained_source_bytes, bytes.len());
                assert!(stats.retained_cst_nodes > 0);
                assert_eq!(stats.retained_mapping_entries, 0);
            }
            RetentionProfile::SemanticOnly => {
                assert_eq!(stats.retained_source_bytes, 0);
                assert_eq!(stats.retained_cst_nodes, 0);
                assert_eq!(stats.retained_mapping_entries, 0);
            }
            RetentionProfile::Hybrid => {
                assert_eq!(stats.retained_source_bytes, bytes.len());
                assert!(stats.retained_cst_nodes > 0);
                assert_eq!(stats.retained_mapping_entries, stats.operations);
                for operation in document.operations() {
                    assert!(document.operation_syntax_range(operation).is_some());
                }
            }
        }

        let operation = document.operations().next().unwrap();
        let dense =
            document.attribute_value(document.attribute_id(operation, "dense_value").unwrap());
        let sparse =
            document.attribute_value(document.attribute_id(operation, "sparse_value").unwrap());
        let resource =
            document.attribute_value(document.attribute_id(operation, "resource_value").unwrap());
        assert!(matches!(
            dense,
            Some(AttributeValue::Large(LargeAttributeValue::Dense(_)))
        ));
        assert!(matches!(
            sparse,
            Some(AttributeValue::Large(LargeAttributeValue::Sparse(_)))
        ));
        assert!(matches!(
            resource,
            Some(AttributeValue::Large(LargeAttributeValue::Resource(_)))
        ));
        assert!(matches!(
            document.attribute_value(document.attribute_id(operation, "wide").unwrap()),
            Some(AttributeValue::WideNumber(_))
        ));
        assert!(matches!(
            document.attribute_value(document.attribute_id(operation, "opaque").unwrap()),
            Some(AttributeValue::Opaque(_))
        ));
    }
}

#[test]
fn retained_source_and_syntax_are_shared_and_outlive_either_owner() {
    let bytes = b"\"known\"() : () -> ()".as_slice();
    let parsed = ParsedFile::parse(bytes).unwrap();
    let exact_cst_bytes = parsed.syntax().tree().exact_retained_bytes();
    let document = lower_proving_fixture_with_retention(
        &parsed,
        LoweringMode::Strict,
        RetentionProfile::Hybrid,
        &SharedRegistry,
    )
    .document
    .unwrap();
    let shared = document.statistics();
    assert!(shared.source_storage_shared);
    assert!(shared.cst_storage_shared);
    assert_eq!(shared.retained_cst_bytes, exact_cst_bytes);
    assert!(shared.direct_owned_bytes > 0);
    assert!(shared.document_index_bytes > 0);

    drop(parsed);
    assert_eq!(document.source_bytes(), Some(bytes));
    assert_eq!(
        document.syntax_tree().unwrap().exact_retained_bytes(),
        exact_cst_bytes
    );
    assert!(!document.statistics().source_storage_shared);
    assert!(!document.statistics().cst_storage_shared);

    let parsed = ParsedFile::parse(bytes).unwrap();
    let document = lower_proving_fixture_with_retention(
        &parsed,
        LoweringMode::Strict,
        RetentionProfile::Hybrid,
        &SharedRegistry,
    )
    .document
    .unwrap();
    drop(document);
    parsed.syntax().tree().verify().unwrap();
    assert_eq!(parsed.original_bytes(), bytes);
}

#[test]
fn malformed_large_values_follow_the_shared_recovery_contract() {
    let bytes = include_bytes!("../../../tests/corpus/mlir-22.1/payload-opaque/malformed.mlir");
    let parsed = ParsedFile::parse(bytes.as_slice()).unwrap();
    let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
    assert!(strict.document.is_none());
    assert!(!strict.diagnostics.is_empty());

    let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
    assert!(!best.semantically_complete);
    let document = best.document.unwrap();
    assert!(!document.is_semantically_complete());
    document.validate().unwrap();
    assert_eq!(
        document.operations().count(),
        4,
        "following operations survive recovery"
    );
}

#[test]
fn each_owned_payload_family_recovers_as_an_invalid_sentinel() {
    for value in [
        "dense<[1, 2} : tensor<2xi32>",
        "sparse<[[0, 1]], [2.0}> : tensor<2x2xf32>",
        "dense_resource<handle} : tensor<2xi32>",
        "0x : i128",
        "#vendor.attr<[1, 2}",
    ] {
        let source = format!("\"bad\"() {{value = {value}}} : () -> ()");
        let parsed = ParsedFile::parse(source.into_bytes()).unwrap();
        let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
        assert!(
            strict.document.is_none(),
            "{value}: {:?}",
            strict.diagnostics
        );
        assert!(!strict.diagnostics.is_empty(), "{value}");

        let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
        let document = best.document.unwrap();
        assert!(!document.is_semantically_complete(), "{value}");
        document.validate().unwrap();
        let operation = document.operations().next().unwrap();
        let attribute = document.attribute_id(operation, "value").unwrap();
        assert!(
            matches!(
                document.attribute_value(attribute),
                Some(AttributeValue::Invalid(_))
            ),
            "{value}: {:?}",
            document.attribute_value(attribute)
        );
    }
}

#[test]
fn opaque_payload_preserves_exact_quoted_body_bytes() {
    let source = include_bytes!("../../../tests/corpus/mlir-22.1/payload-opaque/valid.mlir");
    let parsed = ParsedFile::parse(source.as_slice()).unwrap();
    let document = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
        .document
        .unwrap();
    let operation = document.operations().next().unwrap();
    let value = document
        .attribute_value(document.attribute_id(operation, "opaque").unwrap())
        .unwrap();
    let AttributeValue::Opaque(bytes) = value else {
        panic!("unexpected opaque value: {value:?}");
    };
    assert_eq!(
        bytes.as_ref(),
        b"#vendor.attr<{nested = [\"literal > } ]\", \"escaped \\\\22 quote\", (1, 2)]}>"
    );
    let function_type = document.function_type(operation).unwrap();
    let Some(TypeValue::Function { inputs, .. }) = document.type_value(function_type) else {
        panic!("unexpected opaque type");
    };
    let TypeValue::Opaque(bytes) = &inputs[0] else {
        panic!("unexpected opaque function input");
    };
    assert_eq!(
        bytes.as_ref(),
        b"!vendor.type<{nested = [\"literal > } ]\", (1, 2)]}>"
    );
}

#[test]
fn wide_number_grammar_rejects_empty_or_unprefixed_hex_payloads() {
    for value in ["0x : i128", "0xgg : i128", "abcdef : i128", "0x12 : nope"] {
        let source = format!("\"wide\"() {{value = {value}}} : () -> ()");
        let parsed = ParsedFile::parse(source.into_bytes()).unwrap();
        let strict = lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry);
        assert!(strict.document.is_none(), "{value}");
        assert!(strict.diagnostics.iter().any(|diagnostic| {
            diagnostic.message.contains("unsupported") || diagnostic.message.contains("malformed")
        }));
        let best = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry);
        let document = best.document.unwrap();
        assert!(matches!(
            document.attribute_value(
                document
                    .attribute_id(document.operations().next().unwrap(), "value")
                    .unwrap()
            ),
            Some(AttributeValue::Invalid(_))
        ));
        document.validate().unwrap();
    }
}
