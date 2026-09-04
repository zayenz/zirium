#![no_main]

use libfuzzer_sys::fuzz_target;
use zirium::{
    dialect::DialectRegistry,
    parser::{ParseFileError, ParseLimits, ParsedFile},
    semantic::{Document, LoweringMode, ValueReference, lower_with_dialect_registry},
};

const LIMITS: ParseLimits = ParseLimits {
    max_file_bytes: 4096,
    max_tokens: 256,
    max_delimiter_depth: 8,
    max_payload_bytes: 512,
    max_numeric_literal_bytes: 32,
    max_attribute_depth: 8,
    max_alias_expansion_depth: 8,
};

fuzz_target!(|data: &[u8]| {
    let registry = DialectRegistry::proving();
    let parsed = match ParsedFile::parse_with_limits_and_registry(data, LIMITS, registry) {
        Ok(parsed) => parsed,
        Err(ParseFileError::ResourceLimit(_)) if data.len() > LIMITS.max_file_bytes => return,
        Err(error) => panic!("unexpected parse failure: {error}"),
    };

    for mode in [LoweringMode::Strict, LoweringMode::BestEffort] {
        let lowered = lower_with_dialect_registry(&parsed, mode, registry);
        if let Some(document) = lowered.document {
            inspect_document(&document, registry);
        }
    }
});

fn inspect_document(document: &Document, registry: &DialectRegistry) {
    document.validate_structure().unwrap();
    let _ = document.verify_semantics(registry);
    let _ = document.statistics();
    let _ = document.diagnostics();
    let _ = document.symbol_index_diagnostics(registry);

    for operation in document.operations() {
        document.check_operation(operation).unwrap();
        let _ = document.operation_name(operation);
        let _ = document.operation_source_range(operation);
        let _ = document.operation_is_unparsed(operation);
        let _ = document.operation_unparsed_text(operation);
        let _ = document.operation_location(operation);
        let _ = document.operation_location_value(operation);
        let _ = document.attributes(operation).map(Iterator::count);
        let _ = document.properties(operation).map(Iterator::count);
        let _ = document.checked_lookup_symbol(operation, "missing", registry);

        if let Some(types) = document.result_types(operation) {
            for &ty in types {
                let _ = document.type_spelling(ty);
                let _ = document.type_value(ty);
            }
        }
        if let Some(operands) = document.operands(operation) {
            for &operand in operands {
                if let ValueReference::Resolved(value) = operand {
                    let _ = document.check_value(value);
                    let _ = document.value_key(value);
                    let _ = document.checked_uses(value);
                    let _ = document.checked_dominates(value, operation, registry);
                }
            }
        }
        if let Some(successors) = document.successors(operation) {
            for &successor in successors {
                let _ = document.block(successor.block());
                let _ = document.successor_arguments(successor);
            }
        }
        if let Some(regions) = document.operation_regions(operation) {
            for &region in regions {
                let Some(region) = document.region(region) else {
                    continue;
                };
                let _ = region.parent_operation();
                let Some(blocks) = region.blocks(document) else {
                    continue;
                };
                for &block in blocks {
                    let _ = document.block(block);
                    let _ = document.block_label(block);
                    let _ = document.block_argument_types(block);
                    let _ = document.block_operations(block);
                }
            }
        }
    }
}
