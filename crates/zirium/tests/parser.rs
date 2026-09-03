use std::{fs, path::PathBuf, sync::Arc};

use zirium::{
    NodeId, SyntaxKind,
    lexer::lex,
    parser::{
        ParseDiagnosticKind, ParseFileError, ParseLimits, ParsedFile, ParserLimits, TextEdit,
        parse_brace_fixture, parse_generic_operations, parse_generic_operations_with_limits,
    },
    source::{Source, TextRange},
};

fn corpus(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/generic-proving")
            .join(name),
    )
    .unwrap()
}

#[test]
fn owning_parse_limits_are_fatal_only_for_file_size() {
    let bytes = b"\"after\"() : () -> ()";
    assert!(matches!(
        ParsedFile::parse_with_limits(
            bytes.as_slice(),
            ParseLimits {
                max_file_bytes: bytes.len() - 1,
                ..ParseLimits::default()
            }
        ),
        Err(ParseFileError::ResourceLimit(_))
    ));

    let parsed = ParsedFile::parse_with_limits(
        bytes.as_slice(),
        ParseLimits {
            max_tokens: 1,
            ..ParseLimits::default()
        },
    )
    .unwrap();
    assert_eq!(parsed.original_bytes(), bytes);
    assert!(
        parsed
            .lexer_diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == zirium::lexer::DiagnosticKind::TokenLimit)
    );
    parsed.syntax().tree().verify().unwrap();
}

#[test]
fn nested_attribute_depth_limit_recovers_following_operation_losslessly() {
    let nested = (0..32).fold("1".to_owned(), |value, depth| {
        if depth % 2 == 0 {
            format!("[{value}]")
        } else {
            format!("{{k = {value}}}")
        }
    });
    let bytes =
        format!("\"deep\"() {{a = {nested}}} : () -> ()\n\"after\"() : () -> ()").into_bytes();
    let parsed = ParsedFile::parse_with_limits(
        bytes.as_slice(),
        ParseLimits {
            max_delimiter_depth: 4,
            ..ParseLimits::default()
        },
    )
    .unwrap();
    assert_eq!(parsed.original_bytes(), bytes);
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::DepthLimit)
    );
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    parsed.syntax().tree().verify().unwrap();
}

#[test]
fn nested_location_depth_limit_recovers_following_operation_losslessly() {
    let nested = "(".repeat(32) + "unknown" + &")".repeat(32);
    let bytes = format!("\"deep\"() : () -> () loc({nested})\n\"after\"() : () -> ()").into_bytes();
    let parsed = ParsedFile::parse_with_limits(
        bytes.as_slice(),
        ParseLimits {
            max_delimiter_depth: 4,
            ..ParseLimits::default()
        },
    )
    .unwrap();
    assert_eq!(parsed.original_bytes(), bytes);
    assert!(
        parsed
            .syntax()
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::DepthLimit)
    );
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    parsed.syntax().tree().verify().unwrap();
}

#[test]
fn byte_text_edits_reparse_without_mutating_the_original() {
    let original = ParsedFile::parse(b"\xff\n\"old\"() : () -> ()".as_slice()).unwrap();
    let edited = original
        .apply_text_edits(&[TextEdit {
            range: TextRange::new(3, 6).unwrap(),
            replacement: Arc::from(b"new".as_slice()),
        }])
        .unwrap();
    assert_eq!(original.original_bytes(), b"\xff\n\"old\"() : () -> ()");
    assert_eq!(edited.original_bytes(), b"\xff\n\"new\"() : () -> ()");
    edited.syntax().tree().verify().unwrap();
}

fn core_corpus(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/core-syntax")
            .join(name),
    )
    .unwrap()
}

fn shaped_affine_corpus(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/shaped-affine")
            .join(name),
    )
    .unwrap()
}

fn payload_corpus(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/payload-opaque")
            .join(name),
    )
    .unwrap()
}

fn generic_complete_corpus(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/generic-complete")
            .join(name),
    )
    .unwrap()
}

fn recovery_custom_corpus(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/recovery-custom")
            .join(name),
    )
    .unwrap()
}

fn parse_bytes_with_limits(bytes: &[u8], limits: ParserLimits) -> zirium::parser::ParsedSyntax {
    let source = Source::new(bytes.to_vec()).unwrap();
    parse_generic_operations_with_limits(&lex(&source), limits).unwrap()
}

fn node_text<'a>(tree: &zirium::SyntaxTree, source: &'a Source, node: NodeId) -> &'a [u8] {
    source.slice(tree.text_range(node).unwrap()).unwrap()
}

fn reconstruct(tree: &zirium::SyntaxTree, source: &Source) -> Vec<u8> {
    tree.tokens(tree.root())
        .unwrap()
        .iter()
        .flat_map(|token| source.slice(token.range()).unwrap())
        .copied()
        .collect()
}

#[test]
fn nested_errors_propagate_without_tainting_siblings() {
    let source = Source::new(&b"{} {{"[..]).unwrap();
    let tree = parse_brace_fixture(&lex(&source)).unwrap();
    let children: Vec<_> = tree.children(tree.root()).unwrap().collect();
    assert_eq!(children.len(), 2);
    assert_eq!(tree.has_error(children[0]), Some(false));
    assert_eq!(tree.kind(children[1]), Some(SyntaxKind::Region));
    assert_eq!(tree.has_local_error(children[1]), Some(true));
    assert_eq!(tree.has_error(tree.root()), Some(true));
}

#[test]
fn unmatched_close_is_an_error_node() {
    let source = Source::new(&b"}"[..]).unwrap();
    let tree = parse_brace_fixture(&lex(&source)).unwrap();
    let error = tree.children(tree.root()).unwrap().next().unwrap();
    assert_eq!(tree.kind(error), Some(SyntaxKind::Error));
    assert_eq!(tree.has_local_error(error), Some(true));
    tree.verify().unwrap();
}

#[test]
fn proving_fixture_has_narrow_typed_structure() {
    let bytes = corpus("valid.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let lexed = lex(&source);
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert!(lexed.diagnostics().is_empty());
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    parsed.tree().verify().unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);

    let file = parsed.file();
    let operations: Vec<_> = file.operations().collect();
    assert_eq!(operations.len(), 3);
    let operation_text: Vec<_> = operations
        .iter()
        .map(|operation| node_text(parsed.tree(), &source, operation.id()))
        .collect();
    assert!(operation_text[0].starts_with(b"\"builtin.module\""));
    assert!(operation_text[1].starts_with(b"%0 = \"vendor.make\""));
    assert!(operation_text[2].starts_with(b"\"vendor.consume\""));

    let texts = |kind| {
        file.nodes(kind)
            .map(|id| node_text(parsed.tree(), &source, id))
            .collect::<Vec<_>>()
    };
    assert_eq!(texts(SyntaxKind::Result), vec![b"%0 =".as_slice()]);
    assert_eq!(texts(SyntaxKind::Operand), vec![b"%0".as_slice()]);
    assert_eq!(texts(SyntaxKind::Attribute).len(), 1);
    assert!(texts(SyntaxKind::Attribute)[0].starts_with(b"tag = #vendor.tag<\"x\">"));
    assert_eq!(
        texts(SyntaxKind::OpaqueAttribute),
        vec![b"#vendor.tag<\"x\">".as_slice()]
    );
    assert_eq!(
        texts(SyntaxKind::OpaqueType),
        vec![
            b"!vendor.token<\"x\">".as_slice(),
            b"!vendor.token<\"x\">".as_slice()
        ]
    );
    assert_eq!(texts(SyntaxKind::FunctionType).len(), 3);
    let regions: Vec<_> = file.regions().collect();
    assert_eq!(regions.len(), 1);
    assert!(regions[0].implicit_block().is_some());
}

#[test]
fn contextual_type_and_affine_words_remain_valid_dictionary_keys() {
    let bytes = b"\"keys\"() {x86 = 1, tensor = 2, vector = 3, memref = 4, mod = 5, floordiv = 6, ceildiv = 7} : () -> ()";
    let source = Source::new(bytes.as_slice()).unwrap();
    let lexed = lex(&source);
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert!(lexed.diagnostics().is_empty());
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert_eq!(parsed.file().nodes(SyntaxKind::Attribute).count(), 7);
}

#[test]
fn type_spelling_tokens_remain_valid_affine_identifiers() {
    let bytes = b"\"affine.type_names\"() {map = affine_map<(i32, f32, index) -> (i32 + f32 + index)>} : () -> ()";
    let source = Source::new(bytes.as_slice()).unwrap();
    let lexed = lex(&source);
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert!(lexed.diagnostics().is_empty());
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert_eq!(parsed.file().nodes(SyntaxKind::AffineMap).count(), 1);
}

#[test]
fn spaced_and_compact_shaped_dimensions_are_lossless_and_verified() {
    let bytes = b"\"dimensions\"() : (tensor<2 x 3 x f32>, tensor<2x3xf32>, vector<[4]x8xf32>) -> vector<[4] x 8 x f32>";
    let source = Source::new(bytes.as_slice()).unwrap();
    let lexed = lex(&source);
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert!(lexed.diagnostics().is_empty());
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert_eq!(parsed.file().nodes(SyntaxKind::ShapedDimension).count(), 8);
}

#[test]
fn malformed_corpus_is_lossless_verified_and_recovers_operations() {
    for name in [
        "unterminated-opaque.mlir",
        "malformed-attribute.mlir",
        "malformed-signature.mlir",
        "truncated-operation.mlir",
    ] {
        let bytes = corpus(name);
        let source = Source::new(bytes.clone()).unwrap();
        let parsed = parse_generic_operations(&lex(&source)).unwrap();
        assert_eq!(reconstruct(parsed.tree(), &source), bytes, "{name}");
        parsed.tree().verify().unwrap();
        assert!(
            !parsed.diagnostics().is_empty()
                || parsed.tree().has_error(parsed.tree().root()) == Some(true),
            "{name}"
        );
        let operations: Vec<_> = parsed.file().operations().collect();
        assert!(operations.len() >= 2, "{name}");
        assert!(
            node_text(parsed.tree(), &source, operations[0].id())
                .windows(b"\"builtin.module\"".len())
                .any(|window| window == b"\"builtin.module\"")
        );
    }
}

#[test]
fn core_scalar_alias_symbol_collection_and_location_views_are_lossless() {
    let bytes = core_corpus("valid.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let lexed = lex(&source);
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert!(lexed.diagnostics().is_empty());
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    parsed.tree().verify().unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);

    let file = parsed.file();
    let texts = |kind| {
        file.nodes(kind)
            .map(|id| node_text(parsed.tree(), &source, id))
            .collect::<Vec<_>>()
    };
    for kind in [
        SyntaxKind::IntegerType,
        SyntaxKind::FloatType,
        SyntaxKind::IndexType,
        SyntaxKind::IntegerAttribute,
        SyntaxKind::FloatAttribute,
        SyntaxKind::StringAttribute,
        SyntaxKind::TypeAttribute,
        SyntaxKind::AttributeAlias,
        SyntaxKind::TypeAlias,
        SyntaxKind::SymbolReference,
        SyntaxKind::ArrayAttribute,
        SyntaxKind::DictionaryAttribute,
        SyntaxKind::LocationAttribute,
        SyntaxKind::UnknownLocation,
        SyntaxKind::FileLineColLocation,
        SyntaxKind::NameLocation,
        SyntaxKind::CallSiteLocation,
        SyntaxKind::FusedLocation,
    ] {
        assert!(!texts(kind).is_empty(), "missing {kind:?}");
    }
    assert_eq!(
        texts(SyntaxKind::AttributeAlias),
        vec![b"#attr_alias".as_slice()]
    );
    assert_eq!(
        texts(SyntaxKind::SymbolReference),
        vec![b"@root::@leaf".as_slice()]
    );
    assert_eq!(
        texts(SyntaxKind::TypeAlias),
        vec![b"!type_alias".as_slice(), b"!type_alias".as_slice()]
    );
    assert!(
        texts(SyntaxKind::StringAttribute)
            .iter()
            .any(|text| *text == b"\"s\" : i32")
    );
    assert!(
        texts(SyntaxKind::FloatAttribute)
            .iter()
            .any(|text| *text == b"0x7FC00000 : f32")
    );
}

#[test]
fn malformed_core_values_recover_at_enclosing_boundaries() {
    let bytes = core_corpus("malformed.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert!(!parsed.diagnostics().is_empty());
    let operation_texts = parsed
        .file()
        .operations()
        .map(|operation| node_text(parsed.tree(), &source, operation.id()))
        .collect::<Vec<_>>();
    assert_eq!(operation_texts.len(), 5);
    assert!(
        operation_texts[0]
            .windows(4)
            .any(|window| window == b"kept")
    );
    assert!(
        operation_texts[1]
            .windows(5)
            .any(|window| window == b"after")
    );
    assert!(
        operation_texts[2]
            .windows(4)
            .any(|window| window == b"kept")
    );
    assert!(
        operation_texts[3]
            .windows(4)
            .any(|window| window == b"kept")
    );
    assert!(operation_texts[4].starts_with(b"\"core.after\""));
}

#[test]
fn shaped_and_affine_fixture_has_typed_lossless_structure_and_precedence() {
    let bytes = shaped_affine_corpus("valid.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let lexed = lex(&source);
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert!(lexed.diagnostics().is_empty(), "{:?}", lexed.diagnostics());
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    parsed.tree().verify().unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);

    let file = parsed.file();
    let function_types = file.function_types().collect::<Vec<_>>();
    assert_eq!(function_types.len(), 1);
    assert_eq!(
        function_types[0].tree().kind(function_types[0].id()),
        Some(SyntaxKind::FunctionType)
    );
    let shaped = file.shaped_types().collect::<Vec<_>>();
    assert_eq!(shaped.len(), 7);
    assert!(shaped.iter().all(|ty| ty.tree().kind(ty.id()).is_some()));
    let affine = file.affine_values().collect::<Vec<_>>();
    assert_eq!(affine.len(), 4);
    assert!(
        affine
            .iter()
            .all(|value| value.tree().kind(value.id()).is_some())
    );

    let expressions = file
        .nodes(SyntaxKind::AffineExpression)
        .map(|id| node_text(parsed.tree(), &source, id))
        .collect::<Vec<_>>();
    assert!(expressions.contains(&b"d0 + s0 * 2".as_slice()));
    assert!(expressions.contains(&b"s0 * 2".as_slice()));
    assert!(expressions.contains(&b"(d0 + s0) * 2".as_slice()));
    assert!(expressions.contains(&b"- (d0 + d1)".as_slice()));
    assert!(expressions.contains(&b"loc".as_slice()));
    assert_eq!(file.nodes(SyntaxKind::AffineConstraint).count(), 2);
    assert_eq!(file.nodes(SyntaxKind::ShapedDimension).count(), 9);
    assert_eq!(file.nodes(SyntaxKind::TensorEncoding).count(), 1);
    assert_eq!(file.nodes(SyntaxKind::MemRefLayout).count(), 3);
    assert_eq!(file.nodes(SyntaxKind::MemRefMemorySpace).count(), 3);
    assert_eq!(file.nodes(SyntaxKind::StridedLayout).count(), 1);
}

#[test]
fn shaped_affine_malformed_and_depth_inputs_recover_and_verify() {
    for name in ["malformed.mlir", "depth-limit.mlir"] {
        let bytes = shaped_affine_corpus(name);
        let source = Source::new(bytes.clone()).unwrap();
        let parsed = parse_generic_operations(&lex(&source)).unwrap();
        assert_eq!(reconstruct(parsed.tree(), &source), bytes, "{name}");
        parsed.tree().verify().unwrap();
        assert!(!parsed.diagnostics().is_empty(), "{name}");
    }

    let bytes = shaped_affine_corpus("malformed.mlir");
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    let operations = parsed.file().operations().collect::<Vec<_>>();
    assert_eq!(operations.len(), 3);
    assert!(node_text(parsed.tree(), &source, operations[2].id()).starts_with(b"\"affine.after\""));
    assert_eq!(parsed.tree().has_error(parsed.tree().root()), Some(true));
}

#[test]
fn payload_and_opaque_fixture_is_lossless_flat_and_range_backed() {
    let bytes = payload_corpus("valid.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();

    let payloads = parsed.file().payload_attributes().collect::<Vec<_>>();
    assert_eq!(payloads.len(), 3);
    assert_eq!(
        source.slice(payloads[0].payload_range().unwrap()).unwrap(),
        b"[1, 2, 3]"
    );
    assert_eq!(
        source.slice(payloads[2].handle_range().unwrap()).unwrap(),
        b"resource_handle"
    );
    assert_eq!(
        source.slice(payloads[2].type_range().unwrap()).unwrap(),
        b"tensor<4xi32>"
    );
    let wide = parsed.file().wide_numbers().next().unwrap();
    assert_eq!(
        source.slice(wide.literal_range().unwrap()).unwrap(),
        b"0x1234567890abcdef1234567890abcdef"
    );
    assert_eq!(source.slice(wide.type_range().unwrap()).unwrap(), b"i128");
    assert_eq!(parsed.file().opaque_bodies().count(), 2);
}

#[test]
fn payload_node_counts_do_not_depend_on_element_count() {
    fn counts(elements: usize) -> (usize, usize) {
        let payload = (0..elements).map(|_| "1").collect::<Vec<_>>().join(",");
        let bytes = format!(
            "\"dense\"() {{x = dense<[{}]> : tensor<{}xi32>}} : () -> ()",
            payload, elements
        );
        let parsed = parse_bytes_with_limits(bytes.as_bytes(), ParserLimits::default());
        (
            parsed.tree().node_count(),
            parsed
                .file()
                .nodes(SyntaxKind::DenseElementsAttribute)
                .count(),
        )
    }
    assert_eq!(counts(1), counts(1024));
}

#[test]
fn parser_limits_diagnose_without_losing_bytes_or_following_operations() {
    for limits in [
        ParserLimits {
            max_delimiter_depth: 2,
            ..ParserLimits::default()
        },
        ParserLimits {
            max_payload_bytes: 8,
            ..ParserLimits::default()
        },
        ParserLimits {
            max_numeric_literal_bytes: 8,
            ..ParserLimits::default()
        },
    ] {
        let bytes = payload_corpus("oversized.mlir");
        let source = Source::new(bytes.clone()).unwrap();
        let parsed = parse_generic_operations_with_limits(&lex(&source), limits).unwrap();
        assert_eq!(reconstruct(parsed.tree(), &source), bytes);
        parsed.tree().verify().unwrap();
        assert!(!parsed.diagnostics().is_empty());
        assert_eq!(parsed.file().operations().count(), 2);
    }
}

#[test]
fn malformed_payloads_recover_at_enclosing_boundaries() {
    let bytes = payload_corpus("malformed.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert!(!parsed.diagnostics().is_empty());
    let operations = parsed.file().operations().collect::<Vec<_>>();
    assert_eq!(operations.len(), 4);
    assert!(
        node_text(parsed.tree(), &source, operations[1].id()).starts_with(b"\"payload.after\"")
    );
    assert!(node_text(parsed.tree(), &source, operations[3].id()).starts_with(b"\"opaque.after\""));
}

#[test]
fn complete_generic_fixture_exposes_ordered_typed_components() {
    let bytes = generic_complete_corpus("valid.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let lexed = lex(&source);
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert!(lexed.diagnostics().is_empty(), "{:?}", lexed.diagnostics());
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();

    let operations = parsed.file().operations().collect::<Vec<_>>();
    let results = operations[0].results().collect::<Vec<_>>();
    assert_eq!(results.len(), 2);
    assert!(results[0].number().is_some());
    assert!(results[1].number().is_none());
    assert_eq!(operations[1].operands().count(), 3);
    assert_eq!(parsed.file().nodes(SyntaxKind::OperandUse).count(), 5);

    assert!(operations[2].properties().is_some());
    assert!(operations[2].attributes().is_some());
    let successors = operations[5].successors().collect::<Vec<_>>();
    assert_eq!(successors.len(), 3);
    assert_eq!(
        successors[0].arguments_group().unwrap().arguments().count(),
        2
    );
    assert_eq!(successors[0].arguments().count(), 2);
    assert_eq!(
        successors[1].arguments_group().unwrap().arguments().count(),
        0
    );
    assert_eq!(successors[1].arguments().count(), 0);
    assert!(successors[2].arguments_group().is_none());

    let regions = operations[3].regions().collect::<Vec<_>>();
    assert_eq!(regions.len(), 2);
    let blocks = regions[0].blocks().collect::<Vec<_>>();
    assert_eq!(blocks.len(), 4);
    assert!(blocks[0].label().is_none());
    assert!(blocks[1].label().is_some());
    assert_eq!(blocks[1].arguments().count(), 2);
    let second_region_blocks = regions[1].blocks().collect::<Vec<_>>();
    assert_eq!(second_region_blocks.len(), 1);
    assert!(second_region_blocks[0].label().is_some());
    assert!(operations[3].trailing_location().is_some());
    assert!(parsed.file().nodes(SyntaxKind::LocationAttribute).count() >= 3);

    let kinds = operations[2]
        .components()
        .filter_map(|component| component.tree().kind(component.id()))
        .collect::<Vec<_>>();
    let property = kinds
        .iter()
        .position(|kind| *kind == SyntaxKind::PropertyDict)
        .unwrap();
    let attributes = kinds
        .iter()
        .position(|kind| *kind == SyntaxKind::AttributeDict)
        .unwrap();
    assert!(property < attributes);
}

#[test]
fn complete_generic_malformed_fixture_is_lossless_and_recovers() {
    let bytes = generic_complete_corpus("malformed.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let lexed = lex(&source);
    assert!(lexed.diagnostics().is_empty(), "{:?}", lexed.diagnostics());
    let parsed = parse_generic_operations(&lexed).unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert!(!parsed.diagnostics().is_empty());
    let texts = parsed
        .file()
        .operations()
        .map(|operation| node_text(parsed.tree(), &source, operation.id()))
        .collect::<Vec<_>>();
    for name in [
        b"\"test.after_result\"".as_slice(),
        b"\"test.after_successor\"",
        b"\"test.after_properties\"",
        b"\"test.after_region\"",
        b"\"test.after_location\"",
    ] {
        assert!(
            texts.iter().any(|text| text.starts_with(name)),
            "missing {:?} in {:?}",
            String::from_utf8_lossy(name),
            texts
                .iter()
                .map(|text| String::from_utf8_lossy(text))
                .collect::<Vec<_>>()
        );
    }
    assert!(texts.iter().any(|text| text.starts_with(b"\"test.kept\"")));
}

#[test]
fn every_generic_complete_fixture_reconstructs_and_verifies() {
    for entry in fs::read_dir(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1/generic-complete"),
    )
    .unwrap()
    {
        let path = entry.unwrap().path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("mlir") {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let source = Source::new(bytes.clone()).unwrap();
        let parsed = parse_generic_operations(&lex(&source)).unwrap();
        assert_eq!(
            reconstruct(parsed.tree(), &source),
            bytes,
            "{}",
            path.display()
        );
        parsed.tree().verify().unwrap();
    }
}

#[test]
fn unknown_custom_operations_are_explicit_and_recover_at_ancestors() {
    let bytes = recovery_custom_corpus("malformed.mlir");
    let source = Source::new(bytes.clone()).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert_eq!(
        parsed
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        2
    );
    assert_eq!(
        parsed
            .file()
            .nodes(SyntaxKind::UnparsedCustomOperation)
            .count(),
        2
    );
    let operations = parsed
        .file()
        .operations()
        .map(|operation| node_text(parsed.tree(), &source, operation.id()))
        .collect::<Vec<_>>();
    for expected in [
        b"\"test.after_unknown\"".as_slice(),
        b"\"test.after_array\"",
        b"\"test.after_inner\"",
        b"\"test.outer\"",
    ] {
        assert!(operations.iter().any(|text| text.starts_with(expected)));
    }
}

#[test]
fn unknown_custom_recovery_preserves_result_prefixes() {
    let bytes = b"test.unknown %arg\n%0 = \"test.after\"() : () -> ()\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert_eq!(
        parsed
            .file()
            .nodes(SyntaxKind::UnparsedCustomOperation)
            .count(),
        1
    );
    let operations = parsed
        .file()
        .operations()
        .map(|operation| node_text(parsed.tree(), &source, operation.id()))
        .collect::<Vec<_>>();
    assert_eq!(operations.len(), 2, "{operations:?}");
    let operation = parsed.file().operations().nth(1).unwrap();
    assert!(node_text(parsed.tree(), &source, operation.id()).starts_with(b"%0 ="));

    let bytes = b"vendor.before\n%0=vendor.compact\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    let compact = parsed.syntax().file().operations().nth(1).unwrap();
    assert!(
        node_text(parsed.syntax().tree(), &source, compact.id()).starts_with(b"%0=vendor.compact")
    );
}

#[test]
fn unknown_custom_regions_are_recursively_parsed() {
    let bytes = b"vendor.outer {\n  vendor.inner {\n    vendor.leaf\n  }\n}\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
    assert_eq!(reconstruct(parsed.syntax().tree(), &source), bytes);
    parsed.syntax().tree().verify().unwrap();
    assert_eq!(parsed.syntax().file().operations().count(), 3);
    assert_eq!(parsed.syntax().file().regions().count(), 2);
}

#[test]
fn unknown_custom_attributes_remain_payloads_not_operations_or_regions() {
    let bytes = b"vendor.op %x\n  attributes {flag = true}\nvendor.next\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
    assert_eq!(reconstruct(parsed.syntax().tree(), &source), bytes);
    let operations = parsed
        .syntax()
        .file()
        .operations()
        .map(|operation| node_text(parsed.syntax().tree(), &source, operation.id()))
        .collect::<Vec<_>>();
    assert_eq!(operations.len(), 2, "{operations:?}");
    assert_eq!(parsed.syntax().file().regions().count(), 0);

    let bytes = b"vendor.op attributes {}\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
    assert_eq!(reconstruct(parsed.syntax().tree(), &source), bytes);
    assert_eq!(parsed.syntax().file().operations().count(), 1);
    assert_eq!(parsed.syntax().file().regions().count(), 0);

    let bytes = b"vendor.op { vendor.inner } attributes {}\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
    assert_eq!(reconstruct(parsed.syntax().tree(), &source), bytes);
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    assert_eq!(parsed.syntax().file().regions().count(), 1);

    let bytes = b"vendor.op\n  attributes {\n    \"flag\" = true\n  }\nvendor.next\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
    assert_eq!(reconstruct(parsed.syntax().tree(), &source), bytes);
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    assert_eq!(parsed.syntax().file().regions().count(), 0);

    let bytes = b"vendor.first(@sym)\n  %operand attributes {flag = true}\nvendor.second\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
    assert_eq!(reconstruct(parsed.syntax().tree(), &source), bytes);
    assert_eq!(parsed.syntax().file().operations().count(), 2);
    assert_eq!(parsed.syntax().file().regions().count(), 0);

    for bytes in [
        b"vendor.op {flag}\nvendor.next\n".as_slice(),
        b"vendor.op {\"flag\"}\nvendor.next\n".as_slice(),
    ] {
        let source = Source::new(bytes).unwrap();
        let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
        assert_eq!(reconstruct(parsed.syntax().tree(), &source), bytes);
        assert_eq!(parsed.syntax().file().operations().count(), 2);
        assert_eq!(parsed.syntax().file().regions().count(), 0);
    }
}

#[test]
fn consecutive_unknown_custom_operations_are_individual_nodes() {
    let bytes = b"test.first %arg\ntest.second %arg\n";
    let source = Source::new(bytes.as_slice()).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    assert_eq!(reconstruct(parsed.tree(), &source), bytes);
    parsed.tree().verify().unwrap();
    assert_eq!(
        parsed
            .file()
            .nodes(SyntaxKind::UnparsedCustomOperation)
            .count(),
        2
    );
    assert_eq!(
        parsed
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation)
            .count(),
        2
    );
}

#[test]
fn every_owned_node_has_a_borrowed_structural_view() {
    let bytes = recovery_custom_corpus("malformed.mlir");
    let source = Source::new(bytes).unwrap();
    let parsed = parse_generic_operations(&lex(&source)).unwrap();
    for node in parsed.file().syntax() {
        assert_eq!(node.tree().kind(node.id()), Some(node.kind()));
        assert_eq!(parsed.file().node(node.id()).unwrap().kind(), node.kind());
        assert!(node.text_range().is_some());
        for child in node.children() {
            assert_eq!(child.tree().parent(child.id()), Some(node.id()));
        }
    }
    for node in parsed.file().typed_syntax() {
        assert_eq!(
            node.node().tree().kind(node.node().id()),
            Some(node.node().kind())
        );
    }
}
