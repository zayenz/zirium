use std::{fmt, io};

use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    printer::{PrintError, PrintLayout},
    semantic::{
        LoweringMode, RetentionProfile, lower_with_dialect_registry,
        lower_with_dialect_registry_and_retention,
    },
};

fn round_trip(path: &str) {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/corpus/mlir-22.1")
            .join(path),
    )
    .unwrap();
    let parsed = ParsedFile::parse(bytes).unwrap();
    let first = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
        .document
        .unwrap_or_else(|| panic!("{path} did not lower strictly"));
    let mut printed = Vec::new();
    first.print_io(&mut printed, PrintLayout::Pretty).unwrap();
    let text = String::from_utf8(printed).unwrap();
    let reparsed = ParsedFile::parse(text.as_bytes()).unwrap();
    let relowered =
        lower_with_dialect_registry(&reparsed, LoweringMode::Strict, &DialectRegistry::EMPTY);
    let second = relowered.document.unwrap_or_else(|| {
        let ranges: Vec<_> = relowered
            .diagnostics
            .iter()
            .map(|diagnostic| {
                &text[diagnostic.range.start() as usize..diagnostic.range.end() as usize]
            })
            .collect();
        panic!(
            "{path} did not relower strictly: {:?}\n{ranges:?}\n{text}",
            relowered.diagnostics,
        )
    });
    let mut first_text = String::new();
    let mut second_text = String::new();
    first.print(&mut first_text, PrintLayout::Compact).unwrap();
    second
        .print(&mut second_text, PrintLayout::Compact)
        .unwrap();
    assert!(
        first.structurally_eq(&second),
        "{path}\nfirst: {first_text}\nsecond: {second_text}"
    );
}

#[test]
fn accumulated_semantic_corpus_round_trips_structurally() {
    for path in [
        "semantic-proving/valid.mlir",
        "semantic-proving/forward.mlir",
        "generic-complete/valid.mlir",
        "shaped-affine/semantic-valid.mlir",
        "payload-opaque/valid.mlir",
    ] {
        round_trip(path);
    }
}

#[test]
fn builtin_dense_arrays_print_canonically_and_reparse_typed() {
    let parsed =
        ParsedFile::parse(b"\"test\"() {dimensions = array<i64: 0x1, // comment\n-2>, bits = array<f32: 0x7FC00000>} : () -> ()".to_vec()).unwrap();
    let first = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
        .document
        .unwrap();
    let mut printed = String::new();
    first.print(&mut printed, PrintLayout::Compact).unwrap();
    assert!(printed.contains("array<i64: 0x1, -2>"), "{printed}");
    assert!(printed.contains("array<f32: 0x7FC00000>"), "{printed}");
    let reparsed = ParsedFile::parse(printed.as_bytes()).unwrap();
    let second =
        lower_with_dialect_registry(&reparsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
            .document
            .unwrap();
    assert!(first.structurally_eq(&second));
}

#[test]
fn streams_to_fmt_and_io_sinks_deterministically() {
    let parsed = ParsedFile::parse(
        include_bytes!("../../../tests/corpus/mlir-22.1/generic-complete/valid.mlir").as_slice(),
    )
    .unwrap();
    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
            .document
            .unwrap();
    let mut string = String::new();
    document.print(&mut string, PrintLayout::Compact).unwrap();
    let mut bytes = Vec::new();
    document.print_io(&mut bytes, PrintLayout::Compact).unwrap();
    assert_eq!(string.as_bytes(), bytes);

    struct CountingSink(usize);
    impl fmt::Write for CountingSink {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            self.0 += value.len();
            Ok(())
        }
    }
    let mut custom = CountingSink(0);
    document.print(&mut custom, PrintLayout::Pretty).unwrap();
    assert!(custom.0 > 0);

    let mut buffered = io::BufWriter::new(Vec::new());
    document
        .print_io(&mut buffered, PrintLayout::Pretty)
        .unwrap();
    assert!(!buffered.into_inner().unwrap().is_empty());
}

#[test]
fn escaped_generic_operation_names_survive_canonical_printing() {
    let source = r#""vendor.\22quoted"() : () -> ()"#;
    let document = strict_document(source);
    assert_eq!(
        document.operation_name(document.root_operations()[0]),
        Some(r"vendor.\22quoted")
    );
    let mut printed = String::new();
    document.print(&mut printed, PrintLayout::Compact).unwrap();
    assert_eq!(printed, source);
    let reparsed = strict_document(&printed);
    assert!(document.structurally_eq(&reparsed));
}

#[test]
fn unchanged_preserving_output_keeps_trailing_metadata_byte_exact() {
    let bytes = br##""work"() : () -> ()
{-# dialect_resources: { payload: "quoted \"#-} marker" } #-}
"##;
    let parsed = ParsedFile::parse(bytes.as_slice()).unwrap();
    let document = lower_with_dialect_registry_and_retention(
        &parsed,
        LoweringMode::Strict,
        RetentionProfile::Hybrid,
        &DialectRegistry::EMPTY,
    )
    .document
    .unwrap();
    assert_eq!(parsed.original_bytes(), bytes);
    let mut original = Vec::new();
    parsed.write_original(&mut original).unwrap();
    assert_eq!(original, bytes);
    assert_eq!(
        document.preserving_bytes(PrintLayout::Pretty).unwrap(),
        bytes
    );
    let mut canonical = Vec::new();
    document
        .print_io(&mut canonical, PrintLayout::Pretty)
        .unwrap();
    assert!(
        !canonical
            .windows(b"dialect_resources".len())
            .any(|window| window == b"dialect_resources")
    );
}

#[test]
fn incomplete_documents_fail_before_the_first_sink_write() {
    let parsed = ParsedFile::parse(
        include_bytes!("../../../tests/corpus/mlir-22.1/semantic-proving/unresolved.mlir")
            .as_slice(),
    )
    .unwrap();
    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::BestEffort, &DialectRegistry::EMPTY)
            .document
            .unwrap();
    let mut sink = String::from("unchanged");
    assert!(matches!(
        document.print(&mut sink, PrintLayout::Pretty),
        Err(PrintError::IncompleteDocument)
    ));
    assert_eq!(sink, "unchanged");
}

fn strict_document(source: &str) -> zirium::semantic::Document {
    let parsed = ParsedFile::parse(std::sync::Arc::<[u8]>::from(source.as_bytes())).unwrap();
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY);
    lowered.document.unwrap_or_else(|| {
        panic!(
            "source did not lower strictly: {:?}: {source}",
            lowered.diagnostics
        )
    })
}

#[test]
fn structural_equality_ignores_source_spelling_and_formatting() {
    let first = strict_document(
        r#"%first = "producer"() : () -> i32
"consumer"(%first) : (i32) -> ()
"empty"() : () -> ()
"#,
    );
    let second = strict_document(
        r#"%renamed = "producer"() : () -> i32 // comment

"consumer"(%renamed) : (i32) -> ()
"empty"() : () -> ()
"#,
    );
    assert!(first.structurally_eq(&second));
}

#[test]
fn structural_equality_distinguishes_vector_scalability() {
    let scalable = strict_document(r#"%v = "vector"() : () -> vector<[4]x8xf32>"#);
    let fixed = strict_document(r#"%v = "vector"() : () -> vector<4x8xf32>"#);
    assert!(!scalable.structurally_eq(&fixed));
}

#[test]
fn structural_equality_distinguishes_opaque_memref_parameters() {
    let first = strict_document(
        r#"#stride = 1
%m = "memref"() : () -> memref<4xf32, strided<[#stride], offset: 0>, 3>"#,
    );
    let second = strict_document(
        r#"#stride = 2
%m = "memref"() : () -> memref<4xf32, strided<[#stride], offset: 0>, 3>"#,
    );
    assert!(!first.structurally_eq(&second));
}
