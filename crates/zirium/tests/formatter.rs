use zirium::{
    dialect::DialectRegistry,
    formatter::{AssemblyStyle, FormatError, FormatOptions},
    parser::ParsedFile,
};

fn format(source: &str) -> String {
    let parsed = ParsedFile::parse_with_registry(source.as_bytes(), DialectRegistry::baseline())
        .expect("source parses");
    assert!(parsed.lexer_diagnostics().is_empty());
    String::from_utf8(parsed.formatted_bytes(FormatOptions::default()).unwrap()).unwrap()
}

#[test]
fn source_formatting_preserves_custom_assembly_names_aliases_and_comments() {
    let source = "#word = i32\nmodule {\n// keep this\nfunc.func @twice(%input : i32) -> i32 {\n%sum=arith.addi %input,%input:i32 // trailing\nfunc.return %sum:i32\n}\n}\n";
    let formatted = format(source);
    assert_eq!(
        formatted,
        "#word = i32\nmodule {\n  // keep this\n  func.func @twice(%input: i32) -> i32 {\n    %sum = arith.addi %input, %input : i32 // trailing\n    func.return %sum : i32\n  }\n}\n"
    );
}

#[test]
fn source_formatting_is_idempotent() {
    let source =
        "module {\nfunc.func @f(%x : i32) {\n%y = arith.addi %x, %x : i32\nfunc.return\n}\n}\n";
    let once = format(source);
    assert_eq!(format(&once), once);
}

#[test]
fn unknown_custom_operations_remain_verbatim() {
    let source = "module {\nunknown.op  weird< payload , untouched >\n}\n";
    let formatted = format(source);
    assert!(
        formatted.contains("unknown.op  weird< payload , untouched >"),
        "{formatted}"
    );
}

#[test]
fn width_wraps_at_commas_and_remains_idempotent() {
    let source = "%result = \"test.many\"(%a,%b,%c,%d) : (i32,i32,i32,i32) -> i32\n";
    let parsed = ParsedFile::parse(source.as_bytes()).unwrap();
    let options = FormatOptions {
        line_width: 28,
        ..FormatOptions::default()
    };
    let once = String::from_utf8(parsed.formatted_bytes(options).unwrap()).unwrap();
    assert!(once.contains(",\n  %"), "{once}");

    let reparsed = ParsedFile::parse(once.as_bytes()).unwrap();
    assert_eq!(reparsed.formatted_bytes(options).unwrap(), once.as_bytes());
}

#[test]
fn parsed_files_reject_generic_assembly_formatting() {
    let parsed = ParsedFile::parse(&b"\"test.op\"() : () -> ()\n"[..]).unwrap();
    let error = parsed
        .formatted_bytes(FormatOptions {
            assembly: AssemblyStyle::Generic,
            ..FormatOptions::default()
        })
        .unwrap_err();
    assert!(matches!(
        error,
        FormatError::GenericAssemblyRequiresSemantics
    ));
}

#[test]
fn formatting_preserves_invalid_bytes_in_comments() {
    let source = b"// \xff\n\"test.op\"() : () -> ()\n";
    let parsed = ParsedFile::parse(&source[..]).unwrap();
    let formatted = parsed.formatted_bytes(FormatOptions::default()).unwrap();
    assert_eq!(formatted, source);
}

#[test]
fn source_formatting_keeps_grammar_sensitive_tokens_adjacent() {
    let source = r#"#meta = [@root::@leaf, loc("core.mlir":7:4)]
%pair:2 = "test.results"() : () -> (i32, i32)
"test.uses"(%pair#0, %pair#1) {set = affine_set<(d0) : (d0 >= 0, d0 == 1)>} : (i32, i32) -> ()
"#;
    let formatted = format(source);
    let reparsed = ParsedFile::parse(formatted.as_bytes()).expect("formatted output parses");
    assert!(reparsed.syntax().diagnostics().is_empty(), "{formatted}");
    assert!(formatted.contains("@root::@leaf"), "{formatted}");
    assert!(formatted.contains("%pair:2"), "{formatted}");
    assert!(formatted.contains("%pair#0"), "{formatted}");
    assert!(formatted.contains("d0 >= 0"), "{formatted}");
    assert!(formatted.contains("d0 == 1"), "{formatted}");
}

#[test]
fn standalone_comments_remain_on_their_own_lines() {
    let source = "\"a\"() : () -> ()\n\n// section\n\n// detail\n\"b\"() : () -> ()\n";
    assert_eq!(format(source), source);
}

#[test]
fn excessive_indent_width_is_rejected() {
    let parsed = ParsedFile::parse(&b"module {\n\"test.op\"() : () -> ()\n}\n"[..]).unwrap();
    assert!(
        parsed
            .formatted_bytes(FormatOptions {
                indent_width: 257,
                ..FormatOptions::default()
            })
            .is_err()
    );
}
