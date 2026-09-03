use zirium::lexer::{DiagnosticKind, LexerLimits, TokenKind, lex, lex_with_limits};
use zirium::source::Source;

fn source(bytes: &[u8]) -> Source {
    Source::new(bytes.to_vec()).unwrap()
}

fn assert_lossless(bytes: &[u8]) {
    let source = source(bytes);
    let lexed = lex(&source);
    let mut offset = 0;
    for token in lexed.tokens() {
        let range = token.range();
        assert_eq!(range.start(), offset);
        assert!(range.end() <= source.len());
        offset = range.end();
    }
    assert_eq!(lexed.tokens().last().unwrap().kind(), TokenKind::Eof);
    assert_eq!(lexed.tokens().last().unwrap().range().start(), source.len());
    assert_eq!(lexed.reconstruct(&source), bytes);
}

#[test]
fn covers_the_lexical_checklist() {
    let bytes = br#" // comment
bare_id %0 ^bb1 @name @"quoted" !dialect.ty #dialect.attr
"text\n\t\"\\\4f" 0 42 0x2A 1.25 6.0e-3 i32
(){}[]<>:;,=+*?| - / -> ... {-# #-}"#;
    let source = source(bytes);
    let kinds: Vec<_> = lex(&source)
        .tokens()
        .iter()
        .map(|token| token.kind())
        .collect();
    for expected in [
        TokenKind::Whitespace,
        TokenKind::LineComment,
        TokenKind::BareIdentifier,
        TokenKind::PercentIdentifier,
        TokenKind::CaretIdentifier,
        TokenKind::AtIdentifier,
        TokenKind::ExclamationIdentifier,
        TokenKind::HashIdentifier,
        TokenKind::String,
        TokenKind::Integer,
        TokenKind::Float,
        TokenKind::IntType,
        TokenKind::Minus,
        TokenKind::Slash,
        TokenKind::Arrow,
        TokenKind::Ellipsis,
        TokenKind::FileMetadataBegin,
        TokenKind::FileMetadataEnd,
    ] {
        assert!(kinds.contains(&expected), "missing {expected:?}");
    }
    assert_lossless(bytes);
}

#[test]
fn metadata_end_marker_inside_an_escaped_string_is_not_a_delimiter() {
    let bytes = br##"{-# dialect_resources: { payload: "quoted \"#-} text" } #-}"##;
    let source = source(bytes);
    let kinds = lex(&source)
        .tokens()
        .iter()
        .map(|token| token.kind())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds
            .iter()
            .filter(|&&kind| kind == TokenKind::FileMetadataEnd)
            .count(),
        1
    );
    assert!(kinds.contains(&TokenKind::String));
    assert_lossless(bytes);
}

#[test]
fn corpus_manifest_covers_the_owned_lexer_families() {
    let manifest = include_str!("../../../tests/corpus/mlir-22.1/manifest.toml");
    for family in [
        "trivia-and-line-comments",
        "strings-and-escapes",
        "identifiers",
        "numbers",
        "punctuation",
        "invalid-input-and-limits",
        "eof",
    ] {
        assert!(manifest.contains(&format!("name = \"{family}\"")));
    }
    let fixture = include_bytes!("../../../tests/corpus/mlir-22.1/lexer.mlir");
    assert_lossless(fixture);
}

#[test]
fn malformed_and_arbitrary_bytes_recover() {
    for bytes in [
        &b"\"unterminated\nnext"[..],
        &b"\"bad\\q\""[..],
        &b"\xff\xfe@ok"[..],
        &b"` /"[..],
    ] {
        assert_lossless(bytes);
        assert!(!lex(&source(bytes)).diagnostics().is_empty());
    }
}

#[test]
fn line_lookup_is_lazy_and_handles_common_newlines() {
    let source = source(b"a\nb\r\nc\rd");
    assert!(!source.line_index_is_built());
    assert_eq!(source.line(0), Some(0));
    assert!(source.line_index_is_built());
    assert_eq!(source.line(2), Some(1));
    assert_eq!(source.line(5), Some(2));
    assert_eq!(source.line(7), Some(3));
    assert_eq!(source.line(source.len()), Some(3));
    assert_eq!(source.line(source.len() + 1), None);
}

#[test]
fn limits_report_diagnostics_without_losing_bytes() {
    let source = source(b"one two three");
    let lexed = lex_with_limits(
        &source,
        LexerLimits {
            max_file_bytes: 3,
            max_tokens: 2,
        },
    );
    assert_eq!(lexed.reconstruct(&source), source.bytes());
    assert!(
        lexed
            .diagnostics()
            .iter()
            .any(|d| d.kind() == DiagnosticKind::FileLimit)
    );
    assert!(
        lexed
            .diagnostics()
            .iter()
            .any(|d| d.kind() == DiagnosticKind::TokenLimit)
    );
    assert_eq!(lexed.tokens().last().unwrap().kind(), TokenKind::Eof);
}

#[test]
fn arbitrary_byte_vectors_are_always_lossless() {
    let mut state = 0x9e37_79b9_u32;
    for length in 0..512 {
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bytes.push(state as u8);
        }
        assert_lossless(&bytes);
    }
}

#[test]
fn compact_dimension_x_does_not_split_ordinary_identifiers() {
    let source = source(
        b"x86 xname tensor vector memref mod floordiv ceildiv 2x3xf32 tensor<2 x 3 x f32> vector<[4]x8xf32> vector<[4] x 8 x f32>",
    );
    let lexed = lex(&source);
    assert_eq!(lexed.reconstruct(&source), source.bytes());
    let significant = lexed
        .tokens()
        .iter()
        .filter(|token| !matches!(token.kind(), TokenKind::Whitespace | TokenKind::Eof))
        .map(|token| token.kind())
        .collect::<Vec<_>>();
    assert_eq!(significant[0], TokenKind::BareIdentifier);
    assert_eq!(significant[1], TokenKind::BareIdentifier);
    assert_eq!(
        significant
            .iter()
            .filter(|kind| **kind == TokenKind::X)
            .count(),
        8
    );
}
