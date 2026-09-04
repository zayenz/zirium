//! Lossless tokenization for textual MLIR.
//!
//! [`lex`] and [`lex_with_limits`] retain every source byte through token ranges.
//! Malformed input produces [`Diagnostic`] values instead of stopping the scan.

use crate::source::{Source, TextRange};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Whitespace,
    LineComment,
    String,
    BareIdentifier,
    AtIdentifier,
    PercentIdentifier,
    CaretIdentifier,
    ExclamationIdentifier,
    HashIdentifier,
    Dense,
    Sparse,
    DenseResource,
    Integer,
    WideInteger,
    Float,
    IntType,
    FloatType,
    IndexType,
    Loc,
    Unknown,
    CallSite,
    Fused,
    Tuple,
    Tensor,
    Vector,
    MemRef,
    AffineMap,
    AffineSet,
    Mod,
    FloorDiv,
    CeilDiv,
    Strided,
    X,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Less,
    Greater,
    Colon,
    Comma,
    Equal,
    Plus,
    Star,
    Question,
    VerticalBar,
    Minus,
    Slash,
    Arrow,
    Ellipsis,
    FileMetadataBegin,
    FileMetadataEnd,
    Invalid,
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    kind: TokenKind,
    range: TextRange,
}
impl Token {
    pub fn new(kind: TokenKind, range: TextRange) -> Self {
        Self { kind, range }
    }

    pub fn kind(self) -> TokenKind {
        self.kind
    }
    pub fn range(self) -> TextRange {
        self.range
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticKind {
    FileLimit,
    TokenLimit,
    InvalidByte,
    UnterminatedString,
    InvalidEscape,
    InvalidIdentifier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    kind: DiagnosticKind,
    range: TextRange,
}
impl Diagnostic {
    pub fn kind(&self) -> DiagnosticKind {
        self.kind
    }
    pub fn range(&self) -> TextRange {
        self.range
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LexerLimits {
    pub max_file_bytes: usize,
    pub max_tokens: usize,
}
impl Default for LexerLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: u32::MAX as usize,
            max_tokens: usize::MAX,
        }
    }
}

#[derive(Debug)]
pub struct Lexed {
    tokens: Vec<Token>,
    diagnostics: Vec<Diagnostic>,
}
impl Lexed {
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
    pub fn reconstruct(&self, source: &Source) -> Vec<u8> {
        let mut result = Vec::with_capacity(source.bytes().len());
        for token in &self.tokens {
            if token.kind != TokenKind::Eof {
                result.extend_from_slice(source.slice(token.range).expect("checked lexer range"));
            }
        }
        result
    }
}

/// Tokenizes a source with the default [`LexerLimits`].
///
/// Scanning always returns a token tape ending in [`TokenKind::Eof`]. Problems
/// such as an unterminated token or an exceeded limit appear in
/// [`Lexed::diagnostics`].
pub fn lex(source: &Source) -> Lexed {
    lex_with_limits(source, LexerLimits::default())
}

/// Tokenizes a source with explicit file and token limits.
///
/// Exceeding `max_file_bytes` adds a [`DiagnosticKind::FileLimit`] diagnostic
/// and continues normal tokenization. Reaching `max_tokens` records the
/// remaining input as one [`TokenKind::Invalid`] token, adds a
/// [`DiagnosticKind::TokenLimit`] diagnostic, and stops scanning.
pub fn lex_with_limits(source: &Source, limits: LexerLimits) -> Lexed {
    let bytes = source.bytes();
    let mut lexer = Lexer {
        bytes,
        position: 0,
        tokens: Vec::new(),
        diagnostics: Vec::new(),
    };
    if bytes.len() > limits.max_file_bytes {
        lexer.diagnostic(DiagnosticKind::FileLimit, 0, bytes.len());
    }
    while lexer.position < bytes.len() {
        if lexer.tokens.len() >= limits.max_tokens {
            let start = lexer.position;
            lexer.position = bytes.len();
            lexer.push(TokenKind::Invalid, start, bytes.len());
            lexer.diagnostic(DiagnosticKind::TokenLimit, start, bytes.len());
            break;
        }
        lexer.next_token();
    }
    lexer.push(TokenKind::Eof, bytes.len(), bytes.len());
    Lexed {
        tokens: lexer.tokens,
        diagnostics: lexer.diagnostics,
    }
}

struct Lexer<'a> {
    bytes: &'a [u8],
    position: usize,
    tokens: Vec<Token>,
    diagnostics: Vec<Diagnostic>,
}

impl Lexer<'_> {
    fn next_token(&mut self) {
        let start = self.position;
        let byte = self.bump();
        let kind = match byte {
            b if b.is_ascii_whitespace() => {
                self.eat_while(u8::is_ascii_whitespace);
                TokenKind::Whitespace
            }
            b'/' if self.peek() == Some(b'/') => {
                self.bump();
                self.eat_while(|b| *b != b'\n' && *b != b'\r');
                TokenKind::LineComment
            }
            b'"' => {
                self.string(start);
                TokenKind::String
            }
            b'@' => {
                if self.at_identifier(start) {
                    TokenKind::AtIdentifier
                } else {
                    self.diagnostic(DiagnosticKind::InvalidIdentifier, start, self.position);
                    TokenKind::Invalid
                }
            }
            b'%' => self.prefixed_identifier(start, TokenKind::PercentIdentifier),
            b'^' => self.prefixed_identifier(start, TokenKind::CaretIdentifier),
            b'!' => self.prefixed_identifier(start, TokenKind::ExclamationIdentifier),
            b'#' => {
                if self.bytes.get(self.position..self.position + 2) == Some(b"-}") {
                    self.position += 2;
                    TokenKind::FileMetadataEnd
                } else {
                    self.prefixed_identifier(start, TokenKind::HashIdentifier)
                }
            }
            b'x' if self.at_dimension_x() => TokenKind::X,
            b if is_bare_start(b) => {
                self.eat_while(is_bare_continue);
                if is_integer_type(&self.bytes[start..self.position]) {
                    TokenKind::IntType
                } else if is_float_type(&self.bytes[start..self.position]) {
                    TokenKind::FloatType
                } else if &self.bytes[start..self.position] == b"index" {
                    TokenKind::IndexType
                } else {
                    match &self.bytes[start..self.position] {
                        b"loc" => TokenKind::Loc,
                        b"unknown" => TokenKind::Unknown,
                        b"callsite" => TokenKind::CallSite,
                        b"fused" => TokenKind::Fused,
                        b"tuple" => TokenKind::Tuple,
                        b"tensor" => TokenKind::Tensor,
                        b"vector" => TokenKind::Vector,
                        b"memref" => TokenKind::MemRef,
                        b"affine_map" => TokenKind::AffineMap,
                        b"affine_set" => TokenKind::AffineSet,
                        b"mod" => TokenKind::Mod,
                        b"floordiv" => TokenKind::FloorDiv,
                        b"ceildiv" => TokenKind::CeilDiv,
                        b"strided" => TokenKind::Strided,
                        b"dense" => TokenKind::Dense,
                        b"sparse" => TokenKind::Sparse,
                        b"dense_resource" => TokenKind::DenseResource,
                        _ => TokenKind::BareIdentifier,
                    }
                }
            }
            b if b.is_ascii_digit() => self.number(),
            b'-' if self.peek() == Some(b'>') => {
                self.bump();
                TokenKind::Arrow
            }
            b'-' => TokenKind::Minus,
            b'.' if self.bytes.get(self.position..self.position + 2) == Some(b"..") => {
                self.position += 2;
                TokenKind::Ellipsis
            }
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'{' if self.bytes.get(self.position..self.position + 2) == Some(b"-#") => {
                self.position += 2;
                TokenKind::FileMetadataBegin
            }
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b'<' => TokenKind::Less,
            b'>' => TokenKind::Greater,
            b':' => TokenKind::Colon,
            b',' => TokenKind::Comma,
            b'=' => TokenKind::Equal,
            b'+' => TokenKind::Plus,
            b'*' => TokenKind::Star,
            b'?' => TokenKind::Question,
            b'|' => TokenKind::VerticalBar,
            b'/' => TokenKind::Slash,
            _ => {
                self.diagnostic(DiagnosticKind::InvalidByte, start, self.position);
                TokenKind::Invalid
            }
        };
        self.push(kind, start, self.position);
    }

    fn number(&mut self) -> TokenKind {
        let start = self.position - 1;
        if self.bytes.get(self.position - 1) == Some(&b'0')
            && self.peek() == Some(b'x')
            && self
                .bytes
                .get(self.position + 1)
                .is_some_and(u8::is_ascii_hexdigit)
        {
            self.bump();
            self.eat_while(|b| b.is_ascii_hexdigit());
            return if self.position - start > 18 {
                TokenKind::WideInteger
            } else {
                TokenKind::Integer
            };
        }
        self.eat_while(|b| b.is_ascii_digit());
        let mut float = false;
        if self.peek() == Some(b'.') {
            float = true;
            self.bump();
            self.eat_while(|b| b.is_ascii_digit());
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let mut cursor = self.position + 1;
            if matches!(self.bytes.get(cursor), Some(b'+' | b'-')) {
                cursor += 1;
            }
            if self.bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
                float = true;
                self.position = cursor + 1;
                self.eat_while(|b| b.is_ascii_digit());
            }
        }
        if float {
            TokenKind::Float
        } else {
            TokenKind::Integer
        }
    }

    fn string(&mut self, start: usize) {
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.bump();
                    return;
                }
                b'\n' | b'\x0b' | b'\x0c' => break,
                b'\\' => {
                    let escape = self.position;
                    self.bump();
                    match self.peek() {
                        Some(b'"' | b'\\' | b'n' | b't') => {
                            self.bump();
                        }
                        Some(a)
                            if a.is_ascii_hexdigit()
                                && self
                                    .bytes
                                    .get(self.position + 1)
                                    .is_some_and(u8::is_ascii_hexdigit) =>
                        {
                            self.position += 2;
                        }
                        Some(_) => {
                            self.bump();
                            self.diagnostic(DiagnosticKind::InvalidEscape, escape, self.position);
                        }
                        None => break,
                    }
                }
                _ => {
                    self.bump();
                }
            }
        }
        self.diagnostic(DiagnosticKind::UnterminatedString, start, self.position);
    }

    fn at_identifier(&mut self, start: usize) -> bool {
        if self.peek() == Some(b'"') {
            self.bump();
            self.string(start);
            true
        } else if self.peek().is_some_and(is_bare_start) {
            self.bump();
            self.eat_while(is_at_continue);
            true
        } else {
            false
        }
    }
    fn prefixed_identifier(&mut self, start: usize, valid_kind: TokenKind) -> TokenKind {
        if self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.eat_while(u8::is_ascii_digit);
            return valid_kind;
        }
        if self.peek().is_some_and(is_suffix_start) {
            self.bump();
            self.eat_while(is_suffix_continue);
            return valid_kind;
        }
        self.diagnostic(DiagnosticKind::InvalidIdentifier, start, self.position);
        TokenKind::Invalid
    }
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }
    fn at_dimension_x(&self) -> bool {
        let previous = self
            .tokens
            .iter()
            .rev()
            .find(|token| !matches!(token.kind, TokenKind::Whitespace | TokenKind::LineComment))
            .map(|token| token.kind);
        matches!(
            previous,
            Some(TokenKind::Integer | TokenKind::Question | TokenKind::Star | TokenKind::RBracket)
        ) && is_dimension_x_suffix(
            self.bytes[self.position..]
                .iter()
                .copied()
                .find(|byte| !byte.is_ascii_whitespace()),
        )
    }
    fn bump(&mut self) -> u8 {
        let byte = self.bytes[self.position];
        self.position += 1;
        byte
    }
    fn eat_while(&mut self, predicate: impl Fn(&u8) -> bool) {
        while self.bytes.get(self.position).is_some_and(&predicate) {
            self.position += 1;
        }
    }
    fn push(&mut self, kind: TokenKind, start: usize, end: usize) {
        self.tokens.push(Token {
            kind,
            range: TextRange::new(start as u32, end as u32).expect("checked source length"),
        });
    }
    fn diagnostic(&mut self, kind: DiagnosticKind, start: usize, end: usize) {
        self.diagnostics.push(Diagnostic {
            kind,
            range: TextRange::new(start as u32, end as u32).expect("checked source length"),
        });
    }
}

fn is_bare_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}
fn is_dimension_x_suffix(next: Option<u8>) -> bool {
    next.is_some_and(|byte| {
        byte.is_ascii_digit()
            || matches!(byte, b'?' | b'*')
            || matches!(
                byte,
                b'i' | b's' | b'u' | b'f' | b'b' | b'!' | b't' | b'v' | b'm'
            )
    })
}
fn is_bare_continue(byte: &u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(*byte, b'_' | b'$' | b'.')
}
fn is_at_continue(byte: &u8) -> bool {
    is_bare_continue(byte)
}
fn is_suffix_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$' | b'.' | b'-')
}
fn is_suffix_continue(byte: &u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(*byte, b'_' | b'$' | b'.' | b'-')
}
fn is_integer_type(bytes: &[u8]) -> bool {
    let digits = if bytes.first() == Some(&b'i') {
        &bytes[1..]
    } else if matches!(bytes.first(), Some(b's' | b'u')) && bytes.get(1) == Some(&b'i') {
        &bytes[2..]
    } else {
        return false;
    };
    !digits.is_empty() && digits.iter().all(u8::is_ascii_digit)
}

fn is_float_type(bytes: &[u8]) -> bool {
    matches!(
        bytes,
        b"bf16"
            | b"f16"
            | b"f32"
            | b"f64"
            | b"f80"
            | b"f128"
            | b"tf32"
            | b"f8E4M3"
            | b"f8E5M2"
            | b"f8E4M3FN"
            | b"f8E5M2FNUZ"
            | b"f8E4M3FNUZ"
            | b"f8E4M3B11FNUZ"
            | b"f8E3M4"
            | b"f8E8M0FNU"
            | b"f4E2M1FN"
            | b"f6E2M3FN"
            | b"f6E3M2FN"
    )
}
