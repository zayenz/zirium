use crate::source::TextRange;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Identifier,
    String,
    LParen,
    RParen,
    Comma,
    Pipe,
    Trivia,
    Invalid,
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    kind: TokenKind,
    range: TextRange,
}

impl Token {
    pub fn kind(self) -> TokenKind {
        self.kind
    }
    pub fn range(self) -> TextRange {
        self.range
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticKind {
    QueryTooLarge,
    InvalidToken,
    InvalidEscape,
    UnterminatedString,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    kind: DiagnosticKind,
    range: TextRange,
}

impl Diagnostic {
    pub fn kind(self) -> DiagnosticKind {
        self.kind
    }
    pub fn range(self) -> TextRange {
        self.range
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lexed<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    diagnostics: Vec<Diagnostic>,
}

impl Lexed<'_> {
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
    pub(crate) fn source(&self) -> &str {
        self.source
    }
}

pub const MAX_QUERY_BYTES: usize = u32::MAX as usize;

pub fn query_size_supported(bytes: usize) -> bool {
    u32::try_from(bytes).is_ok()
}

pub fn lex(source: &str) -> Lexed<'_> {
    let bytes = source.as_bytes();
    if !query_size_supported(bytes.len()) {
        return Lexed {
            source,
            tokens: vec![Token {
                kind: TokenKind::Eof,
                range: TextRange::at(0),
            }],
            diagnostics: vec![Diagnostic {
                kind: DiagnosticKind::QueryTooLarge,
                range: TextRange::at(0),
            }],
        };
    }
    let mut tokens = Vec::new();
    let mut diagnostics = Vec::new();
    let mut position = 0;
    while position < bytes.len() {
        let start = position;
        let kind = match bytes[position] {
            b' ' | b'\t' | b'\r' | b'\n' => {
                position += 1;
                while bytes
                    .get(position)
                    .is_some_and(|byte| byte.is_ascii_whitespace())
                {
                    position += 1;
                }
                TokenKind::Trivia
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                position += 1;
                while bytes
                    .get(position)
                    .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                {
                    position += 1;
                }
                TokenKind::Identifier
            }
            b'(' => {
                position += 1;
                TokenKind::LParen
            }
            b')' => {
                position += 1;
                TokenKind::RParen
            }
            b',' => {
                position += 1;
                TokenKind::Comma
            }
            b'|' => {
                position += 1;
                TokenKind::Pipe
            }
            b'"' => {
                position += 1;
                let mut terminated = false;
                while position < bytes.len() {
                    match bytes[position] {
                        b'"' => {
                            position += 1;
                            terminated = true;
                            break;
                        }
                        b'\\' => {
                            let escape_start = position;
                            position += 1;
                            match bytes.get(position) {
                                Some(b'"' | b'\\') => position += 1,
                                Some(_) => {
                                    position += 1;
                                    diagnostics.push(Diagnostic {
                                        kind: DiagnosticKind::InvalidEscape,
                                        range: range(escape_start, position),
                                    });
                                }
                                None => {
                                    diagnostics.push(Diagnostic {
                                        kind: DiagnosticKind::InvalidEscape,
                                        range: range(escape_start, position),
                                    });
                                    break;
                                }
                            }
                        }
                        _ => position += 1,
                    }
                }
                if !terminated {
                    diagnostics.push(Diagnostic {
                        kind: DiagnosticKind::UnterminatedString,
                        range: range(start, position),
                    });
                }
                TokenKind::String
            }
            _ => {
                let ch_len = source[position..].chars().next().map_or(1, char::len_utf8);
                position += ch_len;
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::InvalidToken,
                    range: range(start, position),
                });
                TokenKind::Invalid
            }
        };
        tokens.push(Token {
            kind,
            range: range(start, position),
        });
    }
    tokens.push(Token {
        kind: TokenKind::Eof,
        range: range(position, position),
    });
    Lexed {
        source,
        tokens,
        diagnostics,
    }
}

fn range(start: usize, end: usize) -> TextRange {
    let start = u32::try_from(start).expect("query size checked before lexing");
    let end = u32::try_from(end).expect("query size checked before lexing");
    TextRange::new(start, end).expect("ordered query lexer range")
}
