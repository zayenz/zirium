use crate::source::TextRange;

use super::lexer::{Lexed, Token, TokenKind};

pub const DEFAULT_NESTING_LIMIT: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    predicate: Predicate,
    stages: Vec<Stage>,
    range: TextRange,
}

impl Program {
    pub fn predicate(&self) -> &Predicate {
        &self.predicate
    }
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }
    pub fn range(&self) -> TextRange {
        self.range
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Op {
        name: String,
        range: TextRange,
    },
    HasAttr {
        name: String,
        range: TextRange,
    },
    Attr {
        name: String,
        value: String,
        range: TextRange,
    },
    Not {
        predicate: Box<Predicate>,
        range: TextRange,
    },
    And {
        predicates: Vec<Predicate>,
        range: TextRange,
    },
    Or {
        predicates: Vec<Predicate>,
        range: TextRange,
    },
    Group {
        predicate: Box<Predicate>,
        range: TextRange,
    },
}

impl Predicate {
    pub fn range(&self) -> TextRange {
        match self {
            Self::Op { range, .. }
            | Self::HasAttr { range, .. }
            | Self::Attr { range, .. }
            | Self::Not { range, .. }
            | Self::And { range, .. }
            | Self::Or { range, .. }
            | Self::Group { range, .. } => *range,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Closure {
        range: TextRange,
    },
    Defs {
        range: TextRange,
    },
    Users {
        range: TextRange,
    },
    Parent {
        range: TextRange,
    },
    Children {
        range: TextRange,
    },
    Union {
        predicate: Predicate,
        range: TextRange,
    },
    Intersect {
        predicate: Predicate,
        range: TextRange,
    },
    Except {
        predicate: Predicate,
        range: TextRange,
    },
    SetAttr {
        name: String,
        value: String,
        range: TextRange,
    },
    RemoveAttr {
        name: String,
        range: TextRange,
    },
    Count {
        range: TextRange,
    },
    Root {
        range: TextRange,
    },
}

impl Stage {
    pub fn range(&self) -> TextRange {
        match self {
            Self::Closure { range }
            | Self::Defs { range }
            | Self::Users { range }
            | Self::Parent { range }
            | Self::Children { range }
            | Self::Union { range, .. }
            | Self::Intersect { range, .. }
            | Self::Except { range, .. }
            | Self::SetAttr { range, .. }
            | Self::RemoveAttr { range, .. }
            | Self::Count { range }
            | Self::Root { range } => *range,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    message: &'static str,
    range: TextRange,
}

impl Diagnostic {
    pub fn message(self) -> &'static str {
        self.message
    }
    pub fn range(self) -> TextRange {
        self.range
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    program: Option<Program>,
    diagnostics: Vec<Diagnostic>,
}

impl Parsed {
    pub fn program(&self) -> Option<&Program> {
        self.program.as_ref()
    }
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
    pub fn into_program(self) -> Option<Program> {
        self.program
    }
}

pub fn parse(lexed: &Lexed<'_>) -> Parsed {
    parse_with_nesting_limit(lexed, DEFAULT_NESTING_LIMIT)
}

pub fn parse_with_nesting_limit(lexed: &Lexed<'_>, nesting_limit: usize) -> Parsed {
    let mut parser = Parser {
        source: lexed.source(),
        tokens: lexed.tokens(),
        cursor: 0,
        diagnostics: Vec::new(),
        nesting_limit,
    };
    let program = parser.program();
    Parsed {
        program: parser.diagnostics.is_empty().then_some(program).flatten(),
        diagnostics: parser.diagnostics,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PipelineKind {
    Selection,
    Scalar,
    Root,
}

struct Parser<'a> {
    source: &'a str,
    tokens: &'a [Token],
    cursor: usize,
    diagnostics: Vec<Diagnostic>,
    nesting_limit: usize,
}

impl Parser<'_> {
    fn program(&mut self) -> Option<Program> {
        self.skip_trivia();
        let start = self.current().range().start();
        self.expect_identifier("select", "expected `select`")?;
        self.expect(TokenKind::LParen, "expected `(` after select")?;
        let predicate = self.predicate(1)?;
        self.expect(TokenKind::RParen, "expected `)` after select predicate")?;
        let mut stages = Vec::new();
        let mut kind = PipelineKind::Selection;
        loop {
            self.skip_trivia();
            if self.at(TokenKind::Eof) {
                break;
            }
            if !self.at(TokenKind::Pipe) {
                self.error("expected `|` followed by a pipeline stage");
                self.synchronize(&[TokenKind::Pipe, TokenKind::Eof]);
                if self.at(TokenKind::Eof) {
                    break;
                }
            }
            self.bump();
            self.skip_trivia();
            let stage_start = self.current().range().start();
            let Some(name) = self.identifier_text() else {
                self.error("expected a pipeline stage");
                self.synchronize(&[TokenKind::Pipe, TokenKind::Eof]);
                continue;
            };
            let required_selection = match name.as_str() {
                "closure" | "defs" | "users" | "parent" | "children" | "union" | "intersect"
                | "except" | "set_attr" | "remove_attr" | "count" | "root" => true,
                _ => false,
            };
            if required_selection && kind != PipelineKind::Selection {
                let message = match name.as_str() {
                    "closure" => "closure requires a selection",
                    "defs" => "defs requires a selection",
                    "users" => "users requires a selection",
                    "parent" => "parent requires a selection",
                    "children" => "children requires a selection",
                    "union" => "union requires a selection",
                    "intersect" => "intersect requires a selection",
                    "except" => "except requires a selection",
                    "set_attr" => "set_attr requires a selection",
                    "remove_attr" => "remove_attr requires a selection",
                    "count" => "count requires a selection",
                    _ => "root requires a selection",
                };
                self.diagnostics.push(Diagnostic {
                    message,
                    range: self.previous().range(),
                });
            }
            let stage = match name.as_str() {
                "closure" => Some(Stage::Closure {
                    range: self.span(stage_start),
                }),
                "defs" => Some(Stage::Defs {
                    range: self.span(stage_start),
                }),
                "users" => Some(Stage::Users {
                    range: self.span(stage_start),
                }),
                "parent" => Some(Stage::Parent {
                    range: self.span(stage_start),
                }),
                "children" => Some(Stage::Children {
                    range: self.span(stage_start),
                }),
                "union" => self.predicate_stage(
                    stage_start,
                    "expected `(` after union",
                    |predicate, range| Stage::Union { predicate, range },
                ),
                "intersect" => self.predicate_stage(
                    stage_start,
                    "expected `(` after intersect",
                    |predicate, range| Stage::Intersect { predicate, range },
                ),
                "except" => self.predicate_stage(
                    stage_start,
                    "expected `(` after except",
                    |predicate, range| Stage::Except { predicate, range },
                ),
                "count" => {
                    kind = PipelineKind::Scalar;
                    Some(Stage::Count {
                        range: self.span(stage_start),
                    })
                }
                "root" => {
                    kind = PipelineKind::Root;
                    Some(Stage::Root {
                        range: self.span(stage_start),
                    })
                }
                "set_attr" => self.set_attr(stage_start, 1),
                "remove_attr" => self.remove_attr(stage_start, 1),
                _ => {
                    self.error_at_previous("unknown pipeline operation");
                    self.synchronize(&[TokenKind::Pipe, TokenKind::Eof]);
                    None
                }
            };
            if let Some(stage) = stage {
                stages.push(stage);
            }
        }
        Some(Program {
            predicate,
            stages,
            range: TextRange::new(start, self.current().range().end()).unwrap(),
        })
    }

    fn predicate(&mut self, depth: usize) -> Option<Predicate> {
        self.or_predicate(depth)
    }

    fn or_predicate(&mut self, depth: usize) -> Option<Predicate> {
        let first = self.and_predicate(depth)?;
        let start = first.range().start();
        let mut predicates = vec![first];
        while self.at_identifier("or") {
            self.bump();
            predicates.push(self.and_predicate(depth)?);
        }
        if predicates.len() == 1 {
            predicates.pop()
        } else {
            let end = predicates.last().unwrap().range().end();
            Some(Predicate::Or {
                predicates,
                range: TextRange::new(start, end).unwrap(),
            })
        }
    }

    fn and_predicate(&mut self, depth: usize) -> Option<Predicate> {
        let first = self.not_predicate(depth)?;
        let start = first.range().start();
        let mut predicates = vec![first];
        while self.at_identifier("and") {
            self.bump();
            predicates.push(self.not_predicate(depth)?);
        }
        if predicates.len() == 1 {
            predicates.pop()
        } else {
            let end = predicates.last().unwrap().range().end();
            Some(Predicate::And {
                predicates,
                range: TextRange::new(start, end).unwrap(),
            })
        }
    }

    fn not_predicate(&mut self, depth: usize) -> Option<Predicate> {
        self.skip_trivia();
        let mut starts = Vec::new();
        while self.at_identifier("not") {
            if depth + starts.len() >= self.nesting_limit {
                self.error("query nesting limit exceeded");
                return None;
            }
            starts.push(self.current().range().start());
            self.bump();
        }
        let mut predicate = self.primary_predicate(depth + starts.len())?;
        for start in starts.into_iter().rev() {
            predicate = Predicate::Not {
                range: TextRange::new(start, predicate.range().end()).unwrap(),
                predicate: Box::new(predicate),
            };
        }
        Some(predicate)
    }

    fn primary_predicate(&mut self, depth: usize) -> Option<Predicate> {
        self.skip_trivia();
        let start = self.current().range().start();
        if self.at(TokenKind::LParen) {
            if depth >= self.nesting_limit {
                self.error("query nesting limit exceeded");
                return None;
            }
            self.bump();
            let predicate = self.predicate(depth + 1)?;
            self.expect_recover(
                TokenKind::RParen,
                "expected `)` after predicate",
                &[TokenKind::RParen, TokenKind::Pipe, TokenKind::Eof],
            )?;
            return Some(Predicate::Group {
                predicate: Box::new(predicate),
                range: self.span(start),
            });
        }
        let Some(kind) = self.identifier_text() else {
            self.error("expected a predicate");
            return None;
        };
        match kind.as_str() {
            "op" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after op")?;
                let (name, range) = self.string("expected a quoted operation name")?;
                self.expect(TokenKind::RParen, "expected `)` after operation name")?;
                if name.is_empty() {
                    self.diagnostics.push(Diagnostic {
                        message: "operation name must not be empty",
                        range,
                    });
                }
                Some(Predicate::Op {
                    name,
                    range: self.span(start),
                })
            }
            "has_attr" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after has_attr")?;
                let (name, range) = self.string("expected a quoted attribute name")?;
                self.expect(TokenKind::RParen, "expected `)` after attribute name")?;
                self.check_attribute_name(&name, range);
                Some(Predicate::HasAttr {
                    name,
                    range: self.span(start),
                })
            }
            "attr" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after attr")?;
                let (name, name_range) = self.string("expected a quoted attribute name")?;
                self.expect_recover(
                    TokenKind::Comma,
                    "expected `,` in attr",
                    &[
                        TokenKind::Comma,
                        TokenKind::RParen,
                        TokenKind::Pipe,
                        TokenKind::Eof,
                    ],
                )?;
                let (value, _) = self.string("expected a quoted attribute value")?;
                self.expect_recover(
                    TokenKind::RParen,
                    "expected `)` after attr arguments",
                    &[TokenKind::RParen, TokenKind::Pipe, TokenKind::Eof],
                )?;
                self.check_attribute_name(&name, name_range);
                Some(Predicate::Attr {
                    name,
                    value,
                    range: self.span(start),
                })
            }
            _ => {
                self.error_at_previous("unknown predicate");
                None
            }
        }
    }

    fn check_attribute_name(&mut self, name: &str, range: TextRange) {
        if !valid_attribute_name(name) {
            self.diagnostics.push(Diagnostic {
                message: "attribute name must be a dotted ASCII identifier",
                range,
            });
        }
    }

    fn predicate_stage(
        &mut self,
        start: u32,
        open_message: &'static str,
        make_stage: impl FnOnce(Predicate, TextRange) -> Stage,
    ) -> Option<Stage> {
        self.expect(TokenKind::LParen, open_message)?;
        let predicate = self.predicate(1)?;
        self.expect_recover(
            TokenKind::RParen,
            "expected `)` after set predicate",
            &[TokenKind::RParen, TokenKind::Pipe, TokenKind::Eof],
        )?;
        Some(make_stage(predicate, self.span(start)))
    }

    fn set_attr(&mut self, start: u32, depth: usize) -> Option<Stage> {
        if depth > self.nesting_limit {
            self.error("query nesting limit exceeded");
            return None;
        }
        self.expect(TokenKind::LParen, "expected `(` after set_attr")?;
        let (name, name_range) = self.string("expected a quoted attribute name")?;
        self.expect_recover(
            TokenKind::Comma,
            "expected `,` in set_attr",
            &[
                TokenKind::Comma,
                TokenKind::RParen,
                TokenKind::Pipe,
                TokenKind::Eof,
            ],
        )?;
        let (value, value_range) = self.string("expected a quoted attribute value")?;
        self.expect_recover(
            TokenKind::RParen,
            "expected `)` after set_attr arguments",
            &[TokenKind::RParen, TokenKind::Pipe, TokenKind::Eof],
        )?;
        if !valid_attribute_name(&name) {
            self.diagnostics.push(Diagnostic {
                message: "attribute name must be a dotted ASCII identifier",
                range: name_range,
            });
        }
        if value.chars().any(char::is_control) {
            self.diagnostics.push(Diagnostic {
                message: "attribute string must not contain control characters",
                range: value_range,
            });
        }
        Some(Stage::SetAttr {
            name,
            value,
            range: self.span(start),
        })
    }

    fn remove_attr(&mut self, start: u32, depth: usize) -> Option<Stage> {
        if depth > self.nesting_limit {
            self.error("query nesting limit exceeded");
            return None;
        }
        self.expect(TokenKind::LParen, "expected `(` after remove_attr")?;
        let (name, name_range) = self.string("expected a quoted attribute name")?;
        self.expect_recover(
            TokenKind::RParen,
            "expected `)` after remove_attr argument",
            &[TokenKind::RParen, TokenKind::Pipe, TokenKind::Eof],
        )?;
        self.check_attribute_name(&name, name_range);
        Some(Stage::RemoveAttr {
            name,
            range: self.span(start),
        })
    }

    fn string(&mut self, message: &'static str) -> Option<(String, TextRange)> {
        self.skip_trivia();
        if !self.at(TokenKind::String) {
            self.error(message);
            return None;
        }
        let token = self.bump();
        let text = &self.source[token.range().as_range()];
        let mut value = String::new();
        let inner = text
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or("");
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                if let Some(escaped) = chars.next() {
                    if matches!(escaped, '"' | '\\') {
                        value.push(escaped);
                    }
                }
            } else {
                value.push(ch);
            }
        }
        Some((value, token.range()))
    }

    fn expect_identifier(&mut self, expected: &str, message: &'static str) -> Option<Token> {
        self.skip_trivia();
        if self.at(TokenKind::Identifier) && self.current_text() == expected {
            Some(self.bump())
        } else {
            self.error(message);
            None
        }
    }
    fn at_identifier(&mut self, expected: &str) -> bool {
        self.skip_trivia();
        self.at(TokenKind::Identifier) && self.current_text() == expected
    }
    fn identifier_text(&mut self) -> Option<String> {
        if !self.at(TokenKind::Identifier) {
            return None;
        }
        let token = self.bump();
        Some(self.source[token.range().as_range()].to_owned())
    }
    fn expect(&mut self, kind: TokenKind, message: &'static str) -> Option<Token> {
        self.skip_trivia();
        if self.at(kind) {
            Some(self.bump())
        } else {
            self.error(message);
            None
        }
    }
    fn expect_recover(
        &mut self,
        kind: TokenKind,
        message: &'static str,
        sync: &[TokenKind],
    ) -> Option<Token> {
        self.skip_trivia();
        if self.at(kind) {
            return Some(self.bump());
        }
        self.error(message);
        self.synchronize(sync);
        if self.at(kind) {
            Some(self.bump())
        } else {
            None
        }
    }
    fn synchronize(&mut self, kinds: &[TokenKind]) {
        while !kinds.contains(&self.current().kind()) {
            self.bump();
        }
    }
    fn skip_trivia(&mut self) {
        while self.at(TokenKind::Trivia) {
            self.cursor += 1;
        }
    }
    fn at(&self, kind: TokenKind) -> bool {
        self.current().kind() == kind
    }
    fn current(&self) -> Token {
        self.tokens[self.cursor.min(self.tokens.len() - 1)]
    }
    fn previous(&self) -> Token {
        self.tokens[self.cursor.saturating_sub(1)]
    }
    fn current_text(&self) -> &str {
        &self.source[self.current().range().as_range()]
    }
    fn bump(&mut self) -> Token {
        let token = self.current();
        if token.kind() != TokenKind::Eof {
            self.cursor += 1;
        }
        token
    }
    fn span(&self, start: u32) -> TextRange {
        TextRange::new(start, self.previous().range().end()).unwrap()
    }
    fn error(&mut self, message: &'static str) {
        self.diagnostics.push(Diagnostic {
            message,
            range: self.current().range(),
        });
    }
    fn error_at_previous(&mut self, message: &'static str) {
        self.diagnostics.push(Diagnostic {
            message,
            range: self.previous().range(),
        });
    }
}

fn valid_attribute_name(name: &str) -> bool {
    !name.is_empty()
        && name.split('.').all(|component| {
            let mut chars = component.chars();
            chars
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
                && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        })
}
