use std::collections::HashSet;

use crate::formatter::{AssemblyStyle, FormatOptions, MAX_INDENT_WIDTH};
use crate::source::TextRange;

use super::lexer::{Lexed, Token, TokenKind};

pub const DEFAULT_NESTING_LIMIT: usize = 64;

// Retain the parser paths for existing syntax-tree consumers.
pub use super::model::*;

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
    pub(crate) parse_work: usize,
    pub(crate) parse_results: usize,
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
    parse_with_initial_bindings(lexed, DEFAULT_NESTING_LIMIT, &[])
}

pub fn parse_with_nesting_limit(lexed: &Lexed<'_>, nesting_limit: usize) -> Parsed {
    parse_with_initial_bindings(lexed, nesting_limit, &[])
}

pub(crate) fn parse_with_initial_bindings(
    lexed: &Lexed<'_>,
    nesting_limit: usize,
    initial_bindings: &[&str],
) -> Parsed {
    parse_with_budget(
        lexed,
        nesting_limit,
        initial_bindings,
        super::comprehension::default_limits(),
    )
}

pub(crate) fn parse_with_budget(
    lexed: &Lexed<'_>,
    nesting_limit: usize,
    initial_bindings: &[&str],
    limits: zirql::ProgramLimits,
) -> Parsed {
    let mut budget = zirql::Budget::for_program(limits);
    let mut parser = Parser {
        source: lexed.source(),
        tokens: lexed.tokens(),
        cursor: 0,
        diagnostics: Vec::new(),
        nesting_limit,
        budget: &mut budget,
        bindings: initial_bindings
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
    };
    let program = parser.program();
    Parsed {
        program: parser.diagnostics.is_empty().then_some(program).flatten(),
        diagnostics: parser.diagnostics,
        parse_work: limits.work - parser.budget.remaining_work(),
        parse_results: limits.results - parser.budget.remaining_results().unwrap(),
    }
}

struct Parser<'a, 'b> {
    source: &'a str,
    tokens: &'a [Token],
    cursor: usize,
    diagnostics: Vec<Diagnostic>,
    nesting_limit: usize,
    bindings: HashSet<String>,
    budget: &'b mut zirql::Budget,
}

impl Parser<'_, '_> {
    fn program(&mut self) -> Option<Program> {
        self.skip_trivia();
        let mut parameters = Vec::new();
        let mut parameter_positions = Vec::new();
        if self.at(TokenKind::Eof) {
            return Some(Program {
                parameters,
                parameter_positions,
                statements: Vec::new(),
                expression: Expression {
                    first: Vec::new(),
                    rest: Vec::new(),
                    range: self.current().range(),
                },
                emit_final_result: true,
            });
        }
        let mut statements = Vec::new();
        loop {
            self.skip_trivia();
            if self.at_identifier("do") {
                self.bump();
                let expression = self.expression(0)?;
                self.expect(TokenKind::Semicolon, "expected `;` after do statement")?;
                self.skip_trivia();
                if self.at(TokenKind::Eof) {
                    return Some(Program {
                        parameters,
                        parameter_positions,
                        statements,
                        expression,
                        emit_final_result: false,
                    });
                }
                statements.push(Statement::Do(expression));
                continue;
            }
            if self.at_identifier("let")
                || self.at_identifier("report")
                || self.at_identifier("param")
            {
                let mut reader = zirql::ArgumentReader::new(
                    self.source,
                    self.current().range().start() as usize,
                )
                .ok()?;
                match zirql::read_statement_prefix(&mut reader, self.budget) {
                    Ok(Some(zirql::CommonStatement::Param(declaration))) => {
                        if is_reserved(&declaration.name.value)
                            || self.bindings.contains(&declaration.name.value)
                        {
                            self.sync(declaration.name.range.start);
                            self.error("parameter name is reserved or already defined");
                            return None;
                        }
                        self.sync(reader.position());
                        self.expect(
                            TokenKind::Semicolon,
                            "expected `;` after parameter declaration",
                        )?;
                        let admission = self
                            .budget
                            .charge(
                                declaration.name.value.len() + 2,
                                zirql::Location::Query(declaration.name.range.clone()),
                            )
                            .and_then(|_| {
                                self.budget.charge_results(
                                    2,
                                    zirql::Location::Query(declaration.name.range.clone()),
                                )
                            });
                        if let Err(error) = admission {
                            self.shared_error(error);
                            return None;
                        }
                        self.bindings.insert(declaration.name.value.clone());
                        parameter_positions.push(statements.len());
                        parameters.push(declaration);
                        continue;
                    }
                    Ok(Some(zirql::CommonStatement::Let(name))) => {
                        if is_reserved(&name.value) || self.bindings.contains(&name.value) {
                            self.error("binding name is reserved or already defined");
                            return None;
                        }
                        self.sync(reader.position());
                        let expression = if self.starts_native() {
                            self.expression(0)?
                        } else {
                            self.value_expression(0)?
                        };
                        if !expression.is_read_only() {
                            self.error(
                                "bindings cannot edit or emit; use a separate query statement",
                            );
                            return None;
                        }
                        self.expect(TokenKind::Semicolon, "expected `;` after binding")?;
                        self.bindings.insert(name.value.clone());
                        statements.push(Statement::Binding {
                            name: name.value,
                            expression,
                        });
                        continue;
                    }
                    Ok(Some(zirql::CommonStatement::Report)) => {
                        self.sync(reader.position());
                        let mut expression = self.value_expression(0)?;
                        self.shared_suffix(&mut expression, 0)?;
                        self.skip_trivia();
                        if self.at(TokenKind::Semicolon) {
                            self.bump();
                            self.skip_trivia();
                        }
                        if self.at(TokenKind::Eof) {
                            return Some(Program {
                                parameters,
                                parameter_positions,
                                statements,
                                expression,
                                emit_final_result: true,
                            });
                        }
                        statements.push(Statement::Query(expression));
                        continue;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.shared_error(error);
                        return None;
                    }
                }
            }
            let next = self.tokens[self.cursor + 1..]
                .iter()
                .find(|token| token.kind() != TokenKind::Trivia);
            if self.at(TokenKind::Identifier)
                && next.is_some_and(|token| token.kind() == TokenKind::Equals)
            {
                let name = self.identifier_text()?;
                if is_reserved(&name) || self.bindings.contains(&name) {
                    self.error_at_previous("binding name is reserved or already defined");
                    return None;
                }
                self.expect(TokenKind::Equals, "expected `=` after binding name")?;
                let expression = self.expression(0)?;
                if !expression.is_read_only() {
                    self.error("bindings cannot edit or emit; use a separate query statement");
                    return None;
                }
                self.expect(TokenKind::Semicolon, "expected `;` after binding")?;
                self.bindings.insert(name.clone());
                statements.push(Statement::Binding { name, expression });
                continue;
            }
            let expression = self.expression(0)?;
            self.skip_trivia();
            if self.at(TokenKind::Semicolon) {
                self.bump();
                self.skip_trivia();
                if !self.at(TokenKind::Eof) {
                    statements.push(Statement::Query(expression));
                    continue;
                }
            }
            if !self.at(TokenKind::Eof) {
                self.error("expected `|`, a set operator, `;`, or end of query");
            }
            return Some(Program {
                parameters,
                parameter_positions,
                statements,
                expression,
                emit_final_result: true,
            });
        }
    }

    fn next_kind(&self) -> Option<TokenKind> {
        self.tokens[self.cursor + 1..]
            .iter()
            .find(|token| token.kind() != TokenKind::Trivia)
            .map(|token| token.kind())
    }
    fn sync(&mut self, position: usize) {
        while self.current().kind() != TokenKind::Eof
            && (self.current().range().start() as usize) < position
        {
            self.cursor += 1;
        }
    }
    fn shared_error(&mut self, error: zirql::QueryError) {
        let position = match error.location {
            zirql::Location::Query(range) | zirql::Location::Source(range) => range.start,
        };
        let message = match error.message.as_str() {
            "parameter type must be int, string or bool" => {
                "parameter type must be int, string or bool"
            }
            "integer literal exceeds signed 64-bit range" => {
                "integer literal exceeds signed 64-bit range"
            }
            "invalid JSON string escape or Unicode sequence" => {
                "invalid JSON string escape or Unicode sequence"
            }
            "nonfinite float literal" => "nonfinite float literal",
            "nonzero decimal literal rounded to zero" => "nonzero decimal literal rounded to zero",
            "null is unsupported in constructed values" => {
                "null is unsupported in positional comprehensions"
            }
            "value nesting limit exceeded" => "value nesting limit exceeded",
            "work limit exceeded" => "query work limit exceeded during parsing",
            "result limit exceeded" => "result limit exceeded during parsing",
            _ => "invalid constructed report expression",
        };
        self.diagnostics.push(Diagnostic {
            message,
            range: TextRange::at(position as u32),
        });
    }
    fn shared_value(&mut self, depth: usize) -> Option<super::comprehension::ParsedComprehension> {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
        };
        let nesting = host.nesting;
        match zirql::read_value_expression(&mut reader, &mut host, self.budget, nesting) {
            Ok(value) => {
                self.sync(reader.position());
                Some(super::comprehension::ParsedComprehension::new(value))
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }
    fn shared_projection(&mut self, depth: usize) -> Option<super::report::ParsedProjection> {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
        };
        let nesting = host.nesting;
        match zirql::read_projection(&mut reader, &mut host, self.budget, nesting) {
            Ok(value) => {
                self.sync(reader.position());
                Some(super::report::ParsedProjection::new(value))
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }
    fn shared_native(
        &mut self,
        depth: usize,
        field_only: bool,
    ) -> Option<super::capture::ParsedOperand> {
        let start = self.current().range().start() as usize;
        let end = if field_only {
            self.tokens[self.cursor..]
                .iter()
                .find(|token| {
                    matches!(
                        token.kind(),
                        TokenKind::Pipe
                            | TokenKind::Comma
                            | TokenKind::Semicolon
                            | TokenKind::RParen
                            | TokenKind::RBrace
                            | TokenKind::Eof
                    )
                })
                .map_or(self.source.len(), |token| token.range().start() as usize)
        } else {
            self.source.len()
        };
        let mut reader = zirql::ArgumentReader::new(&self.source[..end], start).ok()?;
        // Bare primitive arguments denote bindings, even when their capture
        // name is also a stage word. Explicit input pipelines remain native.
        if let Err(error) = self.budget.charge(
            self.bindings.iter().map(|name| name.len() + 1).sum(),
            zirql::Location::Query(start..start),
        ) {
            self.shared_error(error);
            return None;
        }
        let mut bindings = self.bindings.clone();
        if !field_only
            && self.at(TokenKind::Identifier)
            && matches!(
                self.next_kind(),
                Some(TokenKind::Comma | TokenKind::RParen | TokenKind::Operator)
            )
            && self.current_text() != "input"
        {
            if let Err(error) = self.budget.charge(
                self.current_text().len() + 1,
                zirql::Location::Query(start..start),
            ) {
                self.shared_error(error);
                return None;
            }
            bindings.insert(self.current_text().to_owned());
        }
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            bindings: &bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
        };
        let nesting = host.nesting;
        match zirql::read_native_expression(&mut reader, &mut host, self.budget, nesting) {
            Ok(value) => {
                self.sync(reader.position());
                Some(super::capture::ParsedOperand::new(value))
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }
    fn shared_match(&mut self, depth: usize) -> Option<super::capture::ParsedMatch> {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
        };
        let nesting = host.nesting;
        match zirql::read_match(&mut reader, &mut host, self.budget, nesting) {
            Ok(value) => {
                self.sync(reader.position());
                Some(super::capture::ParsedMatch::new(value))
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }
    fn starts_native(&mut self) -> bool {
        self.skip_trivia();
        if self.at(TokenKind::Identifier) {
            let name = self.current_text();
            (name == "match" && self.next_kind() == Some(TokenKind::LBrace))
                || self.source[self.current().range().start() as usize..]
                    .split_once('|')
                    .is_some_and(|(source, _)| {
                        let source = source.trim();
                        source.contains('.')
                            && source.bytes().all(|byte| {
                                byte.is_ascii_alphanumeric()
                                    || byte == b'_'
                                    || byte == b'.'
                                    || byte.is_ascii_whitespace()
                            })
                    })
                || is_native_stage(name)
                || self.bindings.contains(name)
                    && matches!(
                        self.next_kind(),
                        Some(TokenKind::Pipe | TokenKind::Semicolon | TokenKind::Eof)
                    )
        } else {
            false
        }
    }
    fn value_expression(&mut self, depth: usize) -> Option<Expression> {
        let start = self.current().range().start();
        let value = self.shared_value(depth)?;
        Some(Expression {
            first: vec![Stage::Comprehension {
                value,
                range: self.span(start),
            }],
            rest: vec![],
            range: self.span(start),
        })
    }
    fn shared_suffix(&mut self, expression: &mut Expression, depth: usize) -> Option<()> {
        while self.at(TokenKind::Pipe) {
            self.bump();
            expression.first.push(self.stage(depth)?);
            self.skip_trivia();
        }
        expression.range = self.span(expression.range.start());
        Some(())
    }

    fn expression(&mut self, depth: usize) -> Option<Expression> {
        self.skip_trivia();
        if depth >= self.nesting_limit {
            self.error("query nesting limit exceeded");
            return None;
        }
        let start = self.current().range().start();
        let first = self.pipeline(depth)?;
        let mut rest = Vec::new();
        loop {
            self.skip_trivia();
            let operator = match self.current_text() {
                "union" => SetOperator::Union,
                "intersect" => SetOperator::Intersect,
                "except" => SetOperator::Except,
                _ => break,
            };
            self.bump();
            rest.push((operator, self.pipeline(depth)?));
        }
        let expression = Expression {
            first,
            rest,
            range: self.span(start),
        };
        if !expression.rest.is_empty()
            && let Some(stage) = expression.first_non_selection_stage()
        {
            self.diagnostics.push(Diagnostic {
                message: "set operands must be selection queries; put edits and count after the grouped set expression",
                range: stage.range(),
            });
        }
        Some(expression)
    }

    fn pipeline(&mut self, depth: usize) -> Option<Vec<Stage>> {
        let mut stages = vec![self.stage(depth)?];
        loop {
            self.skip_trivia();
            if !self.at(TokenKind::Pipe) {
                break;
            }
            self.bump();
            self.skip_trivia();
            if stages.last().is_some_and(Stage::is_terminal) {
                self.error("no stage may follow count");
            }
            stages.push(self.stage(depth)?);
        }
        Some(stages)
    }

    fn stage(&mut self, depth: usize) -> Option<Stage> {
        self.skip_trivia();
        let start = self.current().range().start();
        if self.at_identifier("match") && self.next_kind() == Some(TokenKind::LBrace) {
            let value = self.shared_match(depth)?;
            return Some(Stage::Match {
                value,
                range: self.span(start),
            });
        }
        if self.at(TokenKind::Identifier) && self.next_kind() == Some(TokenKind::Operator) {
            let next = self.tokens[self.cursor + 1..]
                .iter()
                .find(|token| token.kind() != TokenKind::Trivia)?;
            if &self.source[next.range().as_range()] == "." {
                let value = self.shared_native(depth, true)?;
                return Some(Stage::Native {
                    value,
                    range: self.span(start),
                });
            }
        }
        if (self.at(TokenKind::LBracket)
            && super::comprehension::starts(self.source, self.tokens, self.cursor))
            || (self.at_identifier("group_by") && self.next_kind() == Some(TokenKind::LParen))
        {
            let value = self.shared_value(depth)?;
            return Some(Stage::Comprehension {
                value,
                range: self.span(start),
            });
        }
        if self.at_identifier("project") && self.next_kind() == Some(TokenKind::LParen) {
            let value = self.shared_projection(depth)?;
            return Some(Stage::Projection {
                value,
                range: self.span(start),
            });
        }
        if self.at(TokenKind::LBrace) || self.at(TokenKind::LBracket) {
            let value = self.json_literal(depth + 1)?;
            return Some(Stage::Literal {
                value,
                range: self.span(start),
            });
        }
        if self.at(TokenKind::LParen) {
            self.bump();
            let expression = Box::new(self.expression(depth + 1)?);
            self.expect(TokenKind::RParen, "expected `)` after query")?;
            return Some(Stage::Group {
                expression,
                range: self.span(start),
            });
        }
        let Some(name) = self.identifier_text() else {
            self.error("expected a query stage such as input, filter, or defs");
            return None;
        };
        let range = self.span(start);
        Some(match name.as_str() {
            "operand" if self.at(TokenKind::LParen) => {
                self.bump();
                let operation = self.shared_native(depth + 1, false)?;
                self.expect(TokenKind::Comma, "expected `,` before operand index")?;
                self.skip_trivia();
                if !self.at(TokenKind::Integer) {
                    self.error("operand index must be a non-negative integer");
                    return None;
                }
                let Ok(index) = self.current_text().parse::<usize>() else {
                    self.error("operand index is too large");
                    return None;
                };
                self.bump();
                self.expect(TokenKind::RParen, "expected `)` after operand index")?;
                Stage::Operand {
                    operation,
                    index,
                    range: self.span(start),
                }
            }
            "reachable" => Stage::Reachable { range },
            "tally" => Stage::Tally { range },
            "value" => Stage::Value { range },
            "map_by" => {
                self.expect(TokenKind::LParen, "expected `(` after map_by")?;
                let key = Box::new(self.expression(depth + 1)?);
                self.expect(
                    TokenKind::Comma,
                    "expected `,` between map_by key and value",
                )?;
                let value = Box::new(self.expression(depth + 1)?);
                self.expect(TokenKind::RParen, "expected `)` after map_by")?;
                if !key.is_read_only() || !value.is_read_only() {
                    self.error("map_by queries cannot edit or emit");
                    return None;
                }
                Stage::MapBy {
                    key,
                    value,
                    range: self.span(start),
                }
            }
            "input" => Stage::Input { range },
            "closure" => Stage::Closure { range },
            "slice" | "backward_slice" => Stage::Slice { range },
            "forward_slice" => Stage::ForwardSlice { range },
            "defs" | "users" => {
                self.skip_trivia();
                let index = if self.at(TokenKind::LParen) {
                    self.bump();
                    self.skip_trivia();
                    if !self.at(TokenKind::Integer) || self.next_kind() != Some(TokenKind::RParen) {
                        let value = self.shared_value(depth + 1)?;
                        self.expect(TokenKind::RParen, "expected `)` after index")?;
                        return Some(Stage::Computed {
                            value: if name == "defs" {
                                ComputedStage::Defs(value)
                            } else {
                                ComputedStage::Users(value)
                            },
                            range: self.span(start),
                        });
                    }
                    if !self.at(TokenKind::Integer) {
                        self.error("expected a non-negative operand or result index");
                        return None;
                    }
                    let Ok(index) = self.current_text().parse::<usize>() else {
                        self.error("operand or result index is too large");
                        return None;
                    };
                    self.bump();
                    self.expect(TokenKind::RParen, "expected `)` after index")?;
                    Some(index)
                } else {
                    None
                };
                let range = if index.is_some() {
                    self.span(start)
                } else {
                    range
                };
                if name == "defs" {
                    Stage::Defs { index, range }
                } else {
                    Stage::Users { index, range }
                }
            }
            "parent" => Stage::Parent { range },
            "children" => Stage::Children { range },
            "subtree" => Stage::Subtree { range },
            "unique" => Stage::Unique { range },
            "sort" => Stage::Sort { range },
            "sort_by" | "min_by" | "min_all_by" | "max_by" | "max_all_by" => {
                self.expect(TokenKind::LParen, "expected `(` after selector stage")?;
                let selector = Box::new(self.expression(depth + 1)?);
                self.expect(TokenKind::RParen, "expected `)` after selector query")?;
                if !selector.is_read_only() {
                    self.error("selector queries cannot edit or emit");
                    return None;
                }
                let range = self.span(start);
                match name.as_str() {
                    "sort_by" => Stage::SortBy { selector, range },
                    "min_by" => Stage::MinBy { selector, range },
                    "min_all_by" => Stage::MinAllBy { selector, range },
                    "max_by" => Stage::MaxBy { selector, range },
                    "max_all_by" => Stage::MaxAllBy { selector, range },
                    _ => unreachable!(),
                }
            }
            "reverse" => Stage::Reverse { range },
            "head" | "tail" => {
                self.expect(TokenKind::LParen, "expected `(` after head or tail")?;
                self.skip_trivia();
                if !self.at(TokenKind::Integer) || self.next_kind() != Some(TokenKind::RParen) {
                    let value = self.shared_value(depth + 1)?;
                    if matches!(&value.expression.value, zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::Integer(integer)) if *integer < 0)
                    {
                        self.error("expected a non-negative item count");
                        return None;
                    }
                    self.expect(TokenKind::RParen, "expected `)` after item count")?;
                    return Some(Stage::Computed {
                        value: if name == "head" {
                            ComputedStage::Head(value)
                        } else {
                            ComputedStage::Tail(value)
                        },
                        range: self.span(start),
                    });
                }
                if !self.at(TokenKind::Integer) {
                    self.error("expected a non-negative item count");
                    return None;
                }
                let Ok(count) = self.current_text().parse::<usize>() else {
                    self.error("item count is too large");
                    return None;
                };
                self.bump();
                self.expect(TokenKind::RParen, "expected `)` after item count")?;
                let range = self.span(start);
                if name == "head" {
                    Stage::Head { count, range }
                } else {
                    Stage::Tail { count, range }
                }
            }
            "min" => Stage::Min { range },
            "min_all" => Stage::MinAll { range },
            "max" => Stage::Max { range },
            "max_all" => Stage::MaxAll { range },
            "check" => {
                self.skip_trivia();
                let (expected, message) = if self.at(TokenKind::LParen) {
                    self.bump();
                    self.skip_trivia();
                    if self.at(TokenKind::String) {
                        let (message, _) = self.string("expected a check message")?;
                        self.expect(TokenKind::RParen, "expected `)` after check message")?;
                        (None, Some(message))
                    } else {
                        if !self.at(TokenKind::Integer)
                            || self.next_kind() != Some(TokenKind::RParen)
                                && self.next_kind() != Some(TokenKind::Comma)
                        {
                            let value = self.shared_value(depth + 1)?;
                            self.expect(TokenKind::RParen, "expected `)` after check count")?;
                            return Some(Stage::Computed {
                                value: ComputedStage::Check(value),
                                range: self.span(start),
                            });
                        }
                        if !self.at(TokenKind::Integer) {
                            self.error("expected a non-negative item count or check message");
                            return None;
                        }
                        let Ok(expected) = self.current_text().parse::<usize>() else {
                            self.error("item count is too large");
                            return None;
                        };
                        self.bump();
                        self.skip_trivia();
                        let message = if self.at(TokenKind::Comma) {
                            self.bump();
                            Some(self.string("expected a check message after `,`")?.0)
                        } else {
                            None
                        };
                        self.expect(TokenKind::RParen, "expected `)` after check arguments")?;
                        (Some(expected), message)
                    }
                } else {
                    (None, None)
                };
                Stage::Check {
                    expected,
                    message,
                    range: self.span(start),
                }
            }
            "count" => Stage::Count { range },
            "emit" => Stage::Emit { range },
            "format" => {
                let options = self.format_options()?;
                Stage::Format {
                    options,
                    range: self.span(start),
                }
            }
            "json" => Stage::Json { range },
            "markdown" => Stage::Markdown { range },
            "print" => {
                self.expect(TokenKind::LParen, "expected `(` after print")?;
                let (template, _) = self.string("expected a print template string")?;
                self.expect(TokenKind::RParen, "expected `)` after print template")?;
                let parts = self.print_parts(&template)?;
                Stage::Print {
                    parts,
                    range: self.span(start),
                }
            }
            "names" => Stage::Names { range },
            "result_types" => Stage::ResultTypes { range },
            "operand_types" => Stage::OperandTypes { range },
            "filter" => {
                self.expect(TokenKind::LParen, "expected `(` after filter")?;
                let predicate = self.predicate(depth + 1)?;
                self.expect(TokenKind::RParen, "expected `)` after filter predicate")?;
                Stage::Filter {
                    predicate,
                    range: self.span(start),
                }
            }
            "root" => {
                self.expect(TokenKind::LParen, "expected `(` after root")?;
                let predicate = self.predicate(depth + 1)?;
                self.expect(TokenKind::RParen, "expected `)` after root predicate")?;
                Stage::Root {
                    predicate,
                    range: self.span(start),
                }
            }
            "fixpoint" => {
                self.expect(TokenKind::LParen, "expected `(` after fixpoint")?;
                let expression = Box::new(self.expression(depth + 1)?);
                self.expect(TokenKind::RParen, "expected `)` after fixpoint query")?;
                if let Some(stage) = expression.first_non_selection_stage() {
                    self.diagnostics.push(Diagnostic {
                        message: "fixpoint requires a selection query without edits or count",
                        range: stage.range(),
                    });
                }
                Stage::Fixpoint {
                    expression,
                    range: self.span(start),
                }
            }
            "set_attr" => return self.set_attr(start, depth + 1),
            "remove_attr" => return self.remove_attr(start, depth + 1),
            "attr" => return self.attr(start, depth + 1),
            _ if self.bindings.contains(&name) => Stage::Binding { name, range },
            _ => {
                self.error_at_previous("unknown query stage; expected input, filter, navigation, projection, fixpoint, an edit, check, count, emit, format, or json");
                return None;
            }
        })
    }

    fn format_options(&mut self) -> Option<FormatOptions> {
        let mut options = FormatOptions::default();
        self.skip_trivia();
        if !self.at(TokenKind::LParen) {
            return Some(options);
        }
        self.bump();
        self.skip_trivia();
        if self.at(TokenKind::RParen) {
            self.bump();
            return Some(options);
        }
        let mut assembly = false;
        let mut width = false;
        let mut indent = false;
        loop {
            self.skip_trivia();
            if !self.at(TokenKind::Identifier) {
                self.error("expected a format option name");
                return None;
            }
            let name = self.current_text().to_owned();
            self.bump();
            self.expect(TokenKind::Equals, "expected `=` after format option name")?;
            match name.as_str() {
                "assembly" => {
                    if assembly {
                        self.error("duplicate `assembly` format option");
                        return None;
                    }
                    assembly = true;
                    let (value, _) =
                        self.string("expected `source` or `generic` assembly style")?;
                    options.assembly = match value.as_str() {
                        "source" => AssemblyStyle::Source,
                        "generic" => AssemblyStyle::Generic,
                        _ => {
                            self.error_at_previous("expected `source` or `generic` assembly style");
                            return None;
                        }
                    };
                }
                "width" | "indent" => {
                    let duplicate = if name == "width" {
                        &mut width
                    } else {
                        &mut indent
                    };
                    if *duplicate {
                        self.error("duplicate numeric format option");
                        return None;
                    }
                    *duplicate = true;
                    self.skip_trivia();
                    if !self.at(TokenKind::Integer) {
                        self.error("expected a positive integer format option value");
                        return None;
                    }
                    let Ok(value) = self.current_text().parse::<usize>() else {
                        self.error("format option value is too large");
                        return None;
                    };
                    if value == 0 {
                        self.error("format option value must be positive");
                        return None;
                    }
                    if name == "indent" && value > MAX_INDENT_WIDTH {
                        self.error("format indent must not exceed 256");
                        return None;
                    }
                    self.bump();
                    if name == "width" {
                        options.line_width = value;
                    } else {
                        options.indent_width = value;
                    }
                }
                _ => {
                    self.error_at_previous("unknown format option");
                    return None;
                }
            }
            self.skip_trivia();
            if self.at(TokenKind::RParen) {
                self.bump();
                break;
            }
            self.expect(TokenKind::Comma, "expected `,` between format options")?;
        }
        Some(options)
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
            "true" | "false" => {
                if depth >= self.nesting_limit {
                    self.error_at_previous("query nesting limit exceeded");
                    return None;
                }
                Some(Predicate::Bool {
                    value: kind == "true",
                    range: self.span(start),
                })
            }
            "same_value" => {
                self.expect(TokenKind::LParen, "expected `(` after same_value")?;
                let left = self.shared_native(depth + 1, false)?;
                self.expect(TokenKind::Comma, "expected `,` between same_value operands")?;
                let right = self.shared_native(depth + 1, false)?;
                self.expect(TokenKind::RParen, "expected `)` after same_value")?;
                Some(Predicate::SameValue {
                    left,
                    right,
                    range: self.span(start),
                })
            }
            "op" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after op")?;
                self.skip_trivia();
                if !self.at(TokenKind::String) || self.next_kind() != Some(TokenKind::RParen) {
                    let value = self.shared_value(depth + 1)?;
                    self.expect(TokenKind::RParen, "expected `)` after operation name")?;
                    return Some(Predicate::OpValue {
                        name: value,
                        range: self.span(start),
                    });
                }
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
            "dialect" | "result_type" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after predicate")?;
                self.skip_trivia();
                if !self.at(TokenKind::String) || self.next_kind() != Some(TokenKind::RParen) {
                    let value = self.shared_value(depth + 1)?;
                    self.expect(TokenKind::RParen, "expected `)` after predicate argument")?;
                    return Some(Predicate::Computed {
                        value: if kind == "dialect" {
                            ComputedPredicate::Dialect(value)
                        } else {
                            ComputedPredicate::ResultType(value)
                        },
                        range: self.span(start),
                    });
                }
                let (value, value_range) = self.string("expected a quoted name or type")?;
                self.expect(TokenKind::RParen, "expected `)` after predicate argument")?;
                if value.is_empty() {
                    self.diagnostics.push(Diagnostic {
                        message: "name or type must not be empty",
                        range: value_range,
                    });
                }
                if kind == "dialect" {
                    Some(Predicate::Dialect {
                        name: value,
                        range: self.span(start),
                    })
                } else {
                    Some(Predicate::ResultType {
                        spelling: value,
                        range: self.span(start),
                    })
                }
            }
            "has_attr" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after has_attr")?;
                self.skip_trivia();
                if !self.at(TokenKind::String) || self.next_kind() != Some(TokenKind::RParen) {
                    let value = self.shared_value(depth + 1)?;
                    self.expect(TokenKind::RParen, "expected `)` after attribute name")?;
                    return Some(Predicate::Computed {
                        value: ComputedPredicate::HasAttr(value),
                        range: self.span(start),
                    });
                }
                let (name, range) = self.string("expected a quoted attribute name")?;
                self.expect(TokenKind::RParen, "expected `)` after attribute name")?;
                self.check_attribute_name(&name, range);
                Some(Predicate::HasAttr {
                    name,
                    range: self.span(start),
                })
            }
            "string_attr_eq" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after string_attr_eq")?;
                self.skip_trivia();
                if !self.at(TokenKind::String) || self.next_kind() != Some(TokenKind::Comma) {
                    let name = self.shared_value(depth + 1)?;
                    self.expect(TokenKind::Comma, "expected `,` in string_attr_eq")?;
                    let value = self.shared_value(depth + 1)?;
                    self.expect(
                        TokenKind::RParen,
                        "expected `)` after string_attr_eq arguments",
                    )?;
                    return Some(Predicate::Computed {
                        value: ComputedPredicate::Attr { name, value },
                        range: self.span(start),
                    });
                }
                let (name, name_range) = self.string("expected a quoted attribute name")?;
                self.expect_recover(
                    TokenKind::Comma,
                    "expected `,` in string_attr_eq",
                    &[
                        TokenKind::Comma,
                        TokenKind::RParen,
                        TokenKind::Pipe,
                        TokenKind::Eof,
                    ],
                )?;
                self.skip_trivia();
                if !self.at(TokenKind::String) || self.next_kind() != Some(TokenKind::RParen) {
                    let value = self.shared_value(depth + 1)?;
                    self.expect(
                        TokenKind::RParen,
                        "expected `)` after string_attr_eq arguments",
                    )?;
                    let name = super::comprehension::ParsedComprehension::new(zirql::Located::new(
                        zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::String(name)),
                        name_range.as_range(),
                    ));
                    return Some(Predicate::Computed {
                        value: ComputedPredicate::Attr { name, value },
                        range: self.span(start),
                    });
                }
                let (value, _) = self.string("expected a quoted attribute value")?;
                self.expect_recover(
                    TokenKind::RParen,
                    "expected `)` after string_attr_eq arguments",
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

    fn attr(&mut self, start: u32, depth: usize) -> Option<Stage> {
        if depth > self.nesting_limit {
            self.error("query nesting limit exceeded");
            return None;
        }
        self.expect(TokenKind::LParen, "expected `(` after attr")?;
        self.skip_trivia();
        if !self.at(TokenKind::String) || self.next_kind() != Some(TokenKind::RParen) {
            let value = self.shared_value(depth)?;
            self.expect(TokenKind::RParen, "expected `)` after attr argument")?;
            return Some(Stage::Computed {
                value: ComputedStage::Attr(value),
                range: self.span(start),
            });
        }
        let (name, name_range) = self.string("expected a quoted attribute name")?;
        self.expect_recover(
            TokenKind::RParen,
            "expected `)` after attr argument",
            &[TokenKind::RParen, TokenKind::Pipe, TokenKind::Eof],
        )?;
        self.check_attribute_name(&name, name_range);
        Some(Stage::Attr {
            name,
            range: self.span(start),
        })
    }

    fn json_literal(&mut self, depth: usize) -> Option<JsonLiteral> {
        self.skip_trivia();
        if depth >= self.nesting_limit {
            self.error("query nesting limit exceeded");
            return None;
        }
        match self.current().kind() {
            TokenKind::LBrace => {
                self.bump();
                let mut entries = Vec::new();
                self.skip_trivia();
                if !self.at(TokenKind::RBrace) {
                    loop {
                        let (key, _) = self.string("expected a quoted object key")?;
                        let key = self.print_parts(&key)?;
                        self.expect(TokenKind::Colon, "expected `:` after object key")?;
                        entries.push((key, self.json_literal(depth + 1)?));
                        self.skip_trivia();
                        if !self.at(TokenKind::Comma) {
                            break;
                        }
                        self.bump();
                    }
                }
                self.expect(TokenKind::RBrace, "expected `}` after object entries")?;
                Some(JsonLiteral::Object(entries))
            }
            TokenKind::LBracket => {
                self.bump();
                let mut values = Vec::new();
                self.skip_trivia();
                if !self.at(TokenKind::RBracket) {
                    loop {
                        values.push(self.json_literal(depth + 1)?);
                        self.skip_trivia();
                        if !self.at(TokenKind::Comma) {
                            break;
                        }
                        self.bump();
                    }
                }
                self.expect(TokenKind::RBracket, "expected `]` after array elements")?;
                Some(JsonLiteral::Array(values))
            }
            TokenKind::String => {
                let (value, _) = self.string("expected a string")?;
                Some(JsonLiteral::String(self.print_parts(&value)?))
            }
            TokenKind::Integer | TokenKind::Number | TokenKind::Operator => {
                let start = self.current().range().start();
                if self.current_text() == "-" {
                    self.bump();
                    if !matches!(
                        self.current().kind(),
                        TokenKind::Integer | TokenKind::Number
                    ) {
                        self.error("invalid or out-of-range JSON number");
                        return None;
                    }
                }
                let text = &self.source[start as usize..self.current().range().end() as usize];
                let Ok(value) = serde_json::from_str::<serde_json::Number>(text) else {
                    self.error("invalid or out-of-range JSON number");
                    return None;
                };
                self.bump();
                Some(JsonLiteral::Scalar(serde_json::Value::Number(value)))
            }
            TokenKind::Identifier => {
                let name = self.identifier_text()?;
                match name.as_str() {
                    "true" => Some(JsonLiteral::Scalar(serde_json::Value::Bool(true))),
                    "false" => Some(JsonLiteral::Scalar(serde_json::Value::Bool(false))),
                    "null" => Some(JsonLiteral::Scalar(serde_json::Value::Null)),
                    _ if self.bindings.contains(&name) => Some(JsonLiteral::Binding(name)),
                    _ => {
                        self.error_at_previous("expected an earlier binding or a JSON value");
                        None
                    }
                }
            }
            _ => {
                self.error("expected a JSON value or an earlier binding");
                None
            }
        }
    }

    fn print_parts(&mut self, template: &str) -> Option<Vec<PrintPart>> {
        let mut parts = Vec::new();
        let mut literal = String::new();
        let mut chars = template.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '{' | '}' if chars.peek() == Some(&ch) => {
                    chars.next();
                    literal.push(ch);
                }
                '{' => {
                    let mut name = String::new();
                    loop {
                        match chars.next() {
                            Some('}') => break,
                            Some(ch) => name.push(ch),
                            None => {
                                self.error_at_previous(
                                    "unclosed interpolation; use {name} or {{ for a literal brace",
                                );
                                return None;
                            }
                        }
                    }
                    if !self.bindings.contains(&name) {
                        self.error_at_previous(
                            "interpolation requires the name of an earlier binding",
                        );
                        return None;
                    }
                    if !literal.is_empty() {
                        parts.push(PrintPart::Literal(std::mem::take(&mut literal)));
                    }
                    parts.push(PrintPart::Binding(name));
                }
                '}' => {
                    self.error_at_previous("unmatched closing brace; use }} for a literal brace");
                    return None;
                }
                _ => literal.push(ch),
            }
        }
        if !literal.is_empty() {
            parts.push(PrintPart::Literal(literal));
        }
        Some(parts)
    }

    fn string(&mut self, message: &'static str) -> Option<(String, TextRange)> {
        self.skip_trivia();
        if !self.at(TokenKind::String) {
            self.error(message);
            return None;
        }
        let token = self.bump();
        let text = &self.source[token.range().as_range()];
        // Keep multiline query strings while using the JSON escape decoder.
        let normalized = text
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        let Ok(value) = serde_json::from_str::<String>(&normalized) else {
            self.error_at_previous("invalid JSON string escape or Unicode sequence");
            return None;
        };
        Some((value, token.range()))
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

pub(super) fn valid_attribute_name(name: &str) -> bool {
    !name.is_empty()
        && name.split('.').all(|component| {
            let mut chars = component.chars();
            chars
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
                && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        })
}

fn is_native_stage(name: &str) -> bool {
    matches!(
        name,
        "input"
            | "operand"
            | "filter"
            | "reachable"
            | "closure"
            | "slice"
            | "backward_slice"
            | "forward_slice"
            | "defs"
            | "users"
            | "parent"
            | "children"
            | "root"
            | "subtree"
            | "unique"
            | "sort"
            | "sort_by"
            | "min_by"
            | "min_all_by"
            | "max_by"
            | "max_all_by"
            | "reverse"
            | "head"
            | "tail"
            | "min"
            | "min_all"
            | "max"
            | "max_all"
            | "attr"
            | "names"
            | "result_types"
            | "operand_types"
            | "fixpoint"
            | "set_attr"
            | "remove_attr"
            | "check"
            | "count"
            | "emit"
            | "json"
            | "markdown"
            | "print"
            | "tally"
            | "value"
            | "map_by"
            | "project"
    )
}

pub(super) fn is_reserved(name: &str) -> bool {
    matches!(
        name,
        "input"
            | "do"
            | "filter"
            | "closure"
            | "slice"
            | "backward_slice"
            | "forward_slice"
            | "reachable"
            | "defs"
            | "users"
            | "parent"
            | "children"
            | "root"
            | "subtree"
            | "unique"
            | "sort"
            | "sort_by"
            | "reverse"
            | "head"
            | "tail"
            | "min"
            | "min_by"
            | "min_all"
            | "min_all_by"
            | "max"
            | "max_by"
            | "max_all"
            | "max_all_by"
            | "attr"
            | "names"
            | "result_types"
            | "operand_types"
            | "fixpoint"
            | "set_attr"
            | "remove_attr"
            | "check"
            | "count"
            | "emit"
            | "json"
            | "markdown"
            | "print"
            | "tally"
            | "value"
            | "map_by"
            | "union"
            | "intersect"
            | "except"
            | "and"
            | "or"
            | "not"
            | "true"
            | "false"
            | "null"
            | "op"
            | "dialect"
            | "result_type"
            | "has_attr"
            | "string_attr_eq"
    )
}

struct NativeReader<'a> {
    source: &'a str,
    tokens: &'a [Token],
    bindings: &'a HashSet<String>,
    nesting: usize,
}
impl NativeReader<'_> {
    fn parser<'b>(&self, position: usize, budget: &'b mut zirql::Budget) -> Parser<'_, 'b> {
        Parser {
            source: self.source,
            tokens: self.tokens,
            cursor: self
                .tokens
                .partition_point(|token| (token.range().start() as usize) < position),
            diagnostics: vec![],
            nesting_limit: self.nesting,
            bindings: self.bindings.clone(),
            budget,
        }
    }
    fn error(diagnostic: Diagnostic) -> zirql::QueryError {
        zirql::QueryError::query(diagnostic.range.as_range(), diagnostic.message)
    }
}
impl zirql::ValueReaderHost for NativeReader<'_> {
    type Stage = Stage;
    type SetOperator = SetOperator;
    type Emitter = ();
    type Predicate = Predicate;
    fn is_binding(&self, name: &str) -> bool {
        self.bindings.contains(name)
    }
    fn is_stage(&self, name: &str) -> bool {
        is_native_stage(name)
    }
    fn parse_stage(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<zirql::ExpressionStep<Stage, ()>, zirql::QueryError> {
        budget.charge(
            self.bindings.iter().map(|name| name.len() + 1).sum(),
            zirql::Location::Query(reader.position()..reader.position()),
        )?;
        let mut parser = self.parser(reader.position(), budget);
        let result = parser.stage(0);
        if let Some(diagnostic) = parser.diagnostics.first() {
            return Err(Self::error(*diagnostic));
        }
        let stage = result.ok_or_else(|| reader.error("expected a native stage"))?;
        reader.advance(stage.range().end() as usize - reader.position())?;
        Ok(zirql::ExpressionStep::Stage(stage))
    }
    fn parse_predicate_leaf(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<Predicate, zirql::QueryError> {
        budget.charge(
            self.bindings.iter().map(|name| name.len() + 1).sum(),
            zirql::Location::Query(reader.position()..reader.position()),
        )?;
        let mut parser = self.parser(reader.position(), budget);
        let result = parser.primary_predicate(0);
        if let Some(diagnostic) = parser.diagnostics.first() {
            return Err(Self::error(*diagnostic));
        }
        let predicate = result.ok_or_else(|| reader.error("expected a native predicate"))?;
        reader.advance(predicate.range().end() as usize - reader.position())?;
        Ok(predicate)
    }
    fn filter(
        &mut self,
        predicate: zirql::Predicate<Predicate>,
        range: std::ops::Range<usize>,
    ) -> Stage {
        Stage::Filter {
            predicate: lower_shared_predicate(predicate),
            range: TextRange::new(range.start as u32, range.end as u32)
                .expect("ordered query range"),
        }
    }
    fn projection(
        &mut self,
        projection: zirql::Projection<super::report::Operand>,
        range: std::ops::Range<usize>,
    ) -> Stage {
        Stage::Projection {
            value: super::report::ParsedProjection::new(projection),
            range: TextRange::new(range.start as u32, range.end as u32)
                .expect("ordered query range"),
        }
    }
    fn matching(
        &mut self,
        matching: zirql::NativeMatch<super::report::Operand, Predicate>,
        range: std::ops::Range<usize>,
    ) -> Result<Stage, zirql::QueryError> {
        Ok(Stage::Match {
            value: super::capture::ParsedMatch::new(matching),
            range: TextRange::new(range.start as u32, range.end as u32)
                .expect("ordered query range"),
        })
    }
    fn set_operator(&self, word: &str) -> Option<SetOperator> {
        match word {
            "union" => Some(SetOperator::Union),
            "intersect" => Some(SetOperator::Intersect),
            "except" => Some(SetOperator::Except),
            _ => None,
        }
    }
}
fn lower_shared_predicate(predicate: zirql::Predicate<Predicate>) -> Predicate {
    let range = TextRange::new(predicate.range.start as u32, predicate.range.end as u32)
        .expect("shared predicate ranges are ordered source offsets");
    match predicate.value {
        zirql::PredicateKind::Leaf(predicate) => predicate,
        zirql::PredicateKind::Not(predicate) => Predicate::Not {
            predicate: Box::new(lower_shared_predicate(*predicate)),
            range,
        },
        zirql::PredicateKind::Group(predicate) => Predicate::Group {
            predicate: Box::new(lower_shared_predicate(*predicate)),
            range,
        },
        zirql::PredicateKind::And(predicates) => Predicate::And {
            predicates: predicates.into_iter().map(lower_shared_predicate).collect(),
            range,
        },
        zirql::PredicateKind::Or(predicates) => Predicate::Or {
            predicates: predicates.into_iter().map(lower_shared_predicate).collect(),
            range,
        },
    }
}
