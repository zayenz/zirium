use std::{borrow::Cow, collections::HashSet};

use crate::formatter::{AssemblyStyle, FormatOptions, MAX_INDENT_WIDTH};
use crate::source::TextRange;

use super::lexer::{Lexed, Token, TokenKind};

pub const DEFAULT_NESTING_LIMIT: usize = 64;

// Retain the parser paths for existing syntax-tree consumers.
pub use super::model::*;

/// An owned query parser diagnostic with its original byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    message: String,
    range: TextRange,
}

impl Diagnostic {
    pub fn message(&self) -> &str {
        &self.message
    }
    pub fn range(&self) -> TextRange {
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
        function_mode: has_control_values(lexed),
        cursor: 0,
        diagnostics: Vec::new(),
        nesting_limit,
        budget: &mut budget,
        locals: &[],
        bindings: Cow::Owned(
            initial_bindings
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
        ),
    };
    let program = parser.program();
    Parsed {
        program: parser.diagnostics.is_empty().then_some(program).flatten(),
        diagnostics: parser.diagnostics,
        parse_work: limits.work - parser.budget.remaining_work(),
        parse_results: limits.results - parser.budget.remaining_results().unwrap(),
    }
}

fn has_control_values(lexed: &Lexed<'_>) -> bool {
    let mut tokens = lexed
        .tokens()
        .iter()
        .filter(|token| token.kind() != TokenKind::Trivia)
        .peekable();
    let mut pending_if = false;
    let mut parentheses = Vec::new();
    // Carry pending probes through parenthesized segments instead of rescanning
    // each contextual `if` suffix. Every token is visited once.
    while let Some(token) = tokens.next() {
        match token.kind() {
            TokenKind::LParen => {
                parentheses.push(pending_if);
                pending_if = false;
            }
            TokenKind::RParen => {
                pending_if |= parentheses.pop().unwrap_or(false);
            }
            TokenKind::LBrace if pending_if => return true,
            TokenKind::Semicolon | TokenKind::RBracket | TokenKind::Eof => pending_if = false,
            TokenKind::Identifier => {
                let word = &lexed.source()[token.range().as_range()];
                let next = tokens.peek().map(|token| token.kind());
                if (zirql::is_common_control_call(word)
                    || super::functions::is_native_function(word))
                    && matches!(next, Some(TokenKind::LParen | TokenKind::Operator))
                    || word == "let" && matches!(next, Some(TokenKind::LParen | TokenKind::LBrace))
                {
                    return true;
                }
                pending_if |= word == "if";
            }
            _ => {}
        }
    }
    false
}

struct Parser<'a, 'b> {
    source: &'a str,
    tokens: &'a [Token],
    function_mode: bool,
    cursor: usize,
    diagnostics: Vec<Diagnostic>,
    nesting_limit: usize,
    bindings: Cow<'a, HashSet<String>>,
    budget: &'b mut zirql::Budget,
    locals: &'a [&'a str],
}

impl Parser<'_, '_> {
    fn program(&mut self) -> Option<Program> {
        let location = zirql::Location::Query(0..0);
        if let Err(error) = self
            .budget
            .charge(self.bindings.len(), location.clone())
            .and_then(|()| self.budget.charge_results(self.bindings.len(), location))
        {
            self.shared_error(error);
            return None;
        }
        let initial_names = self.bindings.iter().map(String::as_str).collect::<Vec<_>>();
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            bindings: &self.bindings,
            nesting: self.nesting_limit,
            function_mode: self.function_mode,
            program: ProgramSyntax::default(),
        };
        let parsed = zirql::read_program(
            self.source,
            &initial_names,
            &mut host,
            self.budget,
            self.nesting_limit,
        );
        let parsed = match parsed {
            Ok(parsed) => parsed,
            Err(mut error) => {
                if error.message == "expected ';'" {
                    error.message = match host.program.purpose {
                        Some(zirql::ProgramExpression::Suppressed) => {
                            "expected `;` after do statement".to_owned()
                        }
                        Some(zirql::ProgramExpression::Binding) => {
                            "expected `;` after binding".to_owned()
                        }
                        _ => error.message,
                    };
                }
                self.shared_error(error);
                return None;
            }
        };
        Some(frontend_program(parsed, host.program.signatures))
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
        let message = match error.category {
            zirql::QueryErrorCategory::UnsupportedNull => {
                "null is unsupported in positional comprehensions".to_owned()
            }
            zirql::QueryErrorCategory::WorkLimit => {
                "query work limit exceeded during parsing".to_owned()
            }
            zirql::QueryErrorCategory::ResultLimit => {
                "result limit exceeded during parsing".to_owned()
            }
            _ => error.message,
        };
        self.diagnostics.push(Diagnostic {
            message,
            range: TextRange::at(position as u32),
        });
    }
    fn shared_value(&mut self, depth: usize) -> Option<super::comprehension::ParsedComprehension> {
        self.read_shared_value(depth, false)
    }
    fn read_shared_value(
        &mut self,
        depth: usize,
        stop_at_pipe: bool,
    ) -> Option<super::comprehension::ParsedComprehension> {
        let mut end = self.source.len();
        let mut nesting = 0usize;
        for (offset, token) in self.tokens[self.cursor..].iter().enumerate() {
            match token.kind() {
                TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => nesting += 1,
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace if nesting == 0 => {
                    break;
                }
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => nesting -= 1,
                TokenKind::Semicolon if nesting == 0 => break,
                TokenKind::Pipe if nesting == 0 => {
                    if stop_at_pipe {
                        end = token.range().start() as usize;
                        break;
                    }
                    if let Some(next) = self.tokens[self.cursor + offset + 1..]
                        .iter()
                        .find(|t| t.kind() != TokenKind::Trivia)
                        && matches!(
                            &self.source[next.range().as_range()],
                            "json"
                                | "emit"
                                | "markdown"
                                | "print"
                                | "format"
                                | "set_attr"
                                | "remove_attr"
                        )
                    {
                        end = token.range().start() as usize;
                        break;
                    }
                }
                _ => {}
            }
        }
        let mut reader = zirql::ArgumentReader::new(
            &self.source[..end],
            self.current().range().start() as usize,
        )
        .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            function_mode: self.function_mode,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
            program: ProgramSyntax::default(),
        };
        let nesting = host.nesting;
        let locals = self.locals;
        match zirql::read_value_expression_in_scope(
            &mut reader,
            &mut host,
            locals,
            self.budget,
            nesting,
        ) {
            Ok(value) => {
                let value = match super::functions::normalize_outer(
                    value,
                    &self.bindings,
                    self.function_mode,
                    self.budget,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        self.shared_error(error);
                        return None;
                    }
                };
                if !self.function_mode {
                    let written = self.tokens[self.cursor..]
                        .iter()
                        .take_while(|token| (token.range().start() as usize) < reader.position())
                        .filter(|token| token.kind() != TokenKind::Trivia)
                        .collect::<Vec<_>>();
                    if let Some((_, pair)) = written.windows(2).enumerate().find(|(index, pair)| {
                        pair[0].kind() == TokenKind::Identifier
                            && pair[1].kind() == TokenKind::LParen
                            && !is_native_stage(&self.source[pair[0].range().as_range()])
                            && !is_reserved(&self.source[pair[0].range().as_range()])
                            && !zirql::is_common_value_call(
                                &self.source[pair[0].range().as_range()],
                            )
                            && !(&self.source[pair[0].range().as_range()] == "axes"
                                && *index > 0
                                && written[*index - 1].kind() == TokenKind::RBracket)
                            && !matches!(
                                &self.source[pair[0].range().as_range()],
                                "default"
                                    | "map"
                                    | "flat_map"
                                    | "div"
                                    | "mod"
                                    | "for"
                                    | "in"
                                    | "if"
                                    | "else"
                                    | "match"
                                    | "present"
                                    | "op"
                                    | "dialect"
                                    | "result_type"
                                    | "has_attr"
                                    | "string_attr_eq"
                            )
                    }) {
                        self.shared_error(zirql::QueryError::query(
                            pair[0].range().as_range(),
                            "unsupported value call",
                        ));
                        return None;
                    }
                }
                self.sync(reader.position());
                Some(super::comprehension::ParsedComprehension::new(value))
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }
    fn shared_projection(
        &mut self,
        depth: usize,
    ) -> Option<zirql::Projection<super::report::Operand>> {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            function_mode: self.function_mode,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
            program: ProgramSyntax::default(),
        };
        let nesting = host.nesting;
        match zirql::read_projection(&mut reader, &mut host, self.budget, nesting) {
            Ok(mut value) => {
                for (_, field) in &mut value.fields {
                    let expression = std::mem::replace(
                        field,
                        zirql::Located::new(
                            zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::Absent),
                            0..0,
                        ),
                    );
                    *field = match super::functions::normalize_outer(
                        expression,
                        &self.bindings,
                        self.function_mode,
                        self.budget,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            self.shared_error(error);
                            return None;
                        }
                    };
                }
                self.sync(reader.position());
                Some(value)
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
        let mut bindings = self.bindings.as_ref().clone();
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
            function_mode: self.function_mode,
            bindings: &bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
            program: ProgramSyntax::default(),
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
    fn shared_paths(
        &mut self,
        depth: usize,
    ) -> Option<zirql::NativePaths<super::model::NativeStage, super::model::NativeStage>> {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            function_mode: self.function_mode,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
            program: ProgramSyntax::default(),
        };
        let nesting = host.nesting;
        match zirql::read_paths(&mut reader, &mut host, self.budget, nesting) {
            Ok(value) => {
                self.sync(reader.position());
                Some(value)
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }
    fn shared_match(
        &mut self,
        depth: usize,
    ) -> Option<
        zirql::NativeMatch<zirql::NativeBody<super::report::Operand>, super::model::NativeStage>,
    > {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            function_mode: self.function_mode,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
            program: ProgramSyntax::default(),
        };
        let nesting = host.nesting;
        match zirql::read_match(&mut reader, &mut host, self.budget, nesting) {
            Ok(value) => {
                self.sync(reader.position());
                Some(value)
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }
    fn starts_shared_value(&mut self, depth: usize) -> bool {
        match super::comprehension::starts(
            self.source,
            self.tokens,
            self.cursor,
            self.budget,
            self.nesting_limit.saturating_sub(depth),
        ) {
            Ok(starts) => starts,
            Err(error) => {
                self.shared_error(error);
                false
            }
        }
    }
    fn ordering_call(&mut self) -> bool {
        let reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .expect("token boundary");
        match zirql::probe_ordering_call(&reader, self.budget, self.nesting_limit) {
            Ok(constructed) => constructed,
            Err(error) => {
                self.shared_error(error);
                false
            }
        }
    }
    fn starts_native(&mut self) -> bool {
        self.skip_trivia();
        if self.at(TokenKind::Identifier) {
            let constructed_ordering = is_native_stage(self.current_text()) && self.ordering_call();
            let name = self.current_text();
            (name == "match" && self.next_kind() == Some(TokenKind::LBrace))
                || zirql::has_native_field_pipeline(
                    &zirql::ArgumentReader::new(
                        self.source,
                        self.current().range().start() as usize,
                    )
                    .expect("token boundary"),
                )
                || is_native_stage(name) && !constructed_ordering
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
        let range = self.span(start);
        let stage = super::model::value(value);
        Some(Expression {
            first: vec![stage],
            rest: vec![],
            range,
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
            let Some(operator) = SetOperator::from_keyword(self.current_text()) else {
                break;
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
                message: "set operands must be selection queries; put edits and count after the grouped set expression".to_owned(),
                range: stage,
            });
        }
        Some(expression)
    }

    fn pipeline(&mut self, depth: usize) -> Option<Vec<super::model::Step>> {
        let mut stages = vec![self.stage(depth)?];
        loop {
            self.skip_trivia();
            if !self.at(TokenKind::Pipe) {
                break;
            }
            self.bump();
            self.skip_trivia();
            if stages.last().is_some_and(super::model::step_is_terminal) {
                self.error("no stage may follow count");
            }
            stages.push(self.stage(depth)?);
        }
        Some(stages)
    }

    fn stage(&mut self, depth: usize) -> Option<super::model::Step> {
        self.stage_inner(depth, true)
    }
    fn stage_inner(&mut self, depth: usize, shared_values: bool) -> Option<super::model::Step> {
        self.skip_trivia();
        let start = self.current().range().start();
        if (matches!(
            self.current_text(),
            "match"
                | "loop"
                | "fold"
                | "scan"
                | "queue"
                | "pop_front"
                | "push_back_all"
                | "identity_set"
                | "insert"
        ) && matches!(
            self.next_kind(),
            Some(TokenKind::LParen | TokenKind::Operator)
        )) || (self.at_identifier("if")
            && self.function_mode
            && self.next_kind().is_some_and(|kind| {
                matches!(
                    kind,
                    TokenKind::Identifier
                        | TokenKind::LParen
                        | TokenKind::Integer
                        | TokenKind::String
                )
            }))
        {
            let value = self.shared_value(depth)?;
            return Some(super::model::value(value));
        }
        if (self.function_mode || super::functions::is_native_function(self.current_text()))
            && self.at(TokenKind::Identifier)
            && self.next_kind() == Some(TokenKind::LParen)
            && !is_native_stage(self.current_text())
            && !matches!(
                self.current_text(),
                "op" | "dialect" | "result_type" | "has_attr" | "string_attr_eq"
            )
        {
            let value = self.shared_value(depth)?;
            return Some(super::model::value(value));
        }
        if self.at_identifier("paths") && self.next_kind() == Some(TokenKind::LParen) {
            let value = self.shared_paths(depth)?;
            return Some(zirql::Located::new(
                zirql::ExpressionStep::Paths(std::sync::Arc::new(value)),
                super::program::range(self.span(start)),
            ));
        }
        if self.at_identifier("match") && self.next_kind() == Some(TokenKind::LBrace) {
            let value = self.shared_match(depth)?;
            return Some(zirql::Located::new(
                zirql::ExpressionStep::Match(std::sync::Arc::new(value)),
                super::program::range(self.span(start)),
            ));
        }
        if self.at(TokenKind::Identifier) && self.next_kind() == Some(TokenKind::Operator) {
            let next = self.tokens[self.cursor + 1..]
                .iter()
                .find(|token| token.kind() != TokenKind::Trivia)?;
            if &self.source[next.range().as_range()] == "."
                && !(shared_values && self.starts_shared_value(depth))
            {
                let value = self.shared_native(depth, true)?;
                return Some(super::model::native(Stage::Native {
                    value,
                    range: self.span(start),
                }));
            }
        }
        if super::comprehension::constructed_filter_call(self.source, start as usize) {
            self.bump();
            let mut reader =
                zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                    .ok()?;
            let mut host = NativeReader {
                source: self.source,
                tokens: self.tokens,
                bindings: &self.bindings,
                nesting: self.nesting_limit.saturating_sub(depth),
                function_mode: self.function_mode,
                program: ProgramSyntax::default(),
            };
            let nesting = host.nesting;
            let locals = self.locals;
            let value = match zirql::read_constructed_filter(
                &mut reader,
                &mut host,
                locals,
                self.budget,
                nesting,
            ) {
                Ok(predicate) => {
                    self.sync(reader.position());
                    let range = start as usize..reader.position();
                    match super::report::lower_suffix(
                        zirql::Located::new(zirql::ValueSuffix::Filter(predicate), range),
                        self.budget,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            self.shared_error(error);
                            return None;
                        }
                    }
                }
                Err(error) => {
                    self.shared_error(error);
                    return None;
                }
            };
            return Some(super::model::value(value));
        }
        if shared_values && self.starts_shared_value(depth) {
            let reader = zirql::ArgumentReader::new(self.source, start as usize).ok()?;
            let postfix_result = zirql::has_named_value_postfix(&reader);
            let value = self.read_shared_value(depth, postfix_result)?;
            return Some(super::model::value(value));
        }
        if self.at_identifier("project") && self.next_kind() == Some(TokenKind::LParen) {
            let value = self.shared_projection(depth)?;
            return Some(zirql::Located::new(
                zirql::ExpressionStep::Projection(std::sync::Arc::new(value)),
                super::program::range(self.span(start)),
            ));
        }
        if self.at(TokenKind::LBrace) || self.at(TokenKind::LBracket) {
            let value = self.json_literal(depth + 1)?;
            return Some(super::model::native(Stage::Literal {
                value,
                range: self.span(start),
            }));
        }
        if self.at(TokenKind::LParen) {
            self.bump();
            let expression = Box::new(self.expression(depth + 1)?);
            self.expect(TokenKind::RParen, "expected `)` after query")?;
            return Some(super::model::native(Stage::Group {
                expression,
                range: self.span(start),
            }));
        }
        let Some(name) = self.identifier_text() else {
            self.error("expected a query stage such as input, filter, or defs");
            return None;
        };
        let range = self.span(start);
        Some(super::model::native(match name.as_str() {
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
            "tally" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Collection(zirql::Stage::Tally),
                    super::program::range(range),
                ));
            }
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
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Mapping(std::sync::Arc::new(zirql::NativeMap {
                        key: zirql::NativeBody::new(
                            super::report::operand(&key),
                            super::program::range(key.range),
                        ),
                        value: zirql::NativeBody::new(
                            super::report::operand(&value),
                            super::program::range(value.range),
                        ),
                    })),
                    super::program::range(self.span(start)),
                ));
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
                        return Some(super::model::native(Stage::Computed {
                            value: if name == "defs" {
                                ComputedStage::Defs(value)
                            } else {
                                ComputedStage::Users(value)
                            },
                            range: self.span(start),
                        }));
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
            "unique" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Collection(zirql::Stage::Unique),
                    super::program::range(range),
                ));
            }
            "sort" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Ordering(std::sync::Arc::new(zirql::NativeOrdering {
                        order: zirql::NativeOrder::Sort,
                        selector: None,
                    })),
                    super::program::range(range),
                ));
            }
            "sort_by" | "min_by" | "min_all_by" | "max_by" | "max_all_by" => {
                self.expect(TokenKind::LParen, "expected `(` after selector stage")?;
                let selector = Box::new(self.expression(depth + 1)?);
                self.expect(TokenKind::RParen, "expected `)` after selector query")?;
                if !selector.is_read_only() {
                    self.error("selector queries cannot edit or emit");
                    return None;
                }
                let order = match name.as_str() {
                    "sort_by" => zirql::NativeOrder::Sort,
                    "min_by" => zirql::NativeOrder::Min,
                    "max_by" => zirql::NativeOrder::Max,
                    "min_all_by" => zirql::NativeOrder::MinAll,
                    "max_all_by" => zirql::NativeOrder::MaxAll,
                    _ => unreachable!(),
                };
                let step = if selector.is_map_value() && order == zirql::NativeOrder::Sort {
                    zirql::ExpressionStep::MapSorting
                } else {
                    zirql::ExpressionStep::Ordering(std::sync::Arc::new(zirql::NativeOrdering {
                        order,
                        selector: Some(zirql::NativeBody::new(
                            super::report::operand(&selector),
                            super::program::range(selector.range),
                        )),
                    }))
                };
                return Some(zirql::Located::new(
                    step,
                    super::program::range(self.span(start)),
                ));
            }
            "reverse" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Collection(zirql::Stage::Reverse),
                    super::program::range(range),
                ));
            }
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
                    let step = if name == "head" {
                        zirql::ExpressionStep::Head(value.expression.clone())
                    } else {
                        zirql::ExpressionStep::Tail(value.expression.clone())
                    };
                    return Some(zirql::Located::new(
                        step,
                        super::program::range(self.span(start)),
                    ));
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
                    return Some(zirql::Located::new(
                        zirql::ExpressionStep::Collection(zirql::Stage::Head(count)),
                        super::program::range(range),
                    ));
                } else {
                    return Some(zirql::Located::new(
                        zirql::ExpressionStep::Collection(zirql::Stage::Tail(count)),
                        super::program::range(range),
                    ));
                }
            }
            "min" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Ordering(std::sync::Arc::new(zirql::NativeOrdering {
                        order: zirql::NativeOrder::Min,
                        selector: None,
                    })),
                    super::program::range(range),
                ));
            }
            "min_all" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Ordering(std::sync::Arc::new(zirql::NativeOrdering {
                        order: zirql::NativeOrder::MinAll,
                        selector: None,
                    })),
                    super::program::range(range),
                ));
            }
            "max" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Ordering(std::sync::Arc::new(zirql::NativeOrdering {
                        order: zirql::NativeOrder::Max,
                        selector: None,
                    })),
                    super::program::range(range),
                ));
            }
            "max_all" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Ordering(std::sync::Arc::new(zirql::NativeOrdering {
                        order: zirql::NativeOrder::MaxAll,
                        selector: None,
                    })),
                    super::program::range(range),
                ));
            }
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
                            return Some(zirql::Located::new(
                                zirql::ExpressionStep::Check {
                                    operand: Some(value.expression.clone()),
                                    message: None,
                                },
                                super::program::range(self.span(start)),
                            ));
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
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Collection(zirql::Stage::Check { expected, message }),
                    super::program::range(self.span(start)),
                ));
            }
            "count" => {
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Collection(zirql::Stage::Count),
                    super::program::range(range),
                ));
            }
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
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::Filter(std::sync::Arc::new(
                        super::model::shared_predicate(predicate),
                    )),
                    super::program::range(self.span(start)),
                ));
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
            "where_exists" => {
                self.expect(TokenKind::LParen, "expected `(` after where_exists")?;
                let mut reader = zirql::ArgumentReader::new(
                    self.source,
                    self.current().range().start() as usize,
                )
                .ok()?;
                let mut host = NativeReader {
                    source: self.source,
                    tokens: self.tokens,
                    function_mode: self.function_mode,
                    bindings: &self.bindings,
                    nesting: self.nesting_limit.saturating_sub(depth + 1),
                    program: ProgramSyntax::default(),
                };
                let nesting = host.nesting;
                let query = match zirql::read_native_expression(
                    &mut reader,
                    &mut host,
                    self.budget,
                    nesting,
                ) {
                    Ok(query) => query,
                    Err(error) => {
                        self.shared_error(error);
                        return None;
                    }
                };
                self.sync(reader.position());
                self.expect(
                    TokenKind::RParen,
                    "expected `)` after where_exists relation",
                )?;
                if !super::report::operand_is_read_only(&query) {
                    self.error("where_exists relation cannot edit or emit");
                    return None;
                }
                let range = super::program::range(self.span(start));
                return Some(zirql::Located::new(
                    zirql::ExpressionStep::WhereExists(std::sync::Arc::new(
                        zirql::NativeBody::new(query, range.clone()),
                    )),
                    range,
                ));
            }
            "fixpoint" => {
                self.expect(TokenKind::LParen, "expected `(` after fixpoint")?;
                let expression = Box::new(self.expression(depth + 1)?);
                self.expect(TokenKind::RParen, "expected `)` after fixpoint query")?;
                if let Some(stage) = expression.first_non_selection_stage() {
                    self.diagnostics.push(Diagnostic {
                        message: "fixpoint requires a selection query without edits or count"
                            .to_owned(),
                        range: stage,
                    });
                }
                let range = super::program::range(self.span(start));
                let step = if expression.is_single_closure() {
                    return Some(super::model::native(Stage::ClosureFixedPoint {
                        range: self.span(start),
                    }));
                } else {
                    zirql::ExpressionStep::FixedPoint(std::sync::Arc::new(zirql::NativeBody::new(
                        super::report::operand(&expression),
                        range.clone(),
                    )))
                };
                return Some(zirql::Located::new(step, range));
            }
            "set_attr" => return self.set_attr(start, depth + 1).map(super::model::native),
            "remove_attr" => return self.remove_attr(start, depth + 1).map(super::model::native),
            "attr" => return self.attr(start, depth + 1).map(super::model::native),
            _ if self.bindings.contains(&name) => Stage::Binding { name, range },
            _ => {
                self.error_at_previous("unknown query stage; expected input, filter, navigation, projection, fixpoint, an edit, check, count, emit, format, or json");
                return None;
            }
        }))
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
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth).saturating_add(1),
            function_mode: self.function_mode,
            program: ProgramSyntax::default(),
        };
        let locals = self.locals;
        let nesting = host.nesting;
        match zirql::read_native_predicate_argument(
            &mut reader,
            &mut host,
            locals,
            self.budget,
            nesting,
        ) {
            Ok(predicate) => {
                self.sync(reader.position());
                Some(lower_shared_predicate(predicate))
            }
            Err(error) => {
                self.shared_error(error);
                None
            }
        }
    }

    fn primary_predicate(&mut self, depth: usize) -> Option<Predicate> {
        self.skip_trivia();
        let start = self.current().range().start();
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
                        message: "operation name must not be empty".to_owned(),
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
                        message: "name or type must not be empty".to_owned(),
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
                message: "attribute name must be a dotted ASCII identifier".to_owned(),
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
                message: "attribute name must be a dotted ASCII identifier".to_owned(),
                range: name_range,
            });
        }
        if value.chars().any(char::is_control) {
            self.diagnostics.push(Diagnostic {
                message: "attribute string must not contain control characters".to_owned(),
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
            message: message.to_owned(),
            range: self.current().range(),
        });
    }
    fn error_at_previous(&mut self, message: &'static str) {
        self.diagnostics.push(Diagnostic {
            message: message.to_owned(),
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

pub(super) fn is_native_stage(name: &str) -> bool {
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
            | "where_exists"
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
            | "paths"
            | "project"
    )
}

pub(super) fn is_reserved(name: &str) -> bool {
    super::functions::is_native_function(name)
        || (zirql::is_common_keyword(name)
            && matches!(
                name,
                "input" | "and" | "or" | "not" | "true" | "false" | "null"
            ))
        || matches!(
            name,
            "do" | "filter"
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
                | "where_exists"
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
                | "op"
                | "dialect"
                | "result_type"
                | "has_attr"
                | "string_attr_eq"
        )
}

// These owned conversions preserve Zirium's public first/rest frontend shape.
// They do not lower native stages or change emission policy.
fn reader_expression(mut expression: Expression) -> super::report::Operand {
    let range = super::program::range(expression.range);
    let pipeline = |steps, range: std::ops::Range<usize>| {
        zirql::Located::new(
            zirql::ExpressionKind::Sequence {
                input: Box::new(zirql::Located::new(
                    zirql::ExpressionKind::Input,
                    range.clone(),
                )),
                steps,
            },
            range,
        )
    };
    let mut value = pipeline(std::mem::take(&mut expression.first), range.clone());
    for (operator, steps) in std::mem::take(&mut expression.rest) {
        value = zirql::Located::new(
            zirql::ExpressionKind::Set {
                operator,
                left: Box::new(value),
                right: Box::new(pipeline(steps, range.clone())),
            },
            range.clone(),
        );
    }
    value
}

fn frontend_expression(mut expression: super::report::Operand) -> Expression {
    let range = TextRange::new(expression.range.start as u32, expression.range.end as u32)
        .expect("ordered frontend range");
    let mut rest = Vec::new();
    loop {
        match expression.value {
            zirql::ExpressionKind::Set {
                operator,
                left,
                right,
            } => {
                rest.push((operator, frontend_pipeline(*right)));
                expression = *left;
            }
            _ => {
                rest.reverse();
                return Expression {
                    first: frontend_pipeline(expression),
                    rest,
                    range,
                };
            }
        }
    }
}

fn frontend_pipeline(expression: super::report::Operand) -> Vec<super::model::Step> {
    let zirql::ExpressionKind::Sequence { input, steps } = expression.value else {
        unreachable!("the program adapter produces a sequence for every frontend pipeline")
    };
    debug_assert!(matches!(input.value, zirql::ExpressionKind::Input));
    steps
}

fn frontend_program(
    parsed: zirql::ReadProgram<super::model::NativeStage, SetOperator, super::program::Emitter>,
    signatures: Vec<FunctionTypes>,
) -> Program {
    let functions = parsed
        .functions
        .into_iter()
        .zip(signatures)
        .map(|(function, signature)| {
            super::FunctionDefinition::parsed(zirql::PatternFunctionDeclaration {
                name: function.name,
                parameters: function
                    .parameters
                    .into_iter()
                    .zip(signature.parameters)
                    .map(|(parameter, declaration)| {
                        zirql::Located::new(
                            zirql::PatternParameter {
                                pattern: parameter.value.pattern,
                                declaration,
                            },
                            parameter.range,
                        )
                    })
                    .collect(),
                result: signature.result,
                value: function.value,
            })
        })
        .collect();
    let (parameters, parameter_positions) = parsed
        .parameters
        .into_iter()
        .map(|parameter| (parameter.declaration, parameter.position))
        .unzip();
    let mut statements = parsed
        .statements
        .into_iter()
        .map(|statement| match statement.value {
            zirql::Statement::Parameter { .. } | zirql::Statement::BindPattern { .. } => {
                unreachable!("frontend retains parameter positions and native pattern syntax")
            }
            zirql::Statement::Bind { name, expression } => Statement::Binding {
                name: name.value,
                expression: frontend_expression(expression),
            },
            zirql::Statement::BindFields { names, expression } => Statement::PatternBinding {
                names,
                expression: frontend_expression(expression),
            },
            zirql::Statement::Evaluate {
                expression,
                implicit_output,
            } => {
                let expression = frontend_expression(expression);
                if implicit_output.is_some() {
                    Statement::Query(expression)
                } else {
                    Statement::Do(expression)
                }
            }
        })
        .collect::<Vec<_>>();
    let (expression, emit_final_result) = match statements.pop() {
        Some(Statement::Query(expression)) => (expression, true),
        Some(Statement::Do(expression)) => (expression, false),
        Some(_) => unreachable!("completed programs end in an evaluation"),
        None => (
            Expression {
                first: Vec::new(),
                rest: Vec::new(),
                range: TextRange::at(parsed.range.end as u32),
            },
            true,
        ),
    };
    Program {
        functions,
        parameters,
        parameter_positions,
        statements,
        expression,
        emit_final_result,
    }
}

struct FunctionTypes {
    parameters: Vec<zirql::DeclaredType<super::program::Kind>>,
    result: zirql::Located<zirql::DeclaredType<super::program::Kind>>,
}

#[derive(Default)]
enum BindingSyntax {
    Legacy,
    #[default]
    Name,
    Pattern,
}

#[derive(Default)]
struct ProgramSyntax {
    signatures: Vec<FunctionTypes>,
    binding: BindingSyntax,
    purpose: Option<zirql::ProgramExpression>,
    needs_final_expression: bool,
    function_scope: bool,
}

struct NativeReader<'a> {
    source: &'a str,
    tokens: &'a [Token],
    bindings: &'a HashSet<String>,
    nesting: usize,
    function_mode: bool,
    program: ProgramSyntax,
}
impl NativeReader<'_> {
    fn parser<'a, 'b>(
        &'a self,
        position: usize,
        locals: &'a [&'a str],
        budget: &'b mut zirql::Budget,
    ) -> Result<Parser<'a, 'b>, zirql::QueryError> {
        let location = zirql::Location::Query(position..position);
        let mut bindings = if self.program.function_scope {
            Cow::Owned(HashSet::new())
        } else {
            Cow::Borrowed(self.bindings)
        };
        for &name in locals {
            if bindings.contains(name) {
                continue;
            }
            if let Cow::Borrowed(existing) = &bindings {
                budget.charge(
                    existing.iter().map(|name| name.len() + 1).sum(),
                    location.clone(),
                )?;
                budget.charge_results(existing.len(), location.clone())?;
            }
            budget.charge(name.len() + 1, location.clone())?;
            budget.charge_results(1, location.clone())?;
            bindings.to_mut().insert(name.to_owned());
        }
        Ok(Parser {
            source: self.source,
            tokens: self.tokens,
            function_mode: self.function_mode,
            cursor: self
                .tokens
                .partition_point(|token| (token.range().start() as usize) < position),
            diagnostics: vec![],
            nesting_limit: self.nesting,
            bindings,
            locals,
            budget,
        })
    }
    fn error(diagnostic: &Diagnostic) -> zirql::QueryError {
        zirql::QueryError::query(diagnostic.range().as_range(), diagnostic.message())
    }
}
impl zirql::ProgramReaderHost for NativeReader<'_> {
    fn function_name(
        &mut self,
        _: zirql::Located<&str>,
        _: &mut zirql::Budget,
    ) -> Result<(), zirql::QueryError> {
        self.function_mode = true;
        self.program.needs_final_expression = true;
        Ok(())
    }
    fn function_scope(&mut self, active: bool) {
        self.program.function_scope = active;
    }
    fn admit_functions(
        &mut self,
        functions: &mut [zirql::UnresolvedPatternFunctionDeclaration<super::report::Operand>],
        budget: &mut zirql::Budget,
    ) -> Result<(), zirql::QueryError> {
        let ambient = HashSet::new();
        for function in functions {
            let signature = function.resolve_signature(budget, super::functions::resolve_type)?;
            budget.charge_results(
                signature.parameters.len(),
                zirql::Location::Query(signature.name.range.clone()),
            )?;
            self.program.signatures.push(FunctionTypes {
                parameters: signature
                    .parameters
                    .into_iter()
                    .map(|parameter| parameter.value.declaration)
                    .collect(),
                result: signature.result,
            });
            let value = std::mem::replace(
                &mut function.value,
                zirql::Located::new(
                    zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::Absent),
                    0..0,
                ),
            );
            function.value = super::functions::normalize_outer(value, &ambient, true, budget)?;
        }
        Ok(())
    }
    fn admit_name(
        &mut self,
        name: zirql::Located<&str>,
        parameter: bool,
    ) -> Result<(), zirql::QueryError> {
        if is_reserved(name.value) {
            return Err(zirql::QueryError::query(
                name.range.clone(),
                if parameter {
                    "parameter name is reserved or already defined"
                } else {
                    "binding name is reserved or already defined"
                },
            ));
        }
        Ok(())
    }
    fn read_binding(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<Option<zirql::Pattern>, zirql::QueryError> {
        self.program.binding = BindingSyntax::Name;
        let mut probe = zirql::ArgumentReader::new(reader.source(), reader.position())?;
        let Ok(name) = probe.word() else {
            return Ok(None);
        };
        if !probe.consume('=') {
            return Ok(None);
        }
        budget.charge(
            name.value.len() + 1,
            zirql::Location::Query(name.range.clone()),
        )?;
        budget.charge_results(2, zirql::Location::Query(name.range.clone()))?;
        let name = zirql::Located::new(name.value.to_owned(), name.range);
        let range = name.range.clone();
        reader.advance(probe.position() - reader.position())?;
        self.program.binding = BindingSyntax::Legacy;
        Ok(Some(zirql::Located::new(
            zirql::PatternKind::Name(name),
            range,
        )))
    }
    fn binding_pattern(&mut self, pattern: &zirql::Pattern) {
        if !matches!(pattern.value, zirql::PatternKind::Name(_)) {
            self.program.binding = BindingSyntax::Pattern;
            self.function_mode = true;
        }
    }
    fn read_output_prefix(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
    ) -> Option<zirql::ProgramExpression> {
        reader
            .keyword("do")
            .then_some(zirql::ProgramExpression::Suppressed)
    }
    fn read_program_expression(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        names: &[&str],
        purpose: zirql::ProgramExpression,
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<super::report::Operand, zirql::QueryError> {
        self.program.purpose = Some(purpose);
        let mut parser = self.parser(reader.position(), names, budget)?;
        parser.nesting_limit = nesting;
        let expression = match purpose {
            zirql::ProgramExpression::Binding
                if matches!(self.program.binding, BindingSyntax::Pattern) =>
            {
                parser.value_expression(0)
            }
            zirql::ProgramExpression::Binding
                if matches!(self.program.binding, BindingSyntax::Name)
                    && !parser.starts_native() =>
            {
                parser.value_expression(0)
            }
            zirql::ProgramExpression::Report => {
                parser.value_expression(0).and_then(|mut expression| {
                    parser.shared_suffix(&mut expression, 0)?;
                    Some(expression)
                })
            }
            _ => parser.expression(0),
        };
        if let Some(diagnostic) = parser.diagnostics.first() {
            return Err(Self::error(diagnostic));
        }
        let expression = expression.ok_or_else(|| reader.error("expected a query expression"))?;
        if purpose == zirql::ProgramExpression::Binding && !expression.is_read_only() {
            return Err(
                reader.error("bindings cannot edit or emit; use a separate query statement")
            );
        }
        reader.advance(parser.current().range().start() as usize - reader.position())?;
        self.program.needs_final_expression = purpose == zirql::ProgramExpression::Binding;
        Ok(reader_expression(expression))
    }
    fn parameter_statement(
        &mut self,
        _: &zirql::ParameterDeclaration,
        _: usize,
        _: &mut zirql::Budget,
    ) -> Result<Option<zirql::Located<zirql::ReadStatement<Self>>>, zirql::QueryError> {
        self.program.needs_final_expression = true;
        Ok(None)
    }
    fn binding_statement(
        &mut self,
        pattern: zirql::Pattern,
        names: Vec<zirql::Located<String>>,
        expression: super::report::Operand,
        _: &mut zirql::Budget,
    ) -> Result<
        zirql::Statement<super::model::NativeStage, SetOperator, super::program::Emitter>,
        zirql::QueryError,
    > {
        if !matches!(self.program.binding, BindingSyntax::Pattern) {
            let zirql::PatternKind::Name(name) = pattern.value else {
                unreachable!("simple binding pattern")
            };
            return Ok(zirql::Statement::Bind { name, expression });
        }
        let range = pattern.range.start..expression.range.end;
        let mut steps = frontend_pipeline(expression);
        let zirql::ExpressionStep::Value(value) =
            steps.pop().expect("one parsed binding value").value
        else {
            unreachable!("pattern binding reads a value expression")
        };
        let value = super::functions::ParsedBinding::new(
            pattern,
            super::comprehension::ParsedComprehension { expression: value },
        );
        let range =
            TextRange::new(range.start as u32, range.end as u32).expect("ordered binding range");
        let expression = reader_expression(Expression {
            first: vec![super::model::native(Stage::PatternBinding { value, range })],
            rest: Vec::new(),
            range,
        });
        Ok(zirql::Statement::BindFields { names, expression })
    }
    fn implicit_output(
        &mut self,
        purpose: zirql::ProgramExpression,
    ) -> Option<super::program::Emitter> {
        (purpose != zirql::ProgramExpression::Suppressed).then_some(super::program::Emitter::Result)
    }
    fn finish_program(
        &mut self,
        reader: &zirql::ArgumentReader<'_>,
        _: &mut zirql::Budget,
    ) -> Result<(), zirql::QueryError> {
        if self.program.needs_final_expression {
            return Err(reader.error("expected a query stage such as input, filter, or defs"));
        }
        Ok(())
    }
}

impl zirql::ValueReaderHost for NativeReader<'_> {
    type Stage = super::model::NativeStage;
    type SetOperator = SetOperator;
    type Emitter = super::program::Emitter;
    fn retains_native_values(&self) -> bool {
        self.function_mode
    }
    fn field_pipeline_source(&self) -> zirql::FieldPipelineSource {
        zirql::FieldPipelineSource::ValueArgument
    }
    fn is_binding(&self, name: &str) -> bool {
        !self.program.function_scope && self.bindings.contains(name)
    }
    fn is_stage(&self, name: &str) -> bool {
        is_native_stage(name)
    }
    fn parse_stage(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<
        zirql::ExpressionStep<super::model::NativeStage, super::program::Emitter, SetOperator>,
        zirql::QueryError,
    > {
        self.parse_stage_in_scope(reader, &[], budget, self.nesting)
    }
    fn parse_stage_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<
        zirql::ExpressionStep<super::model::NativeStage, super::program::Emitter, SetOperator>,
        zirql::QueryError,
    > {
        let mut parser = self.parser(reader.position(), locals, budget)?;
        parser.nesting_limit = nesting;
        // The common reader has selected a native leaf; do not re-enter it.
        let result = parser.stage_inner(0, false);
        if let Some(diagnostic) = parser.diagnostics.first() {
            return Err(Self::error(diagnostic));
        }
        let stage = result.ok_or_else(|| reader.error("expected a native stage"))?;
        reader.advance(stage.range.end - reader.position())?;
        Ok(stage.value)
    }
    fn parse_constructed_filter_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Option<zirql::ReadStep<Self>>, zirql::QueryError> {
        if !super::comprehension::constructed_filter_call(reader.source(), reader.position()) {
            return Ok(None);
        }
        self.parse_stage_in_scope(reader, locals, budget, nesting)
            .map(Some)
    }
    fn parse_predicate_leaf(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<super::model::NativeStage, zirql::QueryError> {
        self.parse_predicate_leaf_in_scope(reader, &[], budget, self.nesting)
    }
    fn parse_predicate_leaf_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<super::model::NativeStage, zirql::QueryError> {
        let mut parser = self.parser(reader.position(), locals, budget)?;
        parser.nesting_limit = nesting;
        let result = parser.primary_predicate(0);
        if let Some(diagnostic) = parser.diagnostics.first() {
            return Err(Self::error(diagnostic));
        }
        let predicate = result.ok_or_else(|| reader.error("expected a native predicate"))?;
        reader.advance(predicate.range().end() as usize - reader.position())?;
        Ok(std::sync::Arc::new(Stage::Guard {
            range: predicate.range(),
            predicate,
        }))
    }
    fn is_predicate_leaf(&self, name: &str) -> bool {
        matches!(
            name,
            "true"
                | "false"
                | "op"
                | "dialect"
                | "result_type"
                | "has_attr"
                | "string_attr_eq"
                | "same_value"
        )
    }
    fn predicate_expression(
        &mut self,
        predicate: zirql::PredicateArgument<super::report::Operand>,
        range: std::ops::Range<usize>,
    ) -> Result<super::model::NativeStage, zirql::QueryError> {
        let subject = predicate_input(range.clone());
        Ok(std::sync::Arc::new(Stage::Guard {
            predicate: Predicate::Expression {
                value: super::comprehension::ParsedComprehension::new(predicate.bind(subject)),
                range: TextRange::new(range.start as u32, range.end as u32).expect("ordered range"),
            },
            range: TextRange::new(range.start as u32, range.end as u32).expect("ordered range"),
        }))
    }
    fn parse_predicate_leaf_in_argument(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        argument: Option<&zirql::Pattern>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<super::model::NativeStage, zirql::QueryError> {
        let payload = self.parse_predicate_leaf_in_scope(reader, locals, budget, nesting)?;
        let Stage::Guard {
            mut predicate,
            range,
        } = std::sync::Arc::try_unwrap(payload).expect("fresh predicate")
        else {
            unreachable!()
        };
        let mut bind = |value: &mut super::comprehension::ParsedComprehension| -> Result<(), zirql::QueryError> {
            let argument = argument.map(|pattern| zirql::copy_pattern(pattern, budget)).transpose()?;
            let body = std::mem::replace(&mut value.expression, std::sync::Arc::new(zirql::Located::new(zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::Absent), 0..0)));
            let body = std::sync::Arc::try_unwrap(body).expect("fresh parsed argument");
            let subject = predicate_input(body.range.clone());
            value.expression = std::sync::Arc::new(zirql::PredicateArgument { argument, body: Box::new(body) }.bind_value(subject));
            Ok(())
        };
        match &mut predicate {
            Predicate::SameValue { left, right, .. } => {
                bind(&mut left.value)?;
                bind(&mut right.value)?;
            }
            Predicate::OpValue { name, .. } => bind(name)?,
            Predicate::Computed { value, .. } => match value {
                ComputedPredicate::Dialect(value)
                | ComputedPredicate::ResultType(value)
                | ComputedPredicate::HasAttr(value) => bind(value)?,
                ComputedPredicate::Attr { name, value } => {
                    bind(name)?;
                    bind(value)?;
                }
            },
            _ => {}
        }
        Ok(std::sync::Arc::new(Stage::Guard { predicate, range }))
    }
    fn parse_path_edge(
        &mut self,
        edge: zirql::Located<&str>,
        _: &mut zirql::Budget,
    ) -> Result<super::model::NativeStage, zirql::QueryError> {
        if edge.value != "users" {
            return Err(zirql::QueryError::query(
                edge.range,
                "unsupported path edge",
            ));
        }
        Ok(std::sync::Arc::new(Stage::Users {
            index: None,
            range: TextRange::new(edge.range.start as u32, edge.range.end as u32)
                .expect("ordered edge range"),
        }))
    }
    fn set_operator(&self, word: &str) -> Option<SetOperator> {
        SetOperator::from_keyword(word)
    }
}
fn lower_shared_predicate(predicate: zirql::Predicate<super::model::NativeStage>) -> Predicate {
    let range = TextRange::new(predicate.range.start as u32, predicate.range.end as u32)
        .expect("shared predicate ranges are ordered source offsets");
    match predicate.value {
        zirql::PredicateKind::Leaf(predicate) => {
            let Stage::Guard { predicate, .. } =
                std::sync::Arc::try_unwrap(predicate).expect("fresh guard leaf")
            else {
                unreachable!()
            };
            predicate
        }
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

fn predicate_input(
    range: std::ops::Range<usize>,
) -> zirql::ValueExpression<super::report::Operand> {
    zirql::Located::new(
        zirql::ValueExpressionKind::NativeValue(Box::new(zirql::Located::new(
            zirql::ExpressionKind::Input,
            range.clone(),
        ))),
        range,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_error_categories_keep_parsing_wording_and_informative_fallback() {
        let lexed = super::super::lexer::lex("");
        let mut budget = zirql::Budget::for_program(super::super::comprehension::default_limits());
        let mut parser = Parser {
            source: lexed.source(),
            tokens: lexed.tokens(),
            function_mode: false,
            locals: &[],
            cursor: 0,
            diagnostics: Vec::new(),
            nesting_limit: DEFAULT_NESTING_LIMIT,
            bindings: Cow::Owned(HashSet::new()),
            budget: &mut budget,
        };
        parser.shared_error(
            zirql::QueryError::query(7..12, "changed work refusal")
                .with_category(zirql::QueryErrorCategory::WorkLimit),
        );
        parser.shared_error(
            zirql::QueryError::query(8..13, "changed result refusal")
                .with_category(zirql::QueryErrorCategory::ResultLimit),
        );
        parser.shared_error(
            zirql::QueryError::query(9..14, "changed null refusal")
                .with_category(zirql::QueryErrorCategory::UnsupportedNull),
        );
        parser.shared_error(zirql::QueryError::source(20..25, "specific source failure"));
        parser.shared_error(zirql::QueryError::query(
            30..35,
            "null is unsupported in constructed values",
        ));
        assert_eq!(
            parser
                .diagnostics
                .iter()
                .map(|diagnostic| (diagnostic.message(), diagnostic.range().as_range()))
                .collect::<Vec<_>>(),
            vec![
                ("query work limit exceeded during parsing", 7..7),
                ("result limit exceeded during parsing", 8..8),
                ("null is unsupported in positional comprehensions", 9..9),
                ("specific source failure", 20..20),
                ("null is unsupported in constructed values", 30..30),
            ]
        );
    }
}
