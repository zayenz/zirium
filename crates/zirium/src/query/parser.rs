use std::collections::HashSet;

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
        function_mode: has_function_header(lexed) || has_control_values(lexed),
        cursor: 0,
        diagnostics: Vec::new(),
        nesting_limit,
        budget: &mut budget,
        locals: Vec::new(),
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

fn has_function_header(lexed: &Lexed<'_>) -> bool {
    let mut tokens = lexed
        .tokens()
        .iter()
        .filter(|token| token.kind() != TokenKind::Trivia)
        .peekable();
    while let Some(token) = tokens.next() {
        if token.kind() == TokenKind::Identifier
            && &lexed.source()[token.range().as_range()] == "fn"
            && tokens
                .peek()
                .is_some_and(|next| next.kind() == TokenKind::Identifier)
        {
            let mut probe = tokens.clone();
            probe.next();
            if probe
                .next()
                .is_some_and(|next| next.kind() == TokenKind::LParen)
            {
                return true;
            }
        }
    }
    false
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
                if matches!(
                    word,
                    "match"
                        | "loop"
                        | "fold"
                        | "scan"
                        | "queue"
                        | "pop_front"
                        | "push_back_all"
                        | "identity_set"
                        | "insert"
                        | "same_operation"
                        | "user_info"
                ) && matches!(next, Some(TokenKind::LParen | TokenKind::Operator))
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
    bindings: HashSet<String>,
    budget: &'b mut zirql::Budget,
    locals: Vec<String>,
}

impl Parser<'_, '_> {
    fn program(&mut self) -> Option<Program> {
        self.skip_trivia();
        let mut parameters = Vec::new();
        let mut functions = Vec::new();
        let mut parameter_positions = Vec::new();
        if self.at(TokenKind::Eof) {
            return Some(Program {
                functions,
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
            if self.at_identifier("fn") {
                let mut reader = zirql::ArgumentReader::new(
                    self.source,
                    self.current().range().start() as usize,
                )
                .ok()?;
                let mut host = NativeReader {
                    source: self.source,
                    tokens: self.tokens,
                    bindings: &self.bindings,
                    nesting: self.nesting_limit,
                    function_mode: true,
                };
                match zirql::read_pattern_function_declaration(
                    &mut reader,
                    &mut host,
                    self.budget,
                    self.nesting_limit,
                    super::functions::resolve_type,
                ) {
                    Ok(Some(mut declaration)) => {
                        declaration.value = match super::functions::normalize_outer(
                            declaration.value,
                            &self.bindings,
                            self.budget,
                        ) {
                            Ok(value) => value,
                            Err(error) => {
                                self.shared_error(error);
                                return None;
                            }
                        };
                        self.sync(reader.position());
                        self.function_mode = true;
                        functions.push(super::FunctionDefinition::parsed(declaration));
                        continue;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.shared_error(error);
                        return None;
                    }
                }
            }
            if self.at_identifier("do") {
                self.bump();
                let expression = self.expression(0)?;
                self.expect(TokenKind::Semicolon, "expected `;` after do statement")?;
                self.skip_trivia();
                if self.at(TokenKind::Eof) {
                    return Some(Program {
                        functions,
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
                    Ok(Some(zirql::CommonStatement::Let(name)))
                        if !matches!(name.value.as_str(), "_" | "true" | "false") =>
                    {
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
                                functions,
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
                    Ok(None) | Ok(Some(zirql::CommonStatement::Let(_))) => {}
                    Err(error) => {
                        self.shared_error(error);
                        return None;
                    }
                }
            }
            if self.at_identifier("let") && self.next_kind() != Some(TokenKind::Equals) {
                let mut reader = zirql::ArgumentReader::new(
                    self.source,
                    self.current().range().start() as usize,
                )
                .ok()?;
                match zirql::read_pattern_let(&mut reader, self.budget, self.nesting_limit) {
                    Ok(Some(pattern)) => {
                        let names = match zirql::pattern_names(&pattern, self.budget) {
                            Ok(names) => names.into_iter().cloned().collect::<Vec<_>>(),
                            Err(error) => {
                                zirql::dispose_pattern(pattern);
                                self.shared_error(error);
                                return None;
                            }
                        };
                        let mut seen = HashSet::new();
                        for name in &names {
                            if is_reserved(&name.value)
                                || self.bindings.contains(&name.value)
                                || !seen.insert(&name.value)
                            {
                                self.sync(name.range.start);
                                self.error("binding name is reserved or already defined");
                                zirql::dispose_pattern(pattern);
                                return None;
                            }
                        }
                        self.sync(reader.position());
                        self.function_mode = true;
                        let Some(value) = self.shared_value(0) else {
                            zirql::dispose_pattern(pattern);
                            return None;
                        };
                        let range = TextRange::new(
                            pattern.range.start as u32,
                            value.expression.range.end as u32,
                        )
                        .expect("ordered binding range");
                        let value = super::functions::ParsedBinding::new(pattern, value);
                        self.expect(TokenKind::Semicolon, "expected `;` after binding")?;
                        for name in &names {
                            self.bindings.insert(name.value.clone());
                        }
                        statements.push(Statement::PatternBinding {
                            names,
                            expression: Expression {
                                first: vec![Stage::PatternBinding { value, range }],
                                rest: Vec::new(),
                                range,
                            },
                        });
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
                functions,
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
        };
        let nesting = host.nesting;
        let locals: Vec<_> = self.locals.iter().map(String::as_str).collect();
        match zirql::read_value_expression_in_scope(
            &mut reader,
            &mut host,
            &locals,
            self.budget,
            nesting,
        ) {
            Ok(value) => {
                let value = if self.function_mode {
                    match super::functions::normalize_outer(value, &self.bindings, self.budget) {
                        Ok(value) => value,
                        Err(error) => {
                            self.shared_error(error);
                            return None;
                        }
                    }
                } else {
                    value
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
    fn shared_projection(&mut self, depth: usize) -> Option<super::report::ParsedProjection> {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            function_mode: self.function_mode,
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
            function_mode: self.function_mode,
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
    fn shared_paths(&mut self, depth: usize) -> Option<super::capture::ParsedPaths> {
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            function_mode: self.function_mode,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth),
        };
        let nesting = host.nesting;
        match zirql::read_paths(&mut reader, &mut host, self.budget, nesting) {
            Ok(value) => {
                self.sync(reader.position());
                Some(super::capture::ParsedPaths::new(value))
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
            function_mode: self.function_mode,
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
                    && !super::comprehension::ordering_call(
                        self.source,
                        self.current().range().start() as usize,
                    )
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
        let stage = if self.function_mode {
            Stage::FunctionValue { value, range }
        } else {
            Stage::Comprehension { value, range }
        };
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
        self.stage_inner(depth, true)
    }
    fn stage_inner(&mut self, depth: usize, shared_values: bool) -> Option<Stage> {
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
            return Some(Stage::FunctionValue {
                value,
                range: self.span(start),
            });
        }
        if matches!(
            self.current_text(),
            "direct_children" | "cost_annotation" | "same_operation" | "user_info"
        ) && self.next_kind() == Some(TokenKind::LParen)
        {
            let primitive = match self.current_text() {
                "direct_children" => super::functions::Primitive::DirectChildren,
                "cost_annotation" => super::functions::Primitive::CostAnnotation,
                "same_operation" => super::functions::Primitive::SameOperation,
                "user_info" => super::functions::Primitive::UserInfo,
                _ => unreachable!("recognized primitive"),
            };
            self.bump();
            self.expect(TokenKind::LParen, "expected `(`")?;
            let mut argument = self.shared_value(depth + 1)?;
            if primitive == super::functions::Primitive::SameOperation {
                self.expect(TokenKind::Comma, "expected `,` between Operations")?;
                let mut right = self.shared_value(depth + 1)?;
                let take = |value: &mut super::comprehension::ParsedComprehension| {
                    std::mem::replace(
                        std::sync::Arc::get_mut(&mut value.expression)
                            .expect("fresh parser expression"),
                        zirql::Located::new(
                            zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::Absent),
                            0..0,
                        ),
                    )
                };
                let values = vec![take(&mut argument), take(&mut right)];
                argument = super::comprehension::ParsedComprehension::new(zirql::Located::new(
                    zirql::ValueExpressionKind::Tuple(values),
                    start as usize..self.current().range().start() as usize,
                ));
            }
            self.expect(TokenKind::RParen, "expected `)` after native argument")?;
            return Some(Stage::FunctionPrimitive {
                primitive,
                argument,
                range: self.span(start),
            });
        }
        if self.function_mode
            && self.at(TokenKind::Identifier)
            && self.next_kind() == Some(TokenKind::LParen)
            && !is_native_stage(self.current_text())
            && !matches!(
                self.current_text(),
                "op" | "dialect" | "result_type" | "has_attr" | "string_attr_eq"
            )
        {
            let value = self.shared_value(depth)?;
            return Some(Stage::FunctionValue {
                value,
                range: self.span(start),
            });
        }
        if self.at_identifier("paths") && self.next_kind() == Some(TokenKind::LParen) {
            let value = self.shared_paths(depth)?;
            return Some(Stage::Paths {
                value,
                range: self.span(start),
            });
        }
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
            if &self.source[next.range().as_range()] == "."
                && !(shared_values
                    && super::comprehension::starts(self.source, self.tokens, self.cursor))
            {
                let value = self.shared_native(depth, true)?;
                return Some(Stage::Native {
                    value,
                    range: self.span(start),
                });
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
            };
            let nesting = host.nesting;
            let locals: Vec<_> = self.locals.iter().map(String::as_str).collect();
            let value = match zirql::read_constructed_filter(
                &mut reader,
                &mut host,
                &locals,
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
            return Some(Stage::ConstructedSuffix {
                value,
                range: self.span(start),
            });
        }
        if shared_values && super::comprehension::starts(self.source, self.tokens, self.cursor) {
            // Named postfix results leave subsequent stages to the host's
            // native/constructed routing, just as a saved result does.
            let mut prefix = self.tokens[self.cursor..]
                .iter()
                .filter(|token| token.kind() != TokenKind::Trivia);
            let name = prefix.find(|token| token.kind() != TokenKind::LParen);
            let postfix = prefix.find(|token| token.kind() != TokenKind::RParen);
            let postfix_result = name.is_some_and(|token| token.kind() == TokenKind::Identifier)
                && postfix.is_some_and(|token| {
                    token.kind() == TokenKind::Operator
                        && &self.source[token.range().as_range()] == "."
                });
            let value = self.read_shared_value(depth, postfix_result)?;
            let range = self.span(start);
            return Some(if self.function_mode {
                Stage::FunctionValue { value, range }
            } else {
                Stage::Comprehension { value, range }
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
                let selector = super::report::OrderingSelector::new(&selector);
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
                filter_stage(predicate, self.span(start))
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
                let relation = super::report::native_selector(query);
                if !super::report::native_body_is_read_only(&relation) {
                    self.error("where_exists relation cannot edit or emit");
                    return None;
                }
                Stage::WhereExists {
                    relation,
                    range: self.span(start),
                }
            }
            "fixpoint" => {
                self.expect(TokenKind::LParen, "expected `(` after fixpoint")?;
                let expression = Box::new(self.expression(depth + 1)?);
                self.expect(TokenKind::RParen, "expected `)` after fixpoint query")?;
                if let Some(stage) = expression.first_non_selection_stage() {
                    self.diagnostics.push(Diagnostic {
                        message: "fixpoint requires a selection query without edits or count"
                            .to_owned(),
                        range: stage.range(),
                    });
                }
                Stage::Fixpoint {
                    expression: super::model::FixedPointBody::Selection(expression),
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
        let mut reader =
            zirql::ArgumentReader::new(self.source, self.current().range().start() as usize)
                .ok()?;
        let mut host = NativeReader {
            source: self.source,
            tokens: self.tokens,
            bindings: &self.bindings,
            nesting: self.nesting_limit.saturating_sub(depth).saturating_add(1),
            function_mode: self.function_mode,
        };
        let locals: Vec<_> = self.locals.iter().map(String::as_str).collect();
        let nesting = host.nesting;
        match zirql::read_native_predicate_argument(
            &mut reader,
            &mut host,
            &locals,
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
            | "direct_children"
            | "cost_annotation"
            | "same_operation"
            | "user_info"
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
    matches!(
        name,
        "input"
            | "direct_children"
            | "cost_annotation"
            | "same_operation"
            | "user_info"
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
    function_mode: bool,
}
impl NativeReader<'_> {
    fn parser<'b>(
        &self,
        position: usize,
        locals: &[&str],
        budget: &'b mut zirql::Budget,
    ) -> Result<Parser<'_, 'b>, zirql::QueryError> {
        let location = zirql::Location::Query(position..position);
        budget.charge(
            self.bindings.iter().map(|name| name.len() + 1).sum(),
            location.clone(),
        )?;
        let mut bindings = self.bindings.clone();
        for &name in locals {
            if !bindings.contains(name) {
                budget.charge(name.len() + 1, location.clone())?;
                bindings.insert(name.to_owned());
            }
        }
        budget.charge(
            locals.iter().map(|name| name.len() + 1).sum(),
            location.clone(),
        )?;
        budget.charge_results(locals.len(), location)?;
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
            locals: locals.iter().map(|name| (*name).to_owned()).collect(),
            budget,
        })
    }
    fn error(diagnostic: &Diagnostic) -> zirql::QueryError {
        zirql::QueryError::query(diagnostic.range().as_range(), diagnostic.message())
    }
}
impl zirql::ValueReaderHost for NativeReader<'_> {
    type Stage = Stage;
    type SetOperator = SetOperator;
    type Emitter = ();
    type Predicate = Predicate;
    fn retains_native_values(&self) -> bool {
        self.function_mode
    }
    fn field_pipeline_source(&self) -> zirql::FieldPipelineSource {
        zirql::FieldPipelineSource::ValueArgument
    }
    fn value_source(
        &mut self,
        value: zirql::ValueExpression<super::report::Operand>,
        range: std::ops::Range<usize>,
    ) -> Result<Stage, zirql::QueryError> {
        Ok(Stage::FunctionValue {
            value: super::comprehension::ParsedComprehension::new(value),
            range: TextRange::new(range.start as u32, range.end as u32).expect("ordered range"),
        })
    }
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
        self.parse_stage_in_scope(reader, &[], budget, self.nesting)
    }
    fn parse_stage_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<zirql::ExpressionStep<Stage, ()>, zirql::QueryError> {
        let mut parser = self.parser(reader.position(), locals, budget)?;
        parser.nesting_limit = nesting;
        // The common reader has selected a native leaf; do not re-enter it.
        let result = parser.stage_inner(0, false);
        if let Some(diagnostic) = parser.diagnostics.first() {
            return Err(Self::error(diagnostic));
        }
        let stage = result.ok_or_else(|| reader.error("expected a native stage"))?;
        reader.advance(stage.range().end() as usize - reader.position())?;
        Ok(zirql::ExpressionStep::Stage(stage))
    }
    fn parse_constructed_filter_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Option<Stage>, zirql::QueryError> {
        if !super::comprehension::constructed_filter_call(reader.source(), reader.position()) {
            return Ok(None);
        }
        match self.parse_stage_in_scope(reader, locals, budget, nesting)? {
            zirql::ExpressionStep::Stage(stage) => Ok(Some(stage)),
            zirql::ExpressionStep::Emit { .. } => {
                unreachable!("constructed filter is a value stage")
            }
        }
    }
    fn parse_predicate_leaf(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<Predicate, zirql::QueryError> {
        self.parse_predicate_leaf_in_scope(reader, &[], budget, self.nesting)
    }
    fn parse_predicate_leaf_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Predicate, zirql::QueryError> {
        let mut parser = self.parser(reader.position(), locals, budget)?;
        parser.nesting_limit = nesting;
        let result = parser.primary_predicate(0);
        if let Some(diagnostic) = parser.diagnostics.first() {
            return Err(Self::error(diagnostic));
        }
        let predicate = result.ok_or_else(|| reader.error("expected a native predicate"))?;
        reader.advance(predicate.range().end() as usize - reader.position())?;
        Ok(predicate)
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
    ) -> Result<Predicate, zirql::QueryError> {
        let subject = predicate_input(range.clone());
        Ok(Predicate::Expression {
            value: super::comprehension::ParsedComprehension::new(predicate.bind(subject)),
            range: TextRange::new(range.start as u32, range.end as u32).expect("ordered range"),
        })
    }
    fn parse_predicate_leaf_in_argument(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        argument: Option<&zirql::Pattern>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Predicate, zirql::QueryError> {
        let mut predicate = self.parse_predicate_leaf_in_scope(reader, locals, budget, nesting)?;
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
        Ok(predicate)
    }
    fn filter(
        &mut self,
        predicate: zirql::Predicate<Predicate>,
        range: std::ops::Range<usize>,
    ) -> Stage {
        filter_stage(
            lower_shared_predicate(predicate),
            TextRange::new(range.start as u32, range.end as u32).expect("ordered query range"),
        )
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
    fn where_exists(
        &mut self,
        query: super::report::Operand,
        range: std::ops::Range<usize>,
    ) -> Result<Stage, zirql::QueryError> {
        let relation = super::report::native_selector(query);
        if !super::report::native_body_is_read_only(&relation) {
            return Err(zirql::QueryError::query(
                range,
                "where_exists relation cannot edit or emit",
            ));
        }
        Ok(Stage::WhereExists {
            relation,
            range: TextRange::new(range.start as u32, range.end as u32)
                .expect("ordered query range"),
        })
    }
    fn fixed_point(
        &mut self,
        query: super::report::Operand,
        range: std::ops::Range<usize>,
    ) -> Result<Stage, zirql::QueryError> {
        let selector = super::report::OrderingSelector::from_operand(query);
        if !selector.is_read_only() {
            return Err(zirql::QueryError::query(
                range,
                "fixpoint body cannot edit or emit",
            ));
        }
        let super::report::OrderingSelector::Query(value) = selector else {
            unreachable!("native fixed-point body");
        };
        Ok(Stage::Fixpoint {
            expression: super::model::FixedPointBody::Native(value),
            range: TextRange::new(range.start as u32, range.end as u32)
                .expect("ordered query range"),
        })
    }
    fn map_sorting(&mut self, range: std::ops::Range<usize>) -> Result<Stage, zirql::QueryError> {
        let range =
            TextRange::new(range.start as u32, range.end as u32).expect("ordered query range");
        Ok(Stage::SortBy {
            selector: super::report::OrderingSelector::MapValue(range),
            range,
        })
    }
    fn ordering(
        &mut self,
        ordering: zirql::NativeOrdering<super::report::Operand>,
        range: std::ops::Range<usize>,
    ) -> Result<Stage, zirql::QueryError> {
        use zirql::NativeOrder as O;
        let range =
            TextRange::new(range.start as u32, range.end as u32).expect("ordered query range");
        let Some(selector) = ordering.selector else {
            return Ok(match ordering.order {
                O::Sort => Stage::Sort { range },
                O::Min => Stage::Min { range },
                O::Max => Stage::Max { range },
                O::MinAll => Stage::MinAll { range },
                O::MaxAll => Stage::MaxAll { range },
            });
        };
        let selector = if native_map_value(&selector) {
            let selector_range =
                TextRange::new(selector.range.start as u32, selector.range.end as u32)
                    .expect("ordered selector range");
            zirql::dispose_expression(selector);
            super::report::OrderingSelector::MapValue(selector_range)
        } else {
            super::report::OrderingSelector::from_operand(selector)
        };
        if !selector.is_read_only() {
            return Err(zirql::QueryError::query(
                range.as_range(),
                "selector queries cannot edit or emit",
            ));
        }
        Ok(match ordering.order {
            O::Sort => Stage::SortBy { selector, range },
            O::Min => Stage::MinBy { selector, range },
            O::Max => Stage::MaxBy { selector, range },
            O::MinAll => Stage::MinAllBy { selector, range },
            O::MaxAll => Stage::MaxAllBy { selector, range },
        })
    }
    fn mapping(
        &mut self,
        mapping: zirql::NativeMap<super::report::Operand>,
        range: std::ops::Range<usize>,
    ) -> Result<Stage, zirql::QueryError> {
        let mapping = zirql::NativeMap {
            key: super::report::native_selector(mapping.key),
            value: super::report::native_selector(mapping.value),
        };
        if !super::report::map_is_read_only(&mapping) {
            return Err(zirql::QueryError::query(
                range,
                "map_by queries cannot edit or emit",
            ));
        }
        Ok(Stage::NativeMap {
            mapping,
            range: TextRange::new(range.start as u32, range.end as u32)
                .expect("ordered query range"),
        })
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
    fn parse_path_edge(
        &mut self,
        edge: zirql::Located<&str>,
        _: &mut zirql::Budget,
    ) -> Result<Stage, zirql::QueryError> {
        if edge.value != "users" {
            return Err(zirql::QueryError::query(
                edge.range,
                "unsupported path edge",
            ));
        }
        Ok(Stage::Users {
            index: None,
            range: TextRange::new(edge.range.start as u32, edge.range.end as u32)
                .expect("ordered edge range"),
        })
    }
    fn set_operator(&self, word: &str) -> Option<SetOperator> {
        SetOperator::from_keyword(word)
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

fn native_map_value(query: &super::report::Operand) -> bool {
    use zirql::{ExpressionKind as X, ExpressionStep as S};
    match &query.value {
        X::Group(query) => native_map_value(query),
        X::Sequence { input, steps } if matches!(input.value, X::Input) => {
            matches!(
                steps.as_slice(),
                [zirql::Located {
                    value: S::Stage(Stage::Value { .. }),
                    ..
                }]
            )
        }
        _ => false,
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

fn filter_stage(predicate: Predicate, range: TextRange) -> Stage {
    fn value(predicate: &Predicate) -> Option<zirql::ValueExpression<super::report::Operand>> {
        use zirql::ValueExpressionKind as K;
        let range = predicate.range().as_range();
        let kind = match predicate {
            Predicate::Expression { value, .. } => K::NativeValue(Box::new(zirql::Located::new(
                zirql::ExpressionKind::Sequence {
                    input: Box::new(zirql::Located::new(
                        zirql::ExpressionKind::Input,
                        range.clone(),
                    )),
                    steps: vec![
                        zirql::Located::new(
                            zirql::ExpressionStep::Stage(Stage::FunctionValue {
                                value: super::comprehension::ParsedComprehension::new(
                                    zirql::Located::new(K::PredicateSubject, range.clone()),
                                ),
                                range: predicate.range(),
                            }),
                            range.clone(),
                        ),
                        zirql::Located::new(
                            zirql::ExpressionStep::Stage(Stage::FunctionValue {
                                value: value.clone(),
                                range: predicate.range(),
                            }),
                            range.clone(),
                        ),
                    ],
                },
                range.clone(),
            ))),
            Predicate::Bool { value, .. } => K::Literal(zirql::ConstructedValue::Boolean(*value)),
            Predicate::Group { predicate, .. } => return value(predicate),
            Predicate::Not { predicate, .. } => K::Unary {
                operator: zirql::ValueUnaryOperator::Not,
                operand: Box::new(value(predicate)?),
            },
            Predicate::And { predicates, .. } | Predicate::Or { predicates, .. } => {
                let operator = if matches!(predicate, Predicate::And { .. }) {
                    zirql::ValueBinaryOperator::And
                } else {
                    zirql::ValueBinaryOperator::Or
                };
                let mut predicates = predicates.iter();
                let mut result = value(predicates.next()?)?;
                for predicate in predicates {
                    result = zirql::Located::new(
                        K::Binary {
                            operator,
                            left: Box::new(result),
                            right: Box::new(value(predicate)?),
                        },
                        range.clone(),
                    );
                }
                return Some(result);
            }
            _ => return None,
        };
        Some(zirql::Located::new(kind, range))
    }
    let constructed = value(&predicate).map(|body| {
        let range = range.as_range();
        super::comprehension::ParsedComprehension::new(zirql::Located::new(
            zirql::ValueExpressionKind::Filter {
                array: Box::new(zirql::Located::new(
                    zirql::ValueExpressionKind::NativeQuery(Box::new(zirql::Located::new(
                        zirql::ExpressionKind::Input,
                        range.clone(),
                    ))),
                    range.clone(),
                )),
                predicate: zirql::PredicateArgument {
                    argument: None,
                    body: Box::new(body),
                },
            },
            range,
        ))
    });
    Stage::Filter {
        predicate,
        constructed,
        range,
    }
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
            locals: Vec::new(),
            cursor: 0,
            diagnostics: Vec::new(),
            nesting_limit: DEFAULT_NESTING_LIMIT,
            bindings: HashSet::new(),
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
