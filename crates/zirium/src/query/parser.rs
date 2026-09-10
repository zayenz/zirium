use std::collections::HashSet;

use crate::source::TextRange;

use super::lexer::{Lexed, Token, TokenKind};

pub const DEFAULT_NESTING_LIMIT: usize = 64;

/// Set operators have equal precedence and associate left to right.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetOperator {
    Union,
    Intersect,
    Except,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub statements: Vec<Statement>,
    pub(crate) expression: Expression,
    pub(crate) emit_final_result: bool,
}

/// Statements run in order before the final expression. `Do` suppresses only
/// the expression's implicit result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Statement {
    Binding {
        name: String,
        expression: Expression,
    },
    Do(Expression),
    Query(Expression),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrintPart {
    Literal(String),
    Binding(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JsonLiteral {
    Object(Vec<(Vec<PrintPart>, JsonLiteral)>),
    Array(Vec<JsonLiteral>),
    String(Vec<PrintPart>),
    Scalar(serde_json::Value),
    Binding(String),
}

impl Program {
    pub fn expression(&self) -> &Expression {
        &self.expression
    }
    pub fn range(&self) -> TextRange {
        self.expression.range
    }
}

/// Each set operand is a pipeline. Grouping introduces a nested expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expression {
    pub first: Vec<Stage>,
    pub rest: Vec<(SetOperator, Vec<Stage>)>,
    pub range: TextRange,
}

impl Expression {
    pub(crate) fn is_single_closure(&self) -> bool {
        self.rest.is_empty()
            && match self.first.as_slice() {
                [Stage::Closure { .. }] => true,
                [Stage::Group { expression, .. }] => expression.is_single_closure(),
                _ => false,
            }
    }

    pub fn is_selection_only(&self) -> bool {
        self.first
            .iter()
            .chain(self.rest.iter().flat_map(|(_, stages)| stages))
            .all(Stage::is_selection_only)
    }
    fn is_read_only(&self) -> bool {
        self.first
            .iter()
            .chain(self.rest.iter().flat_map(|(_, stages)| stages))
            .all(|stage| match stage {
                Stage::SetAttr { .. }
                | Stage::RemoveAttr { .. }
                | Stage::Emit { .. }
                | Stage::Json { .. }
                | Stage::Markdown { .. }
                | Stage::Print { .. } => false,
                Stage::Group { expression, .. } | Stage::Fixpoint { expression, .. } => {
                    expression.is_read_only()
                }
                Stage::SortBy { selector, .. }
                | Stage::MinBy { selector, .. }
                | Stage::MaxBy { selector, .. } => selector.is_read_only(),
                Stage::MapBy { key, value, .. } => key.is_read_only() && value.is_read_only(),
                _ => true,
            })
    }

    pub(crate) fn ends_with_emission(&self) -> bool {
        self.rest.is_empty()
            && self.first.last().is_some_and(|stage| match stage {
                Stage::Emit { .. }
                | Stage::Json { .. }
                | Stage::Markdown { .. }
                | Stage::Print { .. } => true,
                Stage::Group { expression, .. } => expression.ends_with_emission(),
                _ => false,
            })
    }
    fn is_terminal(&self) -> bool {
        self.first.last().is_some_and(Stage::is_terminal)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Bool {
        value: bool,
        range: TextRange,
    },
    Op {
        name: String,
        range: TextRange,
    },
    Dialect {
        name: String,
        range: TextRange,
    },
    ResultType {
        spelling: String,
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
            Self::Bool { range, .. }
            | Self::Op { range, .. }
            | Self::Dialect { range, .. }
            | Self::ResultType { range, .. }
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
    Literal {
        value: JsonLiteral,
        range: TextRange,
    },
    Markdown {
        range: TextRange,
    },
    Print {
        parts: Vec<PrintPart>,
        range: TextRange,
    },
    Binding {
        name: String,
        range: TextRange,
    },
    Tally {
        range: TextRange,
    },
    MapBy {
        key: Box<Expression>,
        value: Box<Expression>,
        range: TextRange,
    },
    Reachable {
        range: TextRange,
    },
    Input {
        range: TextRange,
    },
    Filter {
        predicate: Predicate,
        range: TextRange,
    },
    Closure {
        range: TextRange,
    },
    Slice {
        range: TextRange,
    },
    Defs {
        index: Option<usize>,
        range: TextRange,
    },
    Users {
        index: Option<usize>,
        range: TextRange,
    },
    Parent {
        range: TextRange,
    },
    Children {
        range: TextRange,
    },
    Root {
        predicate: Predicate,
        range: TextRange,
    },
    Subtree {
        range: TextRange,
    },
    Unique {
        range: TextRange,
    },
    Sort {
        range: TextRange,
    },
    SortBy {
        selector: Box<Expression>,
        range: TextRange,
    },
    Reverse {
        range: TextRange,
    },
    Head {
        count: usize,
        range: TextRange,
    },
    Tail {
        count: usize,
        range: TextRange,
    },
    Min {
        range: TextRange,
    },
    MinBy {
        selector: Box<Expression>,
        range: TextRange,
    },
    Max {
        range: TextRange,
    },
    MaxBy {
        selector: Box<Expression>,
        range: TextRange,
    },
    Attr {
        name: String,
        range: TextRange,
    },
    Names {
        range: TextRange,
    },
    ResultTypes {
        range: TextRange,
    },
    OperandTypes {
        range: TextRange,
    },
    Group {
        expression: Box<Expression>,
        range: TextRange,
    },
    Fixpoint {
        expression: Box<Expression>,
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
    Emit {
        range: TextRange,
    },
    Json {
        range: TextRange,
    },
}

impl Stage {
    pub fn range(&self) -> TextRange {
        match self {
            Self::Literal { range, .. }
            | Self::Markdown { range }
            | Self::Print { range, .. }
            | Self::Binding { range, .. }
            | Self::Tally { range }
            | Self::MapBy { range, .. }
            | Self::Reachable { range }
            | Self::Input { range }
            | Self::Filter { range, .. }
            | Self::Closure { range }
            | Self::Slice { range }
            | Self::Defs { range, .. }
            | Self::Users { range, .. }
            | Self::Parent { range }
            | Self::Children { range }
            | Self::Root { range, .. }
            | Self::Subtree { range }
            | Self::Unique { range }
            | Self::Sort { range }
            | Self::SortBy { range, .. }
            | Self::Reverse { range }
            | Self::Head { range, .. }
            | Self::Tail { range, .. }
            | Self::Min { range }
            | Self::MinBy { range, .. }
            | Self::Max { range }
            | Self::MaxBy { range, .. }
            | Self::Attr { range, .. }
            | Self::Names { range }
            | Self::ResultTypes { range }
            | Self::OperandTypes { range }
            | Self::Group { range, .. }
            | Self::Fixpoint { range, .. }
            | Self::SetAttr { range, .. }
            | Self::RemoveAttr { range, .. }
            | Self::Count { range }
            | Self::Emit { range }
            | Self::Json { range } => *range,
        }
    }
    fn is_selection_only(&self) -> bool {
        match self {
            Self::SetAttr { .. }
            | Self::RemoveAttr { .. }
            | Self::Count { .. }
            | Self::Tally { .. }
            | Self::MapBy { .. }
            | Self::Literal { .. } => false,
            Self::Group { expression, .. } | Self::Fixpoint { expression, .. } => {
                expression.is_selection_only()
            }
            _ => true,
        }
    }
    fn is_terminal(&self) -> bool {
        match self {
            Self::Count { .. } => true,
            Self::Group { expression, .. } => expression.is_terminal(),
            _ => false,
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
        bindings: HashSet::new(),
    };
    let program = parser.program();
    Parsed {
        program: parser.diagnostics.is_empty().then_some(program).flatten(),
        diagnostics: parser.diagnostics,
    }
}

struct Parser<'a> {
    source: &'a str,
    tokens: &'a [Token],
    cursor: usize,
    diagnostics: Vec<Diagnostic>,
    nesting_limit: usize,
    bindings: HashSet<String>,
}

impl Parser<'_> {
    fn program(&mut self) -> Option<Program> {
        self.skip_trivia();
        if self.at(TokenKind::Eof) {
            return Some(Program {
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
                        statements,
                        expression,
                        emit_final_result: false,
                    });
                }
                statements.push(Statement::Do(expression));
                continue;
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
                statements,
                expression,
                emit_final_result: true,
            });
        }
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
        if !expression.rest.is_empty() && !expression.is_selection_only() {
            self.diagnostics.push(Diagnostic {
                message: "set operands must be selection queries; put edits and count after the grouped set expression",
                range: expression.range,
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
            "reachable" => Stage::Reachable { range },
            "tally" => Stage::Tally { range },
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
            "slice" => Stage::Slice { range },
            "defs" | "users" => {
                self.skip_trivia();
                let index = if self.at(TokenKind::LParen) {
                    self.bump();
                    self.skip_trivia();
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
            "sort_by" | "min_by" | "max_by" => {
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
                    "max_by" => Stage::MaxBy { selector, range },
                    _ => unreachable!(),
                }
            }
            "reverse" => Stage::Reverse { range },
            "head" | "tail" => {
                self.expect(TokenKind::LParen, "expected `(` after head or tail")?;
                self.skip_trivia();
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
            "max" => Stage::Max { range },
            "count" => Stage::Count { range },
            "emit" => Stage::Emit { range },
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
                if !expression.is_selection_only() {
                    self.diagnostics.push(Diagnostic {
                        message: "fixpoint requires a selection query without edits or count",
                        range: expression.range,
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
                self.error_at_previous("unknown query stage; expected input, filter, navigation, projection, fixpoint, an edit, count, emit, or json");
                return None;
            }
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
            "dialect" | "result_type" => {
                if depth >= self.nesting_limit {
                    self.error("query nesting limit exceeded");
                    return None;
                }
                self.expect(TokenKind::LParen, "expected `(` after predicate")?;
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
            TokenKind::Integer | TokenKind::Number => {
                let Ok(value) = serde_json::from_str::<serde_json::Number>(self.current_text())
                else {
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

fn is_reserved(name: &str) -> bool {
    matches!(
        name,
        "input"
            | "do"
            | "filter"
            | "closure"
            | "slice"
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
            | "max"
            | "max_by"
            | "attr"
            | "names"
            | "result_types"
            | "operand_types"
            | "fixpoint"
            | "set_attr"
            | "remove_attr"
            | "count"
            | "emit"
            | "json"
            | "markdown"
            | "print"
            | "tally"
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
