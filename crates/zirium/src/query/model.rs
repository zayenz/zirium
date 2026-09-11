//! Shared query representation used by the text parser and library builders.

use crate::source::TextRange;

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
    pub(crate) fn is_read_only(&self) -> bool {
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
                | Stage::MinAllBy { selector, .. }
                | Stage::MaxBy { selector, .. }
                | Stage::MaxAllBy { selector, .. } => selector.is_read_only(),
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
    pub(crate) fn is_terminal(&self) -> bool {
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
    /// Library-only operations with native results.
    Structured {
        step: StructuredStage,
        range: TextRange,
    },
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
    MinAll {
        range: TextRange,
    },
    MinAllBy {
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
    MaxAll {
        range: TextRange,
    },
    MaxAllBy {
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
            Self::Structured { range, .. }
            | Self::Literal { range, .. }
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
            | Self::MinAll { range }
            | Self::MinAllBy { range, .. }
            | Self::Max { range }
            | Self::MaxBy { range, .. }
            | Self::MaxAll { range }
            | Self::MaxAllBy { range, .. }
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
    pub(crate) fn is_terminal(&self) -> bool {
        match self {
            Self::Count { .. } => true,
            Self::Group { expression, .. } => expression.is_terminal(),
            _ => false,
        }
    }
}

/// Structured operations share the evaluator but need no textual syntax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuredStage {
    One,
    MapBy {
        key: Box<Expression>,
        value: Box<Expression>,
    },
    WhereExists(Box<Expression>),
    StringAttr(String),
    Attributes(String),
    Types {
        operands: bool,
    },
    Spellings,
}
