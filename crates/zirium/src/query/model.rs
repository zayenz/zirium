//! Shared query representation used by the text parser and library builders.

use crate::{formatter::FormatOptions, source::TextRange};

/// Set operators have equal precedence and associate left to right.
pub use zirql::NativeSetOperator as SetOperator;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub functions: Vec<super::FunctionDefinition>,
    pub parameters: Vec<zirql::ParameterDeclaration>,
    pub(crate) parameter_positions: Vec<usize>,
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
    PatternBinding {
        names: Vec<zirql::Located<String>>,
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
#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    pub first: Vec<Step>,
    pub rest: Vec<(SetOperator, Vec<Step>)>,
    pub range: TextRange,
}

impl Drop for Expression {
    fn drop(&mut self) {
        for steps in std::iter::once(std::mem::take(&mut self.first)).chain(
            std::mem::take(&mut self.rest)
                .into_iter()
                .map(|(_, steps)| steps),
        ) {
            zirql::dispose_expression(zirql::Located::new(
                zirql::ExpressionKind::Sequence {
                    input: Box::new(zirql::Located::new(zirql::ExpressionKind::Input, 0..0)),
                    steps,
                },
                0..0,
            ));
        }
    }
}

impl Eq for Expression {}

pub type NativeStage = std::sync::Arc<Stage>;
pub type Step =
    zirql::Located<zirql::ExpressionStep<NativeStage, super::program::Emitter, SetOperator>>;

pub(super) fn native(stage: Stage) -> Step {
    let range = super::program::range(stage.range());
    zirql::Located::new(
        zirql::ExpressionStep::Stage(std::sync::Arc::new(stage)),
        range,
    )
}
impl From<Stage> for Step {
    fn from(stage: Stage) -> Self {
        native(stage)
    }
}

pub(super) fn value(value: super::comprehension::ParsedComprehension) -> Step {
    let range = value.expression.range.clone();
    zirql::Located::new(
        zirql::ExpressionStep::Value(std::sync::Arc::clone(&value.expression)),
        range,
    )
}
pub(super) fn step_range(step: &Step) -> TextRange {
    TextRange::new(step.range.start as u32, step.range.end as u32).expect("ordered query range")
}
pub(super) fn step_is_terminal(step: &Step) -> bool {
    matches!(
        &step.value,
        zirql::ExpressionStep::Collection(zirql::Stage::Count)
    ) || matches!(&step.value, zirql::ExpressionStep::Stage(stage) if stage.is_terminal())
}
impl Expression {
    pub(crate) fn is_single_closure(&self) -> bool {
        self.rest.is_empty()
            && match self.first.as_slice() {
                [
                    zirql::Located {
                        value: zirql::ExpressionStep::Stage(stage),
                        ..
                    },
                ] => match stage.as_ref() {
                    Stage::Closure { .. } => true,
                    Stage::Group { expression, .. } => expression.is_single_closure(),
                    _ => false,
                },
                _ => false,
            }
    }
    pub(crate) fn is_map_value(&self) -> bool {
        self.rest.is_empty()
            && match self.first.as_slice() {
                [
                    zirql::Located {
                        value: zirql::ExpressionStep::Stage(stage),
                        ..
                    },
                ] => match stage.as_ref() {
                    Stage::Value { .. } => true,
                    Stage::Group { expression, .. } => expression.is_map_value(),
                    _ => false,
                },
                _ => false,
            }
    }
    pub fn is_selection_only(&self) -> bool {
        self.first_non_selection_stage().is_none()
    }
    pub(crate) fn first_non_selection_stage(&self) -> Option<TextRange> {
        self.first
            .iter()
            .chain(self.rest.iter().flat_map(|(_, steps)| steps))
            .find_map(|step| match &step.value {
                zirql::ExpressionStep::Stage(stage) => stage.first_non_selection_stage(),
                zirql::ExpressionStep::Filter(_)
                | zirql::ExpressionStep::Ordering(_)
                | zirql::ExpressionStep::WhereExists(_)
                | zirql::ExpressionStep::FixedPoint(_)
                | zirql::ExpressionStep::Collection(
                    zirql::Stage::Unique
                    | zirql::Stage::Head(_)
                    | zirql::Stage::Tail(_)
                    | zirql::Stage::Reverse
                    | zirql::Stage::Check { .. },
                )
                | zirql::ExpressionStep::Head(_)
                | zirql::ExpressionStep::Tail(_)
                | zirql::ExpressionStep::Check { .. } => None,
                _ => Some(step_range(step)),
            })
    }
    pub(crate) fn is_read_only(&self) -> bool {
        self.first
            .iter()
            .chain(self.rest.iter().flat_map(|(_, steps)| steps))
            .all(|step| match &step.value {
                zirql::ExpressionStep::Stage(stage) => stage.is_read_only(),
                zirql::ExpressionStep::Emit { .. } => false,
                _ => true,
            })
    }
    pub(crate) fn ends_with_emission(&self) -> bool {
        self.rest.is_empty()
            && self.first.last().is_some_and(|step| match &step.value {
                zirql::ExpressionStep::Emit { .. } => true,
                zirql::ExpressionStep::Stage(stage) => match stage.as_ref() {
                    Stage::Emit { .. }
                    | Stage::Format { .. }
                    | Stage::Json { .. }
                    | Stage::Markdown { .. }
                    | Stage::Print { .. } => true,
                    Stage::Group { expression, .. } => expression.ends_with_emission(),
                    _ => false,
                },
                _ => false,
            })
    }
    pub(crate) fn is_terminal(&self) -> bool {
        self.first.last().is_some_and(step_is_terminal)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Expression {
        value: super::comprehension::ParsedComprehension,
        range: TextRange,
    },
    SameValue {
        left: super::capture::ParsedOperand,
        right: super::capture::ParsedOperand,
        range: TextRange,
    },
    Bool {
        value: bool,
        range: TextRange,
    },
    Computed {
        value: ComputedPredicate,
        range: TextRange,
    },
    OpValue {
        name: super::comprehension::ParsedComprehension,
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
            Self::Expression { range, .. }
            | Self::SameValue { range, .. }
            | Self::Bool { range, .. }
            | Self::Op { range, .. }
            | Self::OpValue { range, .. }
            | Self::Computed { range, .. }
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
pub enum ComputedPredicate {
    Dialect(super::comprehension::ParsedComprehension),
    ResultType(super::comprehension::ParsedComprehension),
    HasAttr(super::comprehension::ParsedComprehension),
    Attr {
        name: super::comprehension::ParsedComprehension,
        value: super::comprehension::ParsedComprehension,
    },
}
impl ComputedPredicate {
    pub(super) fn operands(
        &self,
    ) -> impl Iterator<Item = &super::comprehension::ParsedComprehension> {
        let values = match self {
            Self::Dialect(value) | Self::ResultType(value) | Self::HasAttr(value) => {
                [Some(value), None]
            }
            Self::Attr { name, value } => [Some(name), Some(value)],
        };
        values.into_iter().flatten()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputedStage {
    Defs(super::comprehension::ParsedComprehension),
    Users(super::comprehension::ParsedComprehension),
    Attr(super::comprehension::ParsedComprehension),
}
impl ComputedStage {
    pub(super) fn operand(&self) -> &super::comprehension::ParsedComprehension {
        match self {
            Self::Defs(value) | Self::Users(value) | Self::Attr(value) => value,
        }
    }
    pub(super) fn parameter_type(&self) -> zirql::ParameterType {
        match self {
            Self::Attr(_) => zirql::ParameterType::String,
            Self::Defs(_) | Self::Users(_) => zirql::ParameterType::Int,
        }
    }
    pub(super) fn placeholder(&self, range: TextRange) -> Stage {
        match self {
            Self::Defs(_) => Stage::Defs {
                index: Some(0),
                range,
            },
            Self::Users(_) => Stage::Users {
                index: Some(0),
                range,
            },
            Self::Attr(_) => Stage::Attr {
                name: String::new(),
                range,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Guard {
        predicate: Predicate,
        range: TextRange,
    },
    /// Strict, dialect-aware closure over the selected operations.
    ClosureFixedPoint {
        range: TextRange,
    },
    Parameter {
        index: usize,
        ty: zirql::ParameterType,
        range: TextRange,
    },
    Native {
        value: super::capture::ParsedOperand,
        range: TextRange,
    },
    Operand {
        operation: super::capture::ParsedOperand,
        index: usize,
        range: TextRange,
    },
    FunctionInput {
        value: std::sync::Arc<super::QueryOutput>,
        range: TextRange,
    },
    PatternBinding {
        value: super::functions::ParsedBinding,
        range: TextRange,
    },
    Computed {
        value: ComputedStage,
        range: TextRange,
    },
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
    Value {
        range: TextRange,
    },
    Reachable {
        range: TextRange,
    },
    Input {
        range: TextRange,
    },
    Closure {
        range: TextRange,
    },
    Slice {
        range: TextRange,
    },
    ForwardSlice {
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
    SetAttr {
        name: String,
        value: String,
        range: TextRange,
    },
    RemoveAttr {
        name: String,
        range: TextRange,
    },
    Emit {
        range: TextRange,
    },
    Format {
        options: FormatOptions,
        range: TextRange,
    },
    Json {
        range: TextRange,
    },
}

impl Stage {
    pub(super) fn is_read_only(&self) -> bool {
        match self {
            Stage::SetAttr { .. }
            | Stage::RemoveAttr { .. }
            | Stage::Emit { .. }
            | Stage::Format { .. }
            | Stage::Json { .. }
            | Stage::Markdown { .. }
            | Stage::Print { .. } => false,
            Stage::Group { expression, .. } => expression.is_read_only(),
            _ => true,
        }
    }

    pub fn range(&self) -> TextRange {
        match self {
            Self::Guard { range, .. }
            | Self::ClosureFixedPoint { range }
            | Self::Parameter { range, .. }
            | Self::Native { range, .. }
            | Self::Operand { range, .. }
            | Self::FunctionInput { range, .. }
            | Self::PatternBinding { range, .. }
            | Self::Computed { range, .. }
            | Self::Structured { range, .. }
            | Self::Literal { range, .. }
            | Self::Markdown { range }
            | Self::Print { range, .. }
            | Self::Binding { range, .. }
            | Self::Value { range }
            | Self::Reachable { range }
            | Self::Input { range }
            | Self::Closure { range }
            | Self::Slice { range }
            | Self::ForwardSlice { range }
            | Self::Defs { range, .. }
            | Self::Users { range, .. }
            | Self::Parent { range }
            | Self::Children { range }
            | Self::Root { range, .. }
            | Self::Subtree { range }
            | Self::Attr { range, .. }
            | Self::Names { range }
            | Self::ResultTypes { range }
            | Self::OperandTypes { range }
            | Self::Group { range, .. }
            | Self::SetAttr { range, .. }
            | Self::RemoveAttr { range, .. }
            | Self::Emit { range }
            | Self::Format { range, .. }
            | Self::Json { range } => *range,
        }
    }
    fn is_selection_only(&self) -> bool {
        match self {
            Self::Guard { .. }
            | Self::Native { .. }
            | Self::Operand { .. }
            | Self::SetAttr { .. }
            | Self::RemoveAttr { .. }
            | Self::Value { .. }
            | Self::Literal { .. } => false,
            Self::Group { expression, .. } => expression.is_selection_only(),
            _ => true,
        }
    }
    fn first_non_selection_stage(&self) -> Option<TextRange> {
        match self {
            Self::Group { expression, .. } => expression.first_non_selection_stage(),
            _ if !self.is_selection_only() => Some(self.range()),
            _ => None,
        }
    }
    pub(crate) fn is_terminal(&self) -> bool {
        match self {
            Self::Group { expression, .. } => expression.is_terminal(),
            _ => false,
        }
    }
}

/// Structured operations share the evaluator but need no textual syntax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuredStage {
    One,
    StringAttr(String),
    Attributes(String),
    Types { operands: bool },
    Spellings,
}

pub(super) fn shared_predicate(predicate: Predicate) -> zirql::Predicate<NativeStage> {
    use zirql::PredicateKind as P;
    let range = predicate.range().as_range();
    let value = match predicate {
        Predicate::Not { predicate, .. } => P::Not(Box::new(shared_predicate(*predicate))),
        Predicate::Group { predicate, .. } => P::Group(Box::new(shared_predicate(*predicate))),
        Predicate::And { predicates, .. } => {
            P::And(predicates.into_iter().map(shared_predicate).collect())
        }
        Predicate::Or { predicates, .. } => {
            P::Or(predicates.into_iter().map(shared_predicate).collect())
        }
        predicate => P::Leaf(std::sync::Arc::new(Stage::Guard {
            range: predicate.range(),
            predicate,
        })),
    };
    zirql::Located::new(value, range)
}
