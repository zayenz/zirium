//! Shared diff execution with native comparison handles and host-owned operations.

use super::{
    EvaluationError, EvaluationLimits, EvaluationOptions, OperationTraversal, core_error,
    evaluation_budget, program,
};
use crate::{
    diff::{ChangeField, ChangeId, ChangeKind, Diff, DiffOperation, DiffSide},
    semantic::{AttributeValue, OperationId},
};
use std::{
    collections::{BTreeMap, HashSet},
    ops::Range,
};
use zirql::{
    Bindings, Budget, ExpressionKind, ExpressionStep, Host, Located, NativeInput, ProgramHost,
    Stage, Truth,
};

/// Native results from a diff text program. Selections retain comparison and
/// side ownership until an explicit JSON or MLIR rendering boundary. JSON,
/// Markdown and print stages produce report values rather than emission events.
#[derive(Clone, Debug, PartialEq)]
pub enum DiffProgramOutput {
    Changes(Vec<ChangeId>),
    Operations(DiffSide, Vec<DiffOperation>),
    Strings(Vec<String>),
    Count(usize),
    Map(BTreeMap<String, DiffProgramOutput>),
    Json(serde_json::Value, Option<DiffSide>),
    Text(String),
}

/// A parsed diff program. The diff frontend retains its existing syntax and
/// statement-end output policy while Zirql executes statements and expressions.
#[derive(Debug)]
pub struct DiffProgram {
    pub(super) program: zirql::Program<DiffStage, SetOperator, ()>,
}

impl Drop for DiffProgram {
    fn drop(&mut self) {
        let program = std::mem::replace(&mut self.program, zirql::Program::new(Vec::new(), 0..0));
        dispose_statements(program.into_statements());
    }
}

impl DiffProgram {
    /// Check the diff edit policy before loading either input, including a silent
    /// CLI request whose query is otherwise unevaluated.
    ///
    /// # Errors
    /// Refuses edit stage names outside quoted strings.
    pub fn check_read_only(source: &str) -> Result<(), EvaluationError> {
        super::diff_text::reject_diff_mutations(source).map_err(EvaluationError::new)
    }

    /// Parse and bound a read-only diff program without accessing documents.
    ///
    /// # Errors
    /// Refuses unsupported syntax, edits and excessive expression nesting.
    pub fn parse(source: &str) -> Result<Self, EvaluationError> {
        super::diff_text::parse(source)
    }

    /// Evaluate with legacy per-stream and work limits.
    ///
    /// # Errors
    /// Returns located language, ownership, traversal or callback failures.
    pub fn evaluate(
        &self,
        diff: &Diff<'_>,
        document_names: (&str, &str),
        limits: EvaluationLimits,
        options: EvaluationOptions,
        emit: impl FnMut(&Diff<'_>, DiffProgramOutput, &mut Budget) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_budget(
            diff,
            document_names,
            options,
            &mut evaluation_budget(limits),
            emit,
        )
    }

    /// Run with one caller-owned allowance through nested bodies and callbacks.
    /// Native operation/change values retain comparison ownership, including
    /// the side of an empty operation selection. Earlier callbacks remain
    /// observable after a later failure; callers choose output buffering.
    ///
    /// # Errors
    /// Returns located language, resource, ownership or callback failures.
    pub fn evaluate_with_budget(
        &self,
        diff: &Diff<'_>,
        document_names: (&str, &str),
        options: EvaluationOptions,
        budget: &mut Budget,
        mut emit: impl FnMut(&Diff<'_>, DiffProgramOutput, &mut Budget) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        let names = BTreeMap::from([
            ("before_document".to_owned(), Kind::Strings),
            ("after_document".to_owned(), Kind::Strings),
        ]);
        validate_program(&self.program, names)?;
        let input = initial_changes(diff, budget, &self.program.range())?;
        let mut initial = Vec::new();
        for (name, value) in [
            ("before_document", document_names.0),
            ("after_document", document_names.1),
        ] {
            admit(budget, 1, &(0..0))?;
            program::charge_bytes(budget, value.len(), &(0..0))?;
            initial.push((
                Located::new(name.to_owned(), 0..0),
                DiffProgramOutput::Strings(vec![value.to_owned()]),
            ));
        }
        let bindings = Bindings::new(initial).map_err(core_error)?;
        let mut context = Context {
            diff,
            options,
            emit: &mut emit,
        };
        zirql::evaluate_program(
            &self.program,
            &DiffHost,
            input,
            &mut context,
            bindings,
            budget,
        )
        .map_err(program::evaluation_error)?;
        Ok(())
    }
}

pub(super) type Expression = zirql::Expression<DiffStage, SetOperator, ()>;

#[derive(Debug)]
pub(super) struct OwnedExpression(Option<Expression>);

impl OwnedExpression {
    pub(super) fn new(expression: Expression) -> Self {
        Self(Some(expression))
    }
    pub(super) fn into_inner(mut self) -> Expression {
        self.0
            .take()
            .expect("the owned expression has not been transferred")
    }
}
impl std::ops::Deref for OwnedExpression {
    type Target = Expression;
    fn deref(&self) -> &Expression {
        self.0
            .as_ref()
            .expect("a retained expression has not been transferred")
    }
}
impl Drop for OwnedExpression {
    fn drop(&mut self) {
        if let Some(expression) = self.0.take() {
            dispose_expression(expression);
        }
    }
}

pub(super) fn dispose_statements(
    statements: Vec<Located<zirql::Statement<DiffStage, SetOperator, ()>>>,
) {
    for statement in statements {
        let expression = match statement.value {
            zirql::Statement::Bind { expression, .. }
            | zirql::Statement::Evaluate { expression, .. } => expression,
        };
        dispose_expression(expression);
    }
}

fn dispose_expression(expression: Expression) {
    // A flat set spine is deliberately unbounded by logical nesting. Release
    // its boxes through a worklist instead of recursive Rust drop glue.
    let mut pending = vec![expression];
    while let Some(expression) = pending.pop() {
        match expression.value {
            ExpressionKind::Input | ExpressionKind::Binding(_) => {}
            ExpressionKind::Group(expression) => pending.push(*expression),
            ExpressionKind::Set { left, right, .. } => {
                pending.push(*left);
                pending.push(*right);
            }
            ExpressionKind::Sequence { input, steps } => {
                pending.push(*input);
                // Nested host payloads have their own owned-expression guard.
                drop(steps);
            }
            ExpressionKind::Record(fields) => {
                pending.extend(fields.into_iter().map(|(_, expression)| expression))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum SetOperator {
    Union,
    Intersect,
    Except,
}

#[derive(Clone, Debug)]
pub(super) enum Leaf {
    Bool(bool),
    Change(ChangeKind),
    Changed(ChangeField),
    Op(String),
    Dialect(String),
    ResultType(String),
    HasAttr(String),
    Attr(String, String),
}

#[derive(Debug)]
pub(super) enum DiffStage {
    Stream(zirql::Pipeline<Leaf>),
    SelectionUnique(zirql::Pipeline<Leaf>),
    Nested(Box<OwnedExpression>),
    Input,
    Project(DiffSide),
    Users(Option<usize>),
    Defs(Option<usize>),
    Parent,
    Children,
    Root(zirql::Pipeline<Leaf>),
    Subtree,
    Graph(OperationTraversal),
    Fixpoint(Box<OwnedExpression>),
    Reverse,
    Tail(usize),
    Names,
    Attr(String),
    ResultTypes,
    OperandTypes,
    Sort,
    Extreme {
        maximum: bool,
        all: bool,
    },
    Order {
        kind: Order,
        selector: Box<OwnedExpression>,
    },
    MapBy {
        key: Box<OwnedExpression>,
        value: Box<OwnedExpression>,
    },
    Check(Option<usize>),
    Json,
    Markdown,
    Print(String),
    Noop,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Order {
    Sort,
    Min,
    Max,
    MinAll,
    MaxAll,
}

pub(super) fn sequence(stages: Vec<Located<DiffStage>>, range: Range<usize>) -> Expression {
    Located::new(
        ExpressionKind::Sequence {
            input: Box::new(Located::new(ExpressionKind::Input, range.clone())),
            steps: stages
                .into_iter()
                .map(|stage| Located::new(ExpressionStep::Stage(stage.value), stage.range))
                .collect(),
        },
        range,
    )
}

pub(super) fn nested(expression: Expression) -> Box<OwnedExpression> {
    let range = expression.range.clone();
    Box::new(OwnedExpression::new(Located::new(
        ExpressionKind::Group(Box::new(expression)),
        range,
    )))
}

pub(super) fn stream(
    stage: Stage<Leaf>,
    range: Range<usize>,
) -> Result<DiffStage, EvaluationError> {
    Ok(DiffStage::Stream(
        zirql::Pipeline::with_group_not_nesting(
            vec![Located::new(stage, range.clone())],
            range,
            zirql::MAX_NESTING,
        )
        .map_err(core_error)?,
    ))
}

pub(super) fn initial_changes(
    diff: &Diff<'_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let mut changes = Vec::new();
    for change in diff.change_ids() {
        charge(budget, 1, range)?;
        push(&mut changes, change, budget, range)?;
    }
    Ok(DiffProgramOutput::Changes(changes))
}

pub(super) fn evaluate_typed(
    expression: &Expression,
    diff: &Diff<'_>,
    options: EvaluationOptions,
    budget: &mut Budget,
) -> Result<DiffProgramOutput, EvaluationError> {
    validate_expression(expression, Kind::Changes, &BTreeMap::new())?;
    let input = initial_changes(diff, budget, &expression.range)?;
    let mut discard = |_: &Diff<'_>, _: DiffProgramOutput, _: &mut Budget| Ok(());
    let mut context = Context {
        diff,
        options,
        emit: &mut discard,
    };
    evaluate_expression(
        expression,
        input,
        &mut context,
        &Bindings::default(),
        budget,
    )
}

struct Context<'a, 'document> {
    diff: &'a Diff<'document>,
    options: EvaluationOptions,
    emit:
        &'a mut dyn FnMut(&Diff<'_>, DiffProgramOutput, &mut Budget) -> Result<(), EvaluationError>,
}
struct DiffHost;

fn charge(budget: &mut Budget, units: usize, range: &Range<usize>) -> Result<(), EvaluationError> {
    budget
        .charge(units, zirql::Location::Query(range.clone()))
        .map_err(core_error)
}
fn admit(budget: &mut Budget, units: usize, range: &Range<usize>) -> Result<(), EvaluationError> {
    budget
        .charge_results(units, zirql::Location::Query(range.clone()))
        .map_err(core_error)
}
fn check(budget: &Budget, units: usize, range: &Range<usize>) -> Result<(), EvaluationError> {
    budget
        .check_items(units, zirql::Location::Query(range.clone()))
        .map_err(core_error)
}
fn push<T>(
    values: &mut Vec<T>,
    value: T,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    check(budget, values.len().saturating_add(1), range)?;
    admit(budget, 1, range)?;
    values.push(value);
    Ok(())
}
fn failure(message: impl Into<String>, range: &Range<usize>) -> EvaluationError {
    EvaluationError::new(message).at(range.clone())
}
fn diff_error(error: crate::diff::DiffError) -> EvaluationError {
    EvaluationError::new(error.to_string())
}

impl DiffProgramOutput {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Changes(v) => v.len(),
            Self::Operations(_, v) => v.len(),
            Self::Strings(v) => v.len(),
            Self::Map(v) => v.len(),
            Self::Count(_) | Self::Json(_, _) | Self::Text(_) => 1,
        }
    }
}

fn copy(
    value: &DiffProgramOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    check(budget, value.len(), range)?;
    admit(budget, value.len().max(1), range)?;
    match value {
        DiffProgramOutput::Map(entries) => {
            let mut result = BTreeMap::new();
            for (key, value) in entries {
                program::charge_bytes(budget, key.len(), range)?;
                result.insert(key.clone(), copy(value, budget, range)?);
            }
            Ok(DiffProgramOutput::Map(result))
        }
        DiffProgramOutput::Strings(values) => {
            program::charge_bytes(budget, values.iter().map(String::len).sum(), range)?;
            Ok(DiffProgramOutput::Strings(values.clone()))
        }
        DiffProgramOutput::Text(text) => {
            program::charge_bytes(budget, text.len(), range)?;
            Ok(DiffProgramOutput::Text(text.clone()))
        }
        DiffProgramOutput::Json(value, side) => {
            admit(budget, super::json_size(value), range)?;
            program::charge_bytes(budget, json_bytes(value), range)?;
            Ok(DiffProgramOutput::Json(value.clone(), *side))
        }
        DiffProgramOutput::Changes(_)
        | DiffProgramOutput::Operations(_, _)
        | DiffProgramOutput::Count(_) => Ok(value.clone()),
    }
}

impl ProgramHost<Context<'_, '_>> for DiffHost {
    type Value = DiffProgramOutput;
    type Stage = DiffStage;
    type SetOperator = SetOperator;
    type Emitter = ();
    type Error = EvaluationError;

    fn copy_value(
        &self,
        value: &DiffProgramOutput,
        _: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        copy(value, budget, range).map_err(zirql::EvaluationError::Host)
    }
    fn evaluate_stage(
        &self,
        stage: &DiffStage,
        value: DiffProgramOutput,
        context: &mut Context<'_, '_>,
        bindings: &Bindings<DiffProgramOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        let result =
            if let DiffStage::Stream(pipeline) | DiffStage::SelectionUnique(pipeline) = stage {
                if matches!(stage, DiffStage::SelectionUnique(_))
                    && !matches!(
                        value,
                        DiffProgramOutput::Changes(_) | DiffProgramOutput::Operations(_, _)
                    )
                {
                    Err(failure(
                        "unique requires a change or operation stream",
                        range,
                    ))
                } else {
                    evaluate_stream(pipeline, value, context.diff, budget)
                }
            } else {
                charge(budget, value.len().max(1), range)
                    .and_then(|()| evaluate_stage(stage, value, context, bindings, budget, range))
            };
        result
            .and_then(|value| {
                check(budget, value.len(), range)?;
                Ok(value)
            })
            .map_err(|mut error| {
                if error.range.is_none() || error.range == Some(0..0) {
                    error.range = Some(range.clone());
                }
                zirql::EvaluationError::Host(error)
            })
    }
    fn combine(
        &self,
        operator: &SetOperator,
        left: DiffProgramOutput,
        right: DiffProgramOutput,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        combine(context.diff, left, right, *operator, budget, range)
            .map_err(zirql::EvaluationError::Host)
    }
    fn construct_record(
        &self,
        _: Vec<(String, DiffProgramOutput)>,
        _: &mut Context<'_, '_>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        Err(zirql::EvaluationError::Host(failure(
            "records are unsupported in diff mode",
            range,
        )))
    }
    fn emit(
        &self,
        _: &(),
        value: &DiffProgramOutput,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        let snapshot = copy(value, budget, range).map_err(zirql::EvaluationError::Host)?;
        (context.emit)(context.diff, snapshot, budget).map_err(zirql::EvaluationError::Host)
    }
}

fn evaluate_expression(
    expression: &Expression,
    input: DiffProgramOutput,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
) -> Result<DiffProgramOutput, EvaluationError> {
    zirql::evaluate_expression(expression, &DiffHost, input, context, bindings, budget)
        .map_err(program::evaluation_error)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Row {
    Change(ChangeId),
    Operation(DiffOperation),
    String(String),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Changes,
    Operations(DiffSide),
    Strings,
    Count,
    Map,
    Json,
    Text,
}

fn rows(value: DiffProgramOutput) -> Result<(Kind, Vec<Row>), EvaluationError> {
    Ok(match value {
        DiffProgramOutput::Changes(v) => (Kind::Changes, v.into_iter().map(Row::Change).collect()),
        DiffProgramOutput::Operations(s, v) => (
            Kind::Operations(s),
            v.into_iter().map(Row::Operation).collect(),
        ),
        DiffProgramOutput::Strings(v) => (Kind::Strings, v.into_iter().map(Row::String).collect()),
        _ => return Err(EvaluationError::new("stage requires a stream")),
    })
}
fn from_rows(kind: Kind, rows: Vec<Row>) -> DiffProgramOutput {
    match kind {
        Kind::Changes => DiffProgramOutput::Changes(
            rows.into_iter()
                .map(|row| {
                    let Row::Change(id) = row else {
                        unreachable!("stream kind matches rows")
                    };
                    id
                })
                .collect(),
        ),
        Kind::Operations(side) => DiffProgramOutput::Operations(
            side,
            rows.into_iter()
                .map(|row| {
                    let Row::Operation(id) = row else {
                        unreachable!("stream kind matches rows")
                    };
                    id
                })
                .collect(),
        ),
        Kind::Strings => DiffProgramOutput::Strings(
            rows.into_iter()
                .map(|row| {
                    let Row::String(text) = row else {
                        unreachable!("stream kind matches rows")
                    };
                    text
                })
                .collect(),
        ),
        _ => unreachable!("only stream kinds contain rows"),
    }
}
struct StreamContext<'a, 'document> {
    diff: &'a Diff<'document>,
    kind: Kind,
}

fn evaluate_stream(
    pipeline: &zirql::Pipeline<Leaf>,
    input: DiffProgramOutput,
    diff: &Diff<'_>,
    budget: &mut Budget,
) -> Result<DiffProgramOutput, EvaluationError> {
    let (kind, rows) = rows(input)?;
    let input = NativeInput::from_admitted(
        StreamContext { diff, kind },
        rows,
        budget,
        zirql::Location::Query(pipeline.range()),
    )
    .map_err(core_error)?;
    let result =
        zirql::evaluate(pipeline, &DiffHost, input, budget).map_err(program::evaluation_error)?;
    Ok(match result.value {
        zirql::NativeValue::Stream(rows) => from_rows(result.context.kind, rows),
        zirql::NativeValue::Count(count) => DiffProgramOutput::Count(count),
        zirql::NativeValue::Tally(entries) => DiffProgramOutput::Map(
            entries
                .into_iter()
                .map(|(key, count)| (key, DiffProgramOutput::Count(count)))
                .collect(),
        ),
    })
}

impl Host<Row, StreamContext<'_, '_>, Leaf> for DiffHost {
    type Error = EvaluationError;
    type Key = Row;
    fn stage_work(&self, rows: &[Row], _: &StreamContext<'_, '_>) -> usize {
        rows.len().max(1)
    }
    fn collection_item_work(&self) -> usize {
        0
    }
    fn prepare_stage(
        &self,
        stage: &Located<Stage<Leaf>>,
        context: &mut StreamContext<'_, '_>,
        _: &mut Budget,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        if matches!(stage.value, Stage::Filter(_))
            && !matches!(context.kind, Kind::Changes | Kind::Operations(_))
        {
            return Err(zirql::EvaluationError::Host(failure(
                "filter requires a change or operation stream",
                &stage.range,
            )));
        }
        if matches!(stage.value, Stage::Tally) && context.kind != Kind::Strings {
            return Err(zirql::EvaluationError::Host(failure(
                "tally requires a value stream",
                &stage.range,
            )));
        }
        if let Stage::Filter(predicate) = &stage.value {
            let mut pending = vec![predicate];
            while let Some(predicate) = pending.pop() {
                match &predicate.value {
                    zirql::PredicateKind::Leaf(Leaf::Change(_) | Leaf::Changed(_))
                        if matches!(context.kind, Kind::Operations(_)) =>
                    {
                        return Err(zirql::EvaluationError::Host(failure(
                            "change predicates require a change stream",
                            &stage.range,
                        )));
                    }
                    zirql::PredicateKind::Not(p) | zirql::PredicateKind::Group(p) => {
                        pending.push(p)
                    }
                    zirql::PredicateKind::And(ps) | zirql::PredicateKind::Or(ps) => {
                        pending.extend(ps)
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
    fn prepare_filter_item(
        &self,
        row: &Row,
        context: &mut StreamContext<'_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        validate_row(context.diff, row).map_err(zirql::EvaluationError::Host)
    }
    fn evaluate_leaf(
        &self,
        leaf: &Leaf,
        row: &Row,
        context: &mut StreamContext<'_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<Truth, zirql::EvaluationError<EvaluationError>> {
        let matched =
            matches_leaf(context.diff, row, leaf).map_err(zirql::EvaluationError::Host)?;
        Ok(if matched { Truth::True } else { Truth::False })
    }
    fn identity_key(
        &self,
        row: &Row,
        context: &mut StreamContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Row, zirql::EvaluationError<EvaluationError>> {
        validate_row(context.diff, row).map_err(zirql::EvaluationError::Host)?;
        if let Row::String(text) = row {
            admit(budget, 1, range).map_err(zirql::EvaluationError::Host)?;
            program::charge_bytes(budget, text.len(), range)
                .map_err(zirql::EvaluationError::Host)?;
        }
        Ok(row.clone())
    }
    fn tally_key(
        &self,
        row: Row,
        _: &mut StreamContext<'_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<String, zirql::EvaluationError<EvaluationError>> {
        let Row::String(text) = row else {
            unreachable!("tally preparation requires strings")
        };
        Ok(text)
    }
}

fn validate_row(diff: &Diff<'_>, row: &Row) -> Result<(), EvaluationError> {
    match row {
        Row::Change(id) => {
            diff.change(*id).map_err(diff_error)?;
        }
        Row::Operation(op) => {
            diff.operation_id(*op).map_err(diff_error)?;
        }
        Row::String(_) => {}
    }
    Ok(())
}
fn matches_leaf(diff: &Diff<'_>, row: &Row, leaf: &Leaf) -> Result<bool, EvaluationError> {
    match leaf {
        Leaf::Bool(value) => return Ok(*value),
        Leaf::Change(kind) => {
            let Row::Change(id) = row else {
                return Err(EvaluationError::new("change predicates require changes"));
            };
            let change = diff.change(*id).map_err(diff_error)?;
            return Ok(if *kind == ChangeKind::Moved {
                change.moved()
            } else {
                change.kind() == *kind
            });
        }
        Leaf::Changed(field) => {
            let Row::Change(id) = row else {
                return Err(EvaluationError::new("change predicates require changes"));
            };
            return Ok(diff
                .change(*id)
                .map_err(diff_error)?
                .fields()
                .contains(field));
        }
        _ => {}
    }
    let (side, operation) = match row {
        Row::Change(id) => diff.representative_id(*id).map_err(diff_error)?,
        Row::Operation(op) => (op.side(), diff.operation_id(*op).map_err(diff_error)?),
        Row::String(_) => {
            return Err(EvaluationError::new(
                "operation predicate requires operations",
            ));
        }
    };
    let document = diff.document(side);
    Ok(match leaf {
        Leaf::Op(name) => document.operation_name(operation) == Some(name),
        Leaf::Dialect(name) => document
            .operation_name(operation)
            .and_then(|name| name.split_once('.'))
            .is_some_and(|(dialect, _)| dialect == name),
        Leaf::ResultType(spelling) => document
            .result_types(operation)
            .unwrap_or(&[])
            .iter()
            .any(|&ty| document.type_spelling(ty) == Some(spelling)),
        Leaf::HasAttr(name) => document.attribute_id(operation, name).is_some(),
        Leaf::Attr(name, value) => {
            document
                .attribute_id(operation, name)
                .and_then(|id| document.attribute_value(id))
                .and_then(AttributeValue::decoded_string)
                .as_deref()
                == Some(value)
        }
        Leaf::Bool(_) | Leaf::Change(_) | Leaf::Changed(_) => {
            unreachable!("handled change and Boolean leaves")
        }
    })
}

fn evaluate_stage(
    stage: &DiffStage,
    value: DiffProgramOutput,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let diff = context.diff;
    match stage {
        DiffStage::Stream(_) | DiffStage::SelectionUnique(_) => {
            unreachable!("stream stages use the shared runner")
        }
        DiffStage::Nested(expression) => {
            evaluate_expression(expression, value, context, bindings, budget)
        }
        DiffStage::Input => initial_changes(diff, budget, range),
        DiffStage::Project(side) => {
            let DiffProgramOutput::Changes(changes) = value else {
                return Err(failure("side projection requires changes", range));
            };
            let mut operations = Vec::new();
            for change in changes {
                let operation = match side {
                    DiffSide::Before => diff.before_operation(change),
                    DiffSide::After => diff.after_operation(change),
                }
                .map_err(diff_error)?;
                if let Some(operation) = operation {
                    push(&mut operations, operation, budget, range)?;
                }
            }
            Ok(DiffProgramOutput::Operations(*side, operations))
        }
        DiffStage::Users(_)
        | DiffStage::Defs(_)
        | DiffStage::Parent
        | DiffStage::Children
        | DiffStage::Root(_)
        | DiffStage::Subtree
        | DiffStage::Graph(_) => navigate(diff, value, stage, context.options, budget, range),
        DiffStage::Fixpoint(body) => fixed_point(value, body, context, bindings, budget, range),
        DiffStage::Names
        | DiffStage::Attr(_)
        | DiffStage::ResultTypes
        | DiffStage::OperandTypes => project(diff, value, stage, budget, range),
        DiffStage::Reverse | DiffStage::Tail(_) => {
            let (kind, mut items) = rows(value)?;
            match stage {
                DiffStage::Reverse => items.reverse(),
                DiffStage::Tail(count) if items.len() > *count => {
                    items.drain(..items.len() - count);
                }
                _ => {}
            }
            Ok(from_rows(kind, items))
        }
        DiffStage::Sort | DiffStage::Extreme { .. } => {
            let DiffProgramOutput::Strings(mut strings) = value else {
                return Err(failure("sort and extrema require strings", range));
            };
            if let DiffStage::Extreme { maximum, all } = stage {
                let selected = if *maximum {
                    strings.iter().max()
                } else {
                    strings.iter().min()
                };
                if let Some(selected) = selected {
                    program::charge_bytes(budget, selected.len(), range)?;
                    let selected = selected.clone();
                    strings.retain(|text| text == &selected);
                    if !all {
                        strings.truncate(1);
                    }
                }
            } else {
                strings.sort();
            }
            Ok(DiffProgramOutput::Strings(strings))
        }
        DiffStage::Order { kind, selector } => {
            order(value, *kind, selector, context, bindings, budget, range)
        }
        DiffStage::MapBy {
            key,
            value: expression,
        } => map(value, key, expression, context, bindings, budget, range),
        DiffStage::Check(expected) => {
            let actual = value.len();
            if expected.map_or(actual == 0, |expected| actual != expected) {
                return Err(failure(
                    match expected {
                        Some(expected) => {
                            format!("check failed: expected {expected} items, got {actual}")
                        }
                        None => format!("check failed: expected at least one item, got {actual}"),
                    },
                    range,
                ));
            }
            Ok(value)
        }
        DiffStage::Json => {
            if matches!(value, DiffProgramOutput::Json(_, _)) {
                return Err(failure("json cannot be applied twice", range));
            }
            let side = if let DiffProgramOutput::Operations(side, _) = &value {
                Some(*side)
            } else {
                None
            };
            Ok(DiffProgramOutput::Json(
                to_json(diff, &value, budget, range)?,
                side,
            ))
        }
        DiffStage::Markdown => markdown(diff, value, budget, range).map(DiffProgramOutput::Text),
        DiffStage::Print(template) => {
            interpolate(template, bindings, budget, range).map(DiffProgramOutput::Text)
        }
        DiffStage::Noop => Ok(value),
    }
}

fn operations(
    diff: &Diff<'_>,
    value: DiffProgramOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<(DiffSide, Vec<OperationId>), EvaluationError> {
    let DiffProgramOutput::Operations(side, items) = value else {
        return Err(failure("navigation requires projected operations", range));
    };
    admit(budget, items.len(), range)?;
    let ids = items
        .into_iter()
        .map(|item| {
            if item.side() != side {
                return Err(failure(
                    "operation selection has an incompatible diff side",
                    range,
                ));
            }
            diff.operation_id(item).map_err(diff_error)
        })
        .collect::<Result<_, _>>()?;
    Ok((side, ids))
}
fn navigate(
    diff: &Diff<'_>,
    value: DiffProgramOutput,
    stage: &DiffStage,
    options: EvaluationOptions,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let (side, selected) = operations(diff, value, budget, range)?;
    let document = diff.document(side);
    let ids = match stage {
        DiffStage::Users(index) => super::evaluate_users(document, &selected, *index, budget)?,
        DiffStage::Defs(index) => operation_defs(document, &selected, *index, budget)?,
        DiffStage::Parent => super::evaluate_parent(document, &selected, budget)?,
        DiffStage::Children => super::evaluate_children(document, &selected, budget)?,
        DiffStage::Subtree => breadth_first_subtrees(document, &selected, budget, range)?,
        DiffStage::Root(filter) => {
            let mut roots = Vec::new();
            for operation in selected {
                let mut candidate = Some(operation);
                while let Some(operation) = candidate {
                    charge(budget, 1, range)?;
                    let handle = diff.scoped_operation(side, operation).map_err(diff_error)?;
                    admit(budget, 1, range)?;
                    let matches = evaluate_stream(
                        filter,
                        DiffProgramOutput::Operations(side, vec![handle]),
                        diff,
                        budget,
                    )?;
                    if matches.len() != 0 {
                        push(&mut roots, operation, budget, range)?;
                        break;
                    }
                    candidate = super::operation_parent(document, operation);
                }
            }
            roots
        }
        DiffStage::Graph(traversal) => match traversal {
            OperationTraversal::Reachable => {
                super::evaluate_reachable(document, selected, diff.registry(), options, budget)?
            }
            OperationTraversal::Closure => {
                super::evaluate_closure(document, selected, diff.registry(), false, budget)?
            }
            OperationTraversal::Slice => super::evaluate_slice(document, selected, budget)?,
            OperationTraversal::ForwardSlice => {
                super::evaluate_forward_slice(document, selected, budget)?
            }
        },
        _ => unreachable!("only navigation stages enter navigate"),
    };
    // The domain traversal has already admitted the returned stream.
    Ok(DiffProgramOutput::Operations(
        side,
        ids.into_iter()
            .map(|id| diff.scoped_operation(side, id).map_err(diff_error))
            .collect::<Result<_, _>>()?,
    ))
}

// Diff navigation retains its established operation-result-only definition edge.
// Ordinary defs additionally maps block arguments to their owning operation.
fn operation_defs(
    document: &crate::semantic::Document,
    selected: &[OperationId],
    index: Option<usize>,
    budget: &mut Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    super::collect(
        budget,
        selected
            .iter()
            .flat_map(|&operation| {
                let operands = document.operands(operation).unwrap_or(&[]);
                match index {
                    Some(index) => operands.get(index).map(std::slice::from_ref).unwrap_or(&[]),
                    None => operands,
                }
            })
            .filter_map(|operand| match operand {
                crate::semantic::ValueReference::Resolved(
                    crate::semantic::ValueId::OperationResult { operation, .. },
                ) => Some(*operation),
                crate::semantic::ValueReference::Resolved(
                    crate::semantic::ValueId::BlockArgument { .. },
                )
                | crate::semantic::ValueReference::Invalid(_) => None,
            }),
    )
}

fn breadth_first_subtrees(
    document: &crate::semantic::Document,
    selected: &[OperationId],
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut expanded = Vec::new();
    for &root in selected {
        let mut cursor = expanded.len();
        push(&mut expanded, root, budget, range)?;
        while cursor < expanded.len() {
            charge(budget, 1, range)?;
            let children = super::evaluate_children(document, &[expanded[cursor]], budget)?;
            for child in children {
                check(budget, expanded.len().saturating_add(1), range)?;
                // Moving the already admitted child does not produce another item.
                expanded.push(child);
            }
            cursor += 1;
        }
    }
    Ok(expanded)
}

fn fixed_point(
    value: DiffProgramOutput,
    body: &Expression,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let DiffProgramOutput::Operations(side, mut current) = value else {
        return Err(failure("fixpoint requires projected operations", range));
    };
    let mut checkpoint = copy_operations(side, &current, budget, range)?;
    let mut power = 1usize;
    let mut distance = 0usize;
    loop {
        let input = copy_operations(side, &current, budget, range)?;
        let next = evaluate_expression(body, input, context, bindings, budget)?;
        let DiffProgramOutput::Operations(next_side, next) = next else {
            return Err(failure("fixpoint body must return operations", range));
        };
        if next_side != side {
            return Err(failure(
                "fixpoint body must preserve the projected diff side",
                range,
            ));
        }
        charge(budget, 1, range)?;
        if next == current {
            return Ok(DiffProgramOutput::Operations(side, current));
        }
        if let DiffProgramOutput::Operations(_, checkpoint) = &checkpoint
            && next == *checkpoint
        {
            return Err(failure(
                "fixpoint query cycles without reaching an unchanged selection",
                range,
            ));
        }
        current = next;
        distance = distance.saturating_add(1);
        if distance == power {
            checkpoint = copy_operations(side, &current, budget, range)?;
            power = power.saturating_mul(2);
            distance = 0;
        }
    }
}
fn copy_operations(
    side: DiffSide,
    operations: &[DiffOperation],
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    check(budget, operations.len(), range)?;
    admit(budget, operations.len(), range)?;
    Ok(DiffProgramOutput::Operations(side, operations.to_vec()))
}

fn project(
    diff: &Diff<'_>,
    value: DiffProgramOutput,
    stage: &DiffStage,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let mut strings = Vec::new();
    let (kind, rows) = rows(value)?;
    if !matches!(kind, Kind::Changes | Kind::Operations(_)) {
        return Err(failure("projection requires changes or operations", range));
    }
    for row in rows {
        let (side, operation) = match row {
            Row::Change(id) => diff.representative_id(id).map_err(diff_error)?,
            Row::Operation(op) => (op.side(), diff.operation_id(op).map_err(diff_error)?),
            Row::String(_) => unreachable!("projection validated stream"),
        };
        let document = diff.document(side);
        match stage {
            DiffStage::Names => {
                if let Some(name) = document.operation_name(operation) {
                    add_string(&mut strings, name, budget, range)?;
                }
            }
            DiffStage::Attr(name) => {
                if let Some(id) = document.attribute_id(operation, name) {
                    match document.attribute_value(id) {
                        Some(value @ AttributeValue::String(spelling)) => {
                            check(budget, strings.len().saturating_add(1), range)?;
                            admit(budget, 1, range)?;
                            program::charge_bytes(budget, spelling.len(), range)?;
                            if let Some(text) = value.decoded_string() {
                                strings.push(text);
                            }
                        }
                        Some(AttributeValue::Symbol(path)) => {
                            check(budget, strings.len().saturating_add(1), range)?;
                            admit(budget, 1, range)?;
                            let bytes = path
                                .iter()
                                .map(String::len)
                                .sum::<usize>()
                                .saturating_add(path.len().saturating_sub(1).saturating_mul(2));
                            program::charge_bytes(budget, bytes, range)?;
                            strings.push(path.join("::"));
                        }
                        _ => {
                            if let Some(text) = document.attribute_spelling_value(id) {
                                add_string(&mut strings, text, budget, range)?;
                            }
                        }
                    }
                }
            }
            DiffStage::ResultTypes => {
                for &ty in document.result_types(operation).unwrap_or(&[]) {
                    if let Some(text) = document.type_spelling(ty) {
                        add_string(&mut strings, text, budget, range)?;
                    }
                }
            }
            DiffStage::OperandTypes => {
                for &operand in document.operands(operation).unwrap_or(&[]) {
                    if let Some(text) = document.value_type(operand) {
                        add_string(&mut strings, text, budget, range)?;
                    }
                }
            }
            _ => unreachable!("only projection stages enter project"),
        }
    }
    Ok(DiffProgramOutput::Strings(strings))
}

fn add_string(
    strings: &mut Vec<String>,
    text: &str,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    check(budget, strings.len().saturating_add(1), range)?;
    admit(budget, 1, range)?;
    program::charge_bytes(budget, text.len(), range)?;
    strings.push(text.to_owned());
    Ok(())
}

fn combine(
    diff: &Diff<'_>,
    left: DiffProgramOutput,
    right: DiffProgramOutput,
    operator: SetOperator,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let (left_kind, left) = rows(left)?;
    let (right_kind, right) = rows(right)?;
    if left_kind != right_kind {
        return Err(failure(
            if matches!(
                (left_kind, right_kind),
                (Kind::Operations(_), Kind::Operations(_))
            ) {
                "set operations require operation selections from the same diff side"
            } else {
                "set operations require selections of the same kind"
            },
            range,
        ));
    }
    for row in left.iter().chain(&right) {
        validate_row(diff, row)?;
    }
    charge(budget, left.len().saturating_add(right.len()), range)?;
    let left: HashSet<_> = left.into_iter().collect();
    let right: HashSet<_> = right.into_iter().collect();
    let mut selected = Vec::new();
    let mut append = |row: Row| -> Result<(), EvaluationError> {
        charge(budget, 1, range)?;
        if match operator {
            SetOperator::Union => left.contains(&row) || right.contains(&row),
            SetOperator::Intersect => left.contains(&row) && right.contains(&row),
            SetOperator::Except => left.contains(&row) && !right.contains(&row),
        } {
            push(&mut selected, row, budget, range)?;
        }
        Ok(())
    };
    match left_kind {
        Kind::Changes => {
            for id in diff.change_ids() {
                append(Row::Change(id))?;
            }
        }
        Kind::Operations(side) => {
            for id in diff.document(side).operations() {
                append(Row::Operation(
                    diff.scoped_operation(side, id).map_err(diff_error)?,
                ))?;
            }
        }
        _ => {
            return Err(failure(
                "set operations require selections of the same kind",
                range,
            ));
        }
    }
    Ok(from_rows(left_kind, selected))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SortKey {
    Count(usize),
    Text(String),
}
fn order(
    value: DiffProgramOutput,
    kind: Order,
    selector: &Expression,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let (stream, items) = rows(value)?;
    if !matches!(stream, Kind::Changes | Kind::Operations(_)) {
        return Err(failure(
            "keyed ordering requires changes or operations",
            range,
        ));
    }
    admit(budget, items.len(), range)?;
    let mut keyed = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        admit(budget, 1, range)?;
        let input = from_rows(stream, vec![item.clone()]);
        let key = evaluate_expression(selector, input, context, bindings, budget)?;
        let key = match key {
            DiffProgramOutput::Count(count) => SortKey::Count(count),
            DiffProgramOutput::Strings(mut strings) if strings.len() == 1 => {
                SortKey::Text(strings.pop().unwrap())
            }
            _ => {
                return Err(failure(
                    "ordering key must produce exactly one scalar",
                    range,
                ));
            }
        };
        keyed.push((key, index, item));
    }
    keyed.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    if !keyed.is_empty() && !matches!(kind, Order::Sort) {
        let key = if matches!(kind, Order::Max | Order::MaxAll) {
            &keyed.last().unwrap().0
        } else {
            &keyed[0].0
        };
        if let SortKey::Text(text) = key {
            program::charge_bytes(budget, text.len(), range)?;
        }
        let key = key.clone();
        keyed.retain(|(candidate, _, _)| candidate == &key);
        keyed.sort_by_key(|(_, index, _)| *index);
        if matches!(kind, Order::Min | Order::Max) {
            keyed.truncate(1);
        }
    }
    Ok(from_rows(
        stream,
        keyed.into_iter().map(|(_, _, item)| item).collect(),
    ))
}
fn map(
    value: DiffProgramOutput,
    key: &Expression,
    expression: &Expression,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let (kind, items) = rows(value)?;
    if !matches!(kind, Kind::Changes | Kind::Operations(_)) {
        return Err(failure(
            "map_by requires a change or operation stream",
            range,
        ));
    }
    admit(budget, 1, range)?;
    let mut entries = BTreeMap::new();
    for item in items {
        admit(budget, 1, range)?;
        let input = from_rows(kind, vec![item]);
        let key_input = copy(&input, budget, range)?;
        let result = evaluate_expression(key, key_input, context, bindings, budget)?;
        let DiffProgramOutput::Strings(mut keys) = result else {
            return Err(failure("map_by key must produce exactly one string", range));
        };
        if keys.len() != 1 {
            return Err(failure("map_by key must produce exactly one string", range));
        }
        let key = keys.pop().unwrap();
        if entries.contains_key(&key) {
            return Err(failure(
                format!("map_by encountered duplicate key `{key}`; select unique keys"),
                range,
            ));
        }
        let mapped = evaluate_expression(expression, input, context, bindings, budget)?;
        check(budget, entries.len().saturating_add(1), range)?;
        admit(budget, 1, range)?;
        entries.insert(key, mapped);
    }
    Ok(DiffProgramOutput::Map(entries))
}

fn json_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(text) => text.len(),
        serde_json::Value::Array(values) => values.iter().map(json_bytes).sum(),
        serde_json::Value::Object(entries) => entries
            .iter()
            .map(|(key, value)| key.len() + json_bytes(value))
            .sum(),
        _ => 0,
    }
}

/// Render native diff data only at an explicit JSON boundary.
///
/// # Errors
/// Refuses foreign handles or exhaustion of the caller's result allowance.
pub fn diff_value_to_json(
    diff: &Diff<'_>,
    value: &DiffProgramOutput,
    budget: &mut Budget,
) -> Result<serde_json::Value, EvaluationError> {
    to_json(diff, value, budget, &(0..0))
}
fn to_json(
    diff: &Diff<'_>,
    value: &DiffProgramOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<serde_json::Value, EvaluationError> {
    Ok(match value {
        DiffProgramOutput::Changes(items) => {
            let (units, bytes) = diff.selection_report_size(items).map_err(diff_error)?;
            // First admit copied report data, then charge each serialized chunk
            // before the writer extends its buffer. Parsing the JSON tree is a
            // second materialization of the same bounded report.
            admit(budget, units.saturating_mul(2), range)?;
            program::charge_bytes(budget, bytes.saturating_mul(2), range)?;
            let mut bytes = Vec::new();
            diff.write_selection_json(
                program::BudgetWriter {
                    bytes: &mut bytes,
                    budget,
                    range,
                },
                items,
            )
            .map_err(|error| failure(error.to_string(), range))?;
            serde_json::from_slice(&bytes).map_err(|error| failure(error.to_string(), range))?
        }
        DiffProgramOutput::Operations(side, items) => {
            admit(budget, 1 + items.len().saturating_mul(7), range)?;
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                if item.side() != *side {
                    return Err(failure(
                        "operation selection has an incompatible diff side",
                        range,
                    ));
                }
                let operation = diff.operation_id(*item).map_err(diff_error)?;
                let name = diff.document(*side).operation_name(operation);
                program::charge_bytes(budget, 9 + name.map_or(0, str::len), range)?;
                values.push(serde_json::json!({ "name": name, "range": diff.document(*side).operation_source_range(operation).map(|range| [range.start(),range.end()]) }));
            }
            serde_json::Value::Array(values)
        }
        DiffProgramOutput::Strings(items) => {
            admit(budget, 1 + items.len(), range)?;
            program::charge_bytes(budget, items.iter().map(String::len).sum(), range)?;
            serde_json::json!(items)
        }
        DiffProgramOutput::Count(count) => {
            admit(budget, 1, range)?;
            serde_json::json!(count)
        }
        DiffProgramOutput::Map(entries) => {
            admit(budget, 1 + entries.len(), range)?;
            let mut json = serde_json::Map::new();
            for (key, value) in entries {
                program::charge_bytes(budget, key.len(), range)?;
                json.insert(key.clone(), to_json(diff, value, budget, range)?);
            }
            serde_json::Value::Object(json)
        }
        DiffProgramOutput::Json(value, _) => {
            admit(budget, super::json_size(value), range)?;
            program::charge_bytes(budget, json_bytes(value), range)?;
            value.clone()
        }
        DiffProgramOutput::Text(text) => {
            admit(budget, 1, range)?;
            program::charge_bytes(budget, text.len(), range)?;
            serde_json::Value::String(text.clone())
        }
    })
}
fn markdown(
    diff: &Diff<'_>,
    value: DiffProgramOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<String, EvaluationError> {
    fn cell(text: &str) -> String {
        text.replace('\\', "\\\\")
            .replace('|', "\\|")
            .replace('\n', "<br>")
    }
    admit(budget, 1, range)?;
    let text = match value {
        DiffProgramOutput::Changes(items) => {
            let (units, bytes) = diff.selection_report_size(&items).map_err(diff_error)?;
            program::charge_bytes(
                budget,
                bytes
                    .saturating_add(units.saturating_mul(20))
                    .saturating_mul(6),
                range,
            )?;
            diff.selection_to_markdown(&items).map_err(diff_error)?
        }
        DiffProgramOutput::Strings(items) => {
            program::charge_bytes(budget, 18, range)?;
            let mut output = String::from("| Value |\n| --- |\n");
            for item in items {
                program::charge_bytes(budget, item.len().saturating_mul(4) + 6, range)?;
                output.push_str(&format!("| {} |\n", cell(&item)));
            }
            output
        }
        DiffProgramOutput::Map(entries) => {
            program::charge_bytes(budget, 30, range)?;
            let mut output = String::from("| Key | Value |\n| --- | --- |\n");
            for (key, value) in entries {
                let json = to_json(diff, &value, budget, range)?;
                let mut value = program::serialize_json(&json, budget, range, false)?;
                value.pop();
                program::charge_bytes(
                    budget,
                    (key.len() + value.len()).saturating_mul(4) + 9,
                    range,
                )?;
                output.push_str(&format!("| {} | {} |\n", cell(&key), cell(&value)));
            }
            output
        }
        _ => {
            return Err(failure(
                "markdown requires changes, values, or a map",
                range,
            ));
        }
    };
    Ok(text)
}
fn interpolate(
    template: &str,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<String, EvaluationError> {
    admit(budget, 1, range)?;
    program::charge_bytes(budget, template.len() + 1, range)?;
    let mut output = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        output.push_str(&rest[..open]);
        rest = &rest[open + 1..];
        if rest.starts_with('{') {
            output.push('{');
            rest = &rest[1..];
            continue;
        }
        let close = rest
            .find('}')
            .ok_or_else(|| failure("unterminated interpolation in print", range))?;
        let name = &rest[..close];
        let value = bindings.get(name, range.clone()).map_err(core_error)?;
        match value {
            DiffProgramOutput::Count(count) => {
                charge(
                    budget,
                    usize::try_from(count.checked_ilog10().unwrap_or(0))
                        .expect("usize represents the number of decimal digits")
                        + 1,
                    range,
                )?;
                output.push_str(&count.to_string());
            }
            DiffProgramOutput::Strings(values) if values.len() == 1 => {
                charge(budget, values[0].len(), range)?;
                output.push_str(&values[0]);
            }
            _ => {
                return Err(failure(
                    format!(
                        "string interpolation `{{{name}}}` requires a count or exactly one string"
                    ),
                    range,
                ));
            }
        }
        rest = &rest[close + 1..];
    }
    output.push_str(rest);
    output.push('\n');
    Ok(output)
}

fn validate_program(
    program: &zirql::Program<DiffStage, SetOperator, ()>,
    mut names: BTreeMap<String, Kind>,
) -> Result<(), EvaluationError> {
    for statement in program.statements() {
        let (expression, name) = match &statement.value {
            zirql::Statement::Bind { name, expression } => (expression, Some(name)),
            zirql::Statement::Evaluate { expression, .. } => (expression, None),
        };
        let kind = validate_expression(expression, Kind::Changes, &names)?;
        if let Some(name) = name {
            if matches!(
                name.value.as_str(),
                "before" | "after" | "before_document" | "after_document"
            ) {
                return Err(failure(
                    format!("binding name `{}` is reserved in diff mode", name.value),
                    &name.range,
                ));
            }
            if names.insert(name.value.clone(), kind).is_some() {
                return Err(failure(
                    format!("binding `{}` is already defined", name.value),
                    &name.range,
                ));
            }
        }
    }
    Ok(())
}
fn validate_expression(
    expression: &Expression,
    input: Kind,
    names: &BTreeMap<String, Kind>,
) -> Result<Kind, EvaluationError> {
    // The left-associated set spine is flat language structure, so inspect it
    // iteratively rather than treating each set term as another nesting level.
    let mut rights = Vec::new();
    let mut first = expression;
    while let ExpressionKind::Set { left, right, .. } = &first.value {
        rights.push(right.as_ref());
        first = left;
    }
    let kind = match &first.value {
        ExpressionKind::Input => input,
        ExpressionKind::Binding(name) => *names
            .get(name)
            .ok_or_else(|| failure(format!("undefined binding: {name}"), &first.range))?,
        ExpressionKind::Group(inner) => validate_expression(inner, input, names)?,
        ExpressionKind::Sequence {
            input: expression,
            steps,
        } => {
            let mut kind = validate_expression(expression, input, names)?;
            for step in steps {
                let ExpressionStep::Stage(stage) = &step.value else {
                    unreachable!("diff emitters are value stages")
                };
                kind = validate_stage(stage, kind, names, &step.range)?;
            }
            kind
        }
        ExpressionKind::Set { .. } => unreachable!("the flat set spine was removed"),
        ExpressionKind::Record(_) => {
            return Err(failure(
                "records are unsupported in diff mode",
                &first.range,
            ));
        }
    };
    for right in rights.into_iter().rev() {
        let other = validate_expression(right, input, names)?;
        if kind != other || !matches!(kind, Kind::Changes | Kind::Operations(_)) {
            return Err(failure(
                if matches!((kind, other), (Kind::Operations(_), Kind::Operations(_))) {
                    "set operations require operation selections from the same diff side"
                } else {
                    "set operations require selections of the same kind"
                },
                &expression.range,
            ));
        }
    }
    Ok(kind)
}
fn validate_stage(
    stage: &DiffStage,
    input: Kind,
    names: &BTreeMap<String, Kind>,
    range: &Range<usize>,
) -> Result<Kind, EvaluationError> {
    let require = |valid: bool, message: &str| {
        if valid {
            Ok(())
        } else {
            Err(failure(message, range))
        }
    };
    let selection = matches!(input, Kind::Changes | Kind::Operations(_));
    let stream = selection || input == Kind::Strings;
    Ok(match stage {
        DiffStage::Stream(pipeline) | DiffStage::SelectionUnique(pipeline) => {
            if matches!(stage, DiffStage::SelectionUnique(_)) {
                require(selection, "unique requires a change or operation stream")?;
            }
            let mut kind = input;
            for stage in pipeline.stages() {
                match &stage.value {
                    Stage::Filter(predicate) => {
                        require(selection, "filter requires a change or operation stream")?;
                        let mut pending = vec![predicate];
                        while let Some(predicate) = pending.pop() {
                            match &predicate.value {
                                zirql::PredicateKind::Leaf(Leaf::Change(_) | Leaf::Changed(_)) => {
                                    require(
                                        input == Kind::Changes,
                                        "change predicates require a change stream",
                                    )?
                                }
                                zirql::PredicateKind::Not(p) | zirql::PredicateKind::Group(p) => {
                                    pending.push(p)
                                }
                                zirql::PredicateKind::And(ps) | zirql::PredicateKind::Or(ps) => {
                                    pending.extend(ps)
                                }
                                _ => {}
                            }
                        }
                    }
                    Stage::Head(_) | Stage::Unique => require(stream, "stage requires a stream")?,
                    Stage::Count => {
                        require(stream, "count requires a stream")?;
                        kind = Kind::Count;
                    }
                    Stage::Tally => {
                        require(input == Kind::Strings, "tally requires a value stream")?;
                        kind = Kind::Map;
                    }
                    _ => unreachable!("diff native pipelines contain only shared stages"),
                }
            }
            kind
        }
        DiffStage::Nested(expression) => validate_expression(expression, input, names)?,
        DiffStage::Input => Kind::Changes,
        DiffStage::Project(side) => {
            require(input == Kind::Changes, "side projection requires changes")?;
            Kind::Operations(*side)
        }
        DiffStage::Users(_)
        | DiffStage::Defs(_)
        | DiffStage::Parent
        | DiffStage::Children
        | DiffStage::Root(_)
        | DiffStage::Subtree
        | DiffStage::Graph(_) => {
            require(
                matches!(input, Kind::Operations(_)),
                "navigation requires projected operations",
            )?;
            input
        }
        DiffStage::Fixpoint(body) => {
            require(
                matches!(input, Kind::Operations(_)),
                "fixpoint requires projected operations",
            )?;
            let result = validate_expression(body, input, names)?;
            require(
                matches!(result, Kind::Operations(_)),
                "fixpoint body must return operations",
            )?;
            require(
                result == input,
                "fixpoint body must preserve the projected diff side",
            )?;
            input
        }
        DiffStage::Names
        | DiffStage::Attr(_)
        | DiffStage::ResultTypes
        | DiffStage::OperandTypes => {
            require(selection, "projection requires changes or operations")?;
            Kind::Strings
        }
        DiffStage::Reverse | DiffStage::Tail(_) => {
            require(stream, "stage requires a stream")?;
            input
        }
        DiffStage::Sort | DiffStage::Extreme { .. } => {
            require(input == Kind::Strings, "sort and extrema require strings")?;
            input
        }
        DiffStage::Order { selector, .. } => {
            require(selection, "keyed ordering requires changes or operations")?;
            let kind = validate_expression(selector, input, names)?;
            require(
                matches!(kind, Kind::Strings | Kind::Count),
                "ordering key must produce exactly one scalar",
            )?;
            input
        }
        DiffStage::MapBy { key, value } => {
            require(selection, "map_by requires a change or operation stream")?;
            require(
                validate_expression(key, input, names)? == Kind::Strings,
                "map_by key must produce exactly one string",
            )?;
            validate_expression(value, input, names)?;
            Kind::Map
        }
        DiffStage::Check(_) | DiffStage::Noop => input,
        DiffStage::Json => {
            require(input != Kind::Json, "json cannot be applied twice")?;
            Kind::Json
        }
        DiffStage::Markdown => {
            require(
                matches!(input, Kind::Changes | Kind::Strings | Kind::Map),
                "markdown requires changes, values, or a map",
            )?;
            Kind::Text
        }
        DiffStage::Print(template) => {
            let mut rest = template.as_str();
            while let Some(open) = rest.find('{') {
                rest = &rest[open + 1..];
                if rest.starts_with('{') {
                    rest = &rest[1..];
                    continue;
                }
                let close = rest
                    .find('}')
                    .ok_or_else(|| failure("unterminated interpolation in print", range))?;
                let name = &rest[..close];
                require(
                    names
                        .get(name)
                        .is_some_and(|kind| matches!(kind, Kind::Count | Kind::Strings)),
                    &format!("binding `{name}` is unavailable or incompatible in diff mode"),
                )?;
                rest = &rest[close + 1..];
            }
            Kind::Text
        }
    })
}
