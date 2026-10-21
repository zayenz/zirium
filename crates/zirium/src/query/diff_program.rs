//! Shared diff execution with native comparison handles and host-owned operations.

use super::{
    EvaluationError, EvaluationLimits, EvaluationOptions, OperationTraversal, core_error,
    evaluation_budget, program,
};
use crate::{
    diff::{ChangeField, ChangeId, ChangeKind, Diff, DiffOperation, DiffSide},
    semantic::{AttributeValue, OperationId},
};
use std::sync::Arc;
use std::{borrow::Cow, collections::BTreeMap, ops::Range};
use zirql::ProjectionHost;
use zirql::{
    Bindings, Budget, ExpressionKind, Host, Located, NativeInput, ProgramHost, Stage, Truth,
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
    Constructed(super::comprehension::ConstructedReport),
    Map(Vec<(String, DiffProgramOutput)>),
    Json(serde_json::Value, Option<DiffSide>),
    Text(String),
}

/// A parsed diff program. The diff frontend retains its existing syntax and
/// statement-end output policy while Zirql executes statements and expressions.
#[derive(Debug)]
pub struct DiffProgram {
    pub(super) program: zirql::Program<DiffStage, SetOperator, ()>,
    pub(super) parse_work: usize,
    pub(super) parse_results: usize,
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
        budget
            .charge(
                self.parse_work,
                zirql::Location::Query(self.program.range()),
            )
            .map_err(core_error)?;
        budget
            .charge_results(
                self.parse_results,
                zirql::Location::Query(self.program.range()),
            )
            .map_err(core_error)?;
        let names = BTreeMap::from([
            ("input".to_owned(), Kind::Changes),
            ("before_document".to_owned(), Kind::Strings),
            ("after_document".to_owned(), Kind::Strings),
        ]);
        validate_program(&self.program, names, budget)?;
        let input = initial_changes(diff, budget, &self.program.range())?;
        let mut initial = vec![(
            Located::new("input".to_owned(), 0..0),
            copy(&input, budget, &self.program.range())?,
        )];
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
        zirql::evaluate_program_with(
            &self.program,
            &DiffHost,
            input,
            &mut context,
            bindings,
            budget,
            &mut |expression, input, context, bindings, budget| {
                zirql::evaluate_native_expression(
                    expression,
                    &DiffProjectionHost {
                        reject_string_unique: true,
                    },
                    input,
                    context,
                    bindings,
                    budget,
                )
            },
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
        self.0.take().expect("retained expression")
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
            | zirql::Statement::BindFields { expression, .. }
            | zirql::Statement::Evaluate { expression, .. } => expression,
            zirql::Statement::Parameter { .. } => continue,
            zirql::Statement::BindPattern { pattern, value } => {
                if let Ok(pattern) = std::sync::Arc::try_unwrap(pattern) {
                    zirql::dispose_pattern(pattern);
                }
                if let Ok(value) = std::sync::Arc::try_unwrap(value) {
                    zirql::dispose_native_value(value);
                }
                continue;
            }
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
            ExpressionKind::Field { value, .. } => pending.push(*value),
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

pub(super) use zirql::NativeSetOperator as SetOperator;

#[derive(Debug)]
pub(super) struct SharedValue(pub(super) Arc<zirql::ValueExpression<Expression>>);
impl Clone for SharedValue {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
impl Drop for SharedValue {
    fn drop(&mut self) {
        if let Some(value) = Arc::get_mut(&mut self.0) {
            let value = std::mem::replace(
                value,
                Located::new(
                    zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::Absent),
                    0..0,
                ),
            );
            zirql::dispose_native_value(value);
        }
    }
}
#[derive(Clone, Debug)]
pub(super) enum Leaf {
    Expression(SharedValue),
    Dynamic {
        name: String,
        arguments: Vec<SharedValue>,
    },
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
    Guard(Leaf),
    Input,
    Project(DiffSide),
    Users(Option<usize>),
    Defs(Option<usize>),
    Parent,
    Children,
    Root(zirql::Pipeline<Leaf>),
    Subtree,
    Graph(OperationTraversal),
    Names,
    Attr(String),
    ResultTypes,
    OperandTypes,
    Json,
    Markdown,
    Print(String),
    Noop,
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
    validate_shared(
        expression,
        ValueKind::Native(Kind::Changes),
        &BTreeMap::new(),
        &DiffProjectionHost,
        budget,
    )?;
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
            Self::Count(_) | Self::Json(_, _) | Self::Text(_) | Self::Constructed(_) => 1,
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
        DiffProgramOutput::Constructed(value) => value
            .copy(budget, range)
            .map(DiffProgramOutput::Constructed),
        DiffProgramOutput::Map(entries) => {
            let mut result = Vec::with_capacity(entries.len());
            for (key, value) in entries {
                program::charge_bytes(budget, key.len(), range)?;
                result.push((key.clone(), copy(value, budget, range)?));
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
        let result = charge(budget, value.len().max(1), range)
            .and_then(|()| evaluate_stage(stage, value, context, bindings, budget, range));
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
    zirql::evaluate_native_expression(
        expression,
        &DiffProjectionHost,
        input,
        context,
        bindings,
        budget,
    )
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
    Constructed,
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
struct StreamContext<'a, 'c, 'document> {
    context: &'a mut Context<'c, 'document>,
    bindings: &'a Bindings<DiffProgramOutput>,
    kind: Kind,
}

fn evaluate_stream(
    pipeline: &zirql::Pipeline<Leaf>,
    input: DiffProgramOutput,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
) -> Result<DiffProgramOutput, EvaluationError> {
    let (kind, rows) = rows(input)?;
    let input = NativeInput::from_admitted(
        StreamContext {
            context,
            bindings,
            kind,
        },
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

impl Host<Row, StreamContext<'_, '_, '_>, Leaf> for DiffHost {
    type Error = EvaluationError;
    type Key = Row;
    fn stage_work(&self, rows: &[Row], _: &StreamContext<'_, '_, '_>) -> usize {
        rows.len().max(1)
    }
    fn collection_item_work(&self) -> usize {
        0
    }
    fn prepare_stage(
        &self,
        stage: &Located<Stage<Leaf>>,
        context: &mut StreamContext<'_, '_, '_>,
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
        context: &mut StreamContext<'_, '_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        validate_row(context.context.diff, row).map_err(zirql::EvaluationError::Host)
    }
    fn evaluate_leaf(
        &self,
        leaf: &Leaf,
        row: &Row,
        context: &mut StreamContext<'_, '_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Truth, zirql::EvaluationError<EvaluationError>> {
        let input =
            row_value(row, context.kind, budget, range).map_err(zirql::EvaluationError::Host)?;
        let matched = match leaf {
            Leaf::Expression(value) => {
                let zirql::ValueExpressionKind::PredicateScope { predicate, .. } = &value.0.value
                else {
                    unreachable!("bound predicate")
                };
                zirql::evaluate_native_predicate(
                    predicate,
                    &DiffProjectionHost,
                    input,
                    context.context,
                    context.bindings,
                    budget,
                )?
            }
            Leaf::Dynamic { name, arguments } => {
                let leaf = resolve_leaf(
                    name,
                    arguments,
                    input,
                    context.context,
                    context.bindings,
                    budget,
                )
                .map_err(zirql::EvaluationError::Host)?;
                matches_leaf(context.context.diff, row, &leaf)
                    .map_err(zirql::EvaluationError::Host)?
            }
            _ => matches_leaf(context.context.diff, row, leaf)
                .map_err(zirql::EvaluationError::Host)?,
        };
        Ok(if matched { Truth::True } else { Truth::False })
    }
    fn identity_key(
        &self,
        row: &Row,
        context: &mut StreamContext<'_, '_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Row, zirql::EvaluationError<EvaluationError>> {
        validate_row(context.context.diff, row).map_err(zirql::EvaluationError::Host)?;
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
        _: &mut StreamContext<'_, '_, '_>,
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
        Leaf::Expression(_)
        | Leaf::Dynamic { .. }
        | Leaf::Bool(_)
        | Leaf::Change(_)
        | Leaf::Changed(_) => {
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
        DiffStage::Guard(_) => Err(failure("predicate guard is not an executable stage", range)),
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
        | DiffStage::Graph(_) => navigate(value, stage, context, bindings, budget, range),
        DiffStage::Names
        | DiffStage::Attr(_)
        | DiffStage::ResultTypes
        | DiffStage::OperandTypes => project(diff, value, stage, budget, range),
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
    value: DiffProgramOutput,
    stage: &DiffStage,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    let diff = context.diff;
    let options = context.options;
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
                        context,
                        bindings,
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
    let location = zirql::Location::Query(range.clone());
    match (left, right) {
        (DiffProgramOutput::Changes(left), DiffProgramOutput::Changes(right)) => {
            for id in left.iter().chain(&right) {
                charge(budget, 1, range)?;
                diff.change(*id).map_err(diff_error)?;
            }
            zirql::combine_native_set(
                operator,
                left.iter().copied(),
                right.iter().copied(),
                left.iter().chain(&right).copied().map(Ok),
                budget,
                location,
                |_, _| Ok(()),
            )
            .map(DiffProgramOutput::Changes)
            .map_err(program::evaluation_error)
        }
        (
            DiffProgramOutput::Operations(side, left),
            DiffProgramOutput::Operations(right_side, right),
        ) => {
            if side != right_side {
                return Err(failure(
                    "set operations require operation selections from the same diff side",
                    range,
                ));
            }
            for operation in left.iter().chain(&right) {
                charge(budget, 1, range)?;
                if operation.side() != side {
                    return Err(failure(
                        "set operation handle has a different diff side",
                        range,
                    ));
                }
                let id = diff.operation_id(*operation).map_err(diff_error)?;
                diff.document(side)
                    .check_operation(id)
                    .map_err(|error| failure(error.to_string(), range))?;
            }
            let selected = zirql::combine_native_set(
                operator,
                left.iter().copied(),
                right.iter().copied(),
                left.iter().chain(&right).copied().map(Ok),
                budget,
                location,
                |_, _| Ok(()),
            )
            .map_err(program::evaluation_error)?;
            Ok(DiffProgramOutput::Operations(side, selected))
        }
        (
            DiffProgramOutput::Changes(_)
            | DiffProgramOutput::Operations(_, _)
            | DiffProgramOutput::Strings(_),
            DiffProgramOutput::Changes(_)
            | DiffProgramOutput::Operations(_, _)
            | DiffProgramOutput::Strings(_),
        ) => Err(failure(
            "set operations require selections of the same kind",
            range,
        )),
        _ => Err(failure("stage requires a stream", range)),
    }
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
    zirql::to_report_json_value(
        &mut DiffRenderer { diff, range },
        value,
        budget,
        zirql::Location::Query(range.clone()),
    )
    .map_err(program::evaluation_error)
}

struct DiffRenderer<'a, 'd> {
    diff: &'a Diff<'d>,
    range: &'a Range<usize>,
}
impl zirql::RenderHost for DiffRenderer<'_, '_> {
    type Value = DiffProgramOutput;
    type Error = EvaluationError;

    fn view<'a>(
        &mut self,
        value: &'a DiffProgramOutput,
        _context: zirql::RenderContext,
        budget: &mut Budget,
        _location: &zirql::Location,
    ) -> Result<zirql::RenderView<'a, DiffProgramOutput>, EvaluationError> {
        use zirql::RenderView as View;
        Ok(match value {
            DiffProgramOutput::Constructed(value) => {
                View::Constructed(value.value(), zirql::JsonRepresentation::Ordinary)
            }
            DiffProgramOutput::Map(fields) => View::Record(fields),
            native => View::Projected(self.project_native(native, budget)?),
        })
    }
}
impl DiffRenderer<'_, '_> {
    fn project_native(
        &self,
        value: &DiffProgramOutput,
        budget: &mut Budget,
    ) -> Result<serde_json::Value, EvaluationError> {
        let diff = self.diff;
        let range = self.range;
        Ok(match value {
            DiffProgramOutput::Changes(items) => {
                let (units, bytes) = diff.selection_report_size(items).map_err(diff_error)?;
                admit(budget, units, range)?;
                program::charge_bytes(budget, bytes, range)?;
                diff.selection_json_value(items).map_err(diff_error)?
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
            DiffProgramOutput::Constructed(_) | DiffProgramOutput::Map(_) => {
                unreachable!("borrowed report containers")
            }
        })
    }
}

fn markdown(
    diff: &Diff<'_>,
    value: DiffProgramOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<String, EvaluationError> {
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
        // Diff reports historically retain headers for empty ordinary tables.
        DiffProgramOutput::Strings(ref items) if items.is_empty() => {
            program::charge_bytes(budget, 18, range)?;
            "| Value |\n| --- |\n".to_owned()
        }
        DiffProgramOutput::Map(ref entries) if entries.is_empty() => {
            program::charge_bytes(budget, 30, range)?;
            "| Key | Value |\n| --- | --- |\n".to_owned()
        }
        value @ (DiffProgramOutput::Strings(_)
        | DiffProgramOutput::Map(_)
        | DiffProgramOutput::Constructed(_)) => {
            let bytes = zirql::render_report_markdown(
                &mut DiffRenderer { diff, range },
                &value,
                budget,
                zirql::Location::Query(range.clone()),
            )
            .map_err(program::evaluation_error)?;
            let mut text = String::from_utf8(bytes).expect("Markdown is UTF-8");
            // The diff CLI supplies surrounding document framing itself.
            if text.starts_with('\n') {
                text.remove(0);
            }
            if text.ends_with('\n') {
                text.pop();
            }
            text
        }
        _ => return Err(failure("markdown requires changes or report data", range)),
    };
    Ok(text)
}

enum PrintFragment<'a> {
    Literal(&'a str),
    Binding(&'a str),
}

// Diff print escapes only doubled opening braces; closing braces are literal.
struct PrintFragments<'a> {
    rest: &'a str,
}

impl<'a> PrintFragments<'a> {
    fn new(template: &'a str) -> Self {
        Self { rest: template }
    }
}

impl<'a> Iterator for PrintFragments<'a> {
    type Item = Result<PrintFragment<'a>, &'static str>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }
        let Some(open) = self.rest.find('{') else {
            return Some(Ok(PrintFragment::Literal(std::mem::take(&mut self.rest))));
        };
        if open != 0 {
            let text = &self.rest[..open];
            self.rest = &self.rest[open..];
            return Some(Ok(PrintFragment::Literal(text)));
        }
        self.rest = &self.rest[1..];
        if self.rest.starts_with('{') {
            self.rest = &self.rest[1..];
            return Some(Ok(PrintFragment::Literal("{")));
        }
        let Some(close) = self.rest.find('}') else {
            self.rest = "";
            return Some(Err("unterminated interpolation in print"));
        };
        let name = &self.rest[..close];
        self.rest = &self.rest[close + 1..];
        Some(Ok(PrintFragment::Binding(name)))
    }
}

fn interpolate(
    template: &str,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<String, EvaluationError> {
    // Admit the host's token scan; the core admits output copies separately.
    charge(budget, template.len(), range)?;
    let mut output = zirql::expand_fragments(
        PrintFragments::new(template),
        &mut (),
        budget,
        zirql::Location::Query(range.clone()),
        |part, _, budget| {
            let part = part.map_err(|message| zirql::EvaluationError::Host(failure(message, range)))?;
            Ok(match part {
                PrintFragment::Literal(text) => Cow::Borrowed(text),
                PrintFragment::Binding(name) => {
                    let value = bindings.get(name, range.clone())?;
                    match value {
                        DiffProgramOutput::Count(count) => {
                            let bytes = usize::try_from(count.checked_ilog10().unwrap_or(0))
                                .expect("usize represents the number of decimal digits") + 1;
                            budget.charge(bytes, zirql::Location::Query(range.clone()))?;
                            budget.charge_results(bytes, zirql::Location::Query(range.clone()))?;
                            Cow::Owned(count.to_string())
                        }
                        DiffProgramOutput::Strings(values) if values.len() == 1 => {
                            Cow::Borrowed(values[0].as_str())
                        }
                        _ => {
                            return Err(zirql::EvaluationError::Host(failure(
                                format!("string interpolation `{{{name}}}` requires a count or exactly one string"),
                                range,
                            )));
                        }
                    }
                }
            })
        },
    ).map_err(program::evaluation_error)?;
    output
        .len()
        .checked_add(1)
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| failure("print storage limit exceeded", range))?;
    charge(budget, 1, range)?;
    admit(budget, 1, range)?;
    output.push('\n');
    Ok(output)
}

fn validate_program(
    program: &zirql::Program<DiffStage, SetOperator, ()>,
    mut names: BTreeMap<String, Kind>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    let mut shared_names = value_scope(&names);
    for statement in program.statements() {
        let (expression, name) = match &statement.value {
            zirql::Statement::Parameter { .. }
            | zirql::Statement::BindPattern { .. }
            | zirql::Statement::BindFields { .. } => {
                return Err(EvaluationError::new(
                    "pattern bindings are unsupported in diff queries",
                ));
            }
            zirql::Statement::Bind { name, expression } => (expression, Some(name)),
            zirql::Statement::Evaluate { expression, .. } => (expression, None),
        };
        let shared_kind = zirql::native_entry_kind(
            expression,
            &DiffProjectionHost {
                reject_string_unique: true,
            },
            &ValueKind::Native(Kind::Changes),
            &shared_names,
            budget,
        )
        .map_err(core_error)?;
        if let Some(name) = name {
            if matches!(
                name.value.as_str(),
                "input" | "before" | "after" | "before_document" | "after_document"
            ) {
                return Err(failure(
                    format!("binding name `{}` is reserved in diff mode", name.value),
                    &name.range,
                ));
            }
            let kind = match &shared_kind {
                ValueKind::Native(kind) => *kind,
                ValueKind::Constructed(_) => Kind::Constructed,
            };
            shared_names.insert(name.value.clone(), shared_kind);
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
    Ok(match stage {
        DiffStage::Guard(_) => {
            return Err(failure("predicate guard is not an executable stage", range));
        }
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
        DiffStage::Names
        | DiffStage::Attr(_)
        | DiffStage::ResultTypes
        | DiffStage::OperandTypes => {
            require(selection, "projection requires changes or operations")?;
            Kind::Strings
        }
        DiffStage::Noop => input,
        DiffStage::Json => {
            require(input != Kind::Json, "json cannot be applied twice")?;
            Kind::Json
        }
        DiffStage::Markdown => {
            require(
                matches!(
                    input,
                    Kind::Changes | Kind::Strings | Kind::Map | Kind::Constructed
                ),
                "markdown requires changes or report data",
            )?;
            Kind::Text
        }
        DiffStage::Print(template) => {
            for part in PrintFragments::new(template) {
                if let PrintFragment::Binding(name) =
                    part.map_err(|message| failure(message, range))?
                {
                    require(
                        names
                            .get(name)
                            .is_some_and(|kind| matches!(kind, Kind::Count | Kind::Strings)),
                        &format!("binding `{name}` is unavailable or incompatible in diff mode"),
                    )?;
                }
            }
            Kind::Text
        }
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ValueKind {
    Native(Kind),
    Constructed(Arc<Vec<zirql::ValueType>>),
}
struct DiffProjectionHost {
    reject_string_unique: bool,
}
#[allow(non_upper_case_globals)]
const DiffProjectionHost: DiffProjectionHost = DiffProjectionHost {
    reject_string_unique: false,
};

fn value_scope(names: &BTreeMap<String, Kind>) -> BTreeMap<String, ValueKind> {
    names
        .iter()
        .map(|(name, kind)| (name.clone(), ValueKind::Native(*kind)))
        .collect()
}
fn projection_error(error: EvaluationError) -> zirql::QueryError {
    zirql::QueryError::query(error.range.unwrap_or(0..0), error.message)
}
impl ProgramHost<Context<'_, '_>> for DiffProjectionHost {
    type Value = DiffProgramOutput;
    type Stage = DiffStage;
    type SetOperator = SetOperator;
    type Emitter = ();
    type Error = EvaluationError;
    fn copy_value(
        &self,
        value: &Self::Value,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Self::Value, zirql::EvaluationError<EvaluationError>> {
        DiffHost.copy_value(value, context, budget, range)
    }
    fn evaluate_stage(
        &self,
        stage: &DiffStage,
        value: Self::Value,
        context: &mut Context<'_, '_>,
        bindings: &Bindings<Self::Value>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Self::Value, zirql::EvaluationError<EvaluationError>> {
        ProgramHost::evaluate_stage(&DiffHost, stage, value, context, bindings, budget, range)
    }
    fn combine(
        &self,
        operator: &SetOperator,
        left: Self::Value,
        right: Self::Value,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Self::Value, zirql::EvaluationError<EvaluationError>> {
        DiffHost.combine(operator, left, right, context, budget, range)
    }
    fn construct_record(
        &self,
        fields: Vec<(String, Self::Value)>,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Self::Value, zirql::EvaluationError<EvaluationError>> {
        DiffHost.construct_record(fields, context, budget, range)
    }
    fn emit(
        &self,
        emitter: &(),
        value: &Self::Value,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        DiffHost.emit(emitter, value, context, budget, range)
    }
}
impl ProjectionHost<Context<'_, '_>> for DiffProjectionHost {
    type Kind = ValueKind;
    fn is_constructed_kind(&self, kind: &ValueKind) -> bool {
        matches!(kind, ValueKind::Constructed(_))
    }
    fn subject_kind(
        &self,
        input: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        match input {
            ValueKind::Native(Kind::Changes | Kind::Operations(_) | Kind::Strings) => {
                Ok(input.clone())
            }
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "native diff enumeration requires a stream",
            )),
        }
    }
    fn selection_kind(
        &self,
        input: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        match input {
            ValueKind::Native(Kind::Changes | Kind::Operations(_)) => Ok(input.clone()),
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "relation requires changes or projected operations",
            )),
        }
    }
    fn validate_existence(
        &self,
        input: &ValueKind,
        result: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        if input == result {
            Ok(())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "where_exists relation must preserve its native kind and projected diff side",
            ))
        }
    }
    fn fixed_point_kind(
        &self,
        input: &ValueKind,
        result: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        if !matches!(input, ValueKind::Native(Kind::Operations(_))) {
            return Err(zirql::QueryError::query(
                range.clone(),
                "fixpoint requires projected operations",
            ));
        }
        if input != result {
            return Err(zirql::QueryError::query(
                range.clone(),
                "fixpoint body must preserve the projected diff side",
            ));
        }
        Ok(input.clone())
    }
    fn restore_selection(
        &self,
        input: DiffProgramOutput,
        items: Vec<DiffProgramOutput>,
        _: zirql::SelectionShape,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        let (kind, _) = rows(input).map_err(zirql::EvaluationError::Host)?;
        let mut result = Vec::new();
        for item in items {
            let (item_kind, item_rows) = rows(item).map_err(zirql::EvaluationError::Host)?;
            if item_kind != kind {
                return Err(zirql::QueryError::query(
                    range.clone(),
                    "native diff selection changed kind or side",
                )
                .into());
            }
            for row in item_rows {
                validate_row(context.diff, &row).map_err(zirql::EvaluationError::Host)?;
                push(&mut result, row, budget, range).map_err(zirql::EvaluationError::Host)?;
            }
        }
        Ok(from_rows(kind, result))
    }
    fn admit_selection(
        &self,
        value: DiffProgramOutput,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        let (kind, items) = rows(value).map_err(zirql::EvaluationError::Host)?;
        if !matches!(kind, Kind::Changes | Kind::Operations(_)) {
            return Err(zirql::QueryError::query(
                range.clone(),
                "relation requires changes or projected operations",
            )
            .into());
        }
        for item in &items {
            validate_row(context.diff, item).map_err(zirql::EvaluationError::Host)?;
        }
        check(budget, items.len(), range).map_err(zirql::EvaluationError::Host)?;
        Ok(from_rows(kind, items))
    }
    fn set_kind(
        &self,
        left: &ValueKind,
        right: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        if left == right
            && matches!(
                left,
                ValueKind::Native(Kind::Changes | Kind::Operations(_) | Kind::Strings)
            )
        {
            Ok(left.clone())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "set operands require the same native kind and same diff side",
            ))
        }
    }
    fn same_selection(
        &self,
        left: &DiffProgramOutput,
        right: &DiffProgramOutput,
        _: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<bool, zirql::EvaluationError<EvaluationError>> {
        match (left, right) {
            (DiffProgramOutput::Operations(ls, left), DiffProgramOutput::Operations(rs, right))
                if ls == rs =>
            {
                zirql::ordered_native_equal(
                    left,
                    right,
                    budget,
                    zirql::Location::Query(range.clone()),
                    |a, b, _| Ok(a == b),
                )
            }
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "fixpoint body must preserve the projected diff side",
            )
            .into()),
        }
    }
    fn validate_guard_leaf(
        &self,
        payload: &DiffStage,
        input: &ValueKind,
        names: &BTreeMap<String, ValueKind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        let DiffStage::Guard(leaf) = payload else {
            return Err(zirql::QueryError::query(
                range.clone(),
                "expected diff predicate leaf",
            ));
        };
        if !matches!(
            input,
            ValueKind::Native(Kind::Changes | Kind::Operations(_))
        ) {
            return Err(zirql::QueryError::query(
                range.clone(),
                "filter requires a change or operation stream",
            ));
        }
        if matches!(leaf, Leaf::Change(_) | Leaf::Changed(_))
            && *input != ValueKind::Native(Kind::Changes)
        {
            return Err(zirql::QueryError::query(
                range.clone(),
                "change predicates require a change stream",
            ));
        }
        validate_filter(
            &Located::new(zirql::PredicateKind::Leaf(leaf.clone()), range.clone()),
            input.clone(),
            names,
            budget,
        )
    }
    fn value_arguments<'s>(
        &self,
        payload: &'s DiffStage,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeValueArguments<'s, Context<'_, '_>, Self>, zirql::QueryError> {
        let values: Vec<_> = match payload {
            DiffStage::Guard(Leaf::Expression(value)) => vec![&*value.0],
            DiffStage::Guard(Leaf::Dynamic { arguments, .. }) => {
                arguments.iter().map(|value| &*value.0).collect()
            }
            _ => Vec::new(),
        };
        budget.charge_results(values.len(), zirql::Location::Query(range.clone()))?;
        Ok(values
            .into_iter()
            .map(|value| (value, ValueKind::Native(Kind::Changes)))
            .collect())
    }
    fn scoped_value_arguments<'s>(
        &self,
        payload: &'s DiffStage,
        input: &ValueKind,
        _: &BTreeMap<String, ValueKind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::ScopedNativeValueArguments<'s, Context<'_, '_>, Self>, zirql::QueryError>
    {
        Ok(self
            .value_arguments(payload, budget, range)?
            .into_iter()
            .map(|(expression, _)| zirql::ScopedNativeArgument {
                expression,
                relative: Some(input.clone()),
                locals: Vec::new(),
                allow_effects: false,
            })
            .collect())
    }
    fn evaluate_guard_leaf_with_arguments(
        &self,
        payload: &DiffStage,
        input: &DiffProgramOutput,
        bindings: &Bindings<DiffProgramOutput>,
        arguments: &mut dyn zirql::NativeArguments<Context<'_, '_>, Self>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Truth, zirql::EvaluationError<EvaluationError>> {
        let DiffStage::Guard(leaf) = payload else {
            return Err(
                zirql::QueryError::query(range.clone(), "expected diff predicate leaf").into(),
            );
        };
        let result = match leaf {
            Leaf::Expression(value) => {
                let input = copy(input, budget, range).map_err(zirql::EvaluationError::Host)?;
                arguments.evaluate_predicate(&value.0, input, None, budget)?
            }
            Leaf::Dynamic {
                name,
                arguments: values,
            } => {
                let mut strings = Vec::new();
                for value in values {
                    let input = copy(input, budget, range).map_err(zirql::EvaluationError::Host)?;
                    let value =
                        arguments.evaluate_match_value(&value.0, input, bindings, budget)?;
                    let DiffProgramOutput::Constructed(value) = value.value() else {
                        return Err(zirql::QueryError::query(
                            range.clone(),
                            "diff predicate argument needs a string",
                        )
                        .into());
                    };
                    let zirql::ConstructedValue::String(value) = value.value().data() else {
                        return Err(zirql::QueryError::query(
                            range.clone(),
                            "diff predicate argument needs a string",
                        )
                        .into());
                    };
                    program::charge_bytes(budget, value.len(), range)
                        .map_err(zirql::EvaluationError::Host)?;
                    strings.push(value.clone());
                }
                let leaf = resolved_leaf(name, strings).map_err(zirql::EvaluationError::Host)?;
                let (_, items) =
                    rows(copy(input, budget, range).map_err(zirql::EvaluationError::Host)?)
                        .map_err(zirql::EvaluationError::Host)?;
                let [row] = items.as_slice() else {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "predicate needs one native subject",
                    )
                    .into());
                };
                matches_leaf(arguments.parts().0.diff, row, &leaf)
                    .map_err(zirql::EvaluationError::Host)?
            }
            _ => {
                let (_, items) =
                    rows(copy(input, budget, range).map_err(zirql::EvaluationError::Host)?)
                        .map_err(zirql::EvaluationError::Host)?;
                let [row] = items.as_slice() else {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "predicate needs one native subject",
                    )
                    .into());
                };
                validate_row(arguments.parts().0.diff, row)
                    .map_err(zirql::EvaluationError::Host)?;
                matches_leaf(arguments.parts().0.diff, row, leaf)
                    .map_err(zirql::EvaluationError::Host)?
            }
        };
        Ok(if result { Truth::True } else { Truth::False })
    }
    fn ordering_key_kind(
        &self,
        kind: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        if matches!(kind, ValueKind::Native(Kind::Strings | Kind::Count)) {
            Ok(())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "ordering key must produce exactly one scalar",
            ))
        }
    }
    fn ordering_key(
        &self,
        value: DiffProgramOutput,
        _: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeOrderKey, zirql::EvaluationError<EvaluationError>> {
        match value {
            DiffProgramOutput::Count(count) => Ok(zirql::NativeOrderKey::Count(count)),
            DiffProgramOutput::Strings(mut values) if values.len() == 1 => {
                Ok(zirql::NativeOrderKey::Text(values.pop().unwrap()))
            }
            DiffProgramOutput::Constructed(value) => Ok(zirql::NativeOrderKey::Value(
                value
                    .into_value(budget, range)
                    .map_err(zirql::EvaluationError::Host)?,
            )),
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "ordering key must produce exactly one scalar",
            )
            .into()),
        }
    }
    fn mapping_kind(
        &self,
        key: &ValueKind,
        _: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        if *key == ValueKind::Native(Kind::Strings) {
            Ok(ValueKind::Native(Kind::Map))
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "map_by key must produce exactly one string",
            ))
        }
    }
    fn mapping_keys(
        &self,
        value: DiffProgramOutput,
        _: &mut Context<'_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<Option<Vec<String>>, zirql::EvaluationError<EvaluationError>> {
        Ok(if let DiffProgramOutput::Strings(values) = value {
            Some(values)
        } else {
            None
        })
    }
    fn is_map_kind(&self, kind: &ValueKind) -> bool {
        *kind == ValueKind::Native(Kind::Map)
    }
    fn map_sort_kind(
        &self,
        kind: &ValueKind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        if self.is_map_kind(kind) {
            Ok(kind.clone())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "sort_by(value) requires a map",
            ))
        }
    }
    fn map_entries(
        &self,
        value: DiffProgramOutput,
        _: &mut Context<'_, '_>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<(String, DiffProgramOutput)>, zirql::EvaluationError<EvaluationError>> {
        if let DiffProgramOutput::Map(entries) = value {
            Ok(entries)
        } else {
            Err(zirql::QueryError::query(range.clone(), "expected a native diff map").into())
        }
    }
    fn construct_map(
        &self,
        entries: Vec<(String, DiffProgramOutput)>,
        _: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        check(budget, entries.len(), range).map_err(zirql::EvaluationError::Host)?;
        Ok(DiffProgramOutput::Map(entries))
    }
    fn collection_kind(
        &self,
        operation: &Stage<()>,
        input: &ValueKind,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        if let ValueKind::Constructed(types) = input {
            return zirql::collection_value_types(operation, types, budget, range)
                .map(|types| self.constructed_kind(types));
        }
        let ValueKind::Native(kind) = input else {
            unreachable!()
        };
        let selection = matches!(kind, Kind::Changes | Kind::Operations(_));
        let stream = selection || *kind == Kind::Strings;
        match operation {
            Stage::Unique if self.reject_string_unique && !selection => {
                Err(zirql::QueryError::query(
                    range.clone(),
                    "unique requires a change or operation stream",
                ))
            }
            Stage::Tally if *kind != Kind::Strings => Err(zirql::QueryError::query(
                range.clone(),
                "tally requires a value stream",
            )),
            Stage::Count if stream || *kind == Kind::Map => Ok(ValueKind::Native(Kind::Count)),
            Stage::Tally => Ok(ValueKind::Native(Kind::Map)),
            _ if stream || *kind == Kind::Map => Ok(input.clone()),
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "stage requires a stream or map",
            )),
        }
    }
    fn collection_identity(
        &self,
        item: &DiffProgramOutput,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeCollectionKey, zirql::EvaluationError<EvaluationError>> {
        let (kind, rows) = rows(copy(item, budget, range).map_err(zirql::EvaluationError::Host)?)
            .map_err(zirql::EvaluationError::Host)?;
        let [row] = rows.as_slice() else {
            return Err(zirql::QueryError::query(
                range.clone(),
                "identity requires one diff subject",
            )
            .into());
        };
        validate_row(context.diff, row).map_err(zirql::EvaluationError::Host)?;
        Ok(match row {
            Row::String(value) => zirql::NativeCollectionKey::Text(value.clone()),
            Row::Change(id) => zirql::NativeCollectionKey::Native(zirql::NativeIdentity([
                0,
                id.index() as u64,
                0,
                0,
            ])),
            Row::Operation(id) => zirql::NativeCollectionKey::Native(zirql::NativeIdentity([
                if kind == Kind::Operations(DiffSide::Before) {
                    1
                } else {
                    2
                },
                context
                    .diff
                    .document(id.side())
                    .operation_key(
                        context
                            .diff
                            .operation_id(*id)
                            .map_err(|e| zirql::EvaluationError::Host(diff_error(e)))?,
                    )
                    .map_err(|e| zirql::QueryError::query(range.clone(), e.to_string()))?,
                0,
                0,
            ])),
        })
    }
    fn collection_tally_key(
        &self,
        item: DiffProgramOutput,
        _: &mut Context<'_, '_>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<String, zirql::EvaluationError<EvaluationError>> {
        if let DiffProgramOutput::Strings(mut values) = item
            && values.len() == 1
        {
            Ok(values.pop().unwrap())
        } else {
            Err(zirql::QueryError::query(range.clone(), "tally requires strings").into())
        }
    }
    fn restore_collection(
        &self,
        input: DiffProgramOutput,
        result: zirql::NativeValue<DiffProgramOutput>,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        match result {
            zirql::NativeValue::Stream(items) => self.restore_selection(
                input,
                items,
                zirql::SelectionShape::Preserve,
                context,
                budget,
                range,
            ),
            zirql::NativeValue::Count(count) => Ok(DiffProgramOutput::Count(count)),
            zirql::NativeValue::Tally(entries) => Ok(DiffProgramOutput::Map(
                entries
                    .into_iter()
                    .map(|(key, count)| (key, DiffProgramOutput::Count(count)))
                    .collect(),
            )),
        }
    }

    fn kind(
        &self,
        value: &DiffProgramOutput,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        budget.charge(1, zirql::Location::Query(range.clone()))?;
        Ok(match value {
            DiffProgramOutput::Changes(_) => ValueKind::Native(Kind::Changes),
            DiffProgramOutput::Operations(side, _) => ValueKind::Native(Kind::Operations(*side)),
            DiffProgramOutput::Strings(_) => ValueKind::Native(Kind::Strings),
            DiffProgramOutput::Count(_) => ValueKind::Native(Kind::Count),
            DiffProgramOutput::Map(_) => ValueKind::Native(Kind::Map),
            DiffProgramOutput::Json(_, _) => ValueKind::Native(Kind::Json),
            DiffProgramOutput::Text(_) => ValueKind::Native(Kind::Text),
            DiffProgramOutput::Constructed(value) => {
                ValueKind::Constructed(Arc::clone(&value.types))
            }
        })
    }
    fn scalar_types(
        &self,
        kind: &ValueKind,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Option<Vec<zirql::ValueType>>, zirql::QueryError> {
        use zirql::{ArrayType, ValueType as T};
        Ok(match kind {
            ValueKind::Constructed(types) => Some(zirql::copy_value_types(
                types,
                budget,
                zirql::Location::Query(range.clone()),
            )?),
            ValueKind::Native(Kind::Strings) => {
                Some(vec![T::Array(ArrayType::Elements(vec![T::String]))])
            }
            ValueKind::Native(Kind::Count) => Some(vec![T::Integer]),
            _ => None,
        })
    }
    fn constructed_kind(&self, types: Vec<zirql::ValueType>) -> ValueKind {
        ValueKind::Constructed(Arc::new(types))
    }
    fn stage_kind(
        &self,
        stage: &DiffStage,
        input: &ValueKind,
        names: &BTreeMap<String, ValueKind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<ValueKind, zirql::QueryError> {
        if let DiffStage::Root(pipeline) = stage {
            for step in pipeline.stages() {
                if let zirql::Stage::Filter(predicate) = &step.value {
                    validate_filter(predicate, input.clone(), names, budget)?;
                }
            }
        }
        let ValueKind::Native(input_kind) = input else {
            return Err(zirql::QueryError::query(
                range.clone(),
                "diff stage requires its native stream kind",
            ));
        };
        let native_names = names
            .iter()
            .filter_map(|(name, kind)| match kind {
                ValueKind::Native(kind) => Some((name.clone(), *kind)),
                _ => None,
            })
            .collect();
        let result =
            validate_stage(stage, *input_kind, &native_names, range).map_err(projection_error)?;
        Ok(ValueKind::Native(result))
    }
    fn is_pure(&self, stage: &DiffStage) -> bool {
        !matches!(
            stage,
            DiffStage::Json | DiffStage::Markdown | DiffStage::Print(_) | DiffStage::Noop
        )
    }
    fn nested_queries(
        &self,
        stage: &DiffStage,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeOperands<Context<'_, '_>, Self>, zirql::QueryError> {
        let _ = (stage, budget, range);
        Ok(Vec::new())
    }
    fn subjects(
        &self,
        input: DiffProgramOutput,
        context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<DiffProgramOutput>, zirql::EvaluationError<EvaluationError>> {
        let (kind, rows) = rows(input).map_err(zirql::EvaluationError::Host)?;
        let mut values = Vec::new();
        for row in rows {
            validate_row(context.diff, &row).map_err(zirql::EvaluationError::Host)?;
            values
                .push(row_value(&row, kind, budget, range).map_err(zirql::EvaluationError::Host)?);
        }
        Ok(values)
    }
    fn project_value(
        &self,
        input: DiffProgramOutput,
        _context: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::ConstructedValue, zirql::EvaluationError<EvaluationError>> {
        use zirql::ConstructedValue as V;
        match input {
            DiffProgramOutput::Constructed(value) => value
                .into_value(budget, range)
                .map_err(zirql::EvaluationError::Host),
            DiffProgramOutput::Count(count) => {
                zirql::count_value(count, budget, zirql::Location::Query(range.clone()))
                    .map_err(Into::into)
            }
            DiffProgramOutput::Strings(values) => {
                admit(budget, values.len() + 1, range).map_err(zirql::EvaluationError::Host)?;
                Ok(V::Array(values.into_iter().map(V::String).collect()))
            }
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "native diff subjects require an explicit scalar projection",
            )
            .into()),
        }
    }
    fn lift_constructed(
        &self,
        value: zirql::ConstructedValue,
        types: Vec<zirql::ValueType>,
        _: &mut Context<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<DiffProgramOutput, zirql::EvaluationError<EvaluationError>> {
        super::comprehension::ConstructedReport::with_types(value, types, range.clone(), budget)
            .map(DiffProgramOutput::Constructed)
            .map_err(zirql::EvaluationError::Host)
    }
    fn pattern_constructed<'v>(
        &self,
        value: &'v DiffProgramOutput,
    ) -> Option<&'v zirql::ConstructedValue> {
        match value {
            DiffProgramOutput::Constructed(value) => Some(value.value()),
            _ => None,
        }
    }
}
fn row_value(
    row: &Row,
    kind: Kind,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<DiffProgramOutput, EvaluationError> {
    admit(budget, 1, range)?;
    Ok(match row {
        Row::Change(id) => DiffProgramOutput::Changes(vec![*id]),
        Row::Operation(operation) => {
            let Kind::Operations(side) = kind else {
                unreachable!("operation kind")
            };
            DiffProgramOutput::Operations(side, vec![*operation])
        }
        Row::String(value) => {
            program::charge_bytes(budget, value.len(), range)?;
            DiffProgramOutput::Strings(vec![value.clone()])
        }
    })
}
fn validate_filter(
    predicate: &zirql::Predicate<Leaf>,
    input: ValueKind,
    names: &BTreeMap<String, ValueKind>,
    budget: &mut Budget,
) -> Result<(), zirql::QueryError> {
    let mut pending = vec![predicate];
    while let Some(predicate) = pending.pop() {
        budget.charge(1, zirql::Location::Query(predicate.range.clone()))?;
        match &predicate.value {
            zirql::PredicateKind::Leaf(Leaf::Expression(value)) => {
                let zirql::ValueExpressionKind::PredicateScope { predicate, .. } = &value.0.value
                else {
                    unreachable!("bound predicate")
                };
                zirql::native_predicate_types(
                    predicate,
                    &DiffProjectionHost,
                    input.clone(),
                    names.clone(),
                    budget,
                )?;
            }
            zirql::PredicateKind::Leaf(Leaf::Dynamic { name, arguments }) => {
                if matches!(name.as_str(), "change" | "changed")
                    && input != ValueKind::Native(Kind::Changes)
                {
                    return Err(zirql::QueryError::query(
                        predicate.range.clone(),
                        "change predicates require a change stream",
                    ));
                }
                for argument in arguments {
                    let zirql::ValueExpressionKind::SubjectScope {
                        argument: predicate,
                        ..
                    } = &argument.0.value
                    else {
                        unreachable!("bound native argument")
                    };
                    let types = zirql::native_predicate_value_types(
                        predicate,
                        &DiffProjectionHost,
                        input.clone(),
                        names.clone(),
                        budget,
                    )?;
                    if types.iter().any(|ty| *ty != zirql::ValueType::String) {
                        return Err(zirql::QueryError::query(
                            argument.0.range.clone(),
                            "diff predicate argument needs a string",
                        ));
                    }
                }
            }
            zirql::PredicateKind::Not(value) | zirql::PredicateKind::Group(value) => {
                pending.push(value)
            }
            zirql::PredicateKind::And(values) | zirql::PredicateKind::Or(values) => {
                pending.extend(values)
            }
            _ => {}
        }
    }
    Ok(())
}
fn resolve_leaf(
    name: &str,
    arguments: &[SharedValue],
    input: DiffProgramOutput,
    context: &mut Context<'_, '_>,
    bindings: &Bindings<DiffProgramOutput>,
    budget: &mut Budget,
) -> Result<Leaf, EvaluationError> {
    let mut values = Vec::new();
    for argument in arguments {
        let input = copy(&input, budget, &argument.0.range)?;
        let zirql::ValueExpressionKind::SubjectScope {
            argument: predicate,
            ..
        } = &argument.0.value
        else {
            unreachable!("bound native argument")
        };
        let result = zirql::evaluate_native_predicate_value(
            predicate,
            &DiffProjectionHost,
            input,
            context,
            bindings,
            budget,
        )
        .map_err(program::evaluation_error)?;
        let zirql::ConstructedValue::String(value) = result.value.into_data() else {
            return Err(failure(
                "diff predicate argument needs a string",
                &argument.0.range,
            ));
        };
        values.push(value);
    }
    resolved_leaf(name, values)
}
fn resolved_leaf(name: &str, values: Vec<String>) -> Result<Leaf, EvaluationError> {
    let mut values = values.into_iter();
    let value = values.next().expect("domain signature has one argument");
    Ok(match name {
        "change" => Leaf::Change(value.parse().map_err(diff_error)?),
        "changed" => Leaf::Changed(value.parse().map_err(diff_error)?),
        "op" => Leaf::Op(value),
        "dialect" => Leaf::Dialect(value),
        "result_type" => Leaf::ResultType(value),
        "has_attr" => Leaf::HasAttr(value),
        "attr" => Leaf::Attr(
            value,
            values
                .next()
                .expect("attribute signature has two arguments"),
        ),
        _ => unreachable!("registered diff predicate leaf"),
    })
}
fn validate_shared(
    expression: &Expression,
    input: ValueKind,
    names: &BTreeMap<String, ValueKind>,
    host: &DiffProjectionHost,
    budget: &mut Budget,
) -> Result<ValueKind, EvaluationError> {
    zirql::native_expression_kind(expression, host, &input, names, budget).map_err(core_error)
}

pub(super) fn native_predicate(predicate: zirql::Predicate<DiffStage>) -> zirql::Predicate<Leaf> {
    use zirql::PredicateKind as P;
    let range = predicate.range;
    let value = match predicate.value {
        P::Leaf(DiffStage::Guard(leaf)) => P::Leaf(leaf),
        P::Leaf(_) => unreachable!("native predicate leaf"),
        P::Not(p) => P::Not(Box::new(native_predicate(*p))),
        P::Group(p) => P::Group(Box::new(native_predicate(*p))),
        P::And(ps) => P::And(ps.into_iter().map(native_predicate).collect()),
        P::Or(ps) => P::Or(ps.into_iter().map(native_predicate).collect()),
    };
    Located::new(value, range)
}

pub(super) fn guard_predicate(predicate: zirql::Predicate<Leaf>) -> zirql::Predicate<DiffStage> {
    use zirql::PredicateKind as P;
    let range = predicate.range;
    let value = match predicate.value {
        P::Leaf(leaf) => P::Leaf(DiffStage::Guard(leaf)),
        P::Not(p) => P::Not(Box::new(guard_predicate(*p))),
        P::Group(p) => P::Group(Box::new(guard_predicate(*p))),
        P::And(ps) => P::And(ps.into_iter().map(guard_predicate).collect()),
        P::Or(ps) => P::Or(ps.into_iter().map(guard_predicate).collect()),
    };
    Located::new(value, range)
}
