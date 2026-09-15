//! Typed, immutable queries over diff records and projected operations.

use std::{collections::HashSet, marker::PhantomData};

use crate::{
    diff::{ChangeField, ChangeId, ChangeKind, Diff, DiffOperation, DiffSide},
    semantic::{OperationId, UseSite, ValueId, ValueReference},
};

use super::{EvaluationError, EvaluationLimits, Predicate, model};

#[derive(Clone, Debug)]
enum Stage {
    ChangeFilter(ChangePredicate),
    Project(DiffSide),
    OperationFilter(model::Predicate),
    Users(Option<usize>),
    Defs(Option<usize>),
    Unique,
    Names,
    Count,
}

#[derive(Clone, Debug)]
pub struct DiffQueryExpr<T> {
    stages: Vec<Stage>,
    result: PhantomData<fn() -> T>,
}

pub type ChangeQuery = DiffQueryExpr<Vec<ChangeId>>;
pub type DiffOpQuery = DiffQueryExpr<DiffOperationSelection>;
pub type DiffStringQuery = DiffQueryExpr<Vec<String>>;
pub type DiffCountQuery = DiffQueryExpr<usize>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffOperationSelection {
    side: DiffSide,
    operations: Vec<DiffOperation>,
}

impl DiffOperationSelection {
    pub fn side(&self) -> DiffSide {
        self.side
    }
    pub fn operations(&self) -> &[DiffOperation] {
        &self.operations
    }
    pub fn len(&self) -> usize {
        self.operations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

#[derive(Clone, Debug)]
enum ChangePredicateNode {
    Kind(ChangeKind),
    Field(ChangeField),
    Operation(model::Predicate),
    Not(Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
}

#[derive(Clone, Debug)]
pub struct ChangePredicate(ChangePredicateNode);

pub fn changes() -> ChangeQuery {
    DiffQueryExpr::new()
}
pub fn change_input() -> ChangeQuery {
    DiffQueryExpr::new()
}
pub fn diff_op_input(side: DiffSide) -> DiffOpQuery {
    DiffQueryExpr::<Vec<ChangeId>>::new().append(Stage::Project(side))
}
pub fn change(kind: ChangeKind) -> ChangePredicate {
    ChangePredicate(ChangePredicateNode::Kind(kind))
}
pub fn changed(field: ChangeField) -> ChangePredicate {
    ChangePredicate(ChangePredicateNode::Field(field))
}

pub trait IntoChangePredicate {
    fn into_change_predicate(self) -> ChangePredicate;
}
impl IntoChangePredicate for ChangePredicate {
    fn into_change_predicate(self) -> ChangePredicate {
        self
    }
}
impl IntoChangePredicate for Predicate {
    fn into_change_predicate(self) -> ChangePredicate {
        ChangePredicate(ChangePredicateNode::Operation(self.model()))
    }
}

impl std::ops::BitAnd for ChangePredicate {
    type Output = Self;
    fn bitand(self, right: Self) -> Self {
        Self(ChangePredicateNode::And(vec![self.0, right.0]))
    }
}
impl std::ops::BitAnd<Predicate> for ChangePredicate {
    type Output = Self;
    fn bitand(self, right: Predicate) -> Self {
        self & right.into_change_predicate()
    }
}
impl std::ops::BitOr for ChangePredicate {
    type Output = Self;
    fn bitor(self, right: Self) -> Self {
        Self(ChangePredicateNode::Or(vec![self.0, right.0]))
    }
}
impl std::ops::BitOr<Predicate> for ChangePredicate {
    type Output = Self;
    fn bitor(self, right: Predicate) -> Self {
        self | right.into_change_predicate()
    }
}
impl std::ops::Not for ChangePredicate {
    type Output = Self;
    fn not(self) -> Self {
        Self(ChangePredicateNode::Not(Box::new(self.0)))
    }
}

impl ChangeQuery {
    pub fn filter(&self, predicate: impl IntoChangePredicate) -> Self {
        self.append(Stage::ChangeFilter(predicate.into_change_predicate()))
    }
    pub fn before(&self) -> DiffOpQuery {
        self.append(Stage::Project(DiffSide::Before))
    }
    pub fn after(&self) -> DiffOpQuery {
        self.append(Stage::Project(DiffSide::After))
    }
    pub fn unique(&self) -> Self {
        self.append(Stage::Unique)
    }
    pub fn names(&self) -> DiffStringQuery {
        self.append(Stage::Names)
    }
    pub fn count(&self) -> DiffCountQuery {
        self.append(Stage::Count)
    }
}

impl DiffOpQuery {
    pub fn filter(&self, predicate: Predicate) -> Self {
        self.append(Stage::OperationFilter(predicate.model()))
    }
    pub fn users(&self) -> Self {
        self.append(Stage::Users(None))
    }
    pub fn users_at(&self, index: usize) -> Self {
        self.append(Stage::Users(Some(index)))
    }
    pub fn defs(&self) -> Self {
        self.append(Stage::Defs(None))
    }
    pub fn defs_at(&self, index: usize) -> Self {
        self.append(Stage::Defs(Some(index)))
    }
    pub fn unique(&self) -> Self {
        self.append(Stage::Unique)
    }
    pub fn names(&self) -> DiffStringQuery {
        self.append(Stage::Names)
    }
    pub fn count(&self) -> DiffCountQuery {
        self.append(Stage::Count)
    }
}

impl DiffStringQuery {
    pub fn unique(&self) -> Self {
        self.append(Stage::Unique)
    }
    pub fn count(&self) -> DiffCountQuery {
        self.append(Stage::Count)
    }
}

impl<T> DiffQueryExpr<T> {
    fn new() -> Self {
        Self {
            stages: Vec::new(),
            result: PhantomData,
        }
    }
    fn append<U>(&self, stage: Stage) -> DiffQueryExpr<U> {
        let mut stages = self.stages.clone();
        stages.push(stage);
        DiffQueryExpr {
            stages,
            result: PhantomData,
        }
    }
    pub fn evaluate(&self, diff: &Diff<'_>, limits: EvaluationLimits) -> Result<T, EvaluationError>
    where
        T: DiffQueryResult,
    {
        if self.stages.len() > limits.max_work {
            return Err(EvaluationError::new("query work limit exceeded"));
        }
        let mut work = self.stages.len();
        let mut value = Runtime::Changes(diff.change_ids().collect());
        for stage in &self.stages {
            value = evaluate_stage(diff, value, stage, &mut work, limits)?;
        }
        T::from_runtime(value)
    }
}

impl Diff<'_> {
    pub fn query<T: DiffQueryResult>(
        &self,
        query: &DiffQueryExpr<T>,
    ) -> Result<T, EvaluationError> {
        query.evaluate(self, EvaluationLimits::default())
    }
}

#[doc(hidden)]
pub enum Runtime {
    Changes(Vec<ChangeId>),
    Operations(DiffSide, Vec<DiffOperation>),
    Strings(Vec<String>),
    Count(usize),
}

pub trait DiffQueryResult: Sized {
    fn from_runtime(value: Runtime) -> Result<Self, EvaluationError>;
}
impl DiffQueryResult for Vec<ChangeId> {
    fn from_runtime(value: Runtime) -> Result<Self, EvaluationError> {
        match value {
            Runtime::Changes(v) => Ok(v),
            _ => Err(EvaluationError::new("expected changes")),
        }
    }
}
impl DiffQueryResult for DiffOperationSelection {
    fn from_runtime(value: Runtime) -> Result<Self, EvaluationError> {
        match value {
            Runtime::Operations(side, operations) => Ok(Self { side, operations }),
            _ => Err(EvaluationError::new("expected operations")),
        }
    }
}
impl DiffQueryResult for Vec<String> {
    fn from_runtime(value: Runtime) -> Result<Self, EvaluationError> {
        match value {
            Runtime::Strings(v) => Ok(v),
            _ => Err(EvaluationError::new("expected strings")),
        }
    }
}
impl DiffQueryResult for usize {
    fn from_runtime(value: Runtime) -> Result<Self, EvaluationError> {
        match value {
            Runtime::Count(v) => Ok(v),
            _ => Err(EvaluationError::new("expected count")),
        }
    }
}

fn evaluate_stage(
    diff: &Diff<'_>,
    value: Runtime,
    stage: &Stage,
    work: &mut usize,
    limits: EvaluationLimits,
) -> Result<Runtime, EvaluationError> {
    match stage {
        Stage::ChangeFilter(predicate) => {
            let Runtime::Changes(items) = value else {
                return Err(EvaluationError::new("change predicate requires changes"));
            };
            let mut result = Vec::new();
            for id in items {
                charge(work, 1, limits)?;
                if matches_change(diff, id, &predicate.0)? {
                    result.push(id);
                }
            }
            Ok(Runtime::Changes(result))
        }
        Stage::Project(side) => {
            let Runtime::Changes(items) = value else {
                return Err(EvaluationError::new("side projection requires changes"));
            };
            let operations = items
                .into_iter()
                .filter_map(|id| {
                    match side {
                        DiffSide::Before => diff.before_operation(id),
                        DiffSide::After => diff.after_operation(id),
                    }
                    .ok()
                    .flatten()
                })
                .collect();
            Ok(Runtime::Operations(*side, operations))
        }
        Stage::OperationFilter(predicate) => {
            let Runtime::Operations(side, items) = value else {
                return Err(EvaluationError::new(
                    "operation filter requires a projected side",
                ));
            };
            let result = items
                .into_iter()
                .filter(|item| {
                    matches_operation(
                        predicate,
                        diff.document(side),
                        diff.operation_id(*item).ok(),
                    )
                    .unwrap_or(false)
                })
                .collect();
            Ok(Runtime::Operations(side, result))
        }
        Stage::Users(index) => navigate(diff, value, work, limits, |document, operation| {
            let count = document.result_types(operation).map_or(0, <[_]>::len);
            (0..count)
                .filter(|result| index.is_none_or(|wanted| wanted == *result))
                .flat_map(|result| {
                    document.uses(ValueId::OperationResult {
                        operation,
                        result: result as u32,
                    })
                })
                .map(|site| match site {
                    UseSite::Operand { operation, .. }
                    | UseSite::SuccessorArgument { operation, .. } => operation,
                })
                .collect()
        }),
        Stage::Defs(index) => navigate(diff, value, work, limits, |document, operation| {
            document
                .operands(operation)
                .unwrap_or(&[])
                .iter()
                .enumerate()
                .filter(|(i, _)| index.is_none_or(|wanted| wanted == *i))
                .filter_map(|(_, value)| match value {
                    ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                        Some(*operation)
                    }
                    _ => None,
                })
                .collect()
        }),
        Stage::Unique => match value {
            Runtime::Changes(v) => Ok(Runtime::Changes(dedup(v))),
            Runtime::Operations(s, v) => Ok(Runtime::Operations(s, dedup(v))),
            Runtime::Strings(v) => Ok(Runtime::Strings(dedup(v))),
            Runtime::Count(_) => Err(EvaluationError::new("unique requires a stream")),
        },
        Stage::Names => match value {
            Runtime::Changes(v) => Ok(Runtime::Strings(
                v.into_iter()
                    .filter_map(|id| {
                        let (side, op) = diff.representative_id(id).ok()?;
                        diff.document(side).operation_name(op).map(str::to_owned)
                    })
                    .collect(),
            )),
            Runtime::Operations(side, v) => Ok(Runtime::Strings(
                v.into_iter()
                    .filter_map(|item| {
                        diff.document(side)
                            .operation_name(diff.operation_id(item).ok()?)
                            .map(str::to_owned)
                    })
                    .collect(),
            )),
            _ => Err(EvaluationError::new("names requires changes or operations")),
        },
        Stage::Count => match value {
            Runtime::Changes(v) => Ok(Runtime::Count(v.len())),
            Runtime::Operations(_, v) => Ok(Runtime::Count(v.len())),
            Runtime::Strings(v) => Ok(Runtime::Count(v.len())),
            Runtime::Count(_) => Err(EvaluationError::new("count requires a stream")),
        },
    }
}

fn navigate(
    diff: &Diff<'_>,
    value: Runtime,
    work: &mut usize,
    limits: EvaluationLimits,
    step: impl Fn(&crate::semantic::Document, OperationId) -> Vec<OperationId>,
) -> Result<Runtime, EvaluationError> {
    let Runtime::Operations(side, items) = value else {
        return Err(EvaluationError::new(
            "navigation requires projected operations",
        ));
    };
    let mut result = Vec::new();
    for item in items {
        let ids = step(
            diff.document(side),
            diff.operation_id(item)
                .map_err(|e| EvaluationError::new(e.to_string()))?,
        );
        charge(work, ids.len(), limits)?;
        for operation in ids {
            result.push(
                diff.scoped_operation(side, operation)
                    .map_err(|e| EvaluationError::new(e.to_string()))?,
            );
        }
    }
    if result.len() > limits.max_items {
        return Err(EvaluationError::new("query stream size limit exceeded"));
    }
    Ok(Runtime::Operations(side, result))
}

fn matches_change(
    diff: &Diff<'_>,
    id: ChangeId,
    predicate: &ChangePredicateNode,
) -> Result<bool, EvaluationError> {
    let record = diff
        .change(id)
        .map_err(|e| EvaluationError::new(e.to_string()))?;
    Ok(match predicate {
        ChangePredicateNode::Kind(kind) => {
            if *kind == ChangeKind::Moved {
                record.moved()
            } else {
                record.kind() == *kind
            }
        }
        ChangePredicateNode::Field(field) => record.fields().contains(field),
        ChangePredicateNode::Operation(predicate) => {
            let (side, op) = diff
                .representative_id(id)
                .map_err(|e| EvaluationError::new(e.to_string()))?;
            matches_operation(predicate, diff.document(side), Some(op))?
        }
        ChangePredicateNode::Not(p) => !matches_change(diff, id, p)?,
        ChangePredicateNode::And(ps) => ps
            .iter()
            .all(|p| matches_change(diff, id, p).unwrap_or(false)),
        ChangePredicateNode::Or(ps) => ps
            .iter()
            .any(|p| matches_change(diff, id, p).unwrap_or(false)),
    })
}

fn matches_operation(
    predicate: &model::Predicate,
    document: &crate::semantic::Document,
    operation: Option<OperationId>,
) -> Result<bool, EvaluationError> {
    let Some(operation) = operation else {
        return Ok(false);
    };
    Ok(match predicate {
        model::Predicate::Bool { value, .. } => *value,
        model::Predicate::Op { name, .. } => document.operation_name(operation) == Some(name),
        model::Predicate::Dialect { name, .. } => document
            .operation_name(operation)
            .and_then(|value| value.split_once('.'))
            .is_some_and(|(dialect, _)| dialect == name),
        model::Predicate::ResultType { spelling, .. } => document
            .result_types(operation)
            .unwrap_or(&[])
            .iter()
            .any(|id| document.type_spelling(*id) == Some(spelling)),
        model::Predicate::HasAttr { name, .. } => document.attribute_id(operation, name).is_some(),
        model::Predicate::Attr { name, value, .. } => {
            document
                .attribute_id(operation, name)
                .and_then(|id| document.attribute_value(id))
                .and_then(crate::semantic::AttributeValue::decoded_string)
                .as_deref()
                == Some(value)
        }
        model::Predicate::Not { predicate, .. } => {
            !matches_operation(predicate, document, Some(operation))?
        }
        model::Predicate::And { predicates, .. } => predicates
            .iter()
            .all(|p| matches_operation(p, document, Some(operation)).unwrap_or(false)),
        model::Predicate::Or { predicates, .. } => predicates
            .iter()
            .any(|p| matches_operation(p, document, Some(operation)).unwrap_or(false)),
        model::Predicate::Group { predicate, .. } => {
            matches_operation(predicate, document, Some(operation))?
        }
    })
}

fn charge(
    work: &mut usize,
    amount: usize,
    limits: EvaluationLimits,
) -> Result<(), EvaluationError> {
    *work = work.saturating_add(amount);
    if *work > limits.max_work {
        Err(EvaluationError::new("query work limit exceeded"))
    } else {
        Ok(())
    }
}
fn dedup<T: Clone + Eq + std::hash::Hash>(items: Vec<T>) -> Vec<T> {
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|item| seen.insert(item.clone()))
        .collect()
}
