//! Typed, immutable queries over diff records and projected operations.

use std::marker::PhantomData;

use crate::diff::{ChangeField, ChangeId, ChangeKind, Diff, DiffOperation, DiffSide};

use super::{
    EvaluationError, EvaluationLimits, EvaluationOptions, OperationTraversal, Predicate,
    diff_program::{self, DiffProgramOutput, DiffStage, Expression, Leaf},
    evaluation_budget, model,
};

#[derive(Clone, Debug)]
enum Stage {
    ChangeFilter(ChangePredicate),
    Project(DiffSide),
    OperationFilter(Predicate),
    Users(Option<usize>),
    Defs(Option<usize>),
    Parent,
    Children,
    Root(Predicate),
    Subtree,
    Reachable,
    Closure,
    Slice,
    ForwardSlice,
    Fixpoint(Vec<Stage>),
    Unique,
    Reverse,
    Head(usize),
    Tail(usize),
    Names,
    Attr(String),
    ResultTypes,
    OperandTypes,
    Sort,
    Min(bool),
    Max(bool),
    Count,
}

#[derive(Clone, Debug)]
pub struct DiffQueryExpr<T> {
    stages: Vec<Stage>,
    depth: usize,
    invalid: Option<&'static str>,
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
    Operation(Predicate),
    Not(Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
}

#[derive(Clone, Debug)]
pub struct ChangePredicate {
    node: ChangePredicateNode,
    depth: usize,
    invalid: Option<&'static str>,
}
impl ChangePredicate {
    fn new(node: ChangePredicateNode) -> Self {
        Self {
            node,
            depth: 1,
            invalid: None,
        }
    }
    fn combine(self, right: Self, and: bool) -> Self {
        let flat = |node: &ChangePredicateNode| {
            matches!(
                (and, node),
                (true, ChangePredicateNode::And(_)) | (false, ChangePredicateNode::Or(_))
            )
        };
        let depth = (self.depth - usize::from(flat(&self.node)))
            .max(right.depth - usize::from(flat(&right.node)))
            .saturating_add(1);
        let invalid = self
            .invalid
            .or(right.invalid)
            .or((depth >= super::parser::DEFAULT_NESTING_LIMIT)
                .then_some("predicate nesting limit exceeded"));
        if invalid.is_some() {
            return Self {
                node: ChangePredicateNode::And(Vec::new()),
                depth,
                invalid,
            };
        }
        let mut predicates = Vec::new();
        for node in [self.node, right.node] {
            match node {
                ChangePredicateNode::And(nodes) if and => predicates.extend(nodes),
                ChangePredicateNode::Or(nodes) if !and => predicates.extend(nodes),
                node => predicates.push(node),
            }
        }
        Self {
            node: if and {
                ChangePredicateNode::And(predicates)
            } else {
                ChangePredicateNode::Or(predicates)
            },
            depth,
            invalid,
        }
    }
}

pub fn changes() -> ChangeQuery {
    DiffQueryExpr::new()
}
pub fn change_input() -> ChangeQuery {
    DiffQueryExpr::new()
}
pub fn diff_op_input() -> DiffOpQuery {
    DiffQueryExpr::new()
}
pub fn diff_op_input_for_side(side: DiffSide) -> DiffOpQuery {
    DiffQueryExpr::<Vec<ChangeId>>::new().append(Stage::Project(side))
}
pub fn change(kind: ChangeKind) -> ChangePredicate {
    ChangePredicate::new(ChangePredicateNode::Kind(kind))
}
pub fn changed(field: ChangeField) -> ChangePredicate {
    ChangePredicate::new(ChangePredicateNode::Field(field))
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
        ChangePredicate::new(ChangePredicateNode::Operation(self))
    }
}

impl std::ops::BitAnd for ChangePredicate {
    type Output = Self;
    fn bitand(self, right: Self) -> Self {
        self.combine(right, true)
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
        self.combine(right, false)
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
        let depth = self.depth.saturating_add(1);
        let invalid = self
            .invalid
            .or((depth >= super::parser::DEFAULT_NESTING_LIMIT)
                .then_some("predicate nesting limit exceeded"));
        let node = if invalid.is_some() {
            ChangePredicateNode::And(Vec::new())
        } else {
            ChangePredicateNode::Not(Box::new(self.node))
        };
        Self {
            node,
            depth,
            invalid,
        }
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
    pub fn reverse(&self) -> Self {
        self.append(Stage::Reverse)
    }
    pub fn head(&self, count: usize) -> Self {
        self.append(Stage::Head(count))
    }
    pub fn tail(&self, count: usize) -> Self {
        self.append(Stage::Tail(count))
    }
    pub fn names(&self) -> DiffStringQuery {
        self.append(Stage::Names)
    }
    pub fn attr(&self, name: impl Into<String>) -> DiffStringQuery {
        self.append(Stage::Attr(name.into()))
    }
    pub fn result_types(&self) -> DiffStringQuery {
        self.append(Stage::ResultTypes)
    }
    pub fn operand_types(&self) -> DiffStringQuery {
        self.append(Stage::OperandTypes)
    }
    pub fn count(&self) -> DiffCountQuery {
        self.append(Stage::Count)
    }
}

impl DiffOpQuery {
    pub fn filter(&self, predicate: Predicate) -> Self {
        self.append(Stage::OperationFilter(predicate))
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
    pub fn parent(&self) -> Self {
        self.append(Stage::Parent)
    }
    pub fn children(&self) -> Self {
        self.append(Stage::Children)
    }
    pub fn root(&self, predicate: Predicate) -> Self {
        self.append(Stage::Root(predicate))
    }
    pub fn subtree(&self) -> Self {
        self.append(Stage::Subtree)
    }
    pub fn reachable(&self) -> Self {
        self.append(Stage::Reachable)
    }
    pub fn closure(&self) -> Self {
        self.append(Stage::Closure)
    }
    pub fn slice(&self) -> Self {
        self.append(Stage::Slice)
    }
    pub fn backward_slice(&self) -> Self {
        self.slice()
    }
    pub fn forward_slice(&self) -> Self {
        self.append(Stage::ForwardSlice)
    }
    pub fn fixpoint(&self, body: &DiffOpQuery) -> Self {
        let depth = self.depth.max(body.depth.saturating_add(1));
        let invalid = self
            .invalid
            .or(body.invalid)
            .or((depth > super::parser::DEFAULT_NESTING_LIMIT)
                .then_some("query nesting limit exceeded"));
        if invalid.is_some() {
            return Self {
                stages: Vec::new(),
                depth,
                invalid,
                result: PhantomData,
            };
        }
        let mut result = self.append(Stage::Fixpoint(body.stages.clone()));
        result.depth = depth;
        result
    }
    pub fn unique(&self) -> Self {
        self.append(Stage::Unique)
    }
    pub fn reverse(&self) -> Self {
        self.append(Stage::Reverse)
    }
    pub fn head(&self, count: usize) -> Self {
        self.append(Stage::Head(count))
    }
    pub fn tail(&self, count: usize) -> Self {
        self.append(Stage::Tail(count))
    }
    pub fn names(&self) -> DiffStringQuery {
        self.append(Stage::Names)
    }
    pub fn attr(&self, name: impl Into<String>) -> DiffStringQuery {
        self.append(Stage::Attr(name.into()))
    }
    pub fn result_types(&self) -> DiffStringQuery {
        self.append(Stage::ResultTypes)
    }
    pub fn operand_types(&self) -> DiffStringQuery {
        self.append(Stage::OperandTypes)
    }
    pub fn count(&self) -> DiffCountQuery {
        self.append(Stage::Count)
    }
}

impl DiffStringQuery {
    pub fn sort(&self) -> Self {
        self.append(Stage::Sort)
    }
    pub fn min(&self) -> Self {
        self.append(Stage::Min(false))
    }
    pub fn min_all(&self) -> Self {
        self.append(Stage::Min(true))
    }
    pub fn max(&self) -> Self {
        self.append(Stage::Max(false))
    }
    pub fn max_all(&self) -> Self {
        self.append(Stage::Max(true))
    }
    pub fn unique(&self) -> Self {
        self.append(Stage::Unique)
    }
    pub fn count(&self) -> DiffCountQuery {
        self.append(Stage::Count)
    }
    pub fn reverse(&self) -> Self {
        self.append(Stage::Reverse)
    }
    pub fn head(&self, count: usize) -> Self {
        self.append(Stage::Head(count))
    }
    pub fn tail(&self, count: usize) -> Self {
        self.append(Stage::Tail(count))
    }
}

impl<T> DiffQueryExpr<T> {
    fn new() -> Self {
        Self {
            stages: Vec::new(),
            depth: 1,
            invalid: None,
            result: PhantomData,
        }
    }
    fn append<U>(&self, stage: Stage) -> DiffQueryExpr<U> {
        let mut stages = if self.invalid.is_some() {
            Vec::new()
        } else {
            self.stages.clone()
        };
        if self.invalid.is_none() {
            stages.push(stage);
        }
        DiffQueryExpr {
            stages,
            depth: self.depth,
            invalid: self.invalid,
            result: PhantomData,
        }
    }
    pub fn evaluate(&self, diff: &Diff<'_>, limits: EvaluationLimits) -> Result<T, EvaluationError>
    where
        T: DiffQueryResult,
    {
        self.evaluate_with_options(diff, limits, EvaluationOptions::default())
    }

    pub fn evaluate_with_options(
        &self,
        diff: &Diff<'_>,
        limits: EvaluationLimits,
        options: EvaluationOptions,
    ) -> Result<T, EvaluationError>
    where
        T: DiffQueryResult,
    {
        if let Some(error) = self.invalid {
            return Err(EvaluationError::new(error));
        }
        let expression = lower(&self.stages)?;
        let value = diff_program::evaluate_typed(
            &expression,
            diff,
            options,
            &mut evaluation_budget(limits),
        )?;
        let value = match value {
            DiffProgramOutput::Changes(items) => Runtime::Changes(items),
            DiffProgramOutput::Operations(side, items) => Runtime::Operations(side, items),
            DiffProgramOutput::Strings(items) => Runtime::Strings(items),
            DiffProgramOutput::Count(count) => Runtime::Count(count),
            DiffProgramOutput::Map(_)
            | DiffProgramOutput::Json(_, _)
            | DiffProgramOutput::Text(_) => {
                return Err(EvaluationError::new(
                    "typed diff queries cannot return report values",
                ));
            }
        };
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

fn lower(stages: &[Stage]) -> Result<Expression, EvaluationError> {
    use zirql::{Located, Stage as Shared};
    let mut lowered = Vec::new();
    for stage in stages {
        let native = match stage {
            Stage::ChangeFilter(predicate) => {
                diff_program::stream(Shared::Filter(lower_change(predicate)?), 0..0)?
            }
            Stage::OperationFilter(predicate) => diff_program::stream(
                Shared::Filter(lower_operation(&predicate.checked_model()?)),
                0..0,
            )?,
            Stage::Project(side) => DiffStage::Project(*side),
            Stage::Users(index) => DiffStage::Users(*index),
            Stage::Defs(index) => DiffStage::Defs(*index),
            Stage::Parent => DiffStage::Parent,
            Stage::Children => DiffStage::Children,
            Stage::Subtree => DiffStage::Subtree,
            Stage::Root(predicate) => {
                let DiffStage::Stream(filter) = diff_program::stream(
                    Shared::Filter(lower_operation(&predicate.checked_model()?)),
                    0..0,
                )?
                else {
                    unreachable!("stream builds a native pipeline")
                };
                DiffStage::Root(filter)
            }
            Stage::Reachable => DiffStage::Graph(OperationTraversal::Reachable),
            Stage::Closure => DiffStage::Graph(OperationTraversal::Closure),
            Stage::Slice => DiffStage::Graph(OperationTraversal::Slice),
            Stage::ForwardSlice => DiffStage::Graph(OperationTraversal::ForwardSlice),
            Stage::Fixpoint(body) => DiffStage::Fixpoint(diff_program::nested(lower(body)?)),
            Stage::Unique => diff_program::stream(Shared::Unique, 0..0)?,
            Stage::Head(count) => diff_program::stream(Shared::Head(*count), 0..0)?,
            Stage::Count => diff_program::stream(Shared::Count, 0..0)?,
            Stage::Reverse => DiffStage::Reverse,
            Stage::Tail(count) => DiffStage::Tail(*count),
            Stage::Names => DiffStage::Names,
            Stage::Attr(name) => DiffStage::Attr(name.clone()),
            Stage::ResultTypes => DiffStage::ResultTypes,
            Stage::OperandTypes => DiffStage::OperandTypes,
            Stage::Sort => DiffStage::Sort,
            Stage::Min(all) => DiffStage::Extreme {
                maximum: false,
                all: *all,
            },
            Stage::Max(all) => DiffStage::Extreme {
                maximum: true,
                all: *all,
            },
        };
        lowered.push(Located::new(native, 0..0));
    }
    Ok(diff_program::sequence(lowered, 0..0))
}
fn lower_change(predicate: &ChangePredicate) -> Result<zirql::Predicate<Leaf>, EvaluationError> {
    if let Some(error) = predicate.invalid {
        return Err(EvaluationError::new(error));
    }
    lower_change_node(&predicate.node)
}
fn lower_change_node(
    predicate: &ChangePredicateNode,
) -> Result<zirql::Predicate<Leaf>, EvaluationError> {
    use zirql::PredicateKind as P;
    let node = match predicate {
        ChangePredicateNode::Kind(kind) => P::Leaf(Leaf::Change(*kind)),
        ChangePredicateNode::Field(field) => P::Leaf(Leaf::Changed(*field)),
        ChangePredicateNode::Operation(predicate) => {
            return Ok(lower_operation(&predicate.checked_model()?));
        }
        ChangePredicateNode::Not(predicate) => P::Not(Box::new(lower_change_node(predicate)?)),
        ChangePredicateNode::And(predicates) => P::And(
            predicates
                .iter()
                .map(lower_change_node)
                .collect::<Result<_, _>>()?,
        ),
        ChangePredicateNode::Or(predicates) => P::Or(
            predicates
                .iter()
                .map(lower_change_node)
                .collect::<Result<_, _>>()?,
        ),
    };
    Ok(zirql::Located::new(node, 0..0))
}
fn lower_operation(predicate: &model::Predicate) -> zirql::Predicate<Leaf> {
    use zirql::PredicateKind as P;
    let node = match predicate {
        model::Predicate::Bool { value, .. } => P::Leaf(Leaf::Bool(*value)),
        model::Predicate::Op { name, .. } => P::Leaf(Leaf::Op(name.clone())),
        model::Predicate::Dialect { name, .. } => P::Leaf(Leaf::Dialect(name.clone())),
        model::Predicate::ResultType { spelling, .. } => {
            P::Leaf(Leaf::ResultType(spelling.clone()))
        }
        model::Predicate::HasAttr { name, .. } => P::Leaf(Leaf::HasAttr(name.clone())),
        model::Predicate::Attr { name, value, .. } => {
            P::Leaf(Leaf::Attr(name.clone(), value.clone()))
        }
        model::Predicate::Not { predicate, .. } => P::Not(Box::new(lower_operation(predicate))),
        model::Predicate::Group { predicate, .. } => P::Group(Box::new(lower_operation(predicate))),
        model::Predicate::And { predicates, .. } => {
            P::And(predicates.iter().map(lower_operation).collect())
        }
        model::Predicate::Or { predicates, .. } => {
            P::Or(predicates.iter().map(lower_operation).collect())
        }
    };
    zirql::Located::new(node, 0..0)
}
