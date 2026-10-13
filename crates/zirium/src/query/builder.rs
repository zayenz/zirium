//! Typed, immutable query expressions. Construction never reads a document.

use super::{
    model::{self, Expression, Stage, StructuredStage},
    *,
};
use crate::source::TextRange;
use std::{
    marker::PhantomData,
    ops::{BitAnd, BitOr, Not},
};

fn range() -> TextRange {
    TextRange::at(0)
}
fn expression(stages: Vec<Stage>) -> Expression {
    Expression {
        first: stages,
        rest: Vec::new(),
        range: range(),
    }
}

/// A reusable query returning `T`. Use [`ops`] or [`input`] to start one.
///
/// Query construction is immutable. Execution happens only at [`Document::query`].
/// Nested expressions are limited to 64 levels, including boolean predicates.
///
/// ```
/// use zirium::query::{ops, op};
/// let consumers = ops().filter(op("arith.addi")).users().unique();
/// let histogram = consumers.names().tally();
/// ```
///
/// Methods follow the result type:
///
/// ```compile_fail
/// use zirium::query::ops;
/// let invalid = ops().names().users(); // Strings have no SSA users.
/// ```
#[derive(Clone, Debug)]
pub struct QueryExpr<T> {
    expression: Expression,
    depth: usize,
    nodes: usize,
    invalid: Option<&'static str>,
    result: PhantomData<fn() -> T>,
}

pub type OpQuery = QueryExpr<Vec<OperationId>>;
pub type StringQuery = QueryExpr<Vec<String>>;
pub type CountQuery = QueryExpr<usize>;
pub type MapQuery<T> = QueryExpr<BTreeMap<String, T>>;
pub type TypeQuery = QueryExpr<Vec<TypeId>>;
pub type AttributeQuery = QueryExpr<Vec<AttributeId>>;

impl<T> QueryExpr<T> {
    /// Erase the static result type for language bindings. The expression stays read-only.
    pub fn erased(&self) -> QueryExpr<QueryOutput> {
        QueryExpr {
            expression: self.expression.clone(),
            depth: self.depth,
            nodes: self.nodes,
            invalid: self.invalid,
            result: PhantomData,
        }
    }
    fn start(stages: Vec<Stage>) -> Self {
        Self {
            nodes: stages.len(),
            expression: expression(stages),
            depth: 1,
            invalid: None,
            result: PhantomData,
        }
    }
    fn append<U>(
        &self,
        stage: Stage,
        depth: usize,
        nodes: usize,
        invalid: Option<&'static str>,
    ) -> QueryExpr<U> {
        let depth = self.depth.max(depth);
        let invalid = self
            .invalid
            .or(invalid)
            .or((depth > parser::DEFAULT_NESTING_LIMIT).then_some("query nesting limit exceeded"));
        // Do not retain an excessively nested tree: even dropping it can overflow the stack.
        let mut expression = if invalid.is_some() {
            expression(Vec::new())
        } else {
            self.expression.clone()
        };
        if invalid.is_none() {
            expression.first.push(stage);
        }
        QueryExpr {
            expression,
            depth,
            nodes: self.nodes.saturating_add(nodes),
            invalid,
            result: PhantomData,
        }
    }
    fn stage<U>(&self, stage: Stage) -> QueryExpr<U> {
        self.append(stage, 1, 1, None)
    }
    fn structured<U>(&self, step: StructuredStage) -> QueryExpr<U> {
        self.stage(Stage::Structured {
            step,
            range: range(),
        })
    }
    fn nested<U, V>(&self, other: &QueryExpr<V>, stage: Stage) -> QueryExpr<U> {
        self.append(
            stage,
            other.depth.saturating_add(1),
            other.nodes.saturating_add(1),
            other.invalid,
        )
    }
    /// Execute with a specific registry and evaluation budget, returning native data.
    ///
    /// The result owns its collection storage; operation, type, and attribute IDs
    /// remain scoped to `document`. Evaluation is read-only and deterministic.
    /// Work and result materialization are bounded by `limits`; exceeding either
    /// bound returns [`EvaluationError`]. Cost follows visited stages and produced
    /// items rather than expression-construction time.
    pub fn evaluate(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        limits: EvaluationLimits,
    ) -> Result<T, EvaluationError>
    where
        T: QueryResult,
    {
        T::from_query_output(self.evaluate_output(document, registry, limits)?)
    }
    /// Execute with explicit semantic options and limits.
    ///
    /// This has the same ownership, error, and cost contract as [`Self::evaluate`].
    pub fn evaluate_with_options(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        options: EvaluationOptions,
        limits: EvaluationLimits,
    ) -> Result<T, EvaluationError>
    where
        T: QueryResult,
    {
        T::from_query_output(
            self.evaluate_output_with_options(document, registry, options, limits)?,
        )
    }
    /// Execute to the shared result representation, primarily for language bindings.
    pub fn evaluate_output(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        limits: EvaluationLimits,
    ) -> Result<QueryOutput, EvaluationError> {
        self.evaluate_output_with_options(document, registry, EvaluationOptions::default(), limits)
    }
    pub fn evaluate_output_with_options(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        options: EvaluationOptions,
        limits: EvaluationLimits,
    ) -> Result<QueryOutput, EvaluationError> {
        let mut budget = self.new_budget(limits);
        self.evaluate_output_with_budget(document, registry, options, &mut budget)
    }
    /// Create this request's allowance, retaining legacy policy for ordinary queries.
    pub fn new_budget(&self, limits: EvaluationLimits) -> zirql::Budget {
        if report::has_constructed(&self.expression) {
            zirql::Budget::for_program(ProgramLimits {
                work: limits.max_work,
                stream_items: limits.max_items,
                ..comprehension::default_limits()
            })
        } else {
            evaluation_budget(limits)
        }
    }
    /// Evaluate and retain the caller's remaining allowance for native conversion.
    pub fn evaluate_output_with_budget(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        options: EvaluationOptions,
        budget: &mut zirql::Budget,
    ) -> Result<QueryOutput, EvaluationError> {
        if let Some(message) = self.invalid {
            return Err(EvaluationError::new(message));
        }
        if report::has_constructed(&self.expression) && budget.remaining_results().is_none() {
            return Err(EvaluationError::new(
                "constructed reports require an aggregate result allowance",
            ));
        }
        charge(budget, self.nodes)?;
        program::validate_expression(
            &self.expression,
            program::Kind::Operations,
            &BTreeMap::new(),
            budget,
        )?;
        let selected = QueryOutput::Operations(collect(budget, document.operations())?);
        let mut access = DocumentAccess::Shared(document);
        let mut emit = |_: &Document, _: QueryOutput, _: &mut zirql::Budget| {
            Err(EvaluationError::new("cannot emit in a read-only query"))
        };
        let expressions = program::expressions(std::iter::once(&self.expression));
        let mut context = EvaluationContext {
            document: &mut access,
            registry,
            expressions: &expressions,
            options,
            output: OutputMode::Native,
            emit: &mut emit,
        };
        evaluate_expression(
            &self.expression,
            &mut context,
            selected,
            &zirql::Bindings::default(),
            budget,
        )
    }
    /// Execute a constructed request with explicit aggregate and stream bounds.
    pub fn evaluate_with_program_limits(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        limits: ProgramLimits,
    ) -> Result<T, EvaluationError>
    where
        T: QueryResult,
    {
        T::from_query_output(self.evaluate_output_with_budget(
            document,
            registry,
            EvaluationOptions::default(),
            &mut zirql::Budget::for_program(limits),
        )?)
    }
}

/// Start with all operations in the document, including inside nested queries.
pub fn ops() -> OpQuery {
    QueryExpr::start(vec![Stage::Input { range: range() }])
}
/// Start with the operations supplied by the caller of a nested query.
/// At top level this is all operations in the document.
pub fn input() -> OpQuery {
    QueryExpr::start(Vec::new())
}

impl Document {
    /// Evaluate a structured query with the baseline dialect registry and default limits.
    /// See the [registry contract](https://github.com/zayenz/zirium/blob/main/docs/custom-formats.md#registry-contract-for-each-rust-stage).
    /// The document is not mutated. Returned IDs remain document-owned and may
    /// become stale after later edits; owned strings and collections are snapshots.
    /// Errors include invalid expression structure and exhausted budgets.
    pub fn query<T: QueryResult>(&self, query: &QueryExpr<T>) -> Result<T, EvaluationError> {
        query.evaluate(
            self,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
    }
    /// Evaluate with the registry describing this document's dialect semantics.
    ///
    /// Ownership, mutation visibility, conversion, and error behavior match
    /// [`Self::query`]. Default evaluation limits apply.
    pub fn query_with_registry<T: QueryResult>(
        &self,
        query: &QueryExpr<T>,
        registry: &DialectRegistry,
    ) -> Result<T, EvaluationError> {
        query.evaluate(self, registry, EvaluationLimits::default())
    }
}

impl OpQuery {
    pub fn filter(&self, predicate: Predicate) -> Self {
        self.append(
            Stage::Filter {
                predicate: predicate.inner,
                range: range(),
            },
            predicate.depth + 1,
            predicate.nodes + 1,
            predicate.invalid,
        )
    }
    pub fn root(&self, predicate: Predicate) -> Self {
        self.append(
            Stage::Root {
                predicate: predicate.inner,
                range: range(),
            },
            predicate.depth + 1,
            predicate.nodes + 1,
            predicate.invalid,
        )
    }
    pub fn defs(&self) -> Self {
        self.stage(Stage::Defs {
            index: None,
            range: range(),
        })
    }
    pub fn defs_at(&self, index: usize) -> Self {
        self.stage(Stage::Defs {
            index: Some(index),
            range: range(),
        })
    }
    pub fn users(&self) -> Self {
        self.stage(Stage::Users {
            index: None,
            range: range(),
        })
    }
    pub fn users_at(&self, index: usize) -> Self {
        self.stage(Stage::Users {
            index: Some(index),
            range: range(),
        })
    }
    /// Keep each input operation for which the relative subquery is nonempty.
    pub fn where_exists(&self, query: &OpQuery) -> Self {
        self.nested(
            query,
            Stage::Structured {
                step: StructuredStage::WhereExists(Box::new(query.expression.clone())),
                range: range(),
            },
        )
    }
    /// Build a map. Each key query must return one string; duplicate keys are errors.
    pub fn map_by<T: QueryResult>(
        &self,
        key: &QueryExpr<String>,
        value: &QueryExpr<T>,
    ) -> MapQuery<T> {
        self.append(
            Stage::Structured {
                step: StructuredStage::MapBy {
                    key: Box::new(key.expression.clone()),
                    value: Box::new(value.expression.clone()),
                },
                range: range(),
            },
            key.depth.max(value.depth) + 1,
            key.nodes.saturating_add(value.nodes).saturating_add(1),
            key.invalid.or(value.invalid),
        )
    }
    pub fn string_attr(&self, name: impl Into<String>) -> StringQuery {
        self.structured(StructuredStage::StringAttr(name.into()))
    }
    /// Project attribute spellings using the textual `attr` stage's rules.
    pub fn attr(&self, name: impl Into<String>) -> StringQuery {
        self.stage(Stage::Attr {
            name: name.into(),
            range: range(),
        })
    }
    pub fn attributes(&self, name: impl Into<String>) -> AttributeQuery {
        self.structured(StructuredStage::Attributes(name.into()))
    }
    pub fn result_types(&self) -> TypeQuery {
        self.structured(StructuredStage::Types { operands: false })
    }
    pub fn operand_types(&self) -> TypeQuery {
        self.structured(StructuredStage::Types { operands: true })
    }
    /// Replace the selection repeatedly until it is unchanged. Duplicates matter.
    pub fn fixpoint(&self, body: &OpQuery) -> Self {
        self.nested(
            body,
            Stage::Fixpoint {
                expression: Box::new(body.expression.clone()),
                range: range(),
            },
        )
    }
}

macro_rules! stages {
    ($ty:ty; $($method:ident => $stage:ident : $out:ty),* $(,)?) => {
        impl $ty { $(pub fn $method(&self) -> $out { self.stage(Stage::$stage { range: range() }) })* }
    };
}
stages!(OpQuery; parent => Parent: OpQuery, children => Children: OpQuery, subtree => Subtree: OpQuery,
    closure => Closure: OpQuery, reachable => Reachable: OpQuery, names => Names: StringQuery);
impl OpQuery {
    /// Follow transitive SSA definitions from the current selection.
    ///
    /// Construction clones the existing expression but does not inspect a document.
    /// Evaluation terminates on cycles, preserves deterministic order, and charges
    /// traversal against [`EvaluationLimits::max_work`]. Returned IDs belong to the
    /// evaluated document and later edits can make them stale. Use
    /// [`Self::reachable`] to include control-flow reachability.
    pub fn slice(&self) -> OpQuery {
        self.stage(Stage::Slice { range: range() })
    }

    /// Explicit alias for [`Self::slice`].
    pub fn backward_slice(&self) -> OpQuery {
        self.slice()
    }

    /// Follow transitive SSA users from the current selection.
    pub fn forward_slice(&self) -> OpQuery {
        self.stage(Stage::ForwardSlice { range: range() })
    }
}
stages!(StringQuery; sort => Sort: StringQuery, min => Min: StringQuery, max => Max: StringQuery,
    min_all => MinAll: StringQuery, max_all => MaxAll: StringQuery, tally => Tally: MapQuery<usize>);

macro_rules! stream {
    ($ty:ty) => {
        stages!($ty; unique => Unique: Self, reverse => Reverse: Self, count => Count: CountQuery);
        impl $ty {
            pub fn head(&self, count: usize) -> Self { self.stage(Stage::Head { count, range: range() }) }
            pub fn tail(&self, count: usize) -> Self { self.stage(Stage::Tail { count, range: range() }) }
        }
    };
}
stream!(OpQuery);
stream!(StringQuery);
stream!(TypeQuery);
stream!(AttributeQuery);
impl<T> MapQuery<T> {
    /// Erase map value types for language bindings.
    #[doc(hidden)]
    pub fn map_values_erased(&self) -> MapQuery<QueryOutput> {
        QueryExpr {
            expression: self.expression.clone(),
            depth: self.depth,
            nodes: self.nodes,
            invalid: self.invalid,
            result: PhantomData,
        }
    }
    pub fn count(&self) -> CountQuery {
        self.stage(Stage::Count { range: range() })
    }
}
impl StringQuery {
    /// Require exactly one string and return it as a scalar.
    pub fn one(&self) -> QueryExpr<String> {
        self.structured(StructuredStage::One)
    }
}
impl TypeQuery {
    pub fn spellings(&self) -> StringQuery {
        self.structured(StructuredStage::Spellings)
    }
}
impl AttributeQuery {
    pub fn spellings(&self) -> StringQuery {
        self.structured(StructuredStage::Spellings)
    }
}

macro_rules! sets {
    ($ty:ty) => {
        impl $ty {
            fn set(&self, right: &Self, operator: model::SetOperator) -> Self {
                // A chain of set operators stays flat, just like the textual language.
                let (mut left, left_depth) = match self.expression.first.as_slice() {
                    [Stage::Group { expression, .. }] if !expression.rest.is_empty() => {
                        ((**expression).clone(), self.depth)
                    }
                    _ => (
                        expression(vec![Stage::Group {
                            expression: Box::new(self.expression.clone()),
                            range: range(),
                        }]),
                        self.depth + 2,
                    ),
                };
                left.rest.push((
                    operator,
                    vec![Stage::Group {
                        expression: Box::new(right.expression.clone()),
                        range: range(),
                    }],
                ));
                QueryExpr::<()>::start(Vec::new()).append(
                    Stage::Group {
                        expression: Box::new(left),
                        range: range(),
                    },
                    left_depth.max(right.depth + 2),
                    self.nodes.saturating_add(right.nodes).saturating_add(3),
                    self.invalid.or(right.invalid),
                )
            }
            pub fn union(&self, right: &Self) -> Self {
                self.set(right, model::SetOperator::Union)
            }
            pub fn intersect(&self, right: &Self) -> Self {
                self.set(right, model::SetOperator::Intersect)
            }
            pub fn difference(&self, right: &Self) -> Self {
                self.set(right, model::SetOperator::Except)
            }
        }
    };
}
sets!(OpQuery);
sets!(StringQuery);

/// A scalar string or count usable as an ordering key.
pub trait OrderKey: QueryResult + private::OrderKey {}
impl OrderKey for String {}
impl OrderKey for usize {}
macro_rules! ordering {
    ($($method:ident => $stage:ident),*) => { impl OpQuery { $(
        pub fn $method<T: OrderKey>(&self, selector: &QueryExpr<T>) -> Self {
            self.nested(selector, Stage::$stage { selector: Box::new(selector.expression.clone()), range: range() })
        }
    )* }};
}
ordering!(sort_by => SortBy, min_by => MinBy, max_by => MaxBy, min_all_by => MinAllBy, max_all_by => MaxAllBy);

/// An operation predicate. Combine with `&`, `|`, and `!`.
#[derive(Clone, Debug)]
pub struct Predicate {
    inner: model::Predicate,
    depth: usize,
    nodes: usize,
    invalid: Option<&'static str>,
}
impl Predicate {
    pub(crate) fn model(&self) -> model::Predicate {
        self.inner.clone()
    }

    pub(super) fn checked_model(&self) -> Result<model::Predicate, EvaluationError> {
        if let Some(error) = self.invalid {
            return Err(EvaluationError::new(error));
        }
        Ok(self.model())
    }

    fn new(inner: model::Predicate) -> Self {
        Self {
            inner,
            depth: 1,
            nodes: 1,
            invalid: None,
        }
    }
    fn combine(self, right: Self, and: bool) -> Self {
        let flatten = |predicate: &model::Predicate| {
            matches!(
                (and, predicate),
                (true, model::Predicate::And { .. }) | (false, model::Predicate::Or { .. })
            )
        };
        let left_depth = self.depth - usize::from(flatten(&self.inner));
        let right_depth = right.depth - usize::from(flatten(&right.inner));
        let depth = left_depth.max(right_depth) + 1;
        let invalid = self
            .invalid
            .or(right.invalid)
            .or((depth >= parser::DEFAULT_NESTING_LIMIT)
                .then_some("predicate nesting limit exceeded"));
        let nodes = self.nodes.saturating_add(right.nodes).saturating_add(1);
        if invalid.is_some() {
            return Self {
                inner: model::Predicate::Bool {
                    value: false,
                    range: range(),
                },
                depth,
                nodes,
                invalid,
            };
        }
        let mut predicates = Vec::new();
        for predicate in [self.inner, right.inner] {
            match predicate {
                model::Predicate::And {
                    predicates: children,
                    ..
                } if and => predicates.extend(children),
                model::Predicate::Or {
                    predicates: children,
                    ..
                } if !and => predicates.extend(children),
                predicate => predicates.push(predicate),
            }
        }
        Self {
            inner: if and {
                model::Predicate::And {
                    predicates,
                    range: range(),
                }
            } else {
                model::Predicate::Or {
                    predicates,
                    range: range(),
                }
            },
            depth,
            nodes,
            invalid,
        }
    }
}
impl BitAnd for Predicate {
    type Output = Self;
    fn bitand(self, right: Self) -> Self {
        self.combine(right, true)
    }
}
impl BitOr for Predicate {
    type Output = Self;
    fn bitor(self, right: Self) -> Self {
        self.combine(right, false)
    }
}
impl Not for Predicate {
    type Output = Self;
    fn not(self) -> Self {
        let invalid = self
            .invalid
            .or((self.depth + 1 >= parser::DEFAULT_NESTING_LIMIT)
                .then_some("predicate nesting limit exceeded"));
        let inner = if invalid.is_some() {
            model::Predicate::Bool {
                value: false,
                range: range(),
            }
        } else {
            model::Predicate::Not {
                predicate: Box::new(self.inner),
                range: range(),
            }
        };
        Self {
            inner,
            depth: self.depth.saturating_add(1),
            nodes: self.nodes.saturating_add(1),
            invalid,
        }
    }
}
pub fn op(name: impl Into<String>) -> Predicate {
    Predicate::new(model::Predicate::Op {
        name: name.into(),
        range: range(),
    })
}
pub fn dialect(name: impl Into<String>) -> Predicate {
    Predicate::new(model::Predicate::Dialect {
        name: name.into(),
        range: range(),
    })
}
pub fn has_attr(name: impl Into<String>) -> Predicate {
    Predicate::new(model::Predicate::HasAttr {
        name: name.into(),
        range: range(),
    })
}
pub fn result_type(spelling: impl Into<String>) -> Predicate {
    Predicate::new(model::Predicate::ResultType {
        spelling: spelling.into(),
        range: range(),
    })
}
pub fn string_attr_eq(name: impl Into<String>, value: impl Into<String>) -> Predicate {
    Predicate::new(model::Predicate::Attr {
        name: name.into(),
        value: value.into(),
        range: range(),
    })
}
pub fn always(value: bool) -> Predicate {
    Predicate::new(model::Predicate::Bool {
        value,
        range: range(),
    })
}

mod private {
    pub trait Sealed {}
    pub trait OrderKey {}
    impl OrderKey for String {}
    impl OrderKey for usize {}
}
/// Native return types supported by structured queries.
pub trait QueryResult: private::Sealed + Sized {
    #[doc(hidden)]
    fn from_query_output(output: QueryOutput) -> Result<Self, EvaluationError>;
}
fn wrong_result() -> EvaluationError {
    EvaluationError::new("unexpected structured query result type")
}
macro_rules! result {
    ($ty:ty, $pat:pat => $value:expr) => {
        impl private::Sealed for $ty {}
        impl QueryResult for $ty {
            fn from_query_output(output: QueryOutput) -> Result<Self, EvaluationError> {
                match output {
                    $pat => Ok($value),
                    _ => Err(wrong_result()),
                }
            }
        }
    };
}
result!(Vec<OperationId>, QueryOutput::Operations(value) => value);
result!(Vec<String>, QueryOutput::Values(value) => value);
result!(usize, QueryOutput::Count(value) => value);
result!(String, QueryOutput::Native(NativeValue::String(value)) => value);
result!(Vec<TypeId>, QueryOutput::Native(NativeValue::Types(value)) => value);
result!(Vec<AttributeId>, QueryOutput::Native(NativeValue::Attributes(value)) => value.into_iter().map(|(_, id)| id).collect());
impl<T: QueryResult> private::Sealed for BTreeMap<String, T> {}
impl<T: QueryResult> QueryResult for BTreeMap<String, T> {
    fn from_query_output(output: QueryOutput) -> Result<Self, EvaluationError> {
        let entries = match output {
            QueryOutput::Native(NativeValue::Map(entries)) => entries,
            // The shared tally stage also serves the textual frontend.
            QueryOutput::Map(entries) => entries
                .into_iter()
                .map(|(key, value)| {
                    let count = value
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(wrong_result)?;
                    Ok((key, QueryOutput::Count(count)))
                })
                .collect::<Result<_, EvaluationError>>()?,
            _ => return Err(wrong_result()),
        };
        entries
            .into_iter()
            .map(|(key, value)| Ok((key, T::from_query_output(value)?)))
            .collect()
    }
}

pub(super) fn evaluate_structured(
    step: &StructuredStage,
    context: &mut EvaluationContext<'_, '_, '_>,
    current: QueryOutput,
    bindings: &zirql::Bindings<QueryOutput>,
    budget: &mut zirql::Budget,
) -> Result<QueryOutput, EvaluationError> {
    Ok(match step {
        StructuredStage::One => {
            let QueryOutput::Values(values) = current else {
                return Err(wrong_result());
            };
            let count = values.len();
            let value = zirql::take_one(values, 0..0).map_err(|_| {
                EvaluationError::new(format!("one requires exactly one string, got {count}"))
            })?;
            QueryOutput::Native(NativeValue::String(value))
        }
        StructuredStage::WhereExists(query) => {
            let mut selected = Vec::new();
            for operation in take_operations(current, "where_exists")? {
                let result = evaluate_expression(
                    query,
                    context,
                    QueryOutput::Operations(vec![operation]),
                    bindings,
                    budget,
                )?;
                if !operations(&result, "where_exists")?.is_empty() {
                    program::charge_results(budget, 1, &(0..0))?;
                    selected.push(operation);
                }
            }
            QueryOutput::Operations(selected)
        }
        StructuredStage::MapBy { key, value } => {
            program::charge_results(budget, 1, &(0..0))?;
            let mut entries = BTreeMap::new();
            let mut items = 0usize;
            for operation in take_operations(current, "map_by")? {
                program::charge_results(budget, 1, &(0..0))?;
                let seed = QueryOutput::Operations(vec![operation]);
                let key = String::from_query_output(evaluate_expression(
                    key,
                    context,
                    program::copy_value(&seed, budget, &(0..0))?,
                    bindings,
                    budget,
                )?)?;
                if entries.contains_key(&key) {
                    return Err(EvaluationError::new(format!(
                        "map_by encountered duplicate key `{key}`; select unique keys"
                    )));
                }
                let value = evaluate_expression(value, context, seed, bindings, budget)?;
                items = items.saturating_add(output_size(&value).saturating_add(1));
                check_items(budget, items)?;
                charge(budget, output_size(&value))?;
                program::charge_results(budget, 1, &(0..0))?;
                entries.insert(key, value);
            }
            QueryOutput::Native(NativeValue::Map(entries))
        }
        StructuredStage::StringAttr(name) => {
            let mut values = Vec::new();
            for operation in take_operations(current, "string_attr")? {
                let Some(id) = context.document.attribute_id(operation, name) else {
                    continue;
                };
                let Some(AttributeValue::String(spelling)) = context.document.attribute_value(id)
                else {
                    return Err(EvaluationError::new(format!(
                        "attribute `{name}` is not a string"
                    )));
                };
                program::charge_results(budget, 1, &(0..0))?;
                program::charge_bytes(budget, spelling.len(), &(0..0))?;
                values.push(decode_mlir_string(spelling).ok_or_else(|| {
                    EvaluationError::new(format!("attribute `{name}` cannot be decoded as UTF-8"))
                })?);
            }
            QueryOutput::Values(values)
        }
        StructuredStage::Attributes(name) => {
            let mut values = Vec::new();
            for operation in take_operations(current, "attributes")? {
                if let Some(id) = context.document.attribute_id(operation, name) {
                    program::charge_results(budget, 1, &(0..0))?;
                    program::charge_bytes(budget, name.len(), &(0..0))?;
                    values.push((name.clone(), id));
                }
            }
            QueryOutput::Native(NativeValue::Attributes(values))
        }
        StructuredStage::Types { operands } => {
            let mut values = Vec::new();
            for operation in take_operations(current, "types")? {
                if *operands {
                    for &value in context.document.operands(operation).unwrap_or(&[]) {
                        charge(budget, 1)?;
                        if let Some(id) = context.document.value_type_id(value) {
                            program::charge_results(budget, 1, &(0..0))?;
                            values.push(id);
                            check_items(budget, values.len())?;
                        }
                    }
                } else {
                    let types = context.document.result_types(operation).unwrap_or(&[]);
                    charge(budget, types.len())?;
                    check_items(budget, values.len().saturating_add(types.len()))?;
                    program::charge_results(budget, types.len(), &(0..0))?;
                    values.extend_from_slice(types);
                }
            }
            QueryOutput::Native(NativeValue::Types(values))
        }
        StructuredStage::Spellings => QueryOutput::Values(match current {
            QueryOutput::Native(NativeValue::Types(values)) => values
                .into_iter()
                .map(|id| {
                    context
                        .document
                        .type_spelling(id)
                        .ok_or_else(wrong_result)
                        .and_then(|value| {
                            program::charge_results(budget, 1, &(0..0))?;
                            program::charge_bytes(budget, value.len(), &(0..0))?;
                            Ok(value.to_owned())
                        })
                })
                .collect::<Result<_, _>>()?,
            QueryOutput::Native(NativeValue::Attributes(values)) => values
                .into_iter()
                .map(|(_, id)| {
                    context
                        .document
                        .attribute_spelling_value(id)
                        .ok_or_else(wrong_result)
                        .and_then(|value| {
                            program::charge_results(budget, 1, &(0..0))?;
                            program::charge_bytes(budget, value.len(), &(0..0))?;
                            Ok(value.to_owned())
                        })
                })
                .collect::<Result<_, _>>()?,
            _ => return Err(wrong_result()),
        }),
    })
}

impl private::Sealed for QueryOutput {}
impl QueryResult for QueryOutput {
    fn from_query_output(output: QueryOutput) -> Result<Self, EvaluationError> {
        Ok(output)
    }
}

/// Constructed report data, before JSON or Python rendering.
pub type ReportQuery = QueryExpr<ConstructedReport>;
/// The shared native operand representation used by report expressions.
pub type ReportOperand = zirql::Expression<parser::Stage, parser::SetOperator, ()>;

/// Immutable shared value expression. Native leaves retain typed query trees.
#[derive(Clone, Debug)]
pub struct ReportExpr {
    value: ParsedComprehension,
}
impl ReportExpr {
    fn node(value: zirql::ValueExpressionKind<ReportOperand>) -> Self {
        Self {
            value: ParsedComprehension::new(zirql::Located::new(value, 0..0)),
        }
    }
    fn into_expression(mut self) -> zirql::ValueExpression<ReportOperand> {
        if let Some(expression) = std::sync::Arc::get_mut(&mut self.value.expression) {
            return std::mem::replace(
                expression,
                zirql::Located::new(
                    zirql::ValueExpressionKind::Literal(zirql::ConstructedValue::Integer(0)),
                    0..0,
                ),
            );
        }
        zirql::Located::new(
            zirql::ValueExpressionKind::NativeQuery(Box::new(report::operand(
                &super::builder::expression(vec![Stage::Comprehension {
                    value: self.value,
                    range: range(),
                }]),
            ))),
            0..0,
        )
    }
    pub fn integer(value: i64) -> Self {
        Self::node(zirql::ValueExpressionKind::Literal(
            zirql::ConstructedValue::Integer(value),
        ))
    }
    pub fn boolean(value: bool) -> Self {
        Self::node(zirql::ValueExpressionKind::Literal(
            zirql::ConstructedValue::Boolean(value),
        ))
    }
    pub fn string(value: impl Into<String>) -> Self {
        Self::node(zirql::ValueExpressionKind::Literal(
            zirql::ConstructedValue::String(value.into()),
        ))
    }
    /// Finite floating literal construction; invalid decimals are rejected before storage.
    pub fn float(value: f64) -> Result<Self, EvaluationError> {
        if !value.is_finite() {
            return Err(EvaluationError::new("nonfinite float literal"));
        }
        Ok(Self::node(zirql::ValueExpressionKind::Literal(
            zirql::ConstructedValue::Float(value),
        )))
    }
    pub fn name(name: impl Into<String>) -> Self {
        Self::node(zirql::ValueExpressionKind::Name(name.into()))
    }
    pub fn array(values: impl IntoIterator<Item = Self>) -> Self {
        Self::node(zirql::ValueExpressionKind::Array(
            values.into_iter().map(Self::into_expression).collect(),
        ))
    }
    pub fn record<S: Into<String>>(fields: impl IntoIterator<Item = (S, Self)>) -> Self {
        Self::node(zirql::ValueExpressionKind::Record(
            fields
                .into_iter()
                .map(|(name, value)| {
                    (
                        zirql::Located::new(name.into(), 0..0),
                        value.into_expression(),
                    )
                })
                .collect(),
        ))
    }
    pub fn field(self, name: impl Into<String>) -> Self {
        Self::node(zirql::ValueExpressionKind::Field {
            value: Box::new(self.into_expression()),
            field: zirql::Located::new(name.into(), 0..0),
        })
    }
    pub fn at(self, index: Self) -> Self {
        Self::node(zirql::ValueExpressionKind::ArrayAccess {
            array: Box::new(self.into_expression()),
            index: Box::new(index.into_expression()),
        })
    }
    pub fn unary(self, operator: zirql::ValueUnaryOperator) -> Self {
        Self::node(zirql::ValueExpressionKind::Unary {
            operator,
            operand: Box::new(self.into_expression()),
        })
    }
    pub fn binary(self, operator: zirql::ValueBinaryOperator, right: Self) -> Self {
        Self::node(zirql::ValueExpressionKind::Binary {
            operator,
            left: Box::new(self.into_expression()),
            right: Box::new(right.into_expression()),
        })
    }
    pub fn one(self) -> Self {
        Self::node(zirql::ValueExpressionKind::One(Box::new(
            self.into_expression(),
        )))
    }
    pub fn reduce(self, operator: zirql::ValueReducer) -> Self {
        Self::node(zirql::ValueExpressionKind::Reduce {
            operator,
            array: Box::new(self.into_expression()),
        })
    }
    pub fn length(self) -> Self {
        self.reduce(zirql::ValueReducer::Length)
    }
    pub fn sum(self) -> Self {
        self.reduce(zirql::ValueReducer::Sum)
    }
    pub fn forall(self) -> Self {
        self.reduce(zirql::ValueReducer::Forall)
    }
    pub fn exists(self) -> Self {
        self.reduce(zirql::ValueReducer::Exists)
    }
    pub fn require(condition: Self, value: Self, message: impl Into<String>) -> Self {
        Self::node(zirql::ValueExpressionKind::Require {
            condition: Box::new(condition.into_expression()),
            value: Box::new(value.into_expression()),
            message: zirql::Located::new(message.into(), 0..0),
        })
    }
    pub fn comprehension(head: Self, clauses: impl IntoIterator<Item = ReportGenerator>) -> Self {
        Self::node(zirql::ValueExpressionKind::Comprehension {
            constructor: zirql::ValueConstructor::Array,
            clauses: clauses.into_iter().map(|clause| clause.clause).collect(),
            head: Box::new(head.into_expression()),
        })
    }
    /// Evaluate immutable local bindings in order, then return one constructed value.
    pub fn scope<S: Into<String>>(
        bindings: impl IntoIterator<Item = (S, Self)>,
        value: Self,
    ) -> Self {
        Self::comprehension(
            value,
            bindings
                .into_iter()
                .map(|(name, value)| ReportGenerator::bind(name, value)),
        )
        .one()
    }
}
/// A shared generator clause, with immutable binding and optional known-Boolean filter.
pub struct ReportGenerator {
    clause: zirql::Located<zirql::ValueGenerator<ReportOperand>>,
}
impl ReportGenerator {
    pub fn in_values(
        name: impl Into<String>,
        source: ReportExpr,
        filter: Option<ReportExpr>,
    ) -> Self {
        Self {
            clause: zirql::Located::new(
                zirql::ValueGenerator::In {
                    binders: vec![zirql::Located::new(Some(name.into()), 0..0)],
                    source: source.into_expression(),
                    filter: filter.map(ReportExpr::into_expression),
                },
                0..0,
            ),
        }
    }
    pub fn bind(name: impl Into<String>, value: ReportExpr) -> Self {
        Self {
            clause: zirql::Located::new(
                zirql::ValueGenerator::Bind {
                    binder: zirql::Located::new(Some(name.into()), 0..0),
                    value: value.into_expression(),
                    filter: None,
                },
                0..0,
            ),
        }
    }
}
/// Refer to a saved native operation selection or the local projection subject.
pub fn bound_ops(name: impl Into<String>) -> OpQuery {
    QueryExpr::start(vec![Stage::Binding {
        name: name.into(),
        range: range(),
    }])
}
impl OpQuery {
    pub fn project<S: Into<String>>(
        &self,
        subject: impl Into<String>,
        fields: impl IntoIterator<Item = (S, ReportExpr)>,
    ) -> ReportQuery {
        self.stage(Stage::Projection {
            value: report::ParsedProjection::new(zirql::Projection {
                subject: zirql::Located::new(subject.into(), 0..0),
                fields: fields
                    .into_iter()
                    .map(|(name, value)| {
                        (
                            zirql::Located::new(name.into(), 0..0),
                            value.into_expression(),
                        )
                    })
                    .collect(),
            }),
            range: range(),
        })
    }
}
/// Construct a typed read-only report without requesting an emitter.
pub fn report(value: ReportExpr) -> ReportQuery {
    QueryExpr::start(vec![Stage::Comprehension {
        value: value.value,
        range: range(),
    }])
}
macro_rules! report_query_conversion {
    ($result:ty) => {
        impl From<QueryExpr<$result>> for ReportExpr {
            fn from(query: QueryExpr<$result>) -> Self {
                Self::node(zirql::ValueExpressionKind::NativeQuery(Box::new(
                    report::operand(&query.expression),
                )))
            }
        }
    };
}
report_query_conversion!(Vec<String>);
report_query_conversion!(String);
report_query_conversion!(usize);
report_query_conversion!(ConstructedReport);
impl From<i64> for ReportExpr {
    fn from(value: i64) -> Self {
        Self::integer(value)
    }
}
impl From<bool> for ReportExpr {
    fn from(value: bool) -> Self {
        Self::boolean(value)
    }
}
impl From<String> for ReportExpr {
    fn from(value: String) -> Self {
        Self::string(value)
    }
}
impl From<&str> for ReportExpr {
    fn from(value: &str) -> Self {
        Self::string(value)
    }
}
result!(ConstructedReport, QueryOutput::Native(NativeValue::Constructed(value))=>value);

/// Compare an operation name with a checked shared string expression.
pub fn op_expr(name: ReportExpr) -> Predicate {
    Predicate::new(model::Predicate::OpValue {
        name: name.value,
        range: range(),
    })
}
