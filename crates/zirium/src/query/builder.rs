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
        if let Some(message) = self.invalid {
            return Err(EvaluationError::new(message));
        }
        let mut budget = EvaluationState {
            bindings: BTreeMap::new(),
            remaining: limits.max_work,
            max_items: limits.max_items,
            options,
        };
        budget.charge(self.nodes)?;
        let selected = QueryOutput::Operations(budget.collect(document.operations())?);
        evaluate_expression(
            &self.expression,
            &mut DocumentAccess::Shared(document),
            registry,
            selected,
            &mut |_, _| Err(EvaluationError::new("cannot emit in a read-only query")),
            &mut budget,
        )
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
    pub fn query<T: QueryResult>(&self, query: &QueryExpr<T>) -> Result<T, EvaluationError> {
        query.evaluate(
            self,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
    }
    /// Evaluate with the registry describing this document's dialect semantics.
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
    closure => Closure: OpQuery, slice => Slice: OpQuery, reachable => Reachable: OpQuery, names => Names: StringQuery);
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
    document: &mut DocumentAccess<'_>,
    registry: &DialectRegistry,
    current: QueryOutput,
    emit: &mut impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    budget: &mut EvaluationState,
) -> Result<QueryOutput, EvaluationError> {
    Ok(match step {
        StructuredStage::One => {
            let QueryOutput::Values(mut values) = current else {
                return Err(wrong_result());
            };
            if values.len() != 1 {
                return Err(EvaluationError::new(format!(
                    "one requires exactly one string, got {}",
                    values.len()
                )));
            }
            QueryOutput::Native(NativeValue::String(values.pop().unwrap()))
        }
        StructuredStage::WhereExists(query) => {
            let mut selected = Vec::new();
            for operation in take_operations(current, "where_exists")? {
                let result = evaluate_expression(
                    query,
                    document,
                    registry,
                    QueryOutput::Operations(vec![operation]),
                    emit,
                    budget,
                )?;
                if !operations(&result, "where_exists")?.is_empty() {
                    selected.push(operation);
                }
            }
            QueryOutput::Operations(selected)
        }
        StructuredStage::MapBy { key, value } => {
            let mut entries = BTreeMap::new();
            let mut items = 0usize;
            for operation in take_operations(current, "map_by")? {
                let seed = QueryOutput::Operations(vec![operation]);
                let key = String::from_query_output(evaluate_expression(
                    key,
                    document,
                    registry,
                    seed.clone(),
                    emit,
                    budget,
                )?)?;
                if entries.contains_key(&key) {
                    return Err(EvaluationError::new(format!(
                        "map_by encountered duplicate key `{key}`; select unique keys"
                    )));
                }
                let value = evaluate_expression(value, document, registry, seed, emit, budget)?;
                items = items.saturating_add(output_size(&value).saturating_add(1));
                budget.check_items(items)?;
                budget.charge(output_size(&value))?;
                entries.insert(key, value);
            }
            QueryOutput::Native(NativeValue::Map(entries))
        }
        StructuredStage::StringAttr(name) => {
            let mut values = Vec::new();
            for operation in take_operations(current, "string_attr")? {
                let Some(id) = document.attribute_id(operation, name) else {
                    continue;
                };
                let Some(AttributeValue::String(spelling)) = document.attribute_value(id) else {
                    return Err(EvaluationError::new(format!(
                        "attribute `{name}` is not a string"
                    )));
                };
                values.push(decode_mlir_string(spelling).ok_or_else(|| {
                    EvaluationError::new(format!("attribute `{name}` cannot be decoded as UTF-8"))
                })?);
            }
            QueryOutput::Values(values)
        }
        StructuredStage::Attributes(name) => {
            let values = take_operations(current, "attributes")?
                .into_iter()
                .filter_map(|operation| {
                    document
                        .attribute_id(operation, name)
                        .map(|id| (name.clone(), id))
                })
                .collect();
            QueryOutput::Native(NativeValue::Attributes(values))
        }
        StructuredStage::Types { operands } => {
            let mut values = Vec::new();
            for operation in take_operations(current, "types")? {
                if *operands {
                    for &value in document.operands(operation).unwrap_or(&[]) {
                        budget.charge(1)?;
                        if let Some(id) = document.value_type_id(value) {
                            values.push(id);
                            budget.check_items(values.len())?;
                        }
                    }
                } else {
                    let types = document.result_types(operation).unwrap_or(&[]);
                    budget.charge(types.len())?;
                    budget.check_items(values.len().saturating_add(types.len()))?;
                    values.extend_from_slice(types);
                }
            }
            QueryOutput::Native(NativeValue::Types(values))
        }
        StructuredStage::Spellings => QueryOutput::Values(match current {
            QueryOutput::Native(NativeValue::Types(values)) => values
                .into_iter()
                .map(|id| {
                    document
                        .type_spelling(id)
                        .map(str::to_owned)
                        .ok_or_else(wrong_result)
                })
                .collect::<Result<_, _>>()?,
            QueryOutput::Native(NativeValue::Attributes(values)) => values
                .into_iter()
                .map(|(_, id)| {
                    document
                        .attribute_spelling_value(id)
                        .map(str::to_owned)
                        .ok_or_else(wrong_result)
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
