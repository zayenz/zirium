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
    functions: Vec<FunctionDefinition>,
    function_limits: FunctionLimits,
    parameters: Vec<ParameterDeclaration>,
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

fn merge_parameters(
    left: &[ParameterDeclaration],
    right: &[ParameterDeclaration],
) -> Vec<ParameterDeclaration> {
    let mut parameters = left.to_vec();
    parameters.extend(
        right
            .iter()
            .filter(|declaration| !left.contains(declaration))
            .cloned(),
    );
    parameters
}

impl<T> QueryExpr<T> {
    /// Attach function declarations to this request, retaining conflicts for preflight.
    pub fn with_functions(mut self, functions: Vec<FunctionDefinition>) -> Self {
        self.functions = functions::merge(&self.functions, &functions);
        self
    }
    /// Bound active non-tail calls independently of expression nesting.
    pub fn with_function_limits(mut self, limits: FunctionLimits) -> Self {
        self.function_limits = limits;
        self
    }
    /// Attach immutable required declarations to this root request.
    /// Composition shares identical declarations and retains conflicting declarations
    /// for admission to refuse; existing evaluation methods supply no caller inputs.
    pub fn with_parameters(&self, declarations: Vec<ParameterDeclaration>) -> Self {
        Self {
            functions: self.functions.clone(),
            function_limits: self.function_limits,
            parameters: merge_parameters(&self.parameters, &declarations),
            expression: self.expression.clone(),
            depth: self.depth,
            nodes: self.nodes,
            invalid: self.invalid,
            result: PhantomData,
        }
    }
    pub fn parameters(&self) -> &[ParameterDeclaration] {
        &self.parameters
    }

    /// Evaluate exact typed scalar caller inputs to the query's native result type.
    pub fn evaluate_with_parameters(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        parameters: &[(String, ConstructedValue)],
        limits: EvaluationLimits,
    ) -> Result<T, EvaluationError>
    where
        T: QueryResult,
    {
        let mut budget = self.new_budget(limits);
        T::from_query_output(self.evaluate_output_with_parameters_and_budget(
            document,
            registry,
            EvaluationOptions::default(),
            ParameterInputs::Typed(parameters),
            &mut budget,
        )?)
    }

    /// Erase the static result type for language bindings. The expression stays read-only.
    pub fn erased(&self) -> QueryExpr<QueryOutput> {
        QueryExpr {
            functions: self.functions.clone(),
            function_limits: self.function_limits,
            parameters: self.parameters.clone(),
            expression: self.expression.clone(),
            depth: self.depth,
            nodes: self.nodes,
            invalid: self.invalid,
            result: PhantomData,
        }
    }
    fn start(stages: Vec<Stage>) -> Self {
        Self {
            functions: Vec::new(),
            function_limits: functions::DEFAULT_LIMITS,
            parameters: Vec::new(),
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
            functions: self.functions.clone(),
            function_limits: self.function_limits,
            parameters: self.parameters.clone(),
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
        .with_parameters(other.parameters.clone())
        .with_functions(other.functions.clone())
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
        if !self.functions.is_empty()
            || !self.parameters.is_empty()
            || report::has_constructed(&self.expression)
        {
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
        self.evaluate_output_with_parameters_and_budget(
            document,
            registry,
            options,
            ParameterInputs::Typed(&[]),
            budget,
        )
    }

    /// Admit, preflight and execute with the caller's allowance, retaining it for conversion.
    pub fn evaluate_output_with_parameters_and_budget(
        &self,
        document: &Document,
        registry: &DialectRegistry,
        options: EvaluationOptions,
        parameters: ParameterInputs<'_>,
        budget: &mut Budget,
    ) -> Result<QueryOutput, EvaluationError> {
        if let Some(message) = self.invalid {
            return Err(EvaluationError::new(message));
        }
        if (!self.functions.is_empty()
            || !self.parameters.is_empty()
            || report::has_constructed(&self.expression))
            && budget.remaining_results().is_none()
        {
            return Err(EvaluationError::new(
                "constructed reports require an aggregate result allowance",
            ));
        }
        let admitted =
            program::admit_parameters(&self.parameters, parameters, &["document"], 0..0, budget)?;
        let bindings = zirql::Bindings::new(admitted).map_err(core_error)?;
        let names = program::parameter_names(&self.parameters, budget)?;
        charge(budget, self.nodes)?;
        let function_host = report::ReportHost {
            functions: &self.functions,
        };
        let mut functions = if self.functions.is_empty() {
            None
        } else {
            Some(functions::Session::new(
                &self.functions,
                &function_host,
                self.function_limits,
                budget,
            )?)
        };
        if let Some(session) = &mut functions {
            session.prepare_expression(
                &self.expression,
                program::Kind::Operations,
                &names,
                &function_host,
                budget,
            )?;
        } else {
            program::validate_expression(
                &self.expression,
                program::Kind::Operations,
                &names,
                budget,
            )?;
        }

        let selected = QueryOutput::Operations(collect(budget, document.operations())?);
        let mut access = DocumentAccess::Shared(document);
        let mut emit = |_: &Document, _: QueryOutput, _: &mut zirql::Budget| {
            Err(EvaluationError::new("cannot emit in a read-only query"))
        };
        let function_roots = functions::expression_roots(&self.functions);
        let expressions =
            program::expressions(std::iter::once(&self.expression).chain(function_roots));
        let mut context = EvaluationContext {
            functions: functions.as_ref(),
            current_stage: 0,
            document: &mut access,
            registry,
            expressions: &expressions,
            parameters: Vec::new(),
            options,
            output: OutputMode::Native,
            emit: &mut emit,
        };
        let value =
            evaluate_expression(&self.expression, &mut context, selected, &bindings, budget)?;
        functions::owned(value, budget, &(0..0))
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

/// Direct native edges supported by bounded operation paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathEdge {
    Users,
}

fn path_target(predicate: model::Predicate) -> zirql::Predicate<model::Predicate> {
    use model::Predicate as P;
    let location = program::range(predicate.range());
    let value = match predicate {
        P::Not { predicate, .. } => zirql::PredicateKind::Not(Box::new(path_target(*predicate))),
        P::Group { predicate, .. } => {
            zirql::PredicateKind::Group(Box::new(path_target(*predicate)))
        }
        P::And { predicates, .. } => {
            zirql::PredicateKind::And(predicates.into_iter().map(path_target).collect())
        }
        P::Or { predicates, .. } => {
            zirql::PredicateKind::Or(predicates.into_iter().map(path_target).collect())
        }
        predicate => zirql::PredicateKind::Leaf(predicate),
    };
    zirql::Located::new(value, location)
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
        let functions = predicate.functions.clone();
        let parameters = predicate.parameters.clone();
        self.append(
            Stage::Filter {
                predicate: predicate.inner,
                range: range(),
            },
            predicate.depth + 1,
            predicate.nodes + 1,
            predicate.invalid,
        )
        .with_parameters(parameters)
        .with_functions(functions)
    }
    pub fn root(&self, predicate: Predicate) -> Self {
        let functions = predicate.functions.clone();
        let parameters = predicate.parameters.clone();
        self.append(
            Stage::Root {
                predicate: predicate.inner,
                range: range(),
            },
            predicate.depth + 1,
            predicate.nodes + 1,
            predicate.invalid,
        )
        .with_parameters(parameters)
        .with_functions(functions)
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
    /// Retain one shortest native witness for each source/sink pair within the edge bound.
    pub fn paths(&self, edge: PathEdge, to: Predicate, max_depth: usize) -> CaptureQuery {
        let functions = functions::merge(&self.functions, &to.functions);
        let parameters = merge_parameters(&self.parameters, &to.parameters);
        let edge = match edge {
            PathEdge::Users => Stage::Users {
                index: None,
                range: range(),
            },
        };
        self.append(
            Stage::Paths {
                value: capture::ParsedPaths::new(zirql::NativePaths {
                    edge: zirql::Located::new(edge, 0..0),
                    target: path_target(to.inner),
                    max_depth: zirql::Located::new(max_depth, 0..0),
                    range: 0..0,
                }),
                range: range(),
            },
            to.depth + 1,
            to.nodes + 1,
            to.invalid,
        )
        .with_parameters(parameters)
        .with_functions(functions)
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
        .with_parameters(merge_parameters(&key.parameters, &value.parameters))
        .with_functions(functions::merge(&key.functions, &value.functions))
    }
    pub fn string_attr(&self, name: impl Into<String>) -> StringQuery {
        self.structured(StructuredStage::StringAttr(name.into()))
    }
    /// Project attribute spellings using the textual `attr` stage's rules.
    pub fn attr_expr(&self, name: ReportExpr) -> StringQuery {
        self.stage(Stage::Computed {
            value: model::ComputedStage::Attr(name.value),
            range: range(),
        })
        .with_parameters(name.parameters)
        .with_functions(name.functions)
    }
    pub fn defs_at_expr(&self, index: ReportExpr) -> Self {
        self.stage(Stage::Computed {
            value: model::ComputedStage::Defs(index.value),
            range: range(),
        })
        .with_parameters(index.parameters)
        .with_functions(index.functions)
    }
    pub fn users_at_expr(&self, index: ReportExpr) -> Self {
        self.stage(Stage::Computed {
            value: model::ComputedStage::Users(index.value),
            range: range(),
        })
        .with_parameters(index.parameters)
        .with_functions(index.functions)
    }
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
            pub fn head_expr(&self, count: ReportExpr) -> Self { self.stage(Stage::Computed {value:model::ComputedStage::Head(count.value),range:range()}).with_parameters(count.parameters).with_functions(count.functions) }
            pub fn tail_expr(&self, count: ReportExpr) -> Self { self.stage(Stage::Computed {value:model::ComputedStage::Tail(count.value),range:range()}).with_parameters(count.parameters).with_functions(count.functions) }
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
            functions: self.functions.clone(),
            function_limits: self.function_limits,
            parameters: self.parameters.clone(),
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
                QueryExpr::<()>::start(Vec::new())
                    .append(
                        Stage::Group {
                            expression: Box::new(left),
                            range: range(),
                        },
                        left_depth.max(right.depth + 2),
                        self.nodes.saturating_add(right.nodes).saturating_add(3),
                        self.invalid.or(right.invalid),
                    )
                    .with_parameters(merge_parameters(&self.parameters, &right.parameters))
                    .with_functions(functions::merge(&self.functions, &right.functions))
                    .with_function_limits(self.function_limits)
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
    functions: Vec<FunctionDefinition>,
    parameters: Vec<ParameterDeclaration>,
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
        if !self.parameters.is_empty() {
            return Err(EvaluationError::new(
                "parameterized predicate needs a program scope",
            ));
        }
        Ok(self.model())
    }

    fn new(inner: model::Predicate) -> Self {
        Self {
            functions: Vec::new(),
            parameters: Vec::new(),
            inner,
            depth: 1,
            nodes: 1,
            invalid: None,
        }
    }
    fn combine(self, right: Self, and: bool) -> Self {
        let functions = functions::merge(&self.functions, &right.functions);
        let parameters = merge_parameters(&self.parameters, &right.parameters);
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
                functions,
                parameters,
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
            functions,
            parameters,
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
            functions: self.functions,
            parameters: self.parameters,
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
result!(Vec<OperationId>, QueryOutput::Operations(value) | QueryOutput::Native(NativeValue::OperationSequence(value)) => value);
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
    functions: Vec<FunctionDefinition>,
    parameters: Vec<ParameterDeclaration>,
    value: ParsedComprehension,
}
impl ReportExpr {
    fn with_metadata(
        mut self,
        parameters: Vec<ParameterDeclaration>,
        functions: Vec<FunctionDefinition>,
    ) -> Self {
        self.parameters = parameters;
        self.functions = functions;
        self
    }

    fn node(value: zirql::ValueExpressionKind<ReportOperand>) -> Self {
        Self {
            functions: Vec::new(),
            parameters: Vec::new(),
            value: ParsedComprehension::new(zirql::Located::new(value, 0..0)),
        }
    }
    pub(super) fn into_expression(mut self) -> zirql::ValueExpression<ReportOperand> {
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
                &super::builder::expression(vec![Stage::FunctionValue {
                    value: self.value,
                    range: range(),
                }]),
            ))),
            0..0,
        )
    }
    /// Explicit absence in a read-only report.
    pub fn null() -> Self {
        Self::node(zirql::ValueExpressionKind::Literal(
            zirql::ConstructedValue::Absent,
        ))
    }
    pub fn occurs(self) -> Self {
        let functions = self.functions.clone();
        let parameters = self.parameters.clone();
        Self::node(zirql::ValueExpressionKind::Occurs(Box::new(
            self.into_expression(),
        )))
        .with_metadata(parameters, functions)
    }
    pub fn absent(self) -> Self {
        let functions = self.functions.clone();
        let parameters = self.parameters.clone();
        Self::node(zirql::ValueExpressionKind::Absent(Box::new(
            self.into_expression(),
        )))
        .with_metadata(parameters, functions)
    }
    pub fn lookup(self, key: Self) -> Self {
        let functions = functions::merge(&self.functions, &key.functions);
        let parameters = merge_parameters(&self.parameters, &key.parameters);
        Self::node(zirql::ValueExpressionKind::Lookup {
            value: Box::new(self.into_expression()),
            key: Box::new(key.into_expression()),
        })
        .with_metadata(parameters, functions)
    }
    pub fn has_key(self, key: Self) -> Self {
        let functions = functions::merge(&self.functions, &key.functions);
        let parameters = merge_parameters(&self.parameters, &key.parameters);
        Self::node(zirql::ValueExpressionKind::HasKey {
            value: Box::new(self.into_expression()),
            key: Box::new(key.into_expression()),
        })
        .with_metadata(parameters, functions)
    }
    pub fn default(self, fallback: Self) -> Self {
        let functions = functions::merge(&self.functions, &fallback.functions);
        let parameters = merge_parameters(&self.parameters, &fallback.parameters);
        Self::node(zirql::ValueExpressionKind::Default {
            value: Box::new(self.into_expression()),
            fallback: Box::new(fallback.into_expression()),
        })
        .with_metadata(parameters, functions)
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
        let mut functions = Vec::new();
        let mut parameters = Vec::new();
        let values = values
            .into_iter()
            .map(|value| {
                functions = functions::merge(&functions, &value.functions);
                parameters = merge_parameters(&parameters, &value.parameters);
                value.into_expression()
            })
            .collect();
        Self::node(zirql::ValueExpressionKind::Array(values)).with_metadata(parameters, functions)
    }
    pub fn record<S: Into<String>>(fields: impl IntoIterator<Item = (S, Self)>) -> Self {
        let mut functions = Vec::new();
        let mut parameters = Vec::new();
        let fields = fields
            .into_iter()
            .map(|(name, value)| {
                functions = functions::merge(&functions, &value.functions);
                parameters = merge_parameters(&parameters, &value.parameters);
                (
                    zirql::Located::new(name.into(), 0..0),
                    value.into_expression(),
                )
            })
            .collect();
        Self::node(zirql::ValueExpressionKind::Record(fields)).with_metadata(parameters, functions)
    }
    pub fn field(self, name: impl Into<String>) -> Self {
        let functions = self.functions.clone();
        let parameters = self.parameters.clone();
        Self::node(zirql::ValueExpressionKind::Field {
            value: Box::new(self.into_expression()),
            field: zirql::Located::new(name.into(), 0..0),
        })
        .with_metadata(parameters, functions)
    }
    pub fn at(self, index: Self) -> Self {
        let functions = functions::merge(&self.functions, &index.functions);
        let parameters = merge_parameters(&self.parameters, &index.parameters);
        Self::node(zirql::ValueExpressionKind::ArrayAccess {
            array: Box::new(self.into_expression()),
            index: Box::new(index.into_expression()),
        })
        .with_metadata(parameters, functions)
    }
    pub fn unary(self, operator: zirql::ValueUnaryOperator) -> Self {
        let functions = self.functions.clone();
        let parameters = self.parameters.clone();
        Self::node(zirql::ValueExpressionKind::Unary {
            operator,
            operand: Box::new(self.into_expression()),
        })
        .with_metadata(parameters, functions)
    }
    pub fn binary(self, operator: zirql::ValueBinaryOperator, right: Self) -> Self {
        let functions = functions::merge(&self.functions, &right.functions);
        let parameters = merge_parameters(&self.parameters, &right.parameters);
        Self::node(zirql::ValueExpressionKind::Binary {
            operator,
            left: Box::new(self.into_expression()),
            right: Box::new(right.into_expression()),
        })
        .with_metadata(parameters, functions)
    }
    /// Match an exact case-sensitive substring. An empty fragment matches.
    pub fn contains(self, fragment: Self) -> Self {
        self.binary(zirql::ValueBinaryOperator::Contains, fragment)
    }
    /// Match an exact case-sensitive prefix. An empty prefix matches.
    pub fn starts_with(self, prefix: Self) -> Self {
        self.binary(zirql::ValueBinaryOperator::StartsWith, prefix)
    }
    /// Match an exact case-sensitive suffix. An empty suffix matches.
    pub fn ends_with(self, suffix: Self) -> Self {
        self.binary(zirql::ValueBinaryOperator::EndsWith, suffix)
    }
    pub fn one(self) -> Self {
        let functions = self.functions.clone();
        let parameters = self.parameters.clone();
        Self::node(zirql::ValueExpressionKind::One(Box::new(
            self.into_expression(),
        )))
        .with_metadata(parameters, functions)
    }
    pub fn reduce(self, operator: zirql::ValueReducer) -> Self {
        let functions = self.functions.clone();
        let parameters = self.parameters.clone();
        Self::node(zirql::ValueExpressionKind::Reduce {
            operator,
            array: Box::new(self.into_expression()),
        })
        .with_metadata(parameters, functions)
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
    /// Group positional records by a static field, preserving first-seen keys
    /// and every row's order. Evaluation requires homogeneous supported keys.
    pub fn group_by(self, field: impl Into<String>) -> Self {
        let functions = self.functions.clone();
        let parameters = self.parameters.clone();
        Self::node(zirql::ValueExpressionKind::GroupBy {
            rows: Box::new(self.into_expression()),
            field: zirql::Located::new(field.into(), 0..0),
        })
        .with_metadata(parameters, functions)
    }
    pub fn require(condition: Self, value: Self, message: impl Into<String>) -> Self {
        let functions = functions::merge(&condition.functions, &value.functions);
        let parameters = merge_parameters(&condition.parameters, &value.parameters);
        Self::node(zirql::ValueExpressionKind::Require {
            condition: Box::new(condition.into_expression()),
            value: Box::new(value.into_expression()),
            message: zirql::Located::new(message.into(), 0..0),
        })
        .with_metadata(parameters, functions)
    }
    pub fn comprehension(head: Self, clauses: impl IntoIterator<Item = ReportGenerator>) -> Self {
        let mut functions = head.functions.clone();
        let mut parameters = head.parameters.clone();
        let clauses = clauses
            .into_iter()
            .map(|clause| {
                functions = functions::merge(&functions, &clause.functions);
                parameters = merge_parameters(&parameters, &clause.parameters);
                clause.clause
            })
            .collect();
        Self::node(zirql::ValueExpressionKind::Comprehension {
            constructor: zirql::ValueConstructor::Array,
            clauses,
            head: Box::new(head.into_expression()),
        })
        .with_metadata(parameters, functions)
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
    functions: Vec<FunctionDefinition>,
    parameters: Vec<ParameterDeclaration>,
    clause: zirql::Located<zirql::ValueGenerator<ReportOperand>>,
}
impl ReportGenerator {
    pub fn in_values(
        name: impl Into<String>,
        source: ReportExpr,
        filter: Option<ReportExpr>,
    ) -> Self {
        let functions = functions::merge(
            &source.functions,
            &filter
                .as_ref()
                .map(|value| value.functions.clone())
                .unwrap_or_default(),
        );
        let parameters = merge_parameters(
            &source.parameters,
            &filter
                .as_ref()
                .map(|value| value.parameters.clone())
                .unwrap_or_default(),
        );
        Self {
            functions,
            parameters,
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
            functions: value.functions.clone(),
            parameters: value.parameters.clone(),
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
fn project_rows<T, S: Into<String>>(
    query: &QueryExpr<T>,
    subject: impl Into<String>,
    fields: impl IntoIterator<Item = (S, ReportExpr)>,
) -> ReportQuery {
    let mut functions = Vec::new();
    let mut parameters = Vec::new();
    let fields = fields
        .into_iter()
        .map(|(name, value)| {
            functions = functions::merge(&functions, &value.functions);
            parameters = merge_parameters(&parameters, &value.parameters);
            (
                zirql::Located::new(name.into(), 0..0),
                value.into_expression(),
            )
        })
        .collect();
    query
        .stage(Stage::Projection {
            value: report::ParsedProjection::new(zirql::Projection {
                subject: zirql::Located::new(subject.into(), 0..0),
                fields,
            }),
            range: range(),
        })
        .with_parameters(parameters)
        .with_functions(functions)
}
impl OpQuery {
    pub fn project<S: Into<String>>(
        &self,
        subject: impl Into<String>,
        fields: impl IntoIterator<Item = (S, ReportExpr)>,
    ) -> ReportQuery {
        project_rows(self, subject, fields)
    }
}
/// Construct a typed read-only report without requesting an emitter.
pub fn report(value: ReportExpr) -> ReportQuery {
    QueryExpr::start(vec![Stage::Comprehension {
        value: value.value,
        range: range(),
    }])
    .with_parameters(value.parameters)
    .with_functions(value.functions)
}
macro_rules! report_query_conversion {
    ($result:ty) => {
        impl From<QueryExpr<$result>> for ReportExpr {
            fn from(query: QueryExpr<$result>) -> Self {
                Self::node(zirql::ValueExpressionKind::NativeQuery(Box::new(
                    report::operand(&query.expression),
                )))
                .with_metadata(query.parameters, query.functions)
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
    let functions = name.functions;
    let parameters = name.parameters;
    let mut predicate = Predicate::new(model::Predicate::OpValue {
        name: name.value,
        range: range(),
    });
    predicate.parameters = parameters;
    predicate.functions = functions;
    predicate
}
/// A query returning one exact document-owned SSA value.
pub type SsaQuery = QueryExpr<ValueId>;
/// A query returning ordered native capture records, with an empty-result schema.
pub type CaptureQuery = QueryExpr<CaptureMatches>;
result!(ValueId, QueryOutput::Native(NativeValue::Ssa(value)) => value);
result!(CaptureMatches, QueryOutput::Native(NativeValue::Captures(value)) => value);

impl OpQuery {
    /// Read a required zero-based operand from exactly one selected operation.
    pub fn operand(&self, index: usize) -> SsaQuery {
        let mut query = QueryExpr::start(vec![Stage::Operand {
            operation: capture::ParsedOperand::new(report::operand(&self.expression)),
            index,
            range: range(),
        }])
        .with_parameters(self.parameters.clone())
        .with_functions(self.functions.clone());
        query.depth = self.depth.saturating_add(1);
        query.nodes = self.nodes.saturating_add(1);
        query.invalid = self
            .invalid
            .or((query.depth > parser::DEFAULT_NESTING_LIMIT)
                .then_some("query nesting limit exceeded"));
        query
    }
}
/// Read a previously captured SSA value in a later clause.
pub fn bound_ssa(name: impl Into<String>) -> SsaQuery {
    QueryExpr::start(vec![Stage::Binding {
        name: name.into(),
        range: range(),
    }])
}
/// Select a native operation field from one named capture record.
pub fn capture_ops(subject: impl Into<String>, field: impl Into<String>) -> OpQuery {
    QueryExpr::start(vec![Stage::Native {
        value: capture::ParsedOperand::field(
            subject.into(),
            field.into(),
            capture::NativeKind::Operations,
        ),
        range: range(),
    }])
}
/// Select an exact SSA field from one named capture record.
pub fn capture_ssa(subject: impl Into<String>, field: impl Into<String>) -> SsaQuery {
    QueryExpr::start(vec![Stage::Native {
        value: capture::ParsedOperand::field(
            subject.into(),
            field.into(),
            capture::NativeKind::Ssa,
        ),
        range: range(),
    }])
}

/// Immutable Boolean guard over native SSA queries. Use `&`, `|` and `!` to compose.
#[derive(Clone, Debug)]
pub struct MatchPredicate {
    inner: zirql::Predicate<model::Predicate>,
    functions: Vec<FunctionDefinition>,
    parameters: Vec<ParameterDeclaration>,
    depth: usize,
    nodes: usize,
    invalid: Option<&'static str>,
}
/// Compare full checked ValueId identities, including result/argument position.
pub fn same_value(left: &SsaQuery, right: &SsaQuery) -> MatchPredicate {
    MatchPredicate {
        functions: functions::merge(&left.functions, &right.functions),
        inner: zirql::Located::new(
            zirql::PredicateKind::Leaf(model::Predicate::SameValue {
                left: capture::ParsedOperand::new(report::operand(&left.expression)),
                right: capture::ParsedOperand::new(report::operand(&right.expression)),
                range: range(),
            }),
            0..0,
        ),
        parameters: merge_parameters(&left.parameters, &right.parameters),
        depth: left.depth.max(right.depth).saturating_add(1),
        nodes: left.nodes.saturating_add(right.nodes).saturating_add(1),
        invalid: left.invalid.or(right.invalid),
    }
}
impl MatchPredicate {
    fn combine(self, other: Self, conjunction: bool) -> Self {
        let depth = self.depth.max(other.depth).saturating_add(1);
        let invalid = self
            .invalid
            .or(other.invalid)
            .or((depth > parser::DEFAULT_NESTING_LIMIT).then_some("query nesting limit exceeded"));
        let parameters = merge_parameters(&self.parameters, &other.parameters);
        let functions = functions::merge(&self.functions, &other.functions);
        let nodes = self.nodes.saturating_add(other.nodes).saturating_add(1);
        let inner = if invalid.is_some() {
            zirql::PredicateKind::Leaf(model::Predicate::Bool {
                value: false,
                range: range(),
            })
        } else if conjunction {
            zirql::PredicateKind::And(vec![self.inner, other.inner])
        } else {
            zirql::PredicateKind::Or(vec![self.inner, other.inner])
        };
        Self {
            inner: zirql::Located::new(inner, 0..0),
            functions,
            parameters,
            depth,
            nodes,
            invalid,
        }
    }
}
impl BitAnd for MatchPredicate {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        self.combine(other, true)
    }
}
impl BitOr for MatchPredicate {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.combine(other, false)
    }
}
impl Not for MatchPredicate {
    type Output = Self;
    fn not(self) -> Self {
        let depth = self.depth.saturating_add(1);
        let invalid = self
            .invalid
            .or((depth > parser::DEFAULT_NESTING_LIMIT).then_some("query nesting limit exceeded"));
        let inner = if invalid.is_some() {
            zirql::PredicateKind::Leaf(model::Predicate::Bool {
                value: false,
                range: range(),
            })
        } else {
            zirql::PredicateKind::Not(Box::new(self.inner))
        };
        Self {
            inner: zirql::Located::new(inner, 0..0),
            functions: self.functions,
            parameters: self.parameters,
            depth,
            nodes: self.nodes.saturating_add(1),
            invalid,
        }
    }
}

/// Build required native captures in written clause order. Construction reads no document.
#[derive(Debug)]
pub struct MatchBuilder {
    clauses: Vec<zirql::Located<zirql::MatchClause<capture::ParsedOperand, model::Predicate>>>,
    functions: Vec<FunctionDefinition>,
    parameters: Vec<ParameterDeclaration>,
    depth: usize,
    nodes: usize,
    invalid: Option<&'static str>,
}
impl Clone for MatchBuilder {
    fn clone(&self) -> Self {
        let clauses = self
            .clauses
            .iter()
            .map(|clause| {
                zirql::Located::new(
                    match &clause.value {
                        zirql::MatchClause::From { name, source } => zirql::MatchClause::From {
                            name: name.clone(),
                            source: source.clone(),
                        },
                        zirql::MatchClause::Bind { name, value } => zirql::MatchClause::Bind {
                            name: name.clone(),
                            value: value.clone(),
                        },
                        zirql::MatchClause::Where { predicate } => zirql::MatchClause::Where {
                            predicate: predicate.clone(),
                        },
                    },
                    clause.range.clone(),
                )
            })
            .collect();
        Self {
            clauses,
            functions: self.functions.clone(),
            parameters: self.parameters.clone(),
            depth: self.depth,
            nodes: self.nodes,
            invalid: self.invalid,
        }
    }
}
/// Start an empty finite native match chain.
pub fn matches() -> MatchBuilder {
    MatchBuilder {
        clauses: vec![],
        functions: vec![],
        parameters: vec![],
        depth: 1,
        nodes: 1,
        invalid: None,
    }
}
impl MatchBuilder {
    fn operand<T>(mut self, name: String, query: &QueryExpr<T>, binding: bool) -> Self {
        self.functions = functions::merge(&self.functions, &query.functions);
        self.parameters = merge_parameters(&self.parameters, &query.parameters);
        self.depth = self.depth.max(query.depth.saturating_add(1));
        self.nodes = self.nodes.saturating_add(query.nodes).saturating_add(1);
        self.invalid = self.invalid.or(query.invalid);
        let source = capture::ParsedOperand::new(report::operand(&query.expression));
        let name = zirql::Located::new(name, 0..0);
        let clause = if binding {
            zirql::MatchClause::Bind {
                name,
                value: source,
            }
        } else {
            zirql::MatchClause::From { name, source }
        };
        self.clauses.push(zirql::Located::new(clause, 0..0));
        self
    }
    pub fn from(self, name: impl Into<String>, source: &OpQuery) -> Self {
        self.operand(name.into(), source, false)
    }
    pub fn bind(self, name: impl Into<String>, value: &SsaQuery) -> Self {
        self.operand(name.into(), value, true)
    }
    pub fn where_(mut self, guard: &MatchPredicate) -> Self {
        self.functions = functions::merge(&self.functions, &guard.functions);
        self.parameters = merge_parameters(&self.parameters, &guard.parameters);
        self.depth = self.depth.max(guard.depth.saturating_add(1));
        self.nodes = self.nodes.saturating_add(guard.nodes).saturating_add(1);
        self.invalid = self.invalid.or(guard.invalid);
        self.clauses.push(zirql::Located::new(
            zirql::MatchClause::Where {
                predicate: guard.inner.clone(),
            },
            0..0,
        ));
        self
    }
    pub fn build(self) -> CaptureQuery {
        let native = |value: capture::ParsedOperand| {
            zirql::Located::new(
                zirql::ExpressionKind::Sequence {
                    input: Box::new(zirql::Located::new(zirql::ExpressionKind::Input, 0..0)),
                    steps: vec![zirql::Located::new(
                        zirql::ExpressionStep::Stage(Stage::Native {
                            value,
                            range: range(),
                        }),
                        0..0,
                    )],
                },
                0..0,
            )
        };
        let count = self.clauses.len();
        let clauses = self
            .clauses
            .into_iter()
            .map(|clause| {
                zirql::Located::new(
                    match clause.value {
                        zirql::MatchClause::From { name, source } => zirql::MatchClause::From {
                            name,
                            source: native(source),
                        },
                        zirql::MatchClause::Bind { name, value } => zirql::MatchClause::Bind {
                            name,
                            value: native(value),
                        },
                        zirql::MatchClause::Where { predicate } => {
                            zirql::MatchClause::Where { predicate }
                        }
                    },
                    clause.range,
                )
            })
            .collect();
        let mut query = QueryExpr::start(vec![Stage::Match {
            value: capture::ParsedMatch::new(zirql::NativeMatch {
                clauses,
                range: 0..0,
            }),
            range: range(),
        }])
        .with_parameters(self.parameters)
        .with_functions(self.functions);
        query.depth = self.depth.max(count.saturating_add(1));
        query.nodes = self.nodes;
        query.invalid = self
            .invalid
            .or((query.depth > parser::DEFAULT_NESTING_LIMIT)
                .then_some("query nesting limit exceeded"));
        query
    }
}
impl CaptureQuery {
    /// Explicitly project native capture fields into ordered scalar report rows.
    pub fn project<S: Into<String>>(
        &self,
        subject: impl Into<String>,
        fields: impl IntoIterator<Item = (S, ReportExpr)>,
    ) -> ReportQuery {
        project_rows(self, subject, fields)
    }
}

impl ReportExpr {
    /// Call a declared pure function with explicit inputs.
    pub fn call(name: impl Into<String>, arguments: impl IntoIterator<Item = Self>) -> Self {
        let mut parameters = Vec::new();
        let mut functions = Vec::new();
        let arguments = arguments
            .into_iter()
            .map(|argument| {
                parameters = merge_parameters(&parameters, &argument.parameters);
                functions = functions::merge(&functions, &argument.functions);
                functions::normalize(argument.into_expression())
            })
            .collect();
        Self::node(zirql::ValueExpressionKind::Call {
            function: Located::new(name.into(), 0..0),
            arguments,
        })
        .with_metadata(parameters, functions)
    }
    /// Retain a native query result as an explicit function operand.
    pub fn native<T>(query: &QueryExpr<T>) -> Self {
        Self::node(zirql::ValueExpressionKind::NativeValue(Box::new(
            report::operand(&query.expression),
        )))
        .with_metadata(query.parameters.clone(), query.functions.clone())
    }
    pub fn direct_children(operation: Self) -> Self {
        Self::primitive(functions::Primitive::DirectChildren, operation)
    }
    pub fn cost_annotation(operation: Self) -> Self {
        Self::primitive(functions::Primitive::CostAnnotation, operation)
    }
    fn primitive(primitive: functions::Primitive, argument: Self) -> Self {
        let parameters = argument.parameters.clone();
        let functions = argument.functions.clone();
        Self::node(zirql::ValueExpressionKind::NativeValue(Box::new(
            report::operand(&expression(vec![Stage::FunctionPrimitive {
                primitive,
                argument: ParsedComprehension::new(functions::normalize(
                    argument.into_expression(),
                )),
                range: range(),
            }])),
        )))
        .with_metadata(parameters, functions)
    }
    /// Evaluate only the selected branch; prepare both written branches.
    pub fn conditional(condition: Self, selected: Self, otherwise: Self) -> Self {
        let parameters = merge_parameters(
            &merge_parameters(&condition.parameters, &selected.parameters),
            &otherwise.parameters,
        );
        let functions = functions::merge(
            &functions::merge(&condition.functions, &selected.functions),
            &otherwise.functions,
        );
        Self::node(zirql::ValueExpressionKind::Conditional {
            branches: vec![(condition.into_expression(), selected.into_expression())],
            otherwise: Box::new(otherwise.into_expression()),
        })
        .with_metadata(parameters, functions)
    }
}
/// Start a native or scalar query with a declared function call.
pub fn function_call<T: QueryResult>(
    name: impl Into<String>,
    arguments: impl IntoIterator<Item = ReportExpr>,
) -> QueryExpr<T> {
    let call = ReportExpr::call(name, arguments);
    QueryExpr::start(vec![Stage::FunctionValue {
        value: call.value,
        range: range(),
    }])
    .with_parameters(call.parameters)
    .with_functions(call.functions)
}

impl ReportExpr {
    /// Supply an explicit document-owned operation to a checked function argument.
    pub fn operation(operation: OperationId) -> Self {
        Self::native_input(QueryOutput::Operations(vec![operation]))
    }
    /// Supply an exact SSA identity; admission checks its current document owner.
    pub fn ssa_value(value: ValueId) -> Self {
        Self::native_input(QueryOutput::Native(NativeValue::Ssa(value)))
    }
    /// Supply a sequence whose cardinality remains explicit, including zero or one.
    pub fn operation_sequence(operations: Vec<OperationId>) -> Self {
        Self::native_input(QueryOutput::Native(NativeValue::OperationSequence(
            operations,
        )))
    }
    fn native_input(value: QueryOutput) -> Self {
        Self::node(zirql::ValueExpressionKind::NativeValue(Box::new(
            report::operand(&expression(vec![Stage::FunctionInput {
                value: std::sync::Arc::new(value),
                range: range(),
            }])),
        )))
    }
}
