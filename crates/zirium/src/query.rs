//! Composable operation-selection queries, editing, and output.

use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fmt,
};

use crate::{
    dialect::DialectRegistry,
    semantic::{
        AttributeId, AttributeSpec, AttributeValue, CfBrOp, CfCondBrOp, Document, OperationId,
        Successor, TypeId, UseSite, ValueId, ValueReference, decode_mlir_string,
    },
};

mod builder;
mod comprehension;
pub use comprehension::{ConstructedReport, ParsedComprehension};
mod diff;
mod diff_program;
mod diff_text;
pub use diff_program::{DiffProgram, DiffProgramOutput, diff_value_to_json};
pub mod lexer;
mod model;
pub use builder::*;
pub use diff::*;
pub use zirql::{
    Budget, ConstructedValue, Location as ValueLocation, ProgramLimits, ValueBinaryOperator,
    ValueUnaryOperator, constructed_value_type,
};
mod literal;
pub mod parser;
mod pipeline;
mod program;
mod render;
mod report;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    expression: model::Expression,
    statements: Vec<model::Statement>,
    emit_final_result: bool,
    parse_work: usize,
    parse_results: usize,
}

/// Bounds evaluation work and the number of items in any one stream.
/// Work counts stage inputs, visited predicate nodes, and dependency, ancestor, and subtree visits,
/// not wall-clock time.
#[derive(Clone, Copy, Debug)]
pub struct EvaluationLimits {
    pub max_work: usize,
    pub max_items: usize,
}

impl Default for EvaluationLimits {
    fn default() -> Self {
        Self {
            max_work: 10_000_000,
            max_items: 1_000_000,
        }
    }
}

/// Controls query evaluation policies independently of resource limits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvaluationOptions {
    /// Reject operations whose reference semantics are not registered during `reachable`.
    pub strict_unknown_references: bool,
}

// Document access and host policy stay outside the caller-owned core budget.
struct EvaluationContext<'a, 'document, 'query> {
    document: &'a mut DocumentAccess<'document>,
    registry: &'a DialectRegistry,
    expressions: &'a BTreeMap<usize, program::Expression<'query>>,
    options: EvaluationOptions,
    output: OutputMode,
    emit: &'a mut dyn FnMut(
        &Document,
        QueryOutput,
        &mut zirql::Budget,
    ) -> Result<(), EvaluationError>,
}

#[derive(Clone, Copy)]
enum OutputMode {
    Textual,
    Native,
}

fn evaluation_budget(limits: EvaluationLimits) -> zirql::Budget {
    zirql::Budget::new(zirql::Limits {
        work: limits.max_work,
        items: limits.max_items,
        nesting: parser::DEFAULT_NESTING_LIMIT,
    })
}

fn core_error(error: zirql::QueryError) -> EvaluationError {
    let location = match &error.location {
        zirql::Location::Query(range) => Some(range.clone()),
        _ => None,
    };
    let mut result = EvaluationError::new(match error.message.as_str() {
        "work limit exceeded" => {
            "query work limit exceeded; fixed points may need unique or a larger work limit"
                .to_owned()
        }
        "item limit exceeded" => {
            "query stream size limit exceeded; use unique or a larger item limit".to_owned()
        }
        "nesting limit exceeded" | "nested evaluation call limit exceeded" => {
            "query nesting limit exceeded".to_owned()
        }
        _ => error.message,
    });
    result.range = location;
    result
}

fn charge(budget: &mut zirql::Budget, work: usize) -> Result<(), EvaluationError> {
    budget
        .charge(work, zirql::Location::Query(0..0))
        .map_err(core_error)
}

fn check_items(budget: &zirql::Budget, count: usize) -> Result<(), EvaluationError> {
    budget
        .check_items(count, zirql::Location::Query(0..0))
        .map_err(core_error)
}

fn collect<T>(
    budget: &mut zirql::Budget,
    values: impl Iterator<Item = T>,
) -> Result<Vec<T>, EvaluationError> {
    let mut collected = Vec::new();
    for value in values {
        check_items(budget, collected.len().saturating_add(1))?;
        budget
            .charge_results(1, zirql::Location::Query(0..0))
            .map_err(core_error)?;
        collected.push(value);
    }
    Ok(collected)
}

/// Graph traversal shared by ordinary and diff query evaluation.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationTraversal {
    Reachable,
    Closure,
    Slice,
    ForwardSlice,
}

/// Runs one graph traversal from an explicit operation selection.
#[doc(hidden)]
pub fn evaluate_operation_traversal(
    document: &Document,
    operations: Vec<OperationId>,
    registry: &DialectRegistry,
    traversal: OperationTraversal,
    limits: EvaluationLimits,
    options: EvaluationOptions,
) -> Result<(Vec<OperationId>, usize), EvaluationError> {
    let mut state = evaluation_budget(limits);
    let result = match traversal {
        OperationTraversal::Reachable => {
            evaluate_reachable(document, operations, registry, options, &mut state)
        }
        OperationTraversal::Closure => {
            evaluate_closure(document, operations, registry, false, &mut state)
        }
        OperationTraversal::Slice => evaluate_slice(document, operations, &mut state),
        OperationTraversal::ForwardSlice => {
            evaluate_forward_slice(document, operations, &mut state)
        }
    }?;
    Ok((result, limits.max_work - state.remaining_work()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryOutput {
    /// Native results, including constructed textual reports.
    Native(NativeValue),
    Operations(Vec<OperationId>),
    Values(Vec<String>),
    Count(usize),
    Map(serde_json::Map<String, serde_json::Value>),
    /// A textual report map whose entry order was chosen explicitly.
    RankedMap(Vec<(String, serde_json::Value)>),
    Array(Vec<serde_json::Value>),
    Json(String),
    Text(String),
    Formatted {
        operations: Vec<OperationId>,
        options: crate::formatter::FormatOptions,
    },
}

impl QueryOutput {
    /// Materialize ordinary JSON under the active request allowance.
    pub fn to_json_with_budget(
        &self,
        document: &Document,
        budget: &mut zirql::Budget,
    ) -> Result<serde_json::Value, EvaluationError> {
        let range = match self {
            Self::Native(NativeValue::Constructed(value)) => value.range(),
            _ => 0..0,
        };
        program::to_json(document, self, budget, &range)
    }
}

/// Structured query results that have no corresponding CLI stream kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeValue {
    Constructed(ConstructedReport),
    String(String),
    Types(Vec<TypeId>),
    Attributes(Vec<(String, AttributeId)>),
    Map(BTreeMap<String, QueryOutput>),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SortKey {
    Count(usize),
    Text(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryError {
    pub position: usize,
    pub message: &'static str,
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "query error at byte {}: {}", self.position, self.message)
    }
}

impl std::error::Error for QueryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationError {
    message: String,
    range: Option<std::ops::Range<usize>>,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EvaluationError {}

impl From<zirql::QueryError> for EvaluationError {
    fn from(error: zirql::QueryError) -> Self {
        core_error(error)
    }
}

impl EvaluationError {
    /// Program byte range for a located runtime refusal.
    pub fn range(&self) -> Option<std::ops::Range<usize>> {
        self.range.clone()
    }

    fn at(mut self, range: std::ops::Range<usize>) -> Self {
        if self.range.is_none() || self.range == Some(0..0) {
            self.range = Some(range);
        }
        self
    }

    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            range: None,
        }
    }
}

impl Query {
    pub fn parse(source: &str) -> Result<Self, QueryError> {
        Self::parse_with_program_limits(source, comprehension::default_limits())
    }

    /// Parses a query with the reserved per-document binding available.
    pub fn parse_with_document_context(source: &str) -> Result<Self, QueryError> {
        Self::parse_with_document_context_and_program_limits(
            source,
            comprehension::default_limits(),
        )
    }

    /// Parse with finite work/result bounds for comprehension lowering.
    pub fn parse_with_program_limits(
        source: &str,
        limits: ProgramLimits,
    ) -> Result<Self, QueryError> {
        Self::parse_with_initial_bindings(source, &[], limits)
    }

    /// Parse with the reserved document binding and bounded comprehension lowering.
    pub fn parse_with_document_context_and_program_limits(
        source: &str,
        limits: ProgramLimits,
    ) -> Result<Self, QueryError> {
        Self::parse_with_initial_bindings(source, &["document"], limits)
    }

    fn parse_with_initial_bindings(
        source: &str,
        initial_bindings: &[&str],
        limits: ProgramLimits,
    ) -> Result<Self, QueryError> {
        // Admit an upper bound for lexer storage before allocating tokens. Only
        // actual comprehensions retain this spending for repeated evaluation.
        let mut admission = zirql::Budget::for_program(limits);
        if comprehension::has_syntax(source) {
            let work = source
                .len()
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or(QueryError {
                    position: 0,
                    message: "query allocation size overflow",
                })?;
            admission
                .charge(work, zirql::Location::Query(0..source.len()))
                .map_err(|_| QueryError {
                    position: 0,
                    message: "query work limit exceeded during parsing",
                })?;
            admission
                .charge_results(
                    source.len().saturating_add(1),
                    zirql::Location::Query(0..source.len()),
                )
                .map_err(|_| QueryError {
                    position: 0,
                    message: "result limit exceeded during parsing",
                })?;
        }
        let lexed = lexer::lex(source);
        let parsed = parser::parse_with_budget(
            &lexed,
            limits.nesting,
            initial_bindings,
            ProgramLimits {
                work: admission.remaining_work(),
                results: admission.remaining_results().unwrap(),
                ..limits
            },
        );
        let lexical = lexed.diagnostics().first().map(|diagnostic| {
            let message = match diagnostic.kind() {
                lexer::DiagnosticKind::QueryTooLarge => "query exceeds the supported size",
                lexer::DiagnosticKind::InvalidToken => "invalid token",
                lexer::DiagnosticKind::InvalidEscape => "unsupported string escape",
                lexer::DiagnosticKind::UnterminatedString => "unterminated string",
            };
            (diagnostic.range(), message)
        });
        let syntactic = parsed
            .diagnostics()
            .first()
            .map(|diagnostic| (diagnostic.range(), diagnostic.message()));
        if let Some((range, message)) = [lexical, syntactic]
            .into_iter()
            .flatten()
            .min_by_key(|(range, _)| range.start())
        {
            return Err(QueryError {
                position: range.start() as usize,
                message,
            });
        }
        let (parse_work, parse_results) = if parsed.parse_results != 0 {
            (
                limits.work - admission.remaining_work() + parsed.parse_work,
                limits.results - admission.remaining_results().unwrap() + parsed.parse_results,
            )
        } else {
            (0, 0)
        };
        let program = parsed
            .into_program()
            .expect("diagnostic-free query has a program");
        Ok(Self {
            expression: program.expression,
            statements: program.statements,
            emit_final_result: program.emit_final_result,
            parse_work,
            parse_results,
        })
    }

    /// Evaluates against the current document, invoking `emit` at each explicit
    /// emission and for the implicit final output unless it is suppressed by a
    /// final `do` statement. The callback sees each edit as it exists at that
    /// point. Callers that need all-or-nothing output should buffer emissions
    /// until evaluation succeeds. Edits commit per stage.
    pub fn evaluate(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_options_and_limits(
            document,
            registry,
            EvaluationOptions::default(),
            EvaluationLimits::default(),
            emit,
        )
    }

    pub fn evaluate_with_limits(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        limits: EvaluationLimits,
        emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_options_and_limits(
            document,
            registry,
            EvaluationOptions::default(),
            limits,
            emit,
        )
    }

    pub fn evaluate_with_options_and_limits(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        options: EvaluationOptions,
        limits: EvaluationLimits,
        emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_context_options_and_limits(
            document, registry, None, options, limits, emit,
        )
    }

    pub fn evaluate_with_context_options_and_limits(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        document_name: Option<&str>,
        options: EvaluationOptions,
        limits: EvaluationLimits,
        mut emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_budget(
            document,
            registry,
            document_name,
            options,
            &mut if self.has_comprehensions() {
                zirql::Budget::for_program(ProgramLimits {
                    work: limits.max_work,
                    stream_items: limits.max_items,
                    ..comprehension::default_limits()
                })
            } else {
                evaluation_budget(limits)
            },
            |document, value, _| emit(document, value),
        )
    }

    /// Evaluate with an explicit aggregate allowance in addition to stream bounds.
    pub fn evaluate_with_program_limits(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        document_name: Option<&str>,
        options: EvaluationOptions,
        limits: ProgramLimits,
        mut emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_budget(
            document,
            registry,
            document_name,
            options,
            &mut zirql::Budget::for_program(limits),
            |document, value, _| emit(document, value),
        )
    }

    /// Whether this parsed program uses the bounded constructed report path.
    pub fn has_comprehensions(&self) -> bool {
        self.parse_results != 0
    }

    /// Evaluate and render callbacks with one caller-owned request allowance.
    /// Comprehensions require an aggregate result allowance. Callbacks may see
    /// earlier emissions before a later error; buffer output when needed.
    pub fn evaluate_with_budget(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        document_name: Option<&str>,
        options: EvaluationOptions,
        budget: &mut zirql::Budget,
        mut emit: impl FnMut(&Document, QueryOutput, &mut zirql::Budget) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        if self.has_comprehensions() && budget.remaining_results().is_none() {
            return Err(EvaluationError::new(
                "comprehensions require an aggregate result allowance",
            ));
        }
        budget
            .charge(
                self.parse_work,
                zirql::Location::Query(program::range(self.expression.range)),
            )
            .map_err(core_error)?;
        budget
            .charge_results(
                self.parse_results,
                zirql::Location::Query(program::range(self.expression.range)),
            )
            .map_err(core_error)?;
        program::validate(self, document_name, budget)?;
        let expressions = program::expressions(
            self.statements
                .iter()
                .map(|statement| match statement {
                    model::Statement::Binding { expression, .. }
                    | model::Statement::Do(expression)
                    | model::Statement::Query(expression) => expression,
                })
                .chain(std::iter::once(&self.expression)),
        );
        let program = program::lower(self);
        let mut access = DocumentAccess::Mutable(document);
        let input = QueryOutput::Operations(collect(budget, access.operations())?);
        if let Some(name) = document_name {
            program::charge_results(budget, 1, &(0..0))?;
            program::charge_bytes(budget, name.len(), &(0..0))?;
        }
        let initial = document_name.map(|name| {
            (
                zirql::Located::new("document".to_owned(), 0..0),
                QueryOutput::Values(vec![name.to_owned()]),
            )
        });
        let bindings = zirql::Bindings::new(initial).map_err(core_error)?;
        let mut context = EvaluationContext {
            document: &mut access,
            registry,
            expressions: &expressions,
            options,
            output: OutputMode::Textual,
            emit: &mut emit,
        };
        zirql::evaluate_program(
            &program,
            &program::NativeHost,
            input,
            &mut context,
            bindings,
            budget,
        )
        .map_err(program::evaluation_error)?;
        Ok(())
    }
}

// The program frontend may edit; structured queries only receive shared access.
enum DocumentAccess<'a> {
    Shared(&'a Document),
    Mutable(&'a mut Document),
}

impl std::ops::Deref for DocumentAccess<'_> {
    type Target = Document;
    fn deref(&self) -> &Document {
        match self {
            Self::Shared(document) => document,
            Self::Mutable(document) => document,
        }
    }
}

impl DocumentAccess<'_> {
    fn editable(&mut self) -> Result<&mut Document, EvaluationError> {
        match self {
            Self::Mutable(document) => Ok(document),
            Self::Shared(_) => Err(EvaluationError::new("cannot edit in a read-only query")),
        }
    }
}

fn evaluate_expression(
    expression: &model::Expression,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &zirql::Bindings<QueryOutput>,
    budget: &mut zirql::Budget,
) -> Result<QueryOutput, EvaluationError> {
    let lowered = &context.expressions[&(std::ptr::from_ref(expression) as usize)];
    zirql::evaluate_expression(
        lowered,
        &program::NativeHost,
        input,
        context,
        bindings,
        budget,
    )
    .map_err(program::evaluation_error)
}

// One dispatcher owns all host stages, including those run inside stream segments.
fn evaluate_host_stage(
    stage: &model::Stage,
    context: &mut EvaluationContext<'_, '_, '_>,
    mut current: QueryOutput,
    bindings: &zirql::Bindings<QueryOutput>,
    budget: &mut zirql::Budget,
) -> Result<QueryOutput, EvaluationError> {
    match stage {
        model::Stage::Structured { step, .. } => {
            current = evaluate_structured(step, context, current, bindings, budget)?;
        }
        model::Stage::Comprehension { value, .. } => {
            current = QueryOutput::Native(NativeValue::Constructed(report::evaluate(
                value, context, current, bindings, budget,
            )?));
        }
        model::Stage::Projection { value, range } => {
            current = QueryOutput::Native(NativeValue::Constructed(report::project(
                value,
                context,
                current,
                bindings,
                budget,
                &program::range(*range),
            )?));
        }
        model::Stage::Literal { value, .. } => {
            let value = literal::evaluate(value, context.document, bindings, budget, &mut 0)?;
            if value_depth(&value) > parser::DEFAULT_NESTING_LIMIT {
                return Err(EvaluationError::new("JSON nesting limit exceeded"));
            }
            current = match value {
                serde_json::Value::Object(entries) => QueryOutput::Map(entries),
                serde_json::Value::Array(values) => QueryOutput::Array(values),
                _ => unreachable!("literal stages start with an object or array"),
            };
        }
        model::Stage::Binding { name, .. } => {
            let binding = bindings
                .get(name, program::range(stage.range()))
                .map_err(|_| {
                    EvaluationError::new(format!(
                        "binding `{name}` is unavailable in this evaluation context"
                    ))
                })?;
            charge(budget, output_size(binding).max(1))?;
            current = program::copy_value(binding, budget, &program::range(stage.range()))?;
        }
        model::Stage::Tally { .. } => {
            return Err(EvaluationError::new(
                "tally requires a value stream; use names or attr first",
            ));
        }
        model::Stage::Value { .. } => {
            return Err(EvaluationError::new(
                "value is only valid as the selector in sort_by(value) on a map",
            ));
        }
        model::Stage::MapBy { key, value, .. } => {
            let selected = take_operations(current, "map_by")?;
            program::charge_results(budget, 1, &program::range(stage.range()))?;
            let mut entries = serde_json::Map::new();
            let mut items = 0usize;
            for operation in selected {
                program::charge_results(budget, 1, &program::range(stage.range()))?;
                let item = QueryOutput::Operations(vec![operation]);
                let key = evaluate_expression(
                    key,
                    context,
                    program::copy_value(&item, budget, &program::range(stage.range()))?,
                    bindings,
                    budget,
                )?;
                let QueryOutput::Values(mut keys) = key else {
                    return Err(EvaluationError::new(
                        "map_by key must produce exactly one string",
                    ));
                };
                if keys.len() != 1 {
                    return Err(EvaluationError::new(
                        "map_by key must produce exactly one string",
                    ));
                }
                let key = keys.pop().unwrap();
                if entries.contains_key(&key) {
                    return Err(EvaluationError::new(format!(
                        "map_by encountered duplicate key `{key}`; select unique keys"
                    )));
                }
                let value = evaluate_expression(value, context, item, bindings, budget)?;
                items = items.saturating_add(output_size(&value).saturating_add(1));
                check_items(budget, items)?;
                charge(budget, output_size(&value))?;
                program::charge_results(budget, 1, &program::range(stage.range()))?;
                let value = program::to_json(
                    context.document,
                    &value,
                    budget,
                    &program::range(stage.range()),
                )?;
                if value_depth(&value) >= parser::DEFAULT_NESTING_LIMIT {
                    return Err(EvaluationError::new("map nesting limit exceeded"));
                }
                entries.insert(key, value);
            }
            current = QueryOutput::Map(entries);
        }
        model::Stage::Reachable { .. } => {
            current = QueryOutput::Operations(evaluate_reachable(
                context.document,
                take_operations(current, "reachable")?,
                context.registry,
                context.options,
                budget,
            )?);
        }
        model::Stage::Input { .. } => {
            current = QueryOutput::Operations(collect(budget, context.document.operations())?)
        }
        model::Stage::Filter { .. } => {
            return Err(EvaluationError::new(
                "filter requires operations, but the current stream contains values",
            ));
        }
        model::Stage::Closure { .. } => {
            let selected = take_operations(current, "closure")?;
            current = QueryOutput::Operations(evaluate_closure(
                context.document,
                selected,
                context.registry,
                false,
                budget,
            )?);
        }
        model::Stage::Slice { .. } => {
            let selected = take_operations(current, "slice")?;
            current = QueryOutput::Operations(evaluate_slice(context.document, selected, budget)?);
        }
        model::Stage::ForwardSlice { .. } => {
            let selected = take_operations(current, "forward_slice")?;
            current = QueryOutput::Operations(evaluate_forward_slice(
                context.document,
                selected,
                budget,
            )?);
        }
        model::Stage::Defs { index, .. } => {
            let selected = take_operations(current, "defs")?;
            current = QueryOutput::Operations(evaluate_defs(
                context.document,
                &selected,
                *index,
                budget,
            )?);
        }
        model::Stage::Users { index, .. } => {
            let selected = take_operations(current, "users")?;
            current = QueryOutput::Operations(evaluate_users(
                context.document,
                &selected,
                *index,
                budget,
            )?);
        }
        model::Stage::Parent { .. } => {
            let selected = take_operations(current, "parent")?;
            current =
                QueryOutput::Operations(evaluate_parent(context.document, &selected, budget)?);
        }
        model::Stage::Children { .. } => {
            let selected = take_operations(current, "children")?;
            current =
                QueryOutput::Operations(evaluate_children(context.document, &selected, budget)?);
        }
        model::Stage::Root { predicate, .. } => {
            let selected = take_operations(current, "root")?;
            let resolved = if report::has_dynamic_predicate(predicate) {
                Some(report::resolve_predicate(
                    predicate, context, bindings, budget,
                )?)
            } else {
                None
            };
            current = QueryOutput::Operations(evaluate_root(
                context.document,
                &selected,
                resolved.as_ref().unwrap_or(predicate),
                budget,
            )?);
        }
        model::Stage::Subtree { .. } => {
            let selected = take_operations(current, "subtree")?;
            let mut expanded = Vec::new();
            for operation in selected {
                append_subtree(context.document, operation, &mut expanded, budget)?;
            }
            current = QueryOutput::Operations(expanded);
        }
        model::Stage::Unique { .. } => {
            return Err(EvaluationError::new(
                "unique requires an operation or value stream",
            ));
        }
        model::Stage::Sort { .. } => {
            let QueryOutput::Values(values) = &mut current else {
                return Err(EvaluationError::new(
                    "sort requires a value stream; use names or attr first",
                ));
            };
            values.sort();
        }
        model::Stage::SortBy { selector, .. } => {
            if selector.is_map_value() {
                current = sort_map_by_value(current)?;
            } else {
                if matches!(current, QueryOutput::Map(_) | QueryOutput::RankedMap(_)) {
                    return Err(EvaluationError::new(
                        "sort_by on a map requires the scalar selector value",
                    ));
                }
                let selected = take_operations(current, "sort_by")?;
                current = QueryOutput::Operations(sort_operations_by(
                    selected, selector, context, bindings, budget, false, None,
                )?);
            }
        }
        model::Stage::Reverse { .. } => match &mut current {
            QueryOutput::Operations(values) => values.reverse(),
            QueryOutput::Values(values) => values.reverse(),
            QueryOutput::Native(NativeValue::Types(values)) => values.reverse(),
            QueryOutput::Native(NativeValue::Attributes(values)) => values.reverse(),
            QueryOutput::RankedMap(entries) => entries.reverse(),
            _ => {
                return Err(EvaluationError::new(
                    "reverse requires an operation stream, value stream, or ranked map",
                ));
            }
        },
        model::Stage::Head { count, .. } => truncate_stream(&mut current, *count, false)?,
        model::Stage::Tail { count, .. } => truncate_stream(&mut current, *count, true)?,
        model::Stage::Min { .. } => {
            current = QueryOutput::Values(extreme_value(current, false, false)?);
        }
        model::Stage::Max { .. } => {
            current = QueryOutput::Values(extreme_value(current, true, false)?);
        }
        model::Stage::MinAll { .. } => {
            current = QueryOutput::Values(extreme_value(current, false, true)?);
        }
        model::Stage::MaxAll { .. } => {
            current = QueryOutput::Values(extreme_value(current, true, true)?);
        }
        model::Stage::MinBy { selector, .. } => {
            let selected = take_operations(current, "min_by")?;
            current = QueryOutput::Operations(sort_operations_by(
                selected,
                selector,
                context,
                bindings,
                budget,
                false,
                Some(false),
            )?);
        }
        model::Stage::MaxBy { selector, .. } => {
            let selected = take_operations(current, "max_by")?;
            current = QueryOutput::Operations(sort_operations_by(
                selected,
                selector,
                context,
                bindings,
                budget,
                true,
                Some(false),
            )?);
        }
        model::Stage::MinAllBy { selector, .. } => {
            let selected = take_operations(current, "min_all_by")?;
            current = QueryOutput::Operations(sort_operations_by(
                selected,
                selector,
                context,
                bindings,
                budget,
                false,
                Some(true),
            )?);
        }
        model::Stage::MaxAllBy { selector, .. } => {
            let selected = take_operations(current, "max_all_by")?;
            current = QueryOutput::Operations(sort_operations_by(
                selected,
                selector,
                context,
                bindings,
                budget,
                true,
                Some(true),
            )?);
        }
        model::Stage::Attr { name, .. } => {
            let selected = take_operations(current, "attr")?;
            current =
                QueryOutput::Values(evaluate_attr(context.document, &selected, name, budget)?);
        }
        model::Stage::Names { .. } => {
            let selected = take_operations(current, "names")?;
            current = QueryOutput::Values(collect_strings(
                budget,
                selected
                    .iter()
                    .filter_map(|&op| context.document.operation_name(op)),
            )?);
        }
        model::Stage::ResultTypes { .. } => {
            let selected = take_operations(current, "result_types")?;
            current = QueryOutput::Values(collect_strings(
                budget,
                selected
                    .iter()
                    .flat_map(|&op| context.document.result_types(op).unwrap_or(&[]))
                    .filter_map(|&ty| context.document.type_spelling(ty)),
            )?);
        }
        model::Stage::OperandTypes { .. } => {
            let selected = take_operations(current, "operand_types")?;
            current = QueryOutput::Values(collect_strings(
                budget,
                selected
                    .iter()
                    .flat_map(|&op| context.document.operands(op).unwrap_or(&[]))
                    .filter_map(|&value| context.document.value_type(value)),
            )?);
        }
        model::Stage::Group { expression, .. } => {
            current = evaluate_expression(expression, context, current, bindings, budget)?;
        }
        model::Stage::Fixpoint { expression, .. } => {
            if expression.is_single_closure() {
                current = QueryOutput::Operations(evaluate_closure(
                    context.document,
                    take_operations(current, "closure")?,
                    context.registry,
                    true,
                    budget,
                )?);
                return Ok(current);
            }
            // Brent's cycle detection keeps one checkpoint instead of storing
            // every intermediate selection. Read-only bodies produce deterministic selections.
            let mut checkpoint =
                program::copy_value(&current, budget, &program::range(stage.range()))?;
            let mut power = 1usize;
            let mut distance = 0usize;
            loop {
                let next = evaluate_expression(
                    expression,
                    context,
                    program::copy_value(&current, budget, &program::range(stage.range()))?,
                    bindings,
                    budget,
                )?;
                if matches!(
                    next,
                    QueryOutput::Count(_)
                        | QueryOutput::Json(_)
                        | QueryOutput::Text(_)
                        | QueryOutput::Map(_)
                        | QueryOutput::RankedMap(_)
                        | QueryOutput::Array(_)
                ) {
                    return Err(EvaluationError::new(
                        "fixpoint requires an operation or value stream",
                    ));
                }
                charge(budget, 1)?;
                if next == current {
                    break;
                }
                if next == checkpoint {
                    return Err(EvaluationError {
                        range: None,
                        message: "fixpoint query cycles without reaching an unchanged selection"
                            .into(),
                    });
                }
                current = next;
                distance += 1;
                if distance == power {
                    checkpoint =
                        program::copy_value(&current, budget, &program::range(stage.range()))?;
                    power = power.saturating_mul(2);
                    distance = 0;
                }
            }
        }
        model::Stage::SetAttr { name, value, .. } => {
            let selected = operations(&current, "set_attr")?;
            let mut editor = context
                .document
                .editable()?
                .edit(context.registry)
                .map_err(edit_error)?;
            for operation in selected.iter().copied().collect::<HashSet<_>>() {
                editor
                    .set_attribute(operation, AttributeSpec::string(name.clone(), value))
                    .map_err(edit_error)?;
            }
            editor.commit().map_err(edit_error)?;
        }
        model::Stage::RemoveAttr { name, .. } => {
            let selected = operations(&current, "remove_attr")?;
            let targets = selected
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .into_iter()
                .filter(|&operation| {
                    context.document.operation_is_unparsed(operation) == Some(true)
                        || context.document.attribute_id(operation, name).is_some()
                })
                .collect::<Vec<_>>();
            if !targets.is_empty() {
                let mut editor = context
                    .document
                    .editable()?
                    .edit(context.registry)
                    .map_err(edit_error)?;
                for operation in targets {
                    editor
                        .remove_attribute(operation, name)
                        .map_err(edit_error)?;
                }
                editor.commit().map_err(edit_error)?;
            }
        }
        model::Stage::Check {
            expected, message, ..
        } => {
            let actual = countable_len(&current, "check")?;
            let failure = match expected {
                Some(expected) if actual != *expected => {
                    Some(format!("expected {expected} items, got {actual}"))
                }
                None if actual == 0 => Some("expected at least one item, got 0".to_owned()),
                _ => None,
            };
            if let Some(failure) = failure {
                let message = match message {
                    Some(message) => format!("{failure}: {message}"),
                    None => failure,
                };
                return Err(EvaluationError::new(format!("check failed: {message}")));
            }
        }
        model::Stage::Count { .. } => {
            program::charge_results(budget, 1, &program::range(stage.range()))?;
            return Ok(QueryOutput::Count(countable_len(&current, "count")?));
        }
        model::Stage::Emit { .. }
        | model::Stage::Format { .. }
        | model::Stage::Markdown { .. }
        | model::Stage::Json { .. } => unreachable!("program emitters are shared expression steps"),
        model::Stage::Print { parts, .. } => {
            let mut text = render::interpolate(parts, bindings, budget)?;
            text.push('\n');
            (context.emit)(context.document, QueryOutput::Text(text), budget)?;
        }
    }
    Ok(current)
}

fn operations<'a>(
    output: &'a QueryOutput,
    stage: &str,
) -> Result<&'a [OperationId], EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        _ => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
    }
}

fn take_operations(output: QueryOutput, stage: &str) -> Result<Vec<OperationId>, EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        _ => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
    }
}

fn countable_len(output: &QueryOutput, stage: &str) -> Result<usize, EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected.len()),
        QueryOutput::Values(values) => Ok(values.len()),
        QueryOutput::Map(values) => Ok(values.len()),
        QueryOutput::RankedMap(values) => Ok(values.len()),
        QueryOutput::Array(values) => Ok(values.len()),
        QueryOutput::Native(NativeValue::Types(values)) => Ok(values.len()),
        QueryOutput::Native(NativeValue::Attributes(values)) => Ok(values.len()),
        QueryOutput::Native(NativeValue::Map(values)) => Ok(values.len()),
        QueryOutput::Native(NativeValue::Constructed(_))
        | QueryOutput::Native(NativeValue::String(_))
        | QueryOutput::Count(_)
        | QueryOutput::Json(_)
        | QueryOutput::Text(_)
        | QueryOutput::Formatted { .. } => Err(EvaluationError::new(format!(
            "{stage} requires a stream, map, or array"
        ))),
    }
}

fn output_len(output: &QueryOutput) -> usize {
    match output {
        QueryOutput::Operations(values) => values.len(),
        QueryOutput::Values(values) => values.len(),
        QueryOutput::Map(values) => values.len(),
        QueryOutput::RankedMap(values) => values.len(),
        QueryOutput::Array(values) => values.len(),
        QueryOutput::Native(NativeValue::Types(values)) => values.len(),
        QueryOutput::Native(NativeValue::Attributes(values)) => values.len(),
        QueryOutput::Native(NativeValue::Map(values)) => values.len(),
        QueryOutput::Native(NativeValue::Constructed(_))
        | QueryOutput::Native(NativeValue::String(_))
        | QueryOutput::Count(_)
        | QueryOutput::Json(_)
        | QueryOutput::Text(_)
        | QueryOutput::Formatted { .. } => 1,
    }
}

fn truncate_stream(
    output: &mut QueryOutput,
    count: usize,
    from_end: bool,
) -> Result<(), EvaluationError> {
    fn truncate<T>(values: &mut Vec<T>, count: usize, from_end: bool) {
        if from_end {
            let keep_from = values.len().saturating_sub(count);
            values.drain(..keep_from);
        } else {
            values.truncate(count);
        }
    }
    match output {
        QueryOutput::Operations(values) => truncate(values, count, from_end),
        QueryOutput::Values(values) => truncate(values, count, from_end),
        QueryOutput::Native(NativeValue::Types(values)) => truncate(values, count, from_end),
        QueryOutput::Native(NativeValue::Attributes(values)) => truncate(values, count, from_end),
        QueryOutput::RankedMap(values) => truncate(values, count, from_end),
        _ => {
            return Err(EvaluationError::new(
                "head and tail require an operation stream, value stream, or ranked map",
            ));
        }
    }
    Ok(())
}

fn extreme_value(
    output: QueryOutput,
    maximum: bool,
    retain_all: bool,
) -> Result<Vec<String>, EvaluationError> {
    let QueryOutput::Values(values) = output else {
        return Err(EvaluationError::new(
            "value extrema require a value stream; use names or attr first",
        ));
    };
    let value = if maximum {
        values.iter().max()
    } else {
        values.iter().min()
    }
    .cloned()
    .ok_or_else(|| EvaluationError::new("value extrema require a non-empty value stream"))?;
    if retain_all {
        Ok(values
            .into_iter()
            .filter(|candidate| candidate == &value)
            .collect())
    } else {
        Ok(vec![value])
    }
}

fn sort_map_by_value(output: QueryOutput) -> Result<QueryOutput, EvaluationError> {
    let mut entries: Vec<_> = match output {
        QueryOutput::Map(entries) => entries.into_iter().collect(),
        QueryOutput::RankedMap(entries) => entries,
        _ => {
            return Err(EvaluationError::new("sort_by(value) requires a map"));
        }
    };
    let kind = entries
        .first()
        .map(|(_, value)| scalar_sort_kind(value))
        .transpose()?;
    for (_, value) in &entries {
        if Some(scalar_sort_kind(value)?) != kind {
            return Err(EvaluationError::new(
                "sort_by(value) requires comparable scalar map values of one type",
            ));
        }
    }
    entries.sort_by(|(left_key, left), (right_key, right)| {
        compare_scalar_values(left, right).then_with(|| left_key.cmp(right_key))
    });
    Ok(QueryOutput::RankedMap(entries))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarSortKind {
    Null,
    Bool,
    Number,
    String,
}

fn scalar_sort_kind(value: &serde_json::Value) -> Result<ScalarSortKind, EvaluationError> {
    match value {
        serde_json::Value::Null => Ok(ScalarSortKind::Null),
        serde_json::Value::Bool(_) => Ok(ScalarSortKind::Bool),
        serde_json::Value::Number(_) => Ok(ScalarSortKind::Number),
        serde_json::Value::String(_) => Ok(ScalarSortKind::String),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => Err(EvaluationError::new(
            "sort_by(value) requires scalar map values; project a scalar from nested maps first",
        )),
    }
}

fn compare_scalar_values(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> std::cmp::Ordering {
    match (left, right) {
        (serde_json::Value::Null, serde_json::Value::Null) => std::cmp::Ordering::Equal,
        (serde_json::Value::Bool(left), serde_json::Value::Bool(right)) => left.cmp(right),
        (serde_json::Value::Number(left), serde_json::Value::Number(right)) => {
            let integer = |number: &serde_json::Number| {
                number
                    .as_i64()
                    .map(i128::from)
                    .or_else(|| number.as_u64().map(i128::from))
            };
            match (integer(left), integer(right)) {
                (Some(left), Some(right)) => left.cmp(&right),
                (Some(left), None) => compare_integer_float(left, right.as_f64().unwrap()),
                (None, Some(right)) => {
                    compare_integer_float(right, left.as_f64().unwrap()).reverse()
                }
                (None, None) => left.as_f64().partial_cmp(&right.as_f64()).unwrap(),
            }
        }
        (serde_json::Value::String(left), serde_json::Value::String(right)) => left.cmp(right),
        _ => unreachable!("map values are validated before sorting"),
    }
}

fn compare_integer_float(integer: i128, float: f64) -> std::cmp::Ordering {
    // JSON integers fit in i64/u64. Saturation of this wider cast still orders
    // out-of-range floats correctly; compare fractional parts only on a tie.
    integer
        .cmp(&(float as i128))
        .then_with(|| 0.0f64.partial_cmp(&float.fract()).unwrap())
}

fn sort_operations_by(
    selected: Vec<OperationId>,
    selector: &model::Expression,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &zirql::Bindings<QueryOutput>,
    budget: &mut zirql::Budget,
    descending: bool,
    extrema: Option<bool>,
) -> Result<Vec<OperationId>, EvaluationError> {
    if extrema.is_some() && selected.is_empty() {
        return Err(EvaluationError::new(
            "by extrema require a non-empty operation stream",
        ));
    }
    let mut keyed = Vec::with_capacity(selected.len());
    for operation in selected {
        let output = evaluate_expression(
            selector,
            context,
            QueryOutput::Operations(vec![operation]),
            bindings,
            budget,
        )?;
        let key = match output {
            QueryOutput::Count(value) => SortKey::Count(value),
            QueryOutput::Native(NativeValue::String(value)) => SortKey::Text(value),
            QueryOutput::Values(mut values) if values.len() == 1 => {
                SortKey::Text(values.pop().unwrap())
            }
            _ => {
                return Err(EvaluationError::new(
                    "ordering selector must produce exactly one string or count per operation",
                ));
            }
        };
        keyed.push((operation, key));
    }
    if let Some(retain_all) = extrema {
        let keys = keyed.iter().map(|(_, key)| key);
        let extreme = if descending { keys.max() } else { keys.min() }
            .cloned()
            .expect("extrema reject empty streams before computing keys");
        keyed.retain(|(_, key)| key == &extreme);
        if !retain_all {
            keyed.truncate(1);
        }
    } else {
        keyed.sort_by(|left, right| left.1.cmp(&right.1));
    }
    Ok(keyed.into_iter().map(|(operation, _)| operation).collect())
}

fn value_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(entries) => {
            1 + entries.values().map(value_depth).max().unwrap_or(0)
        }
        serde_json::Value::Array(values) => 1 + values.iter().map(value_depth).max().unwrap_or(0),
        _ => 0,
    }
}

// Bound nested aggregate contents as well as the outer map.
fn json_size(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(entries) => {
            1 + entries
                .values()
                .map(|value| 1 + json_size(value))
                .sum::<usize>()
        }
        serde_json::Value::Array(values) => 1 + values.iter().map(json_size).sum::<usize>(),
        _ => 1,
    }
}

fn output_size(output: &QueryOutput) -> usize {
    match output {
        QueryOutput::Native(NativeValue::Constructed(value)) => value.size(),
        QueryOutput::Native(NativeValue::Map(entries)) => {
            entries.values().map(|value| 1 + output_size(value)).sum()
        }
        QueryOutput::Map(entries) => entries.values().map(|value| 1 + json_size(value)).sum(),
        QueryOutput::RankedMap(entries) => {
            entries.iter().map(|(_, value)| 1 + json_size(value)).sum()
        }
        QueryOutput::Array(values) => 1 + values.iter().map(json_size).sum::<usize>(),
        _ => output_len(output),
    }
}

fn source_ordered(document: &Document, selected: HashSet<OperationId>) -> Vec<OperationId> {
    document
        .operations()
        .filter(|operation| selected.contains(operation))
        .collect()
}

fn evaluate_set_operator(
    document: &Document,
    operator: model::SetOperator,
    left: QueryOutput,
    right: QueryOutput,
) -> Result<QueryOutput, EvaluationError> {
    match (left, right) {
        (QueryOutput::Operations(left), QueryOutput::Operations(right)) => {
            let mut left = left.into_iter().collect::<HashSet<_>>();
            let right = right.into_iter().collect::<HashSet<_>>();
            match operator {
                model::SetOperator::Union => left.extend(right),
                model::SetOperator::Intersect => left.retain(|operation| right.contains(operation)),
                model::SetOperator::Except => left.retain(|operation| !right.contains(operation)),
            }
            Ok(QueryOutput::Operations(source_ordered(document, left)))
        }
        (QueryOutput::Values(mut left), QueryOutput::Values(right)) => {
            let right_set = right.iter().cloned().collect::<HashSet<_>>();
            match operator {
                model::SetOperator::Union => left.extend(right),
                model::SetOperator::Intersect => left.retain(|value| right_set.contains(value)),
                model::SetOperator::Except => left.retain(|value| !right_set.contains(value)),
            }
            retain_unique(&mut left);
            Ok(QueryOutput::Values(left))
        }
        (QueryOutput::Operations(_), QueryOutput::Values(_))
        | (QueryOutput::Values(_), QueryOutput::Operations(_)) => Err(EvaluationError::new(
            "set operands must produce the same kind of stream",
        )),
        _ => Err(EvaluationError::new(
            "set operands require operation or value streams",
        )),
    }
}

fn retain_unique<T: Clone + Eq + std::hash::Hash>(values: &mut Vec<T>) {
    let mut seen = HashSet::new();
    values.retain(|value| seen.insert(value.clone()));
}

fn evaluate_defs(
    document: &Document,
    selected: &[OperationId],
    index: Option<usize>,
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    collect(
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
            .filter_map(|operand| match *operand {
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    Some(operation)
                }
                ValueReference::Resolved(ValueId::BlockArgument { block, .. }) => document
                    .block(block)
                    .and_then(|block| document.region(block.parent_region()))
                    .map(|region| region.parent_operation()),
                ValueReference::Invalid(_) => None,
            }),
    )
}

fn evaluate_users(
    document: &Document,
    selected: &[OperationId],
    index: Option<usize>,
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    collect(
        budget,
        selected
            .iter()
            .flat_map(|&operation| {
                let count = document.result_types(operation).map_or(0, <[_]>::len);
                let results = match index {
                    Some(index) if index < count => index..index + 1,
                    Some(_) => 0..0,
                    None => 0..count,
                };
                results.map(|result| result as u32).flat_map(move |result| {
                    document.uses(ValueId::OperationResult { operation, result })
                })
            })
            .map(|site| match site {
                UseSite::Operand { operation, .. }
                | UseSite::SuccessorArgument { operation, .. } => operation,
            }),
    )
}

fn evaluate_parent(
    document: &Document,
    selected: &[OperationId],
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    collect(
        budget,
        selected.iter().filter_map(|&operation| {
            document
                .operation(operation)?
                .parent_block()
                .and_then(|block| document.block(block))
                .and_then(|block| document.region(block.parent_region()))
                .map(|region| region.parent_operation())
        }),
    )
}

fn evaluate_children(
    document: &Document,
    selected: &[OperationId],
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    collect(
        budget,
        selected
            .iter()
            .flat_map(|&operation| document.operation_regions(operation).unwrap_or(&[]))
            .flat_map(|&region| {
                document
                    .region(region)
                    .and_then(|region| region.blocks(document))
                    .unwrap_or(&[])
            })
            .flat_map(|&block| document.block_operations(block).unwrap_or(&[]))
            .copied(),
    )
}

fn evaluate_root(
    document: &Document,
    selected: &[OperationId],
    predicate: &model::Predicate,
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    let filter = pipeline::root_filter(predicate)?;
    let mut roots = Vec::new();
    for &operation in selected {
        let mut candidate = Some(operation);
        while let Some(operation) = candidate {
            charge(budget, 1)?;
            if pipeline::matches_root(&filter, document, operation, budget)? {
                program::charge_results(budget, 1, &program::range(predicate.range()))?;
                roots.push(operation);
                break;
            }
            candidate = operation_parent(document, operation);
        }
    }
    Ok(roots)
}

fn operation_parent(document: &Document, operation: OperationId) -> Option<OperationId> {
    document
        .operation(operation)?
        .parent_block()
        .and_then(|block| document.block(block))
        .and_then(|block| document.region(block.parent_region()))
        .map(|region| region.parent_operation())
}

fn collect_strings<'a>(
    budget: &mut zirql::Budget,
    strings: impl Iterator<Item = &'a str>,
) -> Result<Vec<String>, EvaluationError> {
    let mut values = Vec::new();
    for value in strings {
        check_items(budget, values.len() + 1)?;
        program::charge_results(budget, 1, &(0..0))?;
        program::charge_bytes(budget, value.len(), &(0..0))?;
        values.push(value.to_owned());
    }
    Ok(values)
}

fn evaluate_attr(
    document: &Document,
    selected: &[OperationId],
    name: &str,
    budget: &mut zirql::Budget,
) -> Result<Vec<String>, EvaluationError> {
    let mut values = Vec::new();
    for &operation in selected {
        let Some(attribute) = document.attribute_id(operation, name) else {
            continue;
        };
        let Some(value) = document.attribute_value(attribute) else {
            continue;
        };
        program::charge_results(budget, 1, &(0..0))?;
        program::charge_bytes(
            budget,
            document
                .attribute_spelling_value(attribute)
                .map_or(0, str::len),
            &(0..0),
        )?;
        values.push(match value {
            AttributeValue::String(spelling) => decode_mlir_string(spelling).ok_or_else(|| EvaluationError::new(format!("attribute `{name}` cannot be decoded as UTF-8; use json to inspect its escaped spelling")))?,
            AttributeValue::Symbol(path) => path.join("::"),
            _ => document.attribute_spelling_value(attribute).unwrap().to_owned(),
        });
    }
    Ok(values)
}

fn output_value(document: &Document, output: &QueryOutput) -> serde_json::Value {
    match output {
        QueryOutput::Native(value) => match value {
            NativeValue::Constructed(_) => {
                unreachable!("constructed reports use bounded rendering")
            }
            NativeValue::String(value) => serde_json::json!(value),
            NativeValue::Types(values) => serde_json::json!(
                values
                    .iter()
                    .map(|&id| document.type_spelling(id))
                    .collect::<Vec<_>>()
            ),
            NativeValue::Attributes(values) => serde_json::json!(
                values
                    .iter()
                    .map(|(_, id)| document.attribute_spelling_value(*id))
                    .collect::<Vec<_>>()
            ),
            NativeValue::Map(values) => serde_json::Value::Object(
                values
                    .iter()
                    .map(|(key, value)| (key.clone(), output_value(document, value)))
                    .collect(),
            ),
        },
        QueryOutput::Operations(selected) => serde_json::Value::Array(
            selected
                .iter()
                .map(|&operation| {
                    let attributes = document
                        .attributes(operation)
                        .into_iter()
                        .flatten()
                        .map(|(name, spelling)| {
                            (
                                name.to_owned(),
                                serde_json::Value::String(spelling.to_owned()),
                            )
                        })
                        .collect::<serde_json::Map<_, _>>();
                    serde_json::json!({
                        "name": document.operation_name(operation),
                        "attributes": attributes,
                        "operand_types": document.operands(operation).unwrap_or(&[]).iter()
                            .map(|&value| document.value_type(value)).collect::<Vec<_>>(),
                        "result_types": document.result_types(operation).unwrap_or(&[]).iter()
                            .map(|&ty| document.type_spelling(ty)).collect::<Vec<_>>(),
                    })
                })
                .collect(),
        ),
        QueryOutput::Values(values) => serde_json::json!(values),
        QueryOutput::Map(values) => serde_json::Value::Object(values.clone()),
        QueryOutput::RankedMap(values) => {
            serde_json::Value::Object(values.iter().cloned().collect())
        }
        QueryOutput::Array(values) => serde_json::Value::Array(values.clone()),
        QueryOutput::Count(count) => serde_json::json!(count),
        QueryOutput::Json(value) | QueryOutput::Text(value) => {
            serde_json::Value::String(value.clone())
        }
        QueryOutput::Formatted { .. } => {
            unreachable!("formatted output is an emission request, not a query value")
        }
    }
}

fn edit_error(error: impl fmt::Display) -> EvaluationError {
    EvaluationError {
        range: None,
        message: format!("edit failed: {error}"),
    }
}

#[derive(Default)]
struct ClosureSelection {
    selected: HashSet<OperationId>,
    pending: VecDeque<OperationId>,
    // A selected operation need not have had its subtree retained yet.
    expanded_subtrees: HashSet<OperationId>,
}

impl ClosureSelection {
    fn insert(
        &mut self,
        operation: OperationId,
        budget: &mut zirql::Budget,
    ) -> Result<(), EvaluationError> {
        if !self.selected.contains(&operation) {
            program::charge_results(budget, 2, &(0..0))?;
            self.selected.insert(operation);
            check_items(budget, self.selected.len())?;
            self.pending.push_back(operation);
        }
        Ok(())
    }
}

// SSA-only traversal: block arguments are boundaries, and regions/callees are
// not expanded implicitly. This is a structural slice, not a standalone program.
fn evaluate_slice(
    document: &Document,
    seeds: Vec<OperationId>,
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selection = ClosureSelection::default();
    for operation in seeds {
        selection.insert(operation, budget)?;
    }
    while let Some(operation) = selection.pending.pop_front() {
        charge(budget, 1)?;
        if document.operation_is_unparsed(operation) == Some(true) {
            return Err(EvaluationError::new(
                "slice requires parsed SSA operands; load the appropriate registry",
            ));
        }
        charge(budget, document.operands(operation).unwrap_or(&[]).len())?;
        for operand in document.operands(operation).unwrap_or(&[]) {
            match *operand {
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    selection.insert(operation, budget)?
                }
                ValueReference::Resolved(ValueId::BlockArgument { .. }) => {}
                ValueReference::Invalid(_) => {
                    return Err(EvaluationError::new(
                        "slice encountered an invalid SSA operand",
                    ));
                }
            }
        }
    }
    Ok(source_ordered(document, selection.selected))
}

// SSA-only traversal in the use direction. As with `users`, this follows uses
// of operation results, including successor arguments, but does not infer
// control-flow, symbol, region, or memory dependencies.
fn evaluate_forward_slice(
    document: &Document,
    seeds: Vec<OperationId>,
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selection = ClosureSelection::default();
    for operation in seeds {
        selection.insert(operation, budget)?;
    }
    while let Some(operation) = selection.pending.pop_front() {
        charge(budget, 1)?;
        let result_count = document.result_types(operation).map_or(0, <[_]>::len);
        for result in 0..result_count {
            for site in document.uses(ValueId::OperationResult {
                operation,
                result: result as u32,
            }) {
                charge(budget, 1)?;
                let user = match site {
                    UseSite::Operand { operation, .. }
                    | UseSite::SuccessorArgument { operation, .. } => operation,
                };
                selection.insert(user, budget)?;
            }
        }
    }
    Ok(source_ordered(document, selection.selected))
}

// Analysis reachability retains bodies and explicit dependencies, without
// widening block arguments or successor edges to their owning function.
fn evaluate_reachable(
    document: &Document,
    seeds: Vec<OperationId>,
    registry: &DialectRegistry,
    options: EvaluationOptions,
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selection = ClosureSelection::default();
    for operation in seeds {
        selection.insert(operation, budget)?;
    }
    while let Some(operation) = selection.pending.pop_front() {
        charge(budget, 1)?;
        let name = document
            .operation_name(operation)
            .unwrap_or("<invalid operation>");
        if document.operation_is_unparsed(operation) == Some(true)
            || (registry.operation(name).is_none()
                && registry.operation_shape(name).is_none()
                && registry.operation_format(name).is_none()
                && registry.operation_grammars(name).is_none())
        {
            if options.strict_unknown_references {
                return Err(EvaluationError::new(format!(
                    "reachable cannot determine reference semantics for `{name}`; load the appropriate registry"
                )));
            }
            continue;
        }
        // Region-bearing operations include their explicitly represented bodies.
        retain_subtree(document, operation, &mut selection, budget)?;
        if let Some(attribute) = registry.call_target_attribute(name) {
            let callee = call_target_spelling(document, operation, attribute).ok_or_else(|| {
                EvaluationError::new(format!(
                    "reachable encountered `{name}` without call-target attribute `{attribute}`"
                ))
            })?;
            let target = document
                .checked_lookup_symbol(operation, &callee, registry)
                .map_err(|error| {
                    EvaluationError::new(format!("reachable could not look up `{callee}`: {error}"))
                })?
                .ok_or_else(|| {
                    EvaluationError::new(format!("reachable could not resolve callee `{callee}`"))
                })?;
            if document.operation_regions(target).unwrap_or(&[]).is_empty() {
                return Err(EvaluationError::new(format!(
                    "reachable cannot inspect external callee `{callee}` without a body"
                )));
            }
            // Include the callee's body, without counting its declaration as an operation in the caller.
            for child in evaluate_children(document, &[target], budget)? {
                retain_subtree(document, child, &mut selection, budget)?;
            }
        } else if registry.symbols(name).uses_symbols {
            return Err(EvaluationError::new(format!(
                "reachable does not yet support symbol references on `{name}`"
            )));
        }
        let successors = document.successors(operation).unwrap_or(&[]);
        if !successors.is_empty() && name != "cf.br" && name != "cf.cond_br" {
            return Err(EvaluationError::new(format!(
                "reachable does not yet support successor references on `{name}`"
            )));
        }
        for successor in successors {
            for &target in document.block_operations(successor.block()).unwrap_or(&[]) {
                selection.insert(target, budget)?;
            }
        }
        charge(budget, document.operands(operation).unwrap_or(&[]).len())?;
        for operand in document.operands(operation).unwrap_or(&[]) {
            match *operand {
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    selection.insert(operation, budget)?
                }
                ValueReference::Resolved(ValueId::BlockArgument { .. }) => {}
                ValueReference::Invalid(_) => {
                    return Err(EvaluationError::new(
                        "reachable encountered an invalid SSA operand",
                    ));
                }
            }
        }
    }
    Ok(source_ordered(document, selection.selected))
}

fn evaluate_closure(
    document: &Document,
    seeds: Vec<OperationId>,
    registry: &DialectRegistry,
    transitive: bool,
    budget: &mut zirql::Budget,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selection = ClosureSelection::default();
    for operation in seeds {
        selection.insert(operation, budget)?;
    }
    let seed_count = selection.pending.len();
    let mut visited = 0;
    while transitive || visited < seed_count {
        let Some(operation) = selection.pending.pop_front() else {
            break;
        };
        charge(budget, 1)?;
        visited += 1;
        let name = document
            .operation_name(operation)
            .unwrap_or("<invalid operation>");
        if document.operation_is_unparsed(operation) == Some(true)
            || (registry.operation(name).is_none()
                && registry.operation_shape(name).is_none()
                && registry.operation_format(name).is_none()
                && registry.operation_grammars(name).is_none())
        {
            return Err(EvaluationError {
                range: None,
                message: format!(
                    "closure cannot determine reference semantics for unregistered operation `{name}`"
                ),
            });
        }
        if let Some(attribute) = registry.call_target_attribute(name) {
            let callee = call_target_spelling(document, operation, attribute).ok_or_else(|| {
                EvaluationError {
                    range: None,
                    message: if name == "func.call" {
                        "closure encountered a func.call without a callee".to_owned()
                    } else {
                        format!(
                            "closure encountered `{name}` without call-target attribute `{attribute}`"
                        )
                    },
                }
            })?;
            let target = document
                .checked_lookup_symbol(operation, &callee, registry)
                .map_err(|error| EvaluationError {
                    range: None,
                    message: if name == "func.call" {
                        format!("closure could not look up func.call callee `{callee}`: {error}")
                    } else {
                        format!(
                            "closure could not look up `{name}` target `{callee}`: {error}"
                        )
                    },
                })?
                .ok_or_else(|| EvaluationError {
                    range: None,
                    message: if name == "func.call" {
                        format!(
                            "closure could not resolve func.call callee `{callee}` in an enclosing symbol table"
                        )
                    } else {
                        format!(
                            "closure could not resolve `{name}` target `{callee}` in an enclosing symbol table"
                        )
                    },
                })?;
            retain_subtree(document, target, &mut selection, budget)?;
        }
        if let Some(branch) = CfBrOp::cast(document, operation) {
            let successor = branch.successor().ok_or_else(|| EvaluationError {
                range: None,
                message: "closure encountered cf.br without a successor".to_owned(),
            })?;
            retain_successor_region(document, successor, name, &mut selection, budget)?;
        } else if let Some(branch) = CfCondBrOp::cast(document, operation) {
            let successors = branch.successors().ok_or_else(|| EvaluationError {
                range: None,
                message: "closure encountered cf.cond_br without successors".to_owned(),
            })?;
            for &successor in successors {
                retain_successor_region(document, successor, name, &mut selection, budget)?;
            }
        }
        if !document.successors(operation).unwrap_or(&[]).is_empty()
            && name != "cf.br"
            && name != "cf.cond_br"
        {
            return Err(EvaluationError {
                range: None,
                message: format!("closure does not yet support successor references on `{name}`"),
            });
        }
        if registry.symbols(name).uses_symbols && registry.call_target_attribute(name).is_none() {
            return Err(EvaluationError {
                range: None,
                message: format!("closure does not yet support symbol references on `{name}`"),
            });
        }
        charge(budget, document.operands(operation).unwrap_or(&[]).len())?;
        for operand in document.operands(operation).unwrap_or(&[]) {
            match *operand {
                ValueReference::Invalid(_) => {
                    return Err(EvaluationError {
                        range: None,
                        message: format!("closure encountered an invalid SSA operand on `{name}`"),
                    });
                }
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    selection.insert(operation, budget)?;
                }
                ValueReference::Resolved(ValueId::BlockArgument { block, .. }) => {
                    let owner = document
                            .block(block)
                            .and_then(|block| document.region(block.parent_region()))
                            .map(|region| region.parent_operation())
                            .ok_or_else(|| EvaluationError {
                                range: None,
                                message: format!(
                                    "closure could not resolve the owning scope for a block argument on `{name}`"
                                ),
                            })?;
                    retain_subtree(document, owner, &mut selection, budget)?;
                }
            }
        }
    }
    Ok(document
        .operations()
        .filter(|operation| selection.selected.contains(operation))
        .collect())
}

fn call_target_spelling(
    document: &Document,
    operation: OperationId,
    attribute: &str,
) -> Option<String> {
    document
        .attribute_id(operation, attribute)
        .or_else(|| {
            (attribute != "callee")
                .then(|| document.attribute_id(operation, "callee"))
                .flatten()
        })
        .and_then(|id| document.attribute_spelling_value(id))
        .map(str::to_owned)
}

fn retain_successor_region(
    document: &Document,
    successor: Successor,
    operation_name: &str,
    selection: &mut ClosureSelection,
    budget: &mut zirql::Budget,
) -> Result<(), EvaluationError> {
    let owner = document
        .block(successor.block())
        .and_then(|block| document.region(block.parent_region()))
        .map(|region| region.parent_operation())
        .ok_or_else(|| EvaluationError {
            range: None,
            message: format!(
                "closure encountered an invalid successor target on `{operation_name}`"
            ),
        })?;
    retain_subtree(document, owner, selection, budget)?;
    Ok(())
}

fn retain_subtree(
    document: &Document,
    operation: OperationId,
    selection: &mut ClosureSelection,
    budget: &mut zirql::Budget,
) -> Result<(), EvaluationError> {
    let mut pending = vec![operation];
    while let Some(operation) = pending.pop() {
        if !selection.expanded_subtrees.insert(operation) {
            continue;
        }
        charge(budget, 1)?;
        selection.insert(operation, budget)?;
        for &region in document.operation_regions(operation).unwrap_or(&[]) {
            for &block in document
                .region(region)
                .and_then(|region| region.blocks(document))
                .unwrap_or(&[])
            {
                pending.extend(document.block_operations(block).unwrap_or(&[]));
            }
        }
    }
    Ok(())
}

fn append_subtree(
    document: &Document,
    operation: OperationId,
    selected: &mut Vec<OperationId>,
    budget: &mut zirql::Budget,
) -> Result<(), EvaluationError> {
    let mut pending = vec![operation];
    while let Some(operation) = pending.pop() {
        charge(budget, 1)?;
        check_items(budget, selected.len().saturating_add(1))?;
        program::charge_results(budget, 1, &(0..0))?;
        selected.push(operation);
        let children = document
            .operation_regions(operation)
            .unwrap_or(&[])
            .iter()
            .flat_map(|&region| {
                document
                    .region(region)
                    .and_then(|region| region.blocks(document))
                    .unwrap_or(&[])
            })
            .flat_map(|&block| document.block_operations(block).unwrap_or(&[]))
            .copied()
            .collect::<Vec<_>>();
        pending.extend(children.into_iter().rev());
    }
    Ok(())
}
