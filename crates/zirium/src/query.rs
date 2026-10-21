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
mod capture;
pub use capture::{CaptureMatches, ParsedOperand};
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
    Budget, ConstructedValue, Located, Location as ValueLocation, NativeCapture,
    ParameterDeclaration, ParameterInputs, ParameterType, ProgramLimits, SignatureType,
    ValueBinaryOperator, ValueUnaryOperator, constructed_value_type, copy_pattern,
};
mod functions;
mod literal;
pub mod parser;
mod pipeline;
mod program;
mod render;
mod report;
mod worklists;
pub use functions::{
    FunctionDefinition, FunctionLimits, FunctionNativeType, FunctionParameter, FunctionType,
    Pattern, PatternKind, PatternParameter,
};
pub use worklists::{
    Computational as ComputationalResult, Sequence as NativeSequence, Tuple as NativeTuple,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    functions: Vec<FunctionDefinition>,
    function_limits: FunctionLimits,
    parameters: Vec<ParameterDeclaration>,
    parameter_positions: Vec<usize>,
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
struct EvaluationContext<'a, 'document> {
    document: &'a mut DocumentAccess<'document>,
    registry: &'a DialectRegistry,
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
    /// Native results, including SSA captures and constructed reports.
    Native(NativeValue),
    Operations(Vec<OperationId>),
    Values(Vec<String>),
    Count(usize),
    Map(Vec<(String, serde_json::Value)>),
    Array(Vec<serde_json::Value>),
    Json(String),
    Text(String),
    Formatted {
        operations: Vec<OperationId>,
        options: crate::formatter::FormatOptions,
    },
}

impl QueryOutput {
    /// Materialize a retained native wrapper under the active request allowance.
    pub fn into_owned_with_budget(self, budget: &mut Budget) -> Result<Self, EvaluationError> {
        functions::owned(self, budget, &(0..0))
    }
    /// Materialize mixed reports through Zirql's shared container traversal.
    /// Native leaves retain Zirium descriptors and owner checks. Constructed
    /// members/keyed arrays use tagged fallback; computational values refuse.
    pub fn to_json_with_budget(
        &self,
        document: &Document,
        budget: &mut zirql::Budget,
    ) -> Result<serde_json::Value, EvaluationError> {
        let range = match self {
            Self::Native(NativeValue::Constructed(value)) => value.range(),
            Self::Native(NativeValue::Captures(value)) => value.range(),
            Self::Native(NativeValue::Computational(value)) => value.range.clone(),
            Self::Native(NativeValue::Tuple(value)) => value.range.clone(),
            Self::Native(NativeValue::Sequence(value)) => value.range.clone(),
            _ => 0..0,
        };
        program::to_json(document, self, budget, &range)
    }
}

/// Native structured data before explicit rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeValue {
    /// An internal queue or identity set, retained for native library use.
    Computational(worklists::Computational),
    /// A positional tuple preserving native members.
    Tuple(worklists::Tuple),
    /// A positional sequence preserving its whole element declaration.
    Sequence(worklists::Sequence),
    Constructed(ConstructedReport),
    /// An owned JSON value exposed by the legacy map-entry view.
    JsonValue(serde_json::Value),
    /// An unchanged argument retains its shared owner across a native callback.
    Retained(std::sync::Arc<QueryOutput>),
    Ssa(ValueId),
    /// An ordered native operation field that remains a sequence at length one.
    OperationSequence(Vec<OperationId>),
    Capture(zirql::NativeCapture<QueryOutput>),
    Captures(CaptureMatches),
    String(String),
    Types(Vec<TypeId>),
    Attributes(Vec<(String, AttributeId)>),
    Map(Vec<(String, QueryOutput)>),
}

/// An owned parsing error at the responsible query byte position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryError {
    pub position: usize,
    pub message: String,
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
        if zirql::has_common_expression_syntax(source, |name| {
            matches!(name, "same_operation" | "user_info")
        }) {
            let work = source
                .len()
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or(QueryError {
                    position: 0,
                    message: "query allocation size overflow".to_owned(),
                })?;
            admission
                .charge(work, zirql::Location::Query(0..source.len()))
                .map_err(|_| QueryError {
                    position: 0,
                    message: "query work limit exceeded during parsing".to_owned(),
                })?;
            admission
                .charge_results(
                    source.len().saturating_add(1),
                    zirql::Location::Query(0..source.len()),
                )
                .map_err(|_| QueryError {
                    position: 0,
                    message: "result limit exceeded during parsing".to_owned(),
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
                message: message.to_owned(),
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
            functions: program.functions,
            function_limits: functions::DEFAULT_LIMITS,
            parameters: program.parameters,
            parameter_positions: program.parameter_positions,
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

    /// Set the active-call bound for the request's prepared shared expressions.
    pub fn with_function_limits(mut self, limits: FunctionLimits) -> Self {
        self.function_limits = limits;
        self
    }
    /// Whether shared syntax requires aggregate result accounting. Native-only
    /// requests keep their existing work and stream bounds.
    pub fn has_comprehensions(&self) -> bool {
        !self.functions.is_empty() || self.parse_results != 0
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
        emit: impl FnMut(&Document, QueryOutput, &mut zirql::Budget) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_parameters_and_budget(
            document,
            registry,
            document_name,
            options,
            ParameterInputs::Typed(&[]),
            budget,
            emit,
        )
    }

    /// Required declarations retained by this text program.
    pub fn parameters(&self) -> &[ParameterDeclaration] {
        &self.parameters
    }

    /// Evaluate typed caller inputs with the default request bounds.
    pub fn evaluate_with_parameters(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        parameters: &[(String, ConstructedValue)],
        mut emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        let limits = EvaluationLimits::default();
        let mut budget = if self.has_comprehensions() {
            Budget::for_program(ProgramLimits {
                work: limits.max_work,
                stream_items: limits.max_items,
                ..comprehension::default_limits()
            })
        } else {
            evaluation_budget(limits)
        };
        self.evaluate_with_parameters_and_budget(
            document,
            registry,
            None,
            EvaluationOptions::default(),
            ParameterInputs::Typed(parameters),
            &mut budget,
            |document, value, _| emit(document, value),
        )
    }

    /// Admit the full argument set, preflight and execute with one request allowance.
    /// Buffer library emissions if a later failure must discard earlier output.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_parameters_and_budget(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        document_name: Option<&str>,
        options: EvaluationOptions,
        parameters: ParameterInputs<'_>,
        budget: &mut Budget,
        mut emit: impl FnMut(&Document, QueryOutput, &mut Budget) -> Result<(), EvaluationError>,
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
        let admitted = program::admit_parameters(
            &self.parameters,
            parameters,
            &["document"],
            program::range(self.expression.range),
            budget,
        )?;
        program::admit_names(
            self.statements
                .iter()
                .map(|statement| match statement {
                    model::Statement::Binding { expression, .. }
                    | model::Statement::PatternBinding { expression, .. }
                    | model::Statement::Do(expression)
                    | model::Statement::Query(expression) => expression,
                })
                .chain(std::iter::once(&self.expression)),
            &self.functions,
            budget,
        )?;
        let function_host = report::ReportHost {
            functions: &self.functions,
        };
        program::charge_results(
            budget,
            5 * admitted.len(),
            &program::range(self.expression.range),
        )?;
        let parameter_names = admitted.iter().map(|(name, _)| name.clone()).collect();
        let program = program::lower(self, parameter_names);
        let mut names = BTreeMap::new();
        if document_name.is_some() {
            names.insert("document".to_owned(), program::Kind::Strings);
        }
        let unit = zirql::PreparedFunctions::prepare_patterns(
            self.functions.iter().map(|f| f.declaration.as_ref()),
            &function_host,
            budget,
        )
        .map_err(functions::error)?;
        let functions = zirql::PreparedProgram::prepare(
            unit,
            &program,
            &names,
            Some(program::Kind::Operations),
            self.function_limits,
            budget,
        )
        .map_err(functions::error)?;
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
            options,
            output: OutputMode::Textual,
            emit: &mut emit,
        };
        functions
            .evaluate(bindings, admitted, Some(input), &mut context, budget)
            .map_err(functions::error)?;
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

// One dispatcher owns all host stages, including those run inside stream segments.
fn evaluate_host_stage(
    stage: &model::Stage,
    context: &mut EvaluationContext<'_, '_>,
    mut current: QueryOutput,
    bindings: &zirql::Bindings<QueryOutput>,
    budget: &mut zirql::Budget,
) -> Result<QueryOutput, EvaluationError> {
    current = functions::owned(current, budget, &program::range(stage.range()))?;
    if let QueryOutput::Native(NativeValue::OperationSequence(operations)) = current {
        current = QueryOutput::Operations(operations);
    }
    match stage {
        model::Stage::FunctionInput { value, range } => {
            worklists::admit_input(value, context, budget, &program::range(*range))
                .map_err(program::evaluation_error)?;
            program::charge_results(budget, 1, &program::range(*range))?;
            current = QueryOutput::Native(NativeValue::Retained(std::sync::Arc::clone(value)));
        }
        model::Stage::PatternBinding { .. } => {
            return Err(EvaluationError::new(
                "pattern initialization is a program statement",
            ));
        }
        model::Stage::Native { .. }
        | model::Stage::Operand { .. }
        | model::Stage::Computed { .. }
        | model::Stage::Group { .. } => {
            return Err(EvaluationError::new(
                "native argument stages require prepared execution",
            ));
        }
        model::Stage::Structured { step, .. } => {
            current = evaluate_structured(step, context, current, bindings, budget)?;
        }
        model::Stage::Literal { value, range } => {
            let value = literal::evaluate(
                value,
                context.document,
                bindings,
                budget,
                &mut 0,
                &program::range(*range),
            )?;
            if value_depth(&value) > parser::DEFAULT_NESTING_LIMIT {
                return Err(EvaluationError::new("JSON nesting limit exceeded"));
            }
            current = match value {
                serde_json::Value::Object(entries) => {
                    let range = program::range(stage.range());
                    capture::slots::<(String, serde_json::Value)>(entries.len(), budget, &range)?;
                    charge(budget, entries.len())?;
                    program::charge_results(budget, entries.len() + 1, &range)?;
                    QueryOutput::Map(entries.into_iter().collect())
                }
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
        model::Stage::Value { .. } => {
            return Err(EvaluationError::new(
                "value is only valid as the selector in sort_by(value) on a map",
            ));
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
        model::Stage::Guard { .. } => {
            return Err(EvaluationError::new(
                "predicate guard is not an executable stage",
            ));
        }
        model::Stage::ClosureFixedPoint { .. } => {
            current = QueryOutput::Operations(evaluate_closure(
                context.document,
                take_operations(current, "closure")?,
                context.registry,
                true,
                budget,
            )?);
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
            current = QueryOutput::Operations(pipeline::evaluate_roots(
                predicate, selected, context, bindings, budget,
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
        model::Stage::Emit { .. }
        | model::Stage::Format { .. }
        | model::Stage::Markdown { .. }
        | model::Stage::Json { .. } => unreachable!("program emitters are shared expression steps"),
        model::Stage::Print { parts, range } => {
            let range = program::range(*range);
            let mut text = render::interpolate(parts, bindings, budget, &range)?;
            text.len()
                .checked_add(1)
                .filter(|bytes| *bytes <= isize::MAX as usize)
                .ok_or_else(|| {
                    EvaluationError::new("print storage limit exceeded").at(range.clone())
                })?;
            budget
                .charge(1, zirql::Location::Query(range.clone()))
                .map_err(core_error)?;
            program::charge_results(budget, 1, &range)?;
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

fn output_len(output: &QueryOutput) -> usize {
    match output {
        QueryOutput::Native(NativeValue::Retained(value)) => output_len(value),
        QueryOutput::Operations(values) => values.len(),
        QueryOutput::Values(values) => values.len(),
        QueryOutput::Map(values) => values.len(),
        QueryOutput::Array(values) => values.len(),
        QueryOutput::Native(NativeValue::OperationSequence(values)) => values.len(),
        QueryOutput::Native(NativeValue::Types(values)) => values.len(),
        QueryOutput::Native(NativeValue::Attributes(values)) => values.len(),
        QueryOutput::Native(NativeValue::Captures(values)) => values.len(),
        QueryOutput::Native(NativeValue::JsonValue(value)) => json_size(value),
        QueryOutput::Native(NativeValue::Sequence(value)) => value.values.len(),
        QueryOutput::Native(
            NativeValue::Computational(_)
            | NativeValue::Tuple(_)
            | NativeValue::Ssa(_)
            | NativeValue::Capture(_),
        ) => 1,
        QueryOutput::Native(NativeValue::Map(values)) => values.len(),
        QueryOutput::Native(NativeValue::Constructed(_))
        | QueryOutput::Native(NativeValue::String(_))
        | QueryOutput::Count(_)
        | QueryOutput::Json(_)
        | QueryOutput::Text(_)
        | QueryOutput::Formatted { .. } => 1,
    }
}

fn ordering_key(
    output: QueryOutput,
    budget: &mut zirql::Budget,
    range: &std::ops::Range<usize>,
) -> Result<zirql::NativeOrderKey, EvaluationError> {
    match output {
        QueryOutput::Native(NativeValue::Capture(mut record)) => {
            let (_, key) = record
                .fields
                .pop()
                .expect("ordering selectors retain one native field");
            ordering_key(key, budget, range)
        }
        QueryOutput::Count(value) => Ok(zirql::NativeOrderKey::Count(value)),
        QueryOutput::Native(NativeValue::String(value)) => Ok(zirql::NativeOrderKey::Text(value)),
        QueryOutput::Values(mut values) if values.len() == 1 => {
            Ok(zirql::NativeOrderKey::Text(values.pop().unwrap()))
        }
        QueryOutput::Native(NativeValue::Retained(value)) => {
            borrowed_ordering_key(&value, budget, range)
        }
        _ => Err(EvaluationError::new(
            "ordering selector must produce exactly one string or count per operation",
        )
        .at(range.clone())),
    }
}

fn borrowed_ordering_key(
    output: &QueryOutput,
    budget: &mut zirql::Budget,
    range: &std::ops::Range<usize>,
) -> Result<zirql::NativeOrderKey, EvaluationError> {
    let text = match functions::unretained(output) {
        QueryOutput::Native(NativeValue::Capture(record)) => {
            let (_, key) = record
                .fields
                .first()
                .expect("ordering selectors retain one native field");
            return borrowed_ordering_key(key, budget, range);
        }
        QueryOutput::Count(value) => return Ok(zirql::NativeOrderKey::Count(*value)),
        QueryOutput::Native(NativeValue::String(value)) => value,
        QueryOutput::Values(values) if values.len() == 1 => &values[0],
        _ => {
            return Err(EvaluationError::new(
                "ordering selector must produce exactly one string or count per operation",
            )
            .at(range.clone()));
        }
    };
    program::charge_bytes(budget, text.len(), range)?;
    program::charge_results(budget, 1, range)?;
    Ok(zirql::NativeOrderKey::Text(text.clone()))
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
        QueryOutput::Native(NativeValue::Tuple(value)) => {
            value.values.iter().fold(1usize, |size, value| {
                size.saturating_add(output_size(value))
            })
        }
        QueryOutput::Native(NativeValue::Sequence(value)) => {
            value.values.iter().fold(1usize, |size, value| {
                size.saturating_add(output_size(value))
            })
        }
        QueryOutput::Native(NativeValue::Constructed(value)) => value.size(),
        QueryOutput::Native(NativeValue::JsonValue(value)) => json_size(value),
        QueryOutput::Native(NativeValue::Map(entries)) => entries
            .iter()
            .map(|(_, value)| 1 + output_size(value))
            .sum(),
        QueryOutput::Map(entries) => entries.iter().map(|(_, value)| 1 + json_size(value)).sum(),
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
    budget: &mut zirql::Budget,
    range: &std::ops::Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let location = zirql::Location::Query(range.clone());
    match (left, right) {
        (
            QueryOutput::Operations(left)
            | QueryOutput::Native(NativeValue::OperationSequence(left)),
            QueryOutput::Operations(right)
            | QueryOutput::Native(NativeValue::OperationSequence(right)),
        ) => {
            for id in left.iter().chain(&right) {
                budget.charge(1, location.clone()).map_err(core_error)?;
                document
                    .check_operation(*id)
                    .map_err(|error| EvaluationError::new(error.to_string()))?;
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
            .map(QueryOutput::Operations)
            .map_err(program::evaluation_error)
        }
        (QueryOutput::Values(left), QueryOutput::Values(right)) => {
            let selected = zirql::combine_native_set(
                operator,
                left.iter().map(String::as_str),
                right.iter().map(String::as_str),
                left.iter().chain(&right).map(|text| Ok(text.as_str())),
                budget,
                location.clone(),
                |text, budget| Ok(budget.charge(text.len(), location.clone())?),
            )
            .map_err(program::evaluation_error)?;
            budget
                .charge(selected.len() + 1, location.clone())
                .map_err(core_error)?;
            program::charge_results(budget, selected.len() + 1, range)?;
            let mut values = Vec::with_capacity(selected.len());
            for text in selected {
                budget
                    .charge(text.len(), location.clone())
                    .map_err(core_error)?;
                values.push(text.to_owned());
            }
            Ok(QueryOutput::Values(values))
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
            NativeValue::JsonValue(value) => value.clone(),
            NativeValue::Retained(_)
            | NativeValue::Map(_)
            | NativeValue::Computational(_)
            | NativeValue::Tuple(_)
            | NativeValue::Sequence(_)
            | NativeValue::Ssa(_)
            | NativeValue::Capture(_)
            | NativeValue::Captures(_)
            | NativeValue::Constructed(_) => {
                unreachable!("constructed reports use bounded rendering")
            }
            NativeValue::OperationSequence(selected) => operation_json(document, selected),
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
        },
        QueryOutput::Operations(selected) => operation_json(document, selected),
        QueryOutput::Values(values) => serde_json::json!(values),
        QueryOutput::Map(values) => serde_json::Value::Object(values.iter().cloned().collect()),
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

fn operation_json(document: &Document, selected: &[OperationId]) -> serde_json::Value {
    serde_json::Value::Array(
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
    )
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

fn fixed_point_equal(
    left: &QueryOutput,
    right: &QueryOutput,
    budget: &mut zirql::Budget,
    range: &std::ops::Range<usize>,
) -> Result<bool, EvaluationError> {
    let location = zirql::Location::Query(range.clone());
    match (left, right) {
        (
            QueryOutput::Operations(left)
            | QueryOutput::Native(NativeValue::OperationSequence(left)),
            QueryOutput::Operations(right)
            | QueryOutput::Native(NativeValue::OperationSequence(right)),
        ) => zirql::ordered_native_equal(left, right, budget, location, |left, right, _| {
            Ok(left == right)
        })
        .map_err(program::evaluation_error),
        (QueryOutput::Values(left), QueryOutput::Values(right)) => zirql::ordered_native_equal(
            left,
            right,
            budget,
            location.clone(),
            |left, right, budget| {
                budget.charge(1, location.clone())?;
                if left.len() != right.len() {
                    return Ok(false);
                }
                for (left, right) in left.bytes().zip(right.bytes()) {
                    budget.charge(1, location.clone())?;
                    if left != right {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
        )
        .map_err(program::evaluation_error),
        _ => {
            budget.charge(1, location).map_err(core_error)?;
            Ok(false)
        }
    }
}
