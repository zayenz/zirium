//! Native SSA primitives and retained shared match trees.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::{Bindings, ExpressionKind as X, Located, MatchHost, NativeCapture};

/// A retained shared path query; cloning shares its immutable native tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedPaths {
    pub(super) paths: Arc<zirql::NativePaths<model::Stage, model::Predicate>>,
}
impl ParsedPaths {
    pub(super) fn new(paths: zirql::NativePaths<model::Stage, model::Predicate>) -> Self {
        Self {
            paths: Arc::new(paths),
        }
    }
}
impl Drop for ParsedPaths {
    fn drop(&mut self) {
        if let Some(paths) = Arc::get_mut(&mut self.paths) {
            let paths = std::mem::replace(
                paths,
                zirql::NativePaths {
                    edge: Located::new(
                        model::Stage::Users {
                            index: None,
                            range: crate::source::TextRange::new(0, 0).expect("empty range"),
                        },
                        0..0,
                    ),
                    target: Located::new(
                        zirql::PredicateKind::Leaf(model::Predicate::Bool {
                            value: false,
                            range: crate::source::TextRange::new(0, 0).expect("empty range"),
                        }),
                        0..0,
                    ),
                    max_depth: Located::new(0, 0..0),
                    range: 0..0,
                },
            );
            zirql::dispose_paths(paths);
        }
    }
}

pub(super) fn path_types(
    value: &ParsedPaths,
    input: &program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &report::ReportHost<'_>,
) -> Result<NativeCapture<program::Kind>, EvaluationError> {
    zirql::path_types(&value.paths, host, input, names, budget).map_err(core_error)
}
pub(super) fn evaluate_paths(
    value: &ParsedPaths,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<CaptureMatches, EvaluationError> {
    let range = &value.paths.range;
    let host = context
        .functions
        .map_or(&report::ReportHost, |session| session.host);
    let kind = report::output_kind(&input, budget, range)?;
    let names = report::runtime_kinds(bindings, budget, range)?;
    let schema = Arc::new(path_types(value, &kind, &names, budget, host)?);
    let values = zirql::evaluate_paths(&value.paths, host, input, context, bindings, budget)
        .map_err(program::evaluation_error)?;
    path_matches(values, schema, budget, range)
}
pub(super) fn evaluate_paths_with_arguments<'q, 'f>(
    value: &ParsedPaths,
    host: &report::ReportHost<'f>,
    input: QueryOutput,
    arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_, 'q>, report::ReportHost<'f>>,
    budget: &mut Budget,
) -> Result<CaptureMatches, EvaluationError> {
    let range = &value.paths.range;
    let schema = {
        let (_, bindings) = arguments.parts();
        let kind = report::output_kind(&input, budget, range)?;
        let names = report::runtime_kinds(bindings, budget, range)?;
        Arc::new(path_types(value, &kind, &names, budget, host)?)
    };
    let values = zirql::evaluate_paths_with_arguments(&value.paths, host, input, arguments, budget)
        .map_err(program::evaluation_error)?;
    path_matches(values, schema, budget, range)
}
fn path_matches(
    values: Vec<QueryOutput>,
    schema: Arc<NativeCapture<program::Kind>>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<CaptureMatches, EvaluationError> {
    slots::<NativeCapture<QueryOutput>>(values.len(), budget, range)?;
    program::charge_results(budget, values.len() + 1, range)?;
    let rows = values
        .into_iter()
        .map(|value| {
            let QueryOutput::Native(NativeValue::Capture(record)) = value else {
                unreachable!("native path record")
            };
            record
        })
        .collect();
    Ok(CaptureMatches {
        rows,
        range: range.clone(),
        schema,
    })
}

fn path_operation(
    value: &QueryOutput,
    document: &Document,
    range: &Range<usize>,
) -> Result<OperationId, EvaluationError> {
    let QueryOutput::Operations(ids) = value else {
        return Err(EvaluationError::new("paths require native operations").at(range.clone()));
    };
    let [id] = ids.as_slice() else {
        return Err(EvaluationError::new("path subject requires one operation").at(range.clone()));
    };
    document
        .check_operation(*id)
        .map_err(|error| EvaluationError::new(error.to_string()).at(range.clone()))?;
    Ok(*id)
}

impl<'q> zirql::PathHost<EvaluationContext<'_, '_, 'q>> for report::ReportHost<'_> {
    type Target = model::Predicate;
    fn validate_path_edge(
        &self,
        edge: &model::Stage,
        input: &program::Kind,
        _: &BTreeMap<String, program::Kind>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        if *input == program::Kind::Operations
            && matches!(edge, model::Stage::Users { index: None, .. })
        {
            return Ok(());
        }
        Err(zirql::QueryError::query(
            range.clone(),
            "users path edge requires native operations",
        ))
    }
    fn validate_target_leaf(
        &self,
        target: &model::Predicate,
        _: &program::Kind,
        names: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        _: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        report::validate_predicate_with_host(target, names, budget, self)
            .map_err(report::query_error)
    }
    fn evaluate_target_leaf(
        &self,
        target: &model::Predicate,
        input: &QueryOutput,
        context: &mut EvaluationContext<'_, '_, 'q>,
        bindings: &Bindings<QueryOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::Truth, zirql::EvaluationError<EvaluationError>> {
        let operation =
            path_operation(input, context.document, range).map_err(zirql::EvaluationError::Host)?;
        let target = report::resolve_predicate(target, context, bindings, budget)
            .map_err(zirql::EvaluationError::Host)?;
        Ok(pipeline::evaluate_leaf(
            &target,
            context.document,
            operation,
        ))
    }
    fn evaluate_target_leaf_with_arguments(
        &self,
        target: &model::Predicate,
        input: &QueryOutput,
        arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_, 'q>, Self>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::Truth, zirql::EvaluationError<EvaluationError>> {
        let operation = {
            let (context, _) = arguments.parts();
            path_operation(input, context.document, range).map_err(zirql::EvaluationError::Host)?
        };
        let target = report::resolve_prepared_predicate(target, arguments, budget)
            .map_err(zirql::EvaluationError::Host)?;
        let (context, _) = arguments.parts();
        Ok(pipeline::evaluate_leaf(
            &target,
            context.document,
            operation,
        ))
    }
    fn path_neighbors(
        &self,
        _: &model::Stage,
        input: &QueryOutput,
        context: &mut EvaluationContext<'_, '_, 'q>,
        _: &Bindings<QueryOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
        let operation =
            path_operation(input, context.document, range).map_err(zirql::EvaluationError::Host)?;
        let neighbors = evaluate_users(context.document, &[operation], None, budget)
            .map_err(zirql::EvaluationError::Host)?;
        slots::<QueryOutput>(neighbors.len(), budget, range)
            .map_err(zirql::EvaluationError::Host)?;
        budget.charge_results(
            neighbors.len() * 2 + 1,
            zirql::Location::Query(range.clone()),
        )?;
        let mut values = Vec::with_capacity(neighbors.len());
        for operation in neighbors {
            context
                .document
                .check_operation(operation)
                .map_err(|error| {
                    zirql::EvaluationError::Host(
                        EvaluationError::new(error.to_string()).at(range.clone()),
                    )
                })?;
            values.push(QueryOutput::Operations(vec![operation]));
        }
        Ok(values)
    }
    fn same_path_identity(
        &self,
        left: &QueryOutput,
        right: &QueryOutput,
        context: &mut EvaluationContext<'_, '_, 'q>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<bool, zirql::EvaluationError<EvaluationError>> {
        Ok(
            path_operation(left, context.document, range).map_err(zirql::EvaluationError::Host)?
                == path_operation(right, context.document, range)
                    .map_err(zirql::EvaluationError::Host)?,
        )
    }
    fn path_steps(
        &self,
        steps: Vec<QueryOutput>,
        context: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        slots::<OperationId>(steps.len(), budget, range).map_err(zirql::EvaluationError::Host)?;
        budget.charge_results(steps.len() + 1, zirql::Location::Query(range.clone()))?;
        let mut operations = Vec::with_capacity(steps.len());
        for step in steps {
            operations.push(
                path_operation(&step, context.document, range)
                    .map_err(zirql::EvaluationError::Host)?,
            );
        }
        Ok(QueryOutput::Native(NativeValue::OperationSequence(
            operations,
        )))
    }
}

/// A retained normalized native operand. Clone shares its immutable tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedOperand {
    pub(super) expression: Arc<report::Operand>,
    pub(super) expected: Option<NativeKind>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeKind {
    Operations,
    Ssa,
}
impl ParsedOperand {
    pub(super) fn new(expression: report::Operand) -> Self {
        Self {
            expression: Arc::new(expression),
            expected: None,
        }
    }
    pub(super) fn field(subject: String, field: String, expected: NativeKind) -> Self {
        Self {
            expression: Arc::new(Located::new(
                X::Field {
                    value: Box::new(Located::new(X::Binding(subject), 0..0)),
                    field: Located::new(field, 0..0),
                },
                0..0,
            )),
            expected: Some(expected),
        }
    }
}
impl Drop for ParsedOperand {
    fn drop(&mut self) {
        if let Some(expression) = Arc::get_mut(&mut self.expression) {
            let expression = std::mem::replace(expression, Located::new(X::Input, 0..0));
            zirql::dispose_expression(expression);
        }
    }
}

/// A parsed or typed shared native match. Hosts retain the common clause tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedMatch {
    pub(super) matching: Arc<zirql::NativeMatch<report::Operand, model::Predicate>>,
}
impl ParsedMatch {
    pub(super) fn new(matching: zirql::NativeMatch<report::Operand, model::Predicate>) -> Self {
        Self {
            matching: Arc::new(matching),
        }
    }
}
impl Drop for ParsedMatch {
    fn drop(&mut self) {
        if let Some(matching) = Arc::get_mut(&mut self.matching) {
            let matching = std::mem::replace(
                matching,
                zirql::NativeMatch {
                    clauses: vec![],
                    range: 0..0,
                },
            );
            zirql::dispose_match(matching);
        }
    }
}

/// Ordered native capture records. Handles retain their original document identity.
/// The private field schema also describes an empty result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureMatches {
    rows: Vec<NativeCapture<QueryOutput>>,
    range: Range<usize>,
    pub(super) schema: Arc<NativeCapture<program::Kind>>,
}
impl CaptureMatches {
    pub fn rows(&self) -> &[NativeCapture<QueryOutput>] {
        &self.rows
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub(super) fn into_rows(self) -> Vec<NativeCapture<QueryOutput>> {
        self.rows
    }
    pub(super) fn copy(
        &self,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Self, EvaluationError> {
        slots::<NativeCapture<QueryOutput>>(self.rows.len(), budget, range)?;
        program::charge_results(budget, 1 + self.rows.len(), range)?;
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            rows.push(copy_record(row, budget, range)?);
        }
        Ok(Self {
            rows,
            range: self.range.clone(),
            schema: Arc::clone(&self.schema),
        })
    }
}

pub(super) fn slots<T>(
    count: usize,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    if count
        .checked_mul(std::mem::size_of::<T>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(
            EvaluationError::new("native capture allocation size overflow").at(range.clone()),
        );
    }
    program::charge_bytes(budget, count, range)
}
pub(super) fn copy_record(
    record: &NativeCapture<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<NativeCapture<QueryOutput>, EvaluationError> {
    slots::<(String, QueryOutput)>(record.fields.len(), budget, range)?;
    program::charge_results(budget, 1 + record.fields.len(), range)?;
    let mut fields = Vec::with_capacity(record.fields.len());
    for (name, value) in &record.fields {
        program::charge_bytes(budget, name.len(), range)?;
        fields.push((name.clone(), program::copy_value(value, budget, range)?));
    }
    Ok(NativeCapture { fields })
}

pub(super) fn types(
    value: &ParsedMatch,
    input: &program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<NativeCapture<program::Kind>, EvaluationError> {
    zirql::match_types(&value.matching, &report::ReportHost, input, names, budget)
        .map_err(core_error)
}
pub(super) fn operand_kind(
    value: &ParsedOperand,
    input: &program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<program::Kind, EvaluationError> {
    let kind =
        zirql::native_expression_kind(&value.expression, &report::ReportHost, input, names, budget)
            .map_err(core_error)?;
    if value.expected.is_some_and(|expected| match expected {
        NativeKind::Operations => kind != program::Kind::Operations,
        NativeKind::Ssa => kind != program::Kind::Ssa,
    }) {
        return Err(
            EvaluationError::new("capture field has an incompatible native kind")
                .at(value.expression.range.clone()),
        );
    }
    Ok(kind)
}
pub(super) fn evaluate(
    value: &ParsedMatch,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<CaptureMatches, EvaluationError> {
    let range = &value.matching.range;
    let input_kind = report::output_kind(&input, budget, range)?;
    let names = report::runtime_kinds(bindings, budget, range)?;
    let schema = Arc::new(types(value, &input_kind, &names, budget)?);
    let values = zirql::evaluate_match(
        &value.matching,
        &report::ReportHost,
        input,
        context,
        bindings,
        budget,
    )
    .map_err(program::evaluation_error)?;
    slots::<NativeCapture<QueryOutput>>(values.len(), budget, range)?;
    program::charge_results(budget, values.len() + 1, range)?;
    let mut rows = Vec::with_capacity(values.len());
    for value in values {
        let QueryOutput::Native(NativeValue::Capture(record)) = value else {
            return Err(EvaluationError::new("unexpected native capture result"));
        };
        rows.push(record);
    }
    Ok(CaptureMatches {
        rows,
        range: range.clone(),
        schema,
    })
}
pub(super) fn evaluate_operand(
    value: &ParsedOperand,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<QueryOutput, EvaluationError> {
    zirql::evaluate_expression(
        &value.expression,
        &report::ReportHost,
        input,
        context,
        bindings,
        budget,
    )
    .map_err(program::evaluation_error)
}
pub(super) fn operand(
    operation: &ParsedOperand,
    index: usize,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let value = evaluate_operand(operation, context, input, bindings, budget)?;
    let QueryOutput::Operations(operations) = value else {
        return Err(EvaluationError::new("operand requires one operation").at(range.clone()));
    };
    let operation = zirql::take_one(operations, range.clone()).map_err(core_error)?;
    context
        .document
        .check_operation(operation)
        .map_err(|error| EvaluationError::new(error.to_string()).at(range.clone()))?;
    budget
        .charge(1, zirql::Location::Query(range.clone()))
        .map_err(core_error)?;
    if context.document.operation_is_unparsed(operation) == Some(true) {
        return Err(EvaluationError::new("operand requires parsed SSA operands").at(range.clone()));
    }
    let reference = context
        .document
        .operands(operation)
        .and_then(|values| values.get(index))
        .ok_or_else(|| {
            EvaluationError::new("required SSA operand is unavailable").at(range.clone())
        })?;
    let ValueReference::Resolved(value) = *reference else {
        return Err(EvaluationError::new("required SSA operand is unresolved").at(range.clone()));
    };
    check_value(context.document, value, range)?;
    program::charge_results(budget, 1, range)?;
    Ok(QueryOutput::Native(NativeValue::Ssa(value)))
}
pub(super) fn check_value(
    document: &Document,
    value: ValueId,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    document
        .check_value(value)
        .map_err(|error| EvaluationError::new(error.to_string()).at(range.clone()))?;
    if let ValueId::BlockArgument { block, .. } = value {
        let owner = document
            .block(block)
            .and_then(|block| document.region(block.parent_region()))
            .map(|region| region.parent_operation())
            .ok_or_else(|| {
                EvaluationError::new("SSA block argument owner is unavailable").at(range.clone())
            })?;
        document
            .check_operation(owner)
            .map_err(|error| EvaluationError::new(error.to_string()).at(range.clone()))?;
    }
    Ok(())
}
impl<'q> MatchHost<EvaluationContext<'_, '_, 'q>> for report::ReportHost<'_> {
    type Guard = model::Predicate;
    fn validate_guard_leaf(
        &self,
        guard: &model::Predicate,
        input: &program::Kind,
        scope: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        match guard {
            model::Predicate::Bool { .. } => Ok(()),
            model::Predicate::SameValue { left, right, .. } => {
                for value in [left, right] {
                    let kind =
                        operand_kind(value, input, scope, budget).map_err(report::query_error)?;
                    if kind != program::Kind::Ssa {
                        return Err(zirql::QueryError::query(
                            value.expression.range.clone(),
                            "same_value requires native SSA values",
                        ));
                    }
                }
                Ok(())
            }
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "unsupported native match guard",
            )),
        }
    }
    fn evaluate_guard_leaf(
        &self,
        guard: &model::Predicate,
        input: &QueryOutput,
        context: &mut EvaluationContext<'_, '_, 'q>,
        bindings: &Bindings<QueryOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::Truth, zirql::EvaluationError<EvaluationError>> {
        let equal = match guard {
            model::Predicate::Bool { value, .. } => *value,
            model::Predicate::SameValue { left, right, .. } => {
                let mut resolve = |query: &ParsedOperand| -> Result<ValueId, zirql::EvaluationError<EvaluationError>> {
                    let input = program::copy_value(input, budget, range).map_err(zirql::EvaluationError::Host)?;
                    let output = evaluate_operand(query, context, input, bindings, budget).map_err(zirql::EvaluationError::Host)?;
                    let QueryOutput::Native(NativeValue::Ssa(value)) = output else {
                        return Err(zirql::QueryError::query(range.clone(), "same_value requires native SSA values").into());
                    };
                    check_value(context.document, value, range).map_err(zirql::EvaluationError::Host)?;
                    Ok(value)
                };
                resolve(left)? == resolve(right)?
            }
            _ => {
                return Err(zirql::QueryError::query(
                    range.clone(),
                    "unsupported native match guard",
                )
                .into());
            }
        };
        Ok(if equal {
            zirql::Truth::True
        } else {
            zirql::Truth::False
        })
    }
}
