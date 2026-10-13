//! Native SSA primitives and retained shared match trees.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::{Bindings, ExpressionKind as X, Located, MatchHost, NativeCapture};

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
impl<'q> MatchHost<EvaluationContext<'_, '_, 'q>> for report::ReportHost {
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
