//! Native SSA primitives and retained shared match trees.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::{ExpressionKind as X, Located, NativeCapture};

pub(super) fn path_operation(
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

/// A native operand retained as a shared value for scoped preparation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedOperand {
    pub(super) value: super::comprehension::ParsedComprehension,
    pub(super) expected: Option<NativeKind>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeKind {
    Operations,
    Ssa,
}
impl ParsedOperand {
    pub(super) fn new(expression: report::Operand) -> Self {
        let range = expression.range.clone();
        Self {
            value: super::comprehension::ParsedComprehension::new(Located::new(
                zirql::ValueExpressionKind::NativeValue(Box::new(expression)),
                range,
            )),
            expected: None,
        }
    }
    pub(super) fn field(subject: String, field: String, expected: NativeKind) -> Self {
        let mut value = Self::new(Located::new(
            X::Field {
                value: Box::new(Located::new(X::Binding(subject), 0..0)),
                field: Located::new(field, 0..0),
            },
            0..0,
        ));
        value.expected = Some(expected);
        value
    }
}

/// A parsed or typed shared native match. Hosts retain the common clause tree.
/// Ordered native capture records. Handles retain their original document identity.
/// The private field schema also describes an empty result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureMatches {
    pub(super) rows: Vec<NativeCapture<QueryOutput>>,
    pub(super) range: Range<usize>,
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

pub(super) fn operand_kind(
    value: &ParsedOperand,
    input: &program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &report::ReportHost<'_>,
) -> Result<program::Kind, EvaluationError> {
    let kind =
        if let zirql::ValueExpressionKind::NativeValue(query) = &value.value.expression.value {
            zirql::native_expression_kind(query, host, input, names, budget)
        } else {
            functions::value_kind(host, &value.value.expression, input.clone(), names, budget)
        }
        .map_err(core_error)?;
    if value.expected.is_some_and(|expected| match expected {
        NativeKind::Operations => kind != program::Kind::Operations,
        NativeKind::Ssa => kind != program::Kind::Ssa,
    }) {
        return Err(
            EvaluationError::new("capture field has an incompatible native kind").at(value
                .value
                .expression
                .range
                .clone()),
        );
    }
    Ok(kind)
}
pub(super) fn native_source(value: ParsedOperand, range: Range<usize>) -> report::Operand {
    Located::new(
        X::Sequence {
            input: Box::new(Located::new(X::Input, range.clone())),
            steps: vec![Located::new(
                zirql::ExpressionStep::Stage(Arc::new(model::Stage::Native {
                    value,
                    range: crate::source::TextRange::new(range.start as u32, range.end as u32)
                        .expect("ordered native source range"),
                })),
                range.clone(),
            )],
        },
        range,
    )
}
pub(super) fn operand_value(
    value: &QueryOutput,
    index: usize,
    context: &mut EvaluationContext<'_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let operations = match functions::unretained(value) {
        QueryOutput::Operations(operations)
        | QueryOutput::Native(NativeValue::OperationSequence(operations)) => operations,
        _ => return Err(EvaluationError::new("operand requires one operation").at(range.clone())),
    };
    let [operation] = operations.as_slice() else {
        return Err(EvaluationError::new("operand requires one operation").at(range.clone()));
    };
    let operation = *operation;
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
