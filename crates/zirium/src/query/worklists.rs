//! Native wrappers for the shared computational runtime.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::{DeclaredType as D, ProgramHost};

/// An opaque computational result. Cloning retains immutable shared storage.
#[derive(Clone, Debug)]
pub struct Computational {
    pub(super) value: Arc<zirql::ComputationalValue<QueryOutput, program::Kind>>,
    pub(super) range: Range<usize>,
}
impl PartialEq for Computational {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
}
impl Eq for Computational {}
/// A native tuple whose members keep their original owners and locations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tuple {
    pub(super) values: Arc<Vec<QueryOutput>>,
    pub(super) range: Range<usize>,
}
/// A native sequence, including the element declaration of an empty result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub(super) values: Arc<Vec<QueryOutput>>,
    pub(super) element: Arc<D<program::Kind>>,
    pub(super) range: Range<usize>,
}

pub(super) fn share(
    value: &QueryOutput,
    host: &report::ReportHost<'_>,
    context: &mut EvaluationContext<'_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<Option<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
    let QueryOutput::Native(NativeValue::Computational(value)) = functions::unretained(value)
    else {
        return Ok(None);
    };
    let shared = value
        .value
        .share(host, context, budget, ValueLocation::Query(range.clone()))?;
    Ok(Some(QueryOutput::Native(NativeValue::Computational(
        Computational {
            value: Arc::new(shared),
            range: value.range.clone(),
        },
    ))))
}

// Explicit builder inputs can carry retained values from an earlier run.
// Check their computational owners once at that boundary, including nested roots.
pub(super) fn admit_input(
    value: &QueryOutput,
    context: &mut EvaluationContext<'_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<(), zirql::EvaluationError<EvaluationError>> {
    let location = ValueLocation::Query(range.clone());
    capture::slots::<&QueryOutput>(1, budget, range).map_err(zirql::EvaluationError::Host)?;
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        budget.charge(1, location.clone())?;
        match functions::unretained(value) {
            QueryOutput::Native(NativeValue::Computational(computational)) => {
                let declaration = computational.value.declaration(budget, &location)?;
                zirql::admit_computational(
                    value,
                    &declaration,
                    &report::ReportHost,
                    context,
                    budget,
                    location.clone(),
                )?;
            }
            QueryOutput::Native(NativeValue::Tuple(tuple)) => {
                capture::slots::<&QueryOutput>(tuple.values.len(), budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                pending.extend(tuple.values.iter());
            }
            QueryOutput::Native(NativeValue::Sequence(sequence)) => {
                budget.check_items(sequence.values.len(), location.clone())?;
                capture::slots::<&QueryOutput>(sequence.values.len(), budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                pending.extend(sequence.values.iter());
            }
            QueryOutput::Native(NativeValue::Capture(record)) => {
                capture::slots::<&QueryOutput>(record.fields.len(), budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                pending.extend(record.fields.iter().map(|(_, value)| value));
            }
            QueryOutput::Native(NativeValue::Captures(records)) => {
                for record in records.rows() {
                    capture::slots::<&QueryOutput>(record.fields.len(), budget, range)
                        .map_err(zirql::EvaluationError::Host)?;
                    pending.extend(record.fields.iter().map(|(_, value)| value));
                }
            }
            QueryOutput::Native(NativeValue::Map(record)) => {
                capture::slots::<&QueryOutput>(record.len(), budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                pending.extend(record.iter().map(|(_, value)| value));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn sequence_items(
    host: &report::ReportHost<'_>,
    value: &QueryOutput,
    element: &D<program::Kind>,
    context: &mut EvaluationContext<'_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<Vec<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
    let location = ValueLocation::Query(range.clone());
    match functions::unretained(value) {
        QueryOutput::Native(NativeValue::Sequence(sequence)) => {
            if sequence.element.as_ref() != element {
                return Err(zirql::QueryError::query(
                    range.clone(),
                    "incompatible native sequence element declaration",
                )
                .into());
            }
            budget.check_items(sequence.values.len(), location.clone())?;
            capture::slots::<QueryOutput>(sequence.values.len(), budget, range)
                .map_err(zirql::EvaluationError::Host)?;
            sequence
                .values
                .iter()
                .map(|v| host.copy_value(v, context, budget, range))
                .collect()
        }
        QueryOutput::Operations(operations)
        | QueryOutput::Native(NativeValue::OperationSequence(operations))
            if matches!(element, D::Native(program::Kind::Operations)) =>
        {
            budget.check_items(operations.len(), location)?;
            capture::slots::<QueryOutput>(operations.len(), budget, range)
                .map_err(zirql::EvaluationError::Host)?;
            let mut values = Vec::with_capacity(operations.len());
            for &operation in operations {
                budget.charge(1, ValueLocation::Query(range.clone()))?;
                context.document.check_operation(operation).map_err(|e| {
                    zirql::EvaluationError::Host(
                        EvaluationError::new(e.to_string()).at(range.clone()),
                    )
                })?;
                values.push(QueryOutput::Operations(vec![operation]));
            }
            Ok(values)
        }
        QueryOutput::Native(NativeValue::Captures(records)) => {
            budget.check_items(records.len(), location)?;
            capture::slots::<QueryOutput>(records.len(), budget, range)
                .map_err(zirql::EvaluationError::Host)?;
            records
                .rows()
                .iter()
                .map(|record| {
                    let value = QueryOutput::Native(NativeValue::Capture(
                        capture::copy_record(record, budget, range)
                            .map_err(zirql::EvaluationError::Host)?,
                    ));
                    zirql::admit_value(
                        &value,
                        element,
                        host,
                        context,
                        budget,
                        ValueLocation::Query(range.clone()),
                    )?;
                    Ok(value)
                })
                .collect()
        }
        _ => Err(zirql::QueryError::query(
            range.clone(),
            "expected an eager native positional sequence",
        )
        .into()),
    }
}

pub(super) fn identity(
    value: &QueryOutput,
    expected: &program::Kind,
    context: &mut EvaluationContext<'_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<zirql::NativeIdentity, zirql::EvaluationError<EvaluationError>> {
    budget.charge(4, ValueLocation::Query(range.clone()))?;
    let document = &context.document;
    let key = match (expected, functions::unretained(value)) {
        (program::Kind::Operations, QueryOutput::Operations(operations))
            if operations.len() == 1 =>
        {
            u128::from(document.operation_key(operations[0]).map_err(|e| {
                zirql::EvaluationError::Host(EvaluationError::new(e.to_string()).at(range.clone()))
            })?)
        }
        (program::Kind::Ssa, QueryOutput::Native(NativeValue::Ssa(value))) => {
            document.check_value(*value).map_err(|e| {
                zirql::EvaluationError::Host(EvaluationError::new(e.to_string()).at(range.clone()))
            })?;
            document.value_key(*value).ok_or_else(|| {
                zirql::QueryError::query(range.clone(), "invalid SSA value identity")
            })?
        }
        _ => {
            return Err(zirql::QueryError::query(
                range.clone(),
                "identity requires one Operation or SSAValue",
            )
            .into());
        }
    };
    let owner = document.owner_identity();
    Ok(zirql::NativeIdentity([
        owner as u64,
        (owner >> 64) as u64,
        key as u64,
        (key >> 64) as u64,
    ]))
}

impl Computational {
    /// Number of retained queue elements or distinct identities.
    pub fn len(&self) -> usize {
        match self.value.as_ref() {
            zirql::ComputationalValue::Queue(value) => value.len(),
            zirql::ComputationalValue::IdentitySet(value) => value.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}
impl Tuple {
    /// Borrow native members without materializing them.
    pub fn values(&self) -> &[QueryOutput] {
        &self.values
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}
impl Sequence {
    /// Borrow immutable post-step snapshots or other native members.
    pub fn values(&self) -> &[QueryOutput] {
        &self.values
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

// Retained declarations include empty element facts. Admit their metadata before
// cloning it; the shared runtime remains responsible for type compatibility.
pub(super) fn declaration(
    value: &D<program::Kind>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<Arc<D<program::Kind>>, zirql::QueryError> {
    let location = ValueLocation::Query(range.clone());
    budget.charge_results(1, location.clone())?;
    Ok(Arc::new(zirql::copy_declaration(value, budget, &location)?))
}
