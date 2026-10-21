//! Typed pure functions retain shared trees and document-owned native handles.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::{
    DeclaredType as D, FunctionHost, ProgramHost, ProjectionHost, RuntimeTypeHost, ValueType,
};

pub use zirql::{FunctionLimits, FunctionParameter, Pattern, PatternKind, PatternParameter};
/// Native signature identities; `SSAValue` is Zirium's checked `ValueId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionNativeType {
    Operation,
    SSAValue,
}
pub type FunctionType = D<FunctionNativeType>;
pub(super) type Declaration = zirql::PatternFunctionDeclaration<report::Operand, program::Kind>;
pub(super) const DEFAULT_LIMITS: FunctionLimits = FunctionLimits { active_depth: 64 };

/// A located function definition shared through typed query composition.
#[derive(Debug)]
pub struct FunctionDefinition {
    pub(super) declaration: Arc<Declaration>,
}
impl Clone for FunctionDefinition {
    fn clone(&self) -> Self {
        Self {
            declaration: Arc::clone(&self.declaration),
        }
    }
}
impl PartialEq for FunctionDefinition {
    fn eq(&self, other: &Self) -> bool {
        self.declaration == other.declaration
    }
}
impl Eq for FunctionDefinition {}
impl Drop for FunctionDefinition {
    fn drop(&mut self) {
        if let Some(d) = Arc::get_mut(&mut self.declaration) {
            for parameter in std::mem::take(&mut d.parameters) {
                zirql::dispose_pattern(parameter.value.pattern);
            }
            let value = std::mem::replace(
                &mut d.value.value,
                zirql::ValueExpressionKind::Literal(ConstructedValue::Absent),
            );
            report::dispose_value(value);
        }
    }
}
impl FunctionDefinition {
    pub(super) fn parsed(declaration: Declaration) -> Self {
        Self {
            declaration: Arc::new(declaration),
        }
    }
}
pub(super) fn merge(
    left: &[FunctionDefinition],
    right: &[FunctionDefinition],
) -> Vec<FunctionDefinition> {
    let mut result = left.to_vec();
    for function in right {
        if !result
            .iter()
            .any(|f| Arc::ptr_eq(&f.declaration, &function.declaration))
        {
            result.push(function.clone());
        }
    }
    result
}
pub(super) fn resolve_type(
    name: Located<&str>,
    _: &mut Budget,
) -> Result<D<program::Kind>, zirql::QueryError> {
    Ok(D::Native(match name.value {
        "Operation" => program::Kind::Operations,
        "SSAValue" => program::Kind::Ssa,
        _ => {
            return Err(zirql::QueryError::query(
                name.range,
                "unknown function signature type",
            ));
        }
    }))
}

pub(super) fn scalar_facts(
    ty: &D<program::Kind>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<Vec<ValueType>, zirql::QueryError> {
    {
        budget.charge(1, ValueLocation::Query(range.clone()))?;
        budget.charge_results(1, ValueLocation::Query(range.clone()))?;
        Ok(match ty {
            D::Value => vec![ValueType::Unestablished],
            D::Dictionary(key, value) => vec![ValueType::Dictionary {
                keys: scalar_facts(key, budget, range)?,
                values: scalar_facts(value, budget, range)?,
            }],
            D::Boolean => vec![ValueType::Boolean],
            D::Integer => vec![ValueType::Integer],
            D::Float => vec![ValueType::Float],
            D::String => vec![ValueType::String],
            D::Optional(value) => {
                let mut facts = scalar_facts(value, budget, range)?;
                facts.push(ValueType::Absent);
                facts
            }
            D::Sequence(value) | D::Array(value) => vec![ValueType::Array(
                zirql::ArrayType::Elements(scalar_facts(value, budget, range)?),
            )],
            D::Tuple(values) => vec![ValueType::Tuple(
                values
                    .iter()
                    .map(|v| scalar_facts(v, budget, range).map(|mut t| t.remove(0)))
                    .collect::<Result<_, _>>()?,
            )],
            D::Record(fields) => vec![ValueType::Record(
                fields
                    .iter()
                    .map(|(n, v)| {
                        budget.charge(n.len() + 1, ValueLocation::Query(range.clone()))?;
                        Ok((n.clone(), scalar_facts(v, budget, range)?.remove(0)))
                    })
                    .collect::<Result<_, zirql::QueryError>>()?,
            )],
            D::Set(element) => vec![ValueType::Set(Some(*element))],
            D::Member(id) => vec![ValueType::Member(*id)],
            D::Keyed(value) => vec![ValueType::KeyedArray(zirql::KeyedArrayType {
                axes: None,
                elements: scalar_facts(value, budget, range)?,
            })],
            D::Native(_) => vec![ValueType::Unestablished],
            D::Queue(_) | D::IdentitySet(_) => vec![ValueType::Unestablished],
        })
    }
}
impl<'c, 'a, 'q> FunctionHost<EvaluationContext<'c, 'a, 'q>> for report::ReportHost<'_> {
    fn native_function(
        &self,
        name: &str,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Option<zirql::NativeFunction<EvaluationContext<'c, 'a, 'q>, Self>>, zirql::QueryError>
    {
        native_function(name, budget, range)
    }
    fn retains_native_admission(&self) -> bool {
        true
    }
    fn resolve_type_name(
        &self,
        name: Located<&str>,
        budget: &mut Budget,
    ) -> Result<D<program::Kind>, zirql::QueryError> {
        budget.charge(
            name.value.len() + 1,
            ValueLocation::Query(name.range.clone()),
        )?;
        resolve_type(name, budget)
    }
    fn prepare_computational(
        &self,
        declaration: Option<&D<program::Kind>>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        budget.charge(1, ValueLocation::Query(range.clone()))?;
        if let Some(declaration) = declaration {
            if matches!(declaration, D::IdentitySet(kind) if !matches!(kind, program::Kind::Operations | program::Kind::Ssa))
            {
                return Err(zirql::QueryError::query(
                    range.clone(),
                    "identity_set requires Operation or SSAValue",
                ));
            }
            self.declaration_kind(declaration, budget, range)?;
        }
        Ok(())
    }
    fn computational_value<'v>(
        &self,
        value: &'v QueryOutput,
    ) -> Option<&'v zirql::ComputationalValue<QueryOutput, program::Kind>> {
        match unretained(value) {
            QueryOutput::Native(NativeValue::Computational(value)) => Some(&value.value),
            _ => None,
        }
    }
    fn lift_computational(
        &self,
        value: zirql::ComputationalValue<QueryOutput, program::Kind>,
        _: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        budget.charge_results(1, ValueLocation::Query(range.clone()))?;
        Ok(QueryOutput::Native(NativeValue::Computational(
            worklists::Computational {
                value: Arc::new(value),
                range: range.clone(),
            },
        )))
    }
    fn construct_tuple(
        &self,
        values: Vec<QueryOutput>,
        _: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        budget.charge_results(values.len() + 1, ValueLocation::Query(range.clone()))?;
        Ok(QueryOutput::Native(NativeValue::Tuple(worklists::Tuple {
            values: Arc::new(values),
            range: range.clone(),
        })))
    }
    fn sequence_items(
        &self,
        value: &QueryOutput,
        element: &D<program::Kind>,
        context: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
        worklists::sequence_items(self, value, element, context, budget, range)
    }
    fn construct_sequence(
        &self,
        values: Vec<QueryOutput>,
        element: &D<program::Kind>,
        _: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        budget.charge_results(values.len() + 1, ValueLocation::Query(range.clone()))?;
        Ok(QueryOutput::Native(NativeValue::Sequence(
            worklists::Sequence {
                values: Arc::new(values),
                element: worklists::declaration(element, budget, range)?,
                range: range.clone(),
            },
        )))
    }
    fn worklist_context(
        &self,
        context: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeContextId, zirql::EvaluationError<EvaluationError>> {
        budget.charge(1, ValueLocation::Query(range.clone()))?;
        let owner = context.document.owner_identity();
        Ok(zirql::NativeContextId([
            owner as u64,
            (owner >> 64) as u64,
            context.document.revision(),
        ]))
    }
    fn native_identity(
        &self,
        value: &QueryOutput,
        expected: &program::Kind,
        context: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeIdentity, zirql::EvaluationError<EvaluationError>> {
        worklists::identity(value, expected, context, budget, range)
    }

    fn declaration_kind(
        &self,
        ty: &D<program::Kind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        {
            budget.charge(1, ValueLocation::Query(range.clone()))?;
            Ok(match ty {
                D::Native(kind) => kind.clone(),
                D::Queue(_) | D::IdentitySet(_) => {
                    program::Kind::Computational(worklists::declaration(ty, budget, range)?)
                }
                D::Tuple(values)
                    if zirql::contains_native(
                        ty,
                        budget,
                        &ValueLocation::Query(range.clone()),
                    )? =>
                {
                    capture::slots::<program::Kind>(values.len(), budget, range)
                        .map_err(report::query_error)?;
                    program::Kind::Tuple(Arc::new(
                        values
                            .iter()
                            .map(|v| self.declaration_kind(v, budget, range))
                            .collect::<Result<_, _>>()?,
                    ))
                }
                D::Sequence(value) | D::Array(value)
                    if matches!(value.as_ref(), D::Native(program::Kind::Operations)) =>
                {
                    program::Kind::Operations
                }
                D::Sequence(value) | D::Array(value)
                    if zirql::contains_native(
                        value,
                        budget,
                        &ValueLocation::Query(range.clone()),
                    )? =>
                {
                    program::Kind::Sequence(worklists::declaration(value, budget, range)?)
                }
                D::Optional(value)
                    if zirql::contains_native(
                        value,
                        budget,
                        &ValueLocation::Query(range.clone()),
                    )? =>
                {
                    self.declaration_kind(value, budget, range)?
                }
                D::Record(fields)
                    if zirql::contains_native(
                        ty,
                        budget,
                        &ValueLocation::Query(range.clone()),
                    )? =>
                {
                    capture::slots::<(String, program::Kind)>(fields.len(), budget, range)
                        .map_err(report::query_error)?;
                    budget.charge_results(fields.len() + 1, ValueLocation::Query(range.clone()))?;
                    let mut kinds = Vec::with_capacity(fields.len());
                    for (name, value) in fields {
                        budget.charge(name.len() + 1, ValueLocation::Query(range.clone()))?;
                        kinds.push((name.clone(), self.declaration_kind(value, budget, range)?));
                    }
                    program::Kind::Capture(Arc::new(zirql::NativeCapture { fields: kinds }))
                }
                _ => self.constructed_kind(scalar_facts(ty, budget, range)?),
            })
        }
    }
    fn is_operation_name(&self, name: &str) -> bool {
        zirql::is_common_operation_name(name)
            || parser::is_reserved(name)
            || matches!(name, "operand" | "project" | "paths" | "match")
    }
}
impl<'q> RuntimeTypeHost<EvaluationContext<'_, '_, 'q>> for report::ReportHost<'_> {
    type Kind = program::Kind;
    fn constructed_value<'v>(&self, value: &'v QueryOutput) -> Option<&'v ConstructedValue> {
        match unretained(value) {
            QueryOutput::Native(NativeValue::Constructed(v)) => Some(v.value()),
            _ => None,
        }
    }
    fn admit_native(
        &self,
        value: &QueryOutput,
        expected: &D<program::Kind>,
        context: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        location: ValueLocation,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        if zirql::admit_computational(value, expected, self, context, budget, location.clone())? {
            return Ok(());
        }
        let range = match &location {
            ValueLocation::Query(r) | ValueLocation::Source(r) => r,
        };
        {
            budget.charge(1, location.clone())?;
            match (expected, unretained(value)) {
                (D::Optional(inner), value) => {
                    return zirql::admit_value(
                        value,
                        inner,
                        self,
                        context,
                        budget,
                        location.clone(),
                    );
                }
                (D::Native(program::Kind::Operations), QueryOutput::Operations(ops))
                    if ops.len() == 1 =>
                {
                    check_operations(context.document, ops, budget, range)?
                }
                (D::Native(program::Kind::Ssa), QueryOutput::Native(NativeValue::Ssa(value))) => {
                    capture::check_value(context.document, *value, range)
                        .map_err(zirql::EvaluationError::Host)?
                }
                (
                    D::Sequence(inner) | D::Array(inner),
                    QueryOutput::Native(NativeValue::OperationSequence(ops)),
                ) if matches!(inner.as_ref(), D::Native(program::Kind::Operations)) => {
                    check_operations(context.document, ops, budget, range)?
                }
                (D::Integer, QueryOutput::Count(value)) if i64::try_from(*value).is_ok() => {}
                (D::String, QueryOutput::Native(NativeValue::String(value))) => {
                    budget.charge(value.len(), location.clone())?;
                }
                (D::Sequence(inner) | D::Array(inner), QueryOutput::Values(values))
                    if matches!(inner.as_ref(), D::String) =>
                {
                    for value in values {
                        budget.charge(value.len() + 1, location.clone())?;
                    }
                }
                (D::Tuple(types), QueryOutput::Native(NativeValue::Tuple(tuple)))
                    if types.len() == tuple.values.len() =>
                {
                    for (value, ty) in tuple.values.iter().zip(types) {
                        zirql::admit_value(value, ty, self, context, budget, location.clone())?;
                    }
                }
                (
                    D::Sequence(element) | D::Array(element),
                    QueryOutput::Native(NativeValue::Sequence(sequence)),
                ) if element.as_ref() == sequence.element.as_ref() => {
                    budget.check_items(sequence.values.len(), location.clone())?;
                    for value in sequence.values.iter() {
                        zirql::admit_value(
                            value,
                            element,
                            self,
                            context,
                            budget,
                            location.clone(),
                        )?;
                    }
                }
                (D::Record(fields), QueryOutput::Native(NativeValue::Capture(record))) => {
                    admit_fields(
                        self,
                        record.fields.iter().map(|(n, v)| (n.as_str(), v)),
                        fields,
                        context,
                        budget,
                        &location,
                    )?
                }
                (D::Record(fields), QueryOutput::Native(NativeValue::Map(record))) => admit_fields(
                    self,
                    record.iter().map(|(n, v)| (n.as_str(), v)),
                    fields,
                    context,
                    budget,
                    &location,
                )?,
                (
                    D::Sequence(inner) | D::Array(inner),
                    QueryOutput::Native(NativeValue::Captures(records)),
                ) if matches!(inner.as_ref(), D::Record(_)) => {
                    let D::Record(fields) = inner.as_ref() else {
                        unreachable!()
                    };
                    budget.check_items(records.len(), location.clone())?;
                    for record in records.rows() {
                        admit_fields(
                            self,
                            record.fields.iter().map(|(n, v)| (n.as_str(), v)),
                            fields,
                            context,
                            budget,
                            &location,
                        )?;
                    }
                }
                (
                    D::Native(kind @ (program::Kind::Capture(_) | program::Kind::Captures(_))),
                    value,
                ) => {
                    admit_view_kind(value, kind, context, budget, range)?;
                }
                _ => {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "function value does not satisfy its declared type",
                    )
                    .into());
                }
            }
            Ok(())
        }
    }
}
fn admit_view_kind(
    value: &QueryOutput,
    expected: &program::Kind,
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<(), zirql::EvaluationError<EvaluationError>> {
    let mut pending = vec![(value, expected)];
    while let Some((value, expected)) = pending.pop() {
        budget.charge(1, ValueLocation::Query(range.clone()))?;
        match (expected, unretained(value)) {
            (
                program::Kind::Operations,
                QueryOutput::Operations(values)
                | QueryOutput::Native(NativeValue::OperationSequence(values)),
            ) => check_operations(context.document, values, budget, range)?,
            (program::Kind::Ssa, QueryOutput::Native(NativeValue::Ssa(value))) => {
                capture::check_value(context.document, *value, range)
                    .map_err(zirql::EvaluationError::Host)?
            }
            (program::Kind::Capture(schema), QueryOutput::Native(NativeValue::Capture(record))) => {
                if schema.fields.len() != record.fields.len() {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "native capture shape changed",
                    )
                    .into());
                }
                budget.charge_results(schema.fields.len(), ValueLocation::Query(range.clone()))?;
                for (label, kind) in &schema.fields {
                    let child = record.get(&Located::new(label.as_str(), range.clone()), budget)?;
                    pending.push((child, kind));
                }
            }
            (
                program::Kind::Captures(schema),
                QueryOutput::Native(NativeValue::Captures(records)),
            ) => {
                budget.charge_results(records.len(), ValueLocation::Query(range.clone()))?;
                for record in records.rows() {
                    if schema.fields.len() != record.fields.len() {
                        return Err(zirql::QueryError::query(
                            range.clone(),
                            "native capture shape changed",
                        )
                        .into());
                    }
                    budget
                        .charge_results(schema.fields.len(), ValueLocation::Query(range.clone()))?;
                    for (label, kind) in &schema.fields {
                        let child =
                            record.get(&Located::new(label.as_str(), range.clone()), budget)?;
                        pending.push((child, kind));
                    }
                }
            }
            (kind, value)
                if report::output_kind(value, budget, range)
                    .map_err(zirql::EvaluationError::Host)?
                    == *kind => {}
            _ => {
                return Err(zirql::QueryError::query(
                    range.clone(),
                    "native view does not satisfy its kind",
                )
                .into());
            }
        }
    }
    Ok(())
}

fn check_operations(
    document: &Document,
    operations: &[OperationId],
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<(), zirql::EvaluationError<EvaluationError>> {
    budget.check_items(operations.len(), ValueLocation::Query(range.clone()))?;
    for &op in operations {
        budget.charge(1, ValueLocation::Query(range.clone()))?;
        document.check_operation(op).map_err(|e| {
            zirql::EvaluationError::Host(EvaluationError::new(e.to_string()).at(range.clone()))
        })?;
    }
    Ok(())
}

pub(super) fn error(
    error: zirql::FunctionError<zirql::EvaluationError<EvaluationError>>,
) -> EvaluationError {
    let mut failure = program::evaluation_error(error.error);
    if !error.calls.is_empty() {
        let calls = error
            .calls
            .into_iter()
            .map(|call| {
                format!(
                    "{} at {}..{}",
                    call.function, call.range.start, call.range.end
                )
            })
            .collect::<Vec<_>>()
            .join(" <- ");
        failure.message = format!("{} (calls: {calls})", failure.message);
    }
    failure
}

const NATIVE_FUNCTION_NAMES: &[&str] = &[
    "direct_children",
    "cost_annotation",
    "same_operation",
    "user_info",
];
pub(super) fn is_native_function(name: &str) -> bool {
    NATIVE_FUNCTION_NAMES.contains(&name)
}
fn native_function<'c, 'a, 'q, 'h>(
    name: &str,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<
    Option<zirql::NativeFunction<EvaluationContext<'c, 'a, 'q>, report::ReportHost<'h>>>,
    zirql::QueryError,
> {
    let Some(index) = NATIVE_FUNCTION_NAMES
        .iter()
        .position(|candidate| *candidate == name)
    else {
        return Ok(None);
    };
    capture::slots::<D<program::Kind>>(8, budget, range).map_err(report::query_error)?;
    budget.charge_results(8, ValueLocation::Query(range.clone()))?;
    let operation = || D::Native(program::Kind::Operations);
    let (parameters, result, invoke): (
        _,
        _,
        zirql::NativeFunctionImplementation<EvaluationContext<'c, 'a, 'q>, report::ReportHost<'h>>,
    ) = match index {
        0 => (
            vec![operation()],
            D::Sequence(Box::new(operation())),
            direct_children,
        ),
        1 => (
            vec![operation()],
            D::Optional(Box::new(D::Integer)),
            cost_annotation,
        ),
        2 => (vec![operation(), operation()], D::Boolean, same_operation),
        3 => {
            budget.charge("targets".len(), ValueLocation::Query(range.clone()))?;
            (
                vec![operation()],
                D::Record(vec![("targets".into(), D::Sequence(Box::new(operation())))]),
                user_info,
            )
        }
        _ => unreachable!("native definition index"),
    };
    Ok(Some(zirql::NativeFunction {
        name: NATIVE_FUNCTION_NAMES[index],
        signature: zirql::OperationSignature { parameters, result },
        invoke,
    }))
}
pub(super) fn unretained(mut value: &QueryOutput) -> &QueryOutput {
    while let QueryOutput::Native(NativeValue::Retained(inner)) = value {
        value = inner;
    }
    value
}
fn admitted_operation(
    value: &QueryOutput,
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<OperationId, EvaluationError> {
    let QueryOutput::Operations(operations) = unretained(value) else {
        return Err(
            EvaluationError::new("native function requires one Operation").at(range.clone()),
        );
    };
    if operations.len() != 1 {
        return Err(
            EvaluationError::new("native function requires one Operation").at(range.clone()),
        );
    }
    check_operations(context.document, operations, budget, range)
        .map_err(program::evaluation_error)?;
    Ok(operations[0])
}
fn same_operation(
    _: &report::ReportHost<'_>,
    values: Vec<Located<QueryOutput>>,
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
    let left = admitted_operation(&values[0].value, context, budget, range)
        .map_err(zirql::EvaluationError::Host)?;
    let right = admitted_operation(&values[1].value, context, budget, range)
        .map_err(zirql::EvaluationError::Host)?;
    ConstructedReport::new(
        ConstructedValue::Boolean(left == right),
        range.clone(),
        budget,
    )
    .map(|v| QueryOutput::Native(NativeValue::Constructed(v)))
    .map_err(zirql::EvaluationError::Host)
}
fn direct_children(
    _: &report::ReportHost<'_>,
    values: Vec<Located<QueryOutput>>,
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
    (|| -> Result<QueryOutput, EvaluationError> {
        let operation = admitted_operation(&values[0].value, context, budget, range)?;

        program::charge_results(budget, 1, range)?;
        evaluate_children(context.document, &[operation], budget)
            .map(|children| QueryOutput::Native(NativeValue::OperationSequence(children)))
    })()
    .map_err(zirql::EvaluationError::Host)
}
fn cost_annotation(
    _: &report::ReportHost<'_>,
    values: Vec<Located<QueryOutput>>,
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
    (|| -> Result<QueryOutput, EvaluationError> {
        let operation = admitted_operation(&values[0].value, context, budget, range)?;

        program::charge_results(budget, 3, range)?;
        let value = match context
            .document
            .attribute_id(operation, "cost")
            .and_then(|id| context.document.attribute_value(id))
        {
            None => ConstructedValue::Absent,
            Some(attribute) => {
                let spelling = context
                    .document
                    .attribute_id(operation, "cost")
                    .and_then(|id| context.document.attribute_spelling_value(id))
                    .unwrap_or("");
                program::charge_bytes(budget, spelling.len() + 1, range)?;
                let integer = attribute.decoded_integer().ok_or_else(|| {
                    EvaluationError::new("cost annotation must be an i64 integer").at(range.clone())
                })?;
                let negative = integer.digits.starts_with('-');
                let magnitude = u64::from_str_radix(
                    integer.digits.trim_start_matches(['+', '-']),
                    integer.radix,
                )
                .map_err(|_| {
                    EvaluationError::new("cost annotation is outside i64").at(range.clone())
                })?;
                let number = if negative {
                    if magnitude == 1u64 << 63 {
                        i64::MIN
                    } else {
                        -i64::try_from(magnitude).map_err(|_| {
                            EvaluationError::new("cost annotation is outside i64").at(range.clone())
                        })?
                    }
                } else {
                    i64::try_from(magnitude).map_err(|_| {
                        EvaluationError::new("cost annotation is outside i64").at(range.clone())
                    })?
                };
                ConstructedValue::Integer(number)
            }
        };
        ConstructedReport::with_types(
            value,
            vec![ValueType::Integer, ValueType::Absent],
            range.clone(),
            budget,
        )
        .map(|v| QueryOutput::Native(NativeValue::Constructed(v)))
    })()
    .map_err(zirql::EvaluationError::Host)
}
fn user_info(
    _: &report::ReportHost<'_>,
    values: Vec<Located<QueryOutput>>,
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
    (|| -> Result<QueryOutput, EvaluationError> {
        let operation = admitted_operation(&values[0].value, context, budget, range)?;

        if !context.document.is_semantically_complete() {
            return Err(
                EvaluationError::new("user_info requires complete semantic adjacency")
                    .at(range.clone()),
            );
        }
        let mut targets = Vec::new();
        for result in 0..context
            .document
            .result_types(operation)
            .map_or(0, <[_]>::len)
        {
            let value = ValueId::OperationResult {
                operation,
                result: u32::try_from(result).map_err(|_| {
                    EvaluationError::new("result index exceeds u32").at(range.clone())
                })?,
            };
            let uses = context.document.query_uses(value, |units| {
                capture::slots::<crate::semantic::UseSite>(units, budget, range)?;
                budget
                    .charge(units, ValueLocation::Query(range.clone()))
                    .map_err(core_error)?;
                budget
                    .charge_results(units, ValueLocation::Query(range.clone()))
                    .map_err(core_error)
            })?;
            budget
                .check_items(
                    targets.len().saturating_add(uses.len()),
                    ValueLocation::Query(range.clone()),
                )
                .map_err(core_error)?;
            capture::slots::<OperationId>(uses.len(), budget, range)?;
            targets.reserve(uses.len());
            for site in uses {
                let user = match site {
                    crate::semantic::UseSite::Operand { operation, .. }
                    | crate::semantic::UseSite::SuccessorArgument { operation, .. } => operation,
                };
                context
                    .document
                    .check_operation(user)
                    .map_err(|e| EvaluationError::new(e.to_string()).at(range.clone()))?;
                targets.push(user);
            }
        }
        program::charge_results(budget, 3, range)?;
        program::charge_bytes(budget, "targets".len() + 1, range)?;
        Ok(QueryOutput::Native(NativeValue::Capture(
            zirql::NativeCapture {
                fields: vec![(
                    "targets".to_owned(),
                    QueryOutput::Native(NativeValue::OperationSequence(targets)),
                )],
            },
        )))
    })()
    .map_err(zirql::EvaluationError::Host)
}

/// Signature-only discovery is safe inside native preparation callbacks. The
/// submitted request separately prepares every complete body in `Session::new`.
pub(super) fn value_kind(
    host: &report::ReportHost<'_>,
    value: &zirql::ValueExpression<report::Operand>,
    relative: program::Kind,
    scope: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<program::Kind, zirql::QueryError> {
    zirql::pattern_function_value_kind(
        host.functions
            .iter()
            .map(|function| function.declaration.as_ref()),
        host,
        value,
        scope,
        Some(relative),
        budget,
    )
}

/// A checked textual top-level pattern binding. Clone retains its immutable tree.
#[derive(Debug, PartialEq)]
pub struct ParsedBinding {
    pub(super) pattern: Arc<Pattern>,
    pub(super) value: ParsedComprehension,
}
impl Clone for ParsedBinding {
    fn clone(&self) -> Self {
        Self {
            pattern: Arc::clone(&self.pattern),
            value: self.value.clone(),
        }
    }
}
impl Eq for ParsedBinding {}
impl Drop for ParsedBinding {
    fn drop(&mut self) {
        if let Some(pattern) = Arc::get_mut(&mut self.pattern) {
            zirql::dispose_pattern(std::mem::replace(
                pattern,
                Located::new(PatternKind::Wildcard, 0..0),
            ));
        }
    }
}
impl ParsedBinding {
    pub(super) fn new(pattern: Pattern, value: ParsedComprehension) -> Self {
        Self {
            pattern: Arc::new(pattern),
            value,
        }
    }
}

type Entry<'q> = zirql::PreparedFunctionValue<'q, report::Operand, program::Kind>;
/// Every submitted request owns this unit, including an empty function list.
/// It stores no document, bindings, budget or result; execution supplies them.
pub(super) struct Session<'q, 'a, 'd> {
    unit: zirql::PreparedFunctions<'q, 'a, EvaluationContext<'a, 'd, 'q>, report::ReportHost<'q>>,
    native_arguments: BTreeMap<(usize, usize), Entry<'q>>,
    bindings: BTreeMap<usize, zirql::PreparedPatternBinding<'q, report::Operand, program::Kind>>,
    expression_aliases: BTreeMap<usize, usize>,
    expressions:
        BTreeMap<usize, Option<zirql::PreparedExpression<'q, report::Operand, program::Kind>>>,
    limits: FunctionLimits,
}
impl<'q, 'a, 'd> Session<'q, 'a, 'd> {
    pub(super) fn new(
        functions: &'q [FunctionDefinition],
        host: &'a report::ReportHost<'q>,
        limits: FunctionLimits,
        budget: &mut Budget,
    ) -> Result<Self, EvaluationError> {
        Ok(Self {
            unit: zirql::PreparedFunctions::prepare_patterns(
                functions.iter().map(|f| f.declaration.as_ref()),
                host,
                budget,
            )
            .map_err(error)?,
            native_arguments: BTreeMap::new(),
            bindings: BTreeMap::new(),
            expressions: BTreeMap::new(),
            expression_aliases: BTreeMap::new(),
            limits,
        })
    }
    pub(super) fn prepare_expression(
        &mut self,
        expression: &'q model::Expression,
        input: program::Kind,
        names: &BTreeMap<String, program::Kind>,
        host: &report::ReportHost<'q>,
        expressions: &'q BTreeMap<usize, report::Operand>,
        budget: &mut Budget,
    ) -> Result<program::Kind, EvaluationError> {
        let source = &expressions[&(expression as *const _ as usize)];
        let native_only = !report::needs_preparation(expression)
            && names
                .values()
                .all(|kind| !matches!(kind, program::Kind::Constructed(_)));
        let result = if native_only {
            let kind = program::Validator {
                host: report::ReportHost {
                    functions: host.functions,
                },
            }
            .expression(expression, input.clone(), names, budget)?;
            self.expressions.insert(source as *const _ as usize, None);
            kind
        } else {
            let entry = self
                .unit
                .prepare_expression(source, names, Some(input.clone()), budget)
                .map_err(error)?;
            let kind = entry.kind().clone();
            self.expressions
                .insert(source as *const _ as usize, Some(entry));
            kind
        };
        // Core preparation owns common bodies. Only opaque legacy groups still
        // need Session entries for their native execution path.
        if !native_only && report::native_groups(source, budget)?.is_empty() {
            return Ok(result);
        }
        for steps in
            std::iter::once(&expression.first).chain(expression.rest.iter().map(|(_, steps)| steps))
        {
            let mut kind = input.clone();
            for step in steps {
                if let zirql::ExpressionStep::Stage(stage) = &step.value {
                    if native_only {
                        for argument in host
                            .scoped_value_arguments(stage, &kind, names, budget, &step.range)
                            .map_err(core_error)?
                        {
                            budget
                                .charge(
                                    names.keys().map(|name| name.len() + 1).sum(),
                                    ValueLocation::Query(step.range.clone()),
                                )
                                .map_err(core_error)?;
                            budget
                                .charge_results(
                                    names.len() + 1,
                                    ValueLocation::Query(step.range.clone()),
                                )
                                .map_err(core_error)?;
                            let mut scope = names.clone();
                            scope.extend(
                                argument
                                    .locals
                                    .into_iter()
                                    .map(|(name, kind)| (name.value, kind)),
                            );
                            let entry = self
                                .unit
                                .prepare_value(
                                    argument.expression,
                                    &scope,
                                    argument.relative,
                                    budget,
                                )
                                .map_err(error)?;
                            self.native_arguments.insert(
                                (
                                    Arc::as_ptr(stage) as usize,
                                    argument.expression as *const _ as usize,
                                ),
                                entry,
                            );
                        }
                    }

                    for nested in report::nested(stage) {
                        let relative = if matches!(stage.as_ref(), model::Stage::Group { .. }) {
                            kind.clone()
                        } else {
                            program::Kind::Operations
                        };
                        self.prepare_expression(
                            nested,
                            relative,
                            names,
                            host,
                            expressions,
                            budget,
                        )?;
                    }
                    if !matches!(
                        stage.as_ref(),
                        model::Stage::Emit { .. }
                            | model::Stage::Format { .. }
                            | model::Stage::Json { .. }
                            | model::Stage::Markdown { .. }
                            | model::Stage::Print { .. }
                    ) {
                        kind = host
                            .stage_kind(stage, &kind, names, budget, &step.range)
                            .map_err(core_error)?;
                    }
                } else {
                    let source = zirql::Located::new(
                        zirql::ExpressionKind::Sequence {
                            input: Box::new(zirql::Located::new(
                                zirql::ExpressionKind::Input,
                                step.range.clone(),
                            )),
                            steps: vec![step.clone()],
                        },
                        step.range.clone(),
                    );
                    kind = zirql::pattern_function_expression_kind(
                        host.functions
                            .iter()
                            .map(|function| function.declaration.as_ref()),
                        host,
                        &source,
                        names,
                        Some(kind),
                        true,
                        budget,
                    )
                    .map_err(core_error)?;
                }
            }
        }
        Ok(result)
    }
    pub(super) fn prepare_program(
        &mut self,
        query: &'q Query,
        document_name: Option<&str>,
        host: &report::ReportHost<'q>,
        program: &'q zirql::Program<model::NativeStage, model::SetOperator, program::Emitter>,
        expressions: &'q BTreeMap<usize, report::Operand>,
        budget: &mut Budget,
    ) -> Result<(), EvaluationError> {
        let mut names = BTreeMap::new();
        if document_name.is_some() {
            names.insert("document".to_owned(), program::Kind::Strings);
        }
        let mut parameters = query
            .parameters
            .iter()
            .zip(&query.parameter_positions)
            .peekable();
        for (index, statement) in query.statements.iter().enumerate() {
            while parameters.peek().is_some_and(|(_, p)| **p == index) {
                let (d, _) = parameters.next().unwrap();
                names.extend(program::parameter_names(std::slice::from_ref(d), budget)?);
            }
            if let model::Statement::PatternBinding { expression, .. } = statement {
                let [
                    zirql::Located {
                        value: zirql::ExpressionStep::Stage(stage),
                        ..
                    },
                ] = expression.first.as_slice()
                else {
                    unreachable!("lowered pattern binding stage")
                };
                let model::Stage::PatternBinding { value, .. } = stage.as_ref() else {
                    unreachable!("lowered pattern binding stage")
                };
                let entry = self
                    .unit
                    .prepare_binding(
                        &value.pattern,
                        &value.value.expression,
                        &names,
                        Some(program::Kind::Operations),
                        budget,
                    )
                    .map_err(error)?;
                for (name, kind) in entry.bindings() {
                    budget
                        .charge(
                            name.value.len() + 1,
                            ValueLocation::Query(name.range.clone()),
                        )
                        .map_err(core_error)?;
                    names.insert(name.value.clone(), kind.clone());
                }
                budget
                    .charge_results(1, ValueLocation::Query(program::range(expression.range)))
                    .map_err(core_error)?;
                self.bindings.insert(Arc::as_ptr(stage) as usize, entry);
                continue;
            }
            let expression = match statement {
                model::Statement::Binding { expression, .. }
                | model::Statement::PatternBinding { expression, .. }
                | model::Statement::Do(expression)
                | model::Statement::Query(expression) => expression,
            };
            let kind = self.prepare_expression(
                expression,
                program::Kind::Operations,
                &names,
                host,
                expressions,
                budget,
            )?;
            if let model::Statement::Binding { name, .. } = statement {
                names.insert(name.clone(), kind);
            }
        }
        for (d, _) in parameters {
            names.extend(program::parameter_names(std::slice::from_ref(d), budget)?);
        }
        self.prepare_expression(
            &query.expression,
            program::Kind::Operations,
            &names,
            host,
            expressions,
            budget,
        )?;
        // Program assembly keeps parameter introduction and implicit emission local.
        for statement in program.statements() {
            let source = match &statement.value {
                zirql::Statement::Bind { expression, .. }
                | zirql::Statement::BindFields { expression, .. }
                | zirql::Statement::Evaluate { expression, .. } => expression,
            };
            let Some(model_expression) = query
                .statements
                .iter()
                .map(|statement| match statement {
                    model::Statement::Binding { expression, .. }
                    | model::Statement::PatternBinding { expression, .. }
                    | model::Statement::Do(expression)
                    | model::Statement::Query(expression) => expression,
                })
                .chain(std::iter::once(&query.expression))
                .find(|expression| program::range(expression.range) == source.range)
            else {
                continue;
            };
            let original = &expressions[&(model_expression as *const _ as usize)];
            // These entries borrow the same common bodies; statement bindings still supply their lexical scope at execution.
            self.expression_aliases
                .insert(source as *const _ as usize, original as *const _ as usize);
        }
        Ok(())
    }
    pub(super) fn run_expression(
        &self,
        expression: &report::Operand,
        context: &mut EvaluationContext<'a, 'd, 'q>,
        input: QueryOutput,
        bindings: &zirql::Bindings<QueryOutput>,
        budget: &mut Budget,
    ) -> Result<QueryOutput, EvaluationError> {
        if let zirql::ExpressionKind::Sequence { steps, .. } = &expression.value
            && let [
                zirql::Located {
                    value: zirql::ExpressionStep::Stage(stage),
                    ..
                },
            ] = steps.as_slice()
            && matches!(
                stage.as_ref(),
                model::Stage::Parameter { .. } | model::Stage::PatternBinding { .. }
            )
        {
            return program::NativeHost
                .evaluate_stage(stage, input, context, bindings, budget, &expression.range)
                .map_err(program::evaluation_error);
        }
        let identity = expression as *const _ as usize;
        let identity = self
            .expression_aliases
            .get(&identity)
            .copied()
            .unwrap_or(identity);
        let entry = self.expressions.get(&identity).ok_or_else(|| {
            EvaluationError::new("expression was not prepared").at(expression.range.clone())
        })?;
        let value = match entry {
            Some(entry) => self
                .unit
                .evaluate_expression(entry, bindings, Some(&input), context, budget, self.limits)
                .map_err(error)?,
            None => zirql::evaluate_expression(
                expression,
                &program::NativeHost,
                input,
                context,
                bindings,
                budget,
            )
            .map_err(program::evaluation_error)?,
        };
        owned(value, budget, &expression.range)
    }
    pub(super) fn run_binding(
        &self,
        context: &mut EvaluationContext<'a, 'd, 'q>,
        input: &QueryOutput,
        bindings: &zirql::Bindings<QueryOutput>,
        budget: &mut Budget,
    ) -> Result<QueryOutput, EvaluationError> {
        let entry = self
            .bindings
            .get(&context.current_stage)
            .ok_or_else(|| EvaluationError::new("pattern binding was not prepared"))?;
        self.unit
            .evaluate_binding(entry, bindings, Some(input), context, budget, self.limits)
            .map(|capture| QueryOutput::Native(NativeValue::Capture(capture)))
            .map_err(error)
    }
    pub(super) fn run_native_argument(
        &self,
        value: &zirql::ValueExpression<report::Operand>,
        context: &mut EvaluationContext<'a, 'd, 'q>,
        input: &QueryOutput,
        bindings: &zirql::Bindings<QueryOutput>,
        budget: &mut Budget,
    ) -> Result<QueryOutput, EvaluationError> {
        let entry = self
            .native_arguments
            .get(&(context.current_stage, value as *const _ as usize))
            .ok_or_else(|| {
                EvaluationError::new("native argument was not prepared in this scope")
                    .at(value.range.clone())
            })?;
        self.unit
            .evaluate(entry, bindings, Some(input), context, budget, self.limits)
            .map_err(error)
    }
}

pub(super) fn owned(
    mut value: QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    while let QueryOutput::Native(NativeValue::Retained(shared)) = value {
        value = match Arc::try_unwrap(shared) {
            Ok(value) => value,
            Err(shared) => program::copy_value(unretained(&shared), budget, range)?,
        };
    }
    Ok(value)
}
pub(super) fn constructed(
    value: QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<ConstructedReport, EvaluationError> {
    match owned(value, budget, range)? {
        QueryOutput::Native(NativeValue::Constructed(value)) => Ok(value),
        value @ QueryOutput::Native(NativeValue::Map(_)) => {
            let value = comprehension::project(&value, budget, range)?;
            ConstructedReport::new(value, range.clone(), budget)
        }
        _ => Err(EvaluationError::new("report requires a constructed value").at(range.clone())),
    }
}

impl FunctionDefinition {
    /// Construct shared declaration data directly, without parsing generated text.
    pub fn new(
        name: impl Into<String>,
        parameters: Vec<FunctionParameter<FunctionNativeType>>,
        result: FunctionType,
        bindings: Vec<(String, ReportExpr)>,
        body: ReportExpr,
    ) -> Result<Self, EvaluationError> {
        Self::with_patterns(
            name,
            parameters
                .into_iter()
                .map(|p| PatternParameter {
                    pattern: Located::new(PatternKind::Name(p.name), 0..0),
                    declaration: p.declaration,
                })
                .collect(),
            result,
            ReportExpr::block(
                bindings
                    .into_iter()
                    .map(|(name, value)| {
                        (
                            Located::new(PatternKind::Name(Located::new(name, 0..0)), 0..0),
                            value,
                        )
                    })
                    .collect(),
                body,
            )?,
        )
    }

    /// Construct whole typed parameter patterns and a pure lexical body.
    /// All arguments are admitted before any parameter pattern installs names.
    pub fn with_patterns(
        name: impl Into<String>,
        parameters: Vec<PatternParameter<FunctionNativeType>>,
        result: FunctionType,
        body: ReportExpr,
    ) -> Result<Self, EvaluationError> {
        let mut budget = Budget::for_program(comprehension::default_limits());
        let mut incoming = parameters.into_iter();
        let mut parameters: Vec<Located<PatternParameter<program::Kind>>> = Vec::new();
        while let Some(parameter) = incoming.next() {
            let declaration = validate_pattern(&parameter.pattern, &mut budget)
                .and_then(|()| native_type(parameter.declaration, 0));
            let declaration = match declaration {
                Ok(declaration) => declaration,
                Err(error) => {
                    zirql::dispose_pattern(parameter.pattern);
                    for parameter in incoming {
                        zirql::dispose_pattern(parameter.pattern);
                    }
                    for parameter in parameters {
                        zirql::dispose_pattern(parameter.value.pattern);
                    }
                    return Err(error);
                }
            };
            parameters.push(Located::new(
                PatternParameter {
                    pattern: parameter.pattern,
                    declaration,
                },
                0..0,
            ));
        }
        let result = match native_type(result, 0) {
            Ok(result) => result,
            Err(error) => {
                for parameter in parameters {
                    zirql::dispose_pattern(parameter.value.pattern);
                }
                return Err(error);
            }
        };
        Ok(Self::parsed(zirql::PatternFunctionDeclaration {
            name: Located::new(name.into(), 0..0),
            parameters,
            result: Located::new(result, 0..0),
            value: normalize(body.into_expression()),
        }))
    }
}

pub(super) fn validate_pattern(
    pattern: &Pattern,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    zirql::pattern_names(pattern, budget)
        .map(|_| ())
        .map_err(core_error)
}

fn native_type(ty: FunctionType, depth: usize) -> Result<D<program::Kind>, EvaluationError> {
    if depth >= parser::DEFAULT_NESTING_LIMIT {
        return Err(EvaluationError::new(
            "function signature nesting limit exceeded",
        ));
    }
    Ok(match ty {
        D::Native(FunctionNativeType::Operation) => D::Native(program::Kind::Operations),
        D::Native(FunctionNativeType::SSAValue) => D::Native(program::Kind::Ssa),
        D::Optional(v) => D::Optional(Box::new(native_type(*v, depth + 1)?)),
        D::Sequence(v) => D::Sequence(Box::new(native_type(*v, depth + 1)?)),
        D::Array(v) => D::Array(Box::new(native_type(*v, depth + 1)?)),
        D::Keyed(v) => D::Keyed(Box::new(native_type(*v, depth + 1)?)),
        D::Tuple(v) => D::Tuple(
            v.into_iter()
                .map(|v| native_type(v, depth + 1))
                .collect::<Result<_, _>>()?,
        ),
        D::Record(v) => D::Record(
            v.into_iter()
                .map(|(n, v)| Ok((n, native_type(v, depth + 1)?)))
                .collect::<Result<_, EvaluationError>>()?,
        ),
        D::Value => D::Value,
        D::Dictionary(key, value) => D::Dictionary(
            Box::new(native_type(*key, depth + 1)?),
            Box::new(native_type(*value, depth + 1)?),
        ),
        D::Boolean => D::Boolean,
        D::Integer => D::Integer,
        D::Float => D::Float,
        D::String => D::String,
        D::Member(v) => D::Member(v),
        D::Set(v) => D::Set(v),
        D::Queue(value) => D::Queue(Box::new(native_type(*value, depth + 1)?)),
        D::IdentitySet(FunctionNativeType::Operation) => D::IdentitySet(program::Kind::Operations),
        D::IdentitySet(FunctionNativeType::SSAValue) => D::IdentitySet(program::Kind::Ssa),
    })
}
// Shared typed values use an explicit source wrapper when another owner keeps
// the tree alive. In a function that wrapper must retain native values too.
pub(super) fn normalize(
    mut value: zirql::ValueExpression<report::Operand>,
) -> zirql::ValueExpression<report::Operand> {
    let mut pending = vec![&mut value];
    while let Some(value) = pending.pop() {
        normalize_wrapper(value);
        zirql::for_each_value_child_mut(value, |child| pending.push(child));
    }
    value
}
/// Normalize outer reads under the frontend's result policy. Constructed-only
/// bodies project scalar data; function-capable bodies may retain native handles.
/// Common admission rejects lexical shadowing before execution.
pub(super) fn normalize_outer(
    mut value: zirql::ValueExpression<report::Operand>,
    names: &std::collections::HashSet<String>,
    retain_native: bool,
    budget: &mut Budget,
) -> Result<zirql::ValueExpression<report::Operand>, zirql::QueryError> {
    let result = zirql::walk_value_mut(&mut value, budget, 64, |value, _, budget| {
        if let zirql::ValueExpressionKind::Name(name) = &value.value
            && names.contains(name)
        {
            budget.charge(name.len() + 2, ValueLocation::Query(value.range.clone()))?;
            budget.charge_results(2, ValueLocation::Query(value.range.clone()))?;
            let query = Box::new(Located::new(
                zirql::ExpressionKind::Binding(name.clone()),
                value.range.clone(),
            ));
            value.value = if retain_native {
                zirql::ValueExpressionKind::NativeValue(query)
            } else {
                zirql::ValueExpressionKind::NativeQuery(query)
            };
        }
        normalize_wrapper(value);
        Ok(())
    });
    if let Err(error) = result {
        zirql::dispose_native_value(value);
        return Err(error);
    }
    Ok(value)
}
fn normalize_wrapper(value: &mut zirql::ValueExpression<report::Operand>) {
    use zirql::ValueExpressionKind as K;
    if matches!(&value.value,K::NativeQuery(q) if matches!(&q.value,zirql::ExpressionKind::Sequence{steps,..} if steps.len()==1 && matches!(&steps[0].value,zirql::ExpressionStep::Value(_))))
    {
        let K::NativeQuery(query) =
            std::mem::replace(&mut value.value, K::Literal(ConstructedValue::Absent))
        else {
            unreachable!()
        };
        value.value = K::NativeValue(query);
    }
}
pub(super) fn expression_roots<'a>(
    functions: &'a [FunctionDefinition],
    budget: &mut Budget,
) -> Result<Vec<&'a model::Expression>, EvaluationError> {
    let mut roots = Vec::new();
    for function in functions {
        report::visit_value(&function.declaration.value, &mut roots, budget)?;
    }
    Ok(roots)
}

fn admit_fields<'v>(
    host: &report::ReportHost<'_>,
    values: impl Iterator<Item = (&'v str, &'v QueryOutput)> + Clone,
    fields: &[(String, D<program::Kind>)],
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    location: &ValueLocation,
) -> Result<(), zirql::EvaluationError<EvaluationError>> {
    let range = match location {
        ValueLocation::Query(r) | ValueLocation::Source(r) => r,
    };
    for (name, _) in values.clone() {
        budget.charge(name.len() + fields.len() + 1, location.clone())?;
        if !fields.iter().any(|(field, _)| field == name) {
            return Err(
                zirql::QueryError::query(range.clone(), "unexpected native record field").into(),
            );
        }
    }
    for (name, ty) in fields {
        budget.charge(name.len() + 1, location.clone())?;
        let mut found = None;
        for (field, value) in values.clone() {
            budget.charge(field.len() + 1, location.clone())?;
            if field == name {
                found = Some(value);
                break;
            }
        }
        match found {
            Some(value) => zirql::admit_value(value, ty, host, context, budget, location.clone())?,
            None if matches!(ty, D::Optional(_)) => {}
            None => {
                return Err(
                    zirql::QueryError::query(range.clone(), "missing native record field").into(),
                );
            }
        }
    }
    Ok(())
}
