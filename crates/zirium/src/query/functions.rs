//! Typed pure functions retain shared trees and document-owned native handles.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::{DeclaredType as D, FunctionHost, ProjectionHost, RuntimeTypeHost, ValueType};

pub use zirql::{FunctionLimits, FunctionParameter};
/// Native signature identities; `SSAValue` is Zirium's checked `ValueId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionNativeType {
    Operation,
    SSAValue,
}
pub type FunctionType = D<FunctionNativeType>;
pub(super) type Declaration = zirql::FunctionDeclaration<report::Operand, program::Kind>;
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
            for (_, value) in std::mem::take(&mut d.bindings) {
                report::dispose_value(value.value);
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
        })
    }
}
impl<'q> FunctionHost<EvaluationContext<'_, '_, 'q>> for report::ReportHost<'_> {
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
                D::Optional(value) if contains_native(value, budget, range)? => {
                    self.declaration_kind(value, budget, range)?
                }
                D::Record(fields) if contains_native(ty, budget, range)? => {
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
                D::Sequence(value) | D::Array(value)
                    if contains_native(value, budget, range)?
                        && matches!(value.as_ref(), D::Record(_)) =>
                {
                    let program::Kind::Capture(schema) =
                        self.declaration_kind(value, budget, range)?
                    else {
                        unreachable!("native record kind")
                    };
                    program::Kind::Captures(schema)
                }
                D::Sequence(value) | D::Array(value)
                    if matches!(value.as_ref(), D::Native(program::Kind::Operations)) =>
                {
                    program::Kind::Operations
                }
                _ => self.constructed_kind(scalar_facts(ty, budget, range)?),
            })
        }
    }
    fn is_operation_name(&self, name: &str) -> bool {
        parser::is_reserved(name)
            || matches!(
                name,
                "direct_children" | "cost_annotation" | "operand" | "project" | "paths" | "match"
            )
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primitive {
    DirectChildren,
    CostAnnotation,
}
pub(super) fn unretained(mut value: &QueryOutput) -> &QueryOutput {
    while let QueryOutput::Native(NativeValue::Retained(inner)) = value {
        value = inner;
    }
    value
}
pub(super) fn primitive(
    primitive: Primitive,
    argument: &QueryOutput,
    context: &mut EvaluationContext<'_, '_, '_>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let QueryOutput::Operations(operations) = unretained(argument) else {
        return Err(
            EvaluationError::new("native primitive requires one Operation").at(range.clone()),
        );
    };
    if operations.len() != 1 {
        return Err(
            EvaluationError::new("native primitive requires one Operation").at(range.clone()),
        );
    }
    check_operations(context.document, operations, budget, range)
        .map_err(program::evaluation_error)?;
    let operation = operations[0];
    match primitive {
        Primitive::DirectChildren => {
            program::charge_results(budget, 1, range)?;
            evaluate_children(context.document, operations, budget)
                .map(|children| QueryOutput::Native(NativeValue::OperationSequence(children)))
        }
        Primitive::CostAnnotation => {
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
                        EvaluationError::new("cost annotation must be an i64 integer")
                            .at(range.clone())
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
                                EvaluationError::new("cost annotation is outside i64")
                                    .at(range.clone())
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
        }
    }
}

/// The declaration pass supplies facts only. Execution and complete admission
/// belong to the shared prepared unit below.
struct Facts<'f, 's> {
    host: &'s report::ReportHost<'f>,
    scope: &'s BTreeMap<String, program::Kind>,
    relative: program::Kind,
}
impl zirql::ValueQueryHost<report::Operand> for Facts<'_, '_> {
    type Error = zirql::QueryError;
    fn function(
        &mut self,
        name: &Located<String>,
        budget: &mut Budget,
    ) -> Result<(usize, usize), zirql::QueryError> {
        for (index, function) in self.host.functions.iter().enumerate() {
            budget.charge(
                name.value.len() + 1,
                ValueLocation::Query(name.range.clone()),
            )?;
            if function.declaration.name.value == name.value {
                return Ok((index, function.declaration.parameters.len()));
            }
        }
        Err(zirql::QueryError::query(
            name.range.clone(),
            format!("undefined function: {}", name.value),
        ))
    }
    fn function_types(
        &mut self,
        name: &Located<String>,
        budget: &mut Budget,
    ) -> Result<Vec<ValueType>, zirql::QueryError> {
        let (index, _) = self.function(name, budget)?;
        scalar_facts(
            &self.host.functions[index].declaration.result.value,
            budget,
            &name.range,
        )
    }
    fn native_value_types(
        &mut self,
        query: &report::Operand,
        locals: &[(&str, Vec<ValueType>)],
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<ValueType>, zirql::QueryError> {
        self.query_types(query, locals, budget, range)
    }
    fn query_types(
        &mut self,
        query: &report::Operand,
        locals: &[(&str, Vec<ValueType>)],
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<ValueType>, zirql::QueryError> {
        let mut scope = copy_scope(self.scope, budget, range)?;
        for (name, types) in locals {
            if !scope.contains_key(*name) {
                scope.insert(
                    (*name).to_owned(),
                    program::Kind::Constructed(Arc::new(zirql::copy_value_types(
                        types,
                        budget,
                        ValueLocation::Query(range.clone()),
                    )?)),
                );
            }
        }
        let kind = zirql::native_expression_kind::<EvaluationContext<'_, '_, '_>, _>(
            query,
            self.host,
            &self.relative,
            &scope,
            budget,
        )?;
        Ok(self
            .host
            .scalar_types(&kind, budget, range)?
            .unwrap_or_else(|| vec![ValueType::Unestablished]))
    }
    fn evaluate_query(
        &mut self,
        _: &report::Operand,
        _: &[(&str, ConstructedValue)],
        _: &mut Budget,
    ) -> Result<ConstructedValue, zirql::QueryError> {
        unreachable!("facts do not execute")
    }
    fn read_binding(
        &mut self,
        _: &str,
        _: &Range<usize>,
        _: &mut Budget,
    ) -> Result<ConstructedValue, zirql::QueryError> {
        unreachable!("facts do not execute")
    }
}
pub(super) fn value_kind(
    host: &report::ReportHost<'_>,
    value: &zirql::ValueExpression<report::Operand>,
    relative: program::Kind,
    scope: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<program::Kind, zirql::QueryError> {
    value_kind_at(host, value, relative, scope, budget, 0)
}
fn value_kind_at(
    host: &report::ReportHost<'_>,
    value: &zirql::ValueExpression<report::Operand>,
    relative: program::Kind,
    scope: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    depth: usize,
) -> Result<program::Kind, zirql::QueryError> {
    use zirql::ValueExpressionKind as V;
    if depth >= parser::DEFAULT_NESTING_LIMIT {
        return Err(zirql::QueryError::query(
            value.range.clone(),
            "function value nesting limit exceeded",
        ));
    }
    budget.charge(1, ValueLocation::Query(value.range.clone()))?;
    if let V::Conditional {
        branches,
        otherwise,
    } = &value.value
    {
        let kind = value_kind_at(host, otherwise, relative.clone(), scope, budget, depth + 1)?;
        let mut same = true;
        for (_, branch) in branches {
            same &=
                value_kind_at(host, branch, relative.clone(), scope, budget, depth + 1)? == kind;
        }
        if same {
            return Ok(kind);
        }
    }
    match &value.value {
        V::Name(name) => scope.get(name).cloned().ok_or_else(|| {
            zirql::QueryError::query(value.range.clone(), format!("undefined binding: {name}"))
        }),
        V::Call { function: name, .. } => {
            let mut facts = Facts {
                host,
                scope,
                relative,
            };
            let (index, _) = zirql::ValueQueryHost::function(&mut facts, name, budget)?;
            host.declaration_kind(
                &host.functions[index].declaration.result.value,
                budget,
                &value.range,
            )
        }
        V::NativeValue(query) | V::NativeQuery(query) => {
            zirql::native_expression_kind::<EvaluationContext<'_, '_, '_>, _>(
                query, host, &relative, scope, budget,
            )
        }
        _ => {
            let mut names = BTreeMap::new();
            for (name, kind) in scope {
                budget.charge(name.len() + 1, ValueLocation::Query(value.range.clone()))?;
                names.insert(
                    name.clone(),
                    host.scalar_types(kind, budget, &value.range)?
                        .unwrap_or_else(|| vec![ValueType::Unestablished]),
                );
            }
            let types = zirql::query_value_types(
                value,
                &names,
                &mut Facts {
                    host,
                    scope,
                    relative,
                },
                budget,
            )?;
            Ok(host.constructed_kind(types))
        }
    }
}

type Entry<'q> = zirql::PreparedFunctionValue<'q, report::Operand, program::Kind>;
/// One submitted request owns this unit and every entry prepared by it. It never
/// stores a document, bindings, budget or result; execution supplies that context.
pub(super) struct Session<'q, 'a, 'd> {
    pub(super) host: &'a report::ReportHost<'q>,
    unit: zirql::PreparedFunctions<'q, 'a, EvaluationContext<'a, 'd, 'q>, report::ReportHost<'q>>,
    entries: BTreeMap<(usize, usize), Entry<'q>>,
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
            host,
            unit: zirql::PreparedFunctions::prepare(
                functions.iter().map(|f| f.declaration.as_ref()),
                host,
                budget,
            )
            .map_err(error)?,
            entries: BTreeMap::new(),
            limits,
        })
    }
    fn prepare_value(
        &mut self,
        stage: &'q model::Stage,
        value: &'q zirql::ValueExpression<report::Operand>,
        input: program::Kind,
        names: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
    ) -> Result<(), EvaluationError> {
        let entry = self
            .unit
            .prepare_value(value, names, Some(input), budget)
            .map_err(error)?;
        budget
            .charge_results(1, ValueLocation::Query(value.range.clone()))
            .map_err(core_error)?;
        self.entries.insert(
            (stage as *const _ as usize, value as *const _ as usize),
            entry,
        );
        Ok(())
    }
    pub(super) fn prepare_expression(
        &mut self,
        expression: &'q model::Expression,
        input: program::Kind,
        names: &BTreeMap<String, program::Kind>,
        host: &report::ReportHost<'q>,
        budget: &mut Budget,
    ) -> Result<program::Kind, EvaluationError> {
        for stages in std::iter::once(&expression.first)
            .chain(expression.rest.iter().map(|(_, stages)| stages))
        {
            let mut kind = input.clone();
            for stage in stages {
                match stage {
                    model::Stage::Comprehension { value, .. }
                    | model::Stage::FunctionValue { value, .. } => {
                        self.prepare_value(stage, &value.expression, kind.clone(), names, budget)?
                    }
                    model::Stage::Projection { value, .. } => {
                        let mut scope = copy_scope(names, budget, &program::range(stage.range()))
                            .map_err(core_error)?;
                        let subject = host
                            .subject_kind(&kind, budget, &value.projection.subject.range)
                            .map_err(core_error)?;
                        scope.insert(value.projection.subject.value.clone(), subject.clone());
                        for (_, value) in &value.projection.fields {
                            self.prepare_value(stage, value, subject.clone(), &scope, budget)?;
                        }
                    }
                    _ => {
                        for (value, relative) in host
                            .value_arguments(stage, budget, &program::range(stage.range()))
                            .map_err(core_error)?
                        {
                            self.prepare_value(stage, value, relative, names, budget)?;
                        }
                    }
                }
                for nested in report::nested(stage) {
                    self.prepare_expression(
                        nested,
                        program::Kind::Operations,
                        names,
                        host,
                        budget,
                    )?;
                }
                kind = program::Validator {
                    host: report::ReportHost {
                        functions: host.functions,
                    },
                }
                .stages(std::slice::from_ref(stage), kind, names, budget)?;
            }
        }
        program::Validator {
            host: report::ReportHost {
                functions: host.functions,
            },
        }
        .expression(expression, input, names, budget)
    }
    pub(super) fn prepare_program(
        &mut self,
        query: &'q Query,
        document_name: Option<&str>,
        host: &report::ReportHost<'q>,
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
            let expression = match statement {
                model::Statement::Binding { expression, .. }
                | model::Statement::Do(expression)
                | model::Statement::Query(expression) => expression,
            };
            let kind = self.prepare_expression(
                expression,
                program::Kind::Operations,
                &names,
                host,
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
            budget,
        )?;
        Ok(())
    }
    pub(super) fn run(
        &self,
        value: &zirql::ValueExpression<report::Operand>,
        context: &mut EvaluationContext<'a, 'd, 'q>,
        input: &QueryOutput,
        bindings: &zirql::Bindings<QueryOutput>,
        budget: &mut Budget,
    ) -> Result<QueryOutput, EvaluationError> {
        let entry = self
            .entries
            .get(&(context.current_stage, value as *const _ as usize))
            .ok_or_else(|| {
                EvaluationError::new("written function entry was not prepared in this scope")
                    .at(value.range.clone())
            })?;
        self.unit
            .evaluate(entry, bindings, Some(input), context, budget, self.limits)
            .map_err(error)
    }
}

pub(super) fn evaluate(
    value: &ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &zirql::Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<QueryOutput, EvaluationError> {
    if let Some(session) = context.functions {
        session.run(&value.expression, context, &input, bindings, budget)
    } else {
        report::evaluate(value, context, input, bindings, budget)
            .map(|v| QueryOutput::Native(NativeValue::Constructed(v)))
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
        let parameters = parameters
            .into_iter()
            .map(|p| {
                Ok(Located::new(
                    FunctionParameter {
                        name: p.name,
                        declaration: native_type(p.declaration, 0)?,
                    },
                    0..0,
                ))
            })
            .collect::<Result<_, EvaluationError>>()?;
        let bindings = bindings
            .into_iter()
            .map(|(name, value)| (Located::new(name, 0..0), normalize(value.into_expression())))
            .collect();
        Ok(Self::parsed(zirql::FunctionDeclaration {
            name: Located::new(name.into(), 0..0),
            parameters,
            result: Located::new(native_type(result, 0)?, 0..0),
            bindings,
            value: normalize(body.into_expression()),
        }))
    }
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
        D::Boolean => D::Boolean,
        D::Integer => D::Integer,
        D::Float => D::Float,
        D::String => D::String,
        D::Member(v) => D::Member(v),
        D::Set(v) => D::Set(v),
    })
}
// Shared typed values use an explicit source wrapper when another owner keeps
// the tree alive. In a function that wrapper must retain native values too.
pub(super) fn normalize(
    mut value: zirql::ValueExpression<report::Operand>,
) -> zirql::ValueExpression<report::Operand> {
    use zirql::ValueExpressionKind as K;
    fn clauses<'v>(
        clauses: &'v mut [Located<zirql::ValueGenerator<report::Operand>>],
        pending: &mut Vec<&'v mut zirql::ValueExpression<report::Operand>>,
    ) {
        for clause in clauses {
            match &mut clause.value {
                zirql::ValueGenerator::In { source, filter, .. }
                | zirql::ValueGenerator::Bind {
                    value: source,
                    filter,
                    ..
                } => {
                    pending.push(source);
                    pending.extend(filter);
                }
            }
        }
    }
    let mut pending = vec![&mut value];
    while let Some(value) = pending.pop() {
        if matches!(&value.value,K::NativeQuery(q) if matches!(&q.value,zirql::ExpressionKind::Sequence{steps,..} if steps.len()==1 && matches!(&steps[0].value,zirql::ExpressionStep::Stage(model::Stage::FunctionValue{..}))))
        {
            let K::NativeQuery(query) =
                std::mem::replace(&mut value.value, K::Literal(ConstructedValue::Absent))
            else {
                unreachable!()
            };
            value.value = K::NativeValue(query);
        }
        match &mut value.value {
            K::Literal(_) | K::Name(_) | K::NativeQuery(_) | K::NativeValue(_) => {}
            K::Call { arguments, .. }
            | K::Array(arguments)
            | K::Tuple(arguments)
            | K::Set(arguments) => pending.extend(arguments),
            K::Occurs(v)
            | K::Absent(v)
            | K::One(v)
            | K::IndexSet(v)
            | K::Entries(v)
            | K::Unary { operand: v, .. }
            | K::Field { value: v, .. }
            | K::TupleAccess { value: v, .. }
            | K::GroupBy { rows: v, .. }
            | K::Reduce { array: v, .. } => pending.push(v),
            K::Record(fields) => pending.extend(fields.iter_mut().map(|(_, v)| v)),
            K::KeyedArray { entries, .. } => {
                for (k, v) in entries {
                    pending.push(k);
                    pending.push(v)
                }
            }
            K::Lookup { value, key } | K::HasKey { value, key } => {
                pending.push(value);
                pending.push(key)
            }
            K::Default { value, fallback } => {
                pending.push(value);
                pending.push(fallback)
            }
            K::Binary { left, right, .. } | K::ZipExact { left, right } => {
                pending.push(left);
                pending.push(right)
            }
            K::ArrayAccess { array, index } => {
                pending.push(array);
                pending.push(index)
            }
            K::Range { lower, upper } => {
                pending.push(lower);
                pending.push(upper)
            }
            K::Require {
                condition, value, ..
            } => {
                pending.push(condition);
                pending.push(value)
            }
            K::Conditional {
                branches,
                otherwise,
            } => {
                for (c, v) in branches {
                    pending.push(c);
                    pending.push(v)
                }
                pending.push(otherwise)
            }
            K::Comprehension {
                clauses: generators,
                head,
                ..
            } => {
                clauses(generators, &mut pending);
                pending.push(head)
            }
            K::KeyedComprehension {
                clauses: generators,
                key,
                head,
                empty_axis_sources,
            } => {
                clauses(generators, &mut pending);
                pending.push(key);
                pending.push(head);
                if let Some(sources) = empty_axis_sources {
                    pending.extend(sources)
                }
            }
        }
    }
    value
}

pub(super) fn expression_roots(functions: &[FunctionDefinition]) -> Vec<&model::Expression> {
    let mut roots = Vec::new();
    for function in functions {
        for (_, value) in &function.declaration.bindings {
            report::visit_value(value, &mut roots);
        }
        report::visit_value(&function.declaration.value, &mut roots);
    }
    roots
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

fn contains_native(
    ty: &D<program::Kind>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<bool, zirql::QueryError> {
    budget.charge(1, ValueLocation::Query(range.clone()))?;
    Ok(match ty {
        D::Native(_) => true,
        D::Optional(value) | D::Sequence(value) | D::Array(value) | D::Keyed(value) => {
            contains_native(value, budget, range)?
        }
        D::Record(fields) => {
            let mut found = false;
            for (name, value) in fields {
                budget.charge(name.len() + 1, ValueLocation::Query(range.clone()))?;
                if contains_native(value, budget, range)? {
                    found = true;
                    break;
                }
            }
            found
        }
        D::Tuple(values) => {
            let mut found = false;
            for value in values {
                if contains_native(value, budget, range)? {
                    found = true;
                    break;
                }
            }
            found
        }
        _ => false,
    })
}
/// Only preparation copies lexical kind metadata; runtime shares value bindings.
pub(super) fn copy_scope(
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<BTreeMap<String, program::Kind>, zirql::QueryError> {
    budget.charge(
        names.keys().map(|name| name.len() + 1).sum(),
        ValueLocation::Query(range.clone()),
    )?;
    budget.charge_results(names.len(), ValueLocation::Query(range.clone()))?;
    Ok(names.clone())
}
