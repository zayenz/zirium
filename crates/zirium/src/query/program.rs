//! Native program lowering. Domain operations and emission snapshots stay in Zirium.
use super::*;
use std::ops::Range;
use zirql::{Bindings, ExpressionStep, Located, ProgramHost};

pub(super) type Expression = super::report::Operand;
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Emitter {
    Result,
    Written(model::NativeStage),
    AlreadyEmitted,
}

pub(super) fn range(range: crate::source::TextRange) -> Range<usize> {
    range.start() as usize..range.end() as usize
}

pub(super) fn lower(
    query: &Query,
    parameter_names: Vec<Located<String>>,
) -> zirql::Program<model::NativeStage, model::SetOperator, Emitter> {
    let mut statements = Vec::new();
    let mut parameters = parameter_names.into_iter().enumerate().peekable();
    for position in 0..=query.statements.len() {
        while parameters
            .peek()
            .is_some_and(|(index, _)| query.parameter_positions[*index] == position)
        {
            let (index, name) = parameters.next().expect("parameter position was checked");
            let location = name.range.clone();
            statements.push(Located::new(
                zirql::Statement::Parameter {
                    declaration: zirql::ParameterDeclaration {
                        name,
                        ty: query.parameters[index].ty.clone(),
                    },
                },
                location,
            ));
        }
        let Some(statement) = query.statements.get(position) else {
            break;
        };
        let (expression, statement) = match statement {
            model::Statement::Binding { name, expression } => (
                expression,
                zirql::Statement::Bind {
                    name: Located::new(name.clone(), range(expression.range)),
                    expression: lower_expression(expression),
                },
            ),
            model::Statement::PatternBinding { expression, .. } => {
                let [
                    Located {
                        value: ExpressionStep::Stage(stage),
                        ..
                    },
                ] = expression.first.as_slice()
                else {
                    unreachable!("frontend pattern initializer")
                };
                let model::Stage::PatternBinding { value, .. } = stage.as_ref() else {
                    unreachable!("frontend pattern initializer")
                };
                (
                    expression,
                    zirql::Statement::BindPattern {
                        pattern: std::sync::Arc::clone(&value.pattern),
                        value: std::sync::Arc::clone(&value.value.expression),
                    },
                )
            }
            model::Statement::Do(expression) => (
                expression,
                zirql::Statement::Evaluate {
                    expression: lower_expression(expression),
                    implicit_output: None,
                },
            ),
            model::Statement::Query(expression) => (
                expression,
                zirql::Statement::Evaluate {
                    expression: lower_expression(expression),
                    implicit_output: Some(Emitter::Result),
                },
            ),
        };
        statements.push(Located::new(statement, range(expression.range)));
    }
    statements.push(Located::new(
        zirql::Statement::Evaluate {
            expression: lower_expression(&query.expression),
            implicit_output: query.emit_final_result.then_some(Emitter::Result),
        },
        range(query.expression.range),
    ));
    zirql::Program::new(statements, range(query.expression.range))
}

pub(super) fn lower_expression(expression: &model::Expression) -> Expression {
    report::operand(expression)
}

/// Check Zirium's reserved projection/capture names inside native operands.
/// No common binding progression or prepared entries are retained here.
pub(super) fn admit_names<'q>(
    expressions: impl Iterator<Item = &'q model::Expression>,
    functions: &[functions::FunctionDefinition],
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    let mut pending: Vec<_> = expressions.collect();
    for function in functions {
        report::visit_value(&function.declaration.value, &mut pending, budget)?;
    }
    while let Some(expression) = pending.pop() {
        for step in expression
            .first
            .iter()
            .chain(expression.rest.iter().flat_map(|(_, steps)| steps))
        {
            report::visit_step(step, &mut pending, budget)?;
        }
    }
    Ok(())
}

pub(super) fn evaluation_error(error: zirql::EvaluationError<EvaluationError>) -> EvaluationError {
    match error {
        zirql::EvaluationError::Core(error) => core_error(error),
        zirql::EvaluationError::Host(error) => error,
    }
}

pub(super) fn admit(
    value: &QueryOutput,
    budget: &mut zirql::Budget,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    budget
        .charge_results(result_units(value), zirql::Location::Query(range.clone()))
        .map_err(core_error)
}

fn result_units(value: &QueryOutput) -> usize {
    match value {
        QueryOutput::Native(NativeValue::Map(entries)) => {
            1 + entries
                .iter()
                .map(|(_, value)| 1 + result_units(value))
                .sum::<usize>()
        }
        QueryOutput::Map(_) => 1 + output_size(value),
        _ => output_size(value).max(1),
    }
}

pub(super) fn charge_results(
    budget: &mut zirql::Budget,
    units: usize,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    budget
        .charge_results(units, zirql::Location::Query(range.clone()))
        .map_err(core_error)
}

pub(super) fn charge_bytes(
    budget: &mut zirql::Budget,
    bytes: usize,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    if budget.remaining_results().is_some() {
        budget
            .charge(bytes, zirql::Location::Query(range.clone()))
            .map_err(core_error)?;
    }
    Ok(())
}

/// Materialize report containers through Zirql; native descriptors stay here.
pub(super) fn to_json(
    document: &Document,
    value: &QueryOutput,
    budget: &mut zirql::Budget,
    range: &Range<usize>,
) -> Result<serde_json::Value, EvaluationError> {
    zirql::to_report_json_value(
        &mut ReportRenderer { document, range },
        value,
        budget,
        zirql::Location::Query(range.clone()),
    )
    .map_err(evaluation_error)
}

struct ReportRenderer<'a> {
    document: &'a Document,
    range: &'a Range<usize>,
}
impl zirql::RenderHost for ReportRenderer<'_> {
    type Value = QueryOutput;
    type Error = EvaluationError;

    fn view<'a>(
        &mut self,
        value: &'a QueryOutput,
        context: zirql::RenderContext,
        budget: &mut Budget,
        _location: &zirql::Location,
    ) -> Result<zirql::RenderView<'a, QueryOutput>, EvaluationError> {
        use zirql::RenderView as View;
        let document = self.document;
        let range = self.range;
        Ok(match value {
            QueryOutput::Native(NativeValue::Retained(value)) => View::Value(value),
            QueryOutput::Native(NativeValue::Computational(value)) => {
                return Err(EvaluationError::new(
                    "computational values cannot be rendered or serialized",
                )
                .at(if range.is_empty() {
                    value.range.clone()
                } else {
                    range.clone()
                }));
            }
            QueryOutput::Native(NativeValue::Tuple(value)) => View::Sequence(&value.values),
            QueryOutput::Native(NativeValue::Sequence(value)) => View::Sequence(&value.values),
            QueryOutput::Native(NativeValue::Map(fields)) => View::Record(fields),
            QueryOutput::Native(NativeValue::Dictionary(value)) => {
                View::Dictionary(value.entries())
            }
            QueryOutput::Values(values)
                if context == zirql::RenderContext::DictionaryKey && values.len() == 1 =>
            {
                charge_bytes(budget, values[0].len(), range)?;
                charge_results(budget, 1, range)?;
                View::Projected(serde_json::Value::String(values[0].clone()))
            }
            QueryOutput::Native(NativeValue::JsonValue(value)) => {
                charge_results(budget, json_size(value), range)?;
                charge_bytes(budget, json_bytes(value), range)?;
                View::Projected(value.clone())
            }
            QueryOutput::Native(NativeValue::Capture(capture)) => View::Capture(capture),
            QueryOutput::Native(NativeValue::Captures(captures)) => View::Captures(captures.rows()),
            QueryOutput::Native(NativeValue::Constructed(value)) => {
                if budget.remaining_results().is_none() {
                    return Err(EvaluationError::new(
                        "constructed reports require an aggregate result allowance",
                    )
                    .at(value.range()));
                }
                View::Constructed(value.value(), zirql::JsonRepresentation::Ordinary)
            }
            QueryOutput::Native(NativeValue::Ssa(value)) => {
                capture::check_value(document, *value, range)?;
                let (kind, index) = match value {
                    ValueId::OperationResult { result, .. } => ("operation_result", *result),
                    ValueId::BlockArgument { argument, .. } => ("block_argument", *argument),
                };
                let ty = document.value_type(ValueReference::Resolved(*value));
                charge_results(budget, 7, range)?;
                charge_bytes(budget, kind.len() + 15 + ty.map_or(0, str::len), range)?;
                View::Projected(serde_json::json!({"kind":kind,"index":index,"type":ty}))
            }
            _ => {
                let (units, bytes) = json_allocation(document, value);
                charge_results(budget, units, range)?;
                charge_bytes(budget, bytes, range)?;
                let mut projected = output_value(document, value);
                if context == zirql::RenderContext::CaptureField
                    && matches!(value, QueryOutput::Operations(values) if values.len() == 1)
                    && let serde_json::Value::Array(values) = &mut projected
                {
                    projected = values.pop().expect("one operation descriptor");
                }
                View::Projected(projected)
            }
        })
    }
}

fn json_allocation(document: &Document, value: &QueryOutput) -> (usize, usize) {
    match value {
        QueryOutput::Operations(rows)
        | QueryOutput::Native(NativeValue::OperationSequence(rows)) => {
            let mut units = 1;
            let mut bytes = 0;
            for &row in rows {
                units += 9;
                bytes += 40 + document.operation_name(row).map_or(0, str::len);
                for (key, value) in document.attributes(row).into_iter().flatten() {
                    units += 2;
                    bytes += key.len() + value.len();
                }
                for &operand in document.operands(row).unwrap_or(&[]) {
                    units += 1;
                    bytes += document.value_type(operand).map_or(0, str::len);
                }
                for &ty in document.result_types(row).unwrap_or(&[]) {
                    units += 1;
                    bytes += document.type_spelling(ty).map_or(0, str::len);
                }
            }
            (units, bytes)
        }
        QueryOutput::Native(NativeValue::Types(rows)) => (
            1 + rows.len(),
            rows.iter()
                .map(|&id| document.type_spelling(id).map_or(0, str::len))
                .sum(),
        ),
        QueryOutput::Native(NativeValue::Attributes(rows)) => (
            1 + rows.len(),
            rows.iter()
                .map(|(_, id)| document.attribute_spelling_value(*id).map_or(0, str::len))
                .sum(),
        ),
        QueryOutput::Values(rows) => (1 + rows.len(), rows.iter().map(String::len).sum()),
        QueryOutput::Map(rows) => (
            1 + rows.iter().map(|(_, v)| 1 + json_size(v)).sum::<usize>(),
            copied_bytes(value),
        ),
        QueryOutput::Array(rows) => (
            1 + rows.iter().map(json_size).sum::<usize>(),
            copied_bytes(value),
        ),
        _ => (1, copied_bytes(value)),
    }
}

pub(super) fn serialize_json(
    value: &serde_json::Value,
    budget: &mut zirql::Budget,
    range: &Range<usize>,
    pretty: bool,
) -> Result<String, EvaluationError> {
    let layout = if pretty {
        zirql::JsonLayout::Pretty
    } else {
        zirql::JsonLayout::Compact
    };
    let mut bytes =
        zirql::render_json_value(value, layout, budget, zirql::Location::Query(range.clone()))
            .map_err(core_error)?;
    charge_results(budget, 1, range)?;
    charge_bytes(budget, 1, range)?;
    bytes.push(b'\n');
    Ok(String::from_utf8(bytes).expect("JSON is UTF-8"))
}

fn json(
    document: &Document,
    value: &QueryOutput,
    budget: &mut zirql::Budget,
    range: &Range<usize>,
) -> Result<String, EvaluationError> {
    let value = to_json(document, value, budget, range)?;
    serialize_json(&value, budget, range, true)
}

pub(super) fn copy_value(
    value: &QueryOutput,
    budget: &mut zirql::Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    if matches!(
        value,
        QueryOutput::Native(
            NativeValue::Computational(_)
                | NativeValue::Tuple(_)
                | NativeValue::Sequence(_)
                | NativeValue::Dictionary(_)
        )
    ) {
        charge_results(budget, 1, range)?;
        return Ok(value.clone());
    }
    if let QueryOutput::Native(NativeValue::Retained(value)) = value {
        charge_results(budget, 1, range)?;
        return Ok(QueryOutput::Native(NativeValue::Retained(
            std::sync::Arc::clone(value),
        )));
    }
    if let QueryOutput::Native(NativeValue::Captures(value)) = value {
        return value
            .copy(budget, range)
            .map(|value| QueryOutput::Native(NativeValue::Captures(value)));
    }
    if let QueryOutput::Native(NativeValue::Capture(value)) = value {
        return capture::copy_record(value, budget, range)
            .map(|value| QueryOutput::Native(NativeValue::Capture(value)));
    }
    if let QueryOutput::Native(NativeValue::Constructed(value)) = value {
        return value
            .copy(budget, range)
            .map(|value| QueryOutput::Native(NativeValue::Constructed(value)));
    }
    if let QueryOutput::Native(NativeValue::Map(entries)) = value {
        charge_results(budget, 1 + entries.len(), range)?;
        let mut result = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            charge_bytes(budget, key.len(), range)?;
            result.push((key.clone(), copy_value(value, budget, range)?));
        }
        return Ok(QueryOutput::Native(NativeValue::Map(result)));
    }
    let textual_map = if let QueryOutput::Map(entries) = value {
        // Check entry storage before scanning keys and nested values.
        let units = entries
            .len()
            .checked_add(1)
            .filter(|_| {
                entries
                    .len()
                    .checked_mul(std::mem::size_of::<(String, serde_json::Value)>())
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
            })
            .ok_or_else(|| {
                EvaluationError::new("native map storage limit exceeded").at(range.clone())
            })?;
        budget
            .charge(units, zirql::Location::Query(range.clone()))
            .map_err(core_error)?;
        true
    } else {
        false
    };
    admit(value, budget, range)?;
    if textual_map
        || matches!(value, QueryOutput::Native(NativeValue::JsonValue(_)))
        || budget.remaining_results().is_some()
    {
        budget
            .charge(copied_bytes(value), zirql::Location::Query(range.clone()))
            .map_err(core_error)?;
    }
    Ok(value.clone())
}

fn copied_bytes(value: &QueryOutput) -> usize {
    match value {
        QueryOutput::Values(values) => values.iter().map(String::len).sum(),
        QueryOutput::Native(NativeValue::Attributes(values)) => {
            values.iter().map(|(name, _)| name.len()).sum()
        }
        QueryOutput::Json(value)
        | QueryOutput::Text(value)
        | QueryOutput::Native(NativeValue::String(value)) => value.len(),
        QueryOutput::Native(NativeValue::Map(entries)) => entries
            .iter()
            .map(|(key, value)| key.len() + copied_bytes(value))
            .sum(),
        QueryOutput::Map(entries) => entries
            .iter()
            .map(|(key, value)| key.len() + json_bytes(value))
            .sum(),
        QueryOutput::Array(values) => values.iter().map(json_bytes).sum(),
        QueryOutput::Native(NativeValue::JsonValue(value)) => json_bytes(value),
        _ => 0,
    }
}
fn json_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(value) => value.len(),
        serde_json::Value::Array(values) => values.iter().map(json_bytes).sum(),
        serde_json::Value::Object(values) => values
            .iter()
            .map(|(key, value)| key.len() + json_bytes(value))
            .sum(),
        _ => 0,
    }
}

pub(super) fn evaluate_native_stage(
    stage: &model::Stage,
    input: QueryOutput,
    context: &mut EvaluationContext<'_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    location: &Range<usize>,
) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
    let result = charge(budget, output_size(&input).max(1))
        .and_then(|()| evaluate_host_stage(stage, context, input, bindings, budget));
    let value = result.map_err(|error| zirql::EvaluationError::Host(error.at(location.clone())))?;
    check_items(budget, output_size(&value)).map_err(zirql::EvaluationError::Host)?;
    Ok(value)
}

pub(super) struct NativeHost;
impl ProgramHost<EvaluationContext<'_, '_>> for NativeHost {
    type Value = QueryOutput;
    type Stage = model::NativeStage;
    type SetOperator = model::SetOperator;
    type Emitter = Emitter;
    type Error = EvaluationError;

    fn copy_value(
        &self,
        value: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut zirql::Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        if let Some(value) = worklists::share(value, &report::ReportHost, context, budget, range)? {
            return Ok(value);
        }
        copy_value(value, budget, range).map_err(zirql::EvaluationError::Host)
    }
    fn evaluate_stage(
        &self,
        stage: &model::NativeStage,
        input: QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        bindings: &Bindings<QueryOutput>,
        budget: &mut zirql::Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        let result = match stage.as_ref() {
            native if pipeline::is_stream_stage(native) && pipeline::is_stream(&input) => {
                let pipeline = pipeline::lower(std::slice::from_ref(native))
                    .map_err(zirql::EvaluationError::Host)?;
                pipeline::evaluate(&pipeline, context, input, bindings, budget)
            }
            native => charge(budget, output_size(&input).max(1))
                .and_then(|()| evaluate_host_stage(native, context, input, bindings, budget)),
        };
        let value =
            result.map_err(|error| zirql::EvaluationError::Host(error.at(range.clone())))?;
        check_items(budget, output_size(&value))
            .map_err(|error| zirql::EvaluationError::Host(error.at(range.clone())))?;
        Ok(value)
    }
    fn combine(
        &self,
        operator: &model::SetOperator,
        left: QueryOutput,
        right: QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut zirql::Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        evaluate_set_operator(context.document, *operator, left, right, budget, range)
            .map_err(|error| zirql::EvaluationError::Host(error.at(range.clone())))
    }
    fn construct_record(
        &self,
        fields: Vec<(String, QueryOutput)>,
        _: &mut EvaluationContext<'_, '_>,
        _: &mut zirql::Budget,
        _: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        Ok(QueryOutput::Native(NativeValue::Capture(
            zirql::NativeCapture { fields },
        )))
    }
    fn capture_record<'v>(
        &self,
        value: &'v QueryOutput,
    ) -> Option<&'v zirql::NativeCapture<QueryOutput>> {
        match value {
            QueryOutput::Native(NativeValue::Capture(record)) => Some(record),
            _ => None,
        }
    }
    fn emit(
        &self,
        emitter: &Emitter,
        value: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut zirql::Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        match emitter {
            Emitter::Result => {
                let value = self.copy_value(value, context, budget, range)?;
                (context.emit)(context.document, value, budget)
                    .map_err(zirql::EvaluationError::Host)?;
            }
            Emitter::Written(stage) => {
                charge(budget, output_size(value).max(1)).map_err(zirql::EvaluationError::Host)?;
                let output = match stage.as_ref() {
                    model::Stage::Emit { .. } => self
                        .copy_value(value, context, budget, range)
                        .map_err(evaluation_error),
                    model::Stage::Format { options, .. } => {
                        let operations =
                            operations(value, "format").map_err(zirql::EvaluationError::Host)?;
                        charge_results(budget, operations.len(), range)
                            .map_err(zirql::EvaluationError::Host)?;
                        Ok(QueryOutput::Formatted {
                            operations: operations.to_vec(),
                            options: *options,
                        })
                    }
                    model::Stage::Json { .. } => {
                        json(context.document, value, budget, range).map(QueryOutput::Json)
                    }
                    model::Stage::Markdown { .. } => {
                        charge_results(budget, 1, range).map_err(zirql::EvaluationError::Host)?;
                        zirql::render_report_markdown(
                            &mut ReportRenderer {
                                document: context.document,
                                range,
                            },
                            value,
                            budget,
                            zirql::Location::Query(range.clone()),
                        )
                        .map(|bytes| {
                            QueryOutput::Text(String::from_utf8(bytes).expect("Markdown is UTF-8"))
                        })
                        .map_err(evaluation_error)
                    }
                    _ => unreachable!("written emitter"),
                }
                .map_err(|error| zirql::EvaluationError::Host(error.at(range.clone())))?;
                (context.emit)(context.document, output, budget)
                    .map_err(zirql::EvaluationError::Host)?;
            }
            Emitter::AlreadyEmitted => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    JsonValue,
    Computational(std::sync::Arc<zirql::DeclaredType<Kind>>),
    Tuple(std::sync::Arc<Vec<Kind>>),
    Sequence(std::sync::Arc<zirql::DeclaredType<Kind>>),
    Operations,
    Ssa,
    Capture(std::sync::Arc<zirql::NativeCapture<Kind>>),
    Captures(std::sync::Arc<zirql::NativeCapture<Kind>>),
    Strings,
    Count,
    Map,
    NativeMap,
    Dictionary(std::sync::Arc<(Kind, Kind)>),
    Array,
    String,
    Types,
    Attributes,
    Constructed(std::sync::Arc<Vec<zirql::ValueType>>),
}

pub(super) fn parameter_names(
    declarations: &[ParameterDeclaration],
    budget: &mut Budget,
) -> Result<BTreeMap<String, Kind>, EvaluationError> {
    let mut names = BTreeMap::new();
    for declaration in declarations {
        charge_bytes(
            budget,
            declaration.name.value.len() + 3,
            &declaration.name.range,
        )?;
        charge_results(budget, 3, &declaration.name.range)?;
        names.insert(
            declaration.name.value.clone(),
            Kind::Constructed(std::sync::Arc::new(vec![declaration.ty.value.into()])),
        );
    }
    Ok(names)
}

pub(super) fn admit_parameters(
    declarations: &[ParameterDeclaration],
    parameters: ParameterInputs<'_>,
    ambient: &[&str],
    range: Range<usize>,
    budget: &mut Budget,
) -> Result<Vec<(Located<String>, QueryOutput)>, EvaluationError> {
    for declaration in declarations {
        charge_bytes(
            budget,
            declaration.name.value.len() + 1,
            &declaration.name.range,
        )?;
        if !zirql::is_unquoted_binding_name(&declaration.name.value)
            || parser::is_reserved(&declaration.name.value)
        {
            return Err(EvaluationError::new(
                "parameter name is reserved or not an ASCII identifier",
            )
            .at(declaration.name.range.clone()));
        }
    }
    zirql::admit_parameters(declarations, parameters, ambient, range, budget)
        .map_err(core_error)?
        .into_iter()
        .map(|(name, value)| {
            let report = ConstructedReport::new(value, name.range.clone(), budget)?;
            Ok((name, QueryOutput::Native(NativeValue::Constructed(report))))
        })
        .collect()
}

pub(super) struct Validator<'f> {
    pub(super) host: report::ReportHost<'f>,
}
impl Validator<'_> {
    pub(super) fn expression(
        &self,
        expression: &model::Expression,
        input: Kind,
        names: &BTreeMap<String, Kind>,
        budget: &mut zirql::Budget,
    ) -> Result<Kind, EvaluationError> {
        if report::needs_preparation(expression) {
            return zirql::pattern_function_expression_kind(
                self.host
                    .functions
                    .iter()
                    .map(|function| function.declaration.as_ref()),
                &self.host,
                &report::operand(expression),
                names,
                Some(input),
                true,
                budget,
            )
            .map_err(core_error);
        }
        fn pipeline(
            validator: &Validator<'_>,
            steps: &[model::Step],
            mut kind: Kind,
            names: &BTreeMap<String, Kind>,
            budget: &mut Budget,
        ) -> Result<Kind, EvaluationError> {
            for step in steps {
                let ExpressionStep::Stage(stage) = &step.value else {
                    return Err(EvaluationError::new(
                        "constructed expression requires preparation",
                    )
                    .at(step.range.clone()));
                };
                kind =
                    validator.stages(std::slice::from_ref(stage.as_ref()), kind, names, budget)?;
            }
            Ok(kind)
        }
        let left = pipeline(self, &expression.first, input.clone(), names, budget)?;
        for (_, steps) in &expression.rest {
            let right = pipeline(self, steps, input.clone(), names, budget)?;
            if left != right || !matches!(left, Kind::Operations | Kind::Strings) {
                return Err(EvaluationError::new(
                    "set operands require operation or value streams of the same kind",
                )
                .at(range(expression.range)));
            }
        }
        Ok(left)
    }

    pub(super) fn stages(
        &self,
        stages: &[model::Stage],
        mut kind: Kind,
        names: &BTreeMap<String, Kind>,
        budget: &mut zirql::Budget,
    ) -> Result<Kind, EvaluationError> {
        use model::{Stage as S, StructuredStage as N};
        for stage in stages {
            let failure = |message: &str| EvaluationError::new(message).at(range(stage.range()));
            let require = |valid, message: &str| if valid { Ok(()) } else { Err(failure(message)) };
            match stage {
                S::PatternBinding { .. } => {
                    return Err(failure("pattern binding requires program preparation"));
                }
                S::FunctionInput { value, .. } => {
                    kind = report::output_kind(value, budget, &range(stage.range()))?;
                }
                S::Native { value, .. } => {
                    kind = capture::operand_kind(value, &kind, names, budget, &self.host)?;
                }
                S::Operand { operation, .. } => {
                    require(
                        capture::operand_kind(operation, &kind, names, budget, &self.host)?
                            == Kind::Operations,
                        "operand requires native operations",
                    )?;
                    kind = Kind::Ssa;
                }
                S::Computed { value, range } => {
                    report::validate_argument_with_host(
                        value.operand(),
                        value.parameter_type().into(),
                        names,
                        budget,
                        &self.host,
                    )?;
                    kind = self.stages(
                        std::slice::from_ref(&value.placeholder(*range)),
                        kind,
                        names,
                        budget,
                    )?;
                }
                S::Input { .. } => kind = Kind::Operations,
                S::Binding { name, .. } => {
                    kind = names.get(name).cloned().ok_or_else(|| {
                        failure(&format!(
                            "binding `{name}` is unavailable in this evaluation context"
                        ))
                    })?
                }
                S::Literal { value, .. } => {
                    validate_literal(value, names)
                        .map_err(|error| error.at(range(stage.range())))?;
                    kind = match value {
                        model::JsonLiteral::Object(_) => Kind::Map,
                        model::JsonLiteral::Array(_) => Kind::Array,
                        _ => unreachable!(),
                    }
                }
                S::Group { expression, .. } => {
                    kind = self.expression(expression, kind, names, budget)?
                }
                S::Guard { predicate, .. } | S::Root { predicate, .. } => {
                    require(
                        kind == Kind::Operations,
                        "stage requires operations, but the current stream contains values",
                    )?;
                    report::validate_predicate_with_host(predicate, names, budget, &self.host)?;
                }
                S::Reachable { .. }
                | S::Closure { .. }
                | S::ClosureFixedPoint { .. }
                | S::Slice { .. }
                | S::ForwardSlice { .. }
                | S::Defs { .. }
                | S::Users { .. }
                | S::Parent { .. }
                | S::Children { .. }
                | S::Subtree { .. }
                | S::SetAttr { .. }
                | S::RemoveAttr { .. } => require(
                    kind == Kind::Operations,
                    "stage requires operations, but the current stream contains values",
                )?,
                S::Names { .. }
                | S::Attr { .. }
                | S::ResultTypes { .. }
                | S::OperandTypes { .. } => {
                    require(kind == Kind::Operations, "projection requires operations")?;
                    kind = Kind::Strings;
                }
                S::Structured { step, .. } => match step {
                    N::One => {
                        require(kind == Kind::Strings, "one requires a string stream")?;
                        kind = Kind::String;
                    }
                    N::StringAttr(_) => {
                        require(kind == Kind::Operations, "string_attr requires operations")?;
                        kind = Kind::Strings;
                    }
                    N::Types { .. } => {
                        require(kind == Kind::Operations, "types requires operations")?;
                        kind = Kind::Types;
                    }
                    N::Attributes(_) => {
                        require(kind == Kind::Operations, "attributes requires operations")?;
                        kind = Kind::Attributes;
                    }
                    N::Spellings => {
                        require(
                            matches!(kind, Kind::Types | Kind::Attributes),
                            "spellings requires native types or attributes",
                        )?;
                        kind = Kind::Strings;
                    }
                },
                S::Format { .. } => {
                    require(kind == Kind::Operations, "format requires operations")?
                }
                S::Markdown { .. } => require(
                    matches!(
                        kind,
                        Kind::Map
                            | Kind::Dictionary(_)
                            | Kind::NativeMap
                            | Kind::Array
                            | Kind::Strings
                            | Kind::Count
                            | Kind::String
                            | Kind::Constructed(_)
                            | Kind::Tuple(_)
                            | Kind::Sequence(_)
                            | Kind::Capture(_)
                            | Kind::Captures(_)
                    ),
                    "markdown requires report data; project operations to rows first",
                )?,
                S::Print { parts, .. } => {
                    validate_interpolation(parts, names)
                        .map_err(|error| error.at(range(stage.range())))?;
                }
                S::Emit { .. } | S::Json { .. } => {}
                S::Value { .. } => {
                    return Err(failure(
                        "value is only valid as the selector in sort_by(value) on a map",
                    ));
                }
            }
        }
        Ok(kind)
    }
}

fn validate_interpolation(
    parts: &[model::PrintPart],
    names: &BTreeMap<String, Kind>,
) -> Result<(), EvaluationError> {
    for part in parts {
        if let model::PrintPart::Binding(name) = part {
            let kind = names.get(name).ok_or_else(|| {
                EvaluationError::new(format!(
                    "binding `{name}` is unavailable in this evaluation context"
                ))
            })?;
            if !matches!(kind, Kind::Strings | Kind::Count | Kind::Array) {
                return Err(EvaluationError::new(
                    "string interpolation requires a count or exactly one string",
                ));
            }
        }
    }
    Ok(())
}

fn validate_literal(
    value: &model::JsonLiteral,
    names: &BTreeMap<String, Kind>,
) -> Result<(), EvaluationError> {
    let binding = |name: &str| {
        if names.contains_key(name) {
            Ok(())
        } else {
            Err(EvaluationError::new(format!(
                "binding `{name}` is unavailable in this evaluation context"
            )))
        }
    };
    match value {
        model::JsonLiteral::Binding(name) => binding(name)?,
        model::JsonLiteral::String(text) => validate_interpolation(text, names)?,
        model::JsonLiteral::Array(values) => {
            for value in values {
                validate_literal(value, names)?;
            }
        }
        model::JsonLiteral::Object(fields) => {
            for (key, value) in fields {
                validate_interpolation(key, names)?;
                validate_literal(value, names)?;
            }
        }
        model::JsonLiteral::Scalar(_) => {}
    }
    Ok(())
}
