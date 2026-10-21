//! Native domain adaptation for shared correlated projection and report values.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::FunctionHost;
use zirql::{
    Bindings, Budget, ConstructedValue as V, ExpressionKind as X, ExpressionStep as Step, Located,
    Location, ProgramHost, ProjectionHost, ValueType,
};

pub(super) type Operand =
    zirql::Expression<model::NativeStage, model::SetOperator, program::Emitter>;
fn map_keys(
    value: QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<Option<Vec<String>>, EvaluationError> {
    Ok(match functions::owned(value, budget, range)? {
        QueryOutput::Values(keys) => Some(keys),
        QueryOutput::Native(NativeValue::String(key)) => {
            program::charge_results(budget, 1, range)?;
            Some(vec![key])
        }
        _ => None,
    })
}
pub(super) fn operand_is_read_only(query: &Operand) -> bool {
    let mut pending = vec![query];
    while let Some(query) = pending.pop() {
        match &query.value {
            X::Input | X::Binding(_) => {}
            X::Group(query) | X::Field { value: query, .. } => pending.push(query),
            X::Sequence { input, steps } => {
                if !steps.iter().all(|step| match &step.value {
                    Step::Stage(stage) => stage.is_read_only(),
                    Step::Emit { .. } => false,
                    _ => true,
                }) {
                    return false;
                }
                pending.push(input);
            }
            X::Set { left, right, .. } => {
                pending.push(right);
                pending.push(left);
            }
            X::Record(fields) => pending.extend(fields.iter().rev().map(|(_, value)| value)),
        }
    }
    true
}

pub(super) fn operand(expression: &model::Expression) -> Operand {
    fn stages(stages: &[model::Step], range: Range<usize>) -> Operand {
        let mut input = Located::new(X::Input, range.clone());
        let mut steps = vec![];
        for (index, step) in stages.iter().enumerate() {
            if let Step::Stage(stage) = &step.value {
                match stage.as_ref() {
                    model::Stage::Binding { name, .. } if index == 0 => {
                        input = Located::new(X::Binding(name.clone()), step.range.clone());
                        continue;
                    }
                    model::Stage::Group { expression, .. } if index == 0 => {
                        input = Located::new(
                            X::Group(Box::new(operand(expression))),
                            step.range.clone(),
                        );
                        continue;
                    }
                    model::Stage::Group { expression, .. } => {
                        steps.push(Located::new(
                            Step::Group(
                                zirql::NativeBody::new(operand(expression), step.range.clone())
                                    .into(),
                            ),
                            step.range.clone(),
                        ));
                        continue;
                    }
                    model::Stage::Emit { .. }
                    | model::Stage::Format { .. }
                    | model::Stage::Json { .. }
                    | model::Stage::Markdown { .. } => {
                        steps.push(Located::new(
                            Step::Emit {
                                emitter: program::Emitter::Written(Arc::clone(stage)),
                                policy: zirql::EmissionPolicy::PassThrough,
                            },
                            step.range.clone(),
                        ));
                        continue;
                    }
                    _ => {}
                }
                steps.push(step.clone());
                if matches!(stage.as_ref(), model::Stage::Print { .. }) {
                    steps.push(Located::new(
                        Step::Emit {
                            emitter: program::Emitter::AlreadyEmitted,
                            policy: zirql::EmissionPolicy::PassThrough,
                        },
                        step.range.clone(),
                    ));
                }
            } else {
                steps.push(step.clone());
            }
        }
        if steps.is_empty() {
            input
        } else {
            Located::new(
                X::Sequence {
                    input: Box::new(input),
                    steps,
                },
                range,
            )
        }
    }
    let range = program::range(expression.range);
    let mut value = stages(&expression.first, range.clone());
    for (operator, right) in &expression.rest {
        value = Located::new(
            X::Set {
                operator: *operator,
                left: Box::new(value),
                right: Box::new(stages(right, range.clone())),
            },
            range.clone(),
        );
    }
    value
}

pub(super) fn lower_suffix(
    suffix: Located<zirql::ValueSuffix<Operand>>,
    budget: &mut Budget,
) -> Result<comprehension::ParsedComprehension, zirql::QueryError> {
    let range = suffix.range.clone();
    budget.charge_results(3, Location::Query(range.clone()))?;
    budget.charge(
        range.len() + std::mem::size_of::<zirql::ValueExpression<Operand>>() * 3,
        Location::Query(range.clone()),
    )?;
    let value = zirql::lower_value_suffix(
        Located::new(
            zirql::ValueExpressionKind::NativeQuery(Box::new(Located::new(
                X::Input,
                range.clone(),
            ))),
            range,
        ),
        suffix,
    )?;
    Ok(comprehension::ParsedComprehension::new(value))
}

pub(super) fn output_kind(
    value: &QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<program::Kind, EvaluationError> {
    use program::Kind as K;
    Ok(match functions::unretained(value) {
        QueryOutput::Native(NativeValue::JsonValue(_)) => K::JsonValue,
        QueryOutput::Native(NativeValue::Computational(value)) => K::Computational(Arc::new(
            value
                .value
                .declaration(budget, &Location::Query(range.clone()))
                .map_err(core_error)?,
        )),
        QueryOutput::Native(NativeValue::Tuple(value)) => {
            capture::slots::<K>(value.values.len(), budget, range)?;
            K::Tuple(Arc::new(
                value
                    .values
                    .iter()
                    .map(|v| output_kind(v, budget, range))
                    .collect::<Result<_, _>>()?,
            ))
        }
        QueryOutput::Native(NativeValue::Sequence(value)) => {
            K::Sequence(Arc::clone(&value.element))
        }
        QueryOutput::Native(NativeValue::Ssa(_)) => K::Ssa,
        QueryOutput::Native(NativeValue::Captures(value)) => K::Captures(Arc::clone(&value.schema)),
        QueryOutput::Native(NativeValue::Capture(record)) => {
            capture::slots::<(String, K)>(record.fields.len(), budget, range)?;
            let mut fields = Vec::with_capacity(record.fields.len());
            for (name, value) in &record.fields {
                program::charge_bytes(budget, name.len(), range)?;
                fields.push((name.clone(), output_kind(value, budget, range)?));
            }
            K::Capture(Arc::new(zirql::NativeCapture { fields }))
        }
        QueryOutput::Operations(_) | QueryOutput::Native(NativeValue::OperationSequence(_)) => {
            K::Operations
        }
        QueryOutput::Values(_) => K::Strings,
        QueryOutput::Count(_) => K::Count,
        QueryOutput::Map(_) => K::Map,
        QueryOutput::Native(NativeValue::Map(_)) => K::NativeMap,
        QueryOutput::Array(_) => K::Array,
        QueryOutput::Native(NativeValue::String(_)) => K::String,
        QueryOutput::Native(NativeValue::Types(_)) => K::Types,
        QueryOutput::Native(NativeValue::Attributes(_)) => K::Attributes,
        QueryOutput::Native(NativeValue::Constructed(value)) => {
            K::Constructed(Arc::clone(&value.types))
        }
        _ => K::Operations,
    })
}

pub(super) struct ReportHost<'f> {
    pub(super) functions: &'f [FunctionDefinition],
}
#[allow(non_upper_case_globals)]
pub(super) const ReportHost: ReportHost<'static> = ReportHost { functions: &[] };
impl ProgramHost<EvaluationContext<'_, '_>> for ReportHost<'_> {
    type Value = QueryOutput;
    type Stage = model::NativeStage;
    type SetOperator = model::SetOperator;
    type Emitter = program::Emitter;
    type Error = EvaluationError;
    fn copy_value(
        &self,
        value: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        if let Some(value) = worklists::share(value, self, context, budget, range)? {
            return Ok(value);
        }
        if let QueryOutput::Native(NativeValue::Constructed(value)) = functions::unretained(value) {
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(QueryOutput::Native(NativeValue::Constructed(value.clone())));
        }
        program::copy_value(value, budget, range).map_err(zirql::EvaluationError::Host)
    }
    fn evaluate_stage(
        &self,
        stage: &model::NativeStage,
        input: QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        bindings: &Bindings<QueryOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        program::NativeHost.evaluate_stage(stage, input, context, bindings, budget, range)
    }
    fn combine(
        &self,
        operator: &model::SetOperator,
        left: QueryOutput,
        right: QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        program::NativeHost.combine(operator, left, right, context, budget, range)
    }
    fn construct_record(
        &self,
        fields: Vec<(String, QueryOutput)>,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        program::NativeHost.construct_record(fields, context, budget, range)
    }
    fn positional_view<'v>(&self, value: &'v QueryOutput) -> Option<&'v [QueryOutput]> {
        match functions::unretained(value) {
            QueryOutput::Native(NativeValue::Sequence(value)) => Some(&value.values),
            _ => None,
        }
    }
    fn tuple_view<'v>(&self, value: &'v QueryOutput) -> Option<&'v [QueryOutput]> {
        match functions::unretained(value) {
            QueryOutput::Native(NativeValue::Tuple(value)) => Some(&value.values),
            _ => None,
        }
    }
    fn capture_record<'v>(
        &self,
        value: &'v QueryOutput,
    ) -> Option<&'v zirql::NativeCapture<QueryOutput>> {
        match functions::unretained(value) {
            QueryOutput::Native(NativeValue::Capture(record)) => Some(record),
            _ => None,
        }
    }
    fn emit(
        &self,
        emitter: &program::Emitter,
        value: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        program::NativeHost.emit(emitter, value, context, budget, range)
    }
}
impl ProjectionHost<EvaluationContext<'_, '_>> for ReportHost<'_> {
    type Kind = program::Kind;
    fn is_constructed_kind(&self, kind: &program::Kind) -> bool {
        matches!(kind, program::Kind::Constructed(_))
    }
    fn collection_argument_kind(
        &self,
        _: &program::Kind,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        Ok(program::Kind::Operations)
    }
    fn collection_argument_input(
        &self,
        _: &QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        budget.charge_results(1, Location::Query(range.clone()))?;
        Ok(QueryOutput::Operations(Vec::new()))
    }

    fn collection_kind(
        &self,
        operation: &zirql::Stage<()>,
        input: &program::Kind,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        use program::Kind as K;
        use zirql::Stage as S;
        if let K::Constructed(types) = input {
            return zirql::collection_value_types(operation, types, budget, range)
                .map(|types| self.constructed_kind(types));
        }
        let stream = matches!(input, K::Operations | K::Strings | K::Types | K::Attributes);
        let collection = stream
            || matches!(
                input,
                K::Map | K::NativeMap | K::Array | K::Captures(_) | K::Sequence(_)
            );
        match operation {
            S::Unique if !stream => Err(zirql::QueryError::query(
                range.clone(),
                "unique requires an operation or value stream",
            )),
            S::Tally if *input != K::Strings => Err(zirql::QueryError::query(
                range.clone(),
                "tally requires a value stream; use names or attr first",
            )),
            S::Count if collection => Ok(K::Count),
            S::Tally => Ok(K::Map),
            _ if collection => Ok(input.clone()),
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "stage requires a stream, map, or array",
            )),
        }
    }
    fn collection_identity(
        &self,
        item: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeCollectionKey, zirql::EvaluationError<EvaluationError>> {
        use zirql::{NativeCollectionKey as K, NativeIdentity};
        budget.charge(1, Location::Query(range.clone()))?;
        match functions::unretained(item) {
            QueryOutput::Operations(_) => {
                worklists::identity(item, &program::Kind::Operations, context, budget, range)
                    .map(K::Native)
            }
            QueryOutput::Values(values) if values.len() == 1 => {
                program::charge_bytes(budget, values[0].len(), range)
                    .map_err(zirql::EvaluationError::Host)?;
                Ok(K::Text(values[0].clone()))
            }
            QueryOutput::Native(NativeValue::Types(values)) if values.len() == 1 => {
                let id = values[0];
                if context.document.type_spelling(id).is_none() {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "invalid native type identity",
                    )
                    .into());
                }
                Ok(K::Native(NativeIdentity(id.native_identity())))
            }
            QueryOutput::Native(NativeValue::Attributes(values)) if values.len() == 1 => {
                let (name, id) = &values[0];
                if context.document.attribute_value(*id).is_none() {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "invalid native attribute identity",
                    )
                    .into());
                }
                program::charge_bytes(budget, name.len(), range)
                    .map_err(zirql::EvaluationError::Host)?;
                Ok(K::Named(name.clone(), NativeIdentity(id.native_identity())))
            }
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "identity requires one admitted native subject",
            )
            .into()),
        }
    }
    fn collection_tally_key(
        &self,
        item: QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<String, zirql::EvaluationError<EvaluationError>> {
        if let QueryOutput::Values(mut values) = item
            && values.len() == 1
        {
            Ok(values.pop().unwrap())
        } else {
            Err(zirql::QueryError::query(range.clone(), "tally requires a value stream").into())
        }
    }
    fn restore_collection(
        &self,
        input: QueryOutput,
        result: zirql::NativeValue<QueryOutput>,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        match result {
            zirql::NativeValue::Stream(items) => self.restore_selection(
                input,
                items,
                zirql::SelectionShape::Preserve,
                context,
                budget,
                range,
            ),
            zirql::NativeValue::Count(count) => Ok(QueryOutput::Count(count)),
            zirql::NativeValue::Tally(entries) => Ok(QueryOutput::Map(
                entries
                    .into_iter()
                    .map(|(key, count)| (key, serde_json::Value::from(count)))
                    .collect(),
            )),
        }
    }

    fn validate_path_edge(
        &self,
        edge: &model::NativeStage,
        input: &program::Kind,
        _: &BTreeMap<String, program::Kind>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        if *input == program::Kind::Operations
            && matches!(edge.as_ref(), model::Stage::Users { index: None, .. })
        {
            return Ok(());
        }
        Err(zirql::QueryError::query(
            range.clone(),
            "users path edge requires native operations",
        ))
    }
    fn path_neighbors(
        &self,
        _: &model::NativeStage,
        input: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        _: &Bindings<QueryOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
        let operation = capture::path_operation(input, context.document, range)
            .map_err(zirql::EvaluationError::Host)?;
        let neighbors = evaluate_users(context.document, &[operation], None, budget)
            .map_err(zirql::EvaluationError::Host)?;
        capture::slots::<QueryOutput>(neighbors.len(), budget, range)
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
        context: &mut EvaluationContext<'_, '_>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<bool, zirql::EvaluationError<EvaluationError>> {
        Ok(capture::path_operation(left, context.document, range)
            .map_err(zirql::EvaluationError::Host)?
            == capture::path_operation(right, context.document, range)
                .map_err(zirql::EvaluationError::Host)?)
    }
    fn path_steps(
        &self,
        steps: Vec<QueryOutput>,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        capture::slots::<OperationId>(steps.len(), budget, range)
            .map_err(zirql::EvaluationError::Host)?;
        budget.charge_results(steps.len() + 1, zirql::Location::Query(range.clone()))?;
        let mut operations = Vec::with_capacity(steps.len());
        for step in steps {
            operations.push(
                capture::path_operation(&step, context.document, range)
                    .map_err(zirql::EvaluationError::Host)?,
            );
        }
        Ok(QueryOutput::Native(NativeValue::OperationSequence(
            operations,
        )))
    }
    fn validate_guard_leaf(
        &self,
        payload: &model::NativeStage,
        input: &program::Kind,
        scope: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        let model::Stage::Guard {
            predicate: guard, ..
        } = payload.as_ref()
        else {
            return Err(zirql::QueryError::query(
                range.clone(),
                "expected predicate guard",
            ));
        };
        match guard {
            model::Predicate::Expression { value, .. } => {
                let kind =
                    functions::value_kind(self, &value.expression, input.clone(), scope, budget)?;
                let types = self.scalar_types(&kind, budget, &value.expression.range)?;
                let Some(types) = types else {
                    return Err(zirql::QueryError::query(
                        value.expression.range.clone(),
                        "match predicate needs a known Boolean",
                    ));
                };
                for ty in types {
                    report::admit_scalar_type(
                        &ty,
                        &zirql::ValueType::Boolean,
                        budget,
                        &value.expression.range,
                        "match predicate needs a known Boolean",
                    )?;
                }
                Ok(())
            }
            model::Predicate::Bool { .. } => Ok(()),
            model::Predicate::SameValue { left, right, .. } => {
                for value in [left, right] {
                    let kind = capture::operand_kind(value, input, scope, budget, self)
                        .map_err(report::query_error)?;
                    if kind != program::Kind::Ssa {
                        return Err(zirql::QueryError::query(
                            value.value.expression.range.clone(),
                            "same_value requires native SSA values",
                        ));
                    }
                }
                Ok(())
            }
            _ => {
                if *input != program::Kind::Operations {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "predicate requires operations",
                    ));
                }
                validate_predicate_with_host(guard, scope, budget, self).map_err(query_error)
            }
        }
    }
    fn evaluate_guard_leaf(
        &self,
        payload: &model::NativeStage,
        input: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        bindings: &Bindings<QueryOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::Truth, zirql::EvaluationError<EvaluationError>> {
        let model::Stage::Guard {
            predicate: guard, ..
        } = payload.as_ref()
        else {
            return Err(zirql::QueryError::query(range.clone(), "expected predicate guard").into());
        };
        if let model::Predicate::Bool { value, .. } = guard {
            return Ok(if *value {
                zirql::Truth::True
            } else {
                zirql::Truth::False
            });
        }
        let input =
            program::copy_value(input, budget, range).map_err(zirql::EvaluationError::Host)?;
        evaluate_predicate_leaf(guard, input, context, bindings, budget)
            .map_err(zirql::EvaluationError::Host)
    }
    fn evaluate_guard_leaf_with_arguments(
        &self,
        payload: &model::NativeStage,
        input: &QueryOutput,
        bindings: &Bindings<QueryOutput>,
        arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_>, Self>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::Truth, zirql::EvaluationError<EvaluationError>> {
        let model::Stage::Guard {
            predicate: guard, ..
        } = payload.as_ref()
        else {
            return Err(zirql::QueryError::query(range.clone(), "expected predicate guard").into());
        };
        if let model::Predicate::Bool { value, .. } = guard {
            return Ok(if *value {
                zirql::Truth::True
            } else {
                zirql::Truth::False
            });
        }
        if let model::Predicate::Expression { .. } = guard {
            let input =
                program::copy_value(input, budget, range).map_err(zirql::EvaluationError::Host)?;
            return report::evaluate_prepared_leaf(guard, input, Some(bindings), arguments, budget);
        }
        if let model::Predicate::SameValue { left, right, .. } = guard {
            let mut resolve = |operand: &capture::ParsedOperand| -> Result<ValueId, zirql::EvaluationError<EvaluationError>> {
                let input = program::copy_value(input, budget, range).map_err(zirql::EvaluationError::Host)?;
                let output = arguments.evaluate_match_value(&operand.value.expression, input, bindings, budget)?;
                let QueryOutput::Native(NativeValue::Ssa(value)) = functions::unretained(output.value()) else {
                    return Err(zirql::QueryError::query(range.clone(), "same_value requires native SSA values").into());
                };
                capture::check_value(arguments.parts().0.document, *value, range).map_err(zirql::EvaluationError::Host)?;
                Ok(*value)
            };
            return Ok(if resolve(left)? == resolve(right)? {
                zirql::Truth::True
            } else {
                zirql::Truth::False
            });
        }
        let input = program::copy_value(input, budget, range)
            .and_then(|input| functions::owned(input, budget, range))
            .map_err(zirql::EvaluationError::Host)?;
        let input = match input {
            QueryOutput::Native(NativeValue::OperationSequence(ids)) => {
                QueryOutput::Operations(ids)
            }
            input => input,
        };
        evaluate_prepared_leaf(guard, input, Some(bindings), arguments, budget)
    }
    fn validate_existence(
        &self,
        input: &program::Kind,
        result: &program::Kind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        if *input == program::Kind::Operations && *result == program::Kind::Operations {
            Ok(())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "where_exists requires native operations",
            ))
        }
    }
    fn allows_fixed_point_emission(&self) -> bool {
        true
    }
    fn fixed_point_kind(
        &self,
        input: &program::Kind,
        result: &program::Kind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        if matches!(input, program::Kind::Operations | program::Kind::Strings)
            && matches!(result, program::Kind::Operations | program::Kind::Strings)
        {
            Ok(result.clone())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "fixpoint requires an operation or value stream",
            ))
        }
    }
    fn set_kind(
        &self,
        left: &program::Kind,
        right: &program::Kind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        if left == right && matches!(left, program::Kind::Operations | program::Kind::Strings) {
            Ok(left.clone())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "set operands require operation or value streams of the same kind",
            ))
        }
    }
    fn selection_kind(
        &self,
        input: &program::Kind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        if matches!(input, program::Kind::Operations | program::Kind::Strings) {
            Ok(input.clone())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "relation requires an operation or value stream",
            ))
        }
    }
    fn selection_items(
        &self,
        input: &QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
        if let QueryOutput::Values(values) = functions::unretained(input) {
            capture::slots::<QueryOutput>(values.len(), budget, range)
                .map_err(zirql::EvaluationError::Host)?;
            capture::slots::<String>(values.len(), budget, range)
                .map_err(zirql::EvaluationError::Host)?;
            budget.charge_results(values.len() * 2, Location::Query(range.clone()))?;
            return values
                .iter()
                .map(|value| {
                    program::charge_bytes(budget, value.len(), range)
                        .map_err(zirql::EvaluationError::Host)?;
                    Ok(QueryOutput::Values(vec![value.clone()]))
                })
                .collect();
        }
        let input = functions::unretained(input);
        let count = match input {
            QueryOutput::Native(NativeValue::Types(values)) => values.len(),
            QueryOutput::Native(NativeValue::Attributes(values)) => values.len(),
            QueryOutput::Array(values) => values.len(),
            QueryOutput::Map(values) => values.len(),
            QueryOutput::Native(NativeValue::Map(values)) => values.len(),
            QueryOutput::Native(NativeValue::Sequence(value)) => value.values.len(),
            QueryOutput::Native(NativeValue::Constructed(value)) => {
                let V::Array(values) = value.value().data() else {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "stage requires a positional collection",
                    )
                    .into());
                };
                values.len()
            }
            _ => {
                return self.subjects(
                    self.copy_value(input, context, budget, range)?,
                    context,
                    budget,
                    range,
                );
            }
        };
        capture::slots::<QueryOutput>(count, budget, range)
            .map_err(zirql::EvaluationError::Host)?;
        budget.charge_results(count + 1, Location::Query(range.clone()))?;
        let values = match input {
            QueryOutput::Native(NativeValue::Types(values)) => {
                capture::slots::<TypeId>(count, budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                budget.charge_results(count, Location::Query(range.clone()))?;
                values
                    .iter()
                    .map(|id| QueryOutput::Native(NativeValue::Types(vec![*id])))
                    .collect()
            }
            QueryOutput::Native(NativeValue::Attributes(values)) => {
                capture::slots::<(String, AttributeId)>(count, budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                budget.charge_results(count, Location::Query(range.clone()))?;
                values
                    .iter()
                    .map(|(name, id)| {
                        program::charge_bytes(budget, name.len(), range)
                            .map_err(zirql::EvaluationError::Host)?;
                        Ok(QueryOutput::Native(NativeValue::Attributes(vec![(
                            name.clone(),
                            *id,
                        )])))
                    })
                    .collect::<Result<_, zirql::EvaluationError<EvaluationError>>>()?
            }
            QueryOutput::Array(_) => {
                let QueryOutput::Array(values) = program::copy_value(input, budget, range)
                    .map_err(zirql::EvaluationError::Host)?
                else {
                    unreachable!("array subjects")
                };
                values
                    .into_iter()
                    .map(|value| QueryOutput::Native(NativeValue::JsonValue(value)))
                    .collect()
            }
            QueryOutput::Map(_) => {
                let QueryOutput::Map(values) = program::copy_value(input, budget, range)
                    .map_err(zirql::EvaluationError::Host)?
                else {
                    unreachable!("map subjects")
                };
                values
                    .into_iter()
                    .map(|(_, value)| QueryOutput::Native(NativeValue::JsonValue(value)))
                    .collect()
            }
            QueryOutput::Native(NativeValue::Map(values)) => values
                .iter()
                .map(|(_, value)| self.copy_value(value, context, budget, range))
                .collect::<Result<_, _>>()?,
            QueryOutput::Native(NativeValue::Sequence(value)) => value
                .values
                .iter()
                .map(|value| self.copy_value(value, context, budget, range))
                .collect::<Result<_, _>>()?,
            QueryOutput::Native(NativeValue::Constructed(value)) => {
                let V::Array(values) = value.value().data() else {
                    unreachable!("positional collection admitted above")
                };
                values
                    .iter()
                    .map(|value| {
                        let value = zirql::copy_constructed_value(
                            value,
                            budget,
                            Location::Query(range.clone()),
                        )?;
                        let ty = zirql::constructed_value_type(
                            &value,
                            budget,
                            Location::Query(range.clone()),
                        )?;
                        self.lift_constructed(value, vec![ty], context, budget, range)
                    })
                    .collect::<Result<_, _>>()?
            }
            _ => unreachable!("selection shape admitted above"),
        };
        Ok(values)
    }
    fn restore_selection(
        &self,
        input: QueryOutput,
        items: Vec<QueryOutput>,
        shape: zirql::SelectionShape,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        let input = functions::owned(input, budget, range).map_err(zirql::EvaluationError::Host)?;
        Ok(match input {
            QueryOutput::Operations(_) => QueryOutput::Operations(
                items
                    .into_iter()
                    .flat_map(|item| match item {
                        QueryOutput::Operations(ids) => ids,
                        _ => unreachable!("operation subjects"),
                    })
                    .collect(),
            ),
            QueryOutput::Native(NativeValue::OperationSequence(_)) => {
                QueryOutput::Native(NativeValue::OperationSequence(
                    items
                        .into_iter()
                        .flat_map(|item| match item {
                            QueryOutput::Operations(ids) => ids,
                            _ => unreachable!("operation subjects"),
                        })
                        .collect(),
                ))
            }
            QueryOutput::Array(_) => QueryOutput::Array(
                items
                    .iter()
                    .map(|value| program::to_json(context.document, value, budget, range))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(zirql::EvaluationError::Host)?,
            ),
            QueryOutput::Native(NativeValue::Types(_)) => QueryOutput::Native(NativeValue::Types(
                items
                    .into_iter()
                    .flat_map(|item| match item {
                        QueryOutput::Native(NativeValue::Types(ids)) => ids,
                        _ => unreachable!("type subjects"),
                    })
                    .collect(),
            )),
            QueryOutput::Native(NativeValue::Attributes(_)) => {
                QueryOutput::Native(NativeValue::Attributes(
                    items
                        .into_iter()
                        .flat_map(|item| match item {
                            QueryOutput::Native(NativeValue::Attributes(ids)) => ids,
                            _ => unreachable!("attribute subjects"),
                        })
                        .collect(),
                ))
            }
            QueryOutput::Native(NativeValue::Constructed(value)) => {
                let values = items
                    .iter()
                    .map(|item| comprehension::project(item, budget, range))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(zirql::EvaluationError::Host)?;
                let types =
                    zirql::copy_value_types(&value.types, budget, Location::Query(range.clone()))?;
                return self.lift_constructed(V::Array(values), types, context, budget, range);
            }
            QueryOutput::Values(_) => QueryOutput::Values(
                items
                    .into_iter()
                    .flat_map(|item| match item {
                        QueryOutput::Values(values) => values,
                        _ => unreachable!("string subjects"),
                    })
                    .collect(),
            ),
            QueryOutput::Native(NativeValue::Captures(value)) => {
                QueryOutput::Native(NativeValue::Captures(CaptureMatches {
                    rows: items
                        .into_iter()
                        .map(|item| match item {
                            QueryOutput::Native(NativeValue::Capture(record)) => record,
                            _ => unreachable!("capture subjects"),
                        })
                        .collect(),
                    range: range.clone(),
                    schema: value.schema,
                }))
            }
            QueryOutput::Native(NativeValue::Sequence(value)) => {
                QueryOutput::Native(NativeValue::Sequence(NativeSequence {
                    values: items.into(),
                    element: value.element,
                    range: range.clone(),
                }))
            }
            value if shape == zirql::SelectionShape::Preserve && items.len() == 1 => {
                items.into_iter().next().unwrap_or(value)
            }
            value => {
                let kind =
                    output_kind(&value, budget, range).map_err(zirql::EvaluationError::Host)?;
                QueryOutput::Native(NativeValue::Sequence(NativeSequence {
                    values: items.into(),
                    element: Arc::new(zirql::DeclaredType::Native(kind)),
                    range: range.clone(),
                }))
            }
        })
    }
    fn admit_selection(
        &self,
        value: QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        let value = functions::owned(value, budget, range).map_err(zirql::EvaluationError::Host)?;
        match value {
            QueryOutput::Operations(_)
            | QueryOutput::Values(_)
            | QueryOutput::Native(NativeValue::OperationSequence(_)) => Ok(value),
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "fixpoint requires an operation or value stream",
            )
            .into()),
        }
    }
    fn same_selection(
        &self,
        left: &QueryOutput,
        right: &QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<bool, zirql::EvaluationError<EvaluationError>> {
        fixed_point_equal(left, right, budget, range).map_err(zirql::EvaluationError::Host)
    }
    fn ordering_key_kind(
        &self,
        kind: &program::Kind,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        if let program::Kind::Constructed(types) = kind {
            for ty in types.iter() {
                if !matches!(
                    ty,
                    ValueType::String
                        | ValueType::Integer
                        | ValueType::Float
                        | ValueType::Unestablished
                ) {
                    return Err(zirql::QueryError::query(
                        range.clone(),
                        "ordering requires numbers or strings",
                    ));
                }
            }
            budget.charge(types.len(), Location::Query(range.clone()))?;
            return Ok(());
        }
        if matches!(
            kind,
            program::Kind::Strings | program::Kind::String | program::Kind::Count
        ) {
            Ok(())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "ordering selector must produce exactly one string or count per operation",
            ))
        }
    }
    fn admit_ordering(
        &self,
        input: &QueryOutput,
        order: zirql::NativeOrder,
        _: &mut EvaluationContext<'_, '_>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        if order != zirql::NativeOrder::Sort && output_size(functions::unretained(input)) == 0 {
            return Err(zirql::QueryError::query(
                range.clone(),
                "extrema require a non-empty stream",
            )
            .into());
        }
        Ok(())
    }
    fn ordering_key(
        &self,
        value: QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeOrderKey, zirql::EvaluationError<EvaluationError>> {
        if let QueryOutput::Native(NativeValue::JsonValue(value)) = functions::unretained(&value) {
            return scalar_map_key(value, budget, range).map(zirql::NativeOrderKey::Scalar);
        }
        if matches!(
            functions::unretained(&value),
            QueryOutput::Native(NativeValue::Constructed(_))
        ) {
            return comprehension::project(&value, budget, range)
                .map(zirql::NativeOrderKey::Value)
                .map_err(zirql::EvaluationError::Host);
        }
        if matches!(
            functions::unretained(&value),
            QueryOutput::Map(_) | QueryOutput::Native(NativeValue::Map(_))
        ) {
            return Err(zirql::QueryError::query(
                range.clone(),
                "sort_by(value) requires scalar map values; project a scalar from nested maps first",
            ).into());
        }
        ordering_key(value, budget, range).map_err(zirql::EvaluationError::Host)
    }
    fn mapping_kind(
        &self,
        key: &program::Kind,
        _: &program::Kind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        if matches!(key, program::Kind::Strings | program::Kind::String) {
            Ok(program::Kind::NativeMap)
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "map_by key must produce exactly one string",
            ))
        }
    }
    fn mapping_keys(
        &self,
        value: QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Option<Vec<String>>, zirql::EvaluationError<EvaluationError>> {
        map_keys(value, budget, range).map_err(zirql::EvaluationError::Host)
    }
    fn is_map_kind(&self, kind: &program::Kind) -> bool {
        matches!(kind, program::Kind::Map | program::Kind::NativeMap)
    }
    fn map_sort_kind(
        &self,
        input: &program::Kind,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        if self.is_map_kind(input) {
            Ok(input.clone())
        } else {
            Err(zirql::QueryError::query(
                range.clone(),
                "sort_by(value) requires a map",
            ))
        }
    }
    fn map_entries(
        &self,
        value: QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<(String, QueryOutput)>, zirql::EvaluationError<EvaluationError>> {
        match functions::owned(value, budget, range).map_err(zirql::EvaluationError::Host)? {
            QueryOutput::Native(NativeValue::Map(entries)) => Ok(entries),
            QueryOutput::Map(entries) => Ok(entries
                .into_iter()
                .map(|(key, value)| (key, QueryOutput::Native(NativeValue::JsonValue(value))))
                .collect()),
            _ => {
                Err(zirql::QueryError::query(range.clone(), "expected a native dictionary").into())
            }
        }
    }
    fn construct_map(
        &self,
        entries: Vec<(String, QueryOutput)>,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        if entries
            .iter()
            .any(|(_, value)| native_map_depth(value) >= parser::DEFAULT_NESTING_LIMIT)
        {
            return Err(
                zirql::QueryError::query(range.clone(), "map nesting limit exceeded").into(),
            );
        }
        let items = entries.iter().fold(0usize, |items, (_, value)| {
            items.saturating_add(output_size(value).saturating_add(1))
        });
        budget.check_items(items, Location::Query(range.clone()))?;
        budget.charge(items, Location::Query(range.clone()))?;
        Ok(QueryOutput::Native(NativeValue::Map(entries)))
    }
    fn restore_map(
        &self,
        input: QueryOutput,
        entries: Vec<(String, QueryOutput)>,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        if matches!(functions::unretained(&input), QueryOutput::Map(_)) {
            let entries = entries
                .into_iter()
                .map(|(key, value)| match value {
                    QueryOutput::Native(NativeValue::JsonValue(value)) => Ok((key, value)),
                    value => program::to_json(context.document, &value, budget, range)
                        .map(|value| (key, value))
                        .map_err(zirql::EvaluationError::Host),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(QueryOutput::Map(entries));
        }
        self.construct_map(entries, context, budget, range)
    }
    fn capture_selection_kind(
        &self,
        schema: zirql::NativeCapture<program::Kind>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        Ok(program::Kind::Captures(Arc::new(schema)))
    }
    fn construct_captures(
        &self,
        rows: Vec<QueryOutput>,
        schema: zirql::NativeCapture<program::Kind>,
        _: &mut EvaluationContext<'_, '_>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        let rows = rows
            .into_iter()
            .map(|row| match row {
                QueryOutput::Native(NativeValue::Capture(record)) => record,
                _ => unreachable!("core capture record"),
            })
            .collect();
        Ok(QueryOutput::Native(NativeValue::Captures(CaptureMatches {
            rows,
            schema: Arc::new(schema),
            range: range.clone(),
        })))
    }

    fn pattern_constructed<'v>(&self, value: &'v QueryOutput) -> Option<&'v V> {
        match functions::unretained(value) {
            QueryOutput::Native(NativeValue::Constructed(value)) => Some(value.value()),
            _ => None,
        }
    }
    fn positional_view_kind(
        &self,
        kind: &program::Kind,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Option<program::Kind>, zirql::QueryError> {
        match kind {
            program::Kind::Sequence(element) => {
                self.declaration_kind(element, budget, range).map(Some)
            }
            _ => Ok(None),
        }
    }
    fn tuple_view_kind<'k>(&self, kind: &'k program::Kind) -> Option<&'k [program::Kind]> {
        match kind {
            program::Kind::Tuple(values) => Some(values),
            _ => None,
        }
    }

    fn value_arguments<'s>(
        &self,
        stage: &'s model::NativeStage,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeValueArguments<'s, EvaluationContext<'_, '_>, Self>, zirql::QueryError>
    {
        let stage = stage.as_ref();
        fn nodes(predicate: &model::Predicate) -> usize {
            match predicate {
                model::Predicate::Not { predicate, .. }
                | model::Predicate::Group { predicate, .. } => 1 + nodes(predicate),
                model::Predicate::And { predicates, .. }
                | model::Predicate::Or { predicates, .. } => {
                    1 + predicates.iter().map(nodes).sum::<usize>()
                }
                _ => 1,
            }
        }
        if let model::Stage::Native { value, .. }
        | model::Stage::Operand {
            operation: value, ..
        } = stage
        {
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(vec![(&value.value.expression, program::Kind::Operations)]);
        }
        if let model::Stage::Guard { predicate, .. } | model::Stage::Root { predicate, .. } = stage
            && !has_dynamic_predicate(predicate)
        {
            return Ok(Vec::new());
        }
        let capacity = match stage {
            model::Stage::Guard { predicate, .. } | model::Stage::Root { predicate, .. } => {
                nodes(predicate) * 2
            }
            model::Stage::Computed { .. } => 1,

            _ => return Ok(Vec::new()),
        };
        budget.charge(capacity * 2, Location::Query(range.clone()))?;
        budget.charge_results(capacity * 2, Location::Query(range.clone()))?;
        let mut values = Vec::with_capacity(capacity);
        let mut pending = Vec::with_capacity(capacity);
        match stage {
            model::Stage::Computed { value, .. } => {
                values.push((&*value.operand().expression, program::Kind::Operations))
            }
            model::Stage::Guard { predicate, .. } | model::Stage::Root { predicate, .. } => {
                pending.push(predicate)
            }

            _ => {}
        }
        while let Some(predicate) = pending.pop() {
            budget.charge(1, Location::Query(range.clone()))?;
            match predicate {
                model::Predicate::Expression { value: name, .. }
                | model::Predicate::OpValue { name, .. } => {
                    values.push((&*name.expression, program::Kind::Operations))
                }
                model::Predicate::Computed { value, .. } => values.extend(
                    value
                        .operands()
                        .map(|value| (&*value.expression, program::Kind::Operations)),
                ),
                model::Predicate::SameValue { left, right, .. } => {
                    values.push((&*left.value.expression, program::Kind::Operations));
                    values.push((&*right.value.expression, program::Kind::Operations));
                }
                model::Predicate::Not { predicate, .. }
                | model::Predicate::Group { predicate, .. } => pending.push(predicate),
                model::Predicate::And { predicates, .. }
                | model::Predicate::Or { predicates, .. } => {
                    pending.extend(predicates.iter().rev())
                }
                _ => {}
            }
        }
        Ok(values)
    }
    fn scoped_value_arguments<'s>(
        &self,
        stage: &'s model::NativeStage,
        input: &program::Kind,
        _names: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<
        zirql::ScopedNativeValueArguments<'s, EvaluationContext<'_, '_>, Self>,
        zirql::QueryError,
    > {
        let shared_stage = stage;
        let stage = stage.as_ref();
        if let model::Stage::Native { value, .. }
        | model::Stage::Operand {
            operation: value, ..
        } = stage
        {
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(vec![zirql::ScopedNativeArgument {
                expression: &value.value.expression,
                relative: Some(input.clone()),
                locals: Vec::new(),
                allow_effects: false,
            }]);
        }
        let arguments = self.value_arguments(shared_stage, budget, range)?;
        capture::slots::<zirql::ScopedNativeArgument<'_, Operand, program::Kind>>(
            arguments.len(),
            budget,
            range,
        )
        .map_err(query_error)?;
        if budget.remaining_results().is_none() {
            budget.charge(arguments.len(), Location::Query(range.clone()))?;
        }
        Ok(arguments
            .into_iter()
            .map(|(expression, kind)| zirql::ScopedNativeArgument {
                expression,
                relative: (!matches!(stage, model::Stage::FunctionInput { .. })).then_some(
                    if matches!(stage, model::Stage::Guard { .. }) {
                        input.clone()
                    } else {
                        kind
                    },
                ),
                locals: Vec::new(),
                allow_effects: false,
            })
            .collect())
    }
    fn evaluate_stage_with_arguments(
        &self,
        stage: &model::NativeStage,
        input: QueryOutput,
        arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_>, Self>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        let stage = stage.as_ref();
        let input = functions::owned(input, budget, range).map_err(zirql::EvaluationError::Host)?;
        match stage {
            model::Stage::Native { value, .. } => {
                let value = arguments.evaluate_native(&value.value.expression, input, budget)?;
                return Ok(QueryOutput::Native(NativeValue::Retained(
                    value.into_shared(),
                )));
            }
            model::Stage::Operand {
                operation, index, ..
            } => {
                let value =
                    arguments.evaluate_native(&operation.value.expression, input, budget)?;
                return capture::operand_value(
                    value.value(),
                    *index,
                    arguments.parts().0,
                    budget,
                    range,
                )
                .map_err(zirql::EvaluationError::Host);
            }
            _ => {}
        }
        if pipeline::is_stream_stage(stage) && pipeline::is_stream(&input) {
            let pipeline = pipeline::lower(std::slice::from_ref(stage))
                .map_err(zirql::EvaluationError::Host)?;
            return pipeline::evaluate_with_arguments(&pipeline, arguments, input, budget)
                .map_err(zirql::EvaluationError::Host);
        }
        if let model::Stage::Root { predicate, .. } = stage {
            let selected = take_operations(input, "root").map_err(zirql::EvaluationError::Host)?;
            return pipeline::evaluate_roots_with_arguments(predicate, selected, arguments, budget)
                .map(QueryOutput::Operations)
                .map_err(zirql::EvaluationError::Host);
        }
        let resolved;
        let stage = match stage {
            model::Stage::Computed { value, range } => {
                resolved = resolve_stage_by(
                    value,
                    *range,
                    &mut |value, budget| {
                        arguments
                            .evaluate(&value.expression, QueryOutput::Operations(vec![]), budget)
                            .map_err(program::evaluation_error)?
                            .into_owned(budget)
                            .map_err(core_error)
                    },
                    budget,
                )
                .map_err(zirql::EvaluationError::Host)?;
                &resolved
            }
            _ => stage,
        };
        let (context, bindings) = arguments.parts();
        program::evaluate_native_stage(stage, input, context, bindings, budget, range)
    }
    fn share_constructed(
        &self,
        value: &QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<Option<Arc<V>>, zirql::EvaluationError<EvaluationError>> {
        Ok(match functions::unretained(value) {
            QueryOutput::Native(NativeValue::Constructed(value)) => Some(value.shared()),
            _ => None,
        })
    }
    fn lift_shared_constructed(
        &self,
        value: Arc<V>,
        types: Vec<ValueType>,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        comprehension::ConstructedReport::from_shared(value, types, range.clone(), budget)
            .map(|value| QueryOutput::Native(NativeValue::Constructed(value)))
            .map_err(zirql::EvaluationError::Host)
    }
    fn kind(
        &self,
        value: &QueryOutput,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        output_kind(value, budget, range).map_err(query_error)
    }
    fn scalar_types(
        &self,
        kind: &program::Kind,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Option<Vec<ValueType>>, zirql::QueryError> {
        use program::Kind as K;
        budget.charge(1, Location::Query(range.clone()))?;
        Ok(match kind {
            K::JsonValue
            | K::NativeMap
            | K::Computational(_)
            | K::Tuple(_)
            | K::Sequence(_)
            | K::Operations
            | K::Ssa
            | K::Capture(_)
            | K::Captures(_)
            | K::Types
            | K::Attributes => None,
            K::Strings => Some(vec![ValueType::Array(zirql::ArrayType::Elements(vec![
                ValueType::String,
            ]))]),
            K::Count => Some(vec![ValueType::Integer]),
            K::String => Some(vec![ValueType::String]),
            K::Constructed(types) => Some(zirql::copy_value_types(
                types,
                budget,
                Location::Query(range.clone()),
            )?),
            K::Map => Some(vec![ValueType::RecordKind]),
            K::Array => Some(vec![ValueType::Array(zirql::ArrayType::Elements(
                Vec::new(),
            ))]),
        })
    }
    fn constructed_kind(&self, types: Vec<ValueType>) -> program::Kind {
        program::Kind::Constructed(Arc::new(types))
    }
    fn expression_kind(
        &self,
        expression: &Operand,
        input: &program::Kind,
        scope: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
    ) -> Result<program::Kind, zirql::QueryError> {
        zirql::pattern_function_expression_kind(
            self.functions
                .iter()
                .map(|function| function.declaration.as_ref()),
            self,
            expression,
            scope,
            Some(input.clone()),
            false,
            budget,
        )
    }
    fn admit_emitter(
        &self,
        emitter: &program::Emitter,
        input: &program::Kind,
        scope: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        _range: &Range<usize>,
    ) -> Result<(), zirql::QueryError> {
        if let program::Emitter::Written(stage) = emitter {
            program::Validator {
                host: Self {
                    functions: self.functions,
                },
            }
            .stages(
                std::slice::from_ref(stage.as_ref()),
                input.clone(),
                scope,
                budget,
            )
            .map_err(query_error)?;
        }
        Ok(())
    }
    fn stage_kind(
        &self,
        stage: &model::NativeStage,
        input: &program::Kind,
        scope: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        let stage = stage.as_ref();
        budget.charge(stage_size(stage), Location::Query(range.clone()))?;
        program::Validator {
            host: Self {
                functions: self.functions,
            },
        }
        .stages(std::slice::from_ref(stage), input.clone(), scope, budget)
        .map_err(query_error)
    }
    fn is_explicit_source(&self, stage: &model::NativeStage) -> bool {
        matches!(stage.as_ref(), model::Stage::FunctionInput { .. })
    }
    fn is_pure(&self, stage: &model::NativeStage) -> bool {
        let stage = stage.as_ref();
        !matches!(
            stage,
            model::Stage::SetAttr { .. }
                | model::Stage::RemoveAttr { .. }
                | model::Stage::Emit { .. }
                | model::Stage::Format { .. }
                | model::Stage::Json { .. }
                | model::Stage::Markdown { .. }
                | model::Stage::Print { .. }
        )
    }
    fn nested_queries(
        &self,
        stage: &model::NativeStage,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeOperands<EvaluationContext<'_, '_>, Self>, zirql::QueryError> {
        let stage = stage.as_ref();
        let mut queries = vec![];
        for expression in nested(stage) {
            budget.charge(expression_size(expression), Location::Query(range.clone()))?;
            queries.push((operand(expression), program::Kind::Operations));
        }
        Ok(queries)
    }
    fn subjects(
        &self,
        input: QueryOutput,
        context: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
        let input = functions::owned(input, budget, range).map_err(zirql::EvaluationError::Host)?;
        let values = match input {
            QueryOutput::Native(NativeValue::Sequence(sequence)) => {
                budget.check_items(sequence.values.len(), Location::Query(range.clone()))?;
                capture::slots::<QueryOutput>(sequence.values.len(), budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                sequence
                    .values
                    .iter()
                    .map(|value| self.copy_value(value, context, budget, range))
                    .collect::<Result<_, _>>()?
            }
            QueryOutput::Operations(operations)
            | QueryOutput::Native(NativeValue::OperationSequence(operations)) => {
                capture::slots::<QueryOutput>(operations.len(), budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                budget.check_items(operations.len(), Location::Query(range.clone()))?;
                budget.charge_results(operations.len() * 2 + 1, Location::Query(range.clone()))?;
                operations
                    .into_iter()
                    .map(|operation| QueryOutput::Operations(vec![operation]))
                    .collect()
            }
            QueryOutput::Native(NativeValue::Captures(value)) => {
                capture::slots::<QueryOutput>(value.len(), budget, range)
                    .map_err(zirql::EvaluationError::Host)?;
                budget.charge_results(value.len() + 1, Location::Query(range.clone()))?;
                value
                    .into_rows()
                    .into_iter()
                    .map(|record| QueryOutput::Native(NativeValue::Capture(record)))
                    .collect()
            }
            value @ QueryOutput::Native(NativeValue::Ssa(_) | NativeValue::Capture(_)) => {
                budget.charge_results(2, Location::Query(range.clone()))?;
                vec![value]
            }
            _ => {
                return Err(zirql::QueryError::query(
                    range.clone(),
                    "native enumeration requires operations, SSA or captures",
                )
                .into());
            }
        };
        Ok(values)
    }
    fn subject_kind(
        &self,
        input: &program::Kind,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        use program::Kind as K;
        match input {
            K::Constructed(types) => {
                let mut elements = Vec::new();
                for ty in types.iter() {
                    match ty {
                        ValueType::Array(
                            zirql::ArrayType::Positions(values)
                            | zirql::ArrayType::Elements(values),
                        )
                        | ValueType::ArrayKind(values) => elements.extend(zirql::copy_value_types(
                            values,
                            budget,
                            Location::Query(range.clone()),
                        )?),
                        _ => {
                            return Err(zirql::QueryError::query(
                                range.clone(),
                                "filter requires a positional collection",
                            ));
                        }
                    }
                }
                Ok(self.constructed_kind(elements))
            }
            K::Sequence(element) => self.declaration_kind(element, budget, range),
            K::Operations | K::Strings | K::Ssa | K::Capture(_) => Ok(input.clone()),
            K::Captures(schema) => Ok(K::Capture(Arc::clone(schema))),
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "native enumeration requires operations, SSA or captures",
            )),
        }
    }
    fn record_kind(
        &self,
        fields: Vec<(String, program::Kind)>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        Ok(program::Kind::Capture(Arc::new(zirql::NativeCapture {
            fields,
        })))
    }
    fn capture_record_kind<'k>(
        &self,
        kind: &'k program::Kind,
    ) -> Option<&'k zirql::NativeCapture<program::Kind>> {
        match kind {
            program::Kind::Capture(schema) => Some(schema),
            _ => None,
        }
    }
    fn project_value(
        &self,
        value: QueryOutput,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<V, zirql::EvaluationError<EvaluationError>> {
        let value = functions::owned(value, budget, range).map_err(zirql::EvaluationError::Host)?;
        match value {
            QueryOutput::Count(count) => Ok(zirql::count_value(
                count,
                budget,
                Location::Query(range.clone()),
            )?),
            QueryOutput::Values(values) => {
                budget.charge_results(values.len() + 1, Location::Query(range.clone()))?;
                Ok(V::Array(values.into_iter().map(V::String).collect()))
            }
            QueryOutput::Native(NativeValue::String(value)) => {
                budget.charge_results(1, Location::Query(range.clone()))?;
                Ok(V::String(value))
            }
            QueryOutput::Native(NativeValue::Constructed(value)) => value
                .into_value(budget, range)
                .map_err(zirql::EvaluationError::Host),
            value => {
                comprehension::project(&value, budget, range).map_err(zirql::EvaluationError::Host)
            }
        }
    }
    fn lift_constructed(
        &self,
        value: V,
        types: Vec<ValueType>,
        _: &mut EvaluationContext<'_, '_>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        comprehension::ConstructedReport::with_types(value, types, range.clone(), budget)
            .map(|value| QueryOutput::Native(NativeValue::Constructed(value)))
            .map_err(zirql::EvaluationError::Host)
    }
}
pub(super) fn query_error(error: EvaluationError) -> zirql::QueryError {
    zirql::QueryError::query(error.range.unwrap_or(0..0), error.message)
}
pub(super) fn nested(stage: &model::Stage) -> Vec<&model::Expression> {
    match stage {
        model::Stage::Group { expression, .. } => vec![expression],
        _ => vec![],
    }
}
fn expression_size(expression: &model::Expression) -> usize {
    expression
        .first
        .iter()
        .chain(expression.rest.iter().flat_map(|(_, stages)| stages))
        .map(|step| match &step.value {
            Step::Stage(stage) => stage_size(stage),
            _ => 1,
        })
        .fold(0usize, usize::saturating_add)
}
fn stage_size(stage: &model::Stage) -> usize {
    use model::{Stage as S, StructuredStage as T};
    let payload = match stage {
        S::Binding { name, .. }
        | S::Attr { name, .. }
        | S::RemoveAttr { name, .. }
        | S::Structured {
            step: T::StringAttr(name) | T::Attributes(name),
            ..
        } => name.len(),
        S::Guard { predicate, .. } | S::Root { predicate, .. } => predicate_size(predicate),
        S::SetAttr { name, value, .. } => name.len().saturating_add(value.len()),
        _ => 0,
    };
    1usize
        .saturating_add(program::range(stage.range()).len())
        .saturating_add(payload)
        .saturating_add(
            nested(stage)
                .into_iter()
                .map(expression_size)
                .fold(0usize, usize::saturating_add),
        )
}
fn predicate_payload(predicate: &model::Predicate) -> usize {
    use model::Predicate as P;
    match predicate {
        P::Op { name, .. } | P::Dialect { name, .. } | P::HasAttr { name, .. } => name.len(),
        P::ResultType { spelling, .. } => spelling.len(),
        P::Attr { name, value, .. } => name.len().saturating_add(value.len()),
        P::And { predicates, .. } | P::Or { predicates, .. } => predicates.len(),
        _ => 0,
    }
}
fn predicate_size(predicate: &model::Predicate) -> usize {
    use model::Predicate as P;
    let children = match predicate {
        P::Not { predicate, .. } | P::Group { predicate, .. } => predicate_size(predicate),
        P::And { predicates, .. } | P::Or { predicates, .. } => predicates
            .iter()
            .map(predicate_size)
            .fold(0usize, usize::saturating_add),
        _ => 0,
    };
    1usize
        .saturating_add(predicate_payload(predicate))
        .saturating_add(children)
}

type ScalarRead<'a> =
    dyn FnMut(&comprehension::ParsedComprehension, &mut Budget) -> Result<V, EvaluationError> + 'a;
fn string_argument(
    value: &comprehension::ParsedComprehension,
    read: &mut ScalarRead<'_>,
    budget: &mut Budget,
) -> Result<String, EvaluationError> {
    let V::String(text) = read(value, budget)?.into_data() else {
        return Err(EvaluationError::new("primitive argument needs a string")
            .at(value.expression.range.clone()));
    };
    Ok(text)
}
fn attribute_argument(
    value: &comprehension::ParsedComprehension,
    read: &mut ScalarRead<'_>,
    budget: &mut Budget,
) -> Result<String, EvaluationError> {
    let text = string_argument(value, read, budget)?;
    if !parser::valid_attribute_name(&text) {
        return Err(
            EvaluationError::new("attribute name must be a dotted ASCII identifier")
                .at(value.expression.range.clone()),
        );
    }
    Ok(text)
}
fn resolve_stage_by(
    value: &model::ComputedStage,
    range: crate::source::TextRange,
    read: &mut ScalarRead<'_>,
    budget: &mut Budget,
) -> Result<model::Stage, EvaluationError> {
    use model::{ComputedStage as C, Stage as S};
    if let C::Attr(value) = value {
        return Ok(S::Attr {
            name: attribute_argument(value, read, budget)?,
            range,
        });
    }
    let V::Integer(integer) = read(value.operand(), budget)?.into_data() else {
        return Err(
            EvaluationError::new("primitive argument needs an int").at(value
                .operand()
                .expression
                .range
                .clone()),
        );
    };
    let count = zirql::nonnegative_integer_to_usize(
        integer,
        ValueLocation::Query(value.operand().expression.range.clone()),
        "primitive argument needs a non-negative int fitting usize",
    )
    .map_err(core_error)?;
    Ok(match value {
        C::Defs(_) => S::Defs {
            index: Some(count),
            range,
        },
        C::Users(_) => S::Users {
            index: Some(count),
            range,
        },
        C::Attr(_) => unreachable!("attribute operand resolved above"),
    })
}
fn resolve_computed_predicate(
    value: &model::ComputedPredicate,
    range: crate::source::TextRange,
    read: &mut ScalarRead<'_>,
    budget: &mut Budget,
) -> Result<model::Predicate, EvaluationError> {
    use model::{ComputedPredicate as C, Predicate as P};
    let name = |value: &comprehension::ParsedComprehension,
                read: &mut ScalarRead<'_>,
                budget: &mut Budget| {
        let text = string_argument(value, read, budget)?;
        if text.is_empty() {
            return Err(EvaluationError::new("name or type must not be empty")
                .at(value.expression.range.clone()));
        }
        Ok(text)
    };
    Ok(match value {
        C::Dialect(value) => P::Dialect {
            name: name(value, read, budget)?,
            range,
        },
        C::ResultType(value) => P::ResultType {
            spelling: name(value, read, budget)?,
            range,
        },
        C::HasAttr(value) => P::HasAttr {
            name: attribute_argument(value, read, budget)?,
            range,
        },
        C::Attr { name, value } => P::Attr {
            name: attribute_argument(name, read, budget)?,
            value: string_argument(value, read, budget)?,
            range,
        },
    })
}
fn op_name_by(
    value: &comprehension::ParsedComprehension,
    read: &mut ScalarRead<'_>,
    budget: &mut Budget,
) -> Result<String, EvaluationError> {
    let name = string_argument(value, read, budget)?;
    if name.is_empty() {
        return Err(EvaluationError::new("operation name must not be empty")
            .at(value.expression.range.clone()));
    }
    Ok(name)
}
pub(super) fn push_expressions<'a>(
    stage: &'a model::Stage,
    pending: &mut Vec<&'a model::Expression>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    match stage {
        model::Stage::Native { value, .. }
        | model::Stage::Operand {
            operation: value, ..
        } => visit_value(&value.value.expression, pending, budget)?,
        model::Stage::PatternBinding { value, .. } => {
            visit_value(&value.value.expression, pending, budget)?
        }
        model::Stage::Computed { value, .. } => {
            visit_value(&value.operand().expression, pending, budget)?
        }
        model::Stage::Guard { predicate, .. } | model::Stage::Root { predicate, .. } => {
            visit_predicate(predicate, pending, budget)?
        }
        _ => {}
    }
    Ok(())
}
fn visit_predicate<'a>(
    predicate: &'a model::Predicate,
    pending: &mut Vec<&'a model::Expression>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    match predicate {
        model::Predicate::SameValue { left, right, .. } => {
            visit_value(&left.value.expression, pending, budget)?;
            visit_value(&right.value.expression, pending, budget)?;
        }
        model::Predicate::Computed { value, .. } => {
            for operand in value.operands() {
                visit_value(&operand.expression, pending, budget)?;
            }
        }
        model::Predicate::Expression { value: name, .. }
        | model::Predicate::OpValue { name, .. } => visit_value(&name.expression, pending, budget)?,
        model::Predicate::Not { predicate, .. } | model::Predicate::Group { predicate, .. } => {
            visit_predicate(predicate, pending, budget)?
        }
        model::Predicate::And { predicates, .. } | model::Predicate::Or { predicates, .. } => {
            for predicate in predicates {
                visit_predicate(predicate, pending, budget)?
            }
        }
        _ => {}
    }
    Ok(())
}
pub(super) fn validate_projection_subject(
    subject: &Located<String>,
) -> Result<(), zirql::QueryError> {
    if subject.value == "_" || parser::is_reserved(&subject.value) {
        return Err(zirql::QueryError::query(
            subject.range.clone(),
            "projection subject shadows an existing or reserved name",
        ));
    }
    Ok(())
}

fn visit_operand<'a>(
    query: &'a Operand,
    pending: &mut Vec<&'a model::Expression>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    budget.within_evaluation(0, Location::Query(query.range.clone()), |budget| {
        zirql::walk_expression(query, budget, 64, |query, budget| {
            if let X::Sequence { steps, .. } = &query.value {
                for step in steps {
                    visit_payloads(step, pending, budget).map_err(query_error)?;
                }
            }
            Ok(())
        })
        .map_err(core_error)
    })
}
pub(super) fn visit_value<'a>(
    value: &'a zirql::ValueExpression<Operand>,
    pending: &mut Vec<&'a model::Expression>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    budget.within_evaluation(0, Location::Query(value.range.clone()), |budget| {
        zirql::walk_value(value, budget, 64, |value, _, budget| {
            if let zirql::ValueExpressionKind::NativeQuery(query)
            | zirql::ValueExpressionKind::NativeValue(query) = &value.value
            {
                visit_operand(query, pending, budget).map_err(query_error)?;
            }
            Ok(())
        })
        .map_err(core_error)
    })
}

pub(super) fn evaluate_predicate_leaf(
    predicate: &model::Predicate,
    input: QueryOutput,
    context: &mut EvaluationContext<'_, '_>,
    _bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<zirql::Truth, EvaluationError> {
    if matches!(
        predicate,
        model::Predicate::Expression { .. } | model::Predicate::SameValue { .. }
    ) {
        return Err(
            EvaluationError::new("native predicate arguments require prepared execution")
                .at(program::range(predicate.range())),
        );
    }
    let resolved = resolve_predicate_by(
        predicate,
        &mut |value, _| {
            Err(
                EvaluationError::new("native predicate arguments require prepared execution")
                    .at(value.expression.range.clone()),
            )
        },
        budget,
    )?;
    let QueryOutput::Operations(operations) = input else {
        return Err(EvaluationError::new("predicate requires operations"));
    };
    let operation =
        zirql::take_one(operations, program::range(predicate.range())).map_err(core_error)?;
    Ok(pipeline::evaluate_leaf(
        &resolved,
        context.document,
        operation,
    ))
}
pub(super) fn evaluate_prepared_leaf(
    predicate: &model::Predicate,
    input: QueryOutput,
    captures: Option<&Bindings<QueryOutput>>,
    arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_>, ReportHost<'_>>,
    budget: &mut Budget,
) -> Result<zirql::Truth, zirql::EvaluationError<EvaluationError>> {
    if let model::Predicate::Expression { value, .. } = predicate {
        let matched = arguments.evaluate_predicate(&value.expression, input, captures, budget)?;
        return Ok(if matched {
            zirql::Truth::True
        } else {
            zirql::Truth::False
        });
    }
    let resolved = resolve_predicate_by(
        predicate,
        &mut |value, budget| {
            let input = program::copy_value(&input, budget, &value.expression.range)?;
            arguments
                .evaluate(&value.expression, input, budget)
                .map_err(program::evaluation_error)?
                .into_owned(budget)
                .map_err(core_error)
        },
        budget,
    )
    .map_err(zirql::EvaluationError::Host)?;
    let QueryOutput::Operations(operations) = input else {
        return Err(zirql::QueryError::query(
            program::range(predicate.range()),
            "predicate requires operations",
        )
        .into());
    };
    let operation = zirql::take_one(operations, program::range(predicate.range()))?;
    Ok(pipeline::evaluate_leaf(
        &resolved,
        arguments.parts().0.document,
        operation,
    ))
}
fn resolve_predicate_by(
    predicate: &model::Predicate,
    read: &mut ScalarRead<'_>,
    budget: &mut Budget,
) -> Result<model::Predicate, EvaluationError> {
    use model::Predicate as P;
    let range = predicate.range();
    let payload = predicate_payload(predicate);
    program::charge_bytes(
        budget,
        1usize.saturating_add(payload),
        &program::range(range),
    )?;
    Ok(match predicate {
        P::Computed { value, .. } => resolve_computed_predicate(value, range, read, budget)?,
        P::OpValue { name, .. } => P::Op {
            name: op_name_by(name, read, budget)?,
            range,
        },
        P::Not { predicate, .. } => P::Not {
            predicate: Box::new(resolve_predicate_by(predicate, read, budget)?),
            range,
        },
        P::Group { predicate, .. } => P::Group {
            predicate: Box::new(resolve_predicate_by(predicate, read, budget)?),
            range,
        },
        P::And { predicates, .. } => P::And {
            predicates: predicates
                .iter()
                .map(|predicate| resolve_predicate_by(predicate, read, budget))
                .collect::<Result<_, _>>()?,
            range,
        },
        P::Or { predicates, .. } => P::Or {
            predicates: predicates
                .iter()
                .map(|predicate| resolve_predicate_by(predicate, read, budget))
                .collect::<Result<_, _>>()?,
            range,
        },
        predicate => predicate.clone(),
    })
}

pub(super) fn has_dynamic_predicate(predicate: &model::Predicate) -> bool {
    use model::Predicate as P;
    match predicate {
        P::Expression { .. } | P::OpValue { .. } | P::Computed { .. } | P::SameValue { .. } => true,
        P::Not { predicate, .. } | P::Group { predicate, .. } => has_dynamic_predicate(predicate),
        P::And { predicates, .. } | P::Or { predicates, .. } => {
            predicates.iter().any(has_dynamic_predicate)
        }
        _ => false,
    }
}

pub(super) fn needs_preparation(expression: &model::Expression) -> bool {
    expression
        .first
        .iter()
        .chain(expression.rest.iter().flat_map(|(_, stages)| stages))
        .any(|step| {
            let Step::Stage(stage) = &step.value else { return true };
            let stage = stage.as_ref();

            matches!(
                stage,
                model::Stage::PatternBinding { .. } | model::Stage::FunctionInput{..} | model::Stage::Native { .. } | model::Stage::Operand { .. } | model::Stage::Computed {..}
            ) || matches!(stage, model::Stage::Guard { predicate, .. } | model::Stage::Root { predicate, .. } if has_dynamic_predicate(predicate))
            || nested(stage).into_iter().any(needs_preparation)
        })
}

pub(super) fn dispose_value(value: zirql::ValueExpressionKind<Operand>) {
    zirql::dispose_native_value(Located::new(value, 0..0));
}

pub(super) fn types_with_host(
    value: &comprehension::ParsedComprehension,
    input: program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &ReportHost<'_>,
) -> Result<Vec<ValueType>, EvaluationError> {
    let kind =
        functions::value_kind(host, &value.expression, input, names, budget).map_err(core_error)?;
    host.scalar_types(&kind, budget, &value.expression.range)
        .map_err(core_error)?
        .ok_or_else(|| {
            EvaluationError::new("native handles require an explicit scalar projection")
                .at(value.expression.range.clone())
        })
}
pub(super) fn validate_argument_with_host(
    value: &comprehension::ParsedComprehension,
    expected: ValueType,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &ReportHost<'_>,
) -> Result<(), EvaluationError> {
    for ty in types_with_host(value, program::Kind::Operations, names, budget, host)? {
        admit_scalar_type(
            &ty,
            &expected,
            budget,
            &value.expression.range,
            "primitive argument has an incompatible scalar type",
        )
        .map_err(core_error)?;
    }
    Ok(())
}

pub(super) fn validate_predicate_with_host(
    predicate: &model::Predicate,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &ReportHost<'_>,
) -> Result<(), EvaluationError> {
    match predicate {
        model::Predicate::Expression { value, .. } => {
            for ty in types_with_host(value, program::Kind::Operations, names, budget, host)? {
                admit_scalar_type(
                    &ty,
                    &ValueType::Boolean,
                    budget,
                    &value.expression.range,
                    "predicate needs a known Boolean",
                )
                .map_err(core_error)?;
            }
        }
        model::Predicate::SameValue { range, .. } => {
            return Err(EvaluationError::new(
                "same_value is only supported in native match guards",
            )
            .at(program::range(*range)));
        }
        model::Predicate::Computed { value, .. } => {
            for operand in value.operands() {
                validate_argument_with_host(operand, ValueType::String, names, budget, host)?;
            }
        }
        model::Predicate::OpValue { name, .. } => {
            for ty in types_with_host(name, program::Kind::Operations, names, budget, host)? {
                admit_scalar_type(
                    &ty,
                    &ValueType::String,
                    budget,
                    &name.expression.range,
                    "op argument needs a string",
                )
                .map_err(core_error)?;
            }
        }
        model::Predicate::Not { predicate, .. } | model::Predicate::Group { predicate, .. } => {
            validate_predicate_with_host(predicate, names, budget, host)?
        }
        model::Predicate::And { predicates, .. } | model::Predicate::Or { predicates, .. } => {
            for predicate in predicates {
                validate_predicate_with_host(predicate, names, budget, host)?
            }
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn admit_scalar_type(
    ty: &ValueType,
    expected: &ValueType,
    budget: &mut Budget,
    range: &Range<usize>,
    context: &str,
) -> Result<(), zirql::QueryError> {
    let expected = match expected {
        ValueType::Boolean => zirql::ScalarType::Boolean,
        ValueType::Integer => zirql::ScalarType::Integer,
        ValueType::Float => zirql::ScalarType::Float,
        ValueType::String => zirql::ScalarType::String,
        _ => {
            return Err(zirql::QueryError::query(
                range.clone(),
                "expected a scalar contract",
            ));
        }
    };
    // Declared scalar contracts establish Zirium's preparation type even when
    // their actual facts are incomplete. Bare unknown facts remain a refusal.
    let policy = if matches!(ty, ValueType::Declared { .. }) {
        zirql::TypeFactPolicy::Deferred
    } else {
        zirql::TypeFactPolicy::Strict
    };
    zirql::admit_scalar_type(
        ty,
        expected,
        policy,
        budget,
        &ValueLocation::Query(range.clone()),
    )
    .map_err(|mut error| {
        if error.category == zirql::QueryErrorCategory::Other {
            error.message = format!("{context}: {}", error.message);
        }
        error
    })
}

pub(super) fn visit_step<'a>(
    step: &'a model::Step,
    pending: &mut Vec<&'a model::Expression>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    visit_payloads(step, pending, budget)?;
    let mut result = Ok(());
    step.value.for_each_value(|value| {
        if result.is_ok() {
            result = visit_value(value, pending, budget);
        }
    });
    result?;
    Ok(())
}

fn visit_payloads<'a>(
    step: &'a model::Step,
    pending: &mut Vec<&'a model::Expression>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    match &step.value {
        Step::Projection(projection) => {
            validate_projection_subject(&projection.subject).map_err(core_error)?
        }
        Step::Match(matching) => {
            for clause in &matching.clauses {
                if let zirql::MatchClause::From { name, .. } | zirql::MatchClause::Bind { name, .. } =
                    &clause.value
                    && (name.value == "_" || parser::is_reserved(&name.value))
                {
                    return Err(
                        EvaluationError::new("capture name is reserved").at(name.range.clone())
                    );
                }
            }
        }
        _ => {}
    }
    let mut result = Ok(());
    step.value.for_each_payload(|stage| {
        if result.is_ok() {
            pending.extend(nested(stage));
            result = push_expressions(stage, pending, budget);
        }
    });
    result
}

pub(super) fn requires_constructed_budget(expression: &model::Expression) -> bool {
    expression
        .first
        .iter()
        .chain(expression.rest.iter().flat_map(|(_, steps)| steps))
        .any(|step| match &step.value {
            Step::Value(_)
            | Step::Projection(_)
            | Step::Match(_)
            | Step::Paths(_)
            | Step::Mapping(_) => true,
            Step::Stage(stage) => {
                matches!(
                    stage.as_ref(),
                    model::Stage::Computed { .. }
                        | model::Stage::Native { .. }
                        | model::Stage::Operand { .. }
                        | model::Stage::FunctionInput { .. }
                        | model::Stage::PatternBinding { .. }
                ) || nested(stage).into_iter().any(requires_constructed_budget)
            }
            Step::Ordering(ordering) => ordering.selector.is_some(),
            Step::Head(_)
            | Step::Tail(_)
            | Step::Check {
                operand: Some(_), ..
            } => true,
            _ => false,
        })
}

fn native_map_depth(value: &QueryOutput) -> usize {
    match functions::unretained(value) {
        QueryOutput::Native(NativeValue::Map(entries)) => {
            1 + entries
                .iter()
                .map(|(_, value)| native_map_depth(value))
                .max()
                .unwrap_or(0)
        }
        QueryOutput::Map(entries) => {
            1 + entries
                .iter()
                .map(|(_, value)| value_depth(value))
                .max()
                .unwrap_or(0)
        }
        QueryOutput::Array(values) => 1 + values.iter().map(value_depth).max().unwrap_or(0),
        QueryOutput::Native(NativeValue::JsonValue(value)) => value_depth(value),
        _ => 0,
    }
}

fn scalar_map_key(
    value: &serde_json::Value,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<zirql::ScalarMapKey, zirql::EvaluationError<EvaluationError>> {
    use zirql::ScalarMapKey as K;
    Ok(match value {
        serde_json::Value::Null => K::Null,
        serde_json::Value::Bool(value) => K::Boolean(*value),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                K::Integer(value)
            } else if let Some(value) = value.as_u64() {
                K::Unsigned(value)
            } else {
                K::Float(value.as_f64().expect("finite JSON number"))
            }
        }
        serde_json::Value::String(value) => {
            budget.charge(value.len(), Location::Query(range.clone()))?;
            budget.charge_results(value.len(), Location::Query(range.clone()))?;
            K::String(value.clone())
        }
        _ => return Err(zirql::QueryError::query(
            range.clone(),
            "sort_by(value) requires scalar map values; project a scalar from nested maps first",
        )
        .into()),
    })
}
