//! Native domain adaptation for shared correlated projection and report values.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::{
    Bindings, Budget, ConstructedValue as V, ExpressionKind as X, ExpressionStep as Step, Located,
    Location, ProgramHost, ProjectionHost, ValueType,
};

pub(super) type Operand = zirql::Expression<model::Stage, model::SetOperator, ()>;
#[derive(Debug, PartialEq)]
pub struct ParsedProjection {
    pub(super) projection: Arc<zirql::Projection<Operand>>,
}
impl Clone for ParsedProjection {
    fn clone(&self) -> Self {
        Self {
            projection: Arc::clone(&self.projection),
        }
    }
}
impl Drop for ParsedProjection {
    fn drop(&mut self) {
        if let Some(projection) = Arc::get_mut(&mut self.projection) {
            for (_, value) in std::mem::take(&mut projection.fields) {
                dispose_value(value.value);
            }
        }
    }
}
// The frontend admits finite scalar literals before retaining these trees.
impl Eq for ParsedProjection {}
impl ParsedProjection {
    pub(super) fn new(projection: zirql::Projection<Operand>) -> Self {
        Self {
            projection: Arc::new(projection),
        }
    }
}

pub(super) fn operand(expression: &model::Expression) -> Operand {
    fn stages(stages: &[model::Stage], range: Range<usize>) -> Operand {
        let mut input = Located::new(X::Input, range.clone());
        let mut steps = vec![];
        for (index, stage) in stages.iter().enumerate() {
            match stage {
                model::Stage::Binding { name, .. } if index == 0 => {
                    input = Located::new(X::Binding(name.clone()), program::range(stage.range()))
                }
                model::Stage::Group { expression, .. } if index == 0 => {
                    input = Located::new(
                        X::Group(Box::new(operand(expression))),
                        program::range(stage.range()),
                    )
                }
                _ => steps.push(Located::new(
                    Step::Stage(stage.clone()),
                    program::range(stage.range()),
                )),
            }
        }
        Located::new(
            X::Sequence {
                input: Box::new(input),
                steps,
            },
            range,
        )
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

pub(super) fn types(
    value: &comprehension::ParsedComprehension,
    input: program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<Vec<ValueType>, EvaluationError> {
    let names = copy_kinds(names, budget, &value.expression.range)?;
    zirql::native_value_types::<EvaluationContext<'_, '_, '_>, _>(
        &value.expression,
        &ReportHost,
        input,
        names,
        budget,
    )
    .map_err(core_error)
}
fn copy_kinds(
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<BTreeMap<String, program::Kind>, EvaluationError> {
    program::charge_bytes(budget, names.keys().map(|name| name.len() + 1).sum(), range)?;
    Ok(names.clone())
}
pub(super) fn projection_types(
    value: &ParsedProjection,
    input: program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<Vec<ValueType>, EvaluationError> {
    if parser::is_reserved(&value.projection.subject.value) || value.projection.subject.value == "_"
    {
        return Err(
            EvaluationError::new("projection subject name is reserved").at(value
                .projection
                .subject
                .range
                .clone()),
        );
    }
    if input != program::Kind::Operations {
        return Err(
            EvaluationError::new("project requires native operations").at(value
                .projection
                .subject
                .range
                .clone()),
        );
    }
    let names = copy_kinds(names, budget, &value.projection.subject.range)?;
    zirql::projection_types::<EvaluationContext<'_, '_, '_>, _>(
        &value.projection,
        &ReportHost,
        input,
        names,
        budget,
    )
    .map_err(core_error)
}
pub(super) fn evaluate(
    value: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<comprehension::ConstructedReport, EvaluationError> {
    let names = runtime_kinds(bindings, budget, &value.expression.range)?;
    let declared = types(value, program::Kind::Operations, &names, budget)?;
    let result = zirql::evaluate_native_value(
        &value.expression,
        &ReportHost,
        input,
        context,
        bindings,
        budget,
    )
    .map_err(program::evaluation_error)?;
    comprehension::ConstructedReport::with_types(
        result,
        declared,
        value.expression.range.clone(),
        budget,
    )
}
pub(super) fn project(
    value: &ParsedProjection,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<comprehension::ConstructedReport, EvaluationError> {
    let names = runtime_kinds(bindings, budget, &value.projection.subject.range)?;
    let declared = projection_types(value, program::Kind::Operations, &names, budget)?;
    let result = zirql::evaluate_projection(
        &value.projection,
        &ReportHost,
        input,
        context,
        bindings,
        budget,
    )
    .map_err(program::evaluation_error)?;
    comprehension::ConstructedReport::with_types(result, declared, range.clone(), budget)
}
fn runtime_kinds(
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<BTreeMap<String, program::Kind>, EvaluationError> {
    let mut names = BTreeMap::new();
    for (name, value) in bindings.iter() {
        program::charge_bytes(budget, name.len() + 1, range)?;
        names.insert(name.to_owned(), output_kind(value));
    }
    Ok(names)
}
fn output_kind(value: &QueryOutput) -> program::Kind {
    use program::Kind as K;
    match value {
        QueryOutput::Operations(_) => K::Operations,
        QueryOutput::Values(_) => K::Strings,
        QueryOutput::Count(_) => K::Count,
        QueryOutput::Map(_)
        | QueryOutput::RankedMap(_)
        | QueryOutput::Native(NativeValue::Map(_)) => K::Map,
        QueryOutput::Array(_) => K::Array,
        QueryOutput::Native(NativeValue::String(_)) => K::String,
        QueryOutput::Native(NativeValue::Types(_)) => K::Types,
        QueryOutput::Native(NativeValue::Attributes(_)) => K::Attributes,
        QueryOutput::Native(NativeValue::Constructed(value)) => {
            K::Constructed(Arc::clone(&value.types))
        }
        _ => K::Operations,
    }
}

pub(super) struct ReportHost;
impl<'q> ProgramHost<EvaluationContext<'_, '_, 'q>> for ReportHost {
    type Value = QueryOutput;
    type Stage = model::Stage;
    type SetOperator = model::SetOperator;
    type Emitter = ();
    type Error = EvaluationError;
    fn copy_value(
        &self,
        value: &QueryOutput,
        _: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        program::copy_value(value, budget, range).map_err(zirql::EvaluationError::Host)
    }
    fn evaluate_stage(
        &self,
        stage: &model::Stage,
        input: QueryOutput,
        context: &mut EvaluationContext<'_, '_, 'q>,
        bindings: &Bindings<QueryOutput>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        let value = if pipeline::is_stream_stage(stage) && pipeline::is_stream(&input) {
            let pipeline = pipeline::lower(std::slice::from_ref(stage))
                .map_err(zirql::EvaluationError::Host)?;
            pipeline::evaluate(&pipeline, context, input, bindings, budget)
        } else {
            charge(budget, output_size(&input).max(1))
                .and_then(|()| evaluate_host_stage(stage, context, input, bindings, budget))
        };
        value.map_err(|error| zirql::EvaluationError::Host(error.at(range.clone())))
    }
    fn combine(
        &self,
        operator: &model::SetOperator,
        left: QueryOutput,
        right: QueryOutput,
        context: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        program::NativeHost.combine(operator, left, right, context, budget, range)
    }
    fn construct_record(
        &self,
        fields: Vec<(String, QueryOutput)>,
        context: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        program::NativeHost.construct_record(fields, context, budget, range)
    }
    fn emit(
        &self,
        _: &(),
        _: &QueryOutput,
        _: &mut EvaluationContext<'_, '_, 'q>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        Err(zirql::QueryError::query(range.clone(), "constructed expressions cannot emit").into())
    }
}
impl<'q> ProjectionHost<EvaluationContext<'_, '_, 'q>> for ReportHost {
    type Kind = program::Kind;
    fn kind(
        &self,
        value: &QueryOutput,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        Ok(output_kind(value))
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
            K::Operations | K::Types | K::Attributes => None,
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
            K::Map | K::Array => Some(vec![ValueType::Unestablished]),
        })
    }
    fn constructed_kind(&self, types: Vec<ValueType>) -> program::Kind {
        program::Kind::Constructed(Arc::new(types))
    }
    fn stage_kind(
        &self,
        stage: &model::Stage,
        input: &program::Kind,
        scope: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        _: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        program::validate_stages(std::slice::from_ref(stage), input.clone(), scope, budget)
            .map_err(query_error)
    }
    fn is_pure(&self, stage: &model::Stage) -> bool {
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
        stage: &model::Stage,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<zirql::NativeOperands<EvaluationContext<'_, '_, 'q>, Self>, zirql::QueryError> {
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
        _: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Vec<QueryOutput>, zirql::EvaluationError<EvaluationError>> {
        let QueryOutput::Operations(operations) = input else {
            return Err(zirql::QueryError::query(
                range.clone(),
                "project requires native operations",
            )
            .into());
        };
        budget.check_items(operations.len(), Location::Query(range.clone()))?;
        budget.charge(operations.len(), Location::Query(range.clone()))?;
        budget.charge_results(operations.len(), Location::Query(range.clone()))?;
        Ok(operations
            .into_iter()
            .map(|operation| QueryOutput::Operations(vec![operation]))
            .collect())
    }
    fn project_value(
        &self,
        value: QueryOutput,
        _: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<V, zirql::EvaluationError<EvaluationError>> {
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
        _: &mut EvaluationContext<'_, '_, 'q>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        comprehension::ConstructedReport::with_types(value, types, range.clone(), budget)
            .map(|value| QueryOutput::Native(NativeValue::Constructed(value)))
            .map_err(zirql::EvaluationError::Host)
    }
}
fn query_error(error: EvaluationError) -> zirql::QueryError {
    zirql::QueryError::query(error.range.unwrap_or(0..0), error.message)
}
fn nested(stage: &model::Stage) -> Vec<&model::Expression> {
    use model::{Stage as S, StructuredStage as T};
    match stage {
        S::Group { expression, .. } | S::Fixpoint { expression, .. } => vec![expression],
        S::SortBy { selector, .. }
        | S::MinBy { selector, .. }
        | S::MinAllBy { selector, .. }
        | S::MaxBy { selector, .. }
        | S::MaxAllBy { selector, .. } => vec![selector],
        S::MapBy { key, value, .. }
        | S::Structured {
            step: T::MapBy { key, value },
            ..
        } => vec![key, value],
        S::Structured {
            step: T::WhereExists(expression),
            ..
        } => vec![expression],
        _ => vec![],
    }
}
fn expression_size(expression: &model::Expression) -> usize {
    expression
        .first
        .iter()
        .chain(expression.rest.iter().flat_map(|(_, stages)| stages))
        .map(|stage| {
            use model::{Stage as S, StructuredStage as T};
            let payload = match stage {
                S::Binding { name, .. }
                | S::Attr { name, .. }
                | S::RemoveAttr { name, .. }
                | S::Structured {
                    step: T::StringAttr(name) | T::Attributes(name),
                    ..
                } => name.len(),
                S::Filter { predicate, .. } | S::Root { predicate, .. } => {
                    predicate_size(predicate)
                }
                S::SetAttr { name, value, .. } => name.len().saturating_add(value.len()),
                S::Check { message, .. } => message.as_ref().map_or(0, String::len),
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
        })
        .fold(0usize, usize::saturating_add)
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

pub(super) fn validate_argument(
    value: &comprehension::ParsedComprehension,
    expected: ValueType,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    if types(value, program::Kind::Operations, names, budget)?
        .iter()
        .any(|ty| *ty != expected)
    {
        return Err(
            EvaluationError::new("primitive argument has an incompatible scalar type")
                .at(value.expression.range.clone()),
        );
    }
    Ok(())
}
fn scalar_argument(
    value: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<V, EvaluationError> {
    evaluate(
        value,
        context,
        QueryOutput::Operations(vec![]),
        bindings,
        budget,
    )?
    .into_value(budget, &value.expression.range)
}
fn string_argument(
    value: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<String, EvaluationError> {
    let V::String(text) = scalar_argument(value, context, bindings, budget)? else {
        return Err(EvaluationError::new("primitive argument needs a string")
            .at(value.expression.range.clone()));
    };
    Ok(text)
}
fn attribute_argument(
    value: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<String, EvaluationError> {
    let text = string_argument(value, context, bindings, budget)?;
    if !parser::valid_attribute_name(&text) {
        return Err(
            EvaluationError::new("attribute name must be a dotted ASCII identifier")
                .at(value.expression.range.clone()),
        );
    }
    Ok(text)
}
pub(super) fn resolve_stage(
    value: &model::ComputedStage,
    range: crate::source::TextRange,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<model::Stage, EvaluationError> {
    use model::{ComputedStage as C, Stage as S};
    if let C::Attr(value) = value {
        return Ok(S::Attr {
            name: attribute_argument(value, context, bindings, budget)?,
            range,
        });
    }
    let V::Integer(integer) = scalar_argument(value.operand(), context, bindings, budget)? else {
        return Err(
            EvaluationError::new("primitive argument needs an int").at(value
                .operand()
                .expression
                .range
                .clone()),
        );
    };
    let count =
        usize::try_from(integer).map_err(|_| {
            EvaluationError::new("primitive argument needs a non-negative int fitting usize")
                .at(value.operand().expression.range.clone())
        })?;
    Ok(match value {
        C::Head(_) => S::Head { count, range },
        C::Tail(_) => S::Tail { count, range },
        C::Defs(_) => S::Defs {
            index: Some(count),
            range,
        },
        C::Users(_) => S::Users {
            index: Some(count),
            range,
        },
        C::Check(_) => S::Check {
            expected: Some(count),
            message: None,
            range,
        },
        C::Attr(_) => unreachable!("attribute operand resolved above"),
    })
}
fn resolve_computed_predicate(
    value: &model::ComputedPredicate,
    range: crate::source::TextRange,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<model::Predicate, EvaluationError> {
    use model::{ComputedPredicate as C, Predicate as P};
    let name = |value: &comprehension::ParsedComprehension,
                context: &mut EvaluationContext<'_, '_, '_>,
                budget: &mut Budget| {
        let text = string_argument(value, context, bindings, budget)?;
        if text.is_empty() {
            return Err(EvaluationError::new("name or type must not be empty")
                .at(value.expression.range.clone()));
        }
        Ok(text)
    };
    Ok(match value {
        C::Dialect(value) => P::Dialect {
            name: name(value, context, budget)?,
            range,
        },
        C::ResultType(value) => P::ResultType {
            spelling: name(value, context, budget)?,
            range,
        },
        C::HasAttr(value) => P::HasAttr {
            name: attribute_argument(value, context, bindings, budget)?,
            range,
        },
        C::Attr { name, value } => P::Attr {
            name: attribute_argument(name, context, bindings, budget)?,
            value: string_argument(value, context, bindings, budget)?,
            range,
        },
    })
}
pub(super) fn validate_predicate(
    predicate: &model::Predicate,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    match predicate {
        model::Predicate::Computed { value, .. } => {
            for operand in value.operands() {
                validate_argument(operand, ValueType::String, names, budget)?;
            }
        }
        model::Predicate::OpValue { name, .. } => {
            if types(name, program::Kind::Operations, names, budget)?
                .iter()
                .any(|ty| *ty != ValueType::String)
            {
                return Err(EvaluationError::new("op argument needs a string")
                    .at(name.expression.range.clone()));
            }
        }
        model::Predicate::Not { predicate, .. } | model::Predicate::Group { predicate, .. } => {
            validate_predicate(predicate, names, budget)?
        }
        model::Predicate::And { predicates, .. } | model::Predicate::Or { predicates, .. } => {
            for predicate in predicates {
                validate_predicate(predicate, names, budget)?
            }
        }
        _ => {}
    }
    Ok(())
}
pub(super) fn op_name(
    value: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<String, EvaluationError> {
    let result = evaluate(
        value,
        context,
        QueryOutput::Operations(vec![]),
        bindings,
        budget,
    )?;
    let V::String(name) = result.into_value(budget, &value.expression.range)? else {
        return Err(
            EvaluationError::new("op argument needs a string").at(value.expression.range.clone())
        );
    };
    if name.is_empty() {
        return Err(EvaluationError::new("operation name must not be empty")
            .at(value.expression.range.clone()));
    }
    Ok(name)
}

pub(super) fn push_expressions<'a>(
    stage: &'a model::Stage,
    pending: &mut Vec<&'a model::Expression>,
) {
    match stage {
        model::Stage::Computed { value, .. } => visit_value(&value.operand().expression, pending),
        model::Stage::Comprehension { value, .. } => visit_value(&value.expression, pending),
        model::Stage::Projection { value, .. } => {
            for (_, expression) in &value.projection.fields {
                visit_value(expression, pending)
            }
        }
        model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } => {
            visit_predicate(predicate, pending)
        }
        _ => {}
    }
}
fn visit_predicate<'a>(predicate: &'a model::Predicate, pending: &mut Vec<&'a model::Expression>) {
    match predicate {
        model::Predicate::Computed { value, .. } => {
            for operand in value.operands() {
                visit_value(&operand.expression, pending);
            }
        }
        model::Predicate::OpValue { name, .. } => visit_value(&name.expression, pending),
        model::Predicate::Not { predicate, .. } | model::Predicate::Group { predicate, .. } => {
            visit_predicate(predicate, pending)
        }
        model::Predicate::And { predicates, .. } | model::Predicate::Or { predicates, .. } => {
            for predicate in predicates {
                visit_predicate(predicate, pending)
            }
        }
        _ => {}
    }
}
fn visit_operand<'a>(query: &'a Operand, pending: &mut Vec<&'a model::Expression>) {
    let mut queries = vec![query];
    while let Some(query) = queries.pop() {
        match &query.value {
            X::Input | X::Binding(_) => {}
            X::Group(query) => queries.push(query),
            X::Sequence { input, steps } => {
                queries.push(input);
                for step in steps {
                    if let Step::Stage(stage) = &step.value {
                        pending.extend(nested(stage));
                        push_expressions(stage, pending);
                    }
                }
            }
            X::Set { left, right, .. } => {
                queries.push(right);
                queries.push(left);
            }
            X::Record(fields) => {
                for (_, query) in fields.iter().rev() {
                    queries.push(query)
                }
            }
        }
    }
}
fn visit_value<'a>(
    value: &'a zirql::ValueExpression<Operand>,
    pending: &mut Vec<&'a model::Expression>,
) {
    use zirql::{ValueExpressionKind as K, ValueGenerator as G};
    let clauses = |clauses: &'a [zirql::Located<G<Operand>>],
                   pending: &mut Vec<&'a model::Expression>| {
        for clause in clauses {
            match &clause.value {
                G::In { source, filter, .. }
                | G::Bind {
                    value: source,
                    filter,
                    ..
                } => {
                    visit_value(source, pending);
                    if let Some(filter) = filter {
                        visit_value(filter, pending);
                    }
                }
            }
        }
    };
    match &value.value {
        K::Literal(_) | K::Name(_) => {}
        K::NativeQuery(query) => visit_operand(query, pending),
        K::One(value)
        | K::IndexSet(value)
        | K::Unary { operand: value, .. }
        | K::Field { value, .. }
        | K::TupleAccess { value, .. }
        | K::Reduce { array: value, .. } => visit_value(value, pending),
        K::Array(values) | K::Tuple(values) | K::Set(values) => {
            for value in values {
                visit_value(value, pending)
            }
        }
        K::Record(fields) => {
            for (_, value) in fields {
                visit_value(value, pending)
            }
        }
        K::KeyedArray { entries, .. } => {
            for (key, value) in entries {
                visit_value(key, pending);
                visit_value(value, pending)
            }
        }
        K::KeyedComprehension {
            clauses: generators,
            key,
            head,
            empty_axis_sources,
        } => {
            clauses(generators, pending);
            visit_value(key, pending);
            visit_value(head, pending);
            if let Some(sources) = empty_axis_sources {
                for source in sources {
                    visit_value(source, pending)
                }
            }
        }
        K::Binary { left, right, .. }
        | K::Range {
            lower: left,
            upper: right,
        }
        | K::ArrayAccess {
            array: left,
            index: right,
        }
        | K::Require {
            condition: left,
            value: right,
            ..
        } => {
            visit_value(left, pending);
            visit_value(right, pending)
        }
        K::Conditional {
            branches,
            otherwise,
        } => {
            for (condition, value) in branches {
                visit_value(condition, pending);
                visit_value(value, pending)
            }
            visit_value(otherwise, pending)
        }
        K::Comprehension {
            clauses: generators,
            head,
            ..
        } => {
            clauses(generators, pending);
            visit_value(head, pending)
        }
    }
}
pub(super) fn resolve_predicate(
    predicate: &model::Predicate,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
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
        P::Computed { value, .. } => {
            resolve_computed_predicate(value, range, context, bindings, budget)?
        }
        P::OpValue { name, .. } => P::Op {
            name: op_name(name, context, bindings, budget)?,
            range,
        },
        P::Not { predicate, .. } => P::Not {
            predicate: Box::new(resolve_predicate(predicate, context, bindings, budget)?),
            range,
        },
        P::Group { predicate, .. } => P::Group {
            predicate: Box::new(resolve_predicate(predicate, context, bindings, budget)?),
            range,
        },
        P::And { predicates, .. } => P::And {
            predicates: predicates
                .iter()
                .map(|predicate| resolve_predicate(predicate, context, bindings, budget))
                .collect::<Result<_, _>>()?,
            range,
        },
        P::Or { predicates, .. } => P::Or {
            predicates: predicates
                .iter()
                .map(|predicate| resolve_predicate(predicate, context, bindings, budget))
                .collect::<Result<_, _>>()?,
            range,
        },
        predicate => predicate.clone(),
    })
}

pub(super) fn has_dynamic_predicate(predicate: &model::Predicate) -> bool {
    use model::Predicate as P;
    match predicate {
        P::OpValue { .. } | P::Computed { .. } => true,
        P::Not { predicate, .. } | P::Group { predicate, .. } => has_dynamic_predicate(predicate),
        P::And { predicates, .. } | P::Or { predicates, .. } => {
            predicates.iter().any(has_dynamic_predicate)
        }
        _ => false,
    }
}

pub(super) fn has_constructed(expression: &model::Expression) -> bool {
    expression
        .first
        .iter()
        .chain(expression.rest.iter().flat_map(|(_, stages)| stages))
        .any(|stage| {
            matches!(
                stage,
                model::Stage::Comprehension { .. } | model::Stage::Projection { .. } | model::Stage::Computed {..}
            ) || matches!(stage, model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } if has_dynamic_predicate(predicate))
            || nested(stage).into_iter().any(has_constructed)
        })
}

pub(super) fn dispose_value(value: zirql::ValueExpressionKind<Operand>) {
    use zirql::{ValueExpressionKind as K, ValueGenerator as G};
    let mut pending = vec![Located::new(value, 0..0)];
    while let Some(value) = pending.pop() {
        let mut generators = |clauses: Vec<Located<G<Operand>>>| {
            for clause in clauses {
                match clause.value {
                    G::In { source, filter, .. }
                    | G::Bind {
                        value: source,
                        filter,
                        ..
                    } => {
                        pending.push(source);
                        if let Some(filter) = filter {
                            pending.push(filter);
                        }
                    }
                }
            }
        };
        match value.value {
            K::Literal(_) | K::Name(_) => {}
            K::NativeQuery(query) => zirql::dispose_expression(*query),
            K::One(value)
            | K::IndexSet(value)
            | K::Unary { operand: value, .. }
            | K::Field { value, .. }
            | K::TupleAccess { value, .. }
            | K::Reduce { array: value, .. } => pending.push(*value),
            K::Array(values) | K::Tuple(values) | K::Set(values) => pending.extend(values),
            K::Record(fields) => pending.extend(fields.into_iter().map(|(_, value)| value)),
            K::KeyedArray { entries, .. } => {
                for (key, value) in entries {
                    pending.push(key);
                    pending.push(value)
                }
            }
            K::KeyedComprehension {
                clauses,
                key,
                head,
                empty_axis_sources,
            } => {
                generators(clauses);
                pending.push(*key);
                pending.push(*head);
                if let Some(sources) = empty_axis_sources {
                    pending.extend(sources)
                }
            }
            K::Binary { left, right, .. }
            | K::Range {
                lower: left,
                upper: right,
            }
            | K::ArrayAccess {
                array: left,
                index: right,
            }
            | K::Require {
                condition: left,
                value: right,
                ..
            } => {
                pending.push(*left);
                pending.push(*right)
            }
            K::Conditional {
                branches,
                otherwise,
            } => {
                for (condition, value) in branches {
                    pending.push(condition);
                    pending.push(value)
                }
                pending.push(*otherwise)
            }
            K::Comprehension { clauses, head, .. } => {
                generators(clauses);
                pending.push(*head)
            }
        }
    }
}
