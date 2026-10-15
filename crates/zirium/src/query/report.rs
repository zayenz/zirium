//! Native domain adaptation for shared correlated projection and report values.
use super::*;
use std::{ops::Range, sync::Arc};
use zirql::FunctionHost;
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

/// A stage-owned selector keeps its prepared lexical body stable across rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderingSelector {
    MapValue(crate::source::TextRange),
    Query(comprehension::ParsedComprehension),
}
impl OrderingSelector {
    pub(super) fn new(expression: &model::Expression) -> Self {
        if expression.is_map_value() {
            Self::MapValue(expression.range)
        } else {
            Self::from_operand(operand(expression))
        }
    }
    pub(super) fn from_operand(expression: Operand) -> Self {
        Self::Query(native_selector(expression))
    }
    pub(super) fn is_map_value(&self) -> bool {
        matches!(self, Self::MapValue(_))
    }
    pub(super) fn range(&self) -> Range<usize> {
        match self {
            Self::MapValue(range) => program::range(*range),
            Self::Query(value) => value.expression.range.clone(),
        }
    }
    pub(super) fn body(&self) -> Option<&comprehension::ParsedComprehension> {
        match self {
            Self::MapValue(_) => None,
            Self::Query(value) => Some(value),
        }
    }
    pub(super) fn is_read_only(&self) -> bool {
        let Some(value) = self.body() else {
            return true;
        };
        let zirql::ValueExpressionKind::NativeValue(query) = &value.expression.value else {
            unreachable!("ordering selectors retain a native query")
        };
        operand_is_read_only(query)
    }
}
/// Retain the selector's native result through the active argument runner.
pub(super) fn native_selector(expression: Operand) -> comprehension::ParsedComprehension {
    let range = expression.range.clone();
    let query_range = crate::source::TextRange::new(range.start as u32, range.end as u32)
        .expect("ordered selector range");
    let expression = Located::new(
        X::Sequence {
            input: Box::new(expression),
            steps: vec![Located::new(
                Step::Stage(model::Stage::RetainSelector { range: query_range }),
                range.clone(),
            )],
        },
        range.clone(),
    );
    comprehension::ParsedComprehension::new(Located::new(
        zirql::ValueExpressionKind::NativeValue(Box::new(expression)),
        range,
    ))
}
fn map_operand(value: &comprehension::ParsedComprehension) -> &Operand {
    let zirql::ValueExpressionKind::NativeValue(query) = &value.expression.value else {
        unreachable!("map selector retains a native query")
    };
    query
}
pub(super) fn map_is_read_only(
    mapping: &zirql::NativeMap<comprehension::ParsedComprehension>,
) -> bool {
    operand_is_read_only(map_operand(&mapping.key))
        && operand_is_read_only(map_operand(&mapping.value))
}
fn map_result(
    value: QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let value = functions::owned(value, budget, range)?;
    let QueryOutput::Native(NativeValue::Capture(mut record)) = value else {
        unreachable!("map selector retains one native field")
    };
    Ok(record.fields.pop().expect("map selector field").1)
}
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
fn evaluate_selector(
    selector: &comprehension::ParsedComprehension,
    input: QueryOutput,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let value = if let Some(session) = context.functions {
        session.run(&selector.expression, context, &input, bindings, budget)?
    } else {
        zirql::evaluate_expression(
            map_operand(selector),
            &ReportHost,
            input,
            context,
            bindings,
            budget,
        )
        .map_err(program::evaluation_error)?
    };
    map_result(value, budget, range)
}
pub(super) fn native_body_is_read_only(value: &comprehension::ParsedComprehension) -> bool {
    operand_is_read_only(map_operand(value))
}
fn exists_operations(
    value: QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<Vec<OperationId>, EvaluationError> {
    match functions::owned(value, budget, range)? {
        QueryOutput::Native(NativeValue::OperationSequence(operations)) => Ok(operations),
        value => take_operations(value, "where_exists"),
    }
}
pub(super) fn evaluate_exists(
    relation: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let selected = exists_operations(input, budget, range)?;
    zirql::filter_native_exists(
        selected,
        budget,
        Location::Query(range.clone()),
        |operation, budget| {
            program::charge_results(budget, 1, range).map_err(zirql::EvaluationError::Host)?;
            let output = evaluate_selector(
                relation,
                QueryOutput::Operations(vec![*operation]),
                context,
                bindings,
                budget,
                range,
            )
            .map_err(zirql::EvaluationError::Host)?;
            exists_operations(output, budget, range).map_err(zirql::EvaluationError::Host)
        },
    )
    .map(QueryOutput::Operations)
    .map_err(program::evaluation_error)
}
pub(super) fn evaluate_map(
    mapping: &zirql::NativeMap<comprehension::ParsedComprehension>,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<QueryOutput, EvaluationError> {
    let input = functions::owned(input, budget, range)?;
    let selected = take_operations(input, "map_by")?;
    let mut items = 0usize;
    let entries = zirql::map_native_by(
        selected,
        context,
        budget,
        Location::Query(range.clone()),
        |operation, context, budget| {
            program::charge_results(budget, 1, range).map_err(zirql::EvaluationError::Host)?;
            let key = evaluate_selector(
                &mapping.key,
                QueryOutput::Operations(vec![*operation]),
                context,
                bindings,
                budget,
                range,
            )
            .map_err(zirql::EvaluationError::Host)?;
            map_keys(key, budget, range).map_err(zirql::EvaluationError::Host)
        },
        |operation, context, budget| {
            program::charge_results(budget, 1, range).map_err(zirql::EvaluationError::Host)?;
            let value = evaluate_selector(
                &mapping.value,
                QueryOutput::Operations(vec![operation]),
                context,
                bindings,
                budget,
                range,
            )
            .map_err(zirql::EvaluationError::Host)?;
            items = items.saturating_add(output_size(&value).saturating_add(1));
            check_items(budget, items).map_err(zirql::EvaluationError::Host)?;
            charge(budget, output_size(&value)).map_err(zirql::EvaluationError::Host)?;
            Ok(value)
        },
    )
    .map_err(program::evaluation_error)?;
    Ok(QueryOutput::Native(NativeValue::Map(entries)))
}
fn operand_is_read_only(query: &Operand) -> bool {
    let mut pending = vec![query];
    while let Some(query) = pending.pop() {
        match &query.value {
            X::Input | X::Binding(_) => {}
            X::Group(query) | X::Field { value: query, .. } => pending.push(query),
            X::Sequence { input, steps } => {
                if !steps.iter().all(|step| match &step.value {
                    Step::Stage(stage) => stage.is_read_only(),
                    Step::Emit { .. } => false,
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
    if let Some(session) = context.functions {
        let result = session.run(&value.expression, context, &input, bindings, budget)?;
        return functions::constructed(result, budget, &value.expression.range);
    }
    let result = zirql::evaluate_native_value_with_types(
        &value.expression,
        &ReportHost,
        input,
        context,
        bindings,
        budget,
    )
    .map_err(program::evaluation_error)?;
    comprehension::ConstructedReport::with_types(
        result.value,
        result.types,
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
    if let Some(session) = context.functions {
        let subjects = ReportHost
            .subjects(input, context, budget, range)
            .map_err(program::evaluation_error)?;
        capture::slots::<V>(subjects.len(), budget, range)?;
        program::charge_results(budget, subjects.len() + 1, range)?;
        let mut rows = Vec::with_capacity(subjects.len());
        for subject in subjects {
            let relative = program::copy_value(&subject, budget, range)?;
            let scope = bindings
                .with_local(
                    Located::new(
                        value.projection.subject.value.clone(),
                        value.projection.subject.range.clone(),
                    ),
                    subject,
                    budget,
                )
                .map_err(core_error)?;
            capture::slots::<(String, V)>(value.projection.fields.len(), budget, range)?;
            program::charge_results(budget, value.projection.fields.len() + 1, range)?;
            let mut fields = Vec::with_capacity(value.projection.fields.len());
            for (name, field) in &value.projection.fields {
                let result = session.run(field, context, &relative, &scope, budget)?;
                let result = functions::owned(result, budget, &field.range)?;
                let result = ReportHost
                    .project_value(result, context, budget, &field.range)
                    .map_err(program::evaluation_error)?;
                program::charge_bytes(budget, name.value.len(), &name.range)?;
                fields.push((name.value.clone(), result));
            }
            rows.push(V::Record(fields));
        }
        return ConstructedReport::new(V::Array(rows), range.clone(), budget);
    }
    let result = zirql::evaluate_projection_with_types(
        &value.projection,
        &ReportHost,
        input,
        context,
        bindings,
        budget,
    )
    .map_err(program::evaluation_error)?;
    comprehension::ConstructedReport::with_types(result.value, result.types, range.clone(), budget)
}
pub(super) fn runtime_kinds(
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<BTreeMap<String, program::Kind>, EvaluationError> {
    let mut names = BTreeMap::new();
    for (name, value) in bindings.iter() {
        program::charge_bytes(budget, name.len() + 1, range)?;
        names.insert(name.to_owned(), output_kind(value, budget, range)?);
    }
    Ok(names)
}
pub(super) fn output_kind(
    value: &QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<program::Kind, EvaluationError> {
    use program::Kind as K;
    Ok(match functions::unretained(value) {
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
impl<'q> ProgramHost<EvaluationContext<'_, '_, 'q>> for ReportHost<'_> {
    type Value = QueryOutput;
    type Stage = model::Stage;
    type SetOperator = model::SetOperator;
    type Emitter = ();
    type Error = EvaluationError;
    fn copy_value(
        &self,
        value: &QueryOutput,
        context: &mut EvaluationContext<'_, '_, 'q>,
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
        _: &(),
        _: &QueryOutput,
        _: &mut EvaluationContext<'_, '_, 'q>,
        _: &mut Budget,
        range: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<EvaluationError>> {
        Err(zirql::QueryError::query(range.clone(), "constructed expressions cannot emit").into())
    }
}
impl<'q> ProjectionHost<EvaluationContext<'_, '_, 'q>> for ReportHost<'_> {
    type Kind = program::Kind;
    fn tuple_view_kind<'k>(&self, kind: &'k program::Kind) -> Option<&'k [program::Kind]> {
        match kind {
            program::Kind::Tuple(values) => Some(values),
            _ => None,
        }
    }

    fn value_arguments<'s>(
        &self,
        stage: &'s model::Stage,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<
        zirql::NativeValueArguments<'s, EvaluationContext<'_, '_, 'q>, Self>,
        zirql::QueryError,
    > {
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
        fn target_nodes(target: &zirql::Predicate<model::Predicate>) -> usize {
            match &target.value {
                zirql::PredicateKind::Leaf(predicate) => nodes(predicate),
                zirql::PredicateKind::Not(target) | zirql::PredicateKind::Group(target) => {
                    1 + target_nodes(target)
                }
                zirql::PredicateKind::And(targets) | zirql::PredicateKind::Or(targets) => {
                    1 + targets.iter().map(target_nodes).sum::<usize>()
                }
            }
        }
        fn targets<'s>(
            target: &'s zirql::Predicate<model::Predicate>,
            pending: &mut Vec<&'s model::Predicate>,
        ) {
            match &target.value {
                zirql::PredicateKind::Leaf(predicate) => pending.push(predicate),
                zirql::PredicateKind::Not(target) | zirql::PredicateKind::Group(target) => {
                    targets(target, pending)
                }
                zirql::PredicateKind::And(values) | zirql::PredicateKind::Or(values) => {
                    for target in values.iter().rev() {
                        targets(target, pending);
                    }
                }
            }
        }
        if let model::Stage::NativeMap { mapping, .. } = stage {
            let mut values = Vec::new();
            budget.charge_results(2, Location::Query(range.clone()))?;
            for selector in [&mapping.key, &mapping.value] {
                budget.charge(
                    operand_size(map_operand(selector)),
                    Location::Query(range.clone()),
                )?;
                values.push((&*selector.expression, program::Kind::Operations));
            }
            return Ok(values);
        }
        if let model::Stage::Fixpoint {
            expression: model::FixedPointBody::Native(value),
            ..
        }
        | model::Stage::WhereExists {
            relation: value, ..
        } = stage
        {
            budget.charge(1, Location::Query(range.clone()))?;
            budget.charge_results(1, Location::Query(range.clone()))?;
            // Resolution records this exact body; scoped preparation supplies
            // the actual incoming operation or string kind.
            return Ok(vec![(&value.expression, program::Kind::Operations)]);
        }
        if let Some((selector, _)) = ordering_selector(stage)
            && let Some(value) = selector.body()
        {
            let zirql::ValueExpressionKind::NativeValue(query) = &value.expression.value else {
                unreachable!("ordering selectors retain a native query")
            };
            budget.charge(operand_size(query), Location::Query(range.clone()))?;
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(vec![(&value.expression, program::Kind::Operations)]);
        }
        if let model::Stage::FunctionValue { value, .. }
        | model::Stage::Comprehension { value, .. }
        | model::Stage::FunctionPrimitive {
            argument: value, ..
        } = stage
        {
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(vec![(&value.expression, program::Kind::Operations)]);
        }
        let capacity = match stage {
            model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } => {
                nodes(predicate) * 2
            }
            model::Stage::Computed { .. } => 1,
            model::Stage::Paths { value, .. } => target_nodes(&value.paths.target) * 2,
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
            model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } => {
                pending.push(predicate)
            }
            model::Stage::Paths { value, .. } => targets(&value.paths.target, &mut pending),
            _ => {}
        }
        while let Some(predicate) = pending.pop() {
            budget.charge(1, Location::Query(range.clone()))?;
            match predicate {
                model::Predicate::OpValue { name, .. } => {
                    values.push((&*name.expression, program::Kind::Operations))
                }
                model::Predicate::Computed { value, .. } => values.extend(
                    value
                        .operands()
                        .map(|value| (&*value.expression, program::Kind::Operations)),
                ),
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
        stage: &'s model::Stage,
        input: &program::Kind,
        _: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<
        zirql::ScopedNativeValueArguments<'s, EvaluationContext<'_, '_, 'q>, Self>,
        zirql::QueryError,
    > {
        let arguments = if let model::Stage::WhereExists { relation, .. } = stage {
            budget.charge(1, Location::Query(range.clone()))?;
            budget.charge_results(1, Location::Query(range.clone()))?;
            vec![(
                &*relation.expression,
                self.subject_kind(input, budget, range)?,
            )]
        } else if let model::Stage::Fixpoint {
            expression: model::FixedPointBody::Native(value),
            ..
        } = stage
        {
            budget.charge(1, Location::Query(range.clone()))?;
            budget.charge_results(1, Location::Query(range.clone()))?;
            vec![(&*value.expression, input.clone())]
        } else {
            self.value_arguments(stage, budget, range)?
        };
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
                relative: (!self.is_explicit_source(stage)).then_some(kind),
                locals: Vec::new(),
            })
            .collect())
    }
    fn evaluate_stage_with_arguments(
        &self,
        stage: &model::Stage,
        input: QueryOutput,
        mut arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_, 'q>, Self>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<QueryOutput, zirql::EvaluationError<EvaluationError>> {
        if let model::Stage::WhereExists { relation, .. } = stage {
            let native_sequence = matches!(
                functions::unretained(&input),
                QueryOutput::Native(NativeValue::OperationSequence(_))
            );
            let selected =
                exists_operations(input, budget, range).map_err(zirql::EvaluationError::Host)?;
            return zirql::filter_native_exists(
                selected,
                budget,
                Location::Query(range.clone()),
                |operation, budget| {
                    program::charge_results(budget, 1, range)
                        .map_err(zirql::EvaluationError::Host)?;
                    let result = arguments.evaluate_native(
                        &relation.expression,
                        QueryOutput::Operations(vec![*operation]),
                        budget,
                    )?;
                    let result = map_result(
                        QueryOutput::Native(NativeValue::Retained(result.into_shared())),
                        budget,
                        range,
                    )
                    .map_err(zirql::EvaluationError::Host)?;
                    exists_operations(result, budget, range).map_err(zirql::EvaluationError::Host)
                },
            )
            .map(|operations| {
                if native_sequence {
                    QueryOutput::Native(NativeValue::OperationSequence(operations))
                } else {
                    QueryOutput::Operations(operations)
                }
            });
        }
        if let model::Stage::NativeMap { mapping, .. } = stage {
            let input =
                functions::owned(input, budget, range).map_err(zirql::EvaluationError::Host)?;
            let input = match input {
                QueryOutput::Native(NativeValue::OperationSequence(operations)) => {
                    QueryOutput::Operations(operations)
                }
                input => input,
            };
            let selected =
                take_operations(input, "map_by").map_err(zirql::EvaluationError::Host)?;
            let mut items = 0usize;
            let entries = zirql::map_native_by(
                selected,
                arguments,
                budget,
                Location::Query(range.clone()),
                |operation, arguments, budget| {
                    program::charge_results(budget, 1, range)
                        .map_err(zirql::EvaluationError::Host)?;
                    let value = arguments.evaluate_native(
                        &mapping.key.expression,
                        QueryOutput::Operations(vec![*operation]),
                        budget,
                    )?;
                    let value = map_result(
                        QueryOutput::Native(NativeValue::Retained(value.into_shared())),
                        budget,
                        range,
                    )
                    .map_err(zirql::EvaluationError::Host)?;
                    map_keys(value, budget, range).map_err(zirql::EvaluationError::Host)
                },
                |operation, arguments, budget| {
                    program::charge_results(budget, 1, range)
                        .map_err(zirql::EvaluationError::Host)?;
                    let value = arguments.evaluate_native(
                        &mapping.value.expression,
                        QueryOutput::Operations(vec![operation]),
                        budget,
                    )?;
                    let value = map_result(
                        QueryOutput::Native(NativeValue::Retained(value.into_shared())),
                        budget,
                        range,
                    )
                    .map_err(zirql::EvaluationError::Host)?;
                    items = items.saturating_add(output_size(&value).saturating_add(1));
                    check_items(budget, items).map_err(zirql::EvaluationError::Host)?;
                    charge(budget, output_size(&value)).map_err(zirql::EvaluationError::Host)?;
                    Ok(value)
                },
            )?;
            return Ok(QueryOutput::Native(NativeValue::Map(entries)));
        }
        if let model::Stage::Fixpoint {
            expression: model::FixedPointBody::Native(value),
            ..
        } = stage
        {
            let input = fixed_point_selection(input, budget, range)
                .map_err(zirql::EvaluationError::Host)?;
            if fixed_point_is_closure(value) {
                let context = arguments.parts().0;
                return evaluate_closure(
                    context.document,
                    take_operations(input, "closure").map_err(zirql::EvaluationError::Host)?,
                    context.registry,
                    true,
                    budget,
                )
                .map(QueryOutput::Operations)
                .map_err(zirql::EvaluationError::Host);
            }
            return zirql::evaluate_native_fixed_point(
                input,
                &mut arguments,
                budget,
                Location::Query(range.clone()),
                |current, arguments, budget| {
                    let input = fixed_point_copy(current, budget, range)
                        .map_err(zirql::EvaluationError::Host)?;
                    let output = arguments.evaluate_native(&value.expression, input, budget)?;
                    let output = QueryOutput::Native(NativeValue::Retained(output.into_shared()));
                    fixed_point_selection(
                        fixed_point_result(output, budget, range)
                            .map_err(zirql::EvaluationError::Host)?,
                        budget,
                        range,
                    )
                    .map_err(zirql::EvaluationError::Host)
                },
                |current, _, budget| {
                    fixed_point_copy(current, budget, range).map_err(zirql::EvaluationError::Host)
                },
                |left, right, _, budget| {
                    fixed_point_equal(left, right, budget, range)
                        .map_err(zirql::EvaluationError::Host)
                },
            );
        }
        if let Some((selector, order)) = ordering_selector(stage)
            && let Some(value) = selector.body()
        {
            let input =
                functions::owned(input, budget, range).map_err(zirql::EvaluationError::Host)?;
            if matches!(input, QueryOutput::Map(_)) && order == zirql::NativeOrder::Sort {
                return Err(zirql::EvaluationError::Host(
                    EvaluationError::new("sort_by on a map requires the scalar selector value")
                        .at(range.clone()),
                ));
            }
            let input = match input {
                QueryOutput::Native(NativeValue::OperationSequence(operations)) => {
                    QueryOutput::Operations(operations)
                }
                input => input,
            };
            let selected = take_operations(input, ordering_name(order))
                .map_err(zirql::EvaluationError::Host)?;
            if order != zirql::NativeOrder::Sort && selected.is_empty() {
                return Err(zirql::EvaluationError::Host(
                    EvaluationError::new("by extrema require a non-empty operation stream")
                        .at(range.clone()),
                ));
            }
            let selected = zirql::order_native_by(
                selected,
                order,
                budget,
                Location::Query(range.clone()),
                |operation, budget| {
                    program::charge_results(budget, 2, range)
                        .map_err(zirql::EvaluationError::Host)?;
                    budget.charge(1, Location::Query(range.clone()))?;
                    let output = arguments.evaluate_native(
                        &value.expression,
                        QueryOutput::Operations(vec![*operation]),
                        budget,
                    )?;
                    borrowed_ordering_key(output.value(), budget, &value.expression.range)
                        .map_err(zirql::EvaluationError::Host)
                },
            )?;
            return Ok(QueryOutput::Operations(selected));
        }
        let prepared = arguments.parts().0.functions.is_some();
        match stage {
            model::Stage::Paths { value, .. } if prepared => {
                return capture::evaluate_paths_with_arguments(
                    value, self, input, arguments, budget,
                )
                .map(|value| QueryOutput::Native(NativeValue::Captures(value)))
                .map_err(zirql::EvaluationError::Host);
            }
            model::Stage::FunctionValue { value, .. }
            | model::Stage::Comprehension { value, .. }
                if prepared =>
            {
                let value = arguments.evaluate_native(
                    &value.expression,
                    QueryOutput::Operations(vec![]),
                    budget,
                )?;
                return Ok(QueryOutput::Native(NativeValue::Retained(
                    value.into_shared(),
                )));
            }
            model::Stage::FunctionPrimitive {
                primitive,
                argument,
                ..
            } => {
                let argument = arguments.evaluate_native(
                    &argument.expression,
                    QueryOutput::Operations(vec![]),
                    budget,
                )?;
                return functions::primitive(
                    *primitive,
                    argument.value(),
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
            model::Stage::Root { predicate, range } if has_dynamic_predicate(predicate) => {
                resolved = model::Stage::Root {
                    predicate: resolve_prepared_predicate(predicate, arguments, budget)
                        .map_err(zirql::EvaluationError::Host)?,
                    range: *range,
                };
                &resolved
            }
            _ => stage,
        };
        let (context, bindings) = arguments.parts();
        self.evaluate_stage(stage, input, context, bindings, budget, range)
    }
    fn share_constructed(
        &self,
        value: &QueryOutput,
        _: &mut EvaluationContext<'_, '_, 'q>,
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
        _: &mut EvaluationContext<'_, '_, 'q>,
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
            K::NativeMap
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
    fn stage_kind(
        &self,
        stage: &model::Stage,
        input: &program::Kind,
        scope: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        _: &Range<usize>,
    ) -> Result<program::Kind, zirql::QueryError> {
        program::Validator {
            host: Self {
                functions: self.functions,
            },
        }
        .stages(std::slice::from_ref(stage), input.clone(), scope, budget)
        .map_err(query_error)
    }
    fn is_explicit_source(&self, stage: &model::Stage) -> bool {
        matches!(
            stage,
            model::Stage::FunctionInput { .. }
                | model::Stage::FunctionValue { .. }
                | model::Stage::FunctionPrimitive { .. }
        )
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
        context: &mut EvaluationContext<'_, '_, 'q>,
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
            K::Sequence(element) => self.declaration_kind(element, budget, range),
            K::Operations | K::Ssa | K::Capture(_) => Ok(input.clone()),
            K::Captures(schema) => Ok(K::Capture(Arc::clone(schema))),
            _ => Err(zirql::QueryError::query(
                range.clone(),
                "native enumeration requires operations, SSA or captures",
            )),
        }
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
        _: &mut EvaluationContext<'_, '_, 'q>,
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
        _: &mut EvaluationContext<'_, '_, 'q>,
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
    use model::{Stage as S, StructuredStage as T};
    match stage {
        S::Group { expression, .. }
        | S::Fixpoint {
            expression: model::FixedPointBody::Selection(expression),
            ..
        } => vec![expression],
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
        .map(stage_size)
        .fold(0usize, usize::saturating_add)
}
fn operand_size(query: &Operand) -> usize {
    let mut size = 0usize;
    let mut pending = vec![query];
    while let Some(query) = pending.pop() {
        size = size.saturating_add(1);
        match &query.value {
            X::Input => {}
            X::Binding(name) => size = size.saturating_add(name.len()),
            X::Group(query) | X::Field { value: query, .. } => pending.push(query),
            X::Sequence { input, steps } => {
                size = steps
                    .iter()
                    .map(|step| match &step.value {
                        Step::Stage(stage) => stage_size(stage),
                        Step::Emit { .. } => 1,
                    })
                    .fold(size, usize::saturating_add);
                pending.push(input);
            }
            X::Set { left, right, .. } => {
                pending.push(right);
                pending.push(left);
            }
            X::Record(fields) => {
                for (name, value) in fields.iter().rev() {
                    size = size.saturating_add(name.value.len());
                    pending.push(value);
                }
            }
        }
    }
    size
}

fn stage_size(stage: &model::Stage) -> usize {
    use model::{Stage as S, StructuredStage as T};
    let payload = match stage {
        S::WhereExists { relation, .. } => operand_size(map_operand(relation)),
        S::NativeMap { mapping, .. } => operand_size(map_operand(&mapping.key))
            .saturating_add(operand_size(map_operand(&mapping.value))),
        S::Binding { name, .. }
        | S::Attr { name, .. }
        | S::RemoveAttr { name, .. }
        | S::Structured {
            step: T::StringAttr(name) | T::Attributes(name),
            ..
        } => name.len(),
        S::Filter { predicate, .. } | S::Root { predicate, .. } => predicate_size(predicate),
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
pub(super) fn op_name(
    value: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<String, EvaluationError> {
    op_name_by(
        value,
        &mut |value, budget| scalar_argument(value, context, bindings, budget),
        budget,
    )
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
pub(super) fn resolve_stage(
    value: &model::ComputedStage,
    range: crate::source::TextRange,
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<model::Stage, EvaluationError> {
    resolve_stage_by(
        value,
        range,
        &mut |value, budget| scalar_argument(value, context, bindings, budget),
        budget,
    )
}

pub(super) fn push_expressions<'a>(
    stage: &'a model::Stage,
    pending: &mut Vec<&'a model::Expression>,
) {
    match stage {
        model::Stage::NativeMap { mapping, .. } => {
            visit_value(&mapping.key.expression, pending);
            visit_value(&mapping.value.expression, pending);
        }

        model::Stage::Fixpoint {
            expression: model::FixedPointBody::Native(value),
            ..
        }
        | model::Stage::WhereExists {
            relation: value, ..
        } => visit_value(&value.expression, pending),
        model::Stage::SortBy { selector, .. }
        | model::Stage::MinBy { selector, .. }
        | model::Stage::MaxBy { selector, .. }
        | model::Stage::MinAllBy { selector, .. }
        | model::Stage::MaxAllBy { selector, .. } => {
            if let Some(value) = selector.body() {
                visit_value(&value.expression, pending);
            }
        }
        model::Stage::Native { value, .. }
        | model::Stage::Operand {
            operation: value, ..
        } => visit_operand(&value.expression, pending),
        model::Stage::Paths { value, .. } => {
            visit_match_predicate(&value.paths.target, pending);
        }
        model::Stage::Match { value, .. } => {
            for clause in &value.matching.clauses {
                match &clause.value {
                    zirql::MatchClause::From { source, .. }
                    | zirql::MatchClause::Bind { value: source, .. } => {
                        visit_operand(source, pending)
                    }
                    zirql::MatchClause::Where { predicate } => {
                        visit_match_predicate(predicate, pending)
                    }
                }
            }
        }
        model::Stage::PatternBinding { value, .. } => visit_value(&value.value.expression, pending),
        model::Stage::Computed { value, .. } => visit_value(&value.operand().expression, pending),
        model::Stage::FunctionValue { value, .. }
        | model::Stage::FunctionPrimitive {
            argument: value, ..
        }
        | model::Stage::Comprehension { value, .. } => visit_value(&value.expression, pending),
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
        model::Predicate::SameValue { left, right, .. } => {
            visit_operand(&left.expression, pending);
            visit_operand(&right.expression, pending);
        }
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
fn visit_match_predicate<'a>(
    predicate: &'a zirql::Predicate<model::Predicate>,
    pending: &mut Vec<&'a model::Expression>,
) {
    match &predicate.value {
        zirql::PredicateKind::Leaf(guard) => visit_predicate(guard, pending),
        zirql::PredicateKind::Not(child) | zirql::PredicateKind::Group(child) => {
            visit_match_predicate(child, pending)
        }
        zirql::PredicateKind::And(children) | zirql::PredicateKind::Or(children) => {
            for child in children {
                visit_match_predicate(child, pending);
            }
        }
    }
}
fn visit_operand<'a>(query: &'a Operand, pending: &mut Vec<&'a model::Expression>) {
    let mut queries = vec![query];
    while let Some(query) = queries.pop() {
        match &query.value {
            X::Input | X::Binding(_) => {}
            X::Field { value, .. } => queries.push(value),
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
pub(super) fn visit_value<'a>(
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
        K::Literal(_) | K::CurrentInput | K::Name(_) | K::IdentitySet { .. } => {}
        K::NativeQuery(query) | K::NativeValue(query) => visit_operand(query, pending),
        K::Block { bindings, value } => {
            for (_, source) in bindings {
                visit_value(source, pending);
            }
            visit_value(value, pending);
        }
        K::ValueMatch { subject, arms } => {
            visit_value(subject, pending);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    visit_value(guard, pending);
                }
                visit_value(&arm.value, pending);
            }
        }
        K::Loop { states, body } => {
            for (_, value) in states {
                visit_value(value, pending);
            }
            visit_value(body, pending);
        }
        K::Fold { items, initial, .. } | K::Scan { items, initial, .. } => {
            visit_value(items, pending);
            visit_value(initial, pending);
        }
        K::Call { arguments, .. } | K::Continue(arguments) => {
            for value in arguments {
                visit_value(value, pending);
            }
        }
        K::Queue { items: value, .. }
        | K::PopFront(value)
        | K::Break(value)
        | K::Occurs(value)
        | K::Absent(value)
        | K::One(value)
        | K::IndexSet(value)
        | K::Entries(value)
        | K::Unique(value)
        | K::Tally(value)
        | K::Unary { operand: value, .. }
        | K::Field { value, .. }
        | K::TupleAccess { value, .. }
        | K::GroupBy { rows: value, .. }
        | K::Reduce { array: value, .. }
        | K::FilterStartsWith { array: value, .. }
        | K::Sort { array: value, .. }
        | K::Extremum { array: value, .. } => visit_value(value, pending),
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
        K::Dictionary(entries) | K::KeyedArray { entries, .. } => {
            for (key, value) in entries {
                visit_value(key, pending);
                visit_value(value, pending)
            }
        }
        K::DictionaryComprehension {
            clauses: generators,
            key,
            head,
        } => {
            clauses(generators, pending);
            visit_value(key, pending);
            visit_value(head, pending);
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
        | K::Lookup {
            value: left,
            key: right,
        }
        | K::HasKey {
            value: left,
            key: right,
        }
        | K::Default {
            value: left,
            fallback: right,
        }
        | K::ZipExact { left, right }
        | K::PushBackAll {
            queue: left,
            items: right,
        }
        | K::Insert {
            set: left,
            item: right,
        }
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
    resolve_predicate_by(
        predicate,
        &mut |value, budget| scalar_argument(value, context, bindings, budget),
        budget,
    )
}
pub(super) fn resolve_prepared_predicate(
    predicate: &model::Predicate,
    arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_, '_>, ReportHost<'_>>,
    budget: &mut Budget,
) -> Result<model::Predicate, EvaluationError> {
    resolve_predicate_by(
        predicate,
        &mut |value, budget| {
            arguments
                .evaluate(&value.expression, QueryOutput::Operations(vec![]), budget)
                .map_err(program::evaluation_error)?
                .into_owned(budget)
                .map_err(core_error)
        },
        budget,
    )
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
        P::OpValue { .. } | P::Computed { .. } => true,
        P::Not { predicate, .. } | P::Group { predicate, .. } => has_dynamic_predicate(predicate),
        P::And { predicates, .. } | P::Or { predicates, .. } => {
            predicates.iter().any(has_dynamic_predicate)
        }
        _ => false,
    }
}

/// Native/control forms need the prepared runner even without declarations.
/// Ordinary report values retain their existing constructed-only path.
pub(super) fn requires_session(expression: &model::Expression) -> bool {
    expression
        .first
        .iter()
        .chain(expression.rest.iter().flat_map(|(_, stages)| stages))
        .any(stage_requires_session)
}
fn stage_requires_session(stage: &model::Stage) -> bool {
    if matches!(
        stage,
        model::Stage::NativeMap { .. } | model::Stage::WhereExists { .. }
    ) {
        return true;
    }
    if ordering_selector(stage).is_some_and(|(selector, _)| selector.body().is_some()) {
        return true;
    }
    match stage {
        model::Stage::PatternBinding { .. }
        | model::Stage::FunctionPrimitive { .. }
        | model::Stage::FunctionInput { .. } => true,
        model::Stage::Comprehension { value, .. } | model::Stage::FunctionValue { value, .. } => {
            value_requires_session(&value.expression)
        }
        model::Stage::Projection { value, .. } => value
            .projection
            .fields
            .iter()
            .any(|(_, value)| value_requires_session(value)),
        model::Stage::Computed { value, .. } => value_requires_session(&value.operand().expression),
        model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } => {
            predicate_requires_session(predicate)
        }
        _ => nested(stage).into_iter().any(requires_session),
    }
}
fn operand_requires_session(query: &Operand) -> bool {
    let mut pending = vec![query];
    while let Some(query) = pending.pop() {
        match &query.value {
            X::Input | X::Binding(_) => {}
            X::Group(query) | X::Field { value: query, .. } => pending.push(query),
            X::Sequence { input, steps } => {
                if steps.iter().any(|step| match &step.value {
                    Step::Stage(stage) => stage_requires_session(stage),
                    Step::Emit { .. } => false,
                }) {
                    return true;
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
    false
}

fn predicate_requires_session(predicate: &model::Predicate) -> bool {
    match predicate {
        model::Predicate::Computed { value, .. } => value
            .operands()
            .any(|value| value_requires_session(&value.expression)),
        model::Predicate::OpValue { name, .. } => value_requires_session(&name.expression),
        model::Predicate::Not { predicate, .. } | model::Predicate::Group { predicate, .. } => {
            predicate_requires_session(predicate)
        }
        model::Predicate::And { predicates, .. } | model::Predicate::Or { predicates, .. } => {
            predicates.iter().any(predicate_requires_session)
        }
        _ => false,
    }
}
pub(super) fn value_requires_session(value: &zirql::ValueExpression<Operand>) -> bool {
    value_requires_session_with_names(value, None)
}

pub(super) fn value_requires_session_in_scope(
    value: &zirql::ValueExpression<Operand>,
    names: &BTreeMap<String, program::Kind>,
) -> bool {
    value_requires_session_with_names(value, Some(names))
}

fn value_requires_session_with_names(
    value: &zirql::ValueExpression<Operand>,
    names: Option<&BTreeMap<String, program::Kind>>,
) -> bool {
    use zirql::ValueExpressionKind as K;
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match &value.value {
            K::Block { .. }
            | K::ValueMatch { .. }
            | K::Call { .. }
            | K::NativeValue(_)
            | K::Loop { .. }
            | K::Fold { .. }
            | K::Scan { .. }
            | K::Break(_)
            | K::Continue(_)
            | K::Queue { .. }
            | K::PopFront(_)
            | K::PushBackAll { .. }
            | K::IdentitySet { .. }
            | K::Insert { .. } => return true,
            K::Literal(_) | K::CurrentInput => {}
            K::Name(name) => {
                if matches!(
                    names.and_then(|names| names.get(name)),
                    Some(
                        program::Kind::Computational(_)
                            | program::Kind::Tuple(_)
                            | program::Kind::Sequence(_)
                            | program::Kind::Capture(_)
                            | program::Kind::Captures(_)
                    )
                ) {
                    return true;
                }
            }
            K::NativeQuery(query) => {
                if operand_requires_session(query) {
                    return true;
                }
            }
            K::Occurs(v)
            | K::Absent(v)
            | K::One(v)
            | K::IndexSet(v)
            | K::Entries(v)
            | K::Unique(v)
            | K::Tally(v)
            | K::Unary { operand: v, .. }
            | K::Field { value: v, .. }
            | K::TupleAccess { value: v, .. }
            | K::GroupBy { rows: v, .. }
            | K::Reduce { array: v, .. }
            | K::FilterStartsWith { array: v, .. }
            | K::Sort { array: v, .. }
            | K::Extremum { array: v, .. } => pending.push(v),
            K::Array(values) | K::Tuple(values) | K::Set(values) => pending.extend(values),
            K::Record(fields) => pending.extend(fields.iter().map(|(_, value)| value)),
            K::Dictionary(entries) | K::KeyedArray { entries, .. } => {
                for (k, v) in entries {
                    pending.push(k);
                    pending.push(v);
                }
            }
            K::Binary { left, right, .. }
            | K::ZipExact { left, right }
            | K::Lookup {
                value: left,
                key: right,
            }
            | K::HasKey {
                value: left,
                key: right,
            }
            | K::Default {
                value: left,
                fallback: right,
            }
            | K::ArrayAccess {
                array: left,
                index: right,
            }
            | K::Range {
                lower: left,
                upper: right,
            }
            | K::Require {
                condition: left,
                value: right,
                ..
            } => {
                pending.push(left);
                pending.push(right);
            }
            K::Conditional {
                branches,
                otherwise,
            } => {
                for (c, v) in branches {
                    pending.push(c);
                    pending.push(v);
                }
                pending.push(otherwise);
            }
            K::Comprehension { clauses, head, .. }
            | K::DictionaryComprehension { clauses, head, .. }
            | K::KeyedComprehension { clauses, head, .. } => {
                for clause in clauses {
                    match &clause.value {
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
                pending.push(head);
                if let K::DictionaryComprehension { key, .. } = &value.value {
                    pending.push(key);
                }
                if let K::KeyedComprehension {
                    key,
                    empty_axis_sources,
                    ..
                } = &value.value
                {
                    pending.push(key);
                    if let Some(sources) = empty_axis_sources {
                        pending.extend(sources);
                    }
                }
            }
        }
    }
    false
}

pub(super) fn has_constructed(expression: &model::Expression) -> bool {
    expression
        .first
        .iter()
        .chain(expression.rest.iter().flat_map(|(_, stages)| stages))
        .any(|stage| {
            matches!(
                stage,
                model::Stage::WhereExists { .. } | model::Stage::Fixpoint { expression: model::FixedPointBody::Native(_), .. } | model::Stage::PatternBinding { .. } | model::Stage::FunctionInput{..} | model::Stage::FunctionValue{..} | model::Stage::FunctionPrimitive{..} | model::Stage::Paths { .. } | model::Stage::Match { .. } | model::Stage::Native { .. } | model::Stage::Operand { .. } | model::Stage::Comprehension { .. } | model::Stage::Projection { .. } | model::Stage::Computed {..}
            ) || matches!(stage, model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } if has_dynamic_predicate(predicate))
            || nested(stage).into_iter().any(has_constructed)
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
    if host.functions.is_empty() && !value_requires_session(&value.expression) {
        return types(value, input, names, budget);
    }
    let kind =
        functions::value_kind(host, &value.expression, input, names, budget).map_err(core_error)?;
    host.scalar_types(&kind, budget, &value.expression.range)
        .map_err(core_error)?
        .ok_or_else(|| {
            EvaluationError::new("report requires a constructed value")
                .at(value.expression.range.clone())
        })
}
pub(super) fn projection_types_with_host(
    value: &ParsedProjection,
    input: program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &ReportHost<'_>,
) -> Result<Vec<ValueType>, EvaluationError> {
    if host.functions.is_empty()
        && value
            .projection
            .fields
            .iter()
            .all(|(_, value)| !value_requires_session(value))
    {
        return projection_types(value, input, names, budget);
    }
    let projection = &value.projection;
    if names.contains_key(&projection.subject.value)
        || parser::is_reserved(&projection.subject.value)
        || projection.subject.value == "_"
    {
        return Err(EvaluationError::new(
            "projection subject shadows an existing or reserved name",
        )
        .at(projection.subject.range.clone()));
    }
    let mut scope =
        functions::copy_scope(names, budget, &projection.subject.range).map_err(core_error)?;
    scope.insert(
        projection.subject.value.clone(),
        host.subject_kind(&input, budget, &projection.subject.range)
            .map_err(core_error)?,
    );
    let mut fields = Vec::new();
    for (name, value) in &projection.fields {
        if fields.iter().any(|(other, _)| other == &name.value) {
            return Err(EvaluationError::new("duplicate projection field").at(name.range.clone()));
        }
        let kind = functions::value_kind(host, value, input.clone(), &scope, budget)
            .map_err(core_error)?;
        let types = host
            .scalar_types(&kind, budget, &value.range)
            .map_err(core_error)?
            .ok_or_else(|| {
                EvaluationError::new("projection field needs a constructed value")
                    .at(value.range.clone())
            })?;
        fields.push((
            name.value.clone(),
            types.into_iter().next().unwrap_or(ValueType::Unestablished),
        ));
    }
    Ok(vec![ValueType::Array(zirql::ArrayType::Elements(vec![
        ValueType::Record(fields),
    ]))])
}
pub(super) fn validate_argument_with_host(
    value: &comprehension::ParsedComprehension,
    expected: ValueType,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &ReportHost<'_>,
) -> Result<(), EvaluationError> {
    if types_with_host(value, program::Kind::Operations, names, budget, host)?
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

pub(super) fn validate_predicate_with_host(
    predicate: &model::Predicate,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &ReportHost<'_>,
) -> Result<(), EvaluationError> {
    match predicate {
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
            if types_with_host(name, program::Kind::Operations, names, budget, host)?
                .iter()
                .any(|ty| *ty != ValueType::String)
            {
                return Err(EvaluationError::new("op argument needs a string")
                    .at(name.expression.range.clone()));
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

pub(super) fn ordering_selector(
    stage: &model::Stage,
) -> Option<(&OrderingSelector, zirql::NativeOrder)> {
    use model::Stage as S;
    use zirql::NativeOrder as O;
    Some(match stage {
        S::SortBy { selector, .. } => (selector, O::Sort),
        S::MinBy { selector, .. } => (selector, O::Min),
        S::MaxBy { selector, .. } => (selector, O::Max),
        S::MinAllBy { selector, .. } => (selector, O::MinAll),
        S::MaxAllBy { selector, .. } => (selector, O::MaxAll),
        _ => return None,
    })
}
fn ordering_name(order: zirql::NativeOrder) -> &'static str {
    match order {
        zirql::NativeOrder::Sort => "sort_by",
        zirql::NativeOrder::Min => "min_by",
        zirql::NativeOrder::Max => "max_by",
        zirql::NativeOrder::MinAll => "min_all_by",
        zirql::NativeOrder::MaxAll => "max_all_by",
    }
}

pub(super) fn native_body_kind(
    value: &comprehension::ParsedComprehension,
    input: &program::Kind,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    host: &ReportHost<'_>,
) -> Result<program::Kind, EvaluationError> {
    let zirql::ValueExpressionKind::NativeValue(query) = &value.expression.value else {
        unreachable!("fixed points retain a native body")
    };
    let kind =
        zirql::native_expression_kind(query, host, input, names, budget).map_err(core_error)?;
    let program::Kind::Capture(fields) = kind else {
        unreachable!("fixed points retain one native selection field")
    };
    Ok(fields.fields[0].1.clone())
}

pub(super) fn fixed_point_is_closure(value: &comprehension::ParsedComprehension) -> bool {
    fn closure(query: &Operand) -> bool {
        match &query.value {
            X::Group(query) => closure(query),
            X::Sequence { input, steps } => match steps.as_slice() {
                [] => closure(input),
                [step] if matches!(input.value, X::Input) => {
                    matches!(step.value, Step::Stage(model::Stage::Closure { .. }))
                }
                _ => false,
            },
            _ => false,
        }
    }
    let zirql::ValueExpressionKind::NativeValue(query) = &value.expression.value else {
        return false;
    };
    let X::Sequence { input, .. } = &query.value else {
        return false;
    };
    closure(input)
}
