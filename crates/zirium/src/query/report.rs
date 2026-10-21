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
    let value = context
        .functions
        .run(&selector.expression, context, &input, bindings, budget)?;
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

/// Preserve tally's following location when native preparation checks one stage
/// at a time. Constructed tally is terminal; native tally may feed map stages.
pub(super) fn retain_tally_continuations(
    query: &mut Operand,
    budget: &mut Budget,
) -> Result<(), zirql::QueryError> {
    let mut pending = vec![query];
    while let Some(query) = pending.pop() {
        match &mut query.value {
            X::Sequence { input, steps } => {
                pending.push(input);
                let mut index = 1;
                while index < steps.len() {
                    if matches!(
                        steps[index - 1].value,
                        Step::Stage(model::Stage::Tally { .. })
                    ) && !matches!(
                        steps[index].value,
                        Step::Stage(model::Stage::TallyContinuation { .. }) | Step::Emit { .. }
                    ) {
                        let range = steps[index].range.clone();
                        budget.charge(
                            std::mem::size_of::<model::Stage>(),
                            Location::Query(range.clone()),
                        )?;
                        budget.charge_results(1, Location::Query(range.clone()))?;
                        let native_range =
                            crate::source::TextRange::new(range.start as u32, range.end as u32)
                                .expect("ordered stage range");
                        steps.insert(
                            index,
                            Located::new(
                                Step::Stage(model::Stage::TallyContinuation {
                                    range: native_range,
                                }),
                                range,
                            ),
                        );
                        index += 1;
                    }
                    index += 1;
                }
            }
            X::Group(query) | X::Field { value: query, .. } => pending.push(query),
            X::Set { left, right, .. } => {
                pending.push(left);
                pending.push(right);
            }
            X::Record(fields) => pending.extend(fields.iter_mut().map(|(_, value)| value)),
            X::Input | X::Binding(_) => {}
        }
    }
    Ok(())
}

/// Constructed and native collections share stage spellings. Only a caller
/// with an established constructed kind uses this shared value tree.
pub(super) fn constructed_suffix(
    stage: &model::Stage,
) -> Option<&comprehension::ParsedComprehension> {
    match stage {
        model::Stage::ConstructedSuffix { value, .. }
        | model::Stage::Count {
            constructed: value, ..
        }
        | model::Stage::Sort {
            constructed: value, ..
        }
        | model::Stage::Unique {
            constructed: value, ..
        }
        | model::Stage::Tally {
            constructed: value, ..
        }
        | model::Stage::Filter {
            constructed: Some(value),
            ..
        } => Some(value),
        _ => None,
    }
}

/// These fixed bodies belong to the overloaded native stages. Retaining them
/// does not opt native requests into constructed allocation accounting; their
/// preparation and execution are charged when the input is constructed.
pub(super) fn collection_suffix(
    suffix: zirql::ValueSuffix<Operand>,
    range: crate::source::TextRange,
) -> comprehension::ParsedComprehension {
    let range = program::range(range);
    let input = Located::new(
        zirql::ValueExpressionKind::NativeQuery(Box::new(Located::new(X::Input, range.clone()))),
        range.clone(),
    );
    comprehension::ParsedComprehension::new(
        zirql::lower_value_suffix(input, Located::new(suffix, range))
            .expect("a suffix on current input cannot follow a terminal tally"),
    )
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

pub(super) fn evaluate(
    value: &comprehension::ParsedComprehension,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<comprehension::ConstructedReport, EvaluationError> {
    let result = context
        .functions
        .run(&value.expression, context, &input, bindings, budget)?;
    functions::constructed(result, budget, &value.expression.range)
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
    let result =
        context
            .functions
            .run_projection(&value.projection, context, input, bindings, budget)?;
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
        if let model::Stage::Projection { value, .. } = stage {
            capture::slots::<(&zirql::ValueExpression<Operand>, program::Kind)>(
                value.projection.fields.len(),
                budget,
                range,
            )
            .map_err(query_error)?;
            return Ok(value
                .projection
                .fields
                .iter()
                .map(|(_, field)| (field, program::Kind::Operations))
                .collect());
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
        | model::Stage::ConstructedSuffix { value, .. }
        | model::Stage::Count {
            constructed: value, ..
        }
        | model::Stage::Sort {
            constructed: value, ..
        }
        | model::Stage::Unique {
            constructed: value, ..
        }
        | model::Stage::Tally {
            constructed: value, ..
        } = stage
        {
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(vec![(&value.expression, program::Kind::Operations)]);
        }
        if let model::Stage::Native { value, .. }
        | model::Stage::Operand {
            operation: value, ..
        } = stage
        {
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(vec![(&value.value.expression, program::Kind::Operations)]);
        }
        if let model::Stage::Match { value, .. } = stage {
            let mut values = Vec::new();
            let mut pending = Vec::new();
            for clause in &value.matching.clauses {
                match &clause.value {
                    zirql::MatchClause::Where { predicate } => targets(predicate, &mut pending),
                    zirql::MatchClause::From { source, .. }
                    | zirql::MatchClause::Bind { value: source, .. } => {
                        budget.charge_results(1, Location::Query(range.clone()))?;
                        values.push((
                            capture::match_source_value(source),
                            program::Kind::Operations,
                        ));
                    }
                }
            }
            while let Some(predicate) = pending.pop() {
                match predicate {
                    model::Predicate::Expression { .. } | model::Predicate::SameValue { .. } => {
                        for value in capture::guard_values(predicate) {
                            budget.charge_results(1, Location::Query(range.clone()))?;
                            values.push((value, program::Kind::Operations));
                        }
                    }
                    model::Predicate::Not { predicate, .. }
                    | model::Predicate::Group { predicate, .. } => pending.push(predicate),
                    model::Predicate::And { predicates, .. }
                    | model::Predicate::Or { predicates, .. } => pending.extend(predicates),
                    _ => {}
                }
            }
            return Ok(values);
        }
        if let model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } = stage
            && !has_dynamic_predicate(predicate)
        {
            if let model::Stage::Filter {
                constructed: Some(value),
                ..
            } = stage
            {
                budget.charge_results(1, Location::Query(range.clone()))?;
                return Ok(vec![(&value.expression, program::Kind::Operations)]);
            }
            return Ok(Vec::new());
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
        if let model::Stage::Filter {
            constructed: Some(value),
            ..
        } = stage
        {
            budget.charge_results(1, Location::Query(range.clone()))?;
            values.push((&*value.expression, program::Kind::Operations));
        }
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
                model::Predicate::Expression { value: name, .. }
                | model::Predicate::OpValue { name, .. } => {
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
        names: &BTreeMap<String, program::Kind>,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<
        zirql::ScopedNativeValueArguments<'s, EvaluationContext<'_, '_, 'q>, Self>,
        zirql::QueryError,
    > {
        if let model::Stage::FunctionValue { value, .. } = stage {
            budget.charge_results(1, Location::Query(range.clone()))?;
            return Ok(vec![zirql::ScopedNativeArgument {
                expression: &value.expression,
                relative: Some(input.clone()),
                locals: Vec::new(),
            }]);
        }
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
            }]);
        }
        if let model::Stage::Match { value, .. } = stage {
            let schema = zirql::match_types(&value.matching, self, input, names, budget)?;
            let mut arguments = Vec::new();
            let mut locals = Vec::new();
            for clause in &value.matching.clauses {
                match &clause.value {
                    zirql::MatchClause::From { name, source }
                    | zirql::MatchClause::Bind {
                        name,
                        value: source,
                    } => {
                        budget.charge_results(1 + locals.len(), Location::Query(range.clone()))?;
                        arguments.push(zirql::ScopedNativeArgument {
                            expression: capture::match_source_value(source),
                            relative: Some(input.clone()),
                            locals: locals.clone(),
                        });
                        let (_, kind) = schema
                            .fields
                            .iter()
                            .find(|(field, _)| field == &name.value)
                            .expect("match schema contains every capture");
                        locals.push((name.clone(), kind.clone()));
                    }
                    zirql::MatchClause::Where { predicate } => {
                        let mut pending = vec![predicate];
                        while let Some(predicate) = pending.pop() {
                            match &predicate.value {
                                zirql::PredicateKind::Leaf(guard) => {
                                    for expression in capture::guard_values(guard) {
                                        budget.charge_results(
                                            1 + locals.len(),
                                            Location::Query(range.clone()),
                                        )?;
                                        arguments.push(zirql::ScopedNativeArgument {
                                            expression,
                                            relative: Some(input.clone()),
                                            locals: locals.clone(),
                                        });
                                    }
                                }
                                zirql::PredicateKind::Not(value)
                                | zirql::PredicateKind::Group(value) => pending.push(value),
                                zirql::PredicateKind::And(values)
                                | zirql::PredicateKind::Or(values) => pending.extend(values),
                            }
                        }
                    }
                }
            }
            return Ok(arguments);
        }
        if matches!(input, program::Kind::Constructed(_)) {
            let value = constructed_suffix(stage);
            if let Some(value) = value {
                budget.charge_results(1, Location::Query(range.clone()))?;
                return Ok(vec![zirql::ScopedNativeArgument {
                    expression: &value.expression,
                    relative: Some(input.clone()),
                    locals: Vec::new(),
                }]);
            }
        }
        // Resolution declares the stable overload body; native input uses the
        // direct stage and must not prepare that body as a value argument.
        if matches!(
            stage,
            model::Stage::Count { .. }
                | model::Stage::Sort { .. }
                | model::Stage::Unique { .. }
                | model::Stage::Tally { .. }
        ) {
            return Ok(Vec::new());
        }
        if let model::Stage::Projection { value, .. } = stage {
            let projection = &value.projection;
            let subject = self.subject_kind(input, budget, &projection.subject.range)?;
            capture::slots::<zirql::ScopedNativeArgument<'_, Operand, program::Kind>>(
                projection.fields.len(),
                budget,
                range,
            )
            .map_err(query_error)?;
            let mut arguments = Vec::with_capacity(projection.fields.len());
            for (_, expression) in &projection.fields {
                capture::slots::<(zirql::Located<String>, program::Kind)>(1, budget, range)
                    .map_err(query_error)?;
                budget.charge(
                    projection.subject.value.len(),
                    Location::Query(projection.subject.range.clone()),
                )?;
                arguments.push(zirql::ScopedNativeArgument {
                    expression,
                    relative: Some(subject.clone()),
                    locals: vec![(projection.subject.clone(), subject.clone())],
                });
            }
            return Ok(arguments);
        }
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
            let mut values = self.value_arguments(stage, budget, range)?;
            if let model::Stage::Filter {
                constructed: Some(value),
                ..
            } = stage
            {
                values.retain(|(expression, _)| !std::ptr::eq(*expression, &*value.expression));
            }
            values
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
        let input = functions::owned(input, budget, range).map_err(zirql::EvaluationError::Host)?;
        if matches!(
            functions::unretained(&input),
            QueryOutput::Native(NativeValue::Constructed(_))
        ) {
            let value = constructed_suffix(stage);
            if let Some(value) = value {
                return arguments
                    .evaluate_native(&value.expression, input, budget)
                    .map(|value| QueryOutput::Native(NativeValue::Retained(value.into_shared())));
            }
        }
        if let model::Stage::Projection { value, .. } = stage {
            let result = arguments.evaluate_projection(&value.projection, input, budget)?;
            return comprehension::ConstructedReport::with_types(
                result.value,
                result.types,
                range.clone(),
                budget,
            )
            .map(|value| QueryOutput::Native(NativeValue::Constructed(value)))
            .map_err(zirql::EvaluationError::Host);
        }
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
            model::Stage::Match { value, .. } => {
                return capture::evaluate_with_arguments(value, self, input, arguments, budget)
                    .map(|value| QueryOutput::Native(NativeValue::Captures(value)))
                    .map_err(zirql::EvaluationError::Host);
            }
            model::Stage::Paths { value, .. } => {
                return capture::evaluate_paths_with_arguments(
                    value, self, input, arguments, budget,
                )
                .map(|value| QueryOutput::Native(NativeValue::Captures(value)))
                .map_err(zirql::EvaluationError::Host);
            }
            model::Stage::FunctionValue { value, .. }
            | model::Stage::Comprehension { value, .. } => {
                let value = arguments.evaluate_native(&value.expression, input, budget)?;
                return Ok(QueryOutput::Native(NativeValue::Retained(
                    value.into_shared(),
                )));
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
        if let model::Stage::FunctionValue { value, .. } = stage
            && matches!(
                value.expression.value,
                zirql::ValueExpressionKind::PredicateScope { .. }
                    | zirql::ValueExpressionKind::SubjectScope { .. }
            )
        {
            return false;
        }
        matches!(
            stage,
            model::Stage::FunctionInput { .. } | model::Stage::FunctionValue { .. }
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
    let count = zirql::nonnegative_integer_to_usize(
        integer,
        ValueLocation::Query(value.operand().expression.range.clone()),
        "primitive argument needs a non-negative int fitting usize",
    )
    .map_err(core_error)?;
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
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    match stage {
        model::Stage::NativeMap { mapping, .. } => {
            visit_value(&mapping.key.expression, pending, budget)?;
            visit_value(&mapping.value.expression, pending, budget)?;
        }

        model::Stage::Fixpoint {
            expression: model::FixedPointBody::Native(value),
            ..
        }
        | model::Stage::WhereExists {
            relation: value, ..
        } => visit_value(&value.expression, pending, budget)?,
        model::Stage::SortBy { selector, .. }
        | model::Stage::MinBy { selector, .. }
        | model::Stage::MaxBy { selector, .. }
        | model::Stage::MinAllBy { selector, .. }
        | model::Stage::MaxAllBy { selector, .. } => {
            if let Some(value) = selector.body() {
                visit_value(&value.expression, pending, budget)?;
            }
        }
        model::Stage::Native { value, .. }
        | model::Stage::Operand {
            operation: value, ..
        } => visit_value(&value.value.expression, pending, budget)?,
        model::Stage::Paths { value, .. } => {
            visit_match_predicate(&value.paths.target, pending, budget)?;
        }
        model::Stage::Match { value, .. } => {
            for clause in &value.matching.clauses {
                match &clause.value {
                    zirql::MatchClause::From { source, .. }
                    | zirql::MatchClause::Bind { value: source, .. } => {
                        visit_operand(source, pending, budget)?
                    }
                    zirql::MatchClause::Where { predicate } => {
                        visit_match_predicate(predicate, pending, budget)?
                    }
                }
            }
        }
        model::Stage::PatternBinding { value, .. } => {
            visit_value(&value.value.expression, pending, budget)?
        }
        model::Stage::Computed { value, .. } => {
            visit_value(&value.operand().expression, pending, budget)?
        }
        model::Stage::FunctionValue { value, .. } | model::Stage::Comprehension { value, .. } => {
            visit_value(&value.expression, pending, budget)?
        }
        model::Stage::Projection { value, .. } => {
            for (_, expression) in &value.projection.fields {
                visit_value(expression, pending, budget)?
            }
        }
        model::Stage::Filter { predicate, .. } | model::Stage::Root { predicate, .. } => {
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
fn visit_match_predicate<'a>(
    predicate: &'a zirql::Predicate<model::Predicate>,
    pending: &mut Vec<&'a model::Expression>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    match &predicate.value {
        zirql::PredicateKind::Leaf(guard) => visit_predicate(guard, pending, budget)?,
        zirql::PredicateKind::Not(child) | zirql::PredicateKind::Group(child) => {
            visit_match_predicate(child, pending, budget)?
        }
        zirql::PredicateKind::And(children) | zirql::PredicateKind::Or(children) => {
            for child in children {
                visit_match_predicate(child, pending, budget)?;
            }
        }
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
                    if let Step::Stage(stage) = &step.value {
                        let nested = nested(stage);
                        budget.charge_results(nested.len(), Location::Query(step.range.clone()))?;
                        pending.extend(nested);
                        push_expressions(stage, pending, budget).map_err(query_error)?;
                    }
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
    context: &mut EvaluationContext<'_, '_, '_>,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<zirql::Truth, EvaluationError> {
    if let model::Predicate::Expression { value, .. } = predicate {
        let result = context
            .functions
            .run(&value.expression, context, &input, bindings, budget)?;
        let result = functions::constructed(result, budget, &value.expression.range)?;
        let V::Boolean(matched) = result.value().data() else {
            return Err(EvaluationError::new("predicate needs a known Boolean")
                .at(value.expression.range.clone()));
        };
        return Ok(if *matched {
            zirql::Truth::True
        } else {
            zirql::Truth::False
        });
    }
    let resolved = resolve_predicate_by(
        predicate,
        &mut |value, budget| {
            let input = program::copy_value(&input, budget, &value.expression.range)?;
            evaluate(value, context, input, bindings, budget)?
                .into_value(budget, &value.expression.range)
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
    arguments: &mut dyn zirql::NativeArguments<EvaluationContext<'_, '_, '_>, ReportHost<'_>>,
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
        P::Expression { .. } | P::OpValue { .. } | P::Computed { .. } => true,
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
                model::Stage::WhereExists { .. } | model::Stage::Fixpoint { expression: model::FixedPointBody::Native(_), .. } | model::Stage::PatternBinding { .. } | model::Stage::FunctionInput{..} | model::Stage::FunctionValue{..} | model::Stage::Paths { .. } | model::Stage::Match { .. } | model::Stage::Native { .. } | model::Stage::Operand { .. } | model::Stage::Comprehension { .. } | model::Stage::ConstructedSuffix { .. } | model::Stage::Projection { .. } | model::Stage::Computed {..}
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
    let kind =
        functions::value_kind(host, &value.expression, input, names, budget).map_err(core_error)?;
    host.scalar_types(&kind, budget, &value.expression.range)
        .map_err(core_error)?
        .ok_or_else(|| {
            EvaluationError::new("native handles require an explicit scalar projection")
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
    let subject = host
        .subject_kind(&input, budget, &projection.subject.range)
        .map_err(core_error)?;
    scope.insert(projection.subject.value.clone(), subject.clone());
    let mut fields = Vec::new();
    for (name, value) in &projection.fields {
        if fields.iter().any(|(other, _)| other == &name.value) {
            return Err(EvaluationError::new("duplicate projection field").at(name.range.clone()));
        }
        let kind = functions::value_kind(host, value, subject.clone(), &scope, budget)
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
