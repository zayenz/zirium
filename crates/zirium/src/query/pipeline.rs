//! Native stream segments adapted to Zirql's collection and predicate execution.

use std::ops::Range;

use super::{
    AttributeId, AttributeValue, Document, EvaluationContext, EvaluationError, NativeValue,
    OperationId, OutputMode, QueryOutput, TypeId, core_error, decode_mlir_string,
    evaluate_host_stage, model,
};
use zirql::{Budget, Host, Located, NativeInput, Pipeline, PredicateKind, Stage, Truth};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Row {
    Operation(OperationId),
    String(String),
    Type(TypeId),
    Attribute(String, AttributeId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    Operations,
    Strings,
    Types,
    Attributes,
}

fn stream_kind(output: &QueryOutput) -> Option<StreamKind> {
    match output {
        QueryOutput::Operations(_) => Some(StreamKind::Operations),
        QueryOutput::Values(_) => Some(StreamKind::Strings),
        QueryOutput::Native(NativeValue::Types(_)) => Some(StreamKind::Types),
        QueryOutput::Native(NativeValue::Attributes(_)) => Some(StreamKind::Attributes),
        _ => None,
    }
}

fn into_rows(output: QueryOutput) -> (StreamKind, Vec<Row>) {
    match output {
        QueryOutput::Operations(items) => (
            StreamKind::Operations,
            items.into_iter().map(Row::Operation).collect(),
        ),
        QueryOutput::Values(items) => (
            StreamKind::Strings,
            items.into_iter().map(Row::String).collect(),
        ),
        QueryOutput::Native(NativeValue::Types(items)) => (
            StreamKind::Types,
            items.into_iter().map(Row::Type).collect(),
        ),
        QueryOutput::Native(NativeValue::Attributes(items)) => (
            StreamKind::Attributes,
            items
                .into_iter()
                .map(|(name, id)| Row::Attribute(name, id))
                .collect(),
        ),
        _ => unreachable!("only native streams enter a core segment"),
    }
}

fn into_output(kind: StreamKind, rows: Vec<Row>) -> QueryOutput {
    match kind {
        StreamKind::Operations => QueryOutput::Operations(
            rows.into_iter()
                .map(|row| {
                    let Row::Operation(id) = row else {
                        unreachable!("host stream kind matches its rows")
                    };
                    id
                })
                .collect(),
        ),
        StreamKind::Strings => QueryOutput::Values(
            rows.into_iter()
                .map(|row| {
                    let Row::String(value) = row else {
                        unreachable!("host stream kind matches its rows")
                    };
                    value
                })
                .collect(),
        ),
        StreamKind::Types => QueryOutput::Native(NativeValue::Types(
            rows.into_iter()
                .map(|row| {
                    let Row::Type(id) = row else {
                        unreachable!("host stream kind matches its rows")
                    };
                    id
                })
                .collect(),
        )),
        StreamKind::Attributes => QueryOutput::Native(NativeValue::Attributes(
            rows.into_iter()
                .map(|row| {
                    let Row::Attribute(name, id) = row else {
                        unreachable!("host stream kind matches its rows")
                    };
                    (name, id)
                })
                .collect(),
        )),
    }
}

pub(super) fn is_stream_stage(stage: &model::Stage) -> bool {
    use model::{Stage as S, StructuredStage as N};
    matches!(
        stage,
        S::Filter { .. }
            | S::Head { .. }
            | S::Count { .. }
            | S::Unique { .. }
            | S::Tally { .. }
            | S::Input { .. }
            | S::Reachable { .. }
            | S::Closure { .. }
            | S::Slice { .. }
            | S::ForwardSlice { .. }
            | S::Defs { .. }
            | S::Users { .. }
            | S::Parent { .. }
            | S::Children { .. }
            | S::Subtree { .. }
            | S::Sort { .. }
            | S::Reverse { .. }
            | S::Tail { .. }
            | S::Min { .. }
            | S::Max { .. }
            | S::MinAll { .. }
            | S::MaxAll { .. }
            | S::Attr { .. }
            | S::Names { .. }
            | S::ResultTypes { .. }
            | S::OperandTypes { .. }
            | S::Structured {
                step: N::StringAttr(_) | N::Attributes(_) | N::Types { .. } | N::Spellings,
                ..
            }
    )
}

pub(super) fn is_stream(output: &QueryOutput) -> bool {
    stream_kind(output).is_some()
}

fn range(range: crate::source::TextRange) -> Range<usize> {
    range.start() as usize..range.end() as usize
}

fn lower_predicate(predicate: &model::Predicate) -> zirql::Predicate<&model::Predicate> {
    use model::Predicate as P;
    let value = match predicate {
        P::Not { predicate, .. } => PredicateKind::Not(Box::new(lower_predicate(predicate))),
        P::Group { predicate, .. } => PredicateKind::Group(Box::new(lower_predicate(predicate))),
        P::And { predicates, .. } => {
            PredicateKind::And(predicates.iter().map(lower_predicate).collect())
        }
        P::Or { predicates, .. } => {
            PredicateKind::Or(predicates.iter().map(lower_predicate).collect())
        }
        P::SameValue { .. }
        | P::Bool { .. }
        | P::Op { .. }
        | P::OpValue { .. }
        | P::Computed { .. }
        | P::Dialect { .. }
        | P::ResultType { .. }
        | P::HasAttr { .. }
        | P::Attr { .. } => PredicateKind::Leaf(predicate),
    };
    Located::new(value, range(predicate.range()))
}

fn evaluation_error(error: zirql::EvaluationError<EvaluationError>) -> EvaluationError {
    match error {
        zirql::EvaluationError::Core(error) => core_error(error),
        zirql::EvaluationError::Host(error) => error,
    }
}

pub(super) type NativePipeline<'q> = Pipeline<&'q model::Predicate, &'q model::Stage>;

pub(super) fn lower(stages: &[model::Stage]) -> Result<NativePipeline<'_>, EvaluationError> {
    let stages = stages
        .iter()
        .map(|stage| {
            let value = match stage {
                model::Stage::Filter { predicate, .. } => Stage::Filter(lower_predicate(predicate)),
                model::Stage::Head { count, .. } => Stage::Head(*count),
                model::Stage::Unique { .. } => Stage::Unique,
                model::Stage::Count { .. } => Stage::Count,
                model::Stage::Tally { .. } => Stage::Tally,
                _ => Stage::Host(stage),
            };
            Located::new(value, range(stage.range()))
        })
        .collect::<Vec<_>>();
    let location = stages
        .first()
        .expect("a core segment has at least one stage")
        .range
        .start..stages.last().unwrap().range.end;
    Pipeline::with_group_not_nesting(stages, location, zirql::MAX_NESTING).map_err(core_error)
}

pub(super) fn evaluate(
    pipeline: &NativePipeline<'_>,
    context: &mut EvaluationContext<'_, '_, '_>,
    input: QueryOutput,
    bindings: &zirql::Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<QueryOutput, EvaluationError> {
    let location = pipeline.range();
    let (kind, items) = into_rows(input);
    let input = NativeInput::from_admitted(
        StreamContext {
            evaluation: context,
            kind,
            bindings,
        },
        items,
        budget,
        zirql::Location::Query(location),
    )
    .map_err(core_error)?;
    let result = zirql::evaluate(pipeline, &StreamHost, input, budget).map_err(evaluation_error)?;
    Ok(match result.value {
        zirql::NativeValue::Stream(rows) => into_output(result.context.kind, rows),
        zirql::NativeValue::Count(count) => QueryOutput::Count(count),
        zirql::NativeValue::Tally(counts) => match result.context.evaluation.output {
            OutputMode::Native => QueryOutput::Native(NativeValue::Map(
                counts
                    .into_iter()
                    .map(|(key, count)| (key, QueryOutput::Count(count)))
                    .collect(),
            )),
            OutputMode::Textual => QueryOutput::Map(
                counts
                    .into_iter()
                    .map(|(key, count)| (key, serde_json::json!(count)))
                    .collect(),
            ),
        },
    })
}

struct StreamContext<'a, 'host, 'document, 'query> {
    evaluation: &'a mut EvaluationContext<'host, 'document, 'query>,
    kind: StreamKind,
    bindings: &'a zirql::Bindings<QueryOutput>,
}

struct StreamHost;

impl<'stage>
    Host<Row, StreamContext<'_, '_, '_, '_>, &'stage model::Predicate, &'stage model::Stage>
    for StreamHost
{
    type Error = EvaluationError;
    type Key = Row;

    fn stage_work(&self, items: &[Row], _: &StreamContext<'_, '_, '_, '_>) -> usize {
        items.len().max(1)
    }

    fn collection_item_work(&self) -> usize {
        0
    }

    fn prepare_stage(
        &self,
        stage: &Located<Stage<&'stage model::Predicate, &'stage model::Stage>>,
        context: &mut StreamContext<'_, '_, '_, '_>,
        _: &mut Budget,
    ) -> Result<(), zirql::EvaluationError<Self::Error>> {
        match stage.value {
            Stage::Filter(_) if context.kind != StreamKind::Operations => {
                Err(zirql::EvaluationError::Host(EvaluationError::new(
                    "filter requires operations, but the current stream contains values",
                )))
            }
            Stage::Tally if context.kind != StreamKind::Strings => {
                Err(zirql::EvaluationError::Host(EvaluationError::new(
                    "tally requires a value stream; use names or attr first",
                )))
            }
            _ => Ok(()),
        }
    }

    fn prepare_filter_item(
        &self,
        _: &Row,
        _: &mut StreamContext<'_, '_, '_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<Self::Error>> {
        Ok(())
    }

    fn evaluate_leaf(
        &self,
        leaf: &&'stage model::Predicate,
        item: &Row,
        context: &mut StreamContext<'_, '_, '_, '_>,
        budget: &mut Budget,
        _: &Range<usize>,
    ) -> Result<Truth, zirql::EvaluationError<Self::Error>> {
        let Row::Operation(operation) = item else {
            unreachable!("filter preparation requires operations")
        };
        if let model::Predicate::OpValue { name, .. } = leaf {
            let name = super::report::op_name(name, context.evaluation, context.bindings, budget)
                .map_err(zirql::EvaluationError::Host)?;
            return Ok(
                if context.evaluation.document.operation_name(*operation) == Some(&name) {
                    Truth::True
                } else {
                    Truth::False
                },
            );
        }
        Ok(evaluate_leaf(leaf, context.evaluation.document, *operation))
    }

    fn evaluate_stage(
        &self,
        stage: &&'stage model::Stage,
        items: Vec<Row>,
        context: &mut StreamContext<'_, '_, '_, '_>,
        budget: &mut Budget,
        _: &Range<usize>,
    ) -> Result<Vec<Row>, zirql::EvaluationError<Self::Error>> {
        let output = evaluate_host_stage(
            stage,
            context.evaluation,
            into_output(context.kind, items),
            context.bindings,
            budget,
        )
        .map_err(zirql::EvaluationError::Host)?;
        let (kind, rows) = into_rows(output);
        context.kind = kind;
        Ok(rows)
    }

    fn identity_key(
        &self,
        item: &Row,
        _: &mut StreamContext<'_, '_, '_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<Row, zirql::EvaluationError<Self::Error>> {
        Ok(item.clone())
    }

    fn tally_key(
        &self,
        item: Row,
        _: &mut StreamContext<'_, '_, '_, '_>,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<String, zirql::EvaluationError<Self::Error>> {
        let Row::String(value) = item else {
            unreachable!("tally preparation requires strings")
        };
        Ok(value)
    }
}

// Only domain leaves live here. Zirql owns every Boolean wrapper and short circuit.
fn evaluate_leaf(
    predicate: &model::Predicate,
    document: &Document,
    operation: OperationId,
) -> Truth {
    let value = match predicate {
        model::Predicate::SameValue { .. } => unreachable!("preflight refuses same_value outside native matches"),
        model::Predicate::Bool { value, .. } => *value,
        model::Predicate::Op { name, .. } => document.operation_name(operation) == Some(name),
        model::Predicate::OpValue {..} | model::Predicate::Computed {..}=>unreachable!("value arguments resolve with their active scope"),
        model::Predicate::Dialect { name, .. } => document.operation_name(operation).and_then(|op| op.split_once('.')).is_some_and(|(dialect, _)| dialect == name),
        model::Predicate::ResultType { spelling, .. } => document.result_types(operation).unwrap_or(&[]).iter().any(|&ty| document.type_spelling(ty) == Some(spelling)),
        model::Predicate::HasAttr { name, .. } => document.attribute_entries(operation).is_some_and(|mut entries| entries.any(|(attribute, _)| attribute == name)),
        model::Predicate::Attr { name, value, .. } => document.attribute_entries(operation).and_then(|mut entries| entries.find(|(attribute, _)| attribute == name)).and_then(|(_, id)| document.attribute_value(id)).is_some_and(|attribute| matches!(attribute, AttributeValue::String(spelling) if decode_mlir_string(spelling).as_deref() == Some(value))),
        model::Predicate::Not { .. } | model::Predicate::And { .. } | model::Predicate::Or { .. } | model::Predicate::Group { .. } => unreachable!("lowering retains only domain leaves"),
    };
    if value { Truth::True } else { Truth::False }
}

pub(super) fn root_filter(
    predicate: &model::Predicate,
) -> Result<Pipeline<&model::Predicate>, EvaluationError> {
    let range = range(predicate.range());
    Pipeline::with_group_not_nesting(
        vec![Located::new(
            Stage::Filter(lower_predicate(predicate)),
            range.clone(),
        )],
        range,
        zirql::MAX_NESTING,
    )
    .map_err(core_error)
}

pub(super) fn matches_root(
    filter: &Pipeline<&model::Predicate>,
    document: &Document,
    operation: OperationId,
    budget: &mut Budget,
) -> Result<bool, EvaluationError> {
    let input = NativeInput::from_admitted(
        document,
        vec![operation],
        budget,
        zirql::Location::Query(filter.range()),
    )
    .map_err(core_error)?;
    let result = zirql::evaluate(filter, &RootHost, input, budget).map_err(evaluation_error)?;
    let zirql::NativeValue::Stream(rows) = result.value else {
        unreachable!("the synthetic root pipeline contains only a filter")
    };
    Ok(!rows.is_empty())
}

struct RootHost;

impl Host<OperationId, &Document, &model::Predicate> for RootHost {
    type Error = EvaluationError;
    type Key = OperationId;

    fn stage_work(&self, _: &[OperationId], _: &&Document) -> usize {
        0
    }
    fn prepare_filter_item(
        &self,
        _: &OperationId,
        _: &mut &Document,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<(), zirql::EvaluationError<Self::Error>> {
        Ok(())
    }
    fn evaluate_leaf(
        &self,
        leaf: &&model::Predicate,
        item: &OperationId,
        document: &mut &Document,
        _: &mut Budget,
        _: &Range<usize>,
    ) -> Result<Truth, zirql::EvaluationError<Self::Error>> {
        Ok(evaluate_leaf(leaf, document, *item))
    }
}
