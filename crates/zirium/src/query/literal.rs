use std::ops::Range;

use super::{EvaluationError, json_size, output_size, parser::JsonLiteral, program, render};
use crate::semantic::Document;

pub(super) fn evaluate(
    literal: &JsonLiteral,
    document: &Document,
    bindings: &zirql::Bindings<super::QueryOutput>,
    budget: &mut zirql::Budget,
    items: &mut usize,
    range: &Range<usize>,
) -> Result<serde_json::Value, EvaluationError> {
    let location = zirql::Location::Query(range.clone());
    budget
        .charge(1, location.clone())
        .map_err(super::core_error)?;
    *items = items.saturating_add(1);
    budget
        .check_items(*items, location.clone())
        .map_err(super::core_error)?;
    Ok(match literal {
        JsonLiteral::Scalar(value) => {
            program::charge_results(budget, 1, range)?;
            value.clone()
        }
        JsonLiteral::String(parts) => {
            serde_json::Value::String(render::interpolate(parts, bindings, budget, range)?)
        }
        JsonLiteral::Binding(name) => {
            let binding = bindings.get(name, range.clone()).map_err(|_| {
                EvaluationError::new(format!(
                    "binding `{name}` is unavailable in this evaluation context"
                ))
                .at(range.clone())
            })?;
            budget
                .charge(output_size(binding), location.clone())
                .map_err(super::core_error)?;
            let value = program::to_json(document, binding, budget, range)?;
            *items = items.saturating_add(json_size(&value).saturating_sub(1));
            budget
                .check_items(*items, location)
                .map_err(super::core_error)?;
            value
        }
        JsonLiteral::Array(elements) => budget.within_evaluation(1, location, |budget| {
            super::capture::slots::<serde_json::Value>(elements.len(), budget, range)?;
            budget
                .charge(elements.len(), zirql::Location::Query(range.clone()))
                .map_err(super::core_error)?;
            program::charge_results(budget, elements.len() + 1, range)?;
            let mut values = Vec::with_capacity(elements.len());
            for element in elements {
                values.push(evaluate(element, document, bindings, budget, items, range)?);
            }
            Ok::<_, EvaluationError>(serde_json::Value::Array(values))
        })?,
        JsonLiteral::Object(entries) => {
            let values = zirql::construct_string_entries(
                entries,
                &mut (document, bindings, items),
                budget,
                location.clone(),
                |(key, _), context, budget| {
                    render::interpolate(key, context.1, budget, range)
                        .map_err(zirql::EvaluationError::Host)
                },
                |(_, value), context, budget| {
                    *context.2 = context.2.saturating_add(1);
                    budget.check_items(*context.2, location.clone())?;
                    evaluate(value, context.0, context.1, budget, context.2, range)
                        .map_err(zirql::EvaluationError::Host)
                },
                |key, _, budget| {
                    let bytes = key
                        .len()
                        .checked_add("duplicate JSON object key `` after interpolation".len())
                        .ok_or_else(|| {
                            zirql::QueryError::query(
                                range.clone(),
                                "JSON object diagnostic size overflow",
                            )
                        })?;
                    budget.charge(bytes, location.clone())?;
                    budget.charge_results(bytes, location.clone())?;
                    Ok(zirql::QueryError::query(
                        range.clone(),
                        format!("duplicate JSON object key `{key}` after interpolation"),
                    ))
                },
            )
            .map_err(program::evaluation_error)?;
            // serde_json owns a second map container; move admitted keys/values.
            super::capture::slots::<(String, serde_json::Value)>(values.len(), budget, range)?;
            budget
                .charge(values.len(), location)
                .map_err(super::core_error)?;
            program::charge_results(budget, values.len() + 1, range)?;
            serde_json::Value::Object(values.into_iter().collect())
        }
    })
}
