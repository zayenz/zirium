use super::{
    EvaluationError, charge, check_items, json_size, output_size, parser::JsonLiteral, program,
    render,
};
use crate::semantic::Document;

pub(super) fn evaluate(
    literal: &JsonLiteral,
    document: &Document,
    bindings: &zirql::Bindings<super::QueryOutput>,
    budget: &mut zirql::Budget,
    items: &mut usize,
) -> Result<serde_json::Value, EvaluationError> {
    charge(budget, 1)?;
    *items = items.saturating_add(1);
    check_items(budget, *items)?;
    program::charge_results(budget, 1, &(0..0))?;
    Ok(match literal {
        JsonLiteral::Scalar(value) => value.clone(),
        JsonLiteral::String(parts) => {
            serde_json::Value::String(render::interpolate(parts, bindings, budget)?)
        }
        JsonLiteral::Binding(name) => {
            let binding = bindings.get(name, 0..0).map_err(|_| {
                EvaluationError::new(format!(
                    "binding `{name}` is unavailable in this evaluation context"
                ))
            })?;
            charge(budget, output_size(binding))?;
            let value = program::to_json(document, binding, budget, &(0..0))?;
            *items = items.saturating_add(json_size(&value).saturating_sub(1));
            check_items(budget, *items)?;
            value
        }
        JsonLiteral::Array(elements) => {
            let mut values = Vec::new();
            for element in elements {
                values.push(evaluate(element, document, bindings, budget, items)?);
            }
            serde_json::Value::Array(values)
        }
        JsonLiteral::Object(entries) => {
            let mut values = serde_json::Map::new();
            for (key, value) in entries {
                let key = render::interpolate(key, bindings, budget)?;
                if values.contains_key(&key) {
                    return Err(EvaluationError::new(format!(
                        "duplicate JSON object key `{key}` after interpolation"
                    )));
                }
                *items = items.saturating_add(1);
                check_items(budget, *items)?;
                program::charge_results(budget, 1, &(0..0))?;
                values.insert(key, evaluate(value, document, bindings, budget, items)?);
            }
            serde_json::Value::Object(values)
        }
    })
}
