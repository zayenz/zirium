use super::{
    EvaluationError, EvaluationState, json_size, output_size, output_value, parser::JsonLiteral,
    render,
};
use crate::semantic::Document;

pub(super) fn evaluate(
    literal: &JsonLiteral,
    document: &Document,
    state: &mut EvaluationState,
    items: &mut usize,
) -> Result<serde_json::Value, EvaluationError> {
    state.charge(1)?;
    *items = items.saturating_add(1);
    state.check_items(*items)?;
    Ok(match literal {
        JsonLiteral::Scalar(value) => value.clone(),
        JsonLiteral::String(parts) => serde_json::Value::String(render::interpolate(parts, state)?),
        JsonLiteral::Binding(name) => {
            let binding = state.bindings.get(name).cloned().ok_or_else(|| {
                EvaluationError::new(format!(
                    "binding `{name}` is unavailable in this evaluation context"
                ))
            })?;
            state.charge(output_size(&binding))?;
            let value = output_value(document, &binding);
            *items = items.saturating_add(json_size(&value).saturating_sub(1));
            state.check_items(*items)?;
            value
        }
        JsonLiteral::Array(elements) => {
            let mut values = Vec::new();
            for element in elements {
                values.push(evaluate(element, document, state, items)?);
            }
            serde_json::Value::Array(values)
        }
        JsonLiteral::Object(entries) => {
            let mut values = serde_json::Map::new();
            for (key, value) in entries {
                let key = render::interpolate(key, state)?;
                if values.contains_key(&key) {
                    return Err(EvaluationError::new(format!(
                        "duplicate JSON object key `{key}` after interpolation"
                    )));
                }
                *items = items.saturating_add(1);
                state.check_items(*items)?;
                values.insert(key, evaluate(value, document, state, items)?);
            }
            serde_json::Value::Object(values)
        }
    })
}
