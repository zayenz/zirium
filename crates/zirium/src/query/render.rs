use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Write;
use std::ops::Range;

use super::{EvaluationError, QueryOutput, charge, check_items, parser::PrintPart};

pub(super) fn interpolate(
    parts: &[PrintPart],
    bindings: &zirql::Bindings<QueryOutput>,
    state: &mut zirql::Budget,
    range: &Range<usize>,
) -> Result<String, EvaluationError> {
    zirql::expand_fragments(
        parts,
        &mut (),
        state,
        zirql::Location::Query(range.clone()),
        |part, _, budget| {
            Ok(match part {
                PrintPart::Literal(value) => Cow::Borrowed(value.as_str()),
                PrintPart::Binding(name) => match bindings.get(name, range.clone()).map_err(|_| {
                    zirql::EvaluationError::Host(EvaluationError::new(format!(
                        "binding `{name}` is unavailable in this evaluation context"
                    )).at(range.clone()))
                })? {
                    QueryOutput::Count(count) => {
                        let bytes = usize::try_from(count.checked_ilog10().unwrap_or(0))
                            .expect("usize represents the number of decimal digits") + 1;
                        budget.charge(bytes, zirql::Location::Query(range.clone()))?;
                        budget.charge_results(bytes, zirql::Location::Query(range.clone()))?;
                        Cow::Owned(count.to_string())
                    }
                    QueryOutput::Values(values) if values.len() == 1 => {
                        Cow::Borrowed(values[0].as_str())
                    }
                    QueryOutput::Array(values) if values.len() == 1 && values[0].is_string() => {
                        Cow::Borrowed(values[0].as_str().expect("one JSON string"))
                    }
                    _ => {
                        return Err(zirql::EvaluationError::Host(EvaluationError::new(format!(
                            "string interpolation `{{{name}}}` requires a count or exactly one string; project or count the binding first"
                        )).at(range.clone())));
                    }
                },
            })
        },
    ).map_err(super::program::evaluation_error)
}

pub(super) fn markdown(
    output: &QueryOutput,
    state: &mut zirql::Budget,
) -> Result<String, EvaluationError> {
    if let Some(entries) = map_entries(output) {
        return markdown_map(&entries, state);
    }
    match output {
        QueryOutput::Array(values) => {
            let values = values.iter().map(scalar).collect::<Result<Vec<_>, _>>()
                .map_err(|_| EvaluationError::new("markdown arrays require scalar elements; use json for nested arrays or objects"))?;
            markdown(&QueryOutput::Values(values), state)
        }
        QueryOutput::Count(count) => Ok(format!("\n{count}\n\n")),
        QueryOutput::Values(values) => {
            table_budget(values.len(), 1, state)?;
            let mut text = header(&["Value"], state)?;
            for value in values {
                row(&mut text, [value.as_str()], state)?;
            }
            Ok(finish(text, values.is_empty()))
        }
        _ => Err(EvaluationError::new(
            "markdown requires values, a count, or a map of scalars or maps of scalars; use names, tally, or map_by to shape operations first",
        )),
    }
}

fn map_entries(output: &QueryOutput) -> Option<Vec<(&String, &serde_json::Value)>> {
    match output {
        QueryOutput::Map(entries) => {
            Some(entries.iter().map(|(key, value)| (key, value)).collect())
        }
        _ => None,
    }
}

fn markdown_map(
    entries: &[(&String, &serde_json::Value)],
    state: &mut zirql::Budget,
) -> Result<String, EvaluationError> {
    if entries.is_empty() {
        return Ok(finish(String::new(), true));
    }
    if entries.iter().all(|(_, value)| value.is_object()) {
        let mut columns = Vec::new();
        let mut seen = HashSet::new();
        for &(key, value) in entries {
            for (column, value) in value.as_object().unwrap() {
                if value.is_array() || value.is_object() {
                    return Err(EvaluationError::new(format!(
                        "markdown cell at row `{key}`, column `{column}` is nested; expected a scalar, use json for deeper data"
                    )));
                }
                if seen.insert(column.as_str()) {
                    columns.push(column.as_str());
                    // Bound the union before expanding sparse rows into a rectangle.
                    table_budget(entries.len(), columns.len() + 1, state)?;
                }
            }
        }
        let mut headers = vec!["Key"];
        headers.extend(&columns);
        table_budget(entries.len(), headers.len(), state)?;
        let mut text = header(&headers, state)?;
        for &(key, value) in entries {
            let values = value.as_object().unwrap();
            let mut cells = vec![key.clone()];
            for column in &columns {
                cells.push(
                    values
                        .get(*column)
                        .map(scalar)
                        .transpose()?
                        .unwrap_or_default(),
                );
            }
            row(&mut text, cells.iter().map(String::as_str), state)?;
        }
        return Ok(finish(text, false));
    }
    table_budget(entries.len(), 2, state)?;
    let mut text = header(&["Key", "Value"], state)?;
    for &(key, value) in entries {
        let value = scalar(value).map_err(|_| EvaluationError::new(format!(
                    "markdown entry `{key}` has an incompatible shape; expected all scalar values or all maps of scalars, use json for mixed or deeper data"
                )))?;
        row(&mut text, [key.as_str(), value.as_str()], state)?;
    }
    Ok(finish(text, false))
}

fn scalar(value: &serde_json::Value) -> Result<String, EvaluationError> {
    match value {
        serde_json::Value::String(value) => Ok(value.clone()),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => Ok(value.to_string()),
        serde_json::Value::Null => Ok(String::new()),
        _ => Err(EvaluationError::new("expected a scalar Markdown cell")),
    }
}

fn table_budget(rows: usize, columns: usize, state: &zirql::Budget) -> Result<(), EvaluationError> {
    check_items(state, rows.saturating_add(1).saturating_mul(columns))
}

fn header(columns: &[&str], state: &mut zirql::Budget) -> Result<String, EvaluationError> {
    super::program::charge_bytes(state, 16, &(0..0))?;
    for column in columns {
        charge(state, column.len().max(1))?;
        super::program::charge_results(state, 1, &(0..0))?;
        super::program::charge_bytes(
            state,
            column.len().saturating_mul(6).saturating_add(16),
            &(0..0),
        )?;
    }
    let mut text = String::from("\n| ");
    text.push_str(
        &columns
            .iter()
            .map(|value| escape(value))
            .collect::<Vec<_>>()
            .join(" | "),
    );
    text.push_str(" |\n| ");
    text.push_str(&vec!["---"; columns.len()].join(" | "));
    text.push_str(" |\n");
    Ok(text)
}

fn row<'a>(
    text: &mut String,
    cells: impl IntoIterator<Item = &'a str>,
    state: &mut zirql::Budget,
) -> Result<(), EvaluationError> {
    super::program::charge_bytes(state, 2, &(0..0))?;
    text.push('|');
    for cell in cells {
        charge(state, cell.len().max(1))?;
        super::program::charge_results(state, 1, &(0..0))?;
        super::program::charge_bytes(
            state,
            cell.len().saturating_mul(6).saturating_add(4),
            &(0..0),
        )?;
        write!(text, " {} |", escape(cell)).expect("writing to a string");
    }
    text.push('\n');
    Ok(())
}

fn finish(mut text: String, empty: bool) -> String {
    if empty {
        "\n_No entries._\n\n".to_owned()
    } else {
        text.push('\n');
        text
    }
}

fn escape(value: &str) -> String {
    let mut text = String::new();
    for ch in value.replace("\r\n", "\n").chars() {
        match ch {
            '&' => text.push_str("&amp;"),
            '<' => text.push_str("&lt;"),
            '>' => text.push_str("&gt;"),
            '\n' | '\r' => text.push_str("<br>"),
            '\\' | '|' | '`' | '*' | '_' | '[' | ']' | '~' => {
                text.push('\\');
                text.push(ch);
            }
            _ => text.push(ch),
        }
    }
    text
}
