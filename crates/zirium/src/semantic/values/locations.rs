//! Location parsing and alias-aware lowering.

use super::*;

pub(super) fn parse_location(spelling: &str) -> Option<LocationValue> {
    let inner = spelling
        .trim()
        .strip_prefix("loc(")?
        .strip_suffix(')')?
        .trim();
    if inner == "unknown" {
        return Some(LocationValue::Unknown);
    }
    if let Some(fused) = inner.strip_prefix("fused") {
        let (metadata, values) = if let Some(rest) = fused.strip_prefix('<') {
            let end = matching_delimiter(rest, '<', '>')?;
            (Some(rest[..end].trim().to_owned()), rest[end + 1..].trim())
        } else {
            (None, fused.trim())
        };
        let values = bracket_inner(values, '[', ']')?;
        return Some(LocationValue::Fused {
            metadata,
            locations: split_top_level_commas(values)
                .iter()
                .map(|value| parse_location_detail(value))
                .collect::<Option<Vec<_>>>()?,
        });
    }
    if let Some(callsite) = inner
        .strip_prefix("callsite(")
        .and_then(|value| value.strip_suffix(')'))
    {
        let (callee, caller) = split_at_keyword(callsite, " at ")?;
        return Some(LocationValue::CallSite {
            callee: Box::new(parse_location_detail(callee)?),
            caller: Box::new(parse_location_detail(caller)?),
        });
    }
    parse_location_detail(inner)
}

pub(in crate::semantic) fn lower_location_value(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> LocationValue {
    let invalid = |doc: &mut Document, code: SemanticDiagnosticCode, message: String| {
        LocationValue::Invalid(push_diagnostic(doc, code, range, message))
    };
    let spelling = spelling.trim();
    if spelling.starts_with('#') {
        let alias = spelling.to_owned();
        let Some((target, _)) = attribute_aliases.get(&alias) else {
            let message = if type_aliases.contains_key(&format!("!{}", &alias[1..])) {
                format!("alias `{alias}` has type kind, expected location")
            } else {
                format!("unresolved location alias `{alias}`")
            };
            let code = if type_aliases.contains_key(&format!("!{}", &alias[1..])) {
                SemanticDiagnosticCode::Location
            } else {
                SemanticDiagnosticCode::UnresolvedReference
            };
            return invalid(doc, code, message);
        };
        if let Err(message) = stack.enter(&alias, "location") {
            return invalid(
                doc,
                stack.diagnostic_code(SemanticDiagnosticCode::Location),
                message,
            );
        }
        let wrapped;
        let target = if target.starts_with("loc(") {
            target.as_str()
        } else {
            wrapped = format!("loc({target})");
            wrapped.as_str()
        };
        let result =
            lower_location_value(target, range, type_aliases, attribute_aliases, stack, doc);
        stack.exit(&alias);
        return result;
    }
    let Some(inner) = spelling
        .strip_prefix("loc(")
        .and_then(|value| value.strip_suffix(')'))
        .map(str::trim)
    else {
        return invalid(
            doc,
            SemanticDiagnosticCode::Location,
            "invalid semantic location".into(),
        );
    };
    if inner.starts_with('#') {
        return lower_location_value(inner, range, type_aliases, attribute_aliases, stack, doc);
    }
    if let Some(fused) = inner.strip_prefix("fused") {
        let (metadata, values) = match fused_parts(fused) {
            Some(parts) => parts,
            None => {
                return invalid(
                    doc,
                    SemanticDiagnosticCode::Location,
                    "malformed fused location".into(),
                );
            }
        };
        let locations = split_top_level_commas(values)
            .iter()
            .map(|value| {
                lower_location_detail(value, range, type_aliases, attribute_aliases, stack, doc)
            })
            .collect();
        return LocationValue::Fused {
            metadata,
            locations,
        };
    }
    if let Some(callsite) = inner
        .strip_prefix("callsite(")
        .and_then(|value| value.strip_suffix(')'))
    {
        let Some((callee, caller)) = split_at_keyword(callsite, " at ") else {
            return invalid(
                doc,
                SemanticDiagnosticCode::Location,
                "malformed callsite location".into(),
            );
        };
        return LocationValue::CallSite {
            callee: Box::new(lower_location_detail(
                callee,
                range,
                type_aliases,
                attribute_aliases,
                stack,
                doc,
            )),
            caller: Box::new(lower_location_detail(
                caller,
                range,
                type_aliases,
                attribute_aliases,
                stack,
                doc,
            )),
        };
    }
    lower_location_detail(inner, range, type_aliases, attribute_aliases, stack, doc)
}

fn lower_location_detail(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> LocationValue {
    let spelling = spelling.trim();
    if spelling.starts_with("loc(") || spelling.starts_with('#') {
        lower_location_value(spelling, range, type_aliases, attribute_aliases, stack, doc)
    } else if spelling.starts_with("callsite(") || spelling.starts_with("fused") {
        let wrapped = format!("loc({spelling})");
        lower_location_value(&wrapped, range, type_aliases, attribute_aliases, stack, doc)
    } else if let Some(stripped) = spelling.strip_prefix('"') {
        let Some(quote_end) = stripped.find('"').map(|index| index + 1) else {
            return LocationValue::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::Location,
                range,
                format!("invalid nested location `{spelling}`"),
            ));
        };
        let name = spelling[..=quote_end].to_owned();
        let rest = spelling[quote_end + 1..].trim();
        if rest.starts_with(':') {
            return parse_location_detail(spelling).unwrap_or_else(|| {
                LocationValue::Invalid(push_diagnostic(
                    doc,
                    SemanticDiagnosticCode::Location,
                    range,
                    format!("invalid nested location `{spelling}`"),
                ))
            });
        }
        let child = if rest.starts_with('(') && rest.ends_with(')') {
            Some(Box::new(lower_location_detail(
                &rest[1..rest.len() - 1],
                range,
                type_aliases,
                attribute_aliases,
                stack,
                doc,
            )))
        } else {
            None
        };
        let has_child = child.is_some();
        LocationValue::Name {
            name,
            child,
            metadata: (!rest.is_empty() && !has_child).then(|| compact(rest)),
        }
    } else {
        parse_location_detail(spelling).unwrap_or_else(|| {
            LocationValue::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::Location,
                range,
                format!("invalid nested location `{spelling}`"),
            ))
        })
    }
}

fn fused_parts(value: &str) -> Option<(Option<String>, &str)> {
    let value = value.trim();
    if let Some(rest) = value.strip_prefix('<') {
        let end = matching_delimiter(rest, '<', '>')?;
        let metadata = Some(rest[..end].trim().to_owned());
        let values = rest[end + 1..].trim();
        Some((metadata, bracket_inner(values, '[', ']')?))
    } else {
        Some((None, bracket_inner(value, '[', ']')?))
    }
}

fn parse_location_detail(value: &str) -> Option<LocationValue> {
    let value = value.trim();
    if value == "unknown" {
        return Some(LocationValue::Unknown);
    }
    if let Some(stripped) = value.strip_prefix('"') {
        let quote_end = stripped.find('"')? + 1;
        let name = value[..=quote_end].to_owned();
        let rest = value[quote_end + 1..].trim();
        if rest.starts_with(':') {
            return parse_file_line_column(value);
        }
        let child = if rest.starts_with('(') && rest.ends_with(')') {
            Some(Box::new(parse_location_detail(&rest[1..rest.len() - 1])?))
        } else {
            None
        };
        let has_child = child.is_some();
        return Some(LocationValue::Name {
            name,
            child,
            metadata: (!rest.is_empty() && !has_child).then(|| compact(rest)),
        });
    }
    parse_file_line_column(value)
}

fn parse_file_line_column(value: &str) -> Option<LocationValue> {
    let mut parts = value.rsplitn(3, ':');
    let column = parts.next()?.trim().parse().ok()?;
    let line = parts.next()?.trim().parse().ok()?;
    Some(LocationValue::FileLineColumn {
        file: parts.next()?.trim().to_owned(),
        line,
        column,
    })
}

fn split_at_keyword<'a>(value: &'a str, keyword: &str) -> Option<(&'a str, &'a str)> {
    let index = value.find(keyword)?;
    Some((&value[..index], &value[index + keyword.len()..]))
}
