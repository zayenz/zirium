//! Type and memref lowering, including alias resolution.

use super::*;

// These parameters keep type interning explicit at the lowering boundary instead of
// hiding mutable lowering state in a broader context object.
#[allow(clippy::too_many_arguments)]
pub(in crate::semantic) fn intern_type(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    types: &mut Interner<TypeValue>,
    spellings: &mut Vec<String>,
    generation: u128,
    doc: &mut Document,
) -> TypeId {
    let value = lower_type_value(spelling, range, type_aliases, attribute_aliases, doc);
    let index = types.intern_value(value);
    if index as usize == spellings.len() {
        spellings.push(spelling.trim().to_owned());
    }
    TypeId::new(index as usize, generation)
}

fn lower_type_value(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    doc: &mut Document,
) -> TypeValue {
    let mut expansion = AliasExpansionState::new(doc.alias_expansion_depth_limit);
    lower_type_value_with_stack(
        spelling,
        range,
        type_aliases,
        attribute_aliases,
        &mut expansion,
        doc,
    )
}

pub(super) fn lower_type_value_with_stack(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    alias_stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> TypeValue {
    let spelling = spelling.trim();
    if let Some((target, _)) = type_aliases.get(spelling)
        && target != spelling
    {
        if let Err(message) = alias_stack.enter(spelling, "type") {
            return TypeValue::Invalid(push_diagnostic(
                doc,
                alias_stack.diagnostic_code(SemanticDiagnosticCode::Type),
                range,
                message,
            ));
        }
        let value = lower_type_value_with_stack(
            target,
            range,
            type_aliases,
            attribute_aliases,
            alias_stack,
            doc,
        );
        alias_stack.exit(spelling);
        return value;
    }
    if !is_composite_type(spelling)
        && let Ok(value) = resolve_type(spelling, type_aliases, attribute_aliases, alias_stack)
    {
        return value;
    }
    if let Some((inputs, results)) = split_arrow(spelling) {
        return TypeValue::Function {
            inputs: split_types(inputs)
                .iter()
                .map(|value| {
                    lower_type_value_with_stack(
                        value,
                        range,
                        type_aliases,
                        attribute_aliases,
                        alias_stack,
                        doc,
                    )
                })
                .collect(),
            results: split_types(results)
                .iter()
                .map(|value| {
                    lower_type_value_with_stack(
                        value,
                        range,
                        type_aliases,
                        attribute_aliases,
                        alias_stack,
                        doc,
                    )
                })
                .collect(),
        };
    }
    if let Some(inner) = angle_inner(spelling, "complex") {
        let element = lower_type_value_with_stack(
            inner,
            range,
            type_aliases,
            attribute_aliases,
            alias_stack,
            doc,
        );
        if matches!(element, TypeValue::Integer { .. } | TypeValue::Float(_)) {
            return TypeValue::Complex(Box::new(element));
        }
        return TypeValue::Invalid(push_diagnostic(
            doc,
            SemanticDiagnosticCode::Type,
            range,
            "invalid element type for complex".into(),
        ));
    }
    if let Some(inner) = angle_inner(spelling, "tuple") {
        return TypeValue::Tuple(
            split_types(inner)
                .iter()
                .map(|value| {
                    lower_type_value_with_stack(
                        value,
                        range,
                        type_aliases,
                        attribute_aliases,
                        alias_stack,
                        doc,
                    )
                })
                .collect(),
        );
    }
    for (prefix, constructor) in [("tensor", 0u8), ("vector", 1), ("memref", 2)] {
        if let Some(inner) = angle_inner(spelling, prefix) {
            let parts = split_top_level_commas(inner);
            let shape = parts.first().copied().unwrap_or("");
            let shape_parts = split_top_level_x(shape);
            if let Some(element) = shape_parts.last() {
                let dimensions = shape_parts[..shape_parts.len().saturating_sub(1)]
                    .iter()
                    .map(|dimension| {
                        let scalable = prefix == "vector" && dimension.starts_with('[');
                        let (size, invalid) = match *dimension {
                            "?" | "*" => (None, None),
                            _ => match dimension.trim_matches(&['[', ']'][..]).parse() {
                                Ok(size) => (Some(size), None),
                                Err(_) => (
                                    None,
                                    Some(push_diagnostic(
                                        doc,
                                        SemanticDiagnosticCode::Type,
                                        range,
                                        format!("invalid {prefix} dimension `{dimension}`"),
                                    )),
                                ),
                            },
                        };
                        ShapedDimension {
                            size,
                            scalable,
                            invalid,
                        }
                    })
                    .collect::<Vec<_>>();
                let element = Box::new(lower_type_value_with_stack(
                    element,
                    range,
                    type_aliases,
                    attribute_aliases,
                    alias_stack,
                    doc,
                ));
                return match constructor {
                    0 => TypeValue::Tensor {
                        dimensions,
                        element,
                        encoding: parts.get(1).map(|value| {
                            Box::new(lower_attribute_value(
                                value,
                                range,
                                type_aliases,
                                attribute_aliases,
                                alias_stack,
                                doc,
                            ))
                        }),
                        unranked: shape_parts.first().copied() == Some("*"),
                    },
                    1 => TypeValue::Vector {
                        scalable: shape_parts[..shape_parts.len().saturating_sub(1)]
                            .iter()
                            .map(|dimension| dimension.starts_with('['))
                            .collect(),
                        dimensions,
                        element,
                    },
                    _ => TypeValue::MemRef {
                        dimensions,
                        element,
                        layout: parts.get(1).map(|value| {
                            lower_memref_layout(
                                value,
                                range,
                                type_aliases,
                                attribute_aliases,
                                alias_stack,
                                doc,
                            )
                        }),
                        memory_space: parts.get(2).map(|value| {
                            Box::new(lower_memref_memory_space(
                                value,
                                range,
                                type_aliases,
                                attribute_aliases,
                                alias_stack,
                                doc,
                            ))
                        }),
                    },
                };
            }
        }
    }
    let message = match resolve_type(spelling, type_aliases, attribute_aliases, alias_stack) {
        Err(message) => message,
        Ok(_) => format!("unsupported or malformed type `{spelling}`"),
    };
    TypeValue::Invalid(push_diagnostic(
        doc,
        SemanticDiagnosticCode::Type,
        range,
        message,
    ))
}

fn lower_memref_layout(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
    doc: &mut Document,
) -> MemRefLayout {
    if spelling.trim().starts_with("affine_map<") {
        return match lower_affine_attribute(spelling.trim(), range, doc) {
            AttributeValue::AffineMap(map) => MemRefLayout::AffineMap(map),
            AttributeValue::Invalid(diagnostic) => MemRefLayout::Invalid(diagnostic),
            _ => MemRefLayout::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::Type,
                range,
                "memref affine layout has wrong kind".into(),
            )),
        };
    }
    if let Some(affine_spelling) =
        match resolve_affine_alias(spelling.trim(), attribute_aliases, expansion) {
            Ok(value) => value,
            Err(message) => {
                return MemRefLayout::Invalid(push_diagnostic(
                    doc,
                    expansion.diagnostic_code(SemanticDiagnosticCode::Type),
                    range,
                    message,
                ));
            }
        }
    {
        return match lower_affine_attribute(affine_spelling, range, doc) {
            AttributeValue::AffineMap(map) => MemRefLayout::AffineMap(map),
            AttributeValue::Invalid(diagnostic) => MemRefLayout::Invalid(diagnostic),
            AttributeValue::IntegerSet(_) => MemRefLayout::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::Type,
                range,
                "integer set has wrong kind for memref affine layout".into(),
            )),
            AttributeValue::Opaque(_)
            | AttributeValue::Large(_)
            | AttributeValue::WideNumber(_)
            | AttributeValue::Type(_)
            | AttributeValue::Boolean(_)
            | AttributeValue::Integer(_)
            | AttributeValue::Float(_)
            | AttributeValue::String(_)
            | AttributeValue::Symbol(_)
            | AttributeValue::Array(_)
            | AttributeValue::DenseArray { .. }
            | AttributeValue::Dictionary(_)
            | AttributeValue::Location(_) => MemRefLayout::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::Type,
                range,
                "affine alias has wrong kind for memref affine layout".into(),
            )),
        };
    }
    match resolve_memref_layout(spelling, type_aliases, attribute_aliases, expansion) {
        Ok(MemRefLayout::Opaque { spelling, .. }) => {
            let parameters = lower_memref_alias_parameters(
                &spelling,
                range,
                type_aliases,
                attribute_aliases,
                expansion,
                doc,
            );
            MemRefLayout::Opaque {
                spelling,
                parameters,
            }
        }
        Ok(layout) => layout,
        Err(message) => {
            if spelling.trim().starts_with("strided<") || spelling.trim().starts_with("affine_map<")
            {
                MemRefLayout::Opaque {
                    spelling: compact(spelling),
                    parameters: lower_memref_alias_parameters(
                        spelling,
                        range,
                        type_aliases,
                        attribute_aliases,
                        expansion,
                        doc,
                    ),
                }
            } else {
                MemRefLayout::Invalid(push_diagnostic(
                    doc,
                    expansion.diagnostic_code(SemanticDiagnosticCode::Type),
                    range,
                    message,
                ))
            }
        }
    }
}

pub(super) fn resolve_affine_alias<'a>(
    spelling: &'a str,
    attribute_aliases: &'a HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<Option<&'a str>, String> {
    let spelling = spelling.trim();
    if !spelling.starts_with('#') || spelling.contains('<') {
        return Ok(None);
    }
    let Some((target, _)) = attribute_aliases.get(spelling) else {
        return Ok(None);
    };
    stack.enter(spelling, "attribute")?;
    let result = if target.trim().starts_with('!') {
        Err(format!(
            "alias `{spelling}` has type kind, expected attribute"
        ))
    } else if target.trim().starts_with("affine_map<") || target.trim().starts_with("affine_set<") {
        Ok(Some(target.trim()))
    } else if target.trim().starts_with('#') {
        resolve_affine_alias(target, attribute_aliases, stack)
    } else {
        Ok(None)
    };
    stack.exit(spelling);
    result
}

fn resolve_memref_layout(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
) -> Result<MemRefLayout, String> {
    let spelling = spelling.trim();
    if spelling.starts_with("strided<") || spelling.starts_with("affine_map<") {
        if let Some(message) =
            first_invalid_memref_alias(spelling, type_aliases, attribute_aliases, expansion)
        {
            return Err(message);
        }
        return Ok(MemRefLayout::Opaque {
            spelling: compact(spelling),
            parameters: Vec::new(),
        });
    }
    resolve_attribute(spelling, type_aliases, attribute_aliases, expansion)
        .and_then(|value| match value {
            AttributeValue::Type(_) => Err("type value has wrong kind".into()),
            value => Ok(MemRefLayout::Attribute(Box::new(value))),
        })
        .map_err(|message| format!("invalid memref layout `{spelling}`: {message}"))
}

fn is_composite_type(spelling: &str) -> bool {
    let spelling = spelling.trim();
    split_arrow(spelling).is_some()
        || ["complex<", "tuple<", "tensor<", "vector<", "memref<"]
            .iter()
            .any(|prefix| spelling.starts_with(prefix))
}

fn lower_memref_alias_parameters(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
    doc: &mut Document,
) -> Vec<AttributeValue> {
    alias_spellings(spelling)
        .into_iter()
        .map(|alias| {
            let value = if alias.starts_with('!') {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    SemanticDiagnosticCode::Type,
                    range,
                    format!("memref layout alias `{alias}` has type kind, expected attribute"),
                ))
            } else {
                lower_attribute_value(
                    &alias,
                    range,
                    type_aliases,
                    attribute_aliases,
                    expansion,
                    doc,
                )
            };
            if matches!(value, AttributeValue::Type(_)) {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    SemanticDiagnosticCode::Type,
                    range,
                    format!("memref layout alias `{alias}` has type kind, expected attribute"),
                ))
            } else {
                value
            }
        })
        .collect()
}

fn first_invalid_memref_alias(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
) -> Option<String> {
    alias_spellings(spelling).into_iter().find_map(|alias| {
        if alias.starts_with('!') {
            return Some(format!(
                "alias `{alias}` has type kind, expected memref layout"
            ));
        }
        match resolve_attribute(&alias, type_aliases, attribute_aliases, expansion) {
            Ok(AttributeValue::Type(_)) => Some(format!(
                "memref layout alias `{alias}` has type kind, expected attribute"
            )),
            Ok(_) => None,
            Err(message) => Some(format!(
                "invalid memref layout parameter `{alias}`: {message}"
            )),
        }
    })
}

fn alias_spellings(spelling: &str) -> Vec<String> {
    let bytes = spelling.as_bytes();
    let mut aliases = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'#' || bytes[index] == b'!' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric()
                    || matches!(bytes[index], b'_' | b'$' | b'.' | b'-'))
            {
                index += 1;
            }
            aliases.push(spelling[start..index].to_owned());
        } else {
            index += 1;
        }
    }
    aliases
}

fn lower_memref_memory_space(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
    doc: &mut Document,
) -> AttributeValue {
    if spelling.trim().starts_with('!') {
        return AttributeValue::Invalid(push_diagnostic(
            doc,
            SemanticDiagnosticCode::Type,
            range,
            format!(
                "memref memory space `{}` has type kind, expected attribute",
                spelling.trim()
            ),
        ));
    }
    let value = lower_attribute_value(
        spelling,
        range,
        type_aliases,
        attribute_aliases,
        expansion,
        doc,
    );
    if matches!(value, AttributeValue::Type(_)) {
        AttributeValue::Invalid(push_diagnostic(
            doc,
            SemanticDiagnosticCode::Type,
            range,
            format!(
                "memref memory space `{}` has type kind, expected attribute",
                spelling.trim()
            ),
        ))
    } else {
        value
    }
}

fn resolve_memref_memory_space(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
) -> Result<AttributeValue, String> {
    if spelling.trim().starts_with('!') {
        return Err(format!(
            "memref memory space `{}` has type kind, expected attribute",
            spelling.trim()
        ));
    }
    resolve_attribute(spelling, type_aliases, attribute_aliases, expansion).and_then(|value| {
        match value {
            AttributeValue::Type(_) => Err(format!(
                "memref memory space `{}` has type kind, expected attribute",
                spelling.trim()
            )),
            value => Ok(value),
        }
    })
}

pub(super) fn resolve_type(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<TypeValue, String> {
    let spelling = spelling.trim();
    if spelling.starts_with('!') && !spelling.contains(['.', '<']) {
        let other = format!("#{}", &spelling[1..]);
        if attribute_aliases.contains_key(&other) {
            return Err(format!(
                "alias `{spelling}` has attribute kind, expected type"
            ));
        }
        let Some((target, _)) = type_aliases.get(spelling) else {
            return Err(format!("unresolved type alias `{spelling}`"));
        };
        stack.enter(spelling, "type")?;
        let result = resolve_type(target, type_aliases, attribute_aliases, stack);
        stack.exit(spelling);
        return result;
    }
    if spelling == "index" {
        return Ok(TypeValue::Index);
    }
    if let Some(width) = spelling.strip_prefix('i').and_then(parse_width) {
        return Ok(TypeValue::Integer {
            width,
            signedness: None,
        });
    }
    if let Some(width) = spelling.strip_prefix("si").and_then(parse_width) {
        return Ok(TypeValue::Integer {
            width,
            signedness: Some(true),
        });
    }
    if let Some(width) = spelling.strip_prefix("ui").and_then(parse_width) {
        return Ok(TypeValue::Integer {
            width,
            signedness: Some(false),
        });
    }
    if is_float_spelling(spelling) {
        return Ok(TypeValue::Float(spelling.to_owned()));
    }
    if let Some((inputs, results)) = split_arrow(spelling) {
        return Ok(TypeValue::Function {
            inputs: split_types(inputs)
                .iter()
                .map(|s| resolve_type(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
            results: split_types(results)
                .iter()
                .map(|s| resolve_type(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
        });
    }
    if let Some(inner) = angle_inner(spelling, "complex") {
        let element = resolve_type(inner, type_aliases, attribute_aliases, stack)?;
        if !matches!(element, TypeValue::Integer { .. } | TypeValue::Float(_)) {
            return Err("invalid element type for complex".into());
        }
        return Ok(TypeValue::Complex(Box::new(element)));
    }
    if let Some(inner) = angle_inner(spelling, "tuple") {
        return Ok(TypeValue::Tuple(
            split_types(inner)
                .iter()
                .map(|s| resolve_type(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
        ));
    }
    for (prefix, constructor) in [("tensor", 0u8), ("vector", 1), ("memref", 2)] {
        if let Some(inner) = angle_inner(spelling, prefix) {
            let parts = split_top_level_commas(inner);
            let shape = parts.first().copied().unwrap_or("");
            let shape_parts = split_top_level_x(shape);
            let Some(element) = shape_parts.last() else {
                return Err(format!("malformed {prefix} type"));
            };
            let unranked = prefix == "tensor" && shape_parts.first().copied() == Some("*");
            let dimensions = shape_parts[..shape_parts.len() - 1]
                .iter()
                .map(|d| {
                    let scalable = prefix == "vector" && d.starts_with('[');
                    let size = if *d == "?" || *d == "*" {
                        None
                    } else {
                        Some(
                            d.trim_matches(&['[', ']'][..])
                                .parse::<u64>()
                                .map_err(|_| format!("invalid {prefix} dimension `{d}`"))?,
                        )
                    };
                    Ok(ShapedDimension {
                        size,
                        scalable,
                        invalid: None,
                    })
                })
                .collect::<Result<Vec<ShapedDimension>, String>>()?;
            let element = Box::new(resolve_type(
                element,
                type_aliases,
                attribute_aliases,
                stack,
            )?);
            return Ok(match constructor {
                0 => TypeValue::Tensor {
                    dimensions,
                    element,
                    encoding: match parts.get(1) {
                        Some(encoding) => Some(Box::new(resolve_attribute(
                            encoding,
                            type_aliases,
                            attribute_aliases,
                            stack,
                        )?)),
                        None => None,
                    },
                    unranked,
                },
                1 => TypeValue::Vector {
                    dimensions,
                    element,
                    scalable: shape_parts[..shape_parts.len() - 1]
                        .iter()
                        .map(|d| d.starts_with('['))
                        .collect(),
                },
                _ => TypeValue::MemRef {
                    dimensions,
                    element,
                    layout: parts
                        .get(1)
                        .map(|value| {
                            resolve_memref_layout(value, type_aliases, attribute_aliases, stack)
                        })
                        .transpose()?,
                    memory_space: parts
                        .get(2)
                        .map(|value| {
                            resolve_memref_memory_space(
                                value,
                                type_aliases,
                                attribute_aliases,
                                stack,
                            )
                            .map(Box::new)
                        })
                        .transpose()?,
                },
            });
        }
    }
    if spelling.starts_with('!') {
        if spelling.contains('<') {
            let Some(open) = spelling.find('<') else {
                unreachable!()
            };
            let Some(close) = matching_delimiter(&spelling[open + 1..], '<', '>') else {
                return Err(format!("malformed opaque type `{spelling}`"));
            };
            if !spelling[open + 1 + close + 1..].trim().is_empty() {
                return Err(format!("trailing garbage after opaque type `{spelling}`"));
            }
        }
        return Ok(TypeValue::Opaque(Arc::from(spelling.as_bytes())));
    }
    Err(format!("unsupported or malformed type `{spelling}`"))
}
