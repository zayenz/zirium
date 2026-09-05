use super::lowering::Interner;

#[derive(Debug)]
pub(super) struct AliasExpansionState {
    limit: usize,
    active: HashSet<String>,
}

impl AliasExpansionState {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit,
            active: HashSet::new(),
        }
    }

    fn enter(&mut self, alias: &str, family: &str) -> Result<(), String> {
        if self.active.contains(alias) {
            return Err(format!("cyclic {family} alias `{alias}`"));
        }
        if self.active.len() >= self.limit {
            return Err(format!(
                "alias expansion depth exceeds limit of {}",
                self.limit
            ));
        }
        self.active.insert(alias.to_owned());
        Ok(())
    }

    fn exit(&mut self, alias: &str) {
        self.active.remove(alias);
    }
}

use super::*;

pub(super) fn text(bytes: &[u8], range: TextRange) -> &str {
    std::str::from_utf8(&bytes[range.start() as usize..range.end() as usize]).unwrap_or("")
}

pub(super) fn first_identifier(spelling: &str, sigil: u8) -> Option<String> {
    let bytes = spelling.as_bytes();
    let start = bytes.iter().position(|byte| *byte == sigil)?;
    let end = bytes[start + 1..]
        .iter()
        .position(|byte| byte.is_ascii_whitespace() || b"#: ,()={}[]".contains(byte))
        .map_or(bytes.len(), |end| start + 1 + end);
    Some(spelling[start + 1..end].to_owned())
}

pub(super) fn argument_type(spelling: &str) -> &str {
    let Some((_, tail)) = spelling.split_once(':') else {
        return "<invalid>";
    };
    tail.trim().split("loc(").next().unwrap_or(tail).trim()
}

pub(super) fn operation_output_types(
    op: crate::parser::OperationSyntax<'_>,
    bytes: &[u8],
) -> Vec<String> {
    let function = op
        .tree()
        .children(op.id())
        .into_iter()
        .flatten()
        .find(|child| op.tree().kind(*child) == Some(SyntaxKind::FunctionType));
    let Some(range) = function.and_then(|node| op.tree().text_range(node)) else {
        return Vec::new();
    };
    let spelling = text(bytes, range);
    let output = spelling
        .split_once("->")
        .map(|(_, output)| output.trim())
        .unwrap_or("");
    split_types(output)
}

fn split_types(spelling: &str) -> Vec<String> {
    let spelling = spelling.trim();
    let inner = spelling
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .unwrap_or(spelling);
    if inner.trim().is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in inner.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        if byte == b'"' {
            quoted = true;
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'>' if index == 0 || inner.as_bytes()[index - 1] != b'-' => depth -= 1,
            b',' if depth == 0 => {
                result.push(inner[start..index].trim().to_owned());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(inner[start..].trim().to_owned());
    result
}

pub(crate) fn split_registered_types(spelling: &str) -> Vec<String> {
    split_types(spelling)
}

// These parameters keep type interning explicit at the lowering boundary instead of
// hiding mutable lowering state in a broader context object.
#[allow(clippy::too_many_arguments)]
pub(super) fn intern_type(
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

fn lower_type_value_with_stack(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    alias_stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> TypeValue {
    let spelling = spelling.trim();
    if let Some((target, _)) = type_aliases.get(spelling) {
        if target != spelling {
            if let Err(message) = alias_stack.enter(spelling, "type") {
                return TypeValue::Invalid(push_diagnostic(doc, range, message));
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
    }
    if !is_composite_type(spelling) {
        if let Ok(value) = resolve_type(spelling, type_aliases, attribute_aliases, alias_stack) {
            return value;
        }
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
    TypeValue::Invalid(push_diagnostic(doc, range, message))
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
                range,
                "memref affine layout has wrong kind".into(),
            )),
        };
    }
    if let Some(affine_spelling) =
        match resolve_affine_alias(spelling.trim(), attribute_aliases, expansion) {
            Ok(value) => value,
            Err(message) => return MemRefLayout::Invalid(push_diagnostic(doc, range, message)),
        }
    {
        return match lower_affine_attribute(affine_spelling, range, doc) {
            AttributeValue::AffineMap(map) => MemRefLayout::AffineMap(map),
            AttributeValue::Invalid(diagnostic) => MemRefLayout::Invalid(diagnostic),
            AttributeValue::IntegerSet(_) => MemRefLayout::Invalid(push_diagnostic(
                doc,
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
                MemRefLayout::Invalid(push_diagnostic(doc, range, message))
            }
        }
    }
}

fn resolve_affine_alias<'a>(
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
    spelling.contains("->")
        || ["tuple<", "tensor<", "vector<", "memref<"]
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

fn resolve_type(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<TypeValue, String> {
    let spelling = spelling.trim();
    if spelling.starts_with('!') && !spelling.contains('<') {
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

fn resolve_attribute(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<AttributeValue, String> {
    let spelling = spelling.trim();
    if spelling == "true" || spelling == "false" {
        return Ok(AttributeValue::Boolean(spelling == "true"));
    }
    if spelling.starts_with("dense_resource<") {
        return balanced_large_attribute(spelling, "dense_resource", LargeAttributeValue::Resource);
    }
    if spelling.starts_with("dense<") {
        return balanced_large_attribute(spelling, "dense", LargeAttributeValue::Dense);
    }
    if spelling.starts_with("sparse<") {
        return balanced_large_attribute(spelling, "sparse", LargeAttributeValue::Sparse);
    }
    if let Some(inner) = angle_inner(spelling, "array") {
        let (element_type, elements) = parse_dense_array(inner)?;
        return Ok(AttributeValue::DenseArray {
            element_type,
            elements,
        });
    }
    if spelling.starts_with('#') && !spelling.contains('<') {
        let other = format!("!{}", &spelling[1..]);
        if type_aliases.contains_key(&other) {
            return Err(format!(
                "alias `{spelling}` has type kind, expected attribute"
            ));
        }
        let Some((target, _)) = attribute_aliases.get(spelling) else {
            return Err(format!("unresolved attribute alias `{spelling}`"));
        };
        stack.enter(spelling, "attribute")?;
        let result = resolve_attribute(target, type_aliases, attribute_aliases, stack);
        stack.exit(spelling);
        return result;
    }
    if spelling.starts_with('@') {
        return Ok(AttributeValue::Symbol(
            spelling
                .split("::")
                .map(|s| s.trim_start_matches('@').to_owned())
                .collect(),
        ));
    }
    if spelling.starts_with('"') {
        return Ok(AttributeValue::String(spelling.to_owned()));
    }
    if let Some(inner) = bracket_inner(spelling, '[', ']') {
        return Ok(AttributeValue::Array(
            split_types(inner)
                .iter()
                .map(|s| resolve_attribute(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
        ));
    }
    if let Some(inner) = bracket_inner(spelling, '{', '}') {
        let mut entries = split_types(inner)
            .into_iter()
            .map(|entry| {
                let (name, value) = entry
                    .split_once('=')
                    .ok_or_else(|| format!("malformed dictionary entry `{entry}`"))?;
                Ok((
                    name.trim().to_owned(),
                    resolve_attribute(value, type_aliases, attribute_aliases, stack)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        return Ok(AttributeValue::Dictionary(entries));
    }
    if spelling.starts_with("loc(") {
        return parse_location(spelling)
            .map(AttributeValue::Location)
            .ok_or_else(|| "invalid semantic location".into());
    }
    if let Ok(ty) = resolve_type(spelling, type_aliases, attribute_aliases, stack) {
        return Ok(AttributeValue::Type(ty));
    }
    let literal = spelling.split(':').next().unwrap_or(spelling).trim();
    if is_valid_wide_number(spelling) {
        return Ok(AttributeValue::WideNumber(Arc::from(spelling.as_bytes())));
    }
    if literal.parse::<i128>().is_ok() {
        return Ok(AttributeValue::Integer(compact(spelling)));
    }
    if literal.parse::<f64>().is_ok() {
        return Ok(AttributeValue::Float(compact(spelling)));
    }
    if spelling.starts_with('#') {
        let Some(open) = spelling.find('<') else {
            return Err(format!("malformed opaque attribute `{spelling}`"));
        };
        let Some(close) = matching_delimiter(&spelling[open + 1..], '<', '>') else {
            return Err(format!("malformed opaque attribute `{spelling}`"));
        };
        if !spelling[open + 1 + close + 1..].trim().is_empty() {
            return Err(format!(
                "trailing garbage after opaque attribute `{spelling}`"
            ));
        }
        return Ok(AttributeValue::Opaque(Arc::from(spelling.as_bytes())));
    }
    Err(format!("unsupported or malformed attribute `{spelling}`"))
}

fn balanced_large_attribute(
    spelling: &str,
    prefix: &str,
    wrap: impl FnOnce(Arc<[u8]>) -> LargeAttributeValue,
) -> Result<AttributeValue, String> {
    let rest = spelling
        .strip_prefix(prefix)
        .and_then(|value| value.strip_prefix('<'))
        .ok_or_else(|| format!("malformed {prefix} payload"))?;
    let close =
        matching_delimiter(rest, '<', '>').ok_or_else(|| format!("malformed {prefix} payload"))?;
    let suffix = rest[close + 1..].trim();
    let Some(suffix) = suffix.strip_prefix(':').map(str::trim) else {
        return Err(format!("malformed {prefix} payload suffix"));
    };
    if suffix.is_empty()
        || resolve_type(
            suffix,
            &HashMap::new(),
            &HashMap::new(),
            &mut AliasExpansionState::new(64),
        )
        .is_err()
    {
        return Err(format!("malformed {prefix} payload suffix"));
    }
    Ok(AttributeValue::Large(wrap(Arc::from(spelling.as_bytes()))))
}

fn is_valid_wide_number(value: &str) -> bool {
    let Some((literal, suffix)) = value.split_once(':') else {
        return false;
    };
    let literal = literal.trim().trim_start_matches(['+', '-']);
    let suffix = suffix.trim();
    if literal.is_empty() || suffix.len() < 2 {
        return false;
    }
    let digits = literal.strip_prefix("0x").unwrap_or(literal);
    let is_hex = literal.starts_with("0x");
    if digits.is_empty()
        || (!is_hex && !digits.bytes().all(|byte| byte.is_ascii_digit()))
        || (is_hex && !digits.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return false;
    }
    let width = suffix
        .strip_prefix('i')
        .or_else(|| suffix.strip_prefix("si"))
        .or_else(|| suffix.strip_prefix("ui"));
    width.is_some_and(|width| parse_width(width).is_some())
}

fn parse_location(spelling: &str) -> Option<LocationValue> {
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

pub(super) fn lower_location_value(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> LocationValue {
    let invalid = |doc: &mut Document, message: String| {
        LocationValue::Invalid(push_diagnostic(doc, range, message))
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
            return invalid(doc, message);
        };
        if let Err(message) = stack.enter(&alias, "location") {
            return invalid(doc, message);
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
        return invalid(doc, "invalid semantic location".into());
    };
    if inner.starts_with('#') {
        return lower_location_value(inner, range, type_aliases, attribute_aliases, stack, doc);
    }
    if let Some(fused) = inner.strip_prefix("fused") {
        let (metadata, values) = match fused_parts(fused) {
            Some(parts) => parts,
            None => return invalid(doc, "malformed fused location".into()),
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
            return invalid(doc, "malformed callsite location".into());
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

fn matching_delimiter(value: &str, _open: char, close: char) -> Option<usize> {
    let mut expected = vec![close];
    let mut quoted = false;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
            continue;
        }
        if character == '"' {
            quoted = true;
            continue;
        }
        if let Some(nested_close) = delimiter_close(character) {
            expected.push(nested_close);
            continue;
        }
        if matches!(character, ')' | ']' | '}' | '>') {
            if expected.pop() != Some(character) {
                return None;
            }
            if expected.is_empty() {
                return Some(index);
            }
        }
    }
    None
}

fn delimiter_close(character: char) -> Option<char> {
    match character {
        '(' => Some(')'),
        '[' => Some(']'),
        '{' => Some('}'),
        '<' => Some('>'),
        _ => None,
    }
}

fn compact(value: &str) -> String {
    value.chars().filter(|c| !c.is_whitespace()).collect()
}
fn angle_inner<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    value
        .strip_prefix(prefix)?
        .trim()
        .strip_prefix('<')?
        .strip_suffix('>')
}
fn bracket_inner(value: &str, open: char, close: char) -> Option<&str> {
    value.trim().strip_prefix(open)?.strip_suffix(close)
}
fn split_arrow(value: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    let mut quoted = false;
    let mut escaped = false;
    let bytes = value.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        let byte = bytes[i];
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        if byte == b'"' {
            quoted = true;
            continue;
        }
        match byte {
            b'(' | b'<' | b'[' | b'{' => depth += 1,
            b')' | b'>' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if &bytes[i..i + 2] == b"->" && depth == 0 {
            return Some((&value[..i], &value[i + 2..]));
        }
    }
    None
}
fn split_top_level_x(value: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    for (i, byte) in value.bytes().enumerate() {
        match byte {
            b'<' | b'(' | b'[' | b'{' => depth += 1,
            b'>' | b')' | b']' | b'}' => depth -= 1,
            b'x' if depth == 0 => {
                result.push(value[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(value[start..].trim());
    result
}

fn split_top_level_commas(value: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'<' | b'(' | b'[' | b'{' => depth += 1,
            b'>' | b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                result.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(value[start..].trim());
    result
}

fn parse_width(value: &str) -> Option<u32> {
    (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

fn is_float_spelling(value: &str) -> bool {
    matches!(
        value,
        "bf16"
            | "tf32"
            | "f16"
            | "f32"
            | "f64"
            | "f80"
            | "f128"
            | "f8E4M3"
            | "f8E5M2"
            | "f8E4M3FN"
            | "f8E5M2FNUZ"
            | "f8E4M3FNUZ"
            | "f8E4M3B11FNUZ"
            | "f8E3M4"
            | "f8E8M0FNU"
            | "f4E2M1FN"
            | "f6E2M3FN"
            | "f6E3M2FN"
    )
}

// Attribute and property dictionaries share this lowering path, whose inputs are
// deliberately passed separately to keep the surrounding lowering state local.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_dictionary(
    dictionary: Option<crate::representation::NodeId>,
    tree: &crate::representation::SyntaxTree,
    bytes: &[u8],
    strings: &mut Interner,
    attributes: &mut Interner<AttributeValue>,
    attribute_spellings: &mut Vec<String>,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    generation: u128,
    kind: &str,
    doc: &mut Document,
) -> Vec<(u32, AttributeId)> {
    let Some(dictionary) = dictionary else {
        return Vec::new();
    };
    let mut seen = HashMap::<u32, TextRange>::new();
    let mut result = tree
        .children(dictionary)
        .into_iter()
        .flatten()
        .filter(|child| tree.kind(*child) == Some(SyntaxKind::Attribute))
        .filter_map(|attribute| {
            let attribute_range = tree.text_range(attribute)?;
            let spelling = text(bytes, attribute_range);
            let (name, value) = spelling.split_once('=').unwrap_or((spelling, ""));
            let name_id = strings.intern(name.trim());
            let duplicate = seen.insert(name_id, attribute_range).map(|previous| {
                push_diagnostic(
                    doc,
                    attribute_range,
                    format!(
                        "duplicate {kind} key `{}` (previous at {})",
                        name.trim(),
                        previous.start()
                    ),
                )
            });
            let value_spelling = value.trim();
            let malformed_numeric_prefix = value_spelling == "0"
                && bytes
                    .get(attribute_range.end() as usize)
                    .is_some_and(|byte| *byte == b'x');
            let integer_suffix = value_spelling
                .split_once(':')
                .map(|(_, suffix)| suffix.trim())
                .and_then(|suffix| {
                    suffix
                        .strip_prefix('i')
                        .or_else(|| suffix.strip_prefix("si"))
                        .or_else(|| suffix.strip_prefix("ui"))
                })
                .is_some_and(|width| parse_width(width).is_some());
            let numeric_payload_candidate = malformed_numeric_prefix
                || (value_spelling
                    .trim_start_matches(['+', '-'])
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_digit())
                    && value_spelling.contains(':'))
                || integer_suffix;
            let affine_value = value_spelling.starts_with("affine_map<")
                || value_spelling.starts_with("affine_set<");
            let owned_payload_candidate = !affine_value
                && (value_spelling.starts_with("dense<")
                    || value_spelling.starts_with("sparse<")
                    || value_spelling.starts_with("dense_resource<")
                    || (value_spelling.starts_with('#') && value_spelling.contains('<'))
                    || numeric_payload_candidate);
            let semantic = if name.trim() == "no_inline" && value_spelling.is_empty() {
                AttributeValue::Opaque(Arc::from(b"unit".as_slice()))
            } else if malformed_numeric_prefix
                || (owned_payload_candidate && tree.has_error(attribute).unwrap_or(false))
            {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    attribute_range,
                    "malformed attribute value".into(),
                ))
            } else {
                let mut expansion = AliasExpansionState::new(doc.alias_expansion_depth_limit);
                lower_attribute_value(
                    value_spelling,
                    attribute_range,
                    type_aliases,
                    attribute_aliases,
                    &mut expansion,
                    doc,
                )
            };
            let semantic = if let Some(diagnostic) = duplicate {
                AttributeValue::Invalid(diagnostic)
            } else if value_spelling.is_empty() && name.trim() != "no_inline" {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    attribute_range,
                    "malformed dictionary entry".into(),
                ))
            } else {
                semantic
            };
            let index = attributes.intern_value(semantic);
            if index as usize == attribute_spellings.len() {
                attribute_spellings.push(value_spelling.to_owned());
            }
            Some((name_id, AttributeId::new(index as usize, generation)))
        })
        .collect::<Vec<_>>();
    result.sort_by_key(|(name, _)| strings.values[*name as usize].clone());
    result
}

pub(super) fn lower_attribute_value(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> AttributeValue {
    lower_attribute_value_with_depth(
        spelling,
        range,
        type_aliases,
        attribute_aliases,
        stack,
        doc,
        0,
    )
}

fn lower_attribute_value_with_depth(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
    depth: usize,
) -> AttributeValue {
    let spelling = spelling.trim();
    if spelling == "true" || spelling == "false" {
        return AttributeValue::Boolean(spelling == "true");
    }
    if spelling == "unit" {
        return AttributeValue::Opaque(Arc::from(b"unit".as_slice()));
    }
    if let Some(inner) = angle_inner(spelling, "array") {
        if depth >= doc.attribute_depth_limit {
            return AttributeValue::Invalid(push_diagnostic(
                doc,
                range,
                "attribute nesting depth limit exceeded".into(),
            ));
        }
        return match parse_dense_array(inner) {
            Ok((element_type, elements)) => AttributeValue::DenseArray {
                element_type,
                elements,
            },
            Err(message) => AttributeValue::Invalid(push_diagnostic(doc, range, message)),
        };
    }
    if let Some(inner) = angle_inner(spelling, "type") {
        return AttributeValue::Type(lower_type_value_with_stack(
            inner,
            range,
            type_aliases,
            attribute_aliases,
            stack,
            doc,
        ));
    }
    if spelling.starts_with("affine_map<") || spelling.starts_with("affine_set<") {
        return lower_affine_attribute(spelling, range, doc);
    }
    if spelling.starts_with('#') && !spelling.contains('<') {
        match resolve_affine_alias(spelling, attribute_aliases, stack) {
            Ok(Some(target)) => return lower_affine_attribute(target, range, doc),
            Err(message) => {
                return AttributeValue::Invalid(push_diagnostic(doc, range, message));
            }
            Ok(None) => {}
        }
    }
    if let Some(inner) = bracket_inner(spelling, '[', ']') {
        if depth >= doc.attribute_depth_limit {
            return AttributeValue::Invalid(push_diagnostic(
                doc,
                range,
                "attribute nesting depth limit exceeded".into(),
            ));
        }
        return AttributeValue::Array(
            split_types(inner)
                .iter()
                .map(|item| {
                    lower_attribute_value_with_depth(
                        item,
                        range,
                        type_aliases,
                        attribute_aliases,
                        stack,
                        doc,
                        depth + 1,
                    )
                })
                .collect(),
        );
    }
    if let Some(inner) = bracket_inner(spelling, '{', '}') {
        if depth >= doc.attribute_depth_limit {
            return AttributeValue::Invalid(push_diagnostic(
                doc,
                range,
                "attribute nesting depth limit exceeded".into(),
            ));
        }
        let mut seen = HashMap::new();
        let mut entries = split_types(inner)
            .into_iter()
            .map(|entry| {
                let Some((name, value)) = entry.split_once('=') else {
                    push_diagnostic(doc, range, format!("malformed dictionary entry `{entry}`"));
                    return (
                        "<invalid>".into(),
                        AttributeValue::Invalid(push_diagnostic(
                            doc,
                            range,
                            "malformed dictionary entry".into(),
                        )),
                    );
                };
                let name = name.trim().to_owned();
                let duplicate = seen.insert(name.clone(), ()).is_some();
                let value = lower_attribute_value_with_depth(
                    value,
                    range,
                    type_aliases,
                    attribute_aliases,
                    stack,
                    doc,
                    depth + 1,
                );
                let value = if duplicate {
                    AttributeValue::Invalid(push_diagnostic(
                        doc,
                        range,
                        format!("duplicate dictionary key `{name}`"),
                    ))
                } else {
                    value
                };
                (name, value)
            })
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        return AttributeValue::Dictionary(entries);
    }
    if spelling.starts_with("loc(") {
        return AttributeValue::Location(lower_location_value(
            spelling,
            range,
            type_aliases,
            attribute_aliases,
            stack,
            doc,
        ));
    }
    if spelling.starts_with('!')
        || spelling.starts_with('i')
        || spelling.starts_with("si")
        || spelling.starts_with("ui")
        || spelling.starts_with('f')
        || spelling.starts_with('b')
        || spelling.starts_with('t')
        || spelling == "index"
        || spelling.starts_with("tensor<")
        || spelling.starts_with("vector<")
        || spelling.starts_with("memref<")
        || spelling.starts_with("tuple<")
        || spelling.contains("->")
    {
        return AttributeValue::Type(lower_type_value_with_stack(
            spelling,
            range,
            type_aliases,
            attribute_aliases,
            stack,
            doc,
        ));
    }
    match resolve_attribute(spelling, type_aliases, attribute_aliases, stack) {
        Ok(value) => value,
        Err(message) => AttributeValue::Invalid(push_diagnostic(doc, range, message)),
    }
}

fn parse_dense_array(inner: &str) -> Result<(String, Vec<AttributeValue>), String> {
    let (element_type, payload) = match inner.split_once(':') {
        Some((ty, values)) => (ty.trim(), Some(values)),
        None => (inner.trim(), None),
    };
    if !matches!(
        element_type,
        "i1" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
    ) {
        return Err(format!(
            "unsupported dense array element type `{element_type}`"
        ));
    }
    let mut elements = Vec::new();
    if let Some(payload) = payload {
        if payload.trim().is_empty() || payload.trim_end().ends_with(',') {
            return Err("malformed dense array payload".into());
        }
        for item in payload.split(',') {
            let item = item.trim();
            let value = if element_type == "i1" {
                match item {
                    "true" => AttributeValue::Boolean(true),
                    "false" => AttributeValue::Boolean(false),
                    _ => {
                        return Err(format!(
                            "dense array element `{item}` does not match `{element_type}`"
                        ));
                    }
                }
            } else if element_type.starts_with('i') {
                if item.parse::<i128>().is_err() {
                    return Err(format!(
                        "dense array element `{item}` does not match `{element_type}`"
                    ));
                }
                AttributeValue::Integer(compact(item))
            } else {
                if item.parse::<f64>().is_err() {
                    return Err(format!(
                        "dense array element `{item}` does not match `{element_type}`"
                    ));
                }
                AttributeValue::Float(compact(item))
            };
            elements.push(value);
        }
    }
    Ok((element_type.to_owned(), elements))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AffineToken {
    Identifier(String),
    Integer(i64),
    InvalidInteger(String),
    InvalidOperator(String),
    Plus,
    Minus,
    Star,
    FloorDiv,
    CeilDiv,
    Mod,
    LParen,
    RParen,
}

fn lower_affine_attribute(spelling: &str, range: TextRange, doc: &mut Document) -> AttributeValue {
    let (kind, inner) = if let Some(inner) = angle_inner(spelling, "affine_map") {
        (SyntaxKind::AffineMap, inner)
    } else if let Some(inner) = angle_inner(spelling, "affine_set") {
        (SyntaxKind::IntegerSet, inner)
    } else {
        let kind = if spelling.trim().starts_with("affine_map") {
            SyntaxKind::AffineMap
        } else {
            SyntaxKind::IntegerSet
        };
        return invalid_affine_attribute(kind, range, doc, "malformed affine value");
    };
    let Some(after_open) = inner.strip_prefix('(') else {
        return invalid_affine_attribute(kind, range, doc, "malformed affine dimension arity");
    };
    let Some(dim_tail_end) = matching_delimiter(after_open, '(', ')') else {
        return invalid_affine_attribute(kind, range, doc, "malformed affine dimension arity");
    };
    let dim_end = dim_tail_end + 1;
    let dimensions = parse_affine_names(&inner[1..dim_end], "dimension", range, doc);
    let mut rest = inner.get(dim_end + 1..).unwrap_or("").trim();
    let symbols = if rest.starts_with('[') {
        let Some(after_open) = rest.strip_prefix('[') else {
            return invalid_affine_attribute(kind, range, doc, "malformed affine symbol arity");
        };
        let Some(tail_end) = matching_delimiter(after_open, '[', ']') else {
            return invalid_affine_attribute(kind, range, doc, "malformed affine symbol arity");
        };
        let end = tail_end + 1;
        let names = parse_affine_names(&rest[1..end], "symbol", range, doc);
        rest = rest.get(end + 1..).unwrap_or("").trim();
        names
    } else {
        Vec::new()
    };
    let separator = if kind == SyntaxKind::AffineMap {
        "->"
    } else {
        ":"
    };
    let Some(body) = rest.strip_prefix(separator).map(str::trim) else {
        return invalid_affine_attribute(
            kind,
            range,
            doc,
            &format!("malformed affine {separator} separator"),
        );
    };
    let Some(body) = bracket_inner(body, '(', ')') else {
        return invalid_affine_attribute(
            kind,
            range,
            doc,
            "malformed affine result or constraint list",
        );
    };
    if kind == SyntaxKind::AffineMap {
        let results = split_affine_items(body)
            .into_iter()
            .map(|expression| {
                lower_affine_expression(expression, &dimensions, &symbols, range, doc)
            })
            .collect::<Vec<_>>();
        let value = AffineMapValue {
            dimensions: dimensions.len() as u32,
            symbols: symbols.len() as u32,
            results,
        };
        let index = intern_affine_map(doc, value);
        AttributeValue::AffineMap(AffineMapId::new(index, doc.generation))
    } else {
        let constraints = split_affine_items(body)
            .into_iter()
            .map(|constraint| {
                lower_integer_constraint(constraint, &dimensions, &symbols, range, doc)
            })
            .collect::<Vec<_>>();
        let value = IntegerSetValue {
            dimensions: dimensions.len() as u32,
            symbols: symbols.len() as u32,
            constraints,
        };
        let index = intern_integer_set(doc, value);
        AttributeValue::IntegerSet(IntegerSetId::new(index, doc.generation))
    }
}

fn invalid_affine_attribute(
    kind: SyntaxKind,
    range: TextRange,
    doc: &mut Document,
    message: &str,
) -> AttributeValue {
    let expression = invalid_affine_expression(doc, range, message);
    if kind == SyntaxKind::AffineMap {
        let index = intern_affine_map(
            doc,
            AffineMapValue {
                dimensions: 0,
                symbols: 0,
                results: vec![expression],
            },
        );
        AttributeValue::AffineMap(AffineMapId::new(index, doc.generation))
    } else {
        let diagnostic = push_diagnostic(doc, range, message.to_owned());
        let right = invalid_affine_expression(doc, range, message);
        let index = intern_integer_set(
            doc,
            IntegerSetValue {
                dimensions: 0,
                symbols: 0,
                constraints: vec![IntegerSetConstraint {
                    left: expression,
                    relation: IntegerSetRelation::Invalid(diagnostic),
                    right,
                }],
            },
        );
        AttributeValue::IntegerSet(IntegerSetId::new(index, doc.generation))
    }
}

fn invalid_affine_expression(doc: &mut Document, range: TextRange, message: &str) -> AffineExprId {
    let diagnostic = push_diagnostic(doc, range, message.to_owned());
    let index = intern_affine_expression(doc, AffineExprValue::Invalid(diagnostic));
    AffineExprId::new(index, doc.generation)
}

fn split_affine_items(value: &str) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut result = Vec::new();
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                result.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(value[start..].trim());
    result
}

fn parse_affine_names(
    value: &str,
    kind: &str,
    range: TextRange,
    doc: &mut Document,
) -> Vec<String> {
    let mut names = Vec::new();
    for name in value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        if !name.bytes().enumerate().all(|(i, byte)| {
            byte == b'_' || byte.is_ascii_alphanumeric() && (i > 0 || !byte.is_ascii_digit())
        }) {
            push_diagnostic(
                doc,
                range,
                format!("malformed affine {kind} identifier `{name}`"),
            );
        }
        if names.iter().any(|existing| existing == name) {
            push_diagnostic(
                doc,
                range,
                format!("duplicate affine {kind} identifier `{name}`"),
            );
        }
        names.push(name.to_owned());
    }
    names
}

fn lower_integer_constraint(
    value: &str,
    dimensions: &[String],
    symbols: &[String],
    range: TextRange,
    doc: &mut Document,
) -> IntegerSetConstraint {
    let mut depth = 0usize;
    let mut found = None;
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b'=' | b'>' | b'<' if depth == 0 => {
                found = Some((index, byte));
                break;
            }
            _ => {}
        }
    }
    let (left, relation, right) = match found {
        Some((index, b'=')) if value[index..].starts_with("==") => (
            &value[..index],
            IntegerSetRelation::Equal,
            &value[index + 2..],
        ),
        Some((index, b'>')) if value[index..].starts_with(">=") => (
            &value[..index],
            IntegerSetRelation::GreaterEqual,
            &value[index + 2..],
        ),
        Some((index, b'<')) if value[index..].starts_with("<=") => (
            &value[..index],
            IntegerSetRelation::LessEqual,
            &value[index + 2..],
        ),
        Some((index, _)) => {
            let diagnostic = push_diagnostic(
                doc,
                range,
                format!("invalid affine constraint operator in `{value}`"),
            );
            (
                &value[..index],
                IntegerSetRelation::Invalid(diagnostic),
                &value[index + 1..],
            )
        }
        None => {
            let diagnostic = push_diagnostic(
                doc,
                range,
                format!("missing affine constraint operator in `{value}`"),
            );
            (value, IntegerSetRelation::Invalid(diagnostic), "")
        }
    };
    IntegerSetConstraint {
        left: lower_affine_expression(left, dimensions, symbols, range, doc),
        relation,
        right: lower_affine_expression(right, dimensions, symbols, range, doc),
    }
}

fn lower_affine_expression(
    value: &str,
    dimensions: &[String],
    symbols: &[String],
    range: TextRange,
    doc: &mut Document,
) -> AffineExprId {
    let tokens = tokenize_affine(value);
    let mut parser = AffineExpressionParser {
        tokens: &tokens,
        position: 0,
        dimensions,
        symbols,
        range,
        doc,
    };
    let expression = parser.expression(0);
    if parser.position != tokens.len() {
        return parser.invalid(format!("malformed affine expression `{}`", value.trim()));
    }
    expression
}

fn tokenize_affine(value: &str) -> Vec<AffineToken> {
    let bytes = value.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let token = match bytes[i] {
            b'+' => {
                i += 1;
                AffineToken::Plus
            }
            b'-' => {
                i += 1;
                AffineToken::Minus
            }
            b'*' => {
                i += 1;
                AffineToken::Star
            }
            b'(' => {
                i += 1;
                AffineToken::LParen
            }
            b')' => {
                i += 1;
                AffineToken::RParen
            }
            b'0'..=b'9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let literal = &value[start..i];
                match literal.parse() {
                    Ok(value) => AffineToken::Integer(value),
                    Err(_) => AffineToken::InvalidInteger(literal.to_owned()),
                }
            }
            b'/' | b'%' => {
                let operator = bytes[i] as char;
                i += 1;
                AffineToken::InvalidOperator(operator.to_string())
            }
            _ => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &value[start..i.max(start + 1)];
                if i == start {
                    i += 1;
                }
                match word {
                    "floordiv" => AffineToken::FloorDiv,
                    "ceildiv" => AffineToken::CeilDiv,
                    "mod" => AffineToken::Mod,
                    _ => AffineToken::Identifier(word.to_owned()),
                }
            }
        };
        tokens.push(token);
    }
    tokens
}

struct AffineExpressionParser<'a> {
    tokens: &'a [AffineToken],
    position: usize,
    dimensions: &'a [String],
    symbols: &'a [String],
    range: TextRange,
    doc: &'a mut Document,
}

impl AffineExpressionParser<'_> {
    fn expression(&mut self, minimum: u8) -> AffineExprId {
        let mut left = self.primary();
        loop {
            let (precedence, operator) = match self.tokens.get(self.position) {
                Some(AffineToken::Plus) => (1, AffineBinaryOperator::Add),
                Some(AffineToken::Minus) => (1, AffineBinaryOperator::Subtract),
                Some(AffineToken::Star) => (2, AffineBinaryOperator::Multiply),
                Some(AffineToken::FloorDiv) => (2, AffineBinaryOperator::FloorDiv),
                Some(AffineToken::CeilDiv) => (2, AffineBinaryOperator::CeilDiv),
                Some(AffineToken::Mod) => (2, AffineBinaryOperator::Mod),
                _ => break,
            };
            if precedence < minimum {
                break;
            }
            self.position += 1;
            let right = self.expression(precedence + 1);
            left = self.intern(AffineExprValue::Binary {
                operator,
                left,
                right,
            });
        }
        left
    }
    fn primary(&mut self) -> AffineExprId {
        match self.tokens.get(self.position).cloned() {
            Some(AffineToken::Integer(value)) => {
                self.position += 1;
                self.intern(AffineExprValue::Constant(value))
            }
            Some(AffineToken::InvalidInteger(literal)) => {
                self.position += 1;
                self.invalid(format!(
                    "affine integer literal `{literal}` is out of range for i64"
                ))
            }
            Some(AffineToken::InvalidOperator(operator)) => {
                self.position += 1;
                self.invalid(format!("unsupported affine operator `{operator}`"))
            }
            Some(AffineToken::Identifier(name)) => {
                self.position += 1;
                if let Some(index) = self.dimensions.iter().position(|value| value == &name) {
                    self.intern(AffineExprValue::Dimension(index as u32))
                } else if let Some(index) = self.symbols.iter().position(|value| value == &name) {
                    self.intern(AffineExprValue::Symbol(index as u32))
                } else {
                    self.invalid(format!(
                        "affine identifier `{name}` exceeds declared dimension/symbol arity"
                    ))
                }
            }
            Some(AffineToken::Minus) => {
                self.position += 1;
                let right = self.primary();
                let zero = self.intern(AffineExprValue::Constant(0));
                self.intern(AffineExprValue::Binary {
                    operator: AffineBinaryOperator::Subtract,
                    left: zero,
                    right,
                })
            }
            Some(AffineToken::LParen) => {
                self.position += 1;
                let value = self.expression(0);
                if self.tokens.get(self.position) == Some(&AffineToken::RParen) {
                    self.position += 1;
                    value
                } else {
                    self.invalid("unclosed affine expression".into())
                }
            }
            _ => self.invalid("missing affine expression operand".into()),
        }
    }
    fn intern(&mut self, value: AffineExprValue) -> AffineExprId {
        let index = intern_affine_expression(self.doc, value);
        AffineExprId::new(index, self.doc.generation)
    }
    fn invalid(&mut self, message: String) -> AffineExprId {
        let diagnostic = push_diagnostic(self.doc, self.range, message);
        self.intern(AffineExprValue::Invalid(diagnostic))
    }
}

fn intern_affine_expression(doc: &mut Document, value: AffineExprValue) -> usize {
    if let Some(index) = doc
        .affine_expressions
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.affine_expressions.len();
        doc.affine_expressions.push(value);
        index
    }
}
fn intern_affine_map(doc: &mut Document, value: AffineMapValue) -> usize {
    if let Some(index) = doc
        .affine_maps
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.affine_maps.len();
        doc.affine_maps.push(value);
        index
    }
}
fn intern_integer_set(doc: &mut Document, value: IntegerSetValue) -> usize {
    if let Some(index) = doc
        .integer_sets
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.integer_sets.len();
        doc.integer_sets.push(value);
        index
    }
}

// Resolution needs both region- and block-scoped definition maps plus the region
// ancestry; keeping them explicit makes the lookup rules visible at each call site.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_value(
    spelling: &str,
    range: TextRange,
    mut region: Option<RegionId>,
    block: Option<BlockId>,
    region_definitions: &HashMap<(Option<RegionId>, String), Vec<ValueId>>,
    block_definitions: &HashMap<(BlockId, String), Vec<ValueId>>,
    region_outer: &HashMap<RegionId, Option<RegionId>>,
    doc: &mut Document,
) -> ValueReference {
    let name = first_identifier(spelling, b'%').unwrap_or_default();
    let number = spelling
        .split_once('#')
        .and_then(|(_, number)| number.trim().split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|number| number.parse::<usize>().ok())
        .unwrap_or(0);
    if let Some(block) = block {
        if let Some(value) = block_definitions
            .get(&(block, name.clone()))
            .and_then(|values| values.get(number))
            .copied()
        {
            return ValueReference::Resolved(value);
        }
    }
    loop {
        if let Some(value) = region_definitions
            .get(&(region, name.clone()))
            .and_then(|values| values.get(number))
            .copied()
        {
            return ValueReference::Resolved(value);
        }
        let Some(current) = region else { break };
        region = region_outer.get(&current).copied().flatten();
    }
    let display = if spelling.contains('#') {
        format!("%{name}#{number}")
    } else {
        format!("%{name}")
    };
    let diagnostic = push_diagnostic(doc, range, format!("unresolved SSA value `{display}`"));
    ValueReference::Invalid(diagnostic)
}

pub(super) fn value_type(doc: &Document, reference: ValueReference) -> Option<&str> {
    let value = match reference {
        ValueReference::Resolved(value) => value,
        ValueReference::Invalid(_) => return None,
    };
    let type_id = match value {
        ValueId::OperationResult { operation, result } => {
            let operation = doc.operation(operation)?;
            *doc.types_lists
                .get(operation.result_types)?
                .get(result as usize)?
        }
        ValueId::BlockArgument { block, argument } => {
            let block = doc.block(block)?;
            *doc.types_lists
                .get(block.argument_types)?
                .get(argument as usize)?
        }
    };
    doc.type_spelling(type_id)
}

pub(super) fn push_diagnostic(
    doc: &mut Document,
    range: TextRange,
    message: String,
) -> DiagnosticId {
    let id = DiagnosticId::new(doc.diagnostics.len(), doc.generation);
    doc.diagnostics.push(SemanticDiagnostic { range, message });
    doc.complete = false;
    id
}

pub(super) fn leading_symbol(spelling: &str) -> Option<&str> {
    let spelling = spelling.trim_start();
    let operation = if spelling.starts_with('%') {
        spelling.split_once('=')?.1
    } else {
        spelling
    };
    let mut parts = operation.split_ascii_whitespace();
    parts.next()?;
    let symbol = parts.next()?.strip_prefix('@')?;
    let end = symbol
        .bytes()
        .position(|byte| b"#: ,()={}[]".contains(&byte))
        .unwrap_or(symbol.len());
    Some(&symbol[..end])
}
