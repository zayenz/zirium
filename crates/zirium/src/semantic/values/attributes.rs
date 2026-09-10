//! Attribute and dictionary lowering, including dense payload validation.

use super::*;

pub(super) fn resolve_attribute(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<AttributeValue, String> {
    let spelling = spelling.trim();
    if spelling == "true" || spelling == "false" {
        return Ok(AttributeValue::Boolean(spelling == "true"));
    }
    if spelling == "unit" {
        return Ok(AttributeValue::Opaque(Arc::from(b"unit".as_slice())));
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
    if spelling.starts_with('#') && !spelling.contains(['.', '<']) {
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
        return parse_symbol_path(spelling)
            .map(AttributeValue::Symbol)
            .ok_or_else(|| format!("malformed symbol reference `{spelling}`"));
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
        let mut entries = split_dictionary_entries(inner)
            .into_iter()
            .map(|entry| {
                let (name, value) = split_dictionary_entry(entry);
                Ok((
                    name.into_owned(),
                    resolve_attribute(
                        value.unwrap_or("unit"),
                        type_aliases,
                        attribute_aliases,
                        stack,
                    )?,
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
    if literal.parse::<i128>().is_ok() {
        return Ok(AttributeValue::Integer(compact(spelling)));
    }
    if is_valid_wide_number(spelling) {
        return Ok(AttributeValue::WideNumber(Arc::from(spelling.as_bytes())));
    }
    if is_valid_hex_float_attribute(spelling) {
        return Ok(AttributeValue::Float(compact(spelling)));
    }
    if literal.parse::<f64>().is_ok() {
        return Ok(AttributeValue::Float(compact(spelling)));
    }
    if spelling.starts_with('#') {
        if spelling.contains('<') {
            let Some(open) = spelling.find('<') else {
                unreachable!()
            };
            let Some(close) = matching_delimiter(&spelling[open + 1..], '<', '>') else {
                return Err(format!("malformed opaque attribute `{spelling}`"));
            };
            if !spelling[open + 1 + close + 1..].trim().is_empty() {
                return Err(format!(
                    "trailing garbage after opaque attribute `{spelling}`"
                ));
            }
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

fn is_valid_hex_float_attribute(value: &str) -> bool {
    let Some((literal, suffix)) = value.split_once(':') else {
        return false;
    };
    let Some(digits) = literal.trim().strip_prefix("0x") else {
        return false;
    };
    let expected_digits = match suffix.trim() {
        "f16" | "bf16" => 4,
        "f32" => 8,
        "f64" => 16,
        _ => return false,
    };
    digits.len() == expected_digits && digits.bytes().all(|byte| byte.is_ascii_hexdigit())
}

// Attribute and property dictionaries share this lowering path, whose inputs are
// deliberately passed separately to keep the surrounding lowering state local.
#[allow(clippy::too_many_arguments)]
pub(in crate::semantic) fn lower_dictionary(
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
            let (name, value) = split_dictionary_entry(spelling);
            let implicit_unit = value.is_none();
            let value = value.unwrap_or("");
            let name_id = strings.intern(name.as_ref());
            let duplicate = seen.insert(name_id, attribute_range).map(|previous| {
                push_diagnostic(
                    doc,
                    SemanticDiagnosticCode::DuplicateDefinition,
                    attribute_range,
                    format!(
                        "duplicate {kind} key `{}` (previous at {})",
                        name,
                        previous.start()
                    ),
                )
            });
            let value_spelling = if implicit_unit { "unit" } else { value.trim() };
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
            let semantic = if malformed_numeric_prefix
                || (owned_payload_candidate && tree.has_error(attribute).unwrap_or(false))
            {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    SemanticDiagnosticCode::Attribute,
                    attribute_range,
                    "malformed attribute value".into(),
                ))
            } else if kind == "inherent attribute"
                && matches!(name.as_ref(), "contracting_dims" | "batching_dims")
                && tree
                    .children(attribute)
                    .into_iter()
                    .flatten()
                    .any(|child| tree.kind(child) == Some(SyntaxKind::OpaqueAttribute))
            {
                // A paired custom clause prints generically as two arrays;
                // keep its original spelling for inspection.
                let mut expansion = AliasExpansionState::new(doc.alias_expansion_depth_limit);
                AttributeValue::Array(
                    tree.children(attribute)
                        .into_iter()
                        .flatten()
                        .filter(|&child| tree.kind(child) == Some(SyntaxKind::OpaqueAttribute))
                        .flat_map(|child| tree.children(child).into_iter().flatten())
                        .filter(|&child| tree.kind(child) == Some(SyntaxKind::ArrayAttribute))
                        .filter_map(|child| tree.text_range(child))
                        .map(|range| {
                            lower_attribute_value(
                                text(bytes, range),
                                range,
                                type_aliases,
                                attribute_aliases,
                                &mut expansion,
                                doc,
                            )
                        })
                        .collect(),
                )
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
            } else if value_spelling.is_empty() {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    SemanticDiagnosticCode::Attribute,
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

pub(in crate::semantic) fn lower_attribute_value(
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
    if spelling.starts_with('"') {
        return AttributeValue::String(spelling.to_owned());
    }
    if let Some(inner) = angle_inner(spelling, "array") {
        if depth >= doc.attribute_depth_limit {
            return AttributeValue::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::ResourceLimit,
                range,
                "attribute nesting depth limit exceeded".into(),
            ));
        }
        return match parse_dense_array(inner) {
            Ok((element_type, elements)) => AttributeValue::DenseArray {
                element_type,
                elements,
            },
            Err(message) => AttributeValue::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::Attribute,
                range,
                message,
            )),
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
                return AttributeValue::Invalid(push_diagnostic(
                    doc,
                    stack.diagnostic_code(SemanticDiagnosticCode::Attribute),
                    range,
                    message,
                ));
            }
            Ok(None) => {}
        }
    }
    if let Some(inner) = bracket_inner(spelling, '[', ']') {
        if depth >= doc.attribute_depth_limit {
            return AttributeValue::Invalid(push_diagnostic(
                doc,
                SemanticDiagnosticCode::ResourceLimit,
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
                SemanticDiagnosticCode::ResourceLimit,
                range,
                "attribute nesting depth limit exceeded".into(),
            ));
        }
        let mut seen = HashMap::new();
        let mut entries = split_dictionary_entries(inner)
            .into_iter()
            .map(|entry| {
                let (name, value) = split_dictionary_entry(entry);
                let name = name.into_owned();
                let duplicate = seen.insert(name.clone(), ()).is_some();
                let value = lower_attribute_value_with_depth(
                    value.unwrap_or("unit"),
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
                        SemanticDiagnosticCode::DuplicateDefinition,
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
        Err(message) => AttributeValue::Invalid(push_diagnostic(
            doc,
            stack.diagnostic_code(SemanticDiagnosticCode::Attribute),
            range,
            message,
        )),
    }
}

fn parse_dense_array(inner: &str) -> Result<(String, Vec<AttributeValue>), String> {
    let inner = without_line_comments(inner);
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
                if !dense_integer_literal_is_valid(element_type, item) {
                    return Err(format!(
                        "dense array element `{item}` does not match `{element_type}`"
                    ));
                }
                AttributeValue::Integer(compact(item))
            } else {
                if !dense_float_literal_is_valid(element_type, item) {
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

fn without_line_comments(value: &str) -> String {
    value
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(in crate::semantic) fn dense_integer_literal_is_valid(
    element_type: &str,
    literal: &str,
) -> bool {
    let width = match element_type {
        "i8" => 8,
        "i16" => 16,
        "i32" => 32,
        "i64" => 64,
        _ => return false,
    };
    if let Some(digits) = literal.strip_prefix("0x") {
        return u128::from_str_radix(digits, 16).is_ok_and(|value| value < (1u128 << width));
    }
    literal.parse::<i128>().is_ok_and(|value| {
        let signed_min = -(1i128 << (width - 1));
        let unsigned_max = (1i128 << width) - 1;
        (signed_min..=unsigned_max).contains(&value)
    })
}

fn dense_float_literal_is_valid(element_type: &str, literal: &str) -> bool {
    if let Some(digits) = literal.strip_prefix("0x") {
        let width = match element_type {
            "f32" => 32,
            "f64" => 64,
            _ => return false,
        };
        return u128::from_str_radix(digits, 16).is_ok_and(|value| value < (1u128 << width));
    }
    literal.parse::<f64>().is_ok()
}
