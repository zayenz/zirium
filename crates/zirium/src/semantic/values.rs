mod affine;
mod attributes;
mod locations;
mod types;

use affine::lower_affine_attribute;
use attributes::resolve_attribute;
pub(super) use attributes::{
    dense_integer_literal_is_valid, lower_attribute_value, lower_dictionary,
};
pub(super) use locations::lower_location_value;
use locations::parse_location;
pub(super) use types::intern_type;
use types::{lower_type_value_with_stack, resolve_affine_alias, resolve_type};

use super::lowering::Interner;
use std::borrow::Cow;

#[derive(Debug)]
pub(super) struct AliasExpansionState {
    limit: usize,
    active: HashSet<String>,
    limit_exceeded: bool,
}

impl AliasExpansionState {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit,
            active: HashSet::new(),
            limit_exceeded: false,
        }
    }

    fn enter(&mut self, alias: &str, family: &str) -> Result<(), String> {
        if self.active.contains(alias) {
            return Err(format!("cyclic {family} alias `{alias}`"));
        }
        if self.active.len() >= self.limit {
            self.limit_exceeded = true;
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

    fn diagnostic_code(&self, fallback: SemanticDiagnosticCode) -> SemanticDiagnosticCode {
        if self.limit_exceeded {
            SemanticDiagnosticCode::ResourceLimit
        } else {
            fallback
        }
    }
}

use super::*;
use crate::{
    lexer::{TokenKind, lex},
    source::Source,
};

pub(super) fn parse_symbol_path(spelling: &str) -> Option<Vec<String>> {
    let source = Source::new(spelling.as_bytes().to_vec()).ok()?;
    let lexed = lex(&source);
    if !lexed.diagnostics().is_empty() {
        return None;
    }
    let tokens = lexed
        .tokens()
        .iter()
        .copied()
        .filter(|token| {
            !matches!(
                token.kind(),
                TokenKind::Whitespace | TokenKind::LineComment | TokenKind::Eof
            )
        })
        .collect::<Vec<_>>();
    if tokens.len() % 3 != 1 {
        return None;
    }
    let mut path = Vec::with_capacity(tokens.len() / 3 + 1);
    for (index, token) in tokens.iter().enumerate() {
        let expected = if index % 3 == 0 {
            TokenKind::AtIdentifier
        } else {
            TokenKind::Colon
        };
        if token.kind() != expected {
            return None;
        }
        if expected == TokenKind::AtIdentifier {
            let component = text(source.bytes(), token.range()).strip_prefix('@')?;
            path.push(decode_mlir_string(component).unwrap_or_else(|| component.to_owned()));
        }
    }
    (!path.is_empty()).then_some(path)
}

fn bare_symbol_component(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'.'))
}

fn quote_mlir_string(value: &str) -> String {
    let mut result = String::with_capacity(value.len() + 2);
    result.push('"');
    for character in value.chars() {
        match character {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\t' => result.push_str("\\t"),
            character if character.is_control() => {
                let mut bytes = [0; 4];
                for byte in character.encode_utf8(&mut bytes).bytes() {
                    use std::fmt::Write as _;
                    write!(result, "\\{byte:02X}").expect("writing to a String cannot fail");
                }
            }
            character => result.push(character),
        }
    }
    result.push('"');
    result
}

fn format_symbol_component(value: &str) -> String {
    if bare_symbol_component(value) {
        value.to_owned()
    } else {
        quote_mlir_string(value)
    }
}

pub(crate) fn format_symbol_path(path: &[String], sigils: bool) -> String {
    path.iter()
        .map(|component| {
            let component = format_symbol_component(component);
            if sigils {
                format!("@{component}")
            } else {
                component
            }
        })
        .collect::<Vec<_>>()
        .join("::")
}

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
    let output = split_arrow(spelling)
        .map(|(_, output)| output.trim())
        .unwrap_or("");
    split_types(output)
}

fn split_types(spelling: &str) -> Vec<String> {
    let spelling = spelling.trim();
    let inner = spelling.strip_prefix('(').and_then(|value| {
        let closing = value.len().checked_sub(1)?;
        (matching_delimiter(value, '(', ')') == Some(closing)).then_some(&value[..closing])
    });
    let inner = inner.unwrap_or(spelling);
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
        if character == '>' && value.as_bytes().get(index.wrapping_sub(1)) == Some(&b'-') {
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

fn strip_dictionary_trivia(mut value: &str) -> &str {
    loop {
        value = value.trim_start();
        let Some(comment) = value.strip_prefix("//") else {
            return value;
        };
        let Some(newline) = comment.find('\n') else {
            return "";
        };
        value = &comment[newline + 1..];
    }
}

fn split_dictionary_entry(value: &str) -> (Cow<'_, str>, Option<&str>) {
    let value = value.trim_start();
    let key_end = if value.starts_with('"') {
        let mut escaped = false;
        value
            .bytes()
            .enumerate()
            .skip(1)
            .find_map(|(index, byte)| {
                if escaped {
                    escaped = false;
                    None
                } else if byte == b'\\' {
                    escaped = true;
                    None
                } else if byte == b'"' {
                    Some(index + 1)
                } else {
                    None
                }
            })
            .unwrap_or(value.len())
    } else {
        value
            .char_indices()
            .find_map(|(index, character)| {
                (character.is_whitespace() || character == '=' || value[index..].starts_with("//"))
                    .then_some(index)
            })
            .unwrap_or(value.len())
    };
    let spelling = &value[..key_end];
    let name = decode_mlir_string(spelling)
        .map(Cow::Owned)
        .unwrap_or(Cow::Borrowed(spelling));
    let remainder = strip_dictionary_trivia(&value[key_end..]);
    (name, remainder.strip_prefix('='))
}

fn split_dictionary_entries(value: &str) -> Vec<&str> {
    let value = value.trim();
    if value.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut quoted = false;
    let mut escaped = false;
    let mut line_comment = false;
    let bytes = value.as_bytes();
    for (index, byte) in bytes.iter().copied().enumerate() {
        if line_comment {
            if byte == b'\n' {
                line_comment = false;
            }
            continue;
        }
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
        if byte == b'/' && bytes.get(index + 1) == Some(&b'/') {
            line_comment = true;
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'>' if index == 0 || bytes[index - 1] != b'-' => depth -= 1,
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

pub(crate) fn split_arrow(value: &str) -> Option<(&str, &str)> {
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
            b')' | b']' | b'}' => depth -= 1,
            b'>' if i == 0 || bytes[i - 1] != b'-' => depth -= 1,
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
            b')' | b']' | b'}' => depth -= 1,
            b'>' if i == 0 || value.as_bytes()[i - 1] != b'-' => depth -= 1,
            b'x' if depth == 0
                && value.as_bytes()[..i]
                    .iter()
                    .rev()
                    .find(|byte| !byte.is_ascii_whitespace())
                    .is_some_and(|byte| {
                        byte.is_ascii_digit() || matches!(byte, b'?' | b'*' | b']')
                    }) =>
            {
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
            b')' | b']' | b'}' => depth -= 1,
            b'>' if index == 0 || value.as_bytes()[index - 1] != b'-' => depth -= 1,
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
    region_parent_blocks: &HashMap<RegionId, BlockId>,
    doc: &mut Document,
) -> ValueReference {
    let name = first_identifier(spelling, b'%').unwrap_or_default();
    let number = spelling
        .split_once('#')
        .and_then(|(_, number)| number.trim().split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|number| number.parse::<usize>().ok())
        .unwrap_or(0);
    if let Some(block) = block
        && let Some(value) = block_definitions
            .get(&(block, name.clone()))
            .and_then(|values| values.get(number))
            .copied()
    {
        return ValueReference::Resolved(value);
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
        if let Some(value) = region_parent_blocks
            .get(&current)
            .and_then(|block| block_definitions.get(&(*block, name.clone())))
            .and_then(|values| values.get(number))
            .copied()
        {
            return ValueReference::Resolved(value);
        }
        region = region_outer.get(&current).copied().flatten();
    }
    let display = if spelling.contains('#') {
        format!("%{name}#{number}")
    } else {
        format!("%{name}")
    };
    let diagnostic = push_diagnostic(
        doc,
        SemanticDiagnosticCode::UnresolvedReference,
        range,
        format!("unresolved SSA value `{display}`"),
    );
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
    code: SemanticDiagnosticCode,
    range: TextRange,
    message: String,
) -> DiagnosticId {
    let id = DiagnosticId::new(doc.diagnostics.len(), doc.generation);
    doc.diagnostics
        .push(SemanticDiagnostic::new(code, range, message));
    doc.complete = false;
    id
}
