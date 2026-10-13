//! Constructed report storage and explicit native scalar conversion.
use super::{EvaluationError, NativeValue, QueryOutput, core_error, lexer, program};
use lexer::{Token, TokenKind as T};
use std::{ops::Range, sync::Arc};
use zirql::{
    Budget, ConstructedValue as V, Location, ValueExpression as BaseExpression, ValueType,
};

pub(super) fn default_limits() -> zirql::ProgramLimits {
    let limits = super::EvaluationLimits::default();
    zirql::ProgramLimits {
        work: limits.max_work,
        results: 1_000_000,
        nesting: super::parser::DEFAULT_NESTING_LIMIT,
        stream_items: limits.max_items,
    }
}

/// A lowered report expression. Cloning shares its immutable tree.
#[derive(Debug, PartialEq)]
pub struct ParsedComprehension {
    pub(super) expression: Arc<BaseExpression<super::report::Operand>>,
}
impl Clone for ParsedComprehension {
    fn clone(&self) -> Self {
        Self {
            expression: Arc::clone(&self.expression),
        }
    }
}
impl Drop for ParsedComprehension {
    fn drop(&mut self) {
        if let Some(expression) = Arc::get_mut(&mut self.expression) {
            let value = std::mem::replace(
                &mut expression.value,
                zirql::ValueExpressionKind::Literal(V::Integer(0)),
            );
            super::report::dispose_value(value);
        }
    }
}
// Only this frontend constructs the tree, and it refuses nonfinite literals.
impl Eq for ParsedComprehension {}

/// A native constructed report with its query location. Cloning shares the
/// immutable value; evaluation snapshots use Zirql's budgeted copy operation.
#[derive(Debug)]
pub struct ConstructedReport {
    value: Arc<V>,
    range: Range<usize>,
    pub(super) types: Arc<Vec<ValueType>>,
}
impl PartialEq for ConstructedReport {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.range == other.range
    }
}
impl Clone for ConstructedReport {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            range: self.range.clone(),
            types: Arc::clone(&self.types),
        }
    }
}
// Private construction checks every float before admitting this public Eq value.
impl Eq for ConstructedReport {}
impl ConstructedReport {
    pub(super) fn new(
        value: V,
        range: Range<usize>,
        budget: &mut Budget,
    ) -> Result<Self, EvaluationError> {
        let types = vec![
            zirql::constructed_value_type(&value, budget, Location::Query(range.clone()))
                .map_err(core_error)?,
        ];
        budget
            .charge(1, Location::Query(range.clone()))
            .map_err(core_error)?;
        Ok(Self {
            value: Arc::new(value),
            range,
            types: Arc::new(types),
        })
    }
    pub(super) fn with_types(
        value: V,
        types: Vec<ValueType>,
        range: Range<usize>,
        budget: &mut Budget,
    ) -> Result<Self, EvaluationError> {
        let mut result = Self::new(value, range, budget)?;
        result.types = Arc::new(types);
        Ok(result)
    }
    pub(super) fn into_value(
        self,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<V, EvaluationError> {
        match Arc::try_unwrap(self.value) {
            Ok(value) => Ok(value),
            Err(value) => {
                zirql::copy_constructed_value(&value, budget, Location::Query(range.clone()))
                    .map_err(core_error)
            }
        }
    }
    pub fn value(&self) -> &V {
        &self.value
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub(super) fn copy(
        &self,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Self, EvaluationError> {
        let value =
            zirql::copy_constructed_value(self.value(), budget, Location::Query(range.clone()))
                .map_err(core_error)?;
        let mut copied = Self::new(value, self.range.clone(), budget)?;
        copied.types = Arc::clone(&self.types);
        Ok(copied)
    }
    pub(super) fn size(&self) -> usize {
        value_size(self.value())
    }
    /// Convert the typed report to ordinary JSON under an aggregate request budget.
    pub fn to_json_with_budget(
        &self,
        budget: &mut Budget,
    ) -> Result<serde_json::Value, EvaluationError> {
        if budget.remaining_results().is_none() {
            return Err(EvaluationError::new(
                "constructed reports require an aggregate result allowance",
            )
            .at(self.range.clone()));
        }
        zirql::constructed_value_type(self.value(), budget, Location::Query(self.range.clone()))
            .map_err(core_error)?;
        to_json(self.value(), budget, &self.range, 0)
    }
    /// Render ordinary pretty JSON, spending bytes before buffering them.
    pub fn render_json_with_budget(&self, budget: &mut Budget) -> Result<String, EvaluationError> {
        let value = self.to_json_with_budget(budget)?;
        program::serialize_json(&value, budget, &self.range, true)
    }
    /// Render one compact document/result envelope with this same allowance.
    pub fn render_ndjson_with_budget(
        &self,
        document_name: &str,
        budget: &mut Budget,
    ) -> Result<String, EvaluationError> {
        let value = self.to_json_with_budget(budget)?;
        admit(budget, 4, document_name.len() + 14, &self.range)?;
        let mut envelope = serde_json::Map::new();
        envelope.insert(
            "document".to_owned(),
            serde_json::Value::String(document_name.to_owned()),
        );
        envelope.insert("result".to_owned(), value);
        program::serialize_json(
            &serde_json::Value::Object(envelope),
            budget,
            &self.range,
            false,
        )
    }
}
fn value_size(value: &V) -> usize {
    match value {
        V::Array(values)
        | V::Tuple(values)
        | V::Set {
            elements: values, ..
        } => values.iter().fold(1usize, |n, v| {
            n.saturating_add(1).saturating_add(value_size(v))
        }),
        V::Record(fields) => fields.iter().fold(1usize, |n, (_, v)| {
            n.saturating_add(1).saturating_add(value_size(v))
        }),
        V::KeyedArray { axes, entries } => {
            let axes_size = axes.iter().fold(1usize, |size, axis| {
                size.saturating_add(1).saturating_add(axis.keys.len())
            });
            entries.iter().fold(axes_size, |size, entry| {
                size.saturating_add(1)
                    .saturating_add(entry.key.len())
                    .saturating_add(value_size(&entry.value))
            })
        }
        V::Boolean(_) | V::Integer(_) | V::Float(_) | V::String(_) | V::Member(_) => 1,
    }
}

// Recognize `for binder in` at this array's own level. `[for]` stays a literal.
// The allocation-free admission scan ignores quoted strings and comments, so
// legacy names and text containing `for` retain their old parsing allowance.
pub(super) fn has_syntax(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut cursor = 0;
    let mut arrays = 0usize;
    let mut records = 0usize;
    let mut groups = 0usize;
    let mut statement_start = true;
    let mut previous_for = false;
    let mut before_previous_for = false;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'"' => {
                cursor += 1;
                while cursor < bytes.len() {
                    match bytes[cursor] {
                        b'\\' => cursor = cursor.saturating_add(2),
                        b'"' => {
                            cursor += 1;
                            break;
                        }
                        _ => cursor += 1,
                    }
                }
                previous_for = false;
                before_previous_for = false;
                statement_start = false;
            }
            b'#' => {
                while cursor < bytes.len() && bytes[cursor] != b'\n' {
                    cursor += 1;
                }
            }
            byte if byte.is_ascii_whitespace() => cursor += 1,
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = cursor;
                cursor += 1;
                while bytes
                    .get(cursor)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
                {
                    cursor += 1;
                }
                let word = &source[start..cursor];
                let mut lookahead =
                    zirql::ArgumentReader::new(source, cursor).expect("source byte boundary");
                lookahead.skip_whitespace();
                let rest = lookahead.remaining();
                if word == "project" && rest.starts_with('(') {
                    return true;
                }
                if statement_start
                    && word == "report"
                    && !rest.is_empty()
                    && !rest.starts_with(['=', '|', ';'])
                {
                    return true;
                }
                if statement_start && word == "param" {
                    let mut probe =
                        zirql::ArgumentReader::new(source, cursor).expect("source byte boundary");
                    if probe.word().is_ok() && probe.consume(':') {
                        return true;
                    }
                }
                if statement_start && word == "let" {
                    let mut probe =
                        zirql::ArgumentReader::new(source, cursor).expect("source byte boundary");
                    if probe.word().is_ok() && probe.consume('=') {
                        return true;
                    }
                }
                if matches!(word, "sum" | "length" | "forall" | "exists" | "require")
                    && rest.starts_with('(')
                {
                    return true;
                }

                if arrays > 0 && before_previous_for && word == "in" {
                    return true;
                }
                before_previous_for = previous_for;
                previous_for = word == "for";
                statement_start = false;
            }
            byte => {
                if byte == b'[' {
                    arrays += 1;
                } else if byte == b']' {
                    arrays = arrays.saturating_sub(1);
                }
                if byte == b'{' {
                    records += 1;
                } else if byte == b'}' {
                    records = records.saturating_sub(1);
                }
                if byte == b'(' {
                    groups += 1;
                } else if byte == b')' {
                    groups = groups.saturating_sub(1);
                }
                statement_start = byte == b';' && arrays == 0 && records == 0 && groups == 0;
                previous_for = false;
                before_previous_for = false;
                cursor += 1;
            }
        }
    }
    false
}

pub(super) fn starts(source: &str, tokens: &[Token], cursor: usize) -> bool {
    let mut depth = 0usize;
    let mut previous = T::LBracket;
    for (i, token) in tokens.iter().enumerate().skip(cursor) {
        match token.kind() {
            T::LBracket | T::LBrace | T::LParen => depth += 1,
            T::RBracket | T::RBrace | T::RParen => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return false;
                }
            }
            T::Eof => return false,
            T::Trivia => continue,
            T::Identifier
                if depth == 1
                    && &source[token.range().as_range()] == "for"
                    && !matches!(previous, T::LBracket | T::Comma) =>
            {
                let mut next = tokens[i + 1..].iter().filter(|t| t.kind() != T::Trivia);
                if next.next().is_some_and(|t| t.kind() == T::Identifier)
                    && next.next().is_some_and(|t| {
                        t.kind() == T::Identifier && &source[t.range().as_range()] == "in"
                    })
                {
                    return true;
                }
            }
            _ => {}
        }
        previous = token.kind();
    }
    false
}

impl ParsedComprehension {
    pub(super) fn new(expression: BaseExpression<super::report::Operand>) -> Self {
        Self {
            expression: Arc::new(expression),
        }
    }
}
fn admit(
    budget: &mut Budget,
    count: usize,
    bytes: usize,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    if count
        .checked_mul(std::mem::size_of::<V>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(EvaluationError::new("value allocation size overflow").at(range.clone()));
    }
    budget
        .charge(
            count.checked_add(bytes).ok_or_else(|| {
                EvaluationError::new("value allocation size overflow").at(range.clone())
            })?,
            Location::Query(range.clone()),
        )
        .map_err(core_error)?;
    budget
        .charge_results(count, Location::Query(range.clone()))
        .map_err(core_error)
}
pub(super) fn project(
    value: &QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<V, EvaluationError> {
    match value {
        QueryOutput::Native(NativeValue::Constructed(value)) => {
            zirql::copy_constructed_value(value.value(), budget, Location::Query(range.clone()))
                .map_err(core_error)
        }
        QueryOutput::Values(values) => {
            admit(budget, 1 + values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                admit(budget, 1, value.len(), range)?;
                result.push(V::String(value.clone()));
            }
            Ok(V::Array(result))
        }
        QueryOutput::Count(count) => {
            zirql::count_value(*count, budget, Location::Query(range.clone())).map_err(core_error)
        }
        QueryOutput::Native(NativeValue::String(value)) => {
            admit(budget, 1, value.len(), range)?;
            Ok(V::String(value.clone()))
        }
        QueryOutput::Array(values) => {
            admit(budget, 1 + values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                result.push(project_json(value, budget, range, 0)?);
            }
            Ok(V::Array(result))
        }
        QueryOutput::Map(fields) => {
            admit(budget, 1 + fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project_json(value, budget, range, 0)?));
            }
            Ok(V::Record(result))
        }
        QueryOutput::RankedMap(fields) => {
            admit(budget, 1 + fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project_json(value, budget, range, 0)?));
            }
            Ok(V::Record(result))
        }
        QueryOutput::Native(NativeValue::Map(fields)) => {
            admit(budget, 1 + fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project(value, budget, range)?));
            }
            Ok(V::Record(result))
        }
        _ => Err(EvaluationError::new(
            "native handles and rendered output require an explicit scalar projection",
        )
        .at(range.clone())),
    }
}
fn project_json(
    value: &serde_json::Value,
    budget: &mut Budget,
    range: &Range<usize>,
    depth: usize,
) -> Result<V, EvaluationError> {
    if depth >= 64 {
        return Err(EvaluationError::new("value nesting limit exceeded").at(range.clone()));
    }
    admit(budget, 1, 0, range)?;
    Ok(match value {
        serde_json::Value::Null => {
            return Err(
                EvaluationError::new("null is unsupported in positional comprehensions")
                    .at(range.clone()),
            );
        }
        serde_json::Value::Bool(v) => V::Boolean(*v),
        serde_json::Value::Number(v) => {
            if v.is_i64() || v.is_u64() {
                V::Integer(v.as_i64().ok_or_else(|| {
                    EvaluationError::new("integer exceeds signed 64-bit range").at(range.clone())
                })?)
            } else {
                V::Float(
                    v.as_f64()
                        .ok_or_else(|| EvaluationError::new("nonfinite float").at(range.clone()))?,
                )
            }
        }
        serde_json::Value::String(v) => {
            admit(budget, 0, v.len(), range)?;
            V::String(v.clone())
        }
        serde_json::Value::Array(values) => {
            admit(budget, values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                result.push(project_json(value, budget, range, depth + 1)?);
            }
            V::Array(result)
        }
        serde_json::Value::Object(fields) => {
            admit(budget, fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project_json(value, budget, range, depth + 1)?));
            }
            V::Record(result)
        }
    })
}
fn to_json(
    value: &V,
    budget: &mut Budget,
    range: &Range<usize>,
    depth: usize,
) -> Result<serde_json::Value, EvaluationError> {
    if depth >= 64 {
        return Err(EvaluationError::new("value nesting limit exceeded").at(range.clone()));
    }
    admit(budget, 1, 0, range)?;
    Ok(match value {
        V::Member(_) | V::KeyedArray { .. } => {
            return Err(EvaluationError::new(
                "nominal and keyed values are unsupported in Zirium reports",
            )
            .at(range.clone()));
        }
        V::Boolean(v) => serde_json::Value::Bool(*v),
        V::Integer(v) => serde_json::Value::Number((*v).into()),
        V::Float(v) => serde_json::Value::Number(
            serde_json::Number::from_f64(*v).expect("private finite report invariant"),
        ),
        V::String(v) => {
            admit(budget, 0, v.len(), range)?;
            serde_json::Value::String(v.clone())
        }
        V::Array(values)
        | V::Tuple(values)
        | V::Set {
            elements: values, ..
        } => {
            admit(budget, values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                result.push(to_json(value, budget, range, depth + 1)?);
            }
            serde_json::Value::Array(result)
        }
        V::Record(fields) => {
            admit(budget, fields.len(), 0, range)?;
            let mut result = serde_json::Map::new();
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.insert(key.clone(), to_json(value, budget, range, depth + 1)?);
            }
            serde_json::Value::Object(result)
        }
    })
}
