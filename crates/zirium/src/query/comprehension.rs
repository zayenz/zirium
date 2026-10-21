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
        Self::from_shared(Arc::new(value), types, range, budget)
    }
    // Core execution has already admitted these immutable data and prepared facts.
    pub(super) fn from_shared(
        value: Arc<V>,
        types: Vec<ValueType>,
        range: Range<usize>,
        budget: &mut Budget,
    ) -> Result<Self, EvaluationError> {
        budget
            .charge(1, Location::Query(range.clone()))
            .map_err(core_error)?;
        Ok(Self {
            value,
            range,
            types: Arc::new(types),
        })
    }
    pub(super) fn shared(&self) -> Arc<V> {
        Arc::clone(&self.value)
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
    /// Convert with Zirql's ordinary codec under the active aggregate budget.
    /// Nominal members and keyed arrays use tagged fallback to retain their shape.
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
        zirql::to_json_value(
            self.value(),
            zirql::JsonRepresentation::Ordinary,
            budget,
            Location::Query(self.range.clone()),
        )
        .map_err(core_error)
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
        V::Declared { value, .. } => 1usize.saturating_add(value_size(value)),
        V::Absent => 1,
        V::Array(values) | V::Tuple(values) => sequence_size(values),
        V::Set { elements, .. } => sequence_size(elements),
        V::Dictionary(entries) => entries.iter().fold(1usize, |size, (key, value)| {
            size.saturating_add(1)
                .saturating_add(value_size(key))
                .saturating_add(value_size(value))
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

fn sequence_size(values: &[V]) -> usize {
    values.iter().fold(1usize, |size, value| {
        size.saturating_add(1).saturating_add(value_size(value))
    })
}

pub(super) fn constructed_filter_call(source: &str, position: usize) -> bool {
    let Ok(mut reader) = zirql::ArgumentReader::new(source, position) else {
        return false;
    };
    reader.word().is_ok_and(|word| word.value == "filter")
        && reader.consume('(')
        && reader.word().is_ok_and(|word| word.value == "starts_with")
        && reader.consume('(')
}

// Native stage precedence and legacy JSON/interpolation stay in this adapter;
// the reader owns shared expression and ordering recognition.
pub(super) fn starts(
    source: &str,
    tokens: &[Token],
    cursor: usize,
    budget: &mut Budget,
    nesting: usize,
) -> Result<bool, zirql::QueryError> {
    let Some(token) = tokens.get(cursor) else {
        return Ok(false);
    };
    if let Some(first) = tokens[cursor..]
        .iter()
        .find(|token| !matches!(token.kind(), T::Trivia | T::LParen))
        && first.kind() == T::Identifier
        && super::parser::is_native_stage(&source[first.range().as_range()])
    {
        let reader = zirql::ArgumentReader::new(source, first.range().start() as usize)?;
        if !zirql::probe_ordering_call(&reader, budget, nesting)? {
            return Ok(false);
        }
    }
    let reader = zirql::ArgumentReader::new(source, token.range().start() as usize)?;
    zirql::starts_common_expression(&reader, budget, nesting)
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
        QueryOutput::Native(NativeValue::Computational(_)) => Err(EvaluationError::new(
            "computational values cannot be projected to report data",
        )
        .at(range.clone())),
        QueryOutput::Native(NativeValue::JsonValue(value)) => project_json(value, budget, range, 0),
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
        serde_json::Value::Null => V::Absent,
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
