//! Composable operation-selection queries, editing, and output.

use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fmt,
};

use crate::{
    dialect::DialectRegistry,
    semantic::{
        AttributeSpec, AttributeValue, CfBrOp, CfCondBrOp, Document, OperationId, Successor,
        UseSite, ValueId, ValueReference, decode_mlir_string,
    },
};

pub mod lexer;
mod literal;
pub mod parser;
mod render;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    expression: parser::Expression,
    statements: Vec<parser::Statement>,
    emit_final_result: bool,
}

/// Bounds evaluation work and the number of items in any one stream.
/// Work counts stage inputs and dependency/subtree visits, not wall-clock time.
#[derive(Clone, Copy, Debug)]
pub struct EvaluationLimits {
    pub max_work: usize,
    pub max_items: usize,
}

impl Default for EvaluationLimits {
    fn default() -> Self {
        Self {
            max_work: 10_000_000,
            max_items: 1_000_000,
        }
    }
}

struct EvaluationState {
    bindings: BTreeMap<String, QueryOutput>,
    remaining: usize,
    max_items: usize,
}

impl EvaluationState {
    fn charge(&mut self, work: usize) -> Result<(), EvaluationError> {
        self.remaining = self.remaining.checked_sub(work).ok_or_else(|| EvaluationError::new(
            "query work limit exceeded; fixed points may need unique or a larger --max-work limit"
        ))?;
        Ok(())
    }

    fn check_items(&self, count: usize) -> Result<(), EvaluationError> {
        if count > self.max_items {
            return Err(EvaluationError::new(
                "query stream size limit exceeded; use unique or a larger --max-items limit",
            ));
        }
        Ok(())
    }

    fn collect<T>(&self, values: impl Iterator<Item = T>) -> Result<Vec<T>, EvaluationError> {
        let values: Vec<_> = values.take(self.max_items.saturating_add(1)).collect();
        self.check_items(values.len())?;
        Ok(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryOutput {
    Operations(Vec<OperationId>),
    Values(Vec<String>),
    Count(usize),
    Map(serde_json::Map<String, serde_json::Value>),
    Array(Vec<serde_json::Value>),
    Json(String),
    Text(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SortKey {
    Count(usize),
    Text(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryError {
    pub position: usize,
    pub message: &'static str,
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "query error at byte {}: {}", self.position, self.message)
    }
}

impl std::error::Error for QueryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationError {
    message: String,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EvaluationError {}

impl EvaluationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Query {
    pub fn parse(source: &str) -> Result<Self, QueryError> {
        let lexed = lexer::lex(source);
        let parsed = parser::parse(&lexed);
        let lexical = lexed.diagnostics().first().map(|diagnostic| {
            let message = match diagnostic.kind() {
                lexer::DiagnosticKind::QueryTooLarge => "query exceeds the supported size",
                lexer::DiagnosticKind::InvalidToken => "invalid token",
                lexer::DiagnosticKind::InvalidEscape => "unsupported string escape",
                lexer::DiagnosticKind::UnterminatedString => "unterminated string",
            };
            (diagnostic.range(), message)
        });
        let syntactic = parsed
            .diagnostics()
            .first()
            .map(|diagnostic| (diagnostic.range(), diagnostic.message()));
        if let Some((range, message)) = [lexical, syntactic]
            .into_iter()
            .flatten()
            .min_by_key(|(range, _)| range.start())
        {
            return Err(QueryError {
                position: range.start() as usize,
                message,
            });
        }
        let program = parsed
            .into_program()
            .expect("diagnostic-free query has a program");
        Ok(Self {
            expression: program.expression,
            statements: program.statements,
            emit_final_result: program.emit_final_result,
        })
    }

    /// Evaluates against the current document, invoking `emit` at each explicit
    /// emission and for the implicit final output unless it is suppressed by a
    /// final `do` statement. The callback sees each edit as it exists at that
    /// point. Callers that need all-or-nothing output should buffer emissions
    /// until evaluation succeeds. Edits commit per stage.
    pub fn evaluate(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_limits(document, registry, EvaluationLimits::default(), emit)
    }

    pub fn evaluate_with_limits(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        limits: EvaluationLimits,
        mut emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        let mut budget = EvaluationState {
            bindings: BTreeMap::new(),
            remaining: limits.max_work,
            max_items: limits.max_items,
        };
        let input = QueryOutput::Operations(budget.collect(document.operations())?);
        for statement in &self.statements {
            let expression = match statement {
                parser::Statement::Binding { expression, .. }
                | parser::Statement::Do(expression)
                | parser::Statement::Query(expression) => expression,
            };
            let output = evaluate_expression(
                expression,
                document,
                registry,
                input.clone(),
                &mut emit,
                &mut budget,
            )?;
            match statement {
                parser::Statement::Binding { name, .. } => {
                    budget.bindings.insert(name.clone(), output);
                }
                parser::Statement::Do(_) => {}
                parser::Statement::Query(_) if !expression.ends_with_emission() => {
                    emit(document, output)?
                }
                parser::Statement::Query(_) => {}
            }
        }
        let output = evaluate_expression(
            &self.expression,
            document,
            registry,
            input,
            &mut emit,
            &mut budget,
        )?;
        if self.emit_final_result && !self.expression.ends_with_emission() {
            emit(document, output)?;
        }
        Ok(())
    }
}

fn evaluate_expression(
    expression: &parser::Expression,
    document: &mut Document,
    registry: &DialectRegistry,
    input: QueryOutput,
    emit: &mut impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    budget: &mut EvaluationState,
) -> Result<QueryOutput, EvaluationError> {
    if expression.rest.is_empty() {
        return evaluate_pipeline(&expression.first, document, registry, input, emit, budget);
    }
    let first = evaluate_pipeline(
        &expression.first,
        document,
        registry,
        input.clone(),
        emit,
        budget,
    )?;
    let mut selected = first;
    for (operator, stages) in &expression.rest {
        let right = evaluate_pipeline(stages, document, registry, input.clone(), emit, budget)?;
        selected = evaluate_set_operator(document, *operator, selected, right)?;
        budget.check_items(output_len(&selected))?;
    }
    Ok(selected)
}

fn evaluate_pipeline(
    stages: &[parser::Stage],
    document: &mut Document,
    registry: &DialectRegistry,
    input: QueryOutput,
    emit: &mut impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    budget: &mut EvaluationState,
) -> Result<QueryOutput, EvaluationError> {
    let mut current = input;
    for stage in stages {
        budget.charge(output_size(&current).max(1))?;
        match stage {
            parser::Stage::Literal { value, .. } => {
                let value = literal::evaluate(value, document, budget, &mut 0)?;
                if value_depth(&value) > parser::DEFAULT_NESTING_LIMIT {
                    return Err(EvaluationError::new("JSON nesting limit exceeded"));
                }
                current = match value {
                    serde_json::Value::Object(entries) => QueryOutput::Map(entries),
                    serde_json::Value::Array(values) => QueryOutput::Array(values),
                    _ => unreachable!("literal stages start with an object or array"),
                };
            }
            parser::Stage::Binding { name, .. } => {
                budget.charge(output_size(&budget.bindings[name]).max(1))?;
                current = budget.bindings[name].clone();
            }
            parser::Stage::Tally { .. } => {
                let QueryOutput::Values(values) = current else {
                    return Err(EvaluationError::new(
                        "tally requires a value stream; use names or attr first",
                    ));
                };
                let mut counts = BTreeMap::<String, usize>::new();
                for value in values {
                    *counts.entry(value).or_default() += 1;
                }
                current = QueryOutput::Map(
                    counts
                        .into_iter()
                        .map(|(key, count)| (key, serde_json::json!(count)))
                        .collect(),
                );
            }
            parser::Stage::MapBy { key, value, .. } => {
                let selected = take_operations(current, "map_by")?;
                let mut entries = serde_json::Map::new();
                let mut items = 0usize;
                for operation in selected {
                    let item = QueryOutput::Operations(vec![operation]);
                    let key =
                        evaluate_expression(key, document, registry, item.clone(), emit, budget)?;
                    let QueryOutput::Values(mut keys) = key else {
                        return Err(EvaluationError::new(
                            "map_by key must produce exactly one string",
                        ));
                    };
                    if keys.len() != 1 {
                        return Err(EvaluationError::new(
                            "map_by key must produce exactly one string",
                        ));
                    }
                    let key = keys.pop().unwrap();
                    if entries.contains_key(&key) {
                        return Err(EvaluationError::new(format!(
                            "map_by encountered duplicate key `{key}`; select unique keys"
                        )));
                    }
                    let value = evaluate_expression(value, document, registry, item, emit, budget)?;
                    items = items.saturating_add(output_size(&value).saturating_add(1));
                    budget.check_items(items)?;
                    budget.charge(output_size(&value))?;
                    let value = output_value(document, &value);
                    if value_depth(&value) >= parser::DEFAULT_NESTING_LIMIT {
                        return Err(EvaluationError::new("map nesting limit exceeded"));
                    }
                    entries.insert(key, value);
                }
                current = QueryOutput::Map(entries);
            }
            parser::Stage::Reachable { .. } => {
                current = QueryOutput::Operations(evaluate_reachable(
                    document,
                    take_operations(current, "reachable")?,
                    registry,
                    budget,
                )?);
            }
            parser::Stage::Input { .. } => {
                current = QueryOutput::Operations(budget.collect(document.operations())?)
            }
            parser::Stage::Filter { predicate, .. } => {
                let selected = operations_mut(&mut current, "filter")?;
                selected.retain(|&operation| evaluate_predicate(predicate, document, operation));
            }
            parser::Stage::Closure { .. } => {
                let selected = take_operations(current, "closure")?;
                current = QueryOutput::Operations(evaluate_closure(
                    document, selected, registry, false, budget,
                )?);
            }
            parser::Stage::Slice { .. } => {
                let selected = take_operations(current, "slice")?;
                current = QueryOutput::Operations(evaluate_slice(document, selected, budget)?);
            }
            parser::Stage::Defs { index, .. } => {
                let selected = take_operations(current, "defs")?;
                current =
                    QueryOutput::Operations(evaluate_defs(document, &selected, *index, budget)?);
            }
            parser::Stage::Users { index, .. } => {
                let selected = take_operations(current, "users")?;
                current =
                    QueryOutput::Operations(evaluate_users(document, &selected, *index, budget)?);
            }
            parser::Stage::Parent { .. } => {
                let selected = take_operations(current, "parent")?;
                current = QueryOutput::Operations(evaluate_parent(document, &selected, budget)?);
            }
            parser::Stage::Children { .. } => {
                let selected = take_operations(current, "children")?;
                current = QueryOutput::Operations(evaluate_children(document, &selected, budget)?);
            }
            parser::Stage::Root { predicate, .. } => {
                let selected = take_operations(current, "root")?;
                current = QueryOutput::Operations(evaluate_root(document, &selected, predicate));
            }
            parser::Stage::Subtree { .. } => {
                let selected = take_operations(current, "subtree")?;
                let mut expanded = Vec::new();
                for operation in selected {
                    append_subtree(document, operation, &mut expanded, budget)?;
                }
                current = QueryOutput::Operations(expanded);
            }
            parser::Stage::Unique { .. } => match &mut current {
                QueryOutput::Operations(selected) => retain_unique(selected),
                QueryOutput::Values(values) => retain_unique(values),
                _ => {
                    return Err(EvaluationError::new(
                        "unique requires an operation or value stream",
                    ));
                }
            },
            parser::Stage::Sort { .. } => {
                let QueryOutput::Values(values) = &mut current else {
                    return Err(EvaluationError::new(
                        "sort requires a value stream; use names or attr first",
                    ));
                };
                values.sort();
            }
            parser::Stage::SortBy { selector, .. } => {
                let selected = take_operations(current, "sort_by")?;
                current = QueryOutput::Operations(sort_operations_by(
                    selected, selector, document, registry, emit, budget, false, None,
                )?);
            }
            parser::Stage::Reverse { .. } => match &mut current {
                QueryOutput::Operations(values) => values.reverse(),
                QueryOutput::Values(values) => values.reverse(),
                _ => {
                    return Err(EvaluationError::new(
                        "reverse requires an operation or value stream",
                    ));
                }
            },
            parser::Stage::Head { count, .. } => truncate_stream(&mut current, *count, false)?,
            parser::Stage::Tail { count, .. } => truncate_stream(&mut current, *count, true)?,
            parser::Stage::Min { .. } => {
                current = QueryOutput::Values(extreme_value(current, false, false)?);
            }
            parser::Stage::Max { .. } => {
                current = QueryOutput::Values(extreme_value(current, true, false)?);
            }
            parser::Stage::MinAll { .. } => {
                current = QueryOutput::Values(extreme_value(current, false, true)?);
            }
            parser::Stage::MaxAll { .. } => {
                current = QueryOutput::Values(extreme_value(current, true, true)?);
            }
            parser::Stage::MinBy { selector, .. } => {
                let selected = take_operations(current, "min_by")?;
                current = QueryOutput::Operations(sort_operations_by(
                    selected,
                    selector,
                    document,
                    registry,
                    emit,
                    budget,
                    false,
                    Some(false),
                )?);
            }
            parser::Stage::MaxBy { selector, .. } => {
                let selected = take_operations(current, "max_by")?;
                current = QueryOutput::Operations(sort_operations_by(
                    selected,
                    selector,
                    document,
                    registry,
                    emit,
                    budget,
                    true,
                    Some(false),
                )?);
            }
            parser::Stage::MinAllBy { selector, .. } => {
                let selected = take_operations(current, "min_all_by")?;
                current = QueryOutput::Operations(sort_operations_by(
                    selected,
                    selector,
                    document,
                    registry,
                    emit,
                    budget,
                    false,
                    Some(true),
                )?);
            }
            parser::Stage::MaxAllBy { selector, .. } => {
                let selected = take_operations(current, "max_all_by")?;
                current = QueryOutput::Operations(sort_operations_by(
                    selected,
                    selector,
                    document,
                    registry,
                    emit,
                    budget,
                    true,
                    Some(true),
                )?);
            }
            parser::Stage::Attr { name, .. } => {
                let selected = take_operations(current, "attr")?;
                current = QueryOutput::Values(evaluate_attr(document, &selected, name)?);
            }
            parser::Stage::Names { .. } => {
                let selected = take_operations(current, "names")?;
                current = QueryOutput::Values(
                    selected
                        .iter()
                        .filter_map(|&op| document.operation_name(op).map(str::to_owned))
                        .collect(),
                );
            }
            parser::Stage::ResultTypes { .. } => {
                let selected = take_operations(current, "result_types")?;
                current = QueryOutput::Values(
                    budget.collect(
                        selected
                            .iter()
                            .flat_map(|&op| document.result_types(op).unwrap_or(&[]))
                            .filter_map(|&ty| document.type_spelling(ty).map(str::to_owned)),
                    )?,
                );
            }
            parser::Stage::OperandTypes { .. } => {
                let selected = take_operations(current, "operand_types")?;
                current = QueryOutput::Values(
                    budget.collect(
                        selected
                            .iter()
                            .flat_map(|&op| document.operands(op).unwrap_or(&[]))
                            .filter_map(|&value| document.value_type(value).map(str::to_owned)),
                    )?,
                );
            }
            parser::Stage::Group { expression, .. } => {
                current =
                    evaluate_expression(expression, document, registry, current, emit, budget)?;
            }
            parser::Stage::Fixpoint { expression, .. } => {
                if expression.is_single_closure() {
                    current = QueryOutput::Operations(evaluate_closure(
                        document,
                        take_operations(current, "closure")?,
                        registry,
                        true,
                        budget,
                    )?);
                    continue;
                }
                // Brent's cycle detection keeps one checkpoint instead of storing
                // every intermediate selection. Read-only bodies produce deterministic selections.
                let mut checkpoint = current.clone();
                let mut power = 1usize;
                let mut distance = 0usize;
                loop {
                    let next = evaluate_expression(
                        expression,
                        document,
                        registry,
                        current.clone(),
                        emit,
                        budget,
                    )?;
                    if matches!(
                        next,
                        QueryOutput::Count(_)
                            | QueryOutput::Json(_)
                            | QueryOutput::Text(_)
                            | QueryOutput::Map(_)
                            | QueryOutput::Array(_)
                    ) {
                        return Err(EvaluationError::new(
                            "fixpoint requires an operation or value stream",
                        ));
                    }
                    budget.charge(1)?;
                    if next == current {
                        break;
                    }
                    if next == checkpoint {
                        return Err(EvaluationError {
                            message:
                                "fixpoint query cycles without reaching an unchanged selection"
                                    .into(),
                        });
                    }
                    current = next;
                    distance += 1;
                    if distance == power {
                        checkpoint = current.clone();
                        power = power.saturating_mul(2);
                        distance = 0;
                    }
                }
            }
            parser::Stage::SetAttr { name, value, .. } => {
                let selected = operations(&current, "set_attr")?;
                let mut editor = document.edit(registry).map_err(edit_error)?;
                let spelling = quote_mlir_string(value);
                for operation in selected.iter().copied().collect::<HashSet<_>>() {
                    editor
                        .set_attribute(
                            operation,
                            AttributeSpec {
                                name: name.clone(),
                                spelling: spelling.clone(),
                                value: AttributeValue::String(spelling.clone()),
                            },
                        )
                        .map_err(edit_error)?;
                }
                editor.commit().map_err(edit_error)?;
            }
            parser::Stage::RemoveAttr { name, .. } => {
                let selected = operations(&current, "remove_attr")?;
                let targets = selected
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .filter(|&operation| {
                        document.operation_is_unparsed(operation) == Some(true)
                            || document.attribute_id(operation, name).is_some()
                    })
                    .collect::<Vec<_>>();
                if !targets.is_empty() {
                    let mut editor = document.edit(registry).map_err(edit_error)?;
                    for operation in targets {
                        editor
                            .remove_attribute(operation, name)
                            .map_err(edit_error)?;
                    }
                    editor.commit().map_err(edit_error)?;
                }
            }
            parser::Stage::Count { .. } => {
                return Ok(QueryOutput::Count(countable_len(&current)?));
            }
            parser::Stage::Emit { .. } => emit(document, current.clone())?,
            parser::Stage::Markdown { .. } => {
                emit(
                    document,
                    QueryOutput::Text(render::markdown(&current, budget)?),
                )?;
            }
            parser::Stage::Print { parts, .. } => {
                let mut text = render::interpolate(parts, budget)?;
                text.push('\n');
                emit(document, QueryOutput::Text(text))?;
            }
            parser::Stage::Json { .. } => emit(
                document,
                QueryOutput::Json(output_json(document, &current)?),
            )?,
        }
        budget.check_items(output_size(&current))?;
    }
    Ok(current)
}

fn operations<'a>(
    output: &'a QueryOutput,
    stage: &str,
) -> Result<&'a [OperationId], EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        _ => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
    }
}

fn operations_mut<'a>(
    output: &'a mut QueryOutput,
    stage: &str,
) -> Result<&'a mut Vec<OperationId>, EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        _ => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
    }
}

fn take_operations(output: QueryOutput, stage: &str) -> Result<Vec<OperationId>, EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        _ => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
    }
}

fn countable_len(output: &QueryOutput) -> Result<usize, EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected.len()),
        QueryOutput::Values(values) => Ok(values.len()),
        QueryOutput::Map(values) => Ok(values.len()),
        QueryOutput::Array(values) => Ok(values.len()),
        QueryOutput::Count(_) | QueryOutput::Json(_) | QueryOutput::Text(_) => Err(
            EvaluationError::new("count requires a stream, map, or array"),
        ),
    }
}

fn output_len(output: &QueryOutput) -> usize {
    match output {
        QueryOutput::Operations(values) => values.len(),
        QueryOutput::Values(values) => values.len(),
        QueryOutput::Map(values) => values.len(),
        QueryOutput::Array(values) => values.len(),
        QueryOutput::Count(_) | QueryOutput::Json(_) | QueryOutput::Text(_) => 1,
    }
}

fn truncate_stream(
    output: &mut QueryOutput,
    count: usize,
    from_end: bool,
) -> Result<(), EvaluationError> {
    fn truncate<T>(values: &mut Vec<T>, count: usize, from_end: bool) {
        if from_end {
            let keep_from = values.len().saturating_sub(count);
            values.drain(..keep_from);
        } else {
            values.truncate(count);
        }
    }
    match output {
        QueryOutput::Operations(values) => truncate(values, count, from_end),
        QueryOutput::Values(values) => truncate(values, count, from_end),
        _ => {
            return Err(EvaluationError::new(
                "head and tail require an operation or value stream",
            ));
        }
    }
    Ok(())
}

fn extreme_value(
    output: QueryOutput,
    maximum: bool,
    retain_all: bool,
) -> Result<Vec<String>, EvaluationError> {
    let QueryOutput::Values(values) = output else {
        return Err(EvaluationError::new(
            "value extrema require a value stream; use names or attr first",
        ));
    };
    let value = if maximum {
        values.iter().max()
    } else {
        values.iter().min()
    }
    .cloned()
    .ok_or_else(|| EvaluationError::new("value extrema require a non-empty value stream"))?;
    if retain_all {
        Ok(values
            .into_iter()
            .filter(|candidate| candidate == &value)
            .collect())
    } else {
        Ok(vec![value])
    }
}

#[allow(clippy::too_many_arguments)]
fn sort_operations_by(
    selected: Vec<OperationId>,
    selector: &parser::Expression,
    document: &mut Document,
    registry: &DialectRegistry,
    emit: &mut impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    budget: &mut EvaluationState,
    descending: bool,
    extrema: Option<bool>,
) -> Result<Vec<OperationId>, EvaluationError> {
    if extrema.is_some() && selected.is_empty() {
        return Err(EvaluationError::new(
            "by extrema require a non-empty operation stream",
        ));
    }
    let mut keyed = Vec::with_capacity(selected.len());
    for operation in selected {
        let output = evaluate_expression(
            selector,
            document,
            registry,
            QueryOutput::Operations(vec![operation]),
            emit,
            budget,
        )?;
        let key = match output {
            QueryOutput::Count(value) => SortKey::Count(value),
            QueryOutput::Values(mut values) if values.len() == 1 => {
                SortKey::Text(values.pop().unwrap())
            }
            _ => {
                return Err(EvaluationError::new(
                    "ordering selector must produce exactly one string or count per operation",
                ));
            }
        };
        keyed.push((operation, key));
    }
    keyed.sort_by(|left, right| {
        let ordering = left.1.cmp(&right.1);
        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });
    if let Some(retain_all) = extrema {
        let retain = if retain_all {
            keyed
                .first()
                .map(|(_, extreme)| {
                    keyed
                        .iter()
                        .take_while(|(_, candidate)| candidate == extreme)
                        .count()
                })
                .unwrap_or(0)
        } else {
            1
        };
        keyed.truncate(retain);
    }
    Ok(keyed.into_iter().map(|(operation, _)| operation).collect())
}

fn value_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(entries) => {
            1 + entries.values().map(value_depth).max().unwrap_or(0)
        }
        serde_json::Value::Array(values) => 1 + values.iter().map(value_depth).max().unwrap_or(0),
        _ => 0,
    }
}

// Bound nested aggregate contents as well as the outer map.
fn json_size(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(entries) => {
            1 + entries
                .values()
                .map(|value| 1 + json_size(value))
                .sum::<usize>()
        }
        serde_json::Value::Array(values) => 1 + values.iter().map(json_size).sum::<usize>(),
        _ => 1,
    }
}

fn output_size(output: &QueryOutput) -> usize {
    match output {
        QueryOutput::Map(entries) => entries.values().map(|value| 1 + json_size(value)).sum(),
        QueryOutput::Array(values) => 1 + values.iter().map(json_size).sum::<usize>(),
        _ => output_len(output),
    }
}

fn source_ordered(document: &Document, selected: HashSet<OperationId>) -> Vec<OperationId> {
    document
        .operations()
        .filter(|operation| selected.contains(operation))
        .collect()
}

fn evaluate_set_operator(
    document: &Document,
    operator: parser::SetOperator,
    left: QueryOutput,
    right: QueryOutput,
) -> Result<QueryOutput, EvaluationError> {
    match (left, right) {
        (QueryOutput::Operations(left), QueryOutput::Operations(right)) => {
            let mut left = left.into_iter().collect::<HashSet<_>>();
            let right = right.into_iter().collect::<HashSet<_>>();
            match operator {
                parser::SetOperator::Union => left.extend(right),
                parser::SetOperator::Intersect => {
                    left.retain(|operation| right.contains(operation))
                }
                parser::SetOperator::Except => left.retain(|operation| !right.contains(operation)),
            }
            Ok(QueryOutput::Operations(source_ordered(document, left)))
        }
        (QueryOutput::Values(mut left), QueryOutput::Values(right)) => {
            let right_set = right.iter().cloned().collect::<HashSet<_>>();
            match operator {
                parser::SetOperator::Union => left.extend(right),
                parser::SetOperator::Intersect => left.retain(|value| right_set.contains(value)),
                parser::SetOperator::Except => left.retain(|value| !right_set.contains(value)),
            }
            retain_unique(&mut left);
            Ok(QueryOutput::Values(left))
        }
        (QueryOutput::Operations(_), QueryOutput::Values(_))
        | (QueryOutput::Values(_), QueryOutput::Operations(_)) => Err(EvaluationError::new(
            "set operands must produce the same kind of stream",
        )),
        _ => Err(EvaluationError::new(
            "set operands require operation or value streams",
        )),
    }
}

fn retain_unique<T: Clone + Eq + std::hash::Hash>(values: &mut Vec<T>) {
    let mut seen = HashSet::new();
    values.retain(|value| seen.insert(value.clone()));
}

fn evaluate_defs(
    document: &Document,
    selected: &[OperationId],
    index: Option<usize>,
    budget: &EvaluationState,
) -> Result<Vec<OperationId>, EvaluationError> {
    budget.collect(
        selected
            .iter()
            .flat_map(|&operation| {
                let operands = document.operands(operation).unwrap_or(&[]);
                match index {
                    Some(index) => operands.get(index).map(std::slice::from_ref).unwrap_or(&[]),
                    None => operands,
                }
            })
            .filter_map(|operand| match *operand {
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    Some(operation)
                }
                ValueReference::Resolved(ValueId::BlockArgument { block, .. }) => document
                    .block(block)
                    .and_then(|block| document.region(block.parent_region()))
                    .map(|region| region.parent_operation()),
                ValueReference::Invalid(_) => None,
            }),
    )
}

fn evaluate_users(
    document: &Document,
    selected: &[OperationId],
    index: Option<usize>,
    budget: &EvaluationState,
) -> Result<Vec<OperationId>, EvaluationError> {
    budget.collect(
        selected
            .iter()
            .flat_map(|&operation| {
                let count = document.result_types(operation).map_or(0, <[_]>::len);
                let results = match index {
                    Some(index) if index < count => index..index + 1,
                    Some(_) => 0..0,
                    None => 0..count,
                };
                results.map(|result| result as u32).flat_map(move |result| {
                    document.uses(ValueId::OperationResult { operation, result })
                })
            })
            .map(|site| match site {
                UseSite::Operand { operation, .. }
                | UseSite::SuccessorArgument { operation, .. } => operation,
            }),
    )
}

fn evaluate_parent(
    document: &Document,
    selected: &[OperationId],
    budget: &EvaluationState,
) -> Result<Vec<OperationId>, EvaluationError> {
    budget.collect(selected.iter().filter_map(|&operation| {
        document
            .operation(operation)?
            .parent_block()
            .and_then(|block| document.block(block))
            .and_then(|block| document.region(block.parent_region()))
            .map(|region| region.parent_operation())
    }))
}

fn evaluate_children(
    document: &Document,
    selected: &[OperationId],
    budget: &EvaluationState,
) -> Result<Vec<OperationId>, EvaluationError> {
    budget.collect(
        selected
            .iter()
            .flat_map(|&operation| document.operation_regions(operation).unwrap_or(&[]))
            .flat_map(|&region| {
                document
                    .region(region)
                    .and_then(|region| region.blocks(document))
                    .unwrap_or(&[])
            })
            .flat_map(|&block| document.block_operations(block).unwrap_or(&[]))
            .copied(),
    )
}

fn evaluate_root(
    document: &Document,
    selected: &[OperationId],
    predicate: &parser::Predicate,
) -> Vec<OperationId> {
    selected
        .iter()
        .filter_map(|&operation| {
            let mut candidate = Some(operation);
            while let Some(operation) = candidate {
                if evaluate_predicate(predicate, document, operation) {
                    return Some(operation);
                }
                candidate = operation_parent(document, operation);
            }
            None
        })
        .collect()
}

fn operation_parent(document: &Document, operation: OperationId) -> Option<OperationId> {
    document
        .operation(operation)?
        .parent_block()
        .and_then(|block| document.block(block))
        .and_then(|block| document.region(block.parent_region()))
        .map(|region| region.parent_operation())
}

fn evaluate_attr(
    document: &Document,
    selected: &[OperationId],
    name: &str,
) -> Result<Vec<String>, EvaluationError> {
    selected.iter().filter_map(|&operation| {
        let attribute = document.attribute_id(operation, name)?;
        Some(match document.attribute_value(attribute)? {
            AttributeValue::String(spelling) => decode_mlir_string(spelling).ok_or_else(||
                EvaluationError::new(format!("attribute `{name}` cannot be decoded as UTF-8; use json to inspect its escaped spelling"))),
            AttributeValue::Symbol(path) => Ok(path.join("::")),
            _ => Ok(document.attribute_spelling_value(attribute)?.to_owned()),
        })
    }).collect()
}

fn output_json(document: &Document, output: &QueryOutput) -> Result<String, EvaluationError> {
    serde_json::to_string_pretty(&output_value(document, output))
        .map(|json| format!("{json}\n"))
        .map_err(|error| EvaluationError::new(format!("could not encode JSON: {error}")))
}

fn output_value(document: &Document, output: &QueryOutput) -> serde_json::Value {
    match output {
        QueryOutput::Operations(selected) => serde_json::Value::Array(
            selected
                .iter()
                .map(|&operation| {
                    let attributes = document
                        .attributes(operation)
                        .into_iter()
                        .flatten()
                        .map(|(name, spelling)| {
                            (
                                name.to_owned(),
                                serde_json::Value::String(spelling.to_owned()),
                            )
                        })
                        .collect::<serde_json::Map<_, _>>();
                    serde_json::json!({
                        "name": document.operation_name(operation),
                        "attributes": attributes,
                        "operand_types": document.operands(operation).unwrap_or(&[]).iter()
                            .map(|&value| document.value_type(value)).collect::<Vec<_>>(),
                        "result_types": document.result_types(operation).unwrap_or(&[]).iter()
                            .map(|&ty| document.type_spelling(ty)).collect::<Vec<_>>(),
                    })
                })
                .collect(),
        ),
        QueryOutput::Values(values) => serde_json::json!(values),
        QueryOutput::Map(values) => serde_json::Value::Object(values.clone()),
        QueryOutput::Array(values) => serde_json::Value::Array(values.clone()),
        QueryOutput::Count(count) => serde_json::json!(count),
        QueryOutput::Json(value) | QueryOutput::Text(value) => {
            serde_json::Value::String(value.clone())
        }
    }
}

fn evaluate_predicate(
    predicate: &parser::Predicate,
    document: &Document,
    operation: OperationId,
) -> bool {
    match predicate {
        parser::Predicate::Bool { value, .. } => *value,
        parser::Predicate::Op { name, .. } => document.operation_name(operation) == Some(name),
        parser::Predicate::Dialect { name, .. } => document.operation_name(operation)
            .and_then(|op| op.split_once('.')).is_some_and(|(dialect, _)| dialect == name),
        parser::Predicate::ResultType { spelling, .. } => document.result_types(operation).unwrap_or(&[])
            .iter().any(|&ty| document.type_spelling(ty) == Some(spelling)),
        parser::Predicate::HasAttr { name, .. } => document
            .attribute_entries(operation)
            .is_some_and(|mut entries| entries.any(|(attribute, _)| attribute == name)),
        parser::Predicate::Attr { name, value, .. } => document
            .attribute_entries(operation)
            .and_then(|mut entries| entries.find(|(attribute, _)| attribute == name))
            .and_then(|(_, id)| document.attribute_value(id))
            .is_some_and(|attribute| matches!(attribute, AttributeValue::String(spelling) if decode_mlir_string(spelling).as_deref() == Some(value))),
        parser::Predicate::Not { predicate, .. } => !evaluate_predicate(predicate, document, operation),
        parser::Predicate::And { predicates, .. } => predicates.iter().all(|predicate| evaluate_predicate(predicate, document, operation)),
        parser::Predicate::Or { predicates, .. } => predicates.iter().any(|predicate| evaluate_predicate(predicate, document, operation)),
        parser::Predicate::Group { predicate, .. } => evaluate_predicate(predicate, document, operation),
    }
}

fn quote_mlir_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn edit_error(error: impl fmt::Display) -> EvaluationError {
    EvaluationError {
        message: format!("edit failed: {error}"),
    }
}

#[derive(Default)]
struct ClosureSelection {
    selected: HashSet<OperationId>,
    pending: VecDeque<OperationId>,
    // A selected operation need not have had its subtree retained yet.
    expanded_subtrees: HashSet<OperationId>,
}

impl ClosureSelection {
    fn insert(
        &mut self,
        operation: OperationId,
        budget: &EvaluationState,
    ) -> Result<(), EvaluationError> {
        if self.selected.insert(operation) {
            budget.check_items(self.selected.len())?;
            self.pending.push_back(operation);
        }
        Ok(())
    }
}

// SSA-only traversal: block arguments are boundaries, and regions/callees are
// not expanded implicitly. This is a structural slice, not a standalone program.
fn evaluate_slice(
    document: &Document,
    seeds: Vec<OperationId>,
    budget: &mut EvaluationState,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selection = ClosureSelection::default();
    for operation in seeds {
        selection.insert(operation, budget)?;
    }
    while let Some(operation) = selection.pending.pop_front() {
        budget.charge(1)?;
        if document.operation_is_unparsed(operation) == Some(true) {
            return Err(EvaluationError::new(
                "slice requires parsed SSA operands; load the appropriate registry",
            ));
        }
        budget.charge(document.operands(operation).unwrap_or(&[]).len())?;
        for operand in document.operands(operation).unwrap_or(&[]) {
            match *operand {
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    selection.insert(operation, budget)?
                }
                ValueReference::Resolved(ValueId::BlockArgument { .. }) => {}
                ValueReference::Invalid(_) => {
                    return Err(EvaluationError::new(
                        "slice encountered an invalid SSA operand",
                    ));
                }
            }
        }
    }
    Ok(source_ordered(document, selection.selected))
}

// Analysis reachability retains bodies and explicit dependencies, without
// widening block arguments or successor edges to their owning function.
fn evaluate_reachable(
    document: &Document,
    seeds: Vec<OperationId>,
    registry: &DialectRegistry,
    budget: &mut EvaluationState,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selection = ClosureSelection::default();
    for operation in seeds {
        selection.insert(operation, budget)?;
    }
    while let Some(operation) = selection.pending.pop_front() {
        budget.charge(1)?;
        let name = document
            .operation_name(operation)
            .unwrap_or("<invalid operation>");
        if document.operation_is_unparsed(operation) == Some(true)
            || (registry.operation(name).is_none()
                && registry.operation_shape(name).is_none()
                && registry.operation_format(name).is_none()
                && registry.operation_grammars(name).is_none())
        {
            return Err(EvaluationError::new(format!(
                "reachable cannot determine reference semantics for `{name}`; load the appropriate registry"
            )));
        }
        // Region-bearing operations include their explicitly represented bodies.
        retain_subtree(document, operation, &mut selection, budget)?;
        if let Some(attribute) = registry.call_target_attribute(name) {
            let callee = call_target_spelling(document, operation, attribute).ok_or_else(|| {
                EvaluationError::new(format!(
                    "reachable encountered `{name}` without call-target attribute `{attribute}`"
                ))
            })?;
            let target = document
                .checked_lookup_symbol(operation, &callee, registry)
                .map_err(|error| {
                    EvaluationError::new(format!("reachable could not look up `{callee}`: {error}"))
                })?
                .ok_or_else(|| {
                    EvaluationError::new(format!("reachable could not resolve callee `{callee}`"))
                })?;
            if document.operation_regions(target).unwrap_or(&[]).is_empty() {
                return Err(EvaluationError::new(format!(
                    "reachable cannot inspect external callee `{callee}` without a body"
                )));
            }
            // Include the callee's body, without counting its declaration as an operation in the caller.
            for child in evaluate_children(document, &[target], budget)? {
                retain_subtree(document, child, &mut selection, budget)?;
            }
        } else if registry.symbols(name).uses_symbols {
            return Err(EvaluationError::new(format!(
                "reachable does not yet support symbol references on `{name}`"
            )));
        }
        let successors = document.successors(operation).unwrap_or(&[]);
        if !successors.is_empty() && name != "cf.br" && name != "cf.cond_br" {
            return Err(EvaluationError::new(format!(
                "reachable does not yet support successor references on `{name}`"
            )));
        }
        for successor in successors {
            for &target in document.block_operations(successor.block()).unwrap_or(&[]) {
                selection.insert(target, budget)?;
            }
        }
        budget.charge(document.operands(operation).unwrap_or(&[]).len())?;
        for operand in document.operands(operation).unwrap_or(&[]) {
            match *operand {
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    selection.insert(operation, budget)?
                }
                ValueReference::Resolved(ValueId::BlockArgument { .. }) => {}
                ValueReference::Invalid(_) => {
                    return Err(EvaluationError::new(
                        "reachable encountered an invalid SSA operand",
                    ));
                }
            }
        }
    }
    Ok(source_ordered(document, selection.selected))
}

fn evaluate_closure(
    document: &Document,
    seeds: Vec<OperationId>,
    registry: &DialectRegistry,
    transitive: bool,
    budget: &mut EvaluationState,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selection = ClosureSelection::default();
    for operation in seeds {
        selection.insert(operation, budget)?;
    }
    let seed_count = selection.pending.len();
    let mut visited = 0;
    while transitive || visited < seed_count {
        let Some(operation) = selection.pending.pop_front() else {
            break;
        };
        budget.charge(1)?;
        visited += 1;
        let name = document
            .operation_name(operation)
            .unwrap_or("<invalid operation>");
        if document.operation_is_unparsed(operation) == Some(true)
            || (registry.operation(name).is_none()
                && registry.operation_shape(name).is_none()
                && registry.operation_format(name).is_none()
                && registry.operation_grammars(name).is_none())
        {
            return Err(EvaluationError {
                message: format!(
                    "closure cannot determine reference semantics for unregistered operation `{name}`"
                ),
            });
        }
        if let Some(attribute) = registry.call_target_attribute(name) {
            let callee = call_target_spelling(document, operation, attribute).ok_or_else(|| {
                EvaluationError {
                    message: if name == "func.call" {
                        "closure encountered a func.call without a callee".to_owned()
                    } else {
                        format!(
                            "closure encountered `{name}` without call-target attribute `{attribute}`"
                        )
                    },
                }
            })?;
            let target = document
                .checked_lookup_symbol(operation, &callee, registry)
                .map_err(|error| EvaluationError {
                    message: if name == "func.call" {
                        format!("closure could not look up func.call callee `{callee}`: {error}")
                    } else {
                        format!(
                            "closure could not look up `{name}` target `{callee}`: {error}"
                        )
                    },
                })?
                .ok_or_else(|| EvaluationError {
                    message: if name == "func.call" {
                        format!(
                            "closure could not resolve func.call callee `{callee}` in an enclosing symbol table"
                        )
                    } else {
                        format!(
                            "closure could not resolve `{name}` target `{callee}` in an enclosing symbol table"
                        )
                    },
                })?;
            retain_subtree(document, target, &mut selection, budget)?;
        }
        if let Some(branch) = CfBrOp::cast(document, operation) {
            let successor = branch.successor().ok_or_else(|| EvaluationError {
                message: "closure encountered cf.br without a successor".to_owned(),
            })?;
            retain_successor_region(document, successor, name, &mut selection, budget)?;
        } else if let Some(branch) = CfCondBrOp::cast(document, operation) {
            let successors = branch.successors().ok_or_else(|| EvaluationError {
                message: "closure encountered cf.cond_br without successors".to_owned(),
            })?;
            for &successor in successors {
                retain_successor_region(document, successor, name, &mut selection, budget)?;
            }
        }
        if !document.successors(operation).unwrap_or(&[]).is_empty()
            && name != "cf.br"
            && name != "cf.cond_br"
        {
            return Err(EvaluationError {
                message: format!("closure does not yet support successor references on `{name}`"),
            });
        }
        if registry.symbols(name).uses_symbols && registry.call_target_attribute(name).is_none() {
            return Err(EvaluationError {
                message: format!("closure does not yet support symbol references on `{name}`"),
            });
        }
        budget.charge(document.operands(operation).unwrap_or(&[]).len())?;
        for operand in document.operands(operation).unwrap_or(&[]) {
            match *operand {
                ValueReference::Invalid(_) => {
                    return Err(EvaluationError {
                        message: format!("closure encountered an invalid SSA operand on `{name}`"),
                    });
                }
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    selection.insert(operation, budget)?;
                }
                ValueReference::Resolved(ValueId::BlockArgument { block, .. }) => {
                    let owner = document
                            .block(block)
                            .and_then(|block| document.region(block.parent_region()))
                            .map(|region| region.parent_operation())
                            .ok_or_else(|| EvaluationError {
                                message: format!(
                                    "closure could not resolve the owning scope for a block argument on `{name}`"
                                ),
                            })?;
                    retain_subtree(document, owner, &mut selection, budget)?;
                }
            }
        }
    }
    Ok(document
        .operations()
        .filter(|operation| selection.selected.contains(operation))
        .collect())
}

fn call_target_spelling(
    document: &Document,
    operation: OperationId,
    attribute: &str,
) -> Option<String> {
    document
        .attribute_id(operation, attribute)
        .or_else(|| {
            (attribute != "callee")
                .then(|| document.attribute_id(operation, "callee"))
                .flatten()
        })
        .and_then(|id| document.attribute_spelling_value(id))
        .map(str::to_owned)
}

fn retain_successor_region(
    document: &Document,
    successor: Successor,
    operation_name: &str,
    selection: &mut ClosureSelection,
    budget: &mut EvaluationState,
) -> Result<(), EvaluationError> {
    let owner = document
        .block(successor.block())
        .and_then(|block| document.region(block.parent_region()))
        .map(|region| region.parent_operation())
        .ok_or_else(|| EvaluationError {
            message: format!(
                "closure encountered an invalid successor target on `{operation_name}`"
            ),
        })?;
    retain_subtree(document, owner, selection, budget)?;
    Ok(())
}

fn retain_subtree(
    document: &Document,
    operation: OperationId,
    selection: &mut ClosureSelection,
    budget: &mut EvaluationState,
) -> Result<(), EvaluationError> {
    let mut pending = vec![operation];
    while let Some(operation) = pending.pop() {
        if !selection.expanded_subtrees.insert(operation) {
            continue;
        }
        budget.charge(1)?;
        selection.insert(operation, budget)?;
        for &region in document.operation_regions(operation).unwrap_or(&[]) {
            for &block in document
                .region(region)
                .and_then(|region| region.blocks(document))
                .unwrap_or(&[])
            {
                pending.extend(document.block_operations(block).unwrap_or(&[]));
            }
        }
    }
    Ok(())
}

fn append_subtree(
    document: &Document,
    operation: OperationId,
    selected: &mut Vec<OperationId>,
    budget: &mut EvaluationState,
) -> Result<(), EvaluationError> {
    let mut pending = vec![operation];
    while let Some(operation) = pending.pop() {
        budget.charge(1)?;
        budget.check_items(selected.len().saturating_add(1))?;
        selected.push(operation);
        let children = document
            .operation_regions(operation)
            .unwrap_or(&[])
            .iter()
            .flat_map(|&region| {
                document
                    .region(region)
                    .and_then(|region| region.blocks(document))
                    .unwrap_or(&[])
            })
            .flat_map(|&block| document.block_operations(block).unwrap_or(&[]))
            .copied()
            .collect::<Vec<_>>();
        pending.extend(children.into_iter().rev());
    }
    Ok(())
}
