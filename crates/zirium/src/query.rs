//! Composable operation-selection queries, editing, and output.

use std::{collections::HashSet, fmt};

use crate::{
    dialect::DialectRegistry,
    semantic::{
        AttributeSpec, AttributeValue, CfBrOp, CfCondBrOp, Document, FuncCallOp, OperationId,
        Successor, UseSite, ValueId, ValueReference,
    },
};

pub mod lexer;
pub mod parser;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    expression: parser::Expression,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryOutput {
    Operations(Vec<OperationId>),
    Values(Vec<String>),
    Count(usize),
    Json(String),
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
        })
    }

    /// Evaluates against the current document, invoking `emit` at each explicit
    /// emission and for the implicit final output. The callback sees each edit
    /// as it exists at that point. Callers that need all-or-nothing output should
    /// buffer emissions until evaluation succeeds. Edits commit per stage.
    pub fn evaluate(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
        mut emit: impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        let input = QueryOutput::Operations(document.operations().collect());
        let output = evaluate_expression(&self.expression, document, registry, input, &mut emit)?;
        if !self.expression.ends_with_emission() {
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
) -> Result<QueryOutput, EvaluationError> {
    if expression.rest.is_empty() {
        return evaluate_pipeline(&expression.first, document, registry, input, emit);
    }
    let first = evaluate_pipeline(&expression.first, document, registry, input.clone(), emit)?;
    let mut selected = first;
    for (operator, stages) in &expression.rest {
        let right = evaluate_pipeline(stages, document, registry, input.clone(), emit)?;
        selected = evaluate_set_operator(document, *operator, selected, right)?;
    }
    Ok(selected)
}

fn evaluate_pipeline(
    stages: &[parser::Stage],
    document: &mut Document,
    registry: &DialectRegistry,
    input: QueryOutput,
    emit: &mut impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
) -> Result<QueryOutput, EvaluationError> {
    let mut current = input;
    for stage in stages {
        match stage {
            parser::Stage::Input { .. } => {
                current = QueryOutput::Operations(document.operations().collect())
            }
            parser::Stage::Filter { predicate, .. } => {
                let selected = operations_mut(&mut current, "filter")?;
                selected.retain(|&operation| evaluate_predicate(predicate, document, operation));
            }
            parser::Stage::Closure { .. } => {
                let selected = take_operations(current, "closure")?;
                current = QueryOutput::Operations(evaluate_closure(document, selected, registry)?);
            }
            parser::Stage::Defs { .. } => {
                let selected = take_operations(current, "defs")?;
                current = QueryOutput::Operations(evaluate_defs(document, &selected));
            }
            parser::Stage::Users { .. } => {
                let selected = take_operations(current, "users")?;
                current = QueryOutput::Operations(evaluate_users(document, &selected));
            }
            parser::Stage::Parent { .. } => {
                let selected = take_operations(current, "parent")?;
                current = QueryOutput::Operations(evaluate_parent(document, &selected));
            }
            parser::Stage::Children { .. } => {
                let selected = take_operations(current, "children")?;
                current = QueryOutput::Operations(evaluate_children(document, &selected));
            }
            parser::Stage::Root { predicate, .. } => {
                let selected = take_operations(current, "root")?;
                current = QueryOutput::Operations(evaluate_root(document, &selected, predicate));
            }
            parser::Stage::Subtree { .. } => {
                let selected = take_operations(current, "subtree")?;
                let mut expanded = Vec::new();
                for operation in selected {
                    append_subtree(document, operation, &mut expanded);
                }
                current = QueryOutput::Operations(expanded);
            }
            parser::Stage::Unique { .. } => match &mut current {
                QueryOutput::Operations(selected) => retain_unique(selected),
                QueryOutput::Values(values) => retain_unique(values),
                QueryOutput::Count(_) | QueryOutput::Json(_) => unreachable!("terminal output"),
            },
            parser::Stage::Attr { name, .. } => {
                let selected = take_operations(current, "attr")?;
                current = QueryOutput::Values(evaluate_attr(document, &selected, name));
            }
            parser::Stage::Group { expression, .. } => {
                current = evaluate_expression(expression, document, registry, current, emit)?;
                if matches!(current, QueryOutput::Count(_) | QueryOutput::Json(_)) {
                    return Ok(current);
                }
            }
            parser::Stage::Fixpoint { expression, .. } => {
                // Brent's cycle detection keeps one checkpoint instead of storing
                // every intermediate selection. Read-only bodies produce deterministic selections.
                let mut checkpoint = current.clone();
                let mut power = 1usize;
                let mut distance = 0usize;
                loop {
                    let next =
                        evaluate_expression(expression, document, registry, current.clone(), emit)?;
                    if matches!(next, QueryOutput::Count(_) | QueryOutput::Json(_)) {
                        unreachable!("parser checks fixpoint body");
                    }
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
            parser::Stage::Count { .. } => return Ok(QueryOutput::Count(output_len(&current))),
            parser::Stage::Emit { .. } => emit(document, current.clone())?,
            parser::Stage::Json { .. } => emit(
                document,
                QueryOutput::Json(output_json(document, &current)?),
            )?,
        }
    }
    Ok(current)
}

fn operations<'a>(
    output: &'a QueryOutput,
    stage: &str,
) -> Result<&'a [OperationId], EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        QueryOutput::Values(_) => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
        QueryOutput::Count(_) | QueryOutput::Json(_) => unreachable!("terminal output"),
    }
}

fn operations_mut<'a>(
    output: &'a mut QueryOutput,
    stage: &str,
) -> Result<&'a mut Vec<OperationId>, EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        QueryOutput::Values(_) => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
        QueryOutput::Count(_) | QueryOutput::Json(_) => unreachable!("terminal output"),
    }
}

fn take_operations(output: QueryOutput, stage: &str) -> Result<Vec<OperationId>, EvaluationError> {
    match output {
        QueryOutput::Operations(selected) => Ok(selected),
        QueryOutput::Values(_) => Err(EvaluationError::new(format!(
            "{stage} requires operations, but the current stream contains values"
        ))),
        QueryOutput::Count(_) | QueryOutput::Json(_) => unreachable!("terminal output"),
    }
}

fn output_len(output: &QueryOutput) -> usize {
    match output {
        QueryOutput::Operations(selected) => selected.len(),
        QueryOutput::Values(values) => values.len(),
        QueryOutput::Count(_) | QueryOutput::Json(_) => unreachable!("terminal output"),
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
        _ => unreachable!("parser checks set operands"),
    }
}

fn retain_unique<T: Clone + Eq + std::hash::Hash>(values: &mut Vec<T>) {
    let mut seen = HashSet::new();
    values.retain(|value| seen.insert(value.clone()));
}

fn evaluate_defs(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
    selected
        .iter()
        .flat_map(|&operation| document.operands(operation).unwrap_or(&[]))
        .filter_map(|operand| match *operand {
            ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => Some(operation),
            ValueReference::Resolved(ValueId::BlockArgument { block, .. }) => document
                .block(block)
                .and_then(|block| document.region(block.parent_region()))
                .map(|region| region.parent_operation()),
            ValueReference::Invalid(_) => None,
        })
        .collect()
}

fn evaluate_users(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
    selected
        .iter()
        .flat_map(|&operation| {
            (0..document.result_types(operation).map_or(0, <[_]>::len) as u32).flat_map(
                move |result| document.uses(ValueId::OperationResult { operation, result }),
            )
        })
        .map(|site| match site {
            UseSite::Operand { operation, .. } | UseSite::SuccessorArgument { operation, .. } => {
                operation
            }
        })
        .collect()
}

fn evaluate_parent(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
    selected
        .iter()
        .filter_map(|&operation| {
            document
                .operation(operation)?
                .parent_block()
                .and_then(|block| document.block(block))
                .and_then(|block| document.region(block.parent_region()))
                .map(|region| region.parent_operation())
        })
        .collect()
}

fn evaluate_children(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
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
        .copied()
        .collect()
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

fn evaluate_attr(document: &Document, selected: &[OperationId], name: &str) -> Vec<String> {
    selected
        .iter()
        .filter_map(|&operation| {
            let attribute = document.attribute_id(operation, name)?;
            match document.attribute_value(attribute)? {
                AttributeValue::String(spelling) => decode_mlir_string(spelling),
                AttributeValue::Symbol(path) => Some(path.join("::")),
                _ => document
                    .attribute_spelling_value(attribute)
                    .map(str::to_owned),
            }
        })
        .collect()
}

fn output_json(document: &Document, output: &QueryOutput) -> Result<String, EvaluationError> {
    let value = match output {
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
                    })
                })
                .collect(),
        ),
        QueryOutput::Values(values) => serde_json::json!(values),
        QueryOutput::Count(_) | QueryOutput::Json(_) => unreachable!("terminal output"),
    };
    serde_json::to_string_pretty(&value)
        .map(|json| format!("{json}\n"))
        .map_err(|error| EvaluationError::new(format!("could not encode JSON: {error}")))
}

fn evaluate_predicate(
    predicate: &parser::Predicate,
    document: &Document,
    operation: OperationId,
) -> bool {
    match predicate {
        parser::Predicate::Bool { value, .. } => *value,
        parser::Predicate::Op { name, .. } => document.operation_name(operation) == Some(name),
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

fn decode_mlir_string(spelling: &str) -> Option<String> {
    let inner = spelling.strip_prefix('"')?.strip_suffix('"')?;
    let bytes = inner.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor] != b'\\' {
            decoded.push(bytes[cursor]);
            cursor += 1;
            continue;
        }
        let escaped = *bytes.get(cursor + 1)?;
        if matches!(escaped, b'\\' | b'"') {
            decoded.push(escaped);
            cursor += 2;
        } else {
            let low = *bytes.get(cursor + 2)?;
            decoded.push((hex_digit(escaped)? << 4) | hex_digit(low)?);
            cursor += 3;
        }
    }
    String::from_utf8(decoded).ok()
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
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

fn evaluate_closure(
    document: &Document,
    seeds: Vec<OperationId>,
    registry: &DialectRegistry,
) -> Result<Vec<OperationId>, EvaluationError> {
    let mut selected = seeds.iter().copied().collect::<HashSet<_>>();
    for operation in seeds {
        let name = document
            .operation_name(operation)
            .unwrap_or("<invalid operation>");
        let shape = registry.operation_shape(name);
        if registry.operation(name).is_none()
            && (shape.is_none() || shape == Some(crate::dialect::OperationShape::CallLike))
            && registry.operation_format(name).is_none()
        {
            return Err(EvaluationError {
                message: format!(
                    "closure cannot determine reference semantics for unregistered operation `{name}`"
                ),
            });
        }
        if let Some(call) = FuncCallOp::cast(document, operation) {
            let callee = call.callee().ok_or_else(|| EvaluationError {
                message: "closure encountered a func.call without a callee".to_owned(),
            })?;
            let target = document
                    .checked_lookup_symbol(operation, callee, registry)
                    .map_err(|error| EvaluationError {
                        message: format!(
                            "closure could not look up func.call callee `{callee}`: {error}"
                        ),
                    })?
                    .ok_or_else(|| EvaluationError {
                        message: format!(
                            "closure could not resolve func.call callee `{callee}` in an enclosing symbol table"
                        ),
                    })?;
            retain_subtree(document, target, &mut selected);
        }
        if let Some(branch) = CfBrOp::cast(document, operation) {
            let successor = branch.successor().ok_or_else(|| EvaluationError {
                message: "closure encountered cf.br without a successor".to_owned(),
            })?;
            retain_successor_region(document, successor, name, &mut selected)?;
        } else if let Some(branch) = CfCondBrOp::cast(document, operation) {
            let successors = branch.successors().ok_or_else(|| EvaluationError {
                message: "closure encountered cf.cond_br without successors".to_owned(),
            })?;
            for &successor in successors {
                retain_successor_region(document, successor, name, &mut selected)?;
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
        if registry.symbols(name).uses_symbols && name != "func.call" {
            return Err(EvaluationError {
                message: format!("closure does not yet support symbol references on `{name}`"),
            });
        }
        for operand in document.operands(operation).unwrap_or(&[]) {
            match *operand {
                ValueReference::Invalid(_) => {
                    return Err(EvaluationError {
                        message: format!("closure encountered an invalid SSA operand on `{name}`"),
                    });
                }
                ValueReference::Resolved(ValueId::OperationResult { operation, .. }) => {
                    selected.insert(operation);
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
                    retain_subtree(document, owner, &mut selected);
                }
            }
        }
    }
    Ok(document
        .operations()
        .filter(|operation| selected.contains(operation))
        .collect())
}

fn retain_successor_region(
    document: &Document,
    successor: Successor,
    operation_name: &str,
    selected: &mut HashSet<OperationId>,
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
    retain_subtree(document, owner, selected);
    Ok(())
}

fn retain_subtree(
    document: &Document,
    operation: OperationId,
    selected: &mut HashSet<OperationId>,
) {
    let mut pending = vec![operation];
    while let Some(operation) = pending.pop() {
        selected.insert(operation);
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
}

fn append_subtree(document: &Document, operation: OperationId, selected: &mut Vec<OperationId>) {
    let mut pending = vec![operation];
    while let Some(operation) = pending.pop() {
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
}
