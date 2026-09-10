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
    Selection(Vec<OperationId>),
    Count(usize),
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
        let selected = document.operations().collect();
        let output =
            evaluate_expression(&self.expression, document, registry, selected, &mut emit)?;
        if !self.expression.ends_with_emit() {
            emit(document, output)?;
        }
        Ok(())
    }
}

fn evaluate_expression(
    expression: &parser::Expression,
    document: &mut Document,
    registry: &DialectRegistry,
    input: Vec<OperationId>,
    emit: &mut impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
) -> Result<QueryOutput, EvaluationError> {
    if expression.rest.is_empty() {
        return evaluate_pipeline(&expression.first, document, registry, input, emit);
    }
    let first = evaluate_pipeline(&expression.first, document, registry, input.clone(), emit)?;
    let QueryOutput::Selection(mut selected) = first else {
        unreachable!("parser checks set operands")
    };
    for (operator, stages) in &expression.rest {
        let QueryOutput::Selection(right) =
            evaluate_pipeline(stages, document, registry, input.clone(), emit)?
        else {
            unreachable!("parser checks set operands")
        };
        let mut left = selected.into_iter().collect::<HashSet<_>>();
        let right = right.into_iter().collect::<HashSet<_>>();
        match operator {
            parser::SetOperator::Union => left.extend(right),
            parser::SetOperator::Intersect => left.retain(|operation| right.contains(operation)),
            parser::SetOperator::Except => left.retain(|operation| !right.contains(operation)),
        }
        selected = source_ordered(document, left);
    }
    Ok(QueryOutput::Selection(selected))
}

fn evaluate_pipeline(
    stages: &[parser::Stage],
    document: &mut Document,
    registry: &DialectRegistry,
    mut selected: Vec<OperationId>,
    emit: &mut impl FnMut(&Document, QueryOutput) -> Result<(), EvaluationError>,
) -> Result<QueryOutput, EvaluationError> {
    for stage in stages {
        match stage {
            parser::Stage::Input { .. } => selected = document.operations().collect(),
            parser::Stage::Filter { predicate, .. } => {
                selected.retain(|&operation| evaluate_predicate(predicate, document, operation));
            }
            parser::Stage::Closure { .. } => {
                selected = evaluate_closure(document, selected, registry)?
            }
            parser::Stage::Defs { .. } => selected = evaluate_defs(document, &selected),
            parser::Stage::Users { .. } => selected = evaluate_users(document, &selected),
            parser::Stage::Parent { .. } => selected = evaluate_parent(document, &selected),
            parser::Stage::Children { .. } => selected = evaluate_children(document, &selected),
            parser::Stage::Root { .. } => {
                let mut expanded = HashSet::new();
                for operation in selected {
                    if !expanded.contains(&operation) {
                        retain_subtree(document, operation, &mut expanded);
                    }
                }
                selected = source_ordered(document, expanded);
            }
            parser::Stage::Group { expression, .. } => {
                match evaluate_expression(expression, document, registry, selected, emit)? {
                    QueryOutput::Selection(result) => selected = result,
                    output => return Ok(output),
                }
            }
            parser::Stage::Fixpoint { expression, .. } => {
                // Brent's cycle detection keeps one checkpoint instead of storing
                // every intermediate selection. Read-only bodies produce deterministic selections.
                let mut checkpoint = selected.clone();
                let mut power = 1usize;
                let mut distance = 0usize;
                loop {
                    let QueryOutput::Selection(next) = evaluate_expression(
                        expression,
                        document,
                        registry,
                        selected.clone(),
                        emit,
                    )?
                    else {
                        unreachable!("parser checks fixpoint body")
                    };
                    if next == selected {
                        break;
                    }
                    if next == checkpoint {
                        return Err(EvaluationError {
                            message:
                                "fixpoint query cycles without reaching an unchanged selection"
                                    .into(),
                        });
                    }
                    selected = next;
                    distance += 1;
                    if distance == power {
                        checkpoint = selected.clone();
                        power = power.saturating_mul(2);
                        distance = 0;
                    }
                }
            }
            parser::Stage::SetAttr { name, value, .. } => {
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
            parser::Stage::Count { .. } => return Ok(QueryOutput::Count(selected.len())),
            parser::Stage::Emit { .. } => emit(document, QueryOutput::Selection(selected.clone()))?,
        }
    }
    Ok(QueryOutput::Selection(selected))
}

fn source_ordered(document: &Document, selected: HashSet<OperationId>) -> Vec<OperationId> {
    document
        .operations()
        .filter(|operation| selected.contains(operation))
        .collect()
}

fn evaluate_defs(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
    let definitions = selected
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
        .collect();
    source_ordered(document, definitions)
}

fn evaluate_users(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
    let users = selected
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
        .collect();
    source_ordered(document, users)
}

fn evaluate_parent(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
    let parents = selected
        .iter()
        .filter_map(|&operation| {
            document
                .operation(operation)?
                .parent_block()
                .and_then(|block| document.block(block))
                .and_then(|block| document.region(block.parent_region()))
                .map(|region| region.parent_operation())
        })
        .collect();
    source_ordered(document, parents)
}

fn evaluate_children(document: &Document, selected: &[OperationId]) -> Vec<OperationId> {
    let children = selected
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
        .collect();
    source_ordered(document, children)
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
