//! Parsing and evaluation for the initial operation selection query.

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
    predicate: parser::Predicate,
    stages: Vec<parser::Stage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryOutput {
    Selection(Vec<OperationId>),
    Count(usize),
    Root,
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
        let predicate = program.predicate().clone();
        let stages = program.stages().to_vec();
        Ok(Self { predicate, stages })
    }

    pub fn operation_name(&self) -> Option<&str> {
        match &self.predicate {
            parser::Predicate::Op { name, .. } => Some(name),
            _ => None,
        }
    }

    pub fn evaluate(
        &self,
        document: &mut Document,
        registry: &DialectRegistry,
    ) -> Result<QueryOutput, EvaluationError> {
        let mut selected = document
            .operations()
            .filter(|&operation| evaluate_predicate(&self.predicate, document, operation))
            .collect::<Vec<_>>();
        let mut output = QueryOutput::Selection(selected.clone());
        for stage in &self.stages {
            match stage {
                parser::Stage::Closure { .. } => {
                    selected = evaluate_closure(document, selected, registry)?
                }
                parser::Stage::Defs { .. } => selected = evaluate_defs(document, &selected),
                parser::Stage::Users { .. } => selected = evaluate_users(document, &selected),
                parser::Stage::Parent { .. } => selected = evaluate_parent(document, &selected),
                parser::Stage::Children { .. } => selected = evaluate_children(document, &selected),
                parser::Stage::Union { predicate, .. } => {
                    let mut combined = selected.into_iter().collect::<HashSet<_>>();
                    combined.extend(matching_operations(document, predicate));
                    selected = source_ordered(document, combined);
                }
                parser::Stage::Intersect { predicate, .. } => {
                    let matching = matching_operations(document, predicate);
                    let intersection = selected
                        .into_iter()
                        .filter(|operation| matching.contains(operation))
                        .collect();
                    selected = source_ordered(document, intersection);
                }
                parser::Stage::Except { predicate, .. } => {
                    let matching = matching_operations(document, predicate);
                    let difference = selected
                        .into_iter()
                        .filter(|operation| !matching.contains(operation))
                        .collect();
                    selected = source_ordered(document, difference);
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
                        .filter(|&operation| document.attribute_id(operation, name).is_some())
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
                parser::Stage::Count { .. } => output = QueryOutput::Count(selected.len()),
                parser::Stage::Root { .. } => output = QueryOutput::Root,
            }
        }
        if matches!(output, QueryOutput::Selection(_)) {
            output = QueryOutput::Selection(selected);
        }
        Ok(output)
    }
}

fn source_ordered(document: &Document, selected: HashSet<OperationId>) -> Vec<OperationId> {
    document
        .operations()
        .filter(|operation| selected.contains(operation))
        .collect()
}

fn matching_operations(document: &Document, predicate: &parser::Predicate) -> HashSet<OperationId> {
    document
        .operations()
        .filter(|&operation| evaluate_predicate(predicate, document, operation))
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
    let mut worklist = seeds;
    while let Some(operation) = worklist.pop() {
        let name = document
            .operation_name(operation)
            .unwrap_or("<invalid operation>");
        if registry.operation(name).is_none() {
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
            retain_subtree(document, target, &mut selected, &mut worklist);
        }
        if let Some(branch) = CfBrOp::cast(document, operation) {
            let successor = branch.successor().ok_or_else(|| EvaluationError {
                message: "closure encountered cf.br without a successor".to_owned(),
            })?;
            retain_successor_region(document, successor, name, &mut selected, &mut worklist)?;
        } else if let Some(branch) = CfCondBrOp::cast(document, operation) {
            let successors = branch.successors().ok_or_else(|| EvaluationError {
                message: "closure encountered cf.cond_br without successors".to_owned(),
            })?;
            for &successor in successors {
                retain_successor_region(document, successor, name, &mut selected, &mut worklist)?;
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
                    enqueue(operation, &mut selected, &mut worklist);
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
                    retain_subtree(document, owner, &mut selected, &mut worklist);
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
    worklist: &mut Vec<OperationId>,
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
    retain_subtree(document, owner, selected, worklist);
    Ok(())
}

fn enqueue(
    operation: OperationId,
    selected: &mut HashSet<OperationId>,
    worklist: &mut Vec<OperationId>,
) {
    if selected.insert(operation) {
        worklist.push(operation);
    }
}

fn retain_subtree(
    document: &Document,
    operation: OperationId,
    selected: &mut HashSet<OperationId>,
    worklist: &mut Vec<OperationId>,
) {
    enqueue(operation, selected, worklist);
    for &region in document.operation_regions(operation).unwrap_or(&[]) {
        for &block in document
            .region(region)
            .and_then(|region| region.blocks(document))
            .unwrap_or(&[])
        {
            for &child in document.block_operations(block).unwrap_or(&[]) {
                retain_subtree(document, child, selected, worklist);
            }
        }
    }
}
