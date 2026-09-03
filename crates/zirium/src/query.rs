//! Parsing and evaluation for the initial operation selection query.

use std::{collections::HashSet, fmt};

use crate::{
    dialect::DialectRegistry,
    semantic::{
        AttributeSpec, AttributeValue, CfBrOp, CfCondBrOp, Document, FuncCallOp, OperationId,
        Successor, ValueId, ValueReference,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    operation_name: String,
    stages: Vec<Stage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Stage {
    Closure,
    SetAttr { name: String, value: String },
    Count,
    Root,
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
        let prefix = "select(op(\"";
        if !source.starts_with(prefix) {
            let position = source
                .bytes()
                .zip(prefix.bytes())
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(source.len().min(prefix.len()));
            return Err(QueryError {
                position,
                message: "expected `select(op(\"name\"))`",
            });
        }
        let rest = &source[prefix.len()..];
        let Some(end) = rest.find('"') else {
            return Err(QueryError {
                position: source.len(),
                message: "unterminated operation name",
            });
        };
        if end == 0 {
            return Err(QueryError {
                position: prefix.len(),
                message: "operation name must not be empty",
            });
        }
        let operation_name = rest[..end].to_owned();
        let suffix_start = prefix.len() + end + 1;
        let suffix = &source[suffix_start..];
        let Some(mut rest) = suffix.strip_prefix("))") else {
            return Err(QueryError {
                position: suffix_start,
                message: "expected `))` after operation name",
            });
        };
        let mut stages = Vec::new();
        let mut kind = PipelineKind::Selection;
        while !rest.is_empty() {
            let Some(next) = rest.strip_prefix(" | ") else {
                return Err(QueryError {
                    position: source.len() - rest.len(),
                    message: "expected ` | ` followed by a pipeline operation",
                });
            };
            let position = source.len() - next.len();
            if let Some(after) = next.strip_prefix("closure") {
                require_selection(kind, position, "closure requires a selection")?;
                stages.push(Stage::Closure);
                rest = after;
            } else if let Some(after) = next.strip_prefix("count") {
                require_selection(kind, position, "count requires a selection")?;
                stages.push(Stage::Count);
                kind = PipelineKind::Scalar;
                rest = after;
            } else if let Some(after) = next.strip_prefix("root") {
                require_selection(kind, position, "root requires a selection")?;
                stages.push(Stage::Root);
                kind = PipelineKind::Root;
                rest = after;
            } else if let Some(after) = next.strip_prefix("set_attr(") {
                require_selection(kind, position, "set_attr requires a selection")?;
                let (name, after) = parse_string(after, source.len() - after.len())?;
                let Some(after) = after.strip_prefix(", ") else {
                    return Err(QueryError {
                        position,
                        message: "expected `, ` in set_attr",
                    });
                };
                let (value, after) = parse_string(after, source.len() - after.len())?;
                let Some(after) = after.strip_prefix(')') else {
                    return Err(QueryError {
                        position,
                        message: "expected `)` after set_attr arguments",
                    });
                };
                if !valid_attribute_name(&name) {
                    return Err(QueryError {
                        position,
                        message: "attribute name must be a dotted ASCII identifier",
                    });
                }
                if value.chars().any(char::is_control) {
                    return Err(QueryError {
                        position,
                        message: "attribute string must not contain control characters",
                    });
                }
                stages.push(Stage::SetAttr { name, value });
                rest = after;
            } else {
                return Err(QueryError {
                    position,
                    message: "unknown pipeline operation",
                });
            }
        }
        Ok(Self {
            operation_name,
            stages,
        })
    }

    pub fn operation_name(&self) -> &str {
        &self.operation_name
    }

    pub fn evaluate(&self, document: &mut Document) -> Result<QueryOutput, EvaluationError> {
        let mut selected = document
            .operations()
            .filter(|&operation| document.operation_name(operation) == Some(self.operation_name()))
            .collect::<Vec<_>>();
        let mut output = QueryOutput::Selection(selected.clone());
        for stage in &self.stages {
            match stage {
                Stage::Closure => selected = evaluate_closure(document, selected)?,
                Stage::SetAttr { name, value } => {
                    let registry = DialectRegistry::proving();
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
                Stage::Count => output = QueryOutput::Count(selected.len()),
                Stage::Root => output = QueryOutput::Root,
            }
        }
        if matches!(output, QueryOutput::Selection(_)) {
            output = QueryOutput::Selection(selected);
        }
        Ok(output)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum PipelineKind {
    Selection,
    Scalar,
    Root,
}

fn require_selection(
    kind: PipelineKind,
    position: usize,
    message: &'static str,
) -> Result<(), QueryError> {
    if kind == PipelineKind::Selection {
        Ok(())
    } else {
        Err(QueryError { position, message })
    }
}

fn parse_string(source: &str, position: usize) -> Result<(String, &str), QueryError> {
    let Some(mut rest) = source.strip_prefix('"') else {
        return Err(QueryError {
            position,
            message: "expected a quoted string",
        });
    };
    let mut value = String::new();
    while let Some(ch) = rest.chars().next() {
        rest = &rest[ch.len_utf8()..];
        match ch {
            '"' => return Ok((value, rest)),
            '\\' => {
                let Some(escaped) = rest.chars().next() else {
                    return Err(QueryError {
                        position,
                        message: "unterminated string escape",
                    });
                };
                rest = &rest[escaped.len_utf8()..];
                match escaped {
                    '"' | '\\' => value.push(escaped),
                    _ => {
                        return Err(QueryError {
                            position,
                            message: "unsupported string escape",
                        });
                    }
                }
            }
            _ => value.push(ch),
        }
    }
    Err(QueryError {
        position,
        message: "unterminated string",
    })
}

fn valid_attribute_name(name: &str) -> bool {
    name.split('.').all(|component| {
        let mut chars = component.chars();
        chars
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
            && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    })
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
) -> Result<Vec<OperationId>, EvaluationError> {
    let registry = DialectRegistry::proving();
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
        if !document.successors(operation).unwrap_or(&[]).is_empty() {
            if name != "cf.br" && name != "cf.cond_br" {
                return Err(EvaluationError {
                    message: format!(
                        "closure does not yet support successor references on `{name}`"
                    ),
                });
            }
        }
        if registry.symbols(name).uses_symbols {
            if name != "func.call" {
                return Err(EvaluationError {
                    message: format!("closure does not yet support symbol references on `{name}`"),
                });
            }
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
