//! Parsing and evaluation for the initial operation selection query.

use std::fmt;

use crate::semantic::{Document, OperationId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    operation_name: String,
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
        let suffix_start = prefix.len() + end + 1;
        if &source[suffix_start..] != "))" {
            return Err(QueryError {
                position: suffix_start,
                message: "expected `))` after operation name",
            });
        }
        Ok(Self {
            operation_name: rest[..end].to_owned(),
        })
    }

    pub fn operation_name(&self) -> &str {
        &self.operation_name
    }

    pub fn evaluate(&self, document: &Document) -> Vec<OperationId> {
        document
            .operations()
            .filter(|&operation| document.operation_name(operation) == Some(self.operation_name()))
            .collect()
    }
}
