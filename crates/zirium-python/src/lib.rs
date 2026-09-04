#![allow(unexpected_cfgs)]

use pyo3::{
    create_exception,
    exceptions::{PyException, PyIOError, PyIndexError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyModule},
};
use std::{
    collections::{HashMap, HashSet},
    io::Read,
    mem::size_of,
    path::PathBuf,
    sync::{Arc, RwLock},
};
use zirium::lexer::TokenKind;
use zirium::{
    NodeId, SyntaxKind,
    dialect::{DialectRegistry, OperationShape as CoreOperationShape},
    parser::{ParseFileError, ParseLimits, ParsedFile},
    printer::{DialectPrintMode, PrintLayout},
    semantic::{
        AttributeId, AttributeSpec, AttributeValue, BlockId, Document as CoreDocument, EditError,
        InsertionPoint, LargeAttributeValue, LoweringMode, OperationId,
        OperationSpec as CoreOperationSpec, RegionId, RetentionProfile, SemanticVerificationError,
        TypeId, TypeSpec, TypeValue, UseSite, ValidationError, ValueId, ValueReference,
        lower_with_dialect_registry_and_retention,
    },
    source::TextRange,
};

create_exception!(zirium._zirium, StaleHandleError, PyException);
create_exception!(zirium._zirium, ForeignHandleError, PyException);
create_exception!(zirium._zirium, SemanticEditError, PyException);
create_exception!(zirium._zirium, StructuralVerificationError, PyException);
create_exception!(zirium._zirium, SemanticVerificationErrorPy, PyException);
create_exception!(zirium._zirium, ResourceLimitError, PyException);

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn py_preserve_error(error: zirium::printer::PreserveError) -> PyErr {
    match error {
        zirium::printer::PreserveError::Io(error) => PyIOError::new_err(error.to_string()),
        error => py_error(error),
    }
}

fn range_tuple(range: Option<TextRange>) -> Option<(u32, u32)> {
    range.map(|range| (range.start(), range.end()))
}

mod editing;
mod registry;
mod semantic;
mod syntax;

use editing::*;
use registry::*;
use semantic::*;
use syntax::*;

fn edit_error(error: EditError) -> PyErr {
    match error {
        EditError::StaleOperation(_) | EditError::StaleBlock(_) | EditError::StaleValue(_) => {
            StaleHandleError::new_err(error.to_string())
        }
        EditError::ForeignOperation(_)
        | EditError::ForeignBlock(_)
        | EditError::ForeignValue(_) => ForeignHandleError::new_err(error.to_string()),
        EditError::Structural(error) => StructuralVerificationError::new_err(error.to_string()),
        EditError::Semantic(error) => SemanticVerificationErrorPy::new_err(error.to_string()),
        _ => SemanticEditError::new_err(error.to_string()),
    }
}

fn structural_error(error: ValidationError) -> PyErr {
    StructuralVerificationError::new_err(error.to_string())
}

fn semantic_error(error: SemanticVerificationError) -> PyErr {
    SemanticVerificationErrorPy::new_err(error.to_string())
}

fn parsed(bytes: Vec<u8>, limits: ParseLimits, registry: RegistryKind) -> PyResult<File> {
    ParsedFile::parse_with_limits_and_registry(bytes, limits, registry.registry())
        .map(|parsed| File {
            parsed: Arc::new(parsed),
            registry,
        })
        .map_err(|error| match error {
            ParseFileError::ResourceLimit(_) => ResourceLimitError::new_err(error.to_string()),
            _ => py_error(error),
        })
}

#[allow(clippy::too_many_arguments)]
fn parse_limits(
    max_file_bytes: Option<usize>,
    max_tokens: Option<usize>,
    max_delimiter_depth: Option<usize>,
    max_payload_bytes: Option<usize>,
    max_numeric_literal_bytes: Option<usize>,
    max_attribute_depth: Option<usize>,
    max_alias_expansion_depth: Option<usize>,
) -> ParseLimits {
    let defaults = ParseLimits::default();
    ParseLimits {
        max_file_bytes: max_file_bytes.unwrap_or(defaults.max_file_bytes),
        max_tokens: max_tokens.unwrap_or(defaults.max_tokens),
        max_delimiter_depth: max_delimiter_depth.unwrap_or(defaults.max_delimiter_depth),
        max_payload_bytes: max_payload_bytes.unwrap_or(defaults.max_payload_bytes),
        max_numeric_literal_bytes: max_numeric_literal_bytes
            .unwrap_or(defaults.max_numeric_literal_bytes),
        max_attribute_depth: max_attribute_depth.unwrap_or(defaults.max_attribute_depth),
        max_alias_expansion_depth: max_alias_expansion_depth
            .unwrap_or(defaults.max_alias_expansion_depth),
    }
}

#[pyfunction(signature = (data, *, registry=None, max_file_bytes=None, max_tokens=None, max_delimiter_depth=None, max_payload_bytes=None, max_numeric_literal_bytes=None, max_attribute_depth=None, max_alias_expansion_depth=None))]
#[allow(clippy::too_many_arguments)]
fn parse_bytes(
    data: &Bound<'_, PyBytes>,
    registry: Option<&DialectRegistryHandle>,
    max_file_bytes: Option<usize>,
    max_tokens: Option<usize>,
    max_delimiter_depth: Option<usize>,
    max_payload_bytes: Option<usize>,
    max_numeric_literal_bytes: Option<usize>,
    max_attribute_depth: Option<usize>,
    max_alias_expansion_depth: Option<usize>,
    py: Python<'_>,
) -> PyResult<File> {
    let bytes = data.as_bytes().to_vec();
    let limits = parse_limits(
        max_file_bytes,
        max_tokens,
        max_delimiter_depth,
        max_payload_bytes,
        max_numeric_literal_bytes,
        max_attribute_depth,
        max_alias_expansion_depth,
    );
    let registry = registry.map_or(RegistryKind::Empty, |registry| registry.kind.clone());
    py.detach(move || parsed(bytes, limits, registry))
}

#[pyfunction(signature = (text, *, registry=None, max_file_bytes=None, max_tokens=None, max_delimiter_depth=None, max_payload_bytes=None, max_numeric_literal_bytes=None, max_attribute_depth=None, max_alias_expansion_depth=None))]
#[allow(clippy::too_many_arguments)]
fn parse_text(
    text: &str,
    registry: Option<&DialectRegistryHandle>,
    max_file_bytes: Option<usize>,
    max_tokens: Option<usize>,
    max_delimiter_depth: Option<usize>,
    max_payload_bytes: Option<usize>,
    max_numeric_literal_bytes: Option<usize>,
    max_attribute_depth: Option<usize>,
    max_alias_expansion_depth: Option<usize>,
    py: Python<'_>,
) -> PyResult<File> {
    let bytes = text.as_bytes().to_vec();
    let limits = parse_limits(
        max_file_bytes,
        max_tokens,
        max_delimiter_depth,
        max_payload_bytes,
        max_numeric_literal_bytes,
        max_attribute_depth,
        max_alias_expansion_depth,
    );
    let registry = registry.map_or(RegistryKind::Empty, |registry| registry.kind.clone());
    py.detach(move || parsed(bytes, limits, registry))
}

#[pyfunction(signature = (path, *, registry=None, max_file_bytes=None, max_tokens=None, max_delimiter_depth=None, max_payload_bytes=None, max_numeric_literal_bytes=None, max_attribute_depth=None, max_alias_expansion_depth=None))]
#[allow(clippy::too_many_arguments)]
fn parse_file(
    path: PathBuf,
    registry: Option<&DialectRegistryHandle>,
    max_file_bytes: Option<usize>,
    max_tokens: Option<usize>,
    max_delimiter_depth: Option<usize>,
    max_payload_bytes: Option<usize>,
    max_numeric_literal_bytes: Option<usize>,
    max_attribute_depth: Option<usize>,
    max_alias_expansion_depth: Option<usize>,
    py: Python<'_>,
) -> PyResult<File> {
    let limits = parse_limits(
        max_file_bytes,
        max_tokens,
        max_delimiter_depth,
        max_payload_bytes,
        max_numeric_literal_bytes,
        max_attribute_depth,
        max_alias_expansion_depth,
    );
    let registry = registry.map_or(RegistryKind::Empty, |registry| registry.kind.clone());
    py.detach(move || {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .map_err(|error| PyIOError::new_err(error.to_string()))?
            .take(limits.max_file_bytes.saturating_add(1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| PyIOError::new_err(error.to_string()))?;
        parsed(bytes, limits, registry)
    })
}

#[pymodule(gil_used = false)]
fn _zirium(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Diagnostic>()?;
    module.add_class::<SyntaxTable>()?;
    module.add_class::<SyntaxOperationTable>()?;
    module.add_class::<SyntaxNode>()?;
    module.add_class::<Token>()?;
    module.add_class::<Operation>()?;
    module.add_class::<File>()?;
    module.add_class::<DialectRegistryHandle>()?;
    module.add_class::<OperationShape>()?;
    module.add_class::<Document>()?;
    module.add_class::<OperationTable>()?;
    module.add_class::<LoweringResult>()?;
    module.add_class::<SemanticDiagnostic>()?;
    module.add_class::<SemanticStatistics>()?;
    module.add_class::<SemanticEdit>()?;
    module.add_class::<OperationSpec>()?;
    module.add_class::<AttributeSpecHandle>()?;
    module.add_class::<SemanticUse>()?;
    module.add_class::<SemanticOperation>()?;
    module.add_class::<SemanticRegion>()?;
    module.add_class::<SemanticBlock>()?;
    module.add_class::<SemanticValue>()?;
    module.add_class::<SemanticType>()?;
    module.add_class::<SemanticAttribute>()?;
    module.add(
        "StaleHandleError",
        module.py().get_type::<StaleHandleError>(),
    )?;
    module.add(
        "ForeignHandleError",
        module.py().get_type::<ForeignHandleError>(),
    )?;
    module.add(
        "SemanticEditError",
        module.py().get_type::<SemanticEditError>(),
    )?;
    module.add(
        "StructuralVerificationError",
        module.py().get_type::<StructuralVerificationError>(),
    )?;
    module.add(
        "SemanticVerificationError",
        module.py().get_type::<SemanticVerificationErrorPy>(),
    )?;
    module.add(
        "ResourceLimitError",
        module.py().get_type::<ResourceLimitError>(),
    )?;
    module.add_function(wrap_pyfunction!(parse_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(parse_text, module)?)?;
    module.add_function(wrap_pyfunction!(parse_file, module)?)?;
    Ok(())
}
