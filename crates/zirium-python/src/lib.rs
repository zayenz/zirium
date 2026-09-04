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

#[derive(Clone)]
enum RegistryKind {
    Empty,
    Core,
    Proving,
    Declarative(Arc<DialectRegistry>),
}

static EMPTY_REGISTRY: DialectRegistry = DialectRegistry::EMPTY;

impl RegistryKind {
    fn registry(&self) -> &DialectRegistry {
        match self {
            Self::Empty => &EMPTY_REGISTRY,
            Self::Core => DialectRegistry::core(),
            Self::Proving => DialectRegistry::proving(),
            Self::Declarative(registry) => registry,
        }
    }
}

#[pyclass(name = "DialectRegistry", frozen, module = "zirium._zirium")]
struct DialectRegistryHandle {
    kind: RegistryKind,
}

#[pyclass(name = "OperationShape", frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct OperationShape {
    shape: CoreOperationShape,
}

#[pymethods]
impl OperationShape {
    #[classattr]
    const FUNC_LIKE: Self = Self {
        shape: CoreOperationShape::FuncLike,
    };

    #[classattr]
    const CALL_LIKE: Self = Self {
        shape: CoreOperationShape::CallLike,
    };
}

#[pymethods]
impl DialectRegistryHandle {
    #[staticmethod]
    fn empty() -> Self {
        Self {
            kind: RegistryKind::Empty,
        }
    }

    #[staticmethod]
    fn proving() -> Self {
        Self {
            kind: RegistryKind::Proving,
        }
    }

    #[staticmethod]
    fn core() -> Self {
        Self {
            kind: RegistryKind::Core,
        }
    }

    #[staticmethod]
    fn declarative(operations: Vec<String>) -> PyResult<Self> {
        let names = operations.iter().map(String::as_str).collect::<Vec<_>>();
        let registry = DialectRegistry::declarative(&names).map_err(py_error)?;
        Ok(Self {
            kind: RegistryKind::Declarative(Arc::new(registry)),
        })
    }

    #[staticmethod]
    fn with_operation_shapes(
        operation_shapes: HashMap<String, PyRef<'_, OperationShape>>,
    ) -> PyResult<Self> {
        let owned = operation_shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.shape))
            .collect::<Vec<_>>();
        let registry = DialectRegistry::with_operation_shapes(&owned).map_err(py_error)?;
        Ok(Self {
            kind: RegistryKind::Declarative(Arc::new(registry)),
        })
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
struct Diagnostic {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    range: (u32, u32),
}

#[pyclass(frozen, module = "zirium._zirium")]
struct SyntaxNode {
    file: Arc<ParsedFile>,
    id: NodeId,
}

#[pymethods]
impl SyntaxNode {
    #[getter]
    fn kind(&self) -> String {
        format!("{:?}", self.node().kind())
    }
    #[getter]
    fn range(&self) -> Option<(u32, u32)> {
        range_tuple(self.node().text_range())
    }
    #[getter]
    fn has_error(&self) -> bool {
        self.node().has_error()
    }
    fn child_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let count = self
            .tree()
            .children(self.id)
            .expect("stored node ID")
            .count();
        PyBytes::new_with(py, count * size_of::<u32>(), |bytes| {
            for (slot, child) in bytes
                .chunks_exact_mut(4)
                .zip(self.tree().children(self.id).expect("stored node ID"))
            {
                slot.copy_from_slice(&(child.index() as u32).to_ne_bytes());
            }
            Ok(())
        })
    }
    fn descendant_range(&self) -> (usize, usize) {
        (
            self.id.index(),
            self.tree().subtree_end(self.id).expect("stored node ID"),
        )
    }
    fn token_range(&self) -> (usize, usize) {
        let range = self.tree().token_indices(self.id).expect("stored node ID");
        (range.start, range.end)
    }
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = self
            .node()
            .text_range()
            .and_then(|range| self.file.source().slice(range))
            .unwrap_or_default();
        PyBytes::new(py, bytes)
    }
    fn as_operation(&self) -> Option<Operation> {
        (self.tree().kind(self.id) == Some(SyntaxKind::Operation)).then(|| Operation {
            file: self.file.clone(),
            id: self.id,
        })
    }
}

fn syntax_table(py: Python<'_>, parsed: &ParsedFile) -> PyResult<SyntaxTable> {
    let tree = parsed.syntax().tree();
    let node_count = tree.node_count();
    let token_count = tree.token_count();
    let node_u16 = |value: fn(&zirium::SyntaxTree, NodeId) -> u16| {
        PyBytes::new_with(py, node_count * 2, |bytes| {
            for (index, slot) in bytes.chunks_exact_mut(2).enumerate() {
                let id = tree.node(index).expect("bounded node index");
                slot.copy_from_slice(&value(tree, id).to_ne_bytes());
            }
            Ok(())
        })
        .map(Bound::unbind)
    };
    let node_u32 = |value: fn(&zirium::SyntaxTree, NodeId) -> u32| {
        PyBytes::new_with(py, node_count * 4, |bytes| {
            for (index, slot) in bytes.chunks_exact_mut(4).enumerate() {
                let id = tree.node(index).expect("bounded node index");
                slot.copy_from_slice(&value(tree, id).to_ne_bytes());
            }
            Ok(())
        })
        .map(Bound::unbind)
    };
    let token_u16 = PyBytes::new_with(py, token_count * 2, |bytes| {
        for (index, slot) in bytes.chunks_exact_mut(2).enumerate() {
            slot.copy_from_slice(&token_kind_code(tree.token_kind(index).unwrap()).to_ne_bytes());
        }
        Ok(())
    })?
    .unbind();
    let token_u32 = |end: bool| {
        PyBytes::new_with(py, token_count * 4, |bytes| {
            for (index, slot) in bytes.chunks_exact_mut(4).enumerate() {
                let range = tree.token(index).unwrap().range();
                let value = if end { range.end() } else { range.start() };
                slot.copy_from_slice(&value.to_ne_bytes());
            }
            Ok(())
        })
        .map(Bound::unbind)
    };
    Ok(SyntaxTable {
        node_kind: node_u16(|tree, id| syntax_kind_code(tree.kind(id).unwrap()))?,
        node_start: node_u32(|tree, id| tree.text_range(id).map_or(u32::MAX, |r| r.start()))?,
        node_end: node_u32(|tree, id| tree.text_range(id).map_or(u32::MAX, |r| r.end()))?,
        node_subtree_end: node_u32(|tree, id| tree.subtree_end(id).unwrap() as u32)?,
        node_flags: PyBytes::new_with(py, node_count, |bytes| {
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = u8::from(tree.has_error(tree.node(index).unwrap()).unwrap());
            }
            Ok(())
        })?
        .unbind(),
        token_kind: token_u16,
        token_start: token_u32(false)?,
        token_end: token_u32(true)?,
    })
}

#[pyclass(frozen, module = "zirium._zirium")]
struct Token {
    file: Arc<ParsedFile>,
    id: usize,
}

#[pyclass(frozen, module = "zirium._zirium")]
struct SyntaxTable {
    #[pyo3(get)]
    node_kind: Py<PyBytes>,
    #[pyo3(get)]
    node_start: Py<PyBytes>,
    #[pyo3(get)]
    node_end: Py<PyBytes>,
    #[pyo3(get)]
    node_subtree_end: Py<PyBytes>,
    #[pyo3(get)]
    node_flags: Py<PyBytes>,
    #[pyo3(get)]
    token_kind: Py<PyBytes>,
    #[pyo3(get)]
    token_start: Py<PyBytes>,
    #[pyo3(get)]
    token_end: Py<PyBytes>,
}

#[pyclass(frozen, module = "zirium._zirium")]
struct SyntaxOperationTable {
    #[pyo3(get)]
    operation_node: Py<PyBytes>,
    #[pyo3(get)]
    result_offsets: Py<PyBytes>,
    #[pyo3(get)]
    result_nodes: Py<PyBytes>,
    #[pyo3(get)]
    operand_offsets: Py<PyBytes>,
    #[pyo3(get)]
    operand_nodes: Py<PyBytes>,
    #[pyo3(get)]
    successor_offsets: Py<PyBytes>,
    #[pyo3(get)]
    successor_nodes: Py<PyBytes>,
    #[pyo3(get)]
    region_offsets: Py<PyBytes>,
    #[pyo3(get)]
    region_nodes: Py<PyBytes>,
}

#[derive(Clone, Copy)]
enum SyntaxRelationship {
    Result,
    Operand,
    Successor,
    Region,
}

fn relationship_count(
    operation: zirium::parser::OperationSyntax<'_>,
    relationship: SyntaxRelationship,
) -> usize {
    match relationship {
        SyntaxRelationship::Result => operation.results().count(),
        SyntaxRelationship::Operand => operation.operands().count(),
        SyntaxRelationship::Successor => operation.successors().count(),
        SyntaxRelationship::Region => operation.regions().count(),
    }
}

fn relationship_offsets(
    parsed: &ParsedFile,
    relationship: SyntaxRelationship,
) -> impl Iterator<Item = u32> + '_ {
    let mut total = 0;
    std::iter::once(0).chain(parsed.syntax().file().operations().map(move |operation| {
        total += relationship_count(operation, relationship) as u32;
        total
    }))
}

fn relationship_column<'a>(
    parsed: &'a ParsedFile,
    relationship: SyntaxRelationship,
) -> Box<dyn Iterator<Item = u32> + 'a> {
    let file = parsed.syntax().file();
    match relationship {
        SyntaxRelationship::Result => Box::new(
            file.operations()
                .flat_map(|operation| operation.results().map(|node| node.id().index() as u32)),
        ),
        SyntaxRelationship::Operand => Box::new(
            file.operations()
                .flat_map(|operation| operation.operands().map(|node| node.id().index() as u32)),
        ),
        SyntaxRelationship::Successor => Box::new(
            file.operations()
                .flat_map(|operation| operation.successors().map(|node| node.id().index() as u32)),
        ),
        SyntaxRelationship::Region => Box::new(
            file.operations()
                .flat_map(|operation| operation.regions().map(|node| node.id().index() as u32)),
        ),
    }
}

fn fill_u32_bytes(bytes: &mut [u8], values: impl Iterator<Item = u32>) {
    for (slot, value) in bytes.chunks_exact_mut(4).zip(values) {
        slot.copy_from_slice(&value.to_ne_bytes());
    }
}

fn u32_bytes(
    py: Python<'_>,
    count: usize,
    values: impl Iterator<Item = u32>,
) -> PyResult<Py<PyBytes>> {
    Ok(PyBytes::new_with(py, count * size_of::<u32>(), |bytes| {
        fill_u32_bytes(bytes, values);
        Ok(())
    })?
    .unbind())
}

fn syntax_operation_table(py: Python<'_>, parsed: &ParsedFile) -> PyResult<SyntaxOperationTable> {
    let operation_count = parsed.syntax().file().operations().count();
    Ok(SyntaxOperationTable {
        operation_node: u32_bytes(
            py,
            operation_count,
            parsed
                .syntax()
                .file()
                .operations()
                .map(|op| op.id().index() as u32),
        )?,
        result_offsets: u32_bytes(
            py,
            operation_count + 1,
            relationship_offsets(parsed, SyntaxRelationship::Result),
        )?,
        result_nodes: u32_bytes(
            py,
            relationship_column(parsed, SyntaxRelationship::Result).count(),
            relationship_column(parsed, SyntaxRelationship::Result),
        )?,
        operand_offsets: u32_bytes(
            py,
            operation_count + 1,
            relationship_offsets(parsed, SyntaxRelationship::Operand),
        )?,
        operand_nodes: u32_bytes(
            py,
            relationship_column(parsed, SyntaxRelationship::Operand).count(),
            relationship_column(parsed, SyntaxRelationship::Operand),
        )?,
        successor_offsets: u32_bytes(
            py,
            operation_count + 1,
            relationship_offsets(parsed, SyntaxRelationship::Successor),
        )?,
        successor_nodes: u32_bytes(
            py,
            relationship_column(parsed, SyntaxRelationship::Successor).count(),
            relationship_column(parsed, SyntaxRelationship::Successor),
        )?,
        region_offsets: u32_bytes(
            py,
            operation_count + 1,
            relationship_offsets(parsed, SyntaxRelationship::Region),
        )?,
        region_nodes: u32_bytes(
            py,
            relationship_column(parsed, SyntaxRelationship::Region).count(),
            relationship_column(parsed, SyntaxRelationship::Region),
        )?,
    })
}

#[pymethods]
impl SyntaxTable {
    #[staticmethod]
    fn node_kind_code(name: &str) -> PyResult<u16> {
        syntax_kind_from_name(name)
            .map(syntax_kind_code)
            .ok_or_else(|| PyValueError::new_err(format!("unknown syntax kind: {name}")))
    }
    #[staticmethod]
    fn node_kind_name(code: u16) -> PyResult<&'static str> {
        SYNTAX_KINDS
            .get(code as usize)
            .copied()
            .map(syntax_kind_name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown syntax kind code: {code}")))
    }
    #[staticmethod]
    fn token_kind_code(name: &str) -> PyResult<u16> {
        TOKEN_KINDS
            .iter()
            .position(|kind| token_kind_name(*kind) == name)
            .map(|code| code as u16)
            .ok_or_else(|| PyValueError::new_err(format!("unknown token kind: {name}")))
    }
    #[staticmethod]
    fn token_kind_name(code: u16) -> PyResult<&'static str> {
        TOKEN_KINDS
            .get(code as usize)
            .copied()
            .map(token_kind_name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown token kind code: {code}")))
    }
}

#[pymethods]
impl Token {
    #[getter]
    fn kind(&self) -> String {
        format!("{:?}", self.token().kind())
    }
    #[getter]
    fn range(&self) -> (u32, u32) {
        let range = self.token().range();
        (range.start(), range.end())
    }
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(
            py,
            self.file
                .source()
                .slice(self.token().range())
                .unwrap_or_default(),
        )
    }
}

impl Token {
    fn token(&self) -> &zirium::lexer::Token {
        self.file
            .syntax()
            .tree()
            .token(self.id)
            .expect("stored token ID belongs to parsed file")
    }
}

impl SyntaxNode {
    fn tree(&self) -> &zirium::SyntaxTree {
        self.file.syntax().tree()
    }
    fn node(&self) -> zirium::parser::SyntaxNode<'_> {
        self.file
            .syntax()
            .file()
            .node(self.id)
            .expect("stored node ID belongs to parsed file")
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
struct Operation {
    file: Arc<ParsedFile>,
    id: NodeId,
}

#[pymethods]
impl Operation {
    #[getter]
    fn range(&self) -> Option<(u32, u32)> {
        range_tuple(self.file.syntax().tree().text_range(self.id))
    }
    #[getter]
    fn has_error(&self) -> bool {
        self.file.syntax().tree().has_error(self.id).unwrap_or(true)
    }
    fn syntax(&self) -> SyntaxNode {
        SyntaxNode {
            file: self.file.clone(),
            id: self.id,
        }
    }
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.syntax().text(py)
    }
    fn properties(&self) -> Option<SyntaxNode> {
        self.view().properties().map(|view| self.wrap(view.id()))
    }
    fn attributes(&self) -> Option<SyntaxNode> {
        self.view().attributes().map(|view| self.wrap(view.id()))
    }
    fn trailing_location(&self) -> Option<SyntaxNode> {
        self.view()
            .trailing_location()
            .map(|view| self.wrap(view.id()))
    }
}

impl Operation {
    fn view(&self) -> zirium::parser::OperationSyntax<'_> {
        self.file
            .syntax()
            .file()
            .operation(self.id)
            .expect("stored operation ID belongs to parsed file")
    }
    fn wrap(&self, id: NodeId) -> SyntaxNode {
        SyntaxNode {
            file: self.file.clone(),
            id,
        }
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
struct File {
    parsed: Arc<ParsedFile>,
    registry: RegistryKind,
}

#[pymethods]
impl File {
    #[getter]
    fn root(&self) -> SyntaxNode {
        SyntaxNode {
            file: self.parsed.clone(),
            id: self.parsed.syntax().tree().root(),
        }
    }
    #[getter]
    fn diagnostics(&self) -> Vec<Diagnostic> {
        let lexer = self
            .parsed
            .lexer_diagnostics()
            .iter()
            .map(|diagnostic| Diagnostic {
                kind: format!("lexer.{:?}", diagnostic.kind()),
                range: (diagnostic.range().start(), diagnostic.range().end()),
            });
        let parser = self
            .parsed
            .syntax()
            .diagnostics()
            .iter()
            .map(|diagnostic| Diagnostic {
                kind: format!("parser.{:?}", diagnostic.kind()),
                range: (diagnostic.range().start(), diagnostic.range().end()),
            });
        lexer.chain(parser).collect()
    }
    fn original_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.parsed.original_bytes())
    }
    fn write_original(&self, path: PathBuf, py: Python<'_>) -> PyResult<()> {
        let parsed = self.parsed.clone();
        py.detach(move || std::fs::write(path, parsed.original_bytes()))
            .map_err(|error| PyIOError::new_err(error.to_string()))
    }
    #[getter]
    fn node_count(&self) -> usize {
        self.parsed.syntax().tree().node_count()
    }
    #[getter]
    fn token_count(&self) -> usize {
        self.parsed.syntax().tree().token_count()
    }
    fn node(&self, index: usize) -> PyResult<SyntaxNode> {
        let id =
            self.parsed.syntax().tree().node(index).ok_or_else(|| {
                PyIndexError::new_err(format!("node index out of range: {index}"))
            })?;
        Ok(SyntaxNode {
            file: self.parsed.clone(),
            id,
        })
    }
    fn token(&self, index: usize) -> PyResult<Token> {
        self.parsed
            .syntax()
            .tree()
            .token(index)
            .ok_or_else(|| PyIndexError::new_err(format!("token index out of range: {index}")))?;
        Ok(Token {
            file: self.parsed.clone(),
            id: index,
        })
    }
    fn syntax_table(&self, py: Python<'_>) -> PyResult<SyntaxTable> {
        syntax_table(py, &self.parsed)
    }
    #[getter]
    fn operation_count(&self) -> usize {
        self.parsed.syntax().file().operations().count()
    }
    fn operation(&self, index: usize) -> PyResult<Operation> {
        let id = self
            .parsed
            .syntax()
            .file()
            .operations()
            .nth(index)
            .ok_or_else(|| PyIndexError::new_err(format!("operation index out of range: {index}")))?
            .id();
        Ok(Operation {
            file: self.parsed.clone(),
            id,
        })
    }
    fn operation_table(&self, py: Python<'_>) -> PyResult<SyntaxOperationTable> {
        syntax_operation_table(py, &self.parsed)
    }
    #[pyo3(signature = (retention=None))]
    fn lower_strict(&self, retention: Option<&str>, py: Python<'_>) -> PyResult<LoweringResult> {
        self.lower(LoweringMode::Strict, retention, py)
    }

    #[pyo3(signature = (retention=None))]
    fn lower_best_effort(
        &self,
        retention: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<LoweringResult> {
        self.lower(LoweringMode::BestEffort, retention, py)
    }
}

impl File {
    fn lower(
        &self,
        mode: LoweringMode,
        retention: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<LoweringResult> {
        let retention = parse_retention(retention.unwrap_or("semantic"))?;
        let parsed = self.parsed.clone();
        let registry = self.registry.clone();
        let result = py.detach(move || {
            lower_with_dialect_registry_and_retention(&parsed, mode, retention, registry.registry())
        });
        let diagnostics = result
            .diagnostics
            .into_iter()
            .map(|diagnostic| SemanticDiagnostic {
                range: (diagnostic.range.start(), diagnostic.range.end()),
                message: diagnostic.message,
            })
            .collect();
        Ok(LoweringResult {
            document: result
                .document
                .map(|document| Document::new(document, self.registry.clone())),
            diagnostics,
            semantically_complete: result.semantically_complete,
        })
    }
}

fn parse_retention(value: &str) -> PyResult<RetentionProfile> {
    match value.replace(['_', '-'], "").to_ascii_lowercase().as_str() {
        "semantic" | "semanticonly" => Ok(RetentionProfile::SemanticOnly),
        "syntax" | "syntaxonly" => Ok(RetentionProfile::SyntaxOnly),
        "hybrid" => Ok(RetentionProfile::Hybrid),
        _ => Err(PyValueError::new_err(
            "retention must be 'semantic', 'syntax', or 'hybrid'",
        )),
    }
}

type SharedDocument = Arc<RwLock<CoreDocument>>;

#[pyclass(frozen, module = "zirium._zirium")]
struct OperationTable {
    #[pyo3(get)]
    revision: u64,
    #[pyo3(get)]
    name_code: Py<PyBytes>,
    #[pyo3(get)]
    source_start: Py<PyBytes>,
    #[pyo3(get)]
    source_end: Py<PyBytes>,
    #[pyo3(get)]
    root_flags: Py<PyBytes>,
    #[pyo3(get)]
    name_offsets: Py<PyBytes>,
    #[pyo3(get)]
    name_bytes: Py<PyBytes>,
    state: SharedDocument,
    ids: Vec<OperationId>,
}

#[pymethods]
impl OperationTable {
    #[getter]
    fn count(&self) -> usize {
        self.ids.len()
    }

    fn operation(&self, index: usize) -> PyResult<SemanticOperation> {
        let id = *self
            .ids
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("operation index out of range"))?;
        read_document(&self.state)?
            .operation(id)
            .ok_or_else(|| stale("operation"))?;
        Ok(SemanticOperation::new(self.state.clone(), id))
    }
}

fn read_document(state: &SharedDocument) -> PyResult<std::sync::RwLockReadGuard<'_, CoreDocument>> {
    state
        .read()
        .map_err(|_| PyValueError::new_err("semantic document lock is poisoned"))
}

fn stale(kind: &str) -> PyErr {
    StaleHandleError::new_err(format!("stale semantic {kind} handle"))
}

fn same_document(left: &SharedDocument, right: &SharedDocument) -> PyResult<()> {
    if Arc::ptr_eq(left, right) {
        Ok(())
    } else {
        Err(ForeignHandleError::new_err(
            "semantic handle belongs to another document",
        ))
    }
}

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

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct AttributeSpecHandle {
    state: SharedDocument,
    spec: AttributeSpec,
}

#[pymethods]
impl AttributeSpecHandle {
    #[new]
    #[pyo3(signature = (attribute, name=None))]
    fn new(attribute: &SemanticAttribute, name: Option<String>) -> PyResult<Self> {
        let document = read_document(&attribute.state)?;
        let value = document
            .attribute_value(attribute.id)
            .cloned()
            .ok_or_else(|| stale("attribute"))?;
        let spelling = document
            .attribute_spelling_value(attribute.id)
            .map(str::to_owned)
            .ok_or_else(|| stale("attribute"))?;
        drop(document);
        Ok(Self {
            state: attribute.state.clone(),
            spec: AttributeSpec {
                name: name.unwrap_or_else(|| attribute.name.clone()),
                spelling,
                value,
            },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct OperationSpec {
    state: SharedDocument,
    spec: CoreOperationSpec,
}

#[pymethods]
impl OperationSpec {
    #[new]
    #[pyo3(signature = (name, operands, result_types, function_type, attributes=Vec::new(), properties=Vec::new()))]
    fn new(
        name: String,
        operands: Vec<SemanticValue>,
        result_types: Vec<SemanticType>,
        function_type: SemanticType,
        attributes: Vec<AttributeSpecHandle>,
        properties: Vec<AttributeSpecHandle>,
    ) -> PyResult<Self> {
        let state = function_type.state.clone();
        for value in &operands {
            same_document(&state, &value.state)?;
        }
        for ty in &result_types {
            same_document(&state, &ty.state)?;
        }
        for attribute in attributes.iter().chain(&properties) {
            same_document(&state, &attribute.state)?;
        }
        let document = read_document(&state)?;
        let operands = operands
            .into_iter()
            .map(|value| match value.value {
                ValueReference::Resolved(value) => Ok(value),
                ValueReference::Invalid(_) => Err(SemanticEditError::new_err(
                    "operation operands must be valid semantic values",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let result_types = result_types
            .into_iter()
            .map(|ty| type_spec(&document, ty.id))
            .collect::<PyResult<Vec<_>>>()?;
        let function_type = type_spec(&document, function_type.id)?;
        drop(document);
        Ok(Self {
            state,
            spec: CoreOperationSpec {
                name,
                operands,
                result_types,
                function_type,
                attributes: attributes.into_iter().map(|value| value.spec).collect(),
                properties: properties.into_iter().map(|value| value.spec).collect(),
            },
        })
    }
}

fn type_spec(document: &CoreDocument, id: TypeId) -> PyResult<TypeSpec> {
    Ok(TypeSpec {
        spelling: document
            .type_spelling(id)
            .map(str::to_owned)
            .ok_or_else(|| stale("type"))?,
        value: document
            .type_value(id)
            .cloned()
            .ok_or_else(|| stale("type"))?,
    })
}

#[derive(Clone)]
enum EditCommand {
    Insert(InsertionPoint, CoreOperationSpec),
    Erase(OperationId),
    RewireOperand(OperationId, usize, ValueId),
    RewireSuccessorArgument(OperationId, usize, usize, ValueId),
    ReplaceResultTypes(OperationId, Vec<TypeSpec>),
    SetAttribute(OperationId, AttributeSpec),
    RemoveAttribute(OperationId, String),
    SetProperty(OperationId, AttributeSpec),
    RemoveProperty(OperationId, String),
    ReplaceAllUses(ValueId, ValueId),
    CompactPools,
}

#[pyclass(module = "zirium._zirium")]
struct SemanticEdit {
    state: SharedDocument,
    registry: RegistryKind,
    commands: Vec<EditCommand>,
    entered: bool,
    closed: bool,
}

impl SemanticEdit {
    fn ensure_open(&self) -> PyResult<()> {
        if self.entered && !self.closed {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "semantic edit commands require an active context manager",
            ))
        }
    }

    fn operation(&self, operation: &SemanticOperation) -> PyResult<OperationId> {
        same_document(&self.state, &operation.state)?;
        Ok(operation.id)
    }

    fn block(&self, block: &SemanticBlock) -> PyResult<BlockId> {
        same_document(&self.state, &block.state)?;
        Ok(block.id)
    }

    fn value(&self, value: &SemanticValue) -> PyResult<ValueId> {
        same_document(&self.state, &value.state)?;
        match value.value {
            ValueReference::Resolved(value) => Ok(value),
            ValueReference::Invalid(_) => Err(SemanticEditError::new_err(
                "edit commands require a valid semantic value",
            )),
        }
    }
}

#[pymethods]
impl SemanticEdit {
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        if slf.entered || slf.closed {
            return Err(PyRuntimeError::new_err(
                "semantic edit context cannot be entered more than once",
            ));
        }
        slf.entered = true;
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
        py: Python<'_>,
    ) -> PyResult<bool> {
        self.ensure_open()?;
        self.closed = true;
        if exc_type.is_some() {
            self.commands.clear();
            return Ok(false);
        }
        let state = self.state.clone();
        let registry = self.registry.clone();
        let commands = std::mem::take(&mut self.commands);
        py.detach(move || {
            let mut document = state
                .write()
                .map_err(|_| PyValueError::new_err("semantic document lock is poisoned"))?;
            let mut editor = document.edit(registry.registry()).map_err(edit_error)?;
            for command in commands {
                match command {
                    EditCommand::Insert(point, spec) => {
                        editor.insert(point, spec).map_err(edit_error)?;
                    }
                    EditCommand::Erase(operation) => editor.erase(operation).map_err(edit_error)?,
                    EditCommand::RewireOperand(operation, index, value) => editor
                        .rewire_operand(operation, index, value)
                        .map_err(edit_error)?,
                    EditCommand::RewireSuccessorArgument(operation, successor, argument, value) => {
                        editor
                            .rewire_successor_argument(operation, successor, argument, value)
                            .map_err(edit_error)?
                    }
                    EditCommand::ReplaceResultTypes(operation, types) => editor
                        .replace_result_types(operation, &types)
                        .map_err(edit_error)?,
                    EditCommand::SetAttribute(operation, value) => {
                        editor.set_attribute(operation, value).map_err(edit_error)?
                    }
                    EditCommand::RemoveAttribute(operation, name) => editor
                        .remove_attribute(operation, &name)
                        .map_err(edit_error)?,
                    EditCommand::SetProperty(operation, value) => {
                        editor.set_property(operation, value).map_err(edit_error)?
                    }
                    EditCommand::RemoveProperty(operation, name) => editor
                        .remove_property(operation, &name)
                        .map_err(edit_error)?,
                    EditCommand::ReplaceAllUses(from, to) => {
                        editor.replace_all_uses(from, to).map_err(edit_error)?;
                    }
                    EditCommand::CompactPools => {
                        editor.compact_pools();
                    }
                }
            }
            editor.commit().map_err(edit_error)
        })?;
        Ok(false)
    }

    fn insert_root(&mut self, index: usize, spec: &OperationSpec) -> PyResult<()> {
        self.ensure_open()?;
        same_document(&self.state, &spec.state)?;
        self.commands.push(EditCommand::Insert(
            InsertionPoint::Root(index),
            spec.spec.clone(),
        ));
        Ok(())
    }

    fn insert(
        &mut self,
        block: &SemanticBlock,
        index: usize,
        spec: &OperationSpec,
    ) -> PyResult<()> {
        self.ensure_open()?;
        same_document(&self.state, &spec.state)?;
        let block = self.block(block)?;
        self.commands.push(EditCommand::Insert(
            InsertionPoint::Block { block, index },
            spec.spec.clone(),
        ));
        Ok(())
    }

    fn erase(&mut self, operation: &SemanticOperation) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        self.commands.push(EditCommand::Erase(operation));
        Ok(())
    }

    fn rewire_operand(
        &mut self,
        operation: &SemanticOperation,
        index: usize,
        value: &SemanticValue,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        let value = self.value(value)?;
        self.commands
            .push(EditCommand::RewireOperand(operation, index, value));
        Ok(())
    }

    fn rewire_successor_argument(
        &mut self,
        operation: &SemanticOperation,
        successor: usize,
        argument: usize,
        value: &SemanticValue,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        let value = self.value(value)?;
        self.commands.push(EditCommand::RewireSuccessorArgument(
            operation, successor, argument, value,
        ));
        Ok(())
    }

    fn replace_result_types(
        &mut self,
        operation: &SemanticOperation,
        types: Vec<SemanticType>,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        for ty in &types {
            same_document(&self.state, &ty.state)?;
        }
        let document = read_document(&self.state)?;
        let types = types
            .into_iter()
            .map(|ty| type_spec(&document, ty.id))
            .collect::<PyResult<Vec<_>>>()?;
        drop(document);
        self.commands
            .push(EditCommand::ReplaceResultTypes(operation, types));
        Ok(())
    }

    fn set_attribute(
        &mut self,
        operation: &SemanticOperation,
        value: &AttributeSpecHandle,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        same_document(&self.state, &value.state)?;
        self.commands
            .push(EditCommand::SetAttribute(operation, value.spec.clone()));
        Ok(())
    }

    fn remove_attribute(&mut self, operation: &SemanticOperation, name: String) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        self.commands
            .push(EditCommand::RemoveAttribute(operation, name));
        Ok(())
    }

    fn set_property(
        &mut self,
        operation: &SemanticOperation,
        value: &AttributeSpecHandle,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        same_document(&self.state, &value.state)?;
        self.commands
            .push(EditCommand::SetProperty(operation, value.spec.clone()));
        Ok(())
    }

    fn remove_property(&mut self, operation: &SemanticOperation, name: String) -> PyResult<()> {
        self.ensure_open()?;
        let operation = self.operation(operation)?;
        self.commands
            .push(EditCommand::RemoveProperty(operation, name));
        Ok(())
    }

    fn replace_all_uses(&mut self, from: &SemanticValue, to: &SemanticValue) -> PyResult<()> {
        self.ensure_open()?;
        let from = self.value(from)?;
        let to = self.value(to)?;
        self.commands.push(EditCommand::ReplaceAllUses(from, to));
        Ok(())
    }

    fn compact_pools(&mut self) -> PyResult<()> {
        self.ensure_open()?;
        self.commands.push(EditCommand::CompactPools);
        Ok(())
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
struct SemanticUse {
    #[pyo3(get)]
    kind: &'static str,
    #[pyo3(get)]
    operation: SemanticOperation,
    #[pyo3(get)]
    index: u32,
    #[pyo3(get)]
    successor: Option<u32>,
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct Document {
    state: SharedDocument,
    registry: RegistryKind,
}

impl Document {
    fn new(document: CoreDocument, registry: RegistryKind) -> Self {
        Self {
            state: Arc::new(RwLock::new(document)),
            registry,
        }
    }
}

#[pymethods]
impl Document {
    #[getter]
    fn semantically_complete(&self) -> PyResult<bool> {
        Ok(read_document(&self.state)?.is_semantically_complete())
    }

    #[getter]
    fn retention(&self) -> PyResult<&'static str> {
        Ok(match read_document(&self.state)?.retention_profile() {
            RetentionProfile::SemanticOnly => "semantic",
            RetentionProfile::SyntaxOnly => "syntax",
            RetentionProfile::Hybrid => "hybrid",
        })
    }

    #[getter]
    fn diagnostics(&self) -> PyResult<Vec<SemanticDiagnostic>> {
        Ok(read_document(&self.state)?
            .diagnostics()
            .iter()
            .map(|diagnostic| SemanticDiagnostic {
                range: (diagnostic.range.start(), diagnostic.range.end()),
                message: diagnostic.message.clone(),
            })
            .collect())
    }

    #[pyo3(signature = (name=None))]
    fn operation_table(&self, py: Python<'_>, name: Option<&str>) -> PyResult<OperationTable> {
        let document = read_document(&self.state)?;
        let filter = match name {
            Some(name) => match document.existing_string_index(name) {
                Some(index) => Some(index),
                None => u32::MAX.into(),
            },
            None => None,
        };
        let mut ids = Vec::new();
        let mut dictionary = Vec::new();
        let mut dense = HashMap::new();
        for id in document.operations() {
            let stored = document.operation_name_index(id).expect("live operation");
            if filter.is_some_and(|wanted| wanted != stored) {
                continue;
            }
            let next = dictionary.len() as u32;
            dense.entry(stored).or_insert_with(|| {
                dictionary.push(stored);
                next
            });
            ids.push(id);
        }
        let roots = document
            .root_operations()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let name_code = PyBytes::new_with(py, ids.len() * 4, |bytes| {
            for (slot, id) in bytes.chunks_exact_mut(4).zip(&ids) {
                let stored = document.operation_name_index(*id).expect("live operation");
                slot.copy_from_slice(&dense[&stored].to_ne_bytes());
            }
            Ok(())
        })?
        .unbind();
        let source_column = |end: bool| {
            PyBytes::new_with(py, ids.len() * 4, |bytes| {
                for (slot, id) in bytes.chunks_exact_mut(4).zip(&ids) {
                    let value = document
                        .operation_source_range(*id)
                        .map_or(u32::MAX, |range| {
                            if range.start() == 0 && range.end() == 0 {
                                u32::MAX
                            } else if end {
                                range.end()
                            } else {
                                range.start()
                            }
                        });
                    slot.copy_from_slice(&value.to_ne_bytes());
                }
                Ok(())
            })
            .map(Bound::unbind)
        };
        let name_bytes_len = dictionary
            .iter()
            .map(|&index| {
                document
                    .string_at(index)
                    .expect("stored string index")
                    .len()
            })
            .sum();
        let name_bytes = PyBytes::new_with(py, name_bytes_len, |bytes| {
            let mut cursor = 0;
            for &index in &dictionary {
                let name = document.string_at(index).expect("stored string index");
                bytes[cursor..cursor + name.len()].copy_from_slice(name.as_bytes());
                cursor += name.len();
            }
            Ok(())
        })?
        .unbind();
        let name_offsets = PyBytes::new_with(py, (dictionary.len() + 1) * 4, |bytes| {
            let mut cursor = 0u32;
            bytes[..4].copy_from_slice(&cursor.to_ne_bytes());
            for (slot, &index) in bytes[4..].chunks_exact_mut(4).zip(&dictionary) {
                cursor += document
                    .string_at(index)
                    .expect("stored string index")
                    .len() as u32;
                slot.copy_from_slice(&cursor.to_ne_bytes());
            }
            Ok(())
        })?
        .unbind();
        let root_flags = PyBytes::new_with(py, ids.len(), |bytes| {
            for (byte, id) in bytes.iter_mut().zip(&ids) {
                *byte = u8::from(roots.contains(id));
            }
            Ok(())
        })?
        .unbind();
        Ok(OperationTable {
            revision: document.revision(),
            name_code,
            source_start: source_column(false)?,
            source_end: source_column(true)?,
            root_flags,
            name_offsets,
            name_bytes,
            state: self.state.clone(),
            ids,
        })
    }

    fn statistics(&self) -> PyResult<SemanticStatistics> {
        let value = read_document(&self.state)?.statistics();
        Ok(SemanticStatistics {
            operations: value.operations,
            regions: value.regions,
            blocks: value.blocks,
            local_types: value.local_types,
            local_attributes: value.local_attributes,
            payload_blobs: value.payload_blobs,
            payload_blob_bytes: value.payload_blob_bytes,
            retained_source_bytes: value.retained_source_bytes,
            direct_owned_bytes: value.direct_owned_bytes,
            document_index_bytes: value.document_index_bytes,
            retained_cst_bytes: value.retained_cst_bytes,
            source_storage_shared: value.source_storage_shared,
            cst_storage_shared: value.cst_storage_shared,
            pooled_list_entries: value.pooled_list_entries,
            use_index_entries: value.use_index_entries,
            symbol_index_entries: value.symbol_index_entries,
            dominance_index_entries: value.dominance_index_entries,
        })
    }

    fn edit(&self) -> SemanticEdit {
        SemanticEdit {
            state: self.state.clone(),
            registry: self.registry.clone(),
            commands: Vec::new(),
            entered: false,
            closed: false,
        }
    }

    fn uses(&self, value: &SemanticValue) -> PyResult<Vec<SemanticUse>> {
        same_document(&self.state, &value.state)?;
        let value = match value.value {
            ValueReference::Resolved(value) => value,
            ValueReference::Invalid(_) => return Err(stale("value")),
        };
        let sites = read_document(&self.state)?
            .checked_uses(value)
            .map_err(edit_error)?;
        Ok(sites
            .into_iter()
            .map(|site| match site {
                UseSite::Operand { operation, index } => SemanticUse {
                    kind: "operand",
                    operation: SemanticOperation::new(self.state.clone(), operation),
                    index,
                    successor: None,
                },
                UseSite::SuccessorArgument {
                    operation,
                    successor,
                    argument,
                } => SemanticUse {
                    kind: "successor_argument",
                    operation: SemanticOperation::new(self.state.clone(), operation),
                    index: argument,
                    successor: Some(successor),
                },
            })
            .collect())
    }

    fn lookup_symbol(
        &self,
        from_operation: &SemanticOperation,
        symbol: &str,
    ) -> PyResult<Option<SemanticOperation>> {
        same_document(&self.state, &from_operation.state)?;
        let result = read_document(&self.state)?
            .checked_lookup_symbol(from_operation.id, symbol, self.registry.registry())
            .map_err(edit_error)?;
        Ok(result.map(|id| SemanticOperation::new(self.state.clone(), id)))
    }

    fn symbol_diagnostics(&self) -> PyResult<Vec<(SemanticOperation, String)>> {
        let diagnostics =
            read_document(&self.state)?.symbol_index_diagnostics(self.registry.registry());
        Ok(diagnostics
            .into_iter()
            .map(|diagnostic| {
                (
                    SemanticOperation::new(self.state.clone(), diagnostic.operation),
                    diagnostic.symbol,
                )
            })
            .collect())
    }

    fn dominates(&self, value: &SemanticValue, operation: &SemanticOperation) -> PyResult<bool> {
        same_document(&self.state, &value.state)?;
        same_document(&self.state, &operation.state)?;
        let value = match value.value {
            ValueReference::Resolved(value) => value,
            ValueReference::Invalid(_) => return Err(stale("value")),
        };
        read_document(&self.state)?
            .checked_dominates(value, operation.id, self.registry.registry())
            .map_err(edit_error)
    }

    fn validate_structure(&self, py: Python<'_>) -> PyResult<()> {
        let state = self.state.clone();
        py.detach(move || {
            read_document(&state)?
                .validate_structure()
                .map_err(structural_error)
        })
    }

    fn verify_semantics(&self, py: Python<'_>) -> PyResult<()> {
        let state = self.state.clone();
        let registry = self.registry.clone();
        py.detach(move || {
            read_document(&state)?
                .verify_semantics(registry.registry())
                .map_err(semantic_error)
        })
    }

    #[pyo3(signature = (compact=false))]
    fn custom_bytes<'py>(&self, py: Python<'py>, compact: bool) -> PyResult<Bound<'py, PyBytes>> {
        let state = self.state.clone();
        let registry = self.registry.clone();
        let bytes = py.detach(move || {
            let document = read_document(&state)?;
            let mut bytes = String::new();
            document
                .print_with_registry(
                    &mut bytes,
                    if compact {
                        PrintLayout::Compact
                    } else {
                        PrintLayout::Pretty
                    },
                    DialectPrintMode::PreferCustom,
                    registry.registry(),
                )
                .map_err(py_error)?;
            Ok::<_, PyErr>(bytes.into_bytes())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(signature = (path, compact=false))]
    fn write_custom(&self, path: PathBuf, compact: bool, py: Python<'_>) -> PyResult<()> {
        let state = self.state.clone();
        let registry = self.registry.clone();
        py.detach(move || {
            read_document(&state)?
                .print_with_registry_to_file(
                    path,
                    if compact {
                        PrintLayout::Compact
                    } else {
                        PrintLayout::Pretty
                    },
                    DialectPrintMode::PreferCustom,
                    registry.registry(),
                )
                .map_err(|error| PyIOError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (compact=false))]
    fn canonical_bytes<'py>(
        &self,
        py: Python<'py>,
        compact: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let state = self.state.clone();
        let bytes = py.detach(move || {
            let document = read_document(&state)?;
            let mut bytes = Vec::new();
            document
                .print_io(
                    &mut bytes,
                    if compact {
                        PrintLayout::Compact
                    } else {
                        PrintLayout::Pretty
                    },
                )
                .map_err(py_error)?;
            Ok::<_, PyErr>(bytes)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(signature = (path, compact=false))]
    fn write_canonical(&self, path: PathBuf, compact: bool, py: Python<'_>) -> PyResult<()> {
        let state = self.state.clone();
        py.detach(move || {
            read_document(&state)?
                .print_to_file(
                    path,
                    if compact {
                        PrintLayout::Compact
                    } else {
                        PrintLayout::Pretty
                    },
                )
                .map_err(|error| PyIOError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (compact=false))]
    fn preserving_bytes<'py>(
        &self,
        py: Python<'py>,
        compact: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let state = self.state.clone();
        let bytes = py.detach(move || {
            read_document(&state)?
                .preserving_bytes(if compact {
                    PrintLayout::Compact
                } else {
                    PrintLayout::Pretty
                })
                .map_err(py_error)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(signature = (path, compact=false))]
    fn write_preserving(&self, path: PathBuf, compact: bool, py: Python<'_>) -> PyResult<()> {
        let state = self.state.clone();
        py.detach(move || {
            read_document(&state)?
                .write_preserving_to_file(
                    path,
                    if compact {
                        PrintLayout::Compact
                    } else {
                        PrintLayout::Pretty
                    },
                )
                .map_err(py_preserve_error)
        })
    }

    fn structurally_equal(&self, other: &Document, py: Python<'_>) -> PyResult<bool> {
        let left = self.state.clone();
        let right = other.state.clone();
        py.detach(move || {
            let left_guard = read_document(&left)?;
            if Arc::ptr_eq(&left, &right) {
                return Ok(left_guard.is_semantically_complete()
                    && left_guard.validate_structure().is_ok());
            }
            let right_guard = read_document(&right)?;
            Ok(left_guard.structurally_eq(&right_guard))
        })
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct LoweringResult {
    #[pyo3(get)]
    document: Option<Document>,
    #[pyo3(get)]
    diagnostics: Vec<SemanticDiagnostic>,
    #[pyo3(get)]
    semantically_complete: bool,
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct SemanticDiagnostic {
    #[pyo3(get)]
    range: (u32, u32),
    #[pyo3(get)]
    message: String,
}

#[pyclass(frozen, module = "zirium._zirium")]
struct SemanticStatistics {
    #[pyo3(get)]
    operations: usize,
    #[pyo3(get)]
    regions: usize,
    #[pyo3(get)]
    blocks: usize,
    #[pyo3(get)]
    local_types: usize,
    #[pyo3(get)]
    local_attributes: usize,
    #[pyo3(get)]
    payload_blobs: usize,
    #[pyo3(get)]
    payload_blob_bytes: usize,
    #[pyo3(get)]
    retained_source_bytes: usize,
    #[pyo3(get)]
    direct_owned_bytes: usize,
    #[pyo3(get)]
    document_index_bytes: usize,
    #[pyo3(get)]
    retained_cst_bytes: usize,
    #[pyo3(get)]
    source_storage_shared: bool,
    #[pyo3(get)]
    cst_storage_shared: bool,
    #[pyo3(get)]
    pooled_list_entries: usize,
    #[pyo3(get)]
    use_index_entries: usize,
    #[pyo3(get)]
    symbol_index_entries: usize,
    #[pyo3(get)]
    dominance_index_entries: usize,
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct SemanticOperation {
    state: SharedDocument,
    id: OperationId,
}

impl SemanticOperation {
    fn new(state: SharedDocument, id: OperationId) -> Self {
        Self { state, id }
    }
}

#[pymethods]
impl SemanticOperation {
    #[getter]
    fn mnemonic(&self) -> PyResult<String> {
        self.name()
    }
    #[getter]
    fn name(&self) -> PyResult<String> {
        read_document(&self.state)?
            .operation_name(self.id)
            .map(str::to_owned)
            .ok_or_else(|| stale("operation"))
    }
    #[getter]
    fn source_range(&self) -> PyResult<Option<(u32, u32)>> {
        let document = read_document(&self.state)?;
        document
            .operation(self.id)
            .ok_or_else(|| stale("operation"))?;
        Ok(document
            .operation_source_range(self.id)
            .map(|range| (range.start(), range.end())))
    }
    #[getter]
    fn is_unparsed(&self) -> PyResult<bool> {
        read_document(&self.state)?
            .operation_is_unparsed(self.id)
            .ok_or_else(|| stale("operation"))
    }
    #[getter]
    fn unparsed_text(&self, py: Python<'_>) -> PyResult<Option<Py<PyBytes>>> {
        let document = read_document(&self.state)?;
        document
            .operation(self.id)
            .ok_or_else(|| stale("operation"))?;
        Ok(document
            .operation_unparsed_text(self.id)
            .map(|bytes| PyBytes::new(py, bytes).into()))
    }
    fn region_count(&self) -> PyResult<usize> {
        Ok(read_document(&self.state)?
            .operation_regions(self.id)
            .ok_or_else(|| stale("operation"))?
            .len())
    }
    fn region(&self, index: usize) -> PyResult<SemanticRegion> {
        let id = *read_document(&self.state)?
            .operation_regions(self.id)
            .ok_or_else(|| stale("operation"))?
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("region index out of range"))?;
        Ok(SemanticRegion {
            state: self.state.clone(),
            id,
        })
    }
    fn operand_count(&self) -> PyResult<usize> {
        Ok(read_document(&self.state)?
            .operands(self.id)
            .ok_or_else(|| stale("operation"))?
            .len())
    }
    fn operand(&self, index: usize) -> PyResult<SemanticValue> {
        let value = *read_document(&self.state)?
            .operands(self.id)
            .ok_or_else(|| stale("operation"))?
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("operand index out of range"))?;
        Ok(SemanticValue {
            state: self.state.clone(),
            value,
        })
    }
    fn result_count(&self) -> PyResult<usize> {
        Ok(read_document(&self.state)?
            .result_types(self.id)
            .ok_or_else(|| stale("operation"))?
            .len())
    }
    fn result(&self, index: usize) -> PyResult<SemanticValue> {
        let document = read_document(&self.state)?;
        let operation = document
            .operation(self.id)
            .ok_or_else(|| stale("operation"))?;
        if index
            >= document
                .result_types(self.id)
                .ok_or_else(|| stale("operation"))?
                .len()
        {
            return Err(PyIndexError::new_err("result index out of range"));
        }
        let value = operation
            .result(self.id, index as u32)
            .expect("bounded result");
        Ok(SemanticValue {
            state: self.state.clone(),
            value: ValueReference::Resolved(value),
        })
    }
    fn result_type(&self, index: usize) -> PyResult<SemanticType> {
        let id = *read_document(&self.state)?
            .result_types(self.id)
            .ok_or_else(|| stale("operation"))?
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("result index out of range"))?;
        Ok(SemanticType {
            state: self.state.clone(),
            id,
        })
    }
    fn attribute_count(&self) -> PyResult<usize> {
        Ok(read_document(&self.state)?
            .attribute_entries(self.id)
            .ok_or_else(|| stale("operation"))?
            .count())
    }
    fn attribute(&self, index: usize) -> PyResult<SemanticAttribute> {
        let document = read_document(&self.state)?;
        document
            .operation(self.id)
            .ok_or_else(|| stale("operation"))?;
        let (name, id) = document
            .attribute_entry(self.id, index)
            .ok_or_else(|| PyIndexError::new_err("attribute index out of range"))?;
        Ok(SemanticAttribute {
            state: self.state.clone(),
            id,
            name: name.to_owned(),
        })
    }
    fn attribute_by_name(&self, name: &str) -> PyResult<Option<SemanticAttribute>> {
        let document = read_document(&self.state)?;
        document
            .operation(self.id)
            .ok_or_else(|| stale("operation"))?;
        Ok(document
            .attribute_id(self.id, name)
            .map(|id| SemanticAttribute {
                state: self.state.clone(),
                id,
                name: name.to_owned(),
            }))
    }
    fn attribute_snapshot(&self) -> PyResult<Vec<(String, String)>> {
        Ok(read_document(&self.state)?
            .attributes(self.id)
            .ok_or_else(|| stale("operation"))?
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect())
    }
    fn property_snapshot(&self) -> PyResult<Vec<(String, String)>> {
        Ok(read_document(&self.state)?
            .properties(self.id)
            .ok_or_else(|| stale("operation"))?
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect())
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct SemanticRegion {
    state: SharedDocument,
    id: RegionId,
}

#[pymethods]
impl SemanticRegion {
    fn block_count(&self) -> PyResult<usize> {
        let document = read_document(&self.state)?;
        Ok(document
            .region(self.id)
            .and_then(|r| r.blocks(&document))
            .ok_or_else(|| stale("region"))?
            .len())
    }
    fn block(&self, index: usize) -> PyResult<SemanticBlock> {
        let document = read_document(&self.state)?;
        let id = *document
            .region(self.id)
            .and_then(|r| r.blocks(&document))
            .ok_or_else(|| stale("region"))?
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("block index out of range"))?;
        Ok(SemanticBlock {
            state: self.state.clone(),
            id,
        })
    }
    fn parent_operation(&self) -> PyResult<SemanticOperation> {
        let id = read_document(&self.state)?
            .region(self.id)
            .ok_or_else(|| stale("region"))?
            .parent_operation();
        Ok(SemanticOperation::new(self.state.clone(), id))
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct SemanticBlock {
    state: SharedDocument,
    id: BlockId,
}

#[pymethods]
impl SemanticBlock {
    #[getter]
    fn label(&self) -> PyResult<Option<String>> {
        Ok(read_document(&self.state)?
            .block_label(self.id)
            .ok_or_else(|| stale("block"))?
            .map(str::to_owned))
    }
    fn operation_count(&self) -> PyResult<usize> {
        Ok(read_document(&self.state)?
            .block_operations(self.id)
            .ok_or_else(|| stale("block"))?
            .len())
    }
    fn operation(&self, index: usize) -> PyResult<SemanticOperation> {
        let id = *read_document(&self.state)?
            .block_operations(self.id)
            .ok_or_else(|| stale("block"))?
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("operation index out of range"))?;
        Ok(SemanticOperation::new(self.state.clone(), id))
    }
    fn argument_count(&self) -> PyResult<usize> {
        Ok(read_document(&self.state)?
            .block_argument_types(self.id)
            .ok_or_else(|| stale("block"))?
            .len())
    }
    fn argument(&self, index: usize) -> PyResult<SemanticValue> {
        let count = read_document(&self.state)?
            .block_argument_types(self.id)
            .ok_or_else(|| stale("block"))?
            .len();
        if index >= count {
            return Err(PyIndexError::new_err("argument index out of range"));
        }
        Ok(SemanticValue {
            state: self.state.clone(),
            value: ValueReference::Resolved(ValueId::BlockArgument {
                block: self.id,
                argument: index as u32,
            }),
        })
    }
    fn parent_region(&self) -> PyResult<SemanticRegion> {
        let id = read_document(&self.state)?
            .block(self.id)
            .ok_or_else(|| stale("block"))?
            .parent_region();
        Ok(SemanticRegion {
            state: self.state.clone(),
            id,
        })
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct SemanticValue {
    state: SharedDocument,
    value: ValueReference,
}

#[pymethods]
impl SemanticValue {
    #[getter]
    fn valid(&self) -> bool {
        let ValueReference::Resolved(value) = self.value else {
            return false;
        };
        read_document(&self.state).is_ok_and(|document| document.check_value(value).is_ok())
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match self.value {
            ValueReference::Resolved(ValueId::OperationResult { .. }) => "operation_result",
            ValueReference::Resolved(ValueId::BlockArgument { .. }) => "block_argument",
            ValueReference::Invalid(_) => "invalid",
        }
    }
    #[getter]
    fn key(&self) -> PyResult<Option<u128>> {
        let ValueReference::Resolved(value) = self.value else {
            return Ok(None);
        };
        Ok(read_document(&self.state)?.value_key(value))
    }
    #[getter]
    fn defining_operation(&self) -> PyResult<Option<SemanticOperation>> {
        let ValueReference::Resolved(value) = self.value else {
            return Ok(None);
        };
        let document = read_document(&self.state)?;
        document.check_value(value).map_err(edit_error)?;
        let ValueId::OperationResult { operation, .. } = value else {
            return Ok(None);
        };
        Ok(Some(SemanticOperation::new(self.state.clone(), operation)))
    }
    #[getter]
    fn type_value(&self) -> PyResult<Option<SemanticType>> {
        let document = read_document(&self.state)?;
        if let ValueReference::Resolved(value) = self.value {
            document.check_value(value).map_err(edit_error)?;
        }
        let id = match self.value {
            ValueReference::Resolved(ValueId::OperationResult { operation, result }) => document
                .result_types(operation)
                .and_then(|types| types.get(result as usize))
                .copied(),
            ValueReference::Resolved(ValueId::BlockArgument { block, argument }) => document
                .block_argument_types(block)
                .and_then(|types| types.get(argument as usize))
                .copied(),
            ValueReference::Invalid(_) => None,
        };
        drop(document);
        Ok(id.map(|id| SemanticType {
            state: self.state.clone(),
            id,
        }))
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct SemanticType {
    state: SharedDocument,
    id: TypeId,
}

#[pymethods]
impl SemanticType {
    #[getter]
    fn spelling(&self) -> PyResult<String> {
        read_document(&self.state)?
            .type_spelling(self.id)
            .map(str::to_owned)
            .ok_or_else(|| stale("type"))
    }
    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        Ok(
            match read_document(&self.state)?
                .type_value(self.id)
                .ok_or_else(|| stale("type"))?
            {
                TypeValue::Integer { .. } => "integer",
                TypeValue::Float(_) => "float",
                TypeValue::Index => "index",
                TypeValue::Tuple(_) => "tuple",
                TypeValue::Tensor { .. } => "tensor",
                TypeValue::Vector { .. } => "vector",
                TypeValue::MemRef { .. } => "memref",
                TypeValue::Function { .. } => "function",
                TypeValue::Opaque(_) => "opaque",
                TypeValue::Invalid(_) => "invalid",
            },
        )
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
struct SemanticAttribute {
    state: SharedDocument,
    id: AttributeId,
    name: String,
}

fn decode_string_attribute(spelling: &str) -> Option<String> {
    let inner = spelling.strip_prefix('"')?.strip_suffix('"')?;
    let bytes = inner.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        index += 1;
        match *bytes.get(index)? {
            b'"' => decoded.push(b'"'),
            b'\\' => decoded.push(b'\\'),
            b'n' => decoded.push(b'\n'),
            b't' => decoded.push(b'\t'),
            high if high.is_ascii_hexdigit() => {
                let low = *bytes.get(index + 1)?;
                if !low.is_ascii_hexdigit() {
                    return None;
                }
                decoded.push(
                    (high as char).to_digit(16)? as u8 * 16 + (low as char).to_digit(16)? as u8,
                );
                index += 1;
            }
            _ => return None,
        }
        index += 1;
    }
    String::from_utf8(decoded).ok()
}

#[pymethods]
impl SemanticAttribute {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        Ok(
            match read_document(&self.state)?
                .attribute_value(self.id)
                .ok_or_else(|| stale("attribute"))?
            {
                AttributeValue::Large(LargeAttributeValue::Dense(_)) => "dense",
                AttributeValue::Large(LargeAttributeValue::Sparse(_)) => "sparse",
                AttributeValue::Large(LargeAttributeValue::Resource(_)) => "resource",
                AttributeValue::Boolean(_) => "boolean",
                AttributeValue::Integer(_) => "integer",
                AttributeValue::Float(_) => "float",
                AttributeValue::String(_) => "string",
                AttributeValue::Type(_) => "type",
                AttributeValue::Symbol(_) => "symbol",
                AttributeValue::Array(_) => "array",
                AttributeValue::Dictionary(_) => "dictionary",
                AttributeValue::Location(_) => "location",
                AttributeValue::AffineMap(_) => "affine_map",
                AttributeValue::IntegerSet(_) => "integer_set",
                AttributeValue::WideNumber(_) => "wide_number",
                AttributeValue::Opaque(_) => "opaque",
                AttributeValue::Invalid(_) => "invalid",
            },
        )
    }
    #[getter]
    fn spelling(&self) -> PyResult<String> {
        read_document(&self.state)?
            .attribute_spelling_value(self.id)
            .map(str::to_owned)
            .ok_or_else(|| stale("attribute"))
    }
    #[getter]
    fn string_value(&self) -> PyResult<Option<String>> {
        Ok(
            match read_document(&self.state)?
                .attribute_value(self.id)
                .ok_or_else(|| stale("attribute"))?
            {
                AttributeValue::String(value) => decode_string_attribute(value),
                _ => None,
            },
        )
    }
    #[getter]
    fn integer_value(&self) -> PyResult<Option<i128>> {
        Ok(
            match read_document(&self.state)?
                .attribute_value(self.id)
                .ok_or_else(|| stale("attribute"))?
            {
                AttributeValue::Integer(value) => value
                    .split(':')
                    .next()
                    .and_then(|value| value.trim().parse().ok()),
                _ => None,
            },
        )
    }
    #[getter]
    fn float_value(&self) -> PyResult<Option<f64>> {
        Ok(
            match read_document(&self.state)?
                .attribute_value(self.id)
                .ok_or_else(|| stale("attribute"))?
            {
                AttributeValue::Float(value) => value
                    .split(':')
                    .next()
                    .and_then(|value| value.trim().parse().ok()),
                _ => None,
            },
        )
    }
    #[getter]
    fn boolean_value(&self) -> PyResult<Option<bool>> {
        Ok(
            match read_document(&self.state)?
                .attribute_value(self.id)
                .ok_or_else(|| stale("attribute"))?
            {
                AttributeValue::Boolean(value) => Some(*value),
                _ => None,
            },
        )
    }
    #[getter]
    fn symbol_value(&self) -> PyResult<Option<String>> {
        Ok(
            match read_document(&self.state)?
                .attribute_value(self.id)
                .ok_or_else(|| stale("attribute"))?
            {
                AttributeValue::Symbol(path) => Some(path.join("::")),
                _ => None,
            },
        )
    }
    #[getter]
    fn payload_byte_length(&self) -> PyResult<Option<usize>> {
        Ok(
            match read_document(&self.state)?
                .attribute_value(self.id)
                .ok_or_else(|| stale("attribute"))?
            {
                AttributeValue::Large(
                    LargeAttributeValue::Dense(bytes)
                    | LargeAttributeValue::Sparse(bytes)
                    | LargeAttributeValue::Resource(bytes),
                ) => Some(bytes.len()),
                _ => None,
            },
        )
    }
    fn raw_buffer<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let bytes = match read_document(&self.state)?
            .attribute_value(self.id)
            .ok_or_else(|| stale("attribute"))?
        {
            AttributeValue::Large(
                LargeAttributeValue::Dense(bytes)
                | LargeAttributeValue::Sparse(bytes)
                | LargeAttributeValue::Resource(bytes),
            ) => Some(bytes.clone()),
            _ => None,
        };
        Ok(bytes.map(|bytes| PyBytes::new(py, &bytes)))
    }
}

macro_rules! kind_catalog {
    ($const_name:ident, $kind:ty, $name_fn:ident, [$($variant:ident),+ $(,)?]) => {
        const $const_name: &[$kind] = &[$(<$kind>::$variant),+];
        fn $name_fn(kind: $kind) -> &'static str {
            match kind { $(<$kind>::$variant => stringify!($variant)),+ }
        }
    };
}

kind_catalog!(
    SYNTAX_KINDS,
    SyntaxKind,
    syntax_kind_name,
    [
        File,
        AliasDefinition,
        Operation,
        DialectOperation,
        ArithConstantValue,
        UnparsedCustomOperation,
        Region,
        Block,
        Result,
        ResultGroup,
        ResultNumber,
        Operand,
        OperandUse,
        SuccessorList,
        Successor,
        SuccessorArguments,
        BlockLabel,
        BlockArgumentList,
        BlockArgument,
        PropertyDict,
        AttributeDict,
        Attribute,
        IntegerType,
        FloatType,
        IndexType,
        IntegerAttribute,
        FloatAttribute,
        BooleanAttribute,
        StringAttribute,
        TypeAttribute,
        AttributeAlias,
        TypeAlias,
        SymbolReference,
        ArrayAttribute,
        DictionaryAttribute,
        LocationAttribute,
        UnknownLocation,
        FileLineColLocation,
        NameLocation,
        CallSiteLocation,
        FusedLocation,
        TrailingLocation,
        FunctionType,
        TupleType,
        TensorType,
        VectorType,
        MemRefType,
        ShapedDimension,
        TensorEncoding,
        MemRefLayout,
        MemRefMemorySpace,
        StridedLayout,
        AffineMap,
        IntegerSet,
        AffineExpression,
        AffineConstraint,
        DenseElementsAttribute,
        SparseElementsAttribute,
        DenseResourceElementsAttribute,
        WideNumber,
        OpaqueAttribute,
        OpaqueAttributeBody,
        OpaqueType,
        OpaqueTypeBody,
        Error,
        FileMetadata,
    ]
);

kind_catalog!(
    TOKEN_KINDS,
    TokenKind,
    token_kind_name,
    [
        Whitespace,
        LineComment,
        String,
        BareIdentifier,
        AtIdentifier,
        PercentIdentifier,
        CaretIdentifier,
        ExclamationIdentifier,
        HashIdentifier,
        Dense,
        Sparse,
        DenseResource,
        Integer,
        WideInteger,
        Float,
        IntType,
        FloatType,
        IndexType,
        Loc,
        Unknown,
        CallSite,
        Fused,
        Tuple,
        Tensor,
        Vector,
        MemRef,
        AffineMap,
        AffineSet,
        Mod,
        FloorDiv,
        CeilDiv,
        Strided,
        X,
        LParen,
        RParen,
        LBrace,
        RBrace,
        LBracket,
        RBracket,
        Less,
        Greater,
        Colon,
        Comma,
        Equal,
        Plus,
        Star,
        Question,
        VerticalBar,
        Minus,
        Slash,
        Arrow,
        Ellipsis,
        FileMetadataBegin,
        FileMetadataEnd,
        Invalid,
        Eof,
    ]
);

fn syntax_kind_code(kind: SyntaxKind) -> u16 {
    kind as u16
}

fn token_kind_code(kind: TokenKind) -> u16 {
    kind as u16
}

fn syntax_kind_from_name(name: &str) -> Option<SyntaxKind> {
    SYNTAX_KINDS
        .iter()
        .copied()
        .find(|kind| syntax_kind_name(*kind) == name)
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

// PyO3's extension-module mode intentionally does not link libpython. The
// focused embedded-Python test opts in with `--cfg python_linked_test` and
// explicit libpython linker flags; ordinary extension builds remain unchanged.
#[allow(unexpected_cfgs)]
#[cfg(all(test, python_linked_test))]
mod packed_syntax_operation_tests {
    use super::*;

    fn decoded(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn all_columns_match_direct_operation_syntax_iterators() {
        Python::initialize();
        for source in [
            b"".as_slice(),
            b"%0 = \"make\"() : () -> i32\n\"use\"(%0)[^next] : (i32) -> ()".as_slice(),
            b"\"outer\"() ({ \"inner\"() : () -> () }) : () -> ()".as_slice(),
            b"%0 = \"broken\"(%0)[^next] ({ \"inner\"(".as_slice(),
        ] {
            let parsed = ParsedFile::parse(source).unwrap();
            let operations = parsed.syntax().file().operations().collect::<Vec<_>>();
            let expected = |relationship| {
                let rows = operations
                    .iter()
                    .map(|op| match relationship {
                        SyntaxRelationship::Result => op
                            .results()
                            .map(|node| node.id().index() as u32)
                            .collect::<Vec<_>>(),
                        SyntaxRelationship::Operand => {
                            op.operands().map(|node| node.id().index() as u32).collect()
                        }
                        SyntaxRelationship::Successor => op
                            .successors()
                            .map(|node| node.id().index() as u32)
                            .collect(),
                        SyntaxRelationship::Region => {
                            op.regions().map(|node| node.id().index() as u32).collect()
                        }
                    })
                    .collect::<Vec<_>>();
                let mut offsets = vec![0];
                for nodes in &rows {
                    offsets.push(offsets.last().unwrap() + nodes.len() as u32);
                }
                (offsets, rows.into_iter().flatten().collect::<Vec<_>>())
            };
            let results = expected(SyntaxRelationship::Result);
            let operands = expected(SyntaxRelationship::Operand);
            let successors = expected(SyntaxRelationship::Successor);
            let regions = expected(SyntaxRelationship::Region);
            Python::attach(|py| {
                let table = syntax_operation_table(py, &parsed).unwrap();
                assert_eq!(
                    decoded(table.operation_node.bind(py).as_bytes()),
                    operations
                        .iter()
                        .map(|op| op.id().index() as u32)
                        .collect::<Vec<_>>()
                );
                for (offsets, nodes, expected) in [
                    (&table.result_offsets, &table.result_nodes, &results),
                    (&table.operand_offsets, &table.operand_nodes, &operands),
                    (
                        &table.successor_offsets,
                        &table.successor_nodes,
                        &successors,
                    ),
                    (&table.region_offsets, &table.region_nodes, &regions),
                ] {
                    assert_eq!(decoded(offsets.bind(py).as_bytes()), expected.0);
                    assert_eq!(decoded(nodes.bind(py).as_bytes()), expected.1);
                }
            });
        }
    }
}
