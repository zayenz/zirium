use super::*;

#[pyclass(frozen, module = "zirium._zirium")]
pub(super) struct Diagnostic {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    range: (u32, u32),
}

#[pyclass(frozen, module = "zirium._zirium")]
pub(super) struct SyntaxNode {
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
        matches!(
            self.tree().kind(self.id),
            Some(SyntaxKind::Operation | SyntaxKind::DialectOperation)
        )
        .then(|| Operation {
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
pub(super) struct Token {
    file: Arc<ParsedFile>,
    id: usize,
}

#[pyclass(frozen, module = "zirium._zirium")]
pub(super) struct SyntaxTable {
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
pub(super) struct SyntaxOperationTable {
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
pub(super) struct Operation {
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
pub(super) struct File {
    pub(super) parsed: Arc<ParsedFile>,
    pub(super) registry: RegistryKind,
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
