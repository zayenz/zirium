use super::*;

pub(super) type SharedDocument = Arc<RwLock<CoreDocument>>;

#[pyclass(frozen, module = "zirium._zirium")]
pub(super) struct OperationTable {
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

pub(super) fn read_document(
    state: &SharedDocument,
) -> PyResult<std::sync::RwLockReadGuard<'_, CoreDocument>> {
    state
        .read()
        .map_err(|_| PyValueError::new_err("semantic document lock is poisoned"))
}

pub(super) fn stale(kind: &str) -> PyErr {
    StaleHandleError::new_err(format!("stale semantic {kind} handle"))
}

pub(super) fn same_document(left: &SharedDocument, right: &SharedDocument) -> PyResult<()> {
    if Arc::ptr_eq(left, right) {
        Ok(())
    } else {
        Err(ForeignHandleError::new_err(
            "semantic handle belongs to another document",
        ))
    }
}

#[pyclass(frozen, module = "zirium._zirium")]
pub(super) struct SemanticUse {
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
pub(super) struct Document {
    state: SharedDocument,
    registry: RegistryKind,
}

impl Document {
    pub(super) fn new(document: CoreDocument, registry: RegistryKind) -> Self {
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
        SemanticEdit::new(self.state.clone(), self.registry.clone())
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
pub(super) struct LoweringResult {
    #[pyo3(get)]
    pub(super) document: Option<Document>,
    #[pyo3(get)]
    pub(super) diagnostics: Vec<SemanticDiagnostic>,
    #[pyo3(get)]
    pub(super) semantically_complete: bool,
}

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
pub(super) struct SemanticDiagnostic {
    #[pyo3(get)]
    pub(super) range: (u32, u32),
    #[pyo3(get)]
    pub(super) message: String,
}

#[pyclass(frozen, module = "zirium._zirium")]
pub(super) struct SemanticStatistics {
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
pub(super) struct SemanticOperation {
    pub(super) state: SharedDocument,
    pub(super) id: OperationId,
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
pub(super) struct SemanticRegion {
    pub(super) state: SharedDocument,
    pub(super) id: RegionId,
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
pub(super) struct SemanticBlock {
    pub(super) state: SharedDocument,
    pub(super) id: BlockId,
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
pub(super) struct SemanticValue {
    pub(super) state: SharedDocument,
    pub(super) value: ValueReference,
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
pub(super) struct SemanticType {
    pub(super) state: SharedDocument,
    pub(super) id: TypeId,
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
pub(super) struct SemanticAttribute {
    pub(super) state: SharedDocument,
    pub(super) id: AttributeId,
    pub(super) name: String,
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
