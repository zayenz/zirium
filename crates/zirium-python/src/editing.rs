use super::*;

#[pyclass(frozen, module = "zirium._zirium")]
#[derive(Clone)]
pub(super) struct AttributeSpecHandle {
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
pub(super) struct OperationSpec {
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
pub(super) struct SemanticEdit {
    state: SharedDocument,
    registry: RegistryKind,
    commands: Vec<EditCommand>,
    entered: bool,
    closed: bool,
}

impl SemanticEdit {
    pub(super) fn new(state: SharedDocument, registry: RegistryKind) -> Self {
        Self {
            state,
            registry,
            commands: Vec::new(),
            entered: false,
            closed: false,
        }
    }

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
