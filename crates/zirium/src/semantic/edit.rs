use super::*;

impl Document {
    /// Starts an atomic edit against a private copy of a complete, valid document.
    ///
    /// Changes become visible only when [`DocumentEditor::commit`] succeeds.
    pub fn edit<'a>(
        &'a mut self,
        registry: &'a DialectRegistry,
    ) -> Result<DocumentEditor<'a>, EditError> {
        if !self.complete {
            return Err(EditError::IncompleteDocument);
        }
        self.validate_structure().map_err(EditError::Structural)?;
        Ok(DocumentEditor {
            working: self.clone(),
            original: self,
            registry,
        })
    }
}

impl DocumentEditor<'_> {
    pub fn document(&self) -> &Document {
        &self.working
    }

    pub fn insert(
        &mut self,
        point: InsertionPoint,
        spec: OperationSpec,
    ) -> Result<OperationId, EditError> {
        let parent = match point {
            InsertionPoint::Root(index) => {
                if index > self.working.root_operations().len() {
                    return Err(EditError::InvalidPosition);
                }
                None
            }
            InsertionPoint::Block { block, .. } => {
                self.working
                    .block(block)
                    .ok_or_else(|| self.block_error(block))?;
                if let InsertionPoint::Block { index, .. } = point {
                    if index > self.working.block_operations(block).unwrap_or(&[]).len() {
                        return Err(EditError::InvalidPosition);
                    }
                }
                Some(block)
            }
        };
        for value in &spec.operands {
            self.require_value(*value)?;
        }
        self.invalidate_syntax_mapping();
        let name = self.intern_string(&spec.name);
        let result_types = spec
            .result_types
            .iter()
            .map(|spec| self.intern_type_spec(spec))
            .collect::<Vec<_>>();
        let function_type = self.intern_type_spec(&spec.function_type);
        let attributes = self.intern_attributes(&spec.attributes);
        let properties = self.intern_attributes(&spec.properties);
        let operands = spec
            .operands
            .iter()
            .copied()
            .map(ValueReference::Resolved)
            .collect::<Vec<_>>();
        let generation = self
            .working
            .operation_identities
            .lock()
            .expect("operation identity allocator is not poisoned")
            .allocate();
        let index = self
            .working
            .operation_alive
            .iter()
            .position(|alive| !alive)
            .unwrap_or(self.working.operations.len());
        let id = OperationId::with_owner(index, generation, self.working.identity.0);
        let operation = Operation {
            id,
            name,
            parent,
            operands: self.working.values.push(&operands),
            result_types: self.working.types_lists.push(&result_types),
            function_type,
            attributes: self.working.attribute_lists.push(&attributes),
            properties: self.working.attribute_lists.push(&properties),
            successors: self.working.successor_lists.push(&[]),
            regions: self.working.region_lists.push(&[]),
            location: None,
            source_range: TextRange::new(0, 0).expect("empty source range is valid"),
            unparsed_text: None,
        };
        if index == self.working.operations.len() {
            self.working.operations.push(operation);
            self.working.operation_generations.push(generation);
            self.working.operation_alive.push(true);
        } else {
            self.working.operations[index] = operation;
            self.working.operation_generations[index] = generation;
            self.working.operation_alive[index] = true;
        }
        self.insert_in_parent(point, id)?;
        Ok(id)
    }

    pub fn erase(&mut self, operation: OperationId) -> Result<(), EditError> {
        let op = self
            .working
            .operation(operation)
            .ok_or_else(|| self.operation_error(operation))?;
        if !self
            .working
            .region_lists
            .get(op.regions)
            .unwrap_or(&[])
            .is_empty()
        {
            return Err(EditError::OwnedRegionsUnsupported);
        }
        if self.has_live_uses(operation) {
            return Err(EditError::LiveUses(operation));
        }
        let parent = op.parent;
        self.remove_from_parent(parent, operation)?;
        self.invalidate_syntax_mapping();
        self.working.operation_alive[operation.index()] = false;
        let tombstone_generation = self
            .working
            .operation_identities
            .lock()
            .expect("operation identity allocator is not poisoned")
            .allocate();
        self.working.operation_generations[operation.index()] = tombstone_generation;
        Ok(())
    }

    pub fn rewire_operand(
        &mut self,
        operation: OperationId,
        operand: usize,
        value: ValueId,
    ) -> Result<(), EditError> {
        self.require_value(value)?;
        let list = self.require_operation(operation)?.operands;
        let mut values = self.working.values.get(list).unwrap_or(&[]).to_vec();
        let target = values
            .get_mut(operand)
            .ok_or(EditError::InvalidOperandIndex)?;
        if self
            .working
            .operation_name(operation)
            .is_some_and(|name| self.registry.operation(name).is_none())
        {
            let expected = self
                .working
                .function_type(operation)
                .and_then(|function_type| self.working.type_value(function_type))
                .and_then(|function_type| match function_type {
                    TypeValue::Function { inputs, .. } => inputs.get(operand),
                    _ => None,
                });
            let actual = self
                .working
                .value_type_value(ValueReference::Resolved(value));
            if expected.is_none() || actual != expected {
                return Err(EditError::TypeMismatch);
            }
        }
        *target = ValueReference::Resolved(value);
        self.mark_block_dirty_for(operation);
        self.working.operations[operation.index()].operands = self.working.values.push(&values);
        Ok(())
    }

    /// Replaces every indexed operand and successor-argument use of `from`.
    pub fn replace_all_uses(&mut self, from: ValueId, to: ValueId) -> Result<usize, EditError> {
        self.require_value(from)?;
        self.require_value(to)?;
        let from_type = self
            .working
            .value_type_id(ValueReference::Resolved(from))
            .ok_or(EditError::InvalidValue(from))?;
        let to_type = self
            .working
            .value_type_id(ValueReference::Resolved(to))
            .ok_or(EditError::InvalidValue(to))?;
        if self.working.type_value(from_type) != self.working.type_value(to_type) {
            return Err(EditError::TypeMismatch);
        }
        let sites = self.working.uses(from);
        for site in &sites {
            match *site {
                UseSite::Operand { operation, index } => {
                    let old = self.require_operation(operation)?.operands;
                    let mut values = self.working.values.get(old).unwrap_or(&[]).to_vec();
                    values[index as usize] = ValueReference::Resolved(to);
                    self.working.operations[operation.index()].operands =
                        self.working.values.push(&values);
                }
                UseSite::SuccessorArgument {
                    operation,
                    successor,
                    argument,
                } => {
                    let old_list = self.require_operation(operation)?.successors;
                    let mut successors = self
                        .working
                        .successor_lists
                        .get(old_list)
                        .unwrap_or(&[])
                        .to_vec();
                    let mut arguments = self
                        .working
                        .successor_arguments(successors[successor as usize])
                        .unwrap_or(&[])
                        .to_vec();
                    arguments[argument as usize] = ValueReference::Resolved(to);
                    successors[successor as usize].arguments = self.working.values.push(&arguments);
                    self.working.operations[operation.index()].successors =
                        self.working.successor_lists.push(&successors);
                }
            }
        }
        if !sites.is_empty() {
            for site in &sites {
                let operation = match *site {
                    UseSite::Operand { operation, .. }
                    | UseSite::SuccessorArgument { operation, .. } => operation,
                };
                self.mark_block_dirty_for(operation);
            }
        }
        Ok(sites.len())
    }

    /// Rebuilds all append-only list pools from live records. Arena slots and
    /// generation-checked public IDs are deliberately left unchanged.
    pub fn compact_pools(&mut self) -> usize {
        let before = self.working.statistics().pooled_list_entries;
        let mut values = ListPool::default();
        let mut type_lists = ListPool::default();
        let mut attribute_lists = ListPool::default();
        let mut successor_lists = ListPool::default();
        let mut region_lists = ListPool::default();
        let mut block_lists = ListPool::default();
        let mut operation_lists = ListPool::default();

        for index in 0..self.working.operations.len() {
            if !self.working.operation_alive[index] {
                continue;
            }
            let operation = &self.working.operations[index];
            let operands = values.push(self.working.values.get(operation.operands).unwrap_or(&[]));
            let result_types = type_lists.push(
                self.working
                    .types_lists
                    .get(operation.result_types)
                    .unwrap_or(&[]),
            );
            let attributes = attribute_lists.push(
                self.working
                    .attribute_lists
                    .get(operation.attributes)
                    .unwrap_or(&[]),
            );
            let properties = attribute_lists.push(
                self.working
                    .attribute_lists
                    .get(operation.properties)
                    .unwrap_or(&[]),
            );
            let mut successors = self
                .working
                .successor_lists
                .get(operation.successors)
                .unwrap_or(&[])
                .to_vec();
            for successor in &mut successors {
                successor.arguments =
                    values.push(self.working.values.get(successor.arguments).unwrap_or(&[]));
            }
            let successors = successor_lists.push(&successors);
            let regions = region_lists.push(
                self.working
                    .region_lists
                    .get(operation.regions)
                    .unwrap_or(&[]),
            );
            let operation = &mut self.working.operations[index];
            operation.operands = operands;
            operation.result_types = result_types;
            operation.attributes = attributes;
            operation.properties = properties;
            operation.successors = successors;
            operation.regions = regions;
        }
        for region in &mut self.working.regions {
            region.blocks =
                block_lists.push(self.working.block_lists.get(region.blocks).unwrap_or(&[]));
        }
        for block in &mut self.working.blocks {
            block.argument_types = type_lists.push(
                self.working
                    .types_lists
                    .get(block.argument_types)
                    .unwrap_or(&[]),
            );
            block.operations = operation_lists.push(
                self.working
                    .operation_lists
                    .get(block.operations)
                    .unwrap_or(&[]),
            );
        }
        let roots = self.working.root_operations().to_vec();
        self.working.roots = operation_lists.push(&roots);
        self.working.values = values;
        self.working.types_lists = type_lists;
        self.working.attribute_lists = attribute_lists;
        self.working.successor_lists = successor_lists;
        self.working.region_lists = region_lists;
        self.working.block_lists = block_lists;
        self.working.operation_lists = operation_lists;
        self.working
            .analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned")
            .dominance = None;
        before.saturating_sub(self.working.statistics().pooled_list_entries)
    }

    /// Rewire one block argument carried on a successor edge.
    pub fn rewire_successor_argument(
        &mut self,
        operation: OperationId,
        successor_index: usize,
        argument_index: usize,
        value: ValueId,
    ) -> Result<(), EditError> {
        let successor_list = self.require_operation(operation)?.successors;
        let old_successor = *self
            .working
            .successor_lists
            .get(successor_list)
            .and_then(|successors| successors.get(successor_index))
            .ok_or(EditError::InvalidSuccessorIndex)?;
        self.require_value(value)?;
        let mut arguments = self
            .working
            .values
            .get(old_successor.arguments)
            .unwrap_or(&[])
            .to_vec();
        let target = arguments
            .get_mut(argument_index)
            .ok_or(EditError::InvalidSuccessorArgumentIndex)?;
        let expected = self
            .working
            .block_argument_types(old_successor.block)
            .and_then(|types| types.get(argument_index))
            .and_then(|type_id| self.working.type_value(*type_id));
        let actual = self
            .working
            .value_type_value(ValueReference::Resolved(value));
        if expected.is_none() || actual != expected {
            return Err(EditError::TypeMismatch);
        }
        *target = ValueReference::Resolved(value);
        self.mark_block_dirty_for(operation);
        let arguments = self.working.values.push(&arguments);
        let mut successors = self
            .working
            .successor_lists
            .get(successor_list)
            .unwrap_or(&[])
            .to_vec();
        successors[successor_index] = Successor {
            arguments,
            ..old_successor
        };
        self.working.operations[operation.index()].successors =
            self.working.successor_lists.push(&successors);
        Ok(())
    }

    pub fn replace_result_types(
        &mut self,
        operation: OperationId,
        types: &[TypeSpec],
    ) -> Result<(), EditError> {
        let old = self
            .working
            .result_types(operation)
            .ok_or_else(|| self.operation_error(operation))?;
        if old.len() != types.len() {
            return Err(EditError::ResultCountChange);
        }
        self.mark_block_dirty_for(operation);
        let types = types
            .iter()
            .map(|spec| self.intern_type_spec(spec))
            .collect::<Vec<_>>();
        self.working.operations[operation.index()].result_types =
            self.working.types_lists.push(&types);
        Ok(())
    }

    pub fn set_attribute(
        &mut self,
        operation: OperationId,
        attribute: AttributeSpec,
    ) -> Result<(), EditError> {
        self.set_named_value(operation, attribute, false)
    }
    pub fn remove_attribute(
        &mut self,
        operation: OperationId,
        name: &str,
    ) -> Result<(), EditError> {
        self.remove_named_value(operation, name, false)
    }
    pub fn set_property(
        &mut self,
        operation: OperationId,
        property: AttributeSpec,
    ) -> Result<(), EditError> {
        self.set_named_value(operation, property, true)
    }
    pub fn remove_property(&mut self, operation: OperationId, name: &str) -> Result<(), EditError> {
        self.remove_named_value(operation, name, true)
    }

    /// Verifies the working copy, then atomically installs it in the original document.
    pub fn commit(mut self) -> Result<(), EditError> {
        self.working
            .validate_structure()
            .map_err(EditError::Structural)?;
        self.working
            .verify_semantics_only(self.registry)
            .map_err(EditError::Semantic)?;
        self.working.revision = self.original.revision.wrapping_add(1);
        *self
            .working
            .analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned") = AnalysisCaches::default();
        *self.original = self.working;
        Ok(())
    }

    fn require_value(&self, value: ValueId) -> Result<(), EditError> {
        match value {
            ValueId::OperationResult { operation, result } => {
                if operation.owner != self.working.identity.0 {
                    return Err(EditError::ForeignValue(value));
                }
                if !self.working.valid_operation(operation) {
                    return Err(EditError::StaleValue(value));
                }
                self.working
                    .result_types(operation)
                    .is_some_and(|types| (result as usize) < types.len())
                    .then_some(())
                    .ok_or(EditError::InvalidValue(value))
            }
            ValueId::BlockArgument { block, argument } => {
                if block.generation != self.working.generation {
                    return Err(EditError::ForeignValue(value));
                }
                self.working
                    .block_argument_types(block)
                    .is_some_and(|types| (argument as usize) < types.len())
                    .then_some(())
                    .ok_or(EditError::InvalidValue(value))
            }
        }
    }

    fn require_operation(&self, operation: OperationId) -> Result<&Operation, EditError> {
        self.working
            .operation(operation)
            .ok_or_else(|| self.operation_error(operation))
    }

    fn operation_error(&self, operation: OperationId) -> EditError {
        if operation.owner != self.working.identity.0 {
            return EditError::ForeignOperation(operation);
        }
        if operation.generation
            == self
                .working
                .operation_generations
                .first()
                .copied()
                .unwrap_or(0)
            && operation.index() >= self.working.operations.len()
        {
            EditError::InvalidOperation(operation)
        } else {
            EditError::StaleOperation(operation)
        }
    }

    fn block_error(&self, block: BlockId) -> EditError {
        if block.generation == self.working.generation {
            EditError::StaleBlock(block)
        } else {
            EditError::ForeignBlock(block)
        }
    }

    fn invalidate_syntax_mapping(&mut self) {
        *self
            .working
            .analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned") = AnalysisCaches::default();
        if self.working.retention_profile != RetentionProfile::SemanticOnly {
            self.working.retention_profile = RetentionProfile::SemanticOnly;
            self.working.retained_source = None;
            self.working.retained_syntax = None;
            self.working.syntax_map = Arc::from([]);
            self.working.blob_ranges = Arc::from([]);
            self.working.dirty_operations.clear();
            self.working.dirty_blocks.clear();
        }
    }

    fn mark_operation_dirty(&mut self, operation: OperationId) {
        *self
            .working
            .analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned") = AnalysisCaches::default();
        if self.working.retention_profile == RetentionProfile::Hybrid {
            self.working.dirty_operations.insert(operation);
        }
    }

    fn mark_block_dirty_for(&mut self, operation: OperationId) {
        *self
            .working
            .analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned") = AnalysisCaches::default();
        if self.working.retention_profile != RetentionProfile::Hybrid {
            return;
        }
        if let Some(block) = self
            .working
            .operation(operation)
            .and_then(Operation::parent_block)
        {
            self.working.dirty_blocks.insert(block);
            if let Some(operations) = self.working.block_operations(block).map(<[_]>::to_vec) {
                for operation in operations {
                    self.working.dirty_operations.remove(&operation);
                }
            }
        } else {
            self.working.dirty_operations.insert(operation);
        }
    }

    fn has_live_uses(&self, definition: OperationId) -> bool {
        self.working.result_types(definition).is_some_and(|types| {
            (0..types.len()).any(|result| {
                !self
                    .working
                    .uses(ValueId::OperationResult {
                        operation: definition,
                        result: result as u32,
                    })
                    .is_empty()
            })
        })
    }

    fn insert_in_parent(
        &mut self,
        point: InsertionPoint,
        operation: OperationId,
    ) -> Result<(), EditError> {
        match point {
            InsertionPoint::Root(index) => {
                let mut values = self.working.root_operations().to_vec();
                if index > values.len() {
                    return Err(EditError::InvalidPosition);
                }
                values.insert(index, operation);
                self.working.roots = self.working.operation_lists.push(&values);
            }
            InsertionPoint::Block { block, index } => {
                let mut values = self
                    .working
                    .block_operations(block)
                    .ok_or(EditError::StaleBlock(block))?
                    .to_vec();
                if index > values.len() {
                    return Err(EditError::InvalidPosition);
                }
                values.insert(index, operation);
                self.working.blocks[block.index()].operations =
                    self.working.operation_lists.push(&values);
            }
        }
        Ok(())
    }

    fn remove_from_parent(
        &mut self,
        parent: Option<BlockId>,
        operation: OperationId,
    ) -> Result<(), EditError> {
        if let Some(block) = parent {
            let mut values = self
                .working
                .block_operations(block)
                .ok_or(EditError::StaleBlock(block))?
                .to_vec();
            values.retain(|value| *value != operation);
            self.working.blocks[block.index()].operations =
                self.working.operation_lists.push(&values);
        } else {
            let mut values = self.working.root_operations().to_vec();
            values.retain(|value| *value != operation);
            self.working.roots = self.working.operation_lists.push(&values);
        }
        Ok(())
    }

    fn intern_string(&mut self, value: &str) -> u32 {
        if let Some(index) = self.working.strings.iter().position(|item| item == value) {
            index as u32
        } else {
            let index = self.working.strings.len() as u32;
            self.working.strings.push(value.to_owned());
            index
        }
    }
    pub(super) fn intern_type_spec(&mut self, spec: &TypeSpec) -> TypeId {
        if let Some(index) = self
            .working
            .types
            .iter()
            .position(|value| value == &spec.value)
        {
            TypeId::new(index, self.working.generation)
        } else {
            let index = self.working.types.len();
            self.working.types.push(spec.value.clone());
            self.working.type_spellings.push(spec.spelling.clone());
            TypeId::new(index, self.working.generation)
        }
    }
    fn intern_attribute(&mut self, spec: &AttributeSpec) -> AttributeId {
        if let Some(index) = self
            .working
            .attributes
            .iter()
            .position(|value| value == &spec.value)
        {
            AttributeId::new(index, self.working.generation)
        } else {
            let index = self.working.attributes.len();
            self.working.attributes.push(spec.value.clone());
            self.working.attribute_spellings.push(spec.spelling.clone());
            AttributeId::new(index, self.working.generation)
        }
    }
    fn intern_attributes(&mut self, specs: &[AttributeSpec]) -> Vec<(u32, AttributeId)> {
        specs
            .iter()
            .map(|spec| {
                let name = self.intern_string(&spec.name);
                let value = self.intern_attribute(spec);
                (name, value)
            })
            .collect()
    }
    fn set_named_value(
        &mut self,
        operation: OperationId,
        spec: AttributeSpec,
        property: bool,
    ) -> Result<(), EditError> {
        let op = self
            .working
            .operation(operation)
            .ok_or_else(|| self.operation_error(operation))?;
        let list = if property {
            op.properties
        } else {
            op.attributes
        };
        let mut values = self
            .working
            .attribute_lists
            .get(list)
            .unwrap_or(&[])
            .to_vec();
        let name = self.intern_string(&spec.name);
        let value = self.intern_attribute(&spec);
        if let Some(entry) = values.iter_mut().find(|(key, _)| *key == name) {
            entry.1 = value;
        } else {
            values.push((name, value));
        }
        let list = self.working.attribute_lists.push(&values);
        self.mark_operation_dirty(operation);
        if property {
            self.working.operations[operation.index()].properties = list;
        } else {
            self.working.operations[operation.index()].attributes = list;
        }
        Ok(())
    }
    fn remove_named_value(
        &mut self,
        operation: OperationId,
        name: &str,
        property: bool,
    ) -> Result<(), EditError> {
        let op = self
            .working
            .operation(operation)
            .ok_or_else(|| self.operation_error(operation))?;
        let list = if property {
            op.properties
        } else {
            op.attributes
        };
        let mut values = self
            .working
            .attribute_lists
            .get(list)
            .unwrap_or(&[])
            .to_vec();
        values.retain(|(key, _)| {
            self.working.strings.get(*key as usize).map(String::as_str) != Some(name)
        });
        let list = self.working.attribute_lists.push(&values);
        self.mark_operation_dirty(operation);
        if property {
            self.working.operations[operation.index()].properties = list;
        } else {
            self.working.operations[operation.index()].attributes = list;
        }
        Ok(())
    }
}
