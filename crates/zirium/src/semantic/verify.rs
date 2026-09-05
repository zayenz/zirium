use super::*;

impl Document {
    /// Checks arena references, ownership, parent links, values, and retention data.
    ///
    /// This check does not run dialect schemas or callbacks.
    ///
    /// # Errors
    ///
    /// Returns the first [`ValidationError`] found in semantic storage.
    pub fn validate_structure(&self) -> Result<(), ValidationError> {
        let valid_op = |id: OperationId| self.valid_operation(id);
        if self.operation_generations.len() != self.operations.len()
            || self.operation_alive.len() != self.operations.len()
        {
            return Err(ValidationError::InvalidOperationStorage);
        }
        let valid_region = |id: RegionId| self.valid(id.index, id.generation, self.regions.len());
        let valid_block = |id: BlockId| self.valid(id.index, id.generation, self.blocks.len());
        if self
            .types
            .iter()
            .any(|value| !valid_type_value(self, value))
            || self
                .attributes
                .iter()
                .any(|value| !valid_attribute_value(self, value))
            || self
                .locations
                .iter()
                .any(|value| !valid_location_value(self, value))
            || !valid_affine_storage(self)
        {
            return Err(ValidationError::InvalidSentinel);
        }
        let retains_syntax = matches!(
            self.retention_profile,
            RetentionProfile::SyntaxOnly | RetentionProfile::Hybrid
        );
        let retains_map = self.retention_profile == RetentionProfile::Hybrid;
        if (retains_syntax != (self.retained_source.is_some() && self.retained_syntax.is_some()))
            || (!retains_syntax
                && (self.retained_source.is_some() || self.retained_syntax.is_some()))
            || (!retains_map && !self.syntax_map.is_empty())
            || (retains_map && self.syntax_map.len() != self.operations.len())
        {
            return Err(ValidationError::InvalidRetention);
        }
        let mut node_ranges = HashSet::new();
        let mut operation_ranges = HashSet::new();
        if let Some(source) = &self.retained_source {
            let Some(tree) = &self.retained_syntax else {
                return Err(ValidationError::InvalidRetention);
            };
            if tree.verify().is_err() {
                return Err(ValidationError::InvalidRetention);
            }
            for node in tree.subtree(tree.root()).into_iter().flatten() {
                let Some(range) = tree.text_range(node) else {
                    return Err(ValidationError::InvalidRetention);
                };
                if range.end() as usize > source.len() {
                    return Err(ValidationError::InvalidRetention);
                }
                node_ranges.insert(range);
                if matches!(
                    tree.kind(node),
                    Some(crate::SyntaxKind::Operation | crate::SyntaxKind::DialectOperation)
                ) {
                    operation_ranges.insert(range);
                }
            }
            if (0..tree.token_count()).any(|index| {
                tree.token(index)
                    .is_none_or(|token| token.range().end() as usize > source.len())
            }) || self
                .blob_ranges
                .iter()
                .any(|range| range.end() as usize > source.len() || !node_ranges.contains(range))
            {
                return Err(ValidationError::InvalidRetention);
            }
        } else if self.retained_syntax.is_some() || !self.blob_ranges.is_empty() {
            return Err(ValidationError::InvalidRetention);
        }
        if self
            .syntax_map
            .windows(2)
            .any(|entries| entries[0].0.index >= entries[1].0.index)
            || self.syntax_map.iter().any(|(id, range)| {
                !self.valid_operation(*id)
                    || self
                        .retained_source
                        .as_ref()
                        .is_none_or(|source| range.end() as usize > source.len())
                    || !operation_ranges.contains(range)
            })
        {
            return Err(ValidationError::InvalidRetention);
        }
        let mut operation_owners = vec![None; self.operations.len()];
        for &root in self
            .operation_lists
            .get(self.roots)
            .ok_or(ValidationError::InvalidList)?
        {
            if !valid_op(root) {
                return Err(ValidationError::StaleOperation(root));
            }
            if operation_owners[root.index()].replace(None).is_some() {
                return Err(ValidationError::ParentChildMismatch);
            }
        }
        for (i, block) in self.blocks.iter().enumerate() {
            let block_id = BlockId::new(i, self.generation);
            for &operation in self
                .operation_lists
                .get(block.operations)
                .ok_or(ValidationError::InvalidList)?
            {
                if !valid_op(operation)
                    || operation_owners[operation.index()]
                        .replace(Some(block_id))
                        .is_some()
                {
                    return Err(ValidationError::ParentChildMismatch);
                }
            }
        }
        let mut region_owners = vec![None; self.regions.len()];
        for (i, op) in self.operations.iter().enumerate() {
            if !self.operation_alive[i] {
                continue;
            }
            let id = OperationId::with_owner(i, self.operation_generations[i], self.identity.0);
            if self.strings.get(op.name as usize).is_none() {
                return Err(ValidationError::InvalidString);
            }
            if let Some(parent) = op.parent {
                if !valid_block(parent) {
                    return Err(ValidationError::StaleBlock(parent));
                }
                if operation_owners[i] != Some(Some(parent)) {
                    return Err(ValidationError::ParentChildMismatch);
                }
            } else if operation_owners[i] != Some(None) {
                return Err(ValidationError::ParentChildMismatch);
            }
            for &region in self
                .region_lists
                .get(op.regions)
                .ok_or(ValidationError::InvalidList)?
            {
                if !valid_region(region) || self.regions[region.index()].parent != id {
                    return Err(ValidationError::ParentChildMismatch);
                }
                if region_owners[region.index()].replace(id).is_some() {
                    return Err(ValidationError::ParentChildMismatch);
                }
            }
            for &ty in self
                .types_lists
                .get(op.result_types)
                .ok_or(ValidationError::InvalidList)?
            {
                if !self.valid(ty.index, ty.generation, self.types.len()) {
                    return Err(ValidationError::StaleType(ty));
                }
            }
            if !self.valid(
                op.function_type.index,
                op.function_type.generation,
                self.types.len(),
            ) {
                return Err(ValidationError::StaleType(op.function_type));
            }
            for &(name, attribute) in self
                .attribute_lists
                .get(op.attributes)
                .ok_or(ValidationError::InvalidList)?
            {
                if self.strings.get(name as usize).is_none() {
                    return Err(ValidationError::InvalidString);
                }
                if !self.valid(attribute.index, attribute.generation, self.attributes.len()) {
                    return Err(ValidationError::StaleAttribute(attribute));
                }
            }
            for &(name, attribute) in self
                .attribute_lists
                .get(op.properties)
                .ok_or(ValidationError::InvalidList)?
            {
                if self.strings.get(name as usize).is_none()
                    || !self.valid(attribute.index, attribute.generation, self.attributes.len())
                {
                    return Err(ValidationError::StaleAttribute(attribute));
                }
            }
            if op.location.is_some_and(|location| {
                !self.valid(location.index, location.generation, self.locations.len())
            }) {
                return Err(ValidationError::InvalidLocation);
            }
            for successor in self
                .successor_lists
                .get(op.successors)
                .ok_or(ValidationError::InvalidList)?
            {
                let invalid_target = successor.invalid.is_some();
                if successor.generation != self.generation
                    || (invalid_target
                        && (self.complete
                            || !self.valid(
                                successor.invalid.unwrap().index() as u32,
                                successor.invalid.unwrap().generation,
                                self.diagnostics.len(),
                            )))
                    || (!invalid_target && !valid_block(successor.block))
                {
                    return Err(ValidationError::InvalidSuccessor);
                }
                for argument in self
                    .values
                    .get(successor.arguments)
                    .ok_or(ValidationError::InvalidList)?
                {
                    match *argument {
                        ValueReference::Resolved(ValueId::OperationResult {
                            operation,
                            result,
                        }) if valid_op(operation)
                            && (result as usize)
                                < self
                                    .types_lists
                                    .get(self.operations[operation.index()].result_types)
                                    .ok_or(ValidationError::InvalidList)?
                                    .len() => {}
                        ValueReference::Resolved(ValueId::BlockArgument { block, argument })
                            if valid_block(block)
                                && (argument as usize)
                                    < self
                                        .types_lists
                                        .get(self.blocks[block.index()].argument_types)
                                        .ok_or(ValidationError::InvalidList)?
                                        .len() => {}
                        ValueReference::Invalid(diagnostic)
                            if !self.complete
                                && self.valid(
                                    diagnostic.index() as u32,
                                    diagnostic.generation,
                                    self.diagnostics.len(),
                                ) => {}
                        _ => return Err(ValidationError::InvalidValue),
                    }
                }
            }
            for operand in self
                .values
                .get(op.operands)
                .ok_or(ValidationError::InvalidList)?
            {
                match *operand {
                    ValueReference::Resolved(ValueId::OperationResult { operation, result }) => {
                        if !valid_op(operation)
                            || result as usize
                                >= self
                                    .types_lists
                                    .get(self.operations[operation.index()].result_types)
                                    .ok_or(ValidationError::InvalidList)?
                                    .len()
                        {
                            return Err(ValidationError::InvalidValue);
                        }
                    }
                    ValueReference::Resolved(ValueId::BlockArgument { block, argument }) => {
                        if !valid_block(block)
                            || argument as usize
                                >= self
                                    .types_lists
                                    .get(self.blocks[block.index()].argument_types)
                                    .ok_or(ValidationError::InvalidList)?
                                    .len()
                        {
                            return Err(ValidationError::InvalidValue);
                        }
                    }
                    ValueReference::Invalid(diagnostic) => {
                        if self.complete
                            || !self.valid(
                                diagnostic.index() as u32,
                                diagnostic.generation,
                                self.diagnostics.len(),
                            )
                        {
                            return Err(ValidationError::InvalidSentinel);
                        }
                    }
                }
            }
        }
        let mut block_owners = vec![None; self.blocks.len()];
        for (i, region) in self.regions.iter().enumerate() {
            if !valid_op(region.parent) {
                return Err(ValidationError::StaleOperation(region.parent));
            }
            let region_id = RegionId::new(i, self.generation);
            if region_owners[i] != Some(region.parent) {
                return Err(ValidationError::ParentChildMismatch);
            }
            for &block in self
                .block_lists
                .get(region.blocks)
                .ok_or(ValidationError::InvalidList)?
            {
                if !valid_block(block)
                    || self.blocks[block.index()].parent != RegionId::new(i, self.generation)
                    || block_owners[block.index()].replace(region_id).is_some()
                {
                    return Err(ValidationError::ParentChildMismatch);
                }
            }
        }
        for (i, block) in self.blocks.iter().enumerate() {
            if !valid_region(block.parent) {
                return Err(ValidationError::StaleRegion(block.parent));
            }
            if block_owners[i] != Some(block.parent) {
                return Err(ValidationError::ParentChildMismatch);
            }
            if block
                .label
                .is_some_and(|label| self.strings.get(label as usize).is_none())
            {
                return Err(ValidationError::InvalidString);
            }
            for &ty in self
                .types_lists
                .get(block.argument_types)
                .ok_or(ValidationError::InvalidList)?
            {
                if !self.valid(ty.index, ty.generation, self.types.len()) {
                    return Err(ValidationError::StaleType(ty));
                }
            }
        }
        Ok(())
    }

    /// Alias for [`Self::validate_structure`].
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.validate_structure()
    }

    /// Runs structural checks followed by registered schemas and verifiers.
    ///
    /// Unregistered operations remain valid generic operations. Registered type
    /// and attribute verifiers run for matching opaque values.
    ///
    /// # Errors
    ///
    /// Returns a structural error, rejects invalid sentinels, or reports the
    /// first schema or registered verifier failure.
    pub fn verify_semantics(
        &self,
        registry: &DialectRegistry,
    ) -> Result<(), SemanticVerificationError> {
        self.validate_structure()
            .map_err(SemanticVerificationError::Structural)?;
        self.verify_semantics_only(registry)
    }

    pub(super) fn verify_semantics_only(
        &self,
        registry: &DialectRegistry,
    ) -> Result<(), SemanticVerificationError> {
        if !self.complete {
            return Err(SemanticVerificationError::InvalidSentinel);
        }
        self.verify_registered_values(registry)?;
        for operation in self.operations() {
            let name = self.operation_name(operation).unwrap_or("<invalid>");
            let Some(descriptor) = registry.operation(name) else {
                continue;
            };
            // The phase-4 proving corpus used a generic `func.func`-named
            // container before the exact schema existed. Keep that explicit
            // handwritten fallback distinct from schema-backed functions.
            if name == "func.func" && self.attribute_id(operation, "function_type").is_none() {
                continue;
            }
            let operands = self.operands(operation).map_or(0, <[_]>::len);
            let results = self.result_types(operation).map_or(0, <[_]>::len);
            if !descriptor.schema.operands.accepts(operands)
                || !descriptor.schema.results.accepts(results)
            {
                return Err(SemanticVerificationError::Schema {
                    operation,
                    message: "operand or result count does not match the registered schema",
                });
            }
            if descriptor
                .schema
                .required_attributes
                .iter()
                .any(|name| self.attribute_id(operation, name).is_none())
            {
                return Err(SemanticVerificationError::Schema {
                    operation,
                    message: "a required registered attribute is missing",
                });
            }
            if let Some(program) = descriptor.assembly {
                program.verify(self, operation).map_err(|message| {
                    SemanticVerificationError::Operation { operation, message }
                })?;
            }
            if let Some(verify) = descriptor.verify {
                verify(self, operation).map_err(|message| {
                    SemanticVerificationError::Operation { operation, message }
                })?;
            }
        }
        self.verify_unregistered_function_types(registry)?;
        self.verify_successor_argument_types()?;
        self.verify_registered_structure(registry)?;
        Ok(())
    }

    fn verify_unregistered_function_types(
        &self,
        registry: &DialectRegistry,
    ) -> Result<(), SemanticVerificationError> {
        for operation in self.operations() {
            let Some(name) = self.operation_name(operation) else {
                continue;
            };
            if registry.operation(name).is_some() {
                continue;
            }
            let Some(TypeValue::Function { inputs, results }) = self
                .function_type(operation)
                .and_then(|function_type| self.type_value(function_type))
            else {
                return Err(SemanticVerificationError::Operation {
                    operation,
                    message: "stored function type is not a function type",
                });
            };
            let operands = self.operands(operation).unwrap_or(&[]);
            if operands.len() != inputs.len()
                || operands
                    .iter()
                    .zip(inputs)
                    .any(|(operand, expected)| self.value_type_value(*operand) != Some(expected))
            {
                return Err(SemanticVerificationError::Operation {
                    operation,
                    message: "operand types do not match the stored function type inputs",
                });
            }
            let result_types = self.result_types(operation).unwrap_or(&[]);
            if result_types.len() != results.len()
                || result_types
                    .iter()
                    .zip(results)
                    .any(|(result, expected)| self.type_value(*result) != Some(expected))
            {
                return Err(SemanticVerificationError::Operation {
                    operation,
                    message: "result types do not match the stored function type outputs",
                });
            }
        }
        Ok(())
    }

    fn verify_successor_argument_types(&self) -> Result<(), SemanticVerificationError> {
        for operation in self.operations() {
            for successor in self.successors(operation).unwrap_or(&[]) {
                let arguments = self.successor_arguments(*successor).unwrap_or(&[]);
                let expected = self.block_argument_types(successor.block).unwrap_or(&[]);
                if arguments.len() != expected.len()
                    || arguments.iter().zip(expected).any(|(argument, expected)| {
                        self.value_type_value(*argument) != self.type_value(*expected)
                    })
                {
                    return Err(SemanticVerificationError::Operation {
                        operation,
                        message: "successor argument types do not match the target block arguments",
                    });
                }
            }
        }
        Ok(())
    }

    fn verify_registered_values(
        &self,
        registry: &DialectRegistry,
    ) -> Result<(), SemanticVerificationError> {
        let mut types = HashSet::new();
        let mut attributes = HashSet::new();
        for value in &self.types {
            verify_type_value(value, registry, &mut types, &mut attributes)?;
        }
        for value in &self.attributes {
            verify_attribute_value(value, registry, &mut types, &mut attributes)?;
        }
        Ok(())
    }

    fn verify_registered_structure(
        &self,
        registry: &DialectRegistry,
    ) -> Result<(), SemanticVerificationError> {
        let context = self.verification_context();
        for operation in self.operations() {
            let Some(name) = self.operation_name(operation) else {
                continue;
            };
            let descriptor = registry.operation(name).filter(|_| {
                name != "func.func" || self.attribute_id(operation, "function_type").is_some()
            });
            if let Some(descriptor) = descriptor {
                for (region_index, region) in self
                    .operation_regions(operation)
                    .unwrap_or(&[])
                    .iter()
                    .enumerate()
                {
                    let requires_terminator = descriptor
                        .regions
                        .get(region_index)
                        .is_some_and(|region| region.requires_terminator);
                    let blocks = self
                        .region(*region)
                        .and_then(|region| region.blocks(self))
                        .unwrap_or(&[]);
                    for block in blocks {
                        let operations = self.block_operations(*block).unwrap_or(&[]);
                        for child in operations.iter().take(operations.len().saturating_sub(1)) {
                            let is_terminator = self
                                .operation_name(*child)
                                .and_then(|name| registry.operation(name))
                                .is_some_and(|descriptor| descriptor.is_terminator);
                            if is_terminator {
                                return Err(SemanticVerificationError::Operation {
                                    operation: *child,
                                    message: "terminator must be the final operation in its block",
                                });
                            }
                        }
                        if requires_terminator {
                            let final_operation = operations.last().copied();
                            let has_terminator = final_operation
                                .and_then(|operation| self.operation_name(operation))
                                .and_then(|name| registry.operation(name))
                                .is_some_and(|descriptor| descriptor.is_terminator);
                            if !has_terminator {
                                return Err(SemanticVerificationError::Operation {
                                    operation: final_operation.unwrap_or(operation),
                                    message: "function block must end with a registered terminator",
                                });
                            }
                        }
                    }
                }
            }
            if descriptor.is_some_and(|descriptor| descriptor.symbols.symbol_table) {
                let mut names = HashMap::new();
                for child in self.direct_operations_in_operation_regions(operation) {
                    let Some(child_name) = self.operation_name(child) else {
                        continue;
                    };
                    if !registry.symbols(child_name).defines_symbol {
                        continue;
                    }
                    let Some(symbol) = self.attribute_spelling(child, "sym_name") else {
                        return Err(SemanticVerificationError::Operation {
                            operation: child,
                            message: "registered symbol definition has no symbol name",
                        });
                    };
                    if names.insert(symbol.to_owned(), child).is_some() {
                        return Err(SemanticVerificationError::Operation {
                            operation: child,
                            message: "duplicate symbol in registered symbol table",
                        });
                    }
                }
            }
        }
        for operation in self.operations() {
            let Some(block) = self.operation(operation).and_then(Operation::parent_block) else {
                continue;
            };
            let position = context
                .operation_positions
                .get(&operation)
                .copied()
                .unwrap_or(0);
            let use_point = ValueUsePoint {
                operation,
                block,
                position,
            };
            let values = self.operands(operation).unwrap_or(&[]).iter().chain(
                self.successors(operation)
                    .unwrap_or(&[])
                    .iter()
                    .flat_map(|successor| self.successor_arguments(*successor).unwrap_or(&[])),
            );
            for value in values {
                let ValueReference::Resolved(value) = value else {
                    continue;
                };
                if !self.value_visible_at(
                    *value,
                    use_point,
                    registry,
                    VisibilityAnalysis::Verification(&context),
                ) {
                    return Err(SemanticVerificationError::Operation {
                        operation,
                        message: "SSA definition does not dominate its use",
                    });
                }
            }
        }
        Ok(())
    }

    fn verification_context(&self) -> VerificationContext {
        let mut context = VerificationContext::default();
        for block_index in 0..self.blocks.len() {
            let block = BlockId::new(block_index, self.generation);
            for (position, operation) in self
                .block_operations(block)
                .unwrap_or(&[])
                .iter()
                .enumerate()
            {
                context.operation_positions.insert(*operation, position);
            }
        }
        for region_index in 0..self.regions.len() {
            let region = RegionId::new(region_index, self.generation);
            let blocks = self
                .region(region)
                .and_then(|region| region.blocks(self))
                .unwrap_or(&[]);
            if blocks.is_empty() {
                continue;
            }
            if blocks.len() == 1 {
                context
                    .block_dominators
                    .insert(blocks[0], HashSet::from([blocks[0]]));
                continue;
            }
            let universe = blocks.iter().copied().collect::<HashSet<_>>();
            let mut predecessors = HashMap::<BlockId, Vec<BlockId>>::new();
            for source in blocks {
                for owner in self.block_operations(*source).unwrap_or(&[]) {
                    for successor in self.successors(*owner).unwrap_or(&[]) {
                        predecessors
                            .entry(successor.block)
                            .or_default()
                            .push(*source);
                    }
                }
            }
            for &block in blocks {
                context.block_dominators.insert(
                    block,
                    if block == blocks[0] {
                        HashSet::from([block])
                    } else {
                        universe.clone()
                    },
                );
            }
            let mut changed = true;
            while changed {
                changed = false;
                for &block in blocks.iter().skip(1) {
                    let mut next = universe.clone();
                    for predecessor in predecessors.get(&block).into_iter().flatten() {
                        if let Some(set) = context.block_dominators.get(predecessor) {
                            next.retain(|candidate| set.contains(candidate));
                        }
                    }
                    next.insert(block);
                    if context.block_dominators.get(&block) != Some(&next) {
                        context.block_dominators.insert(block, next);
                        changed = true;
                    }
                }
            }
        }
        context
    }

    pub(super) fn attribute_spelling(&self, operation: OperationId, name: &str) -> Option<&str> {
        self.attributes(operation)?
            .find_map(|(candidate, value)| (candidate == name).then_some(value))
    }

    fn resolve_call_target(&self, call: OperationId, callee: &str) -> Option<OperationId> {
        let mut parent = self
            .operation(call)
            .and_then(Operation::parent_block)
            .and_then(|block| self.block(block).map(Block::parent_region))
            .and_then(|region| self.region(region).map(Region::parent_operation));
        while let Some(operation) = parent {
            if DialectRegistry::proving()
                .symbols(self.operation_name(operation)?)
                .symbol_table
            {
                return self
                    .direct_operations_in_operation_regions(operation)
                    .into_iter()
                    .find(|candidate| {
                        self.operation_name(*candidate) == Some("func.func")
                            && self
                                .attribute_spelling(*candidate, "sym_name")
                                .is_some_and(|symbol| same_symbol(symbol, callee))
                    });
            }
            parent = self
                .operation(operation)
                .and_then(Operation::parent_block)
                .and_then(|block| self.block(block).map(Block::parent_region))
                .and_then(|region| self.region(region).map(Region::parent_operation));
        }
        None
    }

    pub(super) fn direct_operations_in_operation_regions(
        &self,
        operation: OperationId,
    ) -> Vec<OperationId> {
        self.operation_regions(operation)
            .unwrap_or(&[])
            .iter()
            .flat_map(|region| {
                self.region(*region)
                    .and_then(|region| region.blocks(self))
                    .into_iter()
                    .flatten()
                    .flat_map(|block| self.block_operations(*block).into_iter().flatten().copied())
            })
            .collect()
    }

    fn operation_dominates(
        &self,
        definition: OperationId,
        use_point: ValueUsePoint,
        analysis: &VisibilityAnalysis<'_>,
    ) -> bool {
        let Some(definition_block) = self.operation(definition).and_then(Operation::parent_block)
        else {
            return false;
        };
        if definition_block == use_point.block {
            return analysis
                .operation_position(definition)
                .is_some_and(|position| position < use_point.position);
        }
        let Some(region) = self.block(use_point.block).map(Block::parent_region) else {
            return false;
        };
        let Some(definition_region) = self.block(definition_block).map(Block::parent_region) else {
            return false;
        };
        if definition_region != region {
            return false;
        }
        analysis.block_dominates(region, definition_block, use_point.block)
            && self.operation(use_point.operation).is_some()
    }

    pub(super) fn value_visible_at(
        &self,
        value: ValueId,
        mut use_point: ValueUsePoint,
        registry: &DialectRegistry,
        analysis: VisibilityAnalysis<'_>,
    ) -> bool {
        let definition_block = match value {
            ValueId::OperationResult { operation, .. } => {
                let Some(block) = self.operation(operation).and_then(Operation::parent_block)
                else {
                    return false;
                };
                block
            }
            ValueId::BlockArgument { block, .. } => block,
        };
        let Some(definition_region) = self.block(definition_block).map(Block::parent_region) else {
            return false;
        };
        loop {
            let Some(use_region) = self.block(use_point.block).map(Block::parent_region) else {
                return false;
            };
            if definition_region == use_region {
                if matches!(value, ValueId::OperationResult { .. })
                    && self.region_kind(use_region, registry) == crate::dialect::RegionKind::Graph
                {
                    return true;
                }
                return match value {
                    ValueId::OperationResult { operation, .. } => {
                        self.operation_dominates(operation, use_point, &analysis)
                    }
                    ValueId::BlockArgument { .. } => {
                        analysis.block_dominates(use_region, definition_block, use_point.block)
                    }
                };
            }
            if !self.region_contains_region(definition_region, use_region)
                || self.region_is_isolated(use_region, registry)
            {
                return false;
            }
            let Some(parent) = self.region(use_region).map(Region::parent_operation) else {
                return false;
            };
            let Some(parent_block) = self.operation(parent).and_then(Operation::parent_block)
            else {
                return false;
            };
            // The nested use is visible at the containing operation, so the
            // CFG query at the enclosing region must use that operation's
            // block rather than the original nested block.
            let Some(parent_position) = analysis.operation_position(parent) else {
                return false;
            };
            use_point = ValueUsePoint {
                operation: parent,
                block: parent_block,
                position: parent_position,
            };
        }
    }

    fn region_kind(
        &self,
        region: RegionId,
        registry: &DialectRegistry,
    ) -> crate::dialect::RegionKind {
        let Some(parent) = self.region(region).map(Region::parent_operation) else {
            return crate::dialect::RegionKind::Ssacfg;
        };
        let Some(name) = self.operation_name(parent) else {
            return crate::dialect::RegionKind::Ssacfg;
        };
        let Some(position) = self
            .operation_regions(parent)
            .and_then(|regions| regions.iter().position(|candidate| *candidate == region))
        else {
            return crate::dialect::RegionKind::Ssacfg;
        };
        registry
            .operation(name)
            .and_then(|descriptor| descriptor.regions.get(position))
            .map_or(crate::dialect::RegionKind::Ssacfg, |region| region.kind)
    }

    fn region_is_isolated(&self, region: RegionId, registry: &DialectRegistry) -> bool {
        let Some(parent) = self.region(region).map(Region::parent_operation) else {
            return false;
        };
        let Some(name) = self.operation_name(parent) else {
            return false;
        };
        let Some(position) = self
            .operation_regions(parent)
            .and_then(|regions| regions.iter().position(|candidate| *candidate == region))
        else {
            return false;
        };
        registry
            .operation(name)
            .and_then(|descriptor| descriptor.regions.get(position))
            .is_some_and(|region| region.isolated_from_above)
    }

    pub(super) fn enclosing_operation(&self, operation: OperationId) -> Option<OperationId> {
        self.operation(operation)
            .and_then(Operation::parent_block)
            .and_then(|block| self.block(block).map(Block::parent_region))
            .and_then(|region| self.region(region).map(Region::parent_operation))
    }

    fn region_contains_region(&self, outer: RegionId, mut inner: RegionId) -> bool {
        loop {
            let Some(parent_operation) = self.region(inner).map(Region::parent_operation) else {
                return false;
            };
            let Some(parent_block) = self
                .operation(parent_operation)
                .and_then(Operation::parent_block)
            else {
                return false;
            };
            let Some(parent_region) = self.block(parent_block).map(Block::parent_region) else {
                return false;
            };
            if parent_region == outer {
                return true;
            }
            inner = parent_region;
        }
    }
}

fn same_symbol(left: &str, right: &str) -> bool {
    left.trim_matches('@').trim_matches('"') == right.trim_matches('@').trim_matches('"')
}

pub(super) fn normalize_symbol(symbol: &str) -> &str {
    symbol.trim().trim_matches('@').trim_matches('"')
}

pub(crate) fn verify_builtin_module(
    document: &Document,
    operation: OperationId,
) -> Result<(), &'static str> {
    let regions = document
        .operation_regions(operation)
        .ok_or("missing module region")?;
    if regions.len() != 1 {
        return Err("builtin.module expects exactly one region");
    }
    if document
        .region(regions[0])
        .and_then(|region| region.blocks(document))
        .map_or(0, <[_]>::len)
        != 1
    {
        return Err("builtin.module expects exactly one block");
    }
    Ok(())
}

pub(crate) fn verify_func_func(
    document: &Document,
    operation: OperationId,
) -> Result<(), &'static str> {
    let regions = document
        .operation_regions(operation)
        .ok_or("missing function regions")?;
    if regions.len() > 1 {
        return Err("func.func expects a declaration or one body region");
    }
    let signature = match document
        .attribute_id(operation, "function_type")
        .and_then(|attribute| document.attribute_value(attribute))
    {
        Some(AttributeValue::Type(TypeValue::Function { inputs, results })) => (inputs, results),
        _ => return Err("func.func requires a function type attribute"),
    };
    if let Some(region) = regions.first() {
        let blocks = document
            .region(*region)
            .and_then(|region| region.blocks(document))
            .ok_or("missing function blocks")?;
        if blocks.is_empty() {
            return Err("func.func body must contain a block");
        }
        let arguments = document
            .block_argument_types(blocks[0])
            .ok_or("missing entry arguments")?;
        if arguments.len() != signature.0.len() {
            return Err("func.func entry arguments do not match its signature");
        }
        if arguments
            .iter()
            .zip(signature.0)
            .any(|(actual, expected)| document.type_value(*actual) != Some(expected))
        {
            return Err("func.func entry argument types do not match its signature");
        }
    }
    verify_positional_attribute_list(document, operation, "arg_attrs", signature.0.len())?;
    verify_positional_attribute_list(document, operation, "res_attrs", signature.1.len())?;
    if let Some(no_inline) = document.attribute_id(operation, "no_inline") {
        if !document.attribute_value(no_inline).is_some_and(
            |value| matches!(value, AttributeValue::Opaque(bytes) if bytes.as_ref() == b"unit"),
        ) {
            return Err("func.func no_inline must be the supported unit form");
        }
    }
    Ok(())
}

fn verify_positional_attribute_list(
    document: &Document,
    operation: OperationId,
    name: &str,
    expected: usize,
) -> Result<(), &'static str> {
    let Some(attribute) = document.attribute_id(operation, name) else {
        return Ok(());
    };
    match document.attribute_value(attribute) {
        Some(AttributeValue::Array(values)) if values.len() == expected => Ok(()),
        Some(AttributeValue::Array(_)) => {
            Err("function positional attribute arity does not match signature")
        }
        _ => Err("function positional attributes must be an array"),
    }
}

pub(crate) fn verify_func_call(
    document: &Document,
    operation: OperationId,
) -> Result<(), &'static str> {
    let (inputs, results) = match document.type_value(
        document
            .function_type(operation)
            .ok_or("missing call type")?,
    ) {
        Some(TypeValue::Function { inputs, results }) => (inputs, results),
        _ => return Err("func.call requires a functional type"),
    };
    let operands = document
        .operands(operation)
        .ok_or("missing call operands")?;
    let result_types = document
        .result_types(operation)
        .ok_or("missing call results")?;
    if operands.len() != inputs.len() || result_types.len() != results.len() {
        return Err("func.call operand or result count does not match its type");
    }
    if operands
        .iter()
        .zip(inputs)
        .any(|(operand, expected)| document.value_type_value(*operand) != Some(expected))
        || result_types
            .iter()
            .zip(results)
            .any(|(actual, expected)| document.type_value(*actual) != Some(expected))
    {
        return Err("func.call operand or result type does not match its type");
    }
    let callee = document
        .attribute_spelling(operation, "callee")
        .ok_or("func.call requires a callee")?;
    let Some(function) = document.resolve_call_target(operation, callee) else {
        return Err("func.call callee does not resolve in an enclosing symbol table");
    };
    let function_signature = document
        .attribute_id(function, "function_type")
        .and_then(|attribute| document.attribute_value(attribute))
        .and_then(|value| match value {
            AttributeValue::Type(TypeValue::Function { inputs, results }) => {
                Some((inputs, results))
            }
            _ => None,
        })
        .ok_or("resolved func.func has no valid function signature")?;
    if function_signature.0 != inputs || function_signature.1 != results {
        return Err("func.call type does not match the resolved function signature");
    }
    Ok(())
}

pub(crate) fn verify_cf_cond_br(
    document: &Document,
    operation: OperationId,
) -> Result<(), &'static str> {
    let operands = document.operands(operation).ok_or("missing condition")?;
    if operands.is_empty() || document.value_type(operands[0]) != Some("i1") {
        return Err("cf.cond_br condition must have type i1");
    }
    let successors = document.successors(operation).ok_or("missing successors")?;
    if successors.len() != 2 {
        return Err("cf.cond_br expects exactly two successors");
    }
    for successor in successors {
        let arguments = document
            .successor_arguments(*successor)
            .ok_or("missing successor arguments")?;
        let expected = document
            .block_argument_types(successor.block)
            .ok_or("missing target arguments")?;
        if arguments.len() != expected.len()
            || arguments
                .iter()
                .zip(expected)
                .any(|(argument, expected)| document.value_type_id(*argument) != Some(*expected))
        {
            return Err("cf.cond_br arguments do not match target block");
        }
    }
    if let Some(weights) = document.attribute_spelling(operation, "branch_weights") {
        let Some(payload) = weights.strip_prefix("dense<[") else {
            return Err("cf.cond_br branch_weights must be a dense i32 vector");
        };
        let Some((values, ty)) = payload.split_once("]>") else {
            return Err("cf.cond_br branch_weights has malformed dense payload");
        };
        let values = values.split(',').map(str::trim).collect::<Vec<_>>();
        if values.len() != 2
            || values.iter().any(|value| match value.parse::<i128>() {
                Ok(value) => !(0..=i32::MAX as i128).contains(&value),
                Err(_) => true,
            })
            || ty.trim() != ": vector<2xi32>"
        {
            return Err("cf.cond_br branch_weights must contain two non-negative i32 weights");
        }
    }
    Ok(())
}

pub(crate) fn verify_func_return(
    document: &Document,
    operation: OperationId,
) -> Result<(), &'static str> {
    let block = document
        .operation(operation)
        .and_then(Operation::parent_block)
        .ok_or("func.return must be inside a function")?;
    let region = document.block(block).ok_or("missing parent block")?.parent;
    let function = document
        .region(region)
        .ok_or("missing parent region")?
        .parent;
    if document.operation_name(function) != Some("func.func") {
        return Err("func.return must be directly enclosed by func.func");
    }
    let signature = document
        .attribute_id(function, "function_type")
        .and_then(|attribute| match document.attribute_value(attribute) {
            Some(AttributeValue::Type(value)) => Some(value),
            _ => None,
        });
    let expected =
        match signature.or_else(|| document.type_value(document.function_type(function)?)) {
            Some(TypeValue::Function { results, .. }) => results,
            _ => return Err("func.func has no function signature"),
        };
    let operands = document
        .operands(operation)
        .ok_or("missing return operands")?;
    let operation_signature = document
        .function_type(operation)
        .ok_or("func.return has no operation signature")?;
    let signature_inputs = match document.type_value(operation_signature) {
        Some(TypeValue::Function { inputs, results }) if results.is_empty() => inputs,
        _ => return Err("func.return has an invalid operation signature"),
    };
    if operands.len() != signature_inputs.len()
        || operands
            .iter()
            .zip(signature_inputs)
            .any(|(operand, expected)| document.value_type_value(*operand) != Some(expected))
    {
        return Err("func.return operands do not match its operation signature");
    }
    if operands.len() != expected.len() {
        return Err("func.return operand count does not match function results");
    }
    for (operand, expected) in operands.iter().zip(expected) {
        if document.value_type_value(*operand) != Some(expected) {
            return Err("func.return operand type does not match function result");
        }
    }
    Ok(())
}

pub(crate) fn verify_cf_br(
    document: &Document,
    operation: OperationId,
) -> Result<(), &'static str> {
    let successors = document.successors(operation).ok_or("missing successors")?;
    if successors.len() != 1 {
        return Err("cf.br expects one successor");
    }
    let successor = successors[0];
    let arguments = document
        .successor_arguments(successor)
        .ok_or("missing successor arguments")?;
    let expected = document
        .block_argument_types(successor.block)
        .ok_or("missing target arguments")?;
    if arguments.len() != expected.len() {
        return Err("cf.br argument count does not match target block");
    }
    for (argument, expected) in arguments.iter().zip(expected) {
        if document.value_type_id(*argument) != Some(*expected) {
            return Err("cf.br argument type does not match target block");
        }
    }
    Ok(())
}

fn registered_value_name(spelling: &str) -> &str {
    spelling.split_once('<').map_or(spelling, |(name, _)| name)
}

fn verify_type_value<'a>(
    value: &'a TypeValue,
    registry: &DialectRegistry,
    types: &mut HashSet<&'a TypeValue>,
    attributes: &mut HashSet<&'a AttributeValue>,
) -> Result<(), SemanticVerificationError> {
    if !types.insert(value) {
        return Ok(());
    }
    match value {
        TypeValue::Tuple(values) => {
            for value in values {
                verify_type_value(value, registry, types, attributes)?;
            }
        }
        TypeValue::Tensor {
            element, encoding, ..
        } => {
            verify_type_value(element, registry, types, attributes)?;
            if let Some(encoding) = encoding {
                verify_attribute_value(encoding, registry, types, attributes)?;
            }
        }
        TypeValue::Vector { element, .. } => {
            verify_type_value(element, registry, types, attributes)?;
        }
        TypeValue::MemRef {
            element,
            layout,
            memory_space,
            ..
        } => {
            verify_type_value(element, registry, types, attributes)?;
            if let Some(MemRefLayout::Opaque { parameters, .. }) = layout {
                for parameter in parameters {
                    verify_attribute_value(parameter, registry, types, attributes)?;
                }
            } else if let Some(MemRefLayout::Attribute(attribute)) = layout {
                verify_attribute_value(attribute, registry, types, attributes)?;
            }
            if let Some(memory_space) = memory_space {
                verify_attribute_value(memory_space, registry, types, attributes)?;
            }
        }
        TypeValue::Function { inputs, results } => {
            for value in inputs.iter().chain(results) {
                verify_type_value(value, registry, types, attributes)?;
            }
        }
        TypeValue::Opaque(bytes) => {
            let spelling = String::from_utf8_lossy(bytes);
            if let Some(verify) = registry
                .type_descriptor(registered_value_name(&spelling))
                .and_then(|descriptor| descriptor.verify)
            {
                verify(&spelling).map_err(|message| SemanticVerificationError::Type {
                    spelling: spelling.into_owned(),
                    message,
                })?;
            }
        }
        TypeValue::Integer { .. }
        | TypeValue::Float(_)
        | TypeValue::Index
        | TypeValue::Invalid(_) => {}
    }
    Ok(())
}

fn verify_attribute_value<'a>(
    value: &'a AttributeValue,
    registry: &DialectRegistry,
    types: &mut HashSet<&'a TypeValue>,
    attributes: &mut HashSet<&'a AttributeValue>,
) -> Result<(), SemanticVerificationError> {
    if !attributes.insert(value) {
        return Ok(());
    }
    match value {
        AttributeValue::Type(value) => verify_type_value(value, registry, types, attributes)?,
        AttributeValue::Array(values) => {
            for value in values {
                verify_attribute_value(value, registry, types, attributes)?;
            }
        }
        AttributeValue::DenseArray {
            element_type,
            elements,
        } => {
            for value in elements {
                let valid = match (element_type.as_str(), value) {
                    ("i1", AttributeValue::Boolean(_)) => true,
                    ("i8" | "i16" | "i32" | "i64", AttributeValue::Integer(literal)) => {
                        dense_integer_literal_is_valid(element_type, literal)
                    }
                    ("f32" | "f64", AttributeValue::Float(_)) => true,
                    _ => false,
                };
                if !valid {
                    return Err(SemanticVerificationError::Attribute {
                        spelling: element_type.clone(),
                        message: "dense array element does not match its declared type",
                    });
                }
                verify_attribute_value(value, registry, types, attributes)?;
            }
        }
        AttributeValue::Dictionary(values) => {
            for (_, value) in values {
                verify_attribute_value(value, registry, types, attributes)?;
            }
        }
        AttributeValue::Opaque(bytes) => {
            let spelling = String::from_utf8_lossy(bytes);
            if let Some(verify) = registry
                .attribute_descriptor(registered_value_name(&spelling))
                .and_then(|descriptor| descriptor.verify)
            {
                verify(&spelling).map_err(|message| SemanticVerificationError::Attribute {
                    spelling: spelling.into_owned(),
                    message,
                })?;
            }
        }
        AttributeValue::Boolean(_)
        | AttributeValue::Integer(_)
        | AttributeValue::Float(_)
        | AttributeValue::String(_)
        | AttributeValue::Symbol(_)
        | AttributeValue::Location(_)
        | AttributeValue::AffineMap(_)
        | AttributeValue::IntegerSet(_)
        | AttributeValue::Large(_)
        | AttributeValue::WideNumber(_)
        | AttributeValue::Invalid(_) => {}
    }
    Ok(())
}

fn valid_diagnostic(document: &Document, diagnostic: DiagnosticId) -> bool {
    !document.complete
        && document.valid(
            diagnostic.index,
            diagnostic.generation,
            document.diagnostics.len(),
        )
}

pub(super) fn valid_type_value(document: &Document, value: &TypeValue) -> bool {
    match value {
        TypeValue::Tuple(values) => values.iter().all(|value| valid_type_value(document, value)),
        TypeValue::Function { inputs, results } => inputs
            .iter()
            .chain(results)
            .all(|value| valid_type_value(document, value)),
        TypeValue::Tensor {
            dimensions,
            element,
            encoding,
            ..
        } => {
            dimensions.iter().all(|dimension| {
                dimension
                    .invalid
                    .is_none_or(|diagnostic| valid_diagnostic(document, diagnostic))
            }) && valid_type_value(document, element)
                && encoding
                    .as_deref()
                    .is_none_or(|value| valid_attribute_value(document, value))
        }
        TypeValue::Vector {
            dimensions,
            element,
            ..
        } => {
            dimensions.iter().all(|dimension| {
                dimension
                    .invalid
                    .is_none_or(|diagnostic| valid_diagnostic(document, diagnostic))
            }) && valid_type_value(document, element)
        }
        TypeValue::MemRef {
            dimensions,
            element,
            layout,
            memory_space,
        } => {
            dimensions.iter().all(|dimension| {
                dimension
                    .invalid
                    .is_none_or(|diagnostic| valid_diagnostic(document, diagnostic))
            }) && valid_type_value(document, element)
                && layout.as_ref().is_none_or(|layout| match layout {
                    MemRefLayout::Opaque { parameters, .. } => parameters
                        .iter()
                        .all(|value| valid_attribute_value(document, value)),
                    MemRefLayout::AffineMap(map) => document.affine_map(*map).is_some(),
                    MemRefLayout::Attribute(value) => valid_attribute_value(document, value),
                    MemRefLayout::Invalid(diagnostic) => valid_diagnostic(document, *diagnostic),
                })
                && memory_space
                    .as_ref()
                    .is_none_or(|value| valid_attribute_value(document, value))
        }
        TypeValue::Invalid(diagnostic) => valid_diagnostic(document, *diagnostic),
        _ => true,
    }
}

pub(super) fn valid_attribute_value(document: &Document, value: &AttributeValue) -> bool {
    match value {
        AttributeValue::Type(value) => valid_type_value(document, value),
        AttributeValue::Array(values) => values
            .iter()
            .all(|value| valid_attribute_value(document, value)),
        AttributeValue::DenseArray { elements, .. } => elements
            .iter()
            .all(|value| valid_attribute_value(document, value)),
        AttributeValue::Dictionary(values) => values
            .iter()
            .all(|(_, value)| valid_attribute_value(document, value)),
        AttributeValue::Location(value) => valid_location_value(document, value),
        AttributeValue::AffineMap(map) => document.affine_map(*map).is_some(),
        AttributeValue::IntegerSet(set) => document.integer_set(*set).is_some(),
        AttributeValue::Invalid(diagnostic) => valid_diagnostic(document, *diagnostic),
        _ => true,
    }
}

pub(super) fn valid_affine_storage(document: &Document) -> bool {
    let expression = |id: AffineExprId| document.affine_expression(id).is_some();
    document.affine_expressions.iter().all(|value| match value {
        AffineExprValue::Binary { left, right, .. } => expression(*left) && expression(*right),
        AffineExprValue::Invalid(diagnostic) => valid_diagnostic(document, *diagnostic),
        _ => true,
    }) && document
        .affine_maps
        .iter()
        .all(|map| map.results.iter().all(|id| expression(*id)))
        && document.integer_sets.iter().all(|set| {
            set.constraints.iter().all(|constraint| {
                expression(constraint.left)
                    && expression(constraint.right)
                    && match constraint.relation {
                        IntegerSetRelation::Invalid(diagnostic) => {
                            valid_diagnostic(document, diagnostic)
                        }
                        _ => true,
                    }
            })
        })
}

pub(super) fn valid_location_value(document: &Document, value: &LocationValue) -> bool {
    match value {
        LocationValue::Name { child, .. } => child
            .as_deref()
            .is_none_or(|value| valid_location_value(document, value)),
        LocationValue::CallSite { callee, caller } => {
            valid_location_value(document, callee) && valid_location_value(document, caller)
        }
        LocationValue::Fused { locations, .. } => locations
            .iter()
            .all(|value| valid_location_value(document, value)),
        LocationValue::Invalid(diagnostic) => valid_diagnostic(document, *diagnostic),
        _ => true,
    }
}

impl Document {
    pub(super) fn lookup_symbol_in_index(
        &self,
        from: OperationId,
        path: &[&str],
        index: &SymbolIndex,
    ) -> Option<OperationId> {
        let first = *path.first()?;
        let mut scope = self.enclosing_operation(from);
        while let Some(table) = scope {
            if let Some(mut found) = index
                .scopes
                .get(&table)
                .and_then(|symbols| symbols.get(first))
                .copied()
            {
                for component in path.iter().skip(1) {
                    found = *index.scopes.get(&found)?.get(*component)?;
                }
                return Some(found);
            }
            scope = self.enclosing_operation(table);
        }
        None
    }

    pub(super) fn ensure_use_index(&self) {
        if self
            .analyses
            .borrow()
            .uses
            .as_ref()
            .is_some_and(|index| index.revision == self.revision)
        {
            return;
        }
        let mut uses = HashMap::<ValueId, Vec<UseSite>>::new();
        for operation in self.operations() {
            for (index, value) in self.operands(operation).unwrap_or(&[]).iter().enumerate() {
                if let ValueReference::Resolved(value) = value {
                    uses.entry(*value).or_default().push(UseSite::Operand {
                        operation,
                        index: index as u32,
                    });
                }
            }
            for (successor_index, successor) in
                self.successors(operation).unwrap_or(&[]).iter().enumerate()
            {
                for (argument_index, value) in self
                    .successor_arguments(*successor)
                    .unwrap_or(&[])
                    .iter()
                    .enumerate()
                {
                    if let ValueReference::Resolved(value) = value {
                        uses.entry(*value)
                            .or_default()
                            .push(UseSite::SuccessorArgument {
                                operation,
                                successor: successor_index as u32,
                                argument: argument_index as u32,
                            });
                    }
                }
            }
        }
        self.analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned")
            .uses = Some(UseIndex {
            revision: self.revision,
            uses,
        });
    }

    pub(super) fn ensure_symbol_index(&self, registry: &DialectRegistry) {
        let registry_key = registry.content_identity();
        if self
            .analyses
            .borrow()
            .symbols
            .as_ref()
            .is_some_and(|index| index.revision == self.revision && index.registry == registry_key)
        {
            return;
        }
        let mut scopes = HashMap::<OperationId, HashMap<String, OperationId>>::new();
        for table in self.operations().filter(|operation| {
            self.operation_name(*operation)
                .is_some_and(|name| registry.symbols(name).symbol_table)
        }) {
            let entries = scopes.entry(table).or_default();
            for child in self.direct_operations_in_operation_regions(table) {
                let Some(name) = self.operation_name(child) else {
                    continue;
                };
                if !registry.symbols(name).defines_symbol {
                    continue;
                }
                if let Some(symbol) = self.attribute_spelling(child, "sym_name") {
                    entries.insert(normalize_symbol(symbol).to_owned(), child);
                }
            }
        }
        self.analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned")
            .symbols = Some(SymbolIndex {
            revision: self.revision,
            registry: registry_key,
            scopes,
            diagnostics: Vec::new(),
        });
        let mut diagnostics = Vec::new();
        for operation in self.operations() {
            let Some(name) = self.operation_name(operation) else {
                continue;
            };
            if !registry.symbols(name).uses_symbols {
                continue;
            }
            for (_, attribute) in self.attribute_entries(operation).into_iter().flatten() {
                let Some(AttributeValue::Symbol(path)) = self.attribute_value(attribute) else {
                    continue;
                };
                let spelling = path.join("::");
                let path = spelling
                    .split("::")
                    .map(normalize_symbol)
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>();
                let unresolved = self.analyses.borrow().symbols.as_ref().is_none_or(|index| {
                    self.lookup_symbol_in_index(operation, &path, index)
                        .is_none()
                });
                if unresolved {
                    diagnostics.push(SymbolIndexDiagnostic {
                        operation,
                        symbol: spelling,
                    });
                }
            }
        }
        if let Some(index) = self
            .analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned")
            .symbols
            .as_mut()
        {
            index.diagnostics = diagnostics;
        }
    }

    pub(super) fn ensure_dominance_index(&self, registry: &DialectRegistry) {
        let registry_key = registry.content_identity();
        if self
            .analyses
            .borrow()
            .dominance
            .as_ref()
            .is_some_and(|index| index.revision == self.revision && index.registry == registry_key)
        {
            return;
        }
        let mut index = DominanceIndex {
            revision: self.revision,
            registry: registry_key,
            ..Default::default()
        };
        for block_index in 0..self.blocks.len() {
            let block = BlockId::new(block_index, self.generation);
            for (position, operation) in self
                .block_operations(block)
                .unwrap_or(&[])
                .iter()
                .enumerate()
            {
                index.operation_positions.insert(*operation, position);
            }
        }
        for region_index in 0..self.regions.len() {
            let region_id = RegionId::new(region_index, self.generation);
            let blocks = self
                .region(region_id)
                .and_then(|region| region.blocks(self))
                .unwrap_or(&[]);
            if blocks.is_empty() {
                continue;
            }
            let block_indices = blocks
                .iter()
                .enumerate()
                .map(|(position, block)| (*block, position))
                .collect::<HashMap<_, _>>();
            let mut successors = vec![Vec::new(); blocks.len()];
            let mut predecessors = vec![Vec::new(); blocks.len()];
            for (source_index, source) in blocks.iter().enumerate() {
                for owner in self.block_operations(*source).unwrap_or(&[]) {
                    for successor in self.successors(*owner).unwrap_or(&[]) {
                        let Some(&target_index) = block_indices.get(&successor.block) else {
                            continue;
                        };
                        successors[source_index].push(target_index);
                        predecessors[target_index].push(source_index);
                    }
                }
            }

            let mut reachable = vec![false; blocks.len()];
            let mut postorder = Vec::with_capacity(blocks.len());
            reachable[0] = true;
            let mut dfs = vec![(0usize, 0usize)];
            while let Some((block, next_successor)) = dfs.last_mut() {
                if let Some(&successor) = successors[*block].get(*next_successor) {
                    *next_successor += 1;
                    if !reachable[successor] {
                        reachable[successor] = true;
                        dfs.push((successor, 0));
                    }
                } else {
                    postorder.push(*block);
                    dfs.pop();
                }
            }
            let reverse_postorder = postorder.into_iter().rev().collect::<Vec<_>>();
            let mut rpo_position = vec![usize::MAX; blocks.len()];
            for (position, block) in reverse_postorder.iter().copied().enumerate() {
                rpo_position[block] = position;
            }
            let mut immediate_dominators = vec![None; blocks.len()];
            immediate_dominators[0] = Some(0);
            let mut changed = true;
            while changed {
                changed = false;
                for &block in reverse_postorder.iter().skip(1) {
                    let Some(mut next) = predecessors[block]
                        .iter()
                        .copied()
                        .find(|predecessor| immediate_dominators[*predecessor].is_some())
                    else {
                        continue;
                    };
                    for predecessor in predecessors[block].iter().copied() {
                        if predecessor == next || immediate_dominators[predecessor].is_none() {
                            continue;
                        }
                        let mut left = predecessor;
                        let mut right = next;
                        while left != right {
                            while rpo_position[left] > rpo_position[right] {
                                left =
                                    immediate_dominators[left].expect("reachable block has idom");
                            }
                            while rpo_position[right] > rpo_position[left] {
                                right =
                                    immediate_dominators[right].expect("reachable block has idom");
                            }
                        }
                        next = left;
                    }
                    if immediate_dominators[block] != Some(next) {
                        immediate_dominators[block] = Some(next);
                        changed = true;
                    }
                }
            }
            let mut dominator_tree = vec![Vec::new(); blocks.len()];
            for (block, dominator) in immediate_dominators.iter().copied().enumerate().skip(1) {
                if let Some(dominator) = dominator {
                    dominator_tree[dominator].push(block);
                }
            }
            let mut intervals = vec![None::<(usize, usize)>; blocks.len()];
            let mut timestamp = 0usize;
            let mut traversal = vec![(0usize, false)];
            while let Some((block, exiting)) = traversal.pop() {
                if exiting {
                    let entry = intervals[block].expect("tree entry interval exists").0;
                    intervals[block] = Some((entry, timestamp));
                    timestamp += 1;
                    continue;
                }
                intervals[block] = Some((timestamp, timestamp));
                timestamp += 1;
                traversal.push((block, true));
                traversal.extend(
                    dominator_tree[block]
                        .iter()
                        .rev()
                        .map(|child| (*child, false)),
                );
            }
            index.regions.insert(
                region_id,
                RegionDominance {
                    blocks: blocks.to_vec(),
                    block_indices,
                    successors,
                    predecessors,
                    immediate_dominators,
                    dominator_tree,
                    intervals,
                    reachable,
                },
            );
        }
        self.analyses
            .0
            .write()
            .expect("analysis cache lock is not poisoned")
            .dominance = Some(index);
    }
}
