use super::*;

pub(super) struct Interner<T = String> {
    pub(super) values: Vec<T>,
    by_value: HashMap<T, u32>,
}
impl<T> Default for Interner<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            by_value: HashMap::new(),
        }
    }
}
impl Interner<String> {
    pub(super) fn intern(&mut self, value: &str) -> u32 {
        if let Some(&id) = self.by_value.get(value) {
            id
        } else {
            let id = self.values.len() as u32;
            self.values.push(value.to_owned());
            self.by_value.insert(value.to_owned(), id);
            id
        }
    }
}
impl<T: Clone + Eq + std::hash::Hash> Interner<T> {
    pub(super) fn intern_value(&mut self, value: T) -> u32 {
        if let Some(&id) = self.by_value.get(&value) {
            id
        } else {
            let id = self.values.len() as u32;
            self.values.push(value.clone());
            self.by_value.insert(value, id);
            id
        }
    }
}

/// Lowers a parsed file into semantic-only storage.
///
/// [`LoweringMode::Strict`] returns `None` in [`LoweringResult::document`] when
/// semantic resolution is incomplete. [`LoweringMode::BestEffort`] returns a
/// document with invalid sentinels when recovery is possible. In both modes,
/// inspect [`LoweringResult::diagnostics`].
///
/// The registry must match the one used to parse registered custom syntax.
pub fn lower_with_dialect_registry(
    file: &ParsedFile,
    mode: LoweringMode,
    registry: &DialectRegistry,
) -> LoweringResult {
    lower_with_registry(file, mode, RetentionProfile::SemanticOnly, registry)
}

/// Lowers parsed syntax with explicit recovery, retention, and dialect behavior.
///
/// Inspect [`LoweringResult::diagnostics`] even when a document is returned.
/// [`RetentionProfile::Hybrid`] is required for source-preserving output after
/// semantic edits. [`RetentionProfile::SyntaxOnly`] retains syntax for
/// inspection but does not retain semantic-to-syntax mappings.
///
/// The registry must match the one used during parsing. A mismatched registry
/// can leave custom operations unlowered or apply the wrong verifier contract.
pub fn lower_with_dialect_registry_and_retention(
    file: &ParsedFile,
    mode: LoweringMode,
    retention_profile: RetentionProfile,
    registry: &DialectRegistry,
) -> LoweringResult {
    lower_with_registry(file, mode, retention_profile, registry)
}

fn lower_with_registry(
    file: &ParsedFile,
    mode: LoweringMode,
    retention_profile: RetentionProfile,
    registry: &DialectRegistry,
) -> LoweringResult {
    let identity = allocate_document_identity();
    let generation = identity.0;
    let operation_identities = Arc::new(Mutex::new(OperationIdentityState::default()));
    let operation_generation = operation_identities
        .lock()
        .expect("operation identity allocator is not poisoned")
        .allocate();
    let source = file.source();
    let syntax = file.syntax();
    let ops = syntax.file().operations().collect::<Vec<_>>();
    let regions = syntax.file().regions().collect::<Vec<_>>();
    let blocks = regions
        .iter()
        .flat_map(|region| region.blocks())
        .collect::<Vec<_>>();
    let mut strings = Interner::default();
    let mut types = Interner::<TypeValue>::default();
    let mut type_spellings = Vec::new();
    let mut attrs = Interner::<AttributeValue>::default();
    let mut attribute_spellings = Vec::new();
    let mut locations = Interner::<LocationValue>::default();
    let mut location_spellings = Vec::new();
    let mut type_aliases = HashMap::new();
    let mut attribute_aliases = HashMap::new();
    let mut duplicate_aliases = Vec::new();
    for alias in syntax.file().alias_definitions() {
        let Some(range) = alias.text_range() else {
            continue;
        };
        let spelling = text(source.bytes(), range);
        let Some((name, value)) = spelling.split_once('=') else {
            continue;
        };
        let name = name.trim().to_owned();
        let value = value
            .trim()
            .strip_prefix("type")
            .map(str::trim)
            .unwrap_or(value.trim())
            .to_owned();
        let target = if name.starts_with('!') {
            &mut type_aliases
        } else {
            &mut attribute_aliases
        };
        if target.insert(name.clone(), (value, range)).is_some() {
            duplicate_aliases.push((name, range));
        }
    }
    let op_ids = ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            (
                op.id(),
                OperationId::with_owner(i, operation_generation, identity.0),
            )
        })
        .collect::<HashMap<_, _>>();
    let region_ids = regions
        .iter()
        .enumerate()
        .map(|(i, region)| (region.id(), RegionId::new(i, generation)))
        .collect::<HashMap<_, _>>();
    let block_ids = blocks
        .iter()
        .enumerate()
        .map(|(i, block)| (block.id(), BlockId::new(i, generation)))
        .collect::<HashMap<_, _>>();

    let mut parent_blocks = HashMap::new();
    for block in &blocks {
        for child in syntax.tree().children(block.id()).into_iter().flatten() {
            if matches!(
                syntax.tree().kind(child),
                Some(
                    SyntaxKind::Operation
                        | SyntaxKind::DialectOperation
                        | SyntaxKind::UnparsedCustomOperation
                )
            ) {
                parent_blocks.insert(child, block_ids[&block.id()]);
            }
        }
    }
    let mut region_parents = HashMap::new();
    for op in &ops {
        for region in op.regions() {
            region_parents.insert(region.id(), op_ids[&op.id()]);
        }
    }
    let mut block_regions = HashMap::new();
    for region in &regions {
        let region_id = region_ids[&region.id()];
        for block in region.blocks() {
            block_regions.insert(block_ids[&block.id()], region_id);
        }
    }
    let mut region_outer = HashMap::new();
    for region in &regions {
        let parent = region_parents[&region.id()];
        region_outer.insert(
            region_ids[&region.id()],
            parent_blocks
                .get(&ops[parent.index()].id())
                .and_then(|block| block_regions.get(block))
                .copied(),
        );
    }

    let mut doc = Document {
        generation,
        identity: identity.clone(),
        operation_identities,
        operations: Vec::new(),
        operation_generations: vec![operation_generation; ops.len()],
        operation_alive: vec![true; ops.len()],
        regions: Vec::new(),
        blocks: Vec::new(),
        values: ListPool::default(),
        types_lists: ListPool::default(),
        attribute_lists: ListPool::default(),
        successor_lists: ListPool::default(),
        region_lists: ListPool::default(),
        block_lists: ListPool::default(),
        operation_lists: ListPool::default(),
        strings: Vec::new(),
        types: Vec::new(),
        type_spellings: Vec::new(),
        attributes: Vec::new(),
        attribute_spellings: Vec::new(),
        locations: Vec::new(),
        location_spellings: Vec::new(),
        affine_expressions: Vec::new(),
        affine_maps: Vec::new(),
        integer_sets: Vec::new(),
        diagnostics: Vec::new(),
        roots: List::default(),
        complete: true,
        retention_profile,
        retained_source: matches!(
            retention_profile,
            RetentionProfile::SyntaxOnly | RetentionProfile::Hybrid
        )
        .then(|| source.shared_bytes()),
        retained_syntax: matches!(
            retention_profile,
            RetentionProfile::SyntaxOnly | RetentionProfile::Hybrid
        )
        .then(|| syntax.shared_tree()),
        syntax_map: if retention_profile == RetentionProfile::Hybrid {
            ops.iter()
                .enumerate()
                .filter_map(|(index, op)| {
                    Some((
                        OperationId::with_owner(index, operation_generation, identity.0),
                        op.tree().text_range(op.id())?,
                    ))
                })
                .collect::<Vec<_>>()
                .into()
        } else {
            Arc::from([])
        },
        blob_ranges: if matches!(
            retention_profile,
            RetentionProfile::SyntaxOnly | RetentionProfile::Hybrid
        ) {
            syntax
                .tree()
                .subtree(syntax.tree().root())
                .into_iter()
                .flatten()
                .filter(|node| {
                    matches!(
                        syntax.tree().kind(*node),
                        Some(
                            SyntaxKind::DenseElementsAttribute
                                | SyntaxKind::SparseElementsAttribute
                                | SyntaxKind::DenseResourceElementsAttribute
                                | SyntaxKind::OpaqueAttribute
                                | SyntaxKind::WideNumber
                                | SyntaxKind::OpaqueType
                        )
                    )
                })
                .filter_map(|node| syntax.tree().text_range(node))
                .collect::<Vec<_>>()
                .into()
        } else {
            Arc::from([])
        },
        dirty_operations: HashSet::new(),
        dirty_blocks: HashSet::new(),
        revision: 0,
        analyses: AnalysisStore::default(),
        attribute_depth_limit: file.max_attribute_depth(),
        alias_expansion_depth_limit: file.max_alias_expansion_depth(),
    };
    for (name, range) in duplicate_aliases {
        push_diagnostic(
            &mut doc,
            range,
            format!("duplicate alias definition `{name}`"),
        );
    }

    let syntax_diagnostics = syntax
        .tree()
        .subtree(syntax.tree().root())
        .into_iter()
        .flatten()
        .filter(|node| syntax.tree().has_local_error(*node) == Some(true))
        .filter_map(|node| {
            let message = match syntax.tree().kind(node)? {
                SyntaxKind::ResultNumber | SyntaxKind::ResultGroup => "malformed grouped result",
                SyntaxKind::Successor | SyntaxKind::SuccessorArguments => {
                    "malformed successor argument"
                }
                SyntaxKind::PropertyDict => "malformed property dictionary",
                SyntaxKind::AttributeDict => "malformed attribute dictionary",
                SyntaxKind::BlockLabel
                | SyntaxKind::BlockArgumentList
                | SyntaxKind::BlockArgument
                | SyntaxKind::Region => "malformed block label or argument",
                SyntaxKind::TrailingLocation | SyntaxKind::LocationAttribute => {
                    "malformed trailing location"
                }
                SyntaxKind::DialectOperation => "malformed registered operation",
                _ => return None,
            };
            Some(SemanticDiagnostic {
                range: syntax.tree().text_range(node)?,
                message: message.into(),
            })
        })
        .collect::<Vec<_>>();
    for diagnostic in syntax_diagnostics {
        if !doc.diagnostics.iter().any(|existing| {
            existing.range == diagnostic.range && existing.message == diagnostic.message
        }) {
            push_diagnostic(&mut doc, diagnostic.range, diagnostic.message);
        }
    }
    for diagnostic in file.lexer_diagnostics() {
        push_diagnostic(
            &mut doc,
            diagnostic.range(),
            format!("malformed token: {:?}", diagnostic.kind()),
        );
    }

    let mut region_definitions = HashMap::<(Option<RegionId>, String), Vec<ValueId>>::new();
    let mut block_definitions = HashMap::<(BlockId, String), Vec<ValueId>>::new();
    let mut operation_result_types = Vec::new();
    let mut structurally_lowerable = true;
    struct MatchedLowering {
        name: String,
        shape: Option<OperationShape>,
        lowering: RegisteredLowering,
    }
    let registered = ops
        .iter()
        .map(|op| {
            let range = op.tree().text_range(op.id())?;
            let spelling = text(source.bytes(), range);
            let mnemonic_spelling = text(source.bytes(), op.mnemonic_range()?);
            let mnemonic = mnemonic_spelling
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .unwrap_or(mnemonic_spelling);
            let context = RegisteredLoweringContext {
                spelling,
                mnemonic,
                leading_symbol: op
                    .leading_symbol_range()
                    .map(|range| text(source.bytes(), range)),
                visibility: op
                    .visibility_range()
                    .map(|range| text(source.bytes(), range)),
                arguments: op
                    .arguments()
                    .map(|argument| RegisteredArgument {
                        spelling: text(
                            source.bytes(),
                            argument.tree().text_range(argument.id()).unwrap(),
                        ),
                        attributes: argument
                            .attribute_range()
                            .map(|range| text(source.bytes(), range)),
                    })
                    .collect(),
                function_results: op
                    .function_result_range(source.bytes())
                    .map(|range| text(source.bytes(), range)),
                function_type: op
                    .function_type_range()
                    .map(|range| text(source.bytes(), range)),
            };
            if op.tree().kind(op.id()) == Some(SyntaxKind::Operation) {
                return None;
            }
            if let Some(shape) = registry.operation_shape(mnemonic) {
                return lower_operation_shape(shape, mnemonic, &context).map(|lowering| {
                    MatchedLowering {
                        name: mnemonic.to_owned(),
                        shape: Some(shape),
                        lowering,
                    }
                });
            }
            let descriptor = registry.custom_operation(mnemonic)?;
            descriptor
                .assembly
                .and_then(|program| program.lower(&context))
                .or_else(|| descriptor.lower.and_then(|lower| lower(&context)))
                .map(|lowering| MatchedLowering {
                    name: lowering.name.to_owned(),
                    shape: None,
                    lowering,
                })
        })
        .collect::<Vec<_>>();
    for (i, op) in ops.iter().enumerate() {
        let output_types = registered[i]
            .as_ref()
            .map(|matched| matched.lowering.result_types.clone())
            .unwrap_or_else(|| operation_output_types(*op, source.bytes()));
        let mut result_index = 0usize;
        for result in op.results() {
            let spelling = text(
                source.bytes(),
                result.tree().text_range(result.id()).unwrap(),
            );
            let name = first_identifier(spelling, b'%').unwrap_or_default();
            let count = result
                .number()
                .and_then(|number| {
                    text(source.bytes(), result.tree().text_range(number)?)
                        .bytes()
                        .filter(u8::is_ascii_digit)
                        .fold(None, |value, digit| {
                            Some(value.unwrap_or(0usize) * 10 + (digit - b'0') as usize)
                        })
                })
                .unwrap_or(1usize);
            let available = output_types.len().saturating_sub(result_index);
            let values = (0..count.min(available))
                .map(|offset| ValueId::OperationResult {
                    operation: OperationId::with_owner(i, operation_generation, doc.identity.0),
                    result: (result_index + offset) as u32,
                })
                .collect::<Vec<_>>();
            let scope = parent_blocks
                .get(&op.id())
                .and_then(|block| block_regions.get(block))
                .copied();
            let duplicate = region_definitions.contains_key(&(scope, name.clone()));
            if count == 0 || duplicate {
                push_diagnostic(
                    &mut doc,
                    result.tree().text_range(result.id()).unwrap(),
                    format!("duplicate or empty SSA result group `%{name}`"),
                );
                structurally_lowerable = false;
            } else {
                region_definitions.insert((scope, name.clone()), values);
            }
            result_index += count;
        }
        if result_index != output_types.len() {
            push_diagnostic(
                &mut doc,
                op.tree().text_range(op.id()).unwrap(),
                format!(
                    "result definition count {result_index} does not match result type count {}",
                    output_types.len()
                ),
            );
            structurally_lowerable = false;
        }
        operation_result_types.push(output_types);
    }
    if !structurally_lowerable && mode == LoweringMode::Strict {
        return LoweringResult {
            diagnostics: doc.diagnostics,
            document: None,
            semantically_complete: false,
        };
    }

    let mut block_argument_types = Vec::new();
    for block in &blocks {
        let id = block_ids[&block.id()];
        let mut tys = Vec::new();
        let header_arguments = block
            .arguments()
            .next()
            .is_none()
            .then(|| block_regions.get(&id))
            .flatten()
            .and_then(|region| {
                let syntax_region = &regions[region.index()];
                (syntax_region.blocks().next()?.id() == block.id())
                    .then(|| region_parents.get(&syntax_region.id()))
                    .flatten()
            })
            .and_then(|operation| {
                (registered[operation.index()]
                    .as_ref()
                    .is_some_and(|matched| {
                        matched.shape == Some(OperationShape::FuncLike)
                            || matched.name == "func.func"
                    }))
                .then_some(ops[operation.index()].arguments())
            });
        let syntax_arguments = header_arguments
            .into_iter()
            .flatten()
            .chain(block.arguments());
        for (argument, syntax_argument) in syntax_arguments.enumerate() {
            let spelling = text(
                source.bytes(),
                syntax_argument
                    .tree()
                    .text_range(syntax_argument.id())
                    .unwrap(),
            );
            let name = first_identifier(spelling, b'%').unwrap_or_default();
            let ty = argument_type(spelling);
            let ty_id = intern_type(
                ty,
                syntax_argument
                    .tree()
                    .text_range(syntax_argument.id())
                    .unwrap(),
                &type_aliases,
                &attribute_aliases,
                &mut types,
                &mut type_spellings,
                generation,
                &mut doc,
            );
            tys.push(ty_id);
            if block_definitions.contains_key(&(id, name.clone())) {
                push_diagnostic(
                    &mut doc,
                    syntax_argument
                        .tree()
                        .text_range(syntax_argument.id())
                        .unwrap(),
                    format!("duplicate block argument `%{name}` in block"),
                );
            } else {
                block_definitions.insert(
                    (id, name),
                    vec![ValueId::BlockArgument {
                        block: id,
                        argument: argument as u32,
                    }],
                );
            }
        }
        block_argument_types.push(tys);
    }

    for (i, op) in ops.iter().enumerate() {
        let range = op.tree().text_range(op.id()).unwrap();
        let is_unparsed = op.tree().kind(op.id()) == Some(SyntaxKind::UnparsedCustomOperation);
        let parsed_name = op.mnemonic_range().map(|mnemonic| {
            let spelling = text(source.bytes(), mnemonic);
            spelling
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .unwrap_or(spelling)
        });
        let name = registered[i]
            .as_ref()
            .map(|matched| matched.name.as_str())
            .or(parsed_name)
            .unwrap_or("<invalid>");
        let result_types = operation_result_types[i]
            .iter()
            .map(|spelling| {
                intern_type(
                    spelling,
                    range,
                    &type_aliases,
                    &attribute_aliases,
                    &mut types,
                    &mut type_spellings,
                    generation,
                    &mut doc,
                )
            })
            .collect::<Vec<_>>();
        let function_range = op
            .tree()
            .children(op.id())
            .into_iter()
            .flatten()
            .find(|child| op.tree().kind(*child) == Some(SyntaxKind::FunctionType))
            .and_then(|child| op.tree().text_range(child))
            .unwrap_or(range);
        let function_spelling = registered[i]
            .as_ref()
            .map(|matched| matched.lowering.function_type.as_str())
            .unwrap_or_else(|| text(source.bytes(), function_range));
        let function_type = intern_type(
            function_spelling,
            function_range,
            &type_aliases,
            &attribute_aliases,
            &mut types,
            &mut type_spellings,
            generation,
            &mut doc,
        );
        let operands = op
            .operands()
            .map(|operand| {
                let range = operand.tree().text_range(operand.id()).unwrap();
                resolve_value(
                    text(source.bytes(), range),
                    range,
                    parent_blocks
                        .get(&op.id())
                        .and_then(|block| block_regions.get(block))
                        .copied(),
                    parent_blocks.get(&op.id()).copied(),
                    &region_definitions,
                    &block_definitions,
                    &region_outer,
                    &mut doc,
                )
            })
            .collect::<Vec<_>>();
        let mut attributes = lower_dictionary(
            op.attributes().map(|dict| dict.id()),
            op.tree(),
            source.bytes(),
            &mut strings,
            &mut attrs,
            &mut attribute_spellings,
            &type_aliases,
            &attribute_aliases,
            generation,
            "attribute",
            &mut doc,
        );
        if is_unparsed {
            if let Some(symbol) = leading_symbol(text(source.bytes(), range)) {
                let value = AttributeValue::Symbol(vec![symbol.to_owned()]);
                let index = attrs.intern_value(value);
                if index as usize == attribute_spellings.len() {
                    attribute_spellings.push(format!("@{symbol}"));
                }
                attributes.push((
                    strings.intern("sym_name"),
                    AttributeId::new(index as usize, generation),
                ));
            }
            push_diagnostic(
                &mut doc,
                range,
                format!("unknown custom operation `{name}`"),
            );
        }
        if let Some(matched) = &registered[i] {
            let lowered = &matched.lowering;
            for (name, spelling) in &lowered.attributes {
                let is_inherent = registry
                    .operation(&matched.name)
                    .and_then(|descriptor| descriptor.assembly)
                    .and_then(|program| program.inherent_attribute())
                    == Some(*name);
                if is_inherent
                    && attributes
                        .iter()
                        .any(|(existing, _)| strings.values[*existing as usize] == *name)
                {
                    push_diagnostic(
                        &mut doc,
                        range,
                        format!("duplicate inherent attribute `{name}`"),
                    );
                    doc.complete = false;
                    continue;
                }
                let mut expansion = AliasExpansionState::new(doc.alias_expansion_depth_limit);
                let semantic = lower_attribute_value(
                    spelling,
                    range,
                    &type_aliases,
                    &attribute_aliases,
                    &mut expansion,
                    &mut doc,
                );
                let index = attrs.intern_value(semantic);
                if index as usize == attribute_spellings.len() {
                    attribute_spellings.push(spelling.clone());
                }
                attributes.push((
                    strings.intern(name),
                    AttributeId::new(index as usize, generation),
                ));
            }
            attributes.sort_by_key(|(name, _)| strings.values[*name as usize].clone());
        }
        let properties = lower_dictionary(
            op.properties().map(|dict| dict.id()),
            op.tree(),
            source.bytes(),
            &mut strings,
            &mut attrs,
            &mut attribute_spellings,
            &type_aliases,
            &attribute_aliases,
            generation,
            "property",
            &mut doc,
        );
        let owned_regions = op
            .regions()
            .map(|region| region_ids[&region.id()])
            .collect::<Vec<_>>();
        let location = op.trailing_location().map(|location| {
            let spelling = text(
                source.bytes(),
                location.tree().text_range(location.id()).unwrap(),
            );
            let mut expansion = AliasExpansionState::new(doc.alias_expansion_depth_limit);
            let value = lower_location_value(
                spelling,
                location.tree().text_range(location.id()).unwrap(),
                &type_aliases,
                &attribute_aliases,
                &mut expansion,
                &mut doc,
            );
            let index = locations.intern_value(value);
            if index as usize == location_spellings.len() {
                location_spellings.push(spelling.to_owned());
            }
            LocationId::new(index as usize, generation)
        });
        doc.operations.push(Operation {
            id: OperationId::with_owner(i, operation_generation, doc.identity.0),
            name: strings.intern(name),
            parent: parent_blocks.get(&op.id()).copied(),
            operands: doc.values.push(&operands),
            result_types: doc.types_lists.push(&result_types),
            function_type,
            attributes: doc.attribute_lists.push(&attributes),
            properties: doc.attribute_lists.push(&properties),
            successors: List::default(),
            regions: doc.region_lists.push(&owned_regions),
            location,
            source_range: range,
            unparsed_text: is_unparsed
                .then(|| Arc::from(&source.bytes()[range.start() as usize..range.end() as usize])),
        });
    }

    let mut labels_by_region = HashMap::<RegionId, HashMap<String, Option<BlockId>>>::new();
    for (i, block) in blocks.iter().enumerate() {
        let parent = block_regions[&block_ids[&block.id()]];
        let label = block.label().and_then(|label| {
            let name = first_identifier(
                text(source.bytes(), syntax.tree().text_range(label).unwrap()),
                b'^',
            )?;
            let labels = labels_by_region.entry(parent).or_default();
            if labels.contains_key(&name) {
                push_diagnostic(
                    &mut doc,
                    syntax.tree().text_range(label).unwrap(),
                    format!("duplicate block label `^{name}` in region"),
                );
                labels.insert(name.clone(), None);
            } else {
                labels.insert(name.clone(), Some(BlockId::new(i, generation)));
            }
            Some(strings.intern(&name))
        });
        let operations = syntax
            .tree()
            .children(block.id())
            .into_iter()
            .flatten()
            .filter_map(|child| op_ids.get(&child).copied())
            .collect::<Vec<_>>();
        doc.blocks.push(Block {
            parent,
            label,
            argument_types: doc.types_lists.push(&block_argument_types[i]),
            operations: doc.operation_lists.push(&operations),
        });
    }
    for region in &regions {
        let blocks = region
            .blocks()
            .map(|block| block_ids[&block.id()])
            .collect::<Vec<_>>();
        doc.regions.push(Region {
            generation,
            parent: region_parents[&region.id()],
            blocks: doc.block_lists.push(&blocks),
        });
    }
    doc.types = types.values.clone();
    doc.type_spellings = type_spellings.clone();

    for (i, op) in ops.iter().enumerate() {
        let parent_region = doc.operations[i]
            .parent
            .map(|block| doc.blocks[block.index()].parent);
        let successors = op
            .successors()
            .map(|successor| {
                let Some(range) = successor.tree().text_range(successor.id()) else {
                    let range = op
                        .tree()
                        .text_range(op.id())
                        .unwrap_or_else(|| TextRange::at(source.len()));
                    let diagnostic =
                        push_diagnostic(&mut doc, range, "malformed successor".to_owned());
                    return Successor {
                        block: BlockId::new(usize::MAX, generation),
                        invalid: Some(diagnostic),
                        generation,
                        arguments: doc.values.push(&[]),
                    };
                };
                let spelling = text(source.bytes(), range);
                let label = first_identifier(spelling, b'^').unwrap_or_default();
                let block = parent_region
                    .and_then(|region| labels_by_region.get(&region))
                    .and_then(|labels| labels.get(&label))
                    .copied()
                    .flatten();
                let (block, invalid) = match block {
                    Some(block) => (block, None),
                    None => {
                        let diagnostic = push_diagnostic(
                            &mut doc,
                            range,
                            format!("unresolved block `^{label}`"),
                        );
                        (BlockId::new(usize::MAX, generation), Some(diagnostic))
                    }
                };
                let arguments = successor
                    .arguments()
                    .map(|argument| {
                        match argument.tree().text_range(argument.id()) {
                            Some(range) => resolve_value(
                                text(source.bytes(), range),
                                range,
                                parent_region,
                                doc.operations[i].parent,
                                &region_definitions,
                                &block_definitions,
                                &region_outer,
                                &mut doc,
                            ),
                            None => ValueReference::Invalid(push_diagnostic(
                                &mut doc,
                                range,
                                "malformed successor argument".to_owned(),
                            )),
                        }
                    })
                    .collect::<Vec<_>>();
                let expected = invalid.map_or_else(
                    || {
                        doc.types_lists
                            .get(doc.blocks[block.index()].argument_types)
                            .unwrap_or(&[])
                            .iter()
                            .filter_map(|ty| doc.type_spellings.get(ty.index()).cloned())
                            .collect::<Vec<_>>()
                    },
                    |_| Vec::new(),
                );
                if invalid.is_none() && arguments.len() != expected.len() {
                    push_diagnostic(
                        &mut doc,
                        range,
                        format!(
                            "successor `^{label}` expects {} arguments, got {}",
                            expected.len(),
                            arguments.len()
                        ),
                    );
                }
                for (index, argument) in arguments.iter().enumerate() {
                    if invalid.is_some() {
                        break;
                    }
                    let Some(expected_type) = expected.get(index) else { break };
                    let Some(actual_type) = value_type(&doc, *argument).map(str::to_owned) else {
                        continue;
                    };
                    if actual_type != *expected_type {
                        push_diagnostic(
                            &mut doc,
                            range,
                            format!(
                                "successor `^{label}` argument {index} has type `{actual_type}`, expected `{expected_type}`"
                            ),
                        );
                    }
                }
                Successor {
                    block,
                    invalid,
                    generation,
                    arguments: doc.values.push(&arguments),
                }
            })
            .collect::<Vec<_>>();
        doc.operations[i].successors = doc.successor_lists.push(&successors);
    }

    let roots = ops
        .iter()
        .filter(|op| !parent_blocks.contains_key(&op.id()))
        .map(|op| op_ids[&op.id()])
        .collect::<Vec<_>>();
    doc.roots = doc.operation_lists.push(&roots);
    doc.strings = strings.values;
    doc.types = types.values;
    doc.type_spellings = type_spellings;
    doc.attributes = attrs.values;
    doc.attribute_spellings = attribute_spellings;
    doc.locations = locations.values;
    doc.location_spellings = location_spellings;
    let diagnostics = doc.diagnostics.clone();
    if !doc.complete && mode == LoweringMode::Strict {
        LoweringResult {
            document: None,
            diagnostics,
            semantically_complete: false,
        }
    } else {
        LoweringResult {
            semantically_complete: doc.complete,
            document: Some(doc),
            diagnostics,
        }
    }
}
