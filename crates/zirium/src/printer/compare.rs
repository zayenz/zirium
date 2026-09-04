use super::*;

impl Document {
    pub fn structurally_eq(&self, other: &Self) -> bool {
        if !self.is_semantically_complete()
            || !other.is_semantically_complete()
            || self.validate().is_err()
            || other.validate().is_err()
        {
            return false;
        }
        let Some(maps) = Correspondence::build(self, other) else {
            return false;
        };
        maps.equal_documents(self, other)
    }
}

struct Correspondence {
    operations: HashMap<OperationId, OperationId>,
    regions: HashMap<RegionId, RegionId>,
    blocks: HashMap<BlockId, BlockId>,
}

impl Correspondence {
    fn build(left_doc: &Document, right_doc: &Document) -> Option<Self> {
        let mut maps = Self {
            operations: HashMap::new(),
            regions: HashMap::new(),
            blocks: HashMap::new(),
        };
        let left_roots = left_doc.root_operations();
        let right_roots = right_doc.root_operations();
        if left_roots.len() != right_roots.len() {
            return None;
        }
        for (&left, &right) in left_roots.iter().zip(right_roots) {
            maps.map_operation(left, right, left_doc, right_doc)?;
        }
        Some(maps)
    }

    fn map_operation(
        &mut self,
        left: OperationId,
        right: OperationId,
        left_doc: &Document,
        right_doc: &Document,
    ) -> Option<()> {
        if let Some(mapped) = self.operations.get(&left) {
            return (*mapped == right).then_some(());
        }
        self.operations.insert(left, right);
        let left_regions = left_doc.operation_regions(left)?;
        let right_regions = right_doc.operation_regions(right)?;
        if left_regions.len() != right_regions.len() {
            return None;
        }
        for (&left_region, &right_region) in left_regions.iter().zip(right_regions) {
            self.map_region(left_region, right_region, left_doc, right_doc)?;
        }
        Some(())
    }

    fn map_region(
        &mut self,
        left: RegionId,
        right: RegionId,
        left_doc: &Document,
        right_doc: &Document,
    ) -> Option<()> {
        if let Some(mapped) = self.regions.get(&left) {
            return (*mapped == right).then_some(());
        }
        self.regions.insert(left, right);
        let left_blocks = left_doc.region(left)?.blocks(left_doc)?;
        let right_blocks = right_doc.region(right)?.blocks(right_doc)?;
        if left_blocks.len() != right_blocks.len() {
            return None;
        }
        for (&left_block, &right_block) in left_blocks.iter().zip(right_blocks) {
            self.map_block(left_block, right_block, left_doc, right_doc)?;
        }
        Some(())
    }

    fn map_block(
        &mut self,
        left: BlockId,
        right: BlockId,
        left_doc: &Document,
        right_doc: &Document,
    ) -> Option<()> {
        if let Some(mapped) = self.blocks.get(&left) {
            return (*mapped == right).then_some(());
        }
        self.blocks.insert(left, right);
        let left_args = left_doc.block_argument_types(left)?;
        let right_args = right_doc.block_argument_types(right)?;
        let left_ops = left_doc.block_operations(left)?;
        let right_ops = right_doc.block_operations(right)?;
        if left_args.len() != right_args.len() || left_ops.len() != right_ops.len() {
            return None;
        }
        for (&left_op, &right_op) in left_ops.iter().zip(right_ops) {
            self.map_operation(left_op, right_op, left_doc, right_doc)?;
        }
        Some(())
    }

    fn equal_documents(&self, left: &Document, right: &Document) -> bool {
        self.operations.len() == left.operations().count()
            && self.regions.len()
                == left
                    .operations()
                    .flat_map(|op| left.operation_regions(op).unwrap_or(&[]))
                    .count()
            && self.blocks.len()
                == left
                    .operations()
                    .flat_map(|op| left.operation_regions(op).unwrap_or(&[]))
                    .flat_map(|region| {
                        left.region(*region)
                            .and_then(|r| r.blocks(left))
                            .unwrap_or(&[])
                    })
                    .count()
            && left.root_operations().iter().all(|left_op| {
                self.equal_operation(*left_op, self.operations[left_op], left, right)
            })
    }

    fn equal_operation(
        &self,
        left_id: OperationId,
        right_id: OperationId,
        left: &Document,
        right: &Document,
    ) -> bool {
        if left.operation_name(left_id) != right.operation_name(right_id)
            || !equal_types_by_id(
                left,
                right,
                left.function_type(left_id),
                right.function_type(right_id),
                self,
            )
            || !equal_type_lists(
                left,
                right,
                left.result_types(left_id),
                right.result_types(right_id),
                self,
            )
            || !equal_values(
                left,
                right,
                left.operands(left_id),
                right.operands(right_id),
                self,
            )
            || !equal_entries(
                left,
                right,
                left.operation_attributes(left_id),
                right.operation_attributes(right_id),
                self,
            )
            || !equal_entries(
                left,
                right,
                left.operation_properties(left_id),
                right.operation_properties(right_id),
                self,
            )
            || !equal_locations_by_id(
                left,
                right,
                left.operation_location_id(left_id),
                right.operation_location_id(right_id),
            )
        {
            return false;
        }
        let left_successors = left.successors(left_id).unwrap_or(&[]);
        let right_successors = right.successors(right_id).unwrap_or(&[]);
        if left_successors.len() != right_successors.len() {
            return false;
        }
        for (left_successor, right_successor) in left_successors.iter().zip(right_successors) {
            if self.blocks.get(&left_successor.block) != Some(&right_successor.block)
                || !equal_values(
                    left,
                    right,
                    left.successor_arguments(*left_successor),
                    right.successor_arguments(*right_successor),
                    self,
                )
            {
                return false;
            }
        }
        let left_regions = left.operation_regions(left_id).unwrap_or(&[]);
        let right_regions = right.operation_regions(right_id).unwrap_or(&[]);
        left_regions
            .iter()
            .zip(right_regions)
            .all(|(&lr, &rr)| self.equal_region(lr, rr, left, right))
    }

    fn equal_region(
        &self,
        left_id: RegionId,
        right_id: RegionId,
        left: &Document,
        right: &Document,
    ) -> bool {
        let left_blocks = left
            .region(left_id)
            .and_then(|r| r.blocks(left))
            .unwrap_or(&[]);
        let right_blocks = right
            .region(right_id)
            .and_then(|r| r.blocks(right))
            .unwrap_or(&[]);
        left_blocks.iter().zip(right_blocks).all(|(&lb, &rb)| {
            let left_args = left.block_argument_types(lb);
            let right_args = right.block_argument_types(rb);
            equal_type_lists(left, right, left_args, right_args, self)
                && left
                    .block_operations(lb)
                    .unwrap_or(&[])
                    .iter()
                    .zip(right.block_operations(rb).unwrap_or(&[]))
                    .all(|(&lo, &ro)| self.equal_operation(lo, ro, left, right))
        })
    }
}

fn equal_type_lists(
    left: &Document,
    right: &Document,
    left_values: Option<&[crate::semantic::TypeId]>,
    right_values: Option<&[crate::semantic::TypeId]>,
    maps: &Correspondence,
) -> bool {
    match (left_values, right_values) {
        (Some(left_values), Some(right_values)) => {
            left_values.len() == right_values.len()
                && left_values
                    .iter()
                    .zip(right_values)
                    .all(|(&l, &r)| equal_types_by_id(left, right, Some(l), Some(r), maps))
        }
        (None, None) => true,
        _ => false,
    }
}

fn equal_types_by_id(
    left: &Document,
    right: &Document,
    left_id: Option<crate::semantic::TypeId>,
    right_id: Option<crate::semantic::TypeId>,
    maps: &Correspondence,
) -> bool {
    match (
        left_id.and_then(|id| left.type_value(id)),
        right_id.and_then(|id| right.type_value(id)),
    ) {
        (Some(left_value), Some(right_value)) => {
            equal_types(left, right, left_value, right_value, maps)
        }
        (None, None) => true,
        _ => false,
    }
}

fn equal_types(
    left_doc: &Document,
    right_doc: &Document,
    left: &TypeValue,
    right: &TypeValue,
    maps: &Correspondence,
) -> bool {
    match (left, right) {
        (
            TypeValue::Integer {
                width: lw,
                signedness: ls,
            },
            TypeValue::Integer {
                width: rw,
                signedness: rs,
            },
        ) => lw == rw && ls == rs,
        (TypeValue::Float(l), TypeValue::Float(r)) => l == r,
        (TypeValue::Opaque(l), TypeValue::Opaque(r)) => l == r,
        (TypeValue::Index, TypeValue::Index) => true,
        (TypeValue::Tuple(l), TypeValue::Tuple(r)) => {
            l.len() == r.len()
                && l.iter()
                    .zip(r)
                    .all(|(l, r)| equal_types(left_doc, right_doc, l, r, maps))
        }
        (
            TypeValue::Function {
                inputs: li,
                results: lr,
            },
            TypeValue::Function {
                inputs: ri,
                results: rr,
            },
        ) => {
            equal_type_values(left_doc, right_doc, li, ri, maps)
                && equal_type_values(left_doc, right_doc, lr, rr, maps)
        }
        (
            TypeValue::Tensor {
                dimensions: ld,
                element: le,
                encoding: ln,
                unranked: lu,
            },
            TypeValue::Tensor {
                dimensions: rd,
                element: re,
                encoding: rn,
                unranked: ru,
            },
        ) => {
            lu == ru
                && ld == rd
                && equal_types(left_doc, right_doc, le, re, maps)
                && equal_attribute_options(left_doc, right_doc, ln.as_deref(), rn.as_deref(), maps)
        }
        (
            TypeValue::Vector {
                dimensions: ld,
                element: le,
                scalable: ls,
            },
            TypeValue::Vector {
                dimensions: rd,
                element: re,
                scalable: rs,
            },
        ) => ld == rd && ls == rs && equal_types(left_doc, right_doc, le, re, maps),
        (
            TypeValue::MemRef {
                dimensions: ld,
                element: le,
                layout: ll,
                memory_space: lm,
            },
            TypeValue::MemRef {
                dimensions: rd,
                element: re,
                layout: rl,
                memory_space: rm,
            },
        ) => {
            ld == rd
                && equal_types(left_doc, right_doc, le, re, maps)
                && equal_layouts(left_doc, right_doc, ll.as_ref(), rl.as_ref(), maps)
                && equal_attribute_options(left_doc, right_doc, lm.as_deref(), rm.as_deref(), maps)
        }
        _ => false,
    }
}

fn equal_type_values(
    left_doc: &Document,
    right_doc: &Document,
    left: &[TypeValue],
    right: &[TypeValue],
    maps: &Correspondence,
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(l, r)| equal_types(left_doc, right_doc, l, r, maps))
}

fn equal_attribute_options(
    left_doc: &Document,
    right_doc: &Document,
    left: Option<&AttributeValue>,
    right: Option<&AttributeValue>,
    maps: &Correspondence,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => equal_attributes(left_doc, right_doc, left, right, maps),
        (None, None) => true,
        _ => false,
    }
}

fn equal_layouts(
    left_doc: &Document,
    right_doc: &Document,
    left: Option<&MemRefLayout>,
    right: Option<&MemRefLayout>,
    maps: &Correspondence,
) -> bool {
    match (left, right) {
        (Some(MemRefLayout::AffineMap(l)), Some(MemRefLayout::AffineMap(r))) => {
            equal_affine_maps(left_doc, right_doc, *l, *r)
        }
        (
            Some(MemRefLayout::Opaque {
                spelling: ls,
                parameters: lp,
            }),
            Some(MemRefLayout::Opaque {
                spelling: rs,
                parameters: rp,
            }),
        ) => ls == rs && equal_attribute_values(left_doc, right_doc, lp, rp, maps),
        (Some(MemRefLayout::Attribute(l)), Some(MemRefLayout::Attribute(r))) => {
            equal_attributes(left_doc, right_doc, l, r, maps)
        }
        (None, None) => true,
        _ => false,
    }
}

fn equal_entries(
    left_doc: &Document,
    right_doc: &Document,
    left: Option<&[(u32, crate::semantic::AttributeId)]>,
    right: Option<&[(u32, crate::semantic::AttributeId)]>,
    maps: &Correspondence,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            left.len() == right.len()
                && left.iter().zip(right).all(|(&(lk, lv), &(rk, rv))| {
                    left_doc.string(lk) == right_doc.string(rk)
                        && left_doc
                            .attribute_value(lv)
                            .zip(right_doc.attribute_value(rv))
                            .is_some_and(|(l, r)| equal_attributes(left_doc, right_doc, l, r, maps))
                })
        }
        (None, None) => true,
        _ => false,
    }
}

fn equal_values(
    _left_doc: &Document,
    _right_doc: &Document,
    left: Option<&[ValueReference]>,
    right: Option<&[ValueReference]>,
    maps: &Correspondence,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(&l, &r)| equal_value(l, r, maps))
        }
        (None, None) => true,
        _ => false,
    }
}

fn equal_value(left: ValueReference, right: ValueReference, maps: &Correspondence) -> bool {
    match (left, right) {
        (ValueReference::Resolved(left), ValueReference::Resolved(right)) => match (left, right) {
            (
                ValueId::OperationResult {
                    operation: lo,
                    result: lr,
                },
                ValueId::OperationResult {
                    operation: ro,
                    result: rr,
                },
            ) => maps.operations.get(&lo) == Some(&ro) && lr == rr,
            (
                ValueId::BlockArgument {
                    block: lb,
                    argument: la,
                },
                ValueId::BlockArgument {
                    block: rb,
                    argument: ra,
                },
            ) => maps.blocks.get(&lb) == Some(&rb) && la == ra,
            _ => false,
        },
        (ValueReference::Invalid(_), ValueReference::Invalid(_)) => false,
        _ => false,
    }
}

fn equal_attributes(
    left_doc: &Document,
    right_doc: &Document,
    left: &AttributeValue,
    right: &AttributeValue,
    maps: &Correspondence,
) -> bool {
    match (left, right) {
        (AttributeValue::Integer(l), AttributeValue::Integer(r))
        | (AttributeValue::Float(l), AttributeValue::Float(r))
        | (AttributeValue::String(l), AttributeValue::String(r)) => l == r,
        (AttributeValue::Type(l), AttributeValue::Type(r)) => {
            equal_types(left_doc, right_doc, l, r, maps)
        }
        (AttributeValue::Symbol(l), AttributeValue::Symbol(r)) => l == r,
        (AttributeValue::Array(l), AttributeValue::Array(r)) => {
            equal_attribute_values(left_doc, right_doc, l, r, maps)
        }
        (AttributeValue::Dictionary(l), AttributeValue::Dictionary(r)) => {
            l.len() == r.len()
                && l.iter().zip(r).all(|((lk, lv), (rk, rv))| {
                    lk == rk && equal_attributes(left_doc, right_doc, lv, rv, maps)
                })
        }
        (AttributeValue::Location(l), AttributeValue::Location(r)) => equal_locations(l, r),
        (AttributeValue::AffineMap(l), AttributeValue::AffineMap(r)) => {
            equal_affine_maps(left_doc, right_doc, *l, *r)
        }
        (AttributeValue::IntegerSet(l), AttributeValue::IntegerSet(r)) => {
            equal_integer_sets(left_doc, right_doc, *l, *r)
        }
        (AttributeValue::Large(l), AttributeValue::Large(r)) => l == r,
        (AttributeValue::WideNumber(l), AttributeValue::WideNumber(r))
        | (AttributeValue::Opaque(l), AttributeValue::Opaque(r)) => l == r,
        _ => false,
    }
}

fn equal_attribute_values(
    left_doc: &Document,
    right_doc: &Document,
    left: &[AttributeValue],
    right: &[AttributeValue],
    maps: &Correspondence,
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(l, r)| equal_attributes(left_doc, right_doc, l, r, maps))
}

fn equal_affine_maps(
    left_doc: &Document,
    right_doc: &Document,
    left: crate::semantic::AffineMapId,
    right: crate::semantic::AffineMapId,
) -> bool {
    let (Some(left), Some(right)) = (left_doc.affine_map(left), right_doc.affine_map(right)) else {
        return false;
    };
    left.dimensions == right.dimensions
        && left.symbols == right.symbols
        && left.results.len() == right.results.len()
        && left
            .results
            .iter()
            .zip(&right.results)
            .all(|(&l, &r)| equal_affine_exprs(left_doc, right_doc, l, r))
}

fn equal_affine_exprs(
    left_doc: &Document,
    right_doc: &Document,
    left: crate::semantic::AffineExprId,
    right: crate::semantic::AffineExprId,
) -> bool {
    match (
        left_doc.affine_expression(left),
        right_doc.affine_expression(right),
    ) {
        (Some(AffineExprValue::Dimension(l)), Some(AffineExprValue::Dimension(r)))
        | (Some(AffineExprValue::Symbol(l)), Some(AffineExprValue::Symbol(r))) => l == r,
        (Some(AffineExprValue::Constant(l)), Some(AffineExprValue::Constant(r))) => l == r,
        (
            Some(AffineExprValue::Binary {
                operator: lo,
                left: ll,
                right: lr,
            }),
            Some(AffineExprValue::Binary {
                operator: ro,
                left: rl,
                right: rr,
            }),
        ) => {
            lo == ro
                && equal_affine_exprs(left_doc, right_doc, *ll, *rl)
                && equal_affine_exprs(left_doc, right_doc, *lr, *rr)
        }
        _ => false,
    }
}

fn equal_integer_sets(
    left_doc: &Document,
    right_doc: &Document,
    left: crate::semantic::IntegerSetId,
    right: crate::semantic::IntegerSetId,
) -> bool {
    let (Some(left), Some(right)) = (left_doc.integer_set(left), right_doc.integer_set(right))
    else {
        return false;
    };
    left.dimensions == right.dimensions
        && left.symbols == right.symbols
        && left.constraints.len() == right.constraints.len()
        && left
            .constraints
            .iter()
            .zip(&right.constraints)
            .all(|(l, r)| {
                l.relation == r.relation
                    && equal_affine_exprs(left_doc, right_doc, l.left, r.left)
                    && equal_affine_exprs(left_doc, right_doc, l.right, r.right)
            })
}

fn equal_locations_by_id(
    left_doc: &Document,
    right_doc: &Document,
    left: Option<Option<crate::semantic::LocationId>>,
    right: Option<Option<crate::semantic::LocationId>>,
) -> bool {
    match (left, right) {
        (Some(Some(left)), Some(Some(right))) => left_doc
            .location_value(left)
            .zip(right_doc.location_value(right))
            .is_some_and(|(l, r)| equal_locations(l, r)),
        (Some(None), Some(None)) => true,
        (None, None) => true,
        _ => false,
    }
}

fn equal_locations(left: &LocationValue, right: &LocationValue) -> bool {
    match (left, right) {
        (LocationValue::Unknown, LocationValue::Unknown) => true,
        (
            LocationValue::FileLineColumn {
                file: lf,
                line: ll,
                column: lc,
            },
            LocationValue::FileLineColumn {
                file: rf,
                line: rl,
                column: rc,
            },
        ) => lf == rf && ll == rl && lc == rc,
        (
            LocationValue::Name {
                name: ln,
                child: lc,
                metadata: lm,
            },
            LocationValue::Name {
                name: rn,
                child: rc,
                metadata: rm,
            },
        ) => {
            ln == rn
                && lm == rm
                && match (lc.as_deref(), rc.as_deref()) {
                    (Some(l), Some(r)) => equal_locations(l, r),
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            LocationValue::CallSite {
                callee: lc,
                caller: ll,
            },
            LocationValue::CallSite {
                callee: rc,
                caller: rl,
            },
        ) => equal_locations(lc, rc) && equal_locations(ll, rl),
        (
            LocationValue::Fused {
                metadata: lm,
                locations: ll,
            },
            LocationValue::Fused {
                metadata: rm,
                locations: rl,
            },
        ) => {
            lm == rm
                && ll.len() == rl.len()
                && ll.iter().zip(rl).all(|(l, r)| equal_locations(l, r))
        }
        _ => false,
    }
}
