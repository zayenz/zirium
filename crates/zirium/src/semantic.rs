//! Narrow semantic IR and lowering for Zirium's generic proving fixture.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
    sync::Arc,
    sync::{Mutex, OnceLock, RwLock, Weak},
};

use crate::{
    SyntaxKind,
    dialect::{DialectRegistry, OperationShape, lower_operation_shape},
    parser::ParsedFile,
    source::TextRange,
};

static NEXT_DOCUMENT_IDENTITY: OnceLock<Mutex<u128>> = OnceLock::new();
static LIVE_DOCUMENT_IDENTITIES: OnceLock<Mutex<HashMap<u128, Weak<DocumentIdentity>>>> =
    OnceLock::new();

#[derive(Debug)]
struct DocumentIdentity(u128);

#[derive(Debug)]
struct AliasExpansionState {
    limit: usize,
    active: HashSet<String>,
}

impl AliasExpansionState {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            active: HashSet::new(),
        }
    }

    fn enter(&mut self, alias: &str, family: &str) -> Result<(), String> {
        if self.active.contains(alias) {
            return Err(format!("cyclic {family} alias `{alias}`"));
        }
        if self.active.len() >= self.limit {
            return Err(format!(
                "alias expansion depth exceeds limit of {}",
                self.limit
            ));
        }
        self.active.insert(alias.to_owned());
        Ok(())
    }

    fn exit(&mut self, alias: &str) {
        self.active.remove(alias);
    }
}

impl Drop for DocumentIdentity {
    fn drop(&mut self) {
        if let Some(live) = LIVE_DOCUMENT_IDENTITIES.get() {
            let mut live = live
                .lock()
                .expect("document identity registry is not poisoned");
            if live
                .get(&self.0)
                .is_some_and(|identity| identity.upgrade().is_none())
            {
                live.remove(&self.0);
            }
        }
    }
}

fn allocate_document_identity() -> Arc<DocumentIdentity> {
    let live = LIVE_DOCUMENT_IDENTITIES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut live = live
        .lock()
        .expect("document identity registry is not poisoned");
    let next = NEXT_DOCUMENT_IDENTITY.get_or_init(|| Mutex::new(1));
    let mut next = next
        .lock()
        .expect("document identity allocator is not poisoned");
    loop {
        let candidate = *next;
        *next = next
            .checked_add(1)
            .expect("document identity space exhausted");
        if candidate == 0 {
            continue;
        }
        if live
            .get(&candidate)
            .is_none_or(|identity| identity.upgrade().is_none())
        {
            let identity = Arc::new(DocumentIdentity(candidate));
            live.insert(candidate, Arc::downgrade(&identity));
            return identity;
        }
    }
}

#[derive(Debug, Default)]
struct OperationIdentityState {
    next: u32,
    allocated: HashSet<u32>,
}

impl OperationIdentityState {
    fn allocate(&mut self) -> u32 {
        loop {
            self.next = self.next.wrapping_add(1).max(1);
            if self.allocated.insert(self.next) {
                return self.next;
            }
        }
    }
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name {
            index: u32,
            generation: u128,
        }
        impl $name {
            fn new(index: usize, generation: u128) -> Self {
                Self {
                    index: index as u32,
                    generation,
                }
            }
            fn index(self) -> usize {
                self.index as usize
            }
        }
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperationId {
    index: u32,
    generation: u32,
    owner: u128,
}
impl OperationId {
    fn index(self) -> usize {
        self.index as usize
    }
}
id_type!(RegionId);
id_type!(BlockId);
id_type!(TypeId);
id_type!(AttributeId);
id_type!(LocationId);
id_type!(DiagnosticId);
id_type!(AffineExprId);
id_type!(AffineMapId);
id_type!(IntegerSetId);

impl OperationId {
    fn with_owner(index: usize, generation: u32, owner: u128) -> Self {
        Self {
            index: index as u32,
            generation,
            owner,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RetentionProfile {
    /// Retain source bytes and the parser CST, without semantic-to-syntax mappings.
    SyntaxOnly,
    /// Retain semantic storage only; source bytes, CST, and mappings are discarded.
    SemanticOnly,
    /// Retain source bytes, the parser CST, and sparse operation-to-source mappings.
    Hybrid,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LargeAttributeValue {
    Dense(Arc<[u8]>),
    Sparse(Arc<[u8]>),
    Resource(Arc<[u8]>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AffineBinaryOperator {
    Add,
    Subtract,
    Multiply,
    FloorDiv,
    CeilDiv,
    Mod,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AffineExprValue {
    Dimension(u32),
    Symbol(u32),
    Constant(i64),
    Binary {
        operator: AffineBinaryOperator,
        left: AffineExprId,
        right: AffineExprId,
    },
    Invalid(DiagnosticId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffineMapValue {
    pub dimensions: u32,
    pub symbols: u32,
    pub results: Vec<AffineExprId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntegerSetRelation {
    Equal,
    GreaterEqual,
    LessEqual,
    Invalid(DiagnosticId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegerSetConstraint {
    pub left: AffineExprId,
    pub relation: IntegerSetRelation,
    pub right: AffineExprId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegerSetValue {
    pub dimensions: u32,
    pub symbols: u32,
    pub constraints: Vec<IntegerSetConstraint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeValue {
    Integer {
        width: u32,
        signedness: Option<bool>,
    },
    Float(String),
    Index,
    Tuple(Vec<TypeValue>),
    Tensor {
        dimensions: Vec<ShapedDimension>,
        element: Box<TypeValue>,
        encoding: Option<Box<AttributeValue>>,
        unranked: bool,
    },
    Vector {
        dimensions: Vec<ShapedDimension>,
        element: Box<TypeValue>,
        scalable: Vec<bool>,
    },
    MemRef {
        dimensions: Vec<ShapedDimension>,
        element: Box<TypeValue>,
        layout: Option<MemRefLayout>,
        memory_space: Option<Box<AttributeValue>>,
    },
    Function {
        inputs: Vec<TypeValue>,
        results: Vec<TypeValue>,
    },
    Opaque(Arc<[u8]>),
    Invalid(DiagnosticId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapedDimension {
    pub size: Option<u64>,
    pub scalable: bool,
    pub invalid: Option<DiagnosticId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemRefLayout {
    AffineMap(AffineMapId),
    Opaque {
        spelling: String,
        parameters: Vec<AttributeValue>,
    },
    Attribute(Box<AttributeValue>),
    Invalid(DiagnosticId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttributeValue {
    Boolean(bool),
    Integer(String),
    Float(String),
    String(String),
    Type(TypeValue),
    Symbol(Vec<String>),
    Array(Vec<AttributeValue>),
    Dictionary(Vec<(String, AttributeValue)>),
    Location(LocationValue),
    AffineMap(AffineMapId),
    IntegerSet(IntegerSetId),
    Large(LargeAttributeValue),
    WideNumber(Arc<[u8]>),
    Opaque(Arc<[u8]>),
    Invalid(DiagnosticId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LocationValue {
    Unknown,
    FileLineColumn {
        file: String,
        line: u64,
        column: u64,
    },
    Name {
        name: String,
        child: Option<Box<LocationValue>>,
        metadata: Option<String>,
    },
    CallSite {
        callee: Box<LocationValue>,
        caller: Box<LocationValue>,
    },
    Fused {
        metadata: Option<String>,
        locations: Vec<LocationValue>,
    },
    Invalid(DiagnosticId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueId {
    OperationResult { operation: OperationId, result: u32 },
    BlockArgument { block: BlockId, argument: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueReference {
    Resolved(ValueId),
    Invalid(DiagnosticId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Successor {
    pub block: BlockId,
    invalid: Option<DiagnosticId>,
    generation: u128,
    arguments: List<ValueReference>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringMode {
    /// Reject incomplete lowering by returning no document.
    Strict,
    /// Return a document with invalid sentinels and recovery diagnostics.
    BestEffort,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticDiagnostic {
    pub range: TextRange,
    pub message: String,
}

#[derive(Debug)]
pub struct LoweringResult {
    pub document: Option<Document>,
    pub diagnostics: Vec<SemanticDiagnostic>,
    pub semantically_complete: bool,
}

/// Read-only CST input for a registered lowerer.
pub struct RegisteredLoweringContext<'a> {
    spelling: &'a str,
}

impl<'a> RegisteredLoweringContext<'a> {
    pub fn spelling(&self) -> &'a str {
        self.spelling
    }
}

/// Arena-independent result returned by a registered lowerer.
pub struct RegisteredLowering {
    pub name: &'static str,
    pub result_types: Vec<String>,
    pub function_type: String,
    pub attributes: Vec<(&'static str, String)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DocumentStatistics {
    pub operations: usize,
    pub regions: usize,
    pub blocks: usize,
    pub pooled_list_entries: usize,
    pub local_strings: usize,
    pub local_types: usize,
    pub local_attributes: usize,
    pub affine_expressions: usize,
    pub affine_maps: usize,
    pub integer_sets: usize,
    pub retained_source_bytes: usize,
    pub retained_cst_nodes: usize,
    pub retained_mapping_entries: usize,
    pub direct_owned_bytes: usize,
    pub document_index_bytes: usize,
    pub retained_cst_bytes: usize,
    pub source_storage_shared: bool,
    pub cst_storage_shared: bool,
    pub payload_blob_bytes: usize,
    pub payload_blobs: usize,
    pub use_index_entries: usize,
    pub symbol_index_entries: usize,
    pub dominance_index_entries: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedRegistryStatistics {
    pub strings: usize,
    pub types: usize,
    pub attributes: usize,
}

#[derive(Debug, Default)]
pub struct SharedRegistry;
impl SharedRegistry {
    pub fn statistics(&self) -> SharedRegistryStatistics {
        SharedRegistryStatistics::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct List<T> {
    start: u32,
    len: u32,
    _kind: PhantomData<T>,
}
impl<T> Default for List<T> {
    fn default() -> Self {
        Self {
            start: 0,
            len: 0,
            _kind: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
struct ListPool<T>(Vec<T>);
impl<T> Default for ListPool<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UseSite {
    Operand {
        operation: OperationId,
        index: u32,
    },
    SuccessorArgument {
        operation: OperationId,
        successor: u32,
        argument: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolIndexDiagnostic {
    pub operation: OperationId,
    pub symbol: String,
}

#[derive(Clone, Debug, Default)]
struct UseIndex {
    revision: u64,
    uses: HashMap<ValueId, Vec<UseSite>>,
}

#[derive(Clone, Debug, Default)]
struct SymbolIndex {
    revision: u64,
    registry: u64,
    scopes: HashMap<OperationId, HashMap<String, OperationId>>,
    diagnostics: Vec<SymbolIndexDiagnostic>,
}

#[derive(Clone, Debug, Default)]
struct DominanceIndex {
    revision: u64,
    registry: u64,
    regions: HashMap<RegionId, RegionDominance>,
    operation_positions: HashMap<OperationId, usize>,
}

#[derive(Clone, Debug, Default)]
struct RegionDominance {
    blocks: Vec<BlockId>,
    block_indices: HashMap<BlockId, usize>,
    successors: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
    immediate_dominators: Vec<Option<usize>>,
    dominator_tree: Vec<Vec<usize>>,
    intervals: Vec<Option<(usize, usize)>>,
    reachable: Vec<bool>,
}

impl RegionDominance {
    fn entry_count(&self) -> usize {
        self.blocks.len()
            + self.block_indices.len()
            + self.successors.len()
            + self.successors.iter().map(Vec::len).sum::<usize>()
            + self.predecessors.len()
            + self.predecessors.iter().map(Vec::len).sum::<usize>()
            + self
                .immediate_dominators
                .iter()
                .filter(|dominator| dominator.is_some())
                .count()
            + self.dominator_tree.len()
            + self.dominator_tree.iter().map(Vec::len).sum::<usize>()
            + self
                .intervals
                .iter()
                .filter(|interval| interval.is_some())
                .count()
            + self.reachable.len()
    }
}

#[derive(Default)]
struct VerificationContext {
    block_dominators: HashMap<BlockId, HashSet<BlockId>>,
    operation_positions: HashMap<OperationId, usize>,
}

#[derive(Clone, Copy)]
struct ValueUsePoint {
    operation: OperationId,
    block: BlockId,
    position: usize,
}

enum VisibilityAnalysis<'a> {
    Verification(&'a VerificationContext),
    Indexed(&'a DominanceIndex),
}

impl VisibilityAnalysis<'_> {
    fn operation_position(&self, operation: OperationId) -> Option<usize> {
        match self {
            Self::Verification(context) => context.operation_positions.get(&operation).copied(),
            Self::Indexed(index) => index.operation_positions.get(&operation).copied(),
        }
    }

    fn block_dominates(
        &self,
        region: RegionId,
        definition_block: BlockId,
        use_block: BlockId,
    ) -> bool {
        match self {
            Self::Verification(context) => context
                .block_dominators
                .get(&use_block)
                .is_some_and(|blocks| blocks.contains(&definition_block)),
            Self::Indexed(index) => {
                let Some(region) = index.regions.get(&region) else {
                    return false;
                };
                let (Some(&definition), Some(&use_position)) = (
                    region.block_indices.get(&definition_block),
                    region.block_indices.get(&use_block),
                ) else {
                    return false;
                };
                if !region.reachable[use_position] {
                    return true;
                }
                let (Some((definition_entry, definition_exit)), Some((use_entry, use_exit))) =
                    (region.intervals[definition], region.intervals[use_position])
                else {
                    return false;
                };
                definition_entry <= use_entry && use_exit <= definition_exit
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct AnalysisCaches {
    uses: Option<UseIndex>,
    symbols: Option<SymbolIndex>,
    dominance: Option<DominanceIndex>,
}

#[derive(Debug)]
struct AnalysisStore(RwLock<AnalysisCaches>, Mutex<()>);

impl Default for AnalysisStore {
    fn default() -> Self {
        Self(RwLock::new(AnalysisCaches::default()), Mutex::new(()))
    }
}

impl Clone for AnalysisStore {
    fn clone(&self) -> Self {
        Self(
            RwLock::new(
                self.0
                    .read()
                    .expect("analysis cache lock is not poisoned")
                    .clone(),
            ),
            Mutex::new(()),
        )
    }
}

impl AnalysisStore {
    fn borrow(&self) -> std::sync::RwLockReadGuard<'_, AnalysisCaches> {
        self.0.read().expect("analysis cache lock is not poisoned")
    }
}
impl<T: Copy> ListPool<T> {
    fn push(&mut self, values: &[T]) -> List<T> {
        let list = List {
            start: self.0.len() as u32,
            len: values.len() as u32,
            _kind: PhantomData,
        };
        self.0.extend_from_slice(values);
        list
    }
    fn get(&self, list: List<T>) -> Option<&[T]> {
        self.0
            .get(list.start as usize..list.start as usize + list.len as usize)
    }
}

#[derive(Clone, Debug)]
pub struct Operation {
    id: OperationId,
    name: u32,
    parent: Option<BlockId>,
    operands: List<ValueReference>,
    result_types: List<TypeId>,
    function_type: TypeId,
    attributes: List<(u32, AttributeId)>,
    properties: List<(u32, AttributeId)>,
    successors: List<Successor>,
    regions: List<RegionId>,
    location: Option<LocationId>,
    source_range: TextRange,
    unparsed_text: Option<Arc<[u8]>>,
}
#[derive(Clone, Debug)]
pub struct Region {
    generation: u128,
    parent: OperationId,
    blocks: List<BlockId>,
}
#[derive(Clone, Debug)]
pub struct Block {
    parent: RegionId,
    label: Option<u32>,
    argument_types: List<TypeId>,
    operations: List<OperationId>,
}

#[derive(Clone, Debug)]
pub struct Document {
    generation: u128,
    identity: Arc<DocumentIdentity>,
    operation_identities: Arc<Mutex<OperationIdentityState>>,
    operations: Vec<Operation>,
    operation_generations: Vec<u32>,
    operation_alive: Vec<bool>,
    regions: Vec<Region>,
    blocks: Vec<Block>,
    values: ListPool<ValueReference>,
    types_lists: ListPool<TypeId>,
    attribute_lists: ListPool<(u32, AttributeId)>,
    successor_lists: ListPool<Successor>,
    region_lists: ListPool<RegionId>,
    block_lists: ListPool<BlockId>,
    operation_lists: ListPool<OperationId>,
    strings: Vec<String>,
    types: Vec<TypeValue>,
    type_spellings: Vec<String>,
    attributes: Vec<AttributeValue>,
    attribute_spellings: Vec<String>,
    locations: Vec<LocationValue>,
    location_spellings: Vec<String>,
    affine_expressions: Vec<AffineExprValue>,
    affine_maps: Vec<AffineMapValue>,
    integer_sets: Vec<IntegerSetValue>,
    diagnostics: Vec<SemanticDiagnostic>,
    roots: List<OperationId>,
    complete: bool,
    retention_profile: RetentionProfile,
    retained_source: Option<Arc<[u8]>>,
    retained_syntax: Option<Arc<crate::representation::SyntaxTree>>,
    syntax_map: Arc<[(OperationId, TextRange)]>,
    blob_ranges: Arc<[TextRange]>,
    dirty_operations: HashSet<OperationId>,
    dirty_blocks: HashSet<BlockId>,
    revision: u64,
    analyses: AnalysisStore,
    attribute_depth_limit: usize,
    alias_expansion_depth_limit: usize,
}

impl Document {
    pub fn operations(&self) -> impl Iterator<Item = OperationId> + '_ {
        self.operation_generations
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, _)| self.operation_alive[*index])
            .map(|(index, generation)| OperationId::with_owner(index, generation, self.identity.0))
    }
    pub fn root_operations(&self) -> &[OperationId] {
        self.operation_lists.get(self.roots).unwrap_or(&[])
    }
    pub fn operation(&self, id: OperationId) -> Option<&Operation> {
        self.valid_operation(id)
            .then(|| &self.operations[id.index()])
    }
    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.valid(id.index, id.generation, self.regions.len())
            .then(|| &self.regions[id.index()])
    }
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.valid(id.index, id.generation, self.blocks.len())
            .then(|| &self.blocks[id.index()])
    }
    pub fn operation_name(&self, id: OperationId) -> Option<&str> {
        self.operation(id)
            .and_then(|op| self.strings.get(op.name as usize))
            .map(String::as_str)
    }
    /// Resolves a spelling already present in this document without interning it.
    pub fn existing_string_index(&self, value: &str) -> Option<u32> {
        self.strings
            .iter()
            .position(|item| item == value)
            .map(|index| index as u32)
    }
    pub fn string_at(&self, index: u32) -> Option<&str> {
        self.strings.get(index as usize).map(String::as_str)
    }
    /// Returns the document-local name index stored by an operation.
    pub fn operation_name_index(&self, id: OperationId) -> Option<u32> {
        self.operation(id).map(|operation| operation.name)
    }
    pub fn operands(&self, id: OperationId) -> Option<&[ValueReference]> {
        self.operation(id)
            .and_then(|op| self.values.get(op.operands))
    }
    pub fn result_types(&self, id: OperationId) -> Option<&[TypeId]> {
        self.operation(id)
            .and_then(|op| self.types_lists.get(op.result_types))
    }
    pub fn function_type(&self, id: OperationId) -> Option<TypeId> {
        Some(self.operation(id)?.function_type)
    }
    pub fn operation_regions(&self, id: OperationId) -> Option<&[RegionId]> {
        self.operation(id)
            .and_then(|op| self.region_lists.get(op.regions))
    }
    pub fn properties(&self, id: OperationId) -> Option<impl Iterator<Item = (&str, &str)> + '_> {
        let values = self
            .operation(id)
            .and_then(|op| self.attribute_lists.get(op.properties))?;
        Some(values.iter().filter_map(|(name, value)| {
            Some((
                self.strings.get(*name as usize)?.as_str(),
                self.attribute_spellings.get(value.index())?.as_str(),
            ))
        }))
    }
    pub fn successors(&self, id: OperationId) -> Option<&[Successor]> {
        self.operation(id)
            .and_then(|op| self.successor_lists.get(op.successors))
    }
    pub fn successor_arguments(&self, successor: Successor) -> Option<&[ValueReference]> {
        (successor.generation == self.generation)
            .then(|| self.values.get(successor.arguments))
            .flatten()
    }
    pub fn block_argument_types(&self, id: BlockId) -> Option<&[TypeId]> {
        self.block(id)
            .and_then(|block| self.types_lists.get(block.argument_types))
    }
    pub fn block_label(&self, id: BlockId) -> Option<Option<&str>> {
        let block = self.block(id)?;
        Some(
            block
                .label
                .and_then(|label| self.strings.get(label as usize).map(String::as_str)),
        )
    }
    pub fn operation_location(&self, id: OperationId) -> Option<Option<&str>> {
        let op = self.operation(id)?;
        Some(op.location.and_then(|location| {
            self.location_spellings
                .get(location.index())
                .map(String::as_str)
        }))
    }
    pub fn operation_location_value(&self, id: OperationId) -> Option<Option<&LocationValue>> {
        let op = self.operation(id)?;
        Some(
            op.location
                .and_then(|location| self.locations.get(location.index())),
        )
    }
    pub fn operation_source_range(&self, id: OperationId) -> Option<TextRange> {
        self.operation(id).map(|op| op.source_range)
    }
    pub fn operation_is_unparsed(&self, id: OperationId) -> Option<bool> {
        Some(self.operation(id)?.unparsed_text.is_some())
    }
    pub fn operation_unparsed_text(&self, id: OperationId) -> Option<&[u8]> {
        self.operation(id)?.unparsed_text.as_deref()
    }
    pub fn block_operations(&self, id: BlockId) -> Option<&[OperationId]> {
        self.block(id)
            .and_then(|b| self.operation_lists.get(b.operations))
    }
    pub fn attributes(&self, id: OperationId) -> Option<impl Iterator<Item = (&str, &str)> + '_> {
        let values = self
            .operation(id)
            .and_then(|op| self.attribute_lists.get(op.attributes))?;
        Some(values.iter().filter_map(|(name, value)| {
            Some((
                self.strings.get(*name as usize)?.as_str(),
                self.attribute_spellings.get(value.index())?.as_str(),
            ))
        }))
    }
    pub fn attribute_entries(
        &self,
        id: OperationId,
    ) -> Option<impl Iterator<Item = (&str, AttributeId)> + '_> {
        let values = self
            .operation(id)
            .and_then(|op| self.attribute_lists.get(op.attributes))?;
        Some(
            values.iter().filter_map(|(name, value)| {
                Some((self.strings.get(*name as usize)?.as_str(), *value))
            }),
        )
    }
    pub fn attribute_entry(&self, id: OperationId, index: usize) -> Option<(&str, AttributeId)> {
        let (name, value) = *self
            .attribute_lists
            .get(self.operation(id)?.attributes)?
            .get(index)?;
        Some((self.strings.get(name as usize)?.as_str(), value))
    }
    pub fn attribute_id(&self, id: OperationId, name: &str) -> Option<AttributeId> {
        self.attribute_lists
            .get(self.operation(id)?.attributes)?
            .iter()
            .find_map(|(key, value)| {
                (self.strings.get(*key as usize).map(String::as_str) == Some(name))
                    .then_some(*value)
            })
    }
    pub fn type_spelling(&self, id: TypeId) -> Option<&str> {
        self.valid(id.index, id.generation, self.types.len())
            .then(|| self.type_spellings[id.index()].as_str())
    }
    pub fn type_value(&self, id: TypeId) -> Option<&TypeValue> {
        self.valid(id.index, id.generation, self.types.len())
            .then(|| &self.types[id.index()])
    }
    pub(crate) fn value_type_id(&self, reference: ValueReference) -> Option<TypeId> {
        match reference {
            ValueReference::Resolved(ValueId::OperationResult { operation, result }) => {
                self.result_types(operation)?.get(result as usize).copied()
            }
            ValueReference::Resolved(ValueId::BlockArgument { block, argument }) => self
                .block_argument_types(block)?
                .get(argument as usize)
                .copied(),
            ValueReference::Invalid(_) => None,
        }
    }
    pub(crate) fn value_type(&self, reference: ValueReference) -> Option<&str> {
        self.type_spelling(self.value_type_id(reference)?)
    }
    fn value_type_value(&self, reference: ValueReference) -> Option<&TypeValue> {
        self.type_value(self.value_type_id(reference)?)
    }
    pub(crate) fn value_spelling(&self, reference: ValueReference) -> Option<String> {
        let target = match reference {
            ValueReference::Resolved(value) => value,
            ValueReference::Invalid(_) => return None,
        };
        let mut number = 0;
        for operation in self.operations() {
            for result in 0..self.result_types(operation)?.len() {
                if target
                    == (ValueId::OperationResult {
                        operation,
                        result: result as u32,
                    })
                {
                    return Some(format!("%v{number}"));
                }
                number += 1;
            }
        }
        for operation in self.operations() {
            for region in self.operation_regions(operation)? {
                for block in self.region(*region)?.blocks(self)? {
                    for argument in 0..self.block_argument_types(*block)?.len() {
                        if target
                            == (ValueId::BlockArgument {
                                block: *block,
                                argument: argument as u32,
                            })
                        {
                            return Some(format!("%v{number}"));
                        }
                        number += 1;
                    }
                }
            }
        }
        None
    }
    pub(crate) fn typed_operands_spelling(&self, operation: OperationId) -> Option<String> {
        let operands = self.operands(operation)?;
        if operands.is_empty() {
            return Some(String::new());
        }
        let values = operands
            .iter()
            .map(|value| self.value_spelling(*value))
            .collect::<Option<Vec<_>>>()?
            .join(", ");
        let types = operands
            .iter()
            .map(|value| self.value_type(*value).map(str::to_owned))
            .collect::<Option<Vec<_>>>()?
            .join(", ");
        Some(format!(" {values} : {types}"))
    }
    pub(crate) fn successor_spelling(&self, operation: OperationId) -> Option<String> {
        self.successor_spelling_at(operation, 0)
    }
    pub(crate) fn successor_spelling_at(
        &self,
        operation: OperationId,
        index: usize,
    ) -> Option<String> {
        let successor = *self.successors(operation)?.get(index)?;
        let mut block_number = 0;
        let mut label = None;
        for owner in self.operations() {
            for region in self.operation_regions(owner)? {
                for block in self.region(*region)?.blocks(self)? {
                    if *block == successor.block {
                        label = Some(block_number);
                    }
                    block_number += 1;
                }
            }
        }
        let arguments = self.successor_arguments(successor)?;
        let suffix = if arguments.is_empty() {
            String::new()
        } else {
            let entries = arguments
                .iter()
                .map(|value| {
                    Some(format!(
                        "{} : {}",
                        self.value_spelling(*value)?,
                        self.value_type(*value)?
                    ))
                })
                .collect::<Option<Vec<_>>>()?
                .join(", ");
            format!("({entries})")
        };
        Some(format!("^bb{}{suffix}", label?))
    }
    pub fn attribute_value(&self, id: AttributeId) -> Option<&AttributeValue> {
        self.valid(id.index, id.generation, self.attributes.len())
            .then(|| &self.attributes[id.index()])
    }
    pub fn attribute_spelling_value(&self, id: AttributeId) -> Option<&str> {
        self.valid(id.index, id.generation, self.attributes.len())
            .then(|| self.attribute_spellings[id.index()].as_str())
    }
    pub fn affine_expression(&self, id: AffineExprId) -> Option<&AffineExprValue> {
        self.valid(id.index, id.generation, self.affine_expressions.len())
            .then(|| &self.affine_expressions[id.index()])
    }
    pub fn affine_map(&self, id: AffineMapId) -> Option<&AffineMapValue> {
        self.valid(id.index, id.generation, self.affine_maps.len())
            .then(|| &self.affine_maps[id.index()])
    }
    pub fn integer_set(&self, id: IntegerSetId) -> Option<&IntegerSetValue> {
        self.valid(id.index, id.generation, self.integer_sets.len())
            .then(|| &self.integer_sets[id.index()])
    }
    pub fn diagnostics(&self) -> &[SemanticDiagnostic] {
        &self.diagnostics
    }
    pub fn is_semantically_complete(&self) -> bool {
        self.complete
    }
    pub(crate) fn operation_attributes(&self, id: OperationId) -> Option<&[(u32, AttributeId)]> {
        self.operation(id)
            .and_then(|op| self.attribute_lists.get(op.attributes))
    }
    pub(crate) fn operation_properties(&self, id: OperationId) -> Option<&[(u32, AttributeId)]> {
        self.operation(id)
            .and_then(|op| self.attribute_lists.get(op.properties))
    }
    pub(crate) fn string(&self, id: u32) -> Option<&str> {
        self.strings.get(id as usize).map(String::as_str)
    }
    pub(crate) fn operation_location_id(&self, id: OperationId) -> Option<Option<LocationId>> {
        self.operation(id).map(|op| op.location)
    }
    pub(crate) fn location_value(&self, id: LocationId) -> Option<&LocationValue> {
        self.valid(id.index, id.generation, self.locations.len())
            .then(|| &self.locations[id.index()])
    }
    pub fn statistics(&self) -> DocumentStatistics {
        let (mut payload_blobs, mut payload_blob_bytes) =
            self.attributes
                .iter()
                .fold((0usize, 0usize), |(count, bytes), value| match value {
                    AttributeValue::Large(LargeAttributeValue::Dense(blob))
                    | AttributeValue::Large(LargeAttributeValue::Sparse(blob))
                    | AttributeValue::Large(LargeAttributeValue::Resource(blob))
                    | AttributeValue::WideNumber(blob)
                    | AttributeValue::Opaque(blob) => (count + 1, bytes + blob.len()),
                    _ => (count, bytes),
                });
        for value in &self.types {
            if let TypeValue::Opaque(blob) = value {
                payload_blobs += 1;
                payload_blob_bytes += blob.len();
            }
        }
        let analyses = self
            .analyses
            .0
            .read()
            .expect("analysis cache lock is not poisoned");
        let document_index_bytes = self.syntax_map.len()
            * std::mem::size_of::<(OperationId, TextRange)>()
            + self.blob_ranges.len() * std::mem::size_of::<TextRange>()
            + analyses.uses.as_ref().map_or(0, |index| {
                index
                    .uses
                    .values()
                    .map(|uses| uses.capacity() * std::mem::size_of::<UseSite>())
                    .sum()
            });
        let direct_owned_bytes = self.operations.capacity() * std::mem::size_of::<Operation>()
            + self
                .operations
                .iter()
                .filter_map(|operation| operation.unparsed_text.as_ref())
                .map(|text| text.len())
                .sum::<usize>()
            + self.operation_generations.capacity() * std::mem::size_of::<u32>()
            + self.operation_alive.capacity() * std::mem::size_of::<bool>()
            + self.regions.capacity() * std::mem::size_of::<Region>()
            + self.blocks.capacity() * std::mem::size_of::<Block>()
            + self.values.0.capacity() * std::mem::size_of::<ValueReference>()
            + self.types_lists.0.capacity() * std::mem::size_of::<TypeId>()
            + self.attribute_lists.0.capacity() * std::mem::size_of::<(u32, AttributeId)>()
            + self.successor_lists.0.capacity() * std::mem::size_of::<Successor>()
            + self.region_lists.0.capacity() * std::mem::size_of::<RegionId>()
            + self.block_lists.0.capacity() * std::mem::size_of::<BlockId>()
            + self.operation_lists.0.capacity() * std::mem::size_of::<OperationId>()
            + self.strings.capacity() * std::mem::size_of::<String>()
            + self.strings.iter().map(String::capacity).sum::<usize>()
            + self.types.capacity() * std::mem::size_of::<TypeValue>()
            + self.type_spellings.capacity() * std::mem::size_of::<String>()
            + self
                .type_spellings
                .iter()
                .map(String::capacity)
                .sum::<usize>()
            + self.attributes.capacity() * std::mem::size_of::<AttributeValue>()
            + self.attribute_spellings.capacity() * std::mem::size_of::<String>()
            + self
                .attribute_spellings
                .iter()
                .map(String::capacity)
                .sum::<usize>()
            + self.locations.capacity() * std::mem::size_of::<LocationValue>()
            + self.location_spellings.capacity() * std::mem::size_of::<String>()
            + self
                .location_spellings
                .iter()
                .map(String::capacity)
                .sum::<usize>()
            + self.affine_expressions.capacity() * std::mem::size_of::<AffineExprValue>()
            + self.affine_maps.capacity() * std::mem::size_of::<AffineMapValue>()
            + self.integer_sets.capacity() * std::mem::size_of::<IntegerSetValue>()
            + self.diagnostics.capacity() * std::mem::size_of::<SemanticDiagnostic>()
            + self.dirty_operations.capacity() * std::mem::size_of::<OperationId>()
            + self.dirty_blocks.capacity() * std::mem::size_of::<BlockId>();
        DocumentStatistics {
            operations: self.operations.len(),
            regions: self.regions.len(),
            blocks: self.blocks.len(),
            pooled_list_entries: self.values.0.len()
                + self.types_lists.0.len()
                + self.attribute_lists.0.len()
                + self.successor_lists.0.len()
                + self.region_lists.0.len()
                + self.block_lists.0.len()
                + self.operation_lists.0.len(),
            local_strings: self.strings.len(),
            local_types: self.types.len(),
            local_attributes: self.attributes.len(),
            affine_expressions: self.affine_expressions.len(),
            affine_maps: self.affine_maps.len(),
            integer_sets: self.integer_sets.len(),
            retained_source_bytes: self.retained_source.as_ref().map_or(0, |s| s.len()),
            retained_cst_nodes: self.retained_syntax.as_ref().map_or(0, |s| s.node_count()),
            retained_mapping_entries: self.syntax_map.len(),
            direct_owned_bytes,
            document_index_bytes,
            retained_cst_bytes: self
                .retained_syntax
                .as_ref()
                .map_or(0, |s| s.exact_retained_bytes()),
            source_storage_shared: self
                .retained_source
                .as_ref()
                .is_some_and(|s| Arc::strong_count(s) > 1),
            cst_storage_shared: self
                .retained_syntax
                .as_ref()
                .is_some_and(|s| Arc::strong_count(s) > 1),
            payload_blob_bytes,
            payload_blobs,
            use_index_entries: analyses
                .uses
                .as_ref()
                .map_or(0, |index| index.uses.values().map(Vec::len).sum()),
            symbol_index_entries: analyses
                .symbols
                .as_ref()
                .map_or(0, |index| index.scopes.values().map(HashMap::len).sum()),
            dominance_index_entries: analyses.dominance.as_ref().map_or(0, |index| {
                index.operation_positions.len()
                    + index
                        .regions
                        .values()
                        .map(RegionDominance::entry_count)
                        .sum::<usize>()
            }),
        }
    }
    pub fn retention_profile(&self) -> RetentionProfile {
        self.retention_profile
    }
    pub fn source_bytes(&self) -> Option<&[u8]> {
        self.retained_source.as_deref()
    }
    pub fn syntax_tree(&self) -> Option<&crate::representation::SyntaxTree> {
        self.retained_syntax.as_deref()
    }
    pub fn operation_syntax_range(&self, id: OperationId) -> Option<TextRange> {
        self.valid_operation(id)
            .then(|| {
                self.syntax_map
                    .binary_search_by_key(&id.index, |(op, _)| op.index)
                    .ok()
            })
            .flatten()
            .map(|index| self.syntax_map[index].1)
    }
    pub(crate) fn dirty_operations(&self) -> &HashSet<OperationId> {
        &self.dirty_operations
    }
    pub(crate) fn dirty_blocks(&self) -> &HashSet<BlockId> {
        &self.dirty_blocks
    }
    fn valid(&self, index: u32, generation: u128, len: usize) -> bool {
        generation == self.generation && (index as usize) < len
    }

    fn valid_operation(&self, id: OperationId) -> bool {
        id.owner == self.identity.0
            && self.operation_alive.get(id.index()).copied() == Some(true)
            && self.operation_generations.get(id.index()).copied() == Some(id.generation)
    }

    pub fn check_operation(&self, operation: OperationId) -> Result<(), EditError> {
        if operation.owner != self.identity.0 {
            return Err(EditError::ForeignOperation(operation));
        }
        if self.valid_operation(operation) {
            return Ok(());
        }
        if operation.generation == self.operation_generations.first().copied().unwrap_or(0)
            && operation.index() >= self.operations.len()
        {
            Err(EditError::InvalidOperation(operation))
        } else {
            Err(EditError::StaleOperation(operation))
        }
    }

    pub fn check_value(&self, value: ValueId) -> Result<(), EditError> {
        match value {
            ValueId::OperationResult { operation, result } => {
                self.check_operation(operation)
                    .map_err(|error| match error {
                        EditError::ForeignOperation(_) => EditError::ForeignValue(value),
                        EditError::StaleOperation(_) => EditError::StaleValue(value),
                        _ => EditError::InvalidValue(value),
                    })?;
                self.result_types(operation)
                    .is_some_and(|types| (result as usize) < types.len())
                    .then_some(())
                    .ok_or(EditError::InvalidValue(value))
            }
            ValueId::BlockArgument { block, argument } => {
                if block.generation != self.generation {
                    return Err(EditError::ForeignValue(value));
                }
                self.block_argument_types(block)
                    .is_some_and(|types| (argument as usize) < types.len())
                    .then_some(())
                    .ok_or(EditError::InvalidValue(value))
            }
        }
    }

    /// Returns a stable, document-local identity for a resolved value.
    pub fn value_key(&self, value: ValueId) -> Option<u128> {
        self.check_value(value).ok()?;
        Some(match value {
            ValueId::OperationResult { operation, result } => {
                ((operation.generation as u128) << 65)
                    | ((operation.index as u128) << 33)
                    | ((result as u128) << 1)
            }
            ValueId::BlockArgument { block, argument } => {
                ((block.index as u128) << 33) | ((argument as u128) << 1) | 1
            }
        })
    }

    pub fn checked_uses(&self, value: ValueId) -> Result<Vec<UseSite>, EditError> {
        self.check_value(value)?;
        Ok(self.uses(value))
    }

    pub fn checked_lookup_symbol(
        &self,
        from: OperationId,
        symbol: &str,
        registry: &DialectRegistry,
    ) -> Result<Option<OperationId>, EditError> {
        self.check_operation(from)?;
        Ok(self.lookup_symbol(from, symbol, registry))
    }

    pub fn checked_dominates(
        &self,
        value: ValueId,
        operation: OperationId,
        registry: &DialectRegistry,
    ) -> Result<bool, EditError> {
        self.check_value(value)?;
        self.check_operation(operation)?;
        Ok(self.dominates(value, operation, registry))
    }

    /// Monotonically changes after each committed semantic transaction.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns all operand and successor-argument uses of a value. The backing
    /// index is built on the first query and is not retained by parse/lower-only clients.
    pub fn uses(&self, value: ValueId) -> Vec<UseSite> {
        self.ensure_use_index();
        self.analyses
            .borrow()
            .uses
            .as_ref()
            .and_then(|index| index.uses.get(&value))
            .cloned()
            .unwrap_or_default()
    }

    /// Looks up a symbol from an operation through registered enclosing symbol
    /// tables. Unregistered operations and attributes never enter this index.
    pub fn lookup_symbol(
        &self,
        from: OperationId,
        symbol: &str,
        registry: &DialectRegistry,
    ) -> Option<OperationId> {
        let _query = self
            .analyses
            .1
            .lock()
            .expect("registry query lock is not poisoned");
        self.ensure_symbol_index(registry);
        let path = symbol
            .split("::")
            .map(normalize_symbol)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        let caches = self
            .analyses
            .0
            .read()
            .expect("analysis cache lock is not poisoned");
        let index = caches.symbols.as_ref()?;
        self.lookup_symbol_in_index(from, &path, index)
    }

    fn lookup_symbol_in_index(
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

    /// Reports unresolved references found only on operations registered as
    /// symbol users.
    pub fn symbol_index_diagnostics(
        &self,
        registry: &DialectRegistry,
    ) -> Vec<SymbolIndexDiagnostic> {
        let _query = self
            .analyses
            .1
            .lock()
            .expect("registry query lock is not poisoned");
        self.ensure_symbol_index(registry);
        self.analyses
            .borrow()
            .symbols
            .as_ref()
            .map(|index| index.diagnostics.clone())
            .unwrap_or_default()
    }

    /// Answers whether a value is visible and dominates an operation according
    /// to registered region metadata and the verifier's conservative CFG rules.
    pub fn dominates(
        &self,
        value: ValueId,
        use_operation: OperationId,
        registry: &DialectRegistry,
    ) -> bool {
        let _query = self
            .analyses
            .1
            .lock()
            .expect("registry query lock is not poisoned");
        let valid_value = match value {
            ValueId::OperationResult { operation, result } => self
                .result_types(operation)
                .is_some_and(|types| (result as usize) < types.len()),
            ValueId::BlockArgument { block, argument } => self
                .block_argument_types(block)
                .is_some_and(|types| (argument as usize) < types.len()),
        };
        if !valid_value {
            return false;
        }
        self.ensure_dominance_index(registry);
        let Some(use_block) = self
            .operation(use_operation)
            .and_then(Operation::parent_block)
        else {
            return false;
        };
        let use_position = self
            .analyses
            .borrow()
            .dominance
            .as_ref()
            .and_then(|index| index.operation_positions.get(&use_operation).copied());
        let Some(use_position) = use_position else {
            return false;
        };
        let analyses = self.analyses.borrow();
        let Some(index) = analyses.dominance.as_ref() else {
            return false;
        };
        self.value_visible_at(
            value,
            ValueUsePoint {
                operation: use_operation,
                block: use_block,
                position: use_position,
            },
            registry,
            VisibilityAnalysis::Indexed(index),
        )
    }

    fn ensure_use_index(&self) {
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

    fn ensure_symbol_index(&self, registry: &DialectRegistry) {
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

    fn ensure_dominance_index(&self, registry: &DialectRegistry) {
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
                if tree.kind(node) == Some(crate::SyntaxKind::Operation) {
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

    /// Compatibility alias for callers predating the explicit structural name.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.validate_structure()
    }

    /// Runs structural checks followed by registered operation schemas and verifiers.
    pub fn verify_semantics(
        &self,
        registry: &DialectRegistry,
    ) -> Result<(), SemanticVerificationError> {
        self.validate_structure()
            .map_err(SemanticVerificationError::Structural)?;
        self.verify_semantics_only(registry)
    }

    fn verify_semantics_only(
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
        self.verify_registered_structure(registry)?;
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

    fn attribute_spelling(&self, operation: OperationId, name: &str) -> Option<&str> {
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

    fn direct_operations_in_operation_regions(&self, operation: OperationId) -> Vec<OperationId> {
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

    fn value_visible_at(
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

    fn enclosing_operation(&self, operation: OperationId) -> Option<OperationId> {
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

fn normalize_symbol(symbol: &str) -> &str {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticVerificationError {
    Structural(ValidationError),
    InvalidSentinel,
    Schema {
        operation: OperationId,
        message: &'static str,
    },
    Operation {
        operation: OperationId,
        message: &'static str,
    },
    Type {
        spelling: String,
        message: &'static str,
    },
    Attribute {
        spelling: String,
        message: &'static str,
    },
}

impl fmt::Display for SemanticVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural(error) => write!(f, "structural verification failed: {error}"),
            Self::InvalidSentinel => f.write_str("semantic document contains invalid values"),
            Self::Schema { operation, message } => {
                write!(f, "operation {operation:?} violates its schema: {message}")
            }
            Self::Operation { operation, message } => {
                write!(f, "operation {operation:?} failed verification: {message}")
            }
            Self::Type { spelling, message } => {
                write!(f, "type `{spelling}` failed verification: {message}")
            }
            Self::Attribute { spelling, message } => {
                write!(f, "attribute `{spelling}` failed verification: {message}")
            }
        }
    }
}

impl std::error::Error for SemanticVerificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structural(error) => Some(error),
            _ => None,
        }
    }
}

/// Typed registered view that coexists with the generic `Document` API.
pub struct ArithConstantOp<'a> {
    document: &'a Document,
    operation: OperationId,
}

pub struct BuiltinModuleOp<'a> {
    document: &'a Document,
    operation: OperationId,
}
impl<'a> BuiltinModuleOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "builtin.module").then_some(Self {
            document,
            operation,
        })
    }
    pub fn symbol(&self) -> Option<&'a str> {
        self.document.attribute_spelling(self.operation, "sym_name")
    }
    pub fn region(&self) -> Option<RegionId> {
        self.document
            .operation_regions(self.operation)?
            .first()
            .copied()
    }
}

pub struct FuncFuncOp<'a> {
    document: &'a Document,
    operation: OperationId,
}
impl<'a> FuncFuncOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "func.func").then_some(Self {
            document,
            operation,
        })
    }
    pub fn symbol(&self) -> Option<&'a str> {
        self.document.attribute_spelling(self.operation, "sym_name")
    }
    pub fn signature(&self) -> Option<&'a AttributeValue> {
        self.document
            .attribute_id(self.operation, "function_type")
            .and_then(|id| self.document.attribute_value(id))
    }
}

pub struct FuncCallOp<'a> {
    document: &'a Document,
    operation: OperationId,
}
impl<'a> FuncCallOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "func.call").then_some(Self {
            document,
            operation,
        })
    }
    pub fn callee(&self) -> Option<&'a str> {
        self.document.attribute_spelling(self.operation, "callee")
    }
    pub fn operands(&self) -> Option<&'a [ValueReference]> {
        self.document.operands(self.operation)
    }
}

pub struct CfCondBrOp<'a> {
    document: &'a Document,
    operation: OperationId,
}
impl<'a> CfCondBrOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "cf.cond_br").then_some(Self {
            document,
            operation,
        })
    }
    pub fn condition(&self) -> Option<ValueReference> {
        self.document.operands(self.operation)?.first().copied()
    }
    pub fn successors(&self) -> Option<&'a [Successor]> {
        self.document.successors(self.operation)
    }
}

impl<'a> ArithConstantOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "arith.constant").then_some(Self {
            document,
            operation,
        })
    }

    pub fn operation(&self) -> OperationId {
        self.operation
    }

    pub fn value(&self) -> Option<&'a AttributeValue> {
        self.document
            .attribute_id(self.operation, "value")
            .and_then(|attribute| self.document.attribute_value(attribute))
    }
}

pub struct ArithAddiOp<'a> {
    document: &'a Document,
    operation: OperationId,
}
impl<'a> ArithAddiOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "arith.addi").then_some(Self {
            document,
            operation,
        })
    }
    pub fn operands(&self) -> Option<&'a [ValueReference]> {
        self.document.operands(self.operation)
    }
    pub fn result_type(&self) -> Option<&'a TypeValue> {
        self.document
            .result_types(self.operation)?
            .first()
            .and_then(|ty| self.document.type_value(*ty))
    }
}

pub struct FuncReturnOp<'a> {
    document: &'a Document,
    operation: OperationId,
}
impl<'a> FuncReturnOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "func.return").then_some(Self {
            document,
            operation,
        })
    }
    pub fn operands(&self) -> Option<&'a [ValueReference]> {
        self.document.operands(self.operation)
    }
}

pub struct CfBrOp<'a> {
    document: &'a Document,
    operation: OperationId,
}
impl<'a> CfBrOp<'a> {
    pub fn cast(document: &'a Document, operation: OperationId) -> Option<Self> {
        registered_schema_matches(document, operation, "cf.br").then_some(Self {
            document,
            operation,
        })
    }
    pub fn successor(&self) -> Option<Successor> {
        self.document.successors(self.operation)?.first().copied()
    }
}

fn registered_schema_matches(document: &Document, operation: OperationId, name: &str) -> bool {
    let Some(descriptor) = DialectRegistry::proving().operation(name) else {
        return false;
    };
    document.operation_name(operation) == Some(descriptor.name)
        && document
            .operands(operation)
            .is_some_and(|operands| descriptor.schema.operands.accepts(operands.len()))
        && document
            .result_types(operation)
            .is_some_and(|results| descriptor.schema.results.accepts(results.len()))
        && descriptor
            .schema
            .required_attributes
            .iter()
            .all(|name| document.attribute_id(operation, name).is_some())
}

impl Operation {
    pub fn parent_block(&self) -> Option<BlockId> {
        self.parent
    }
    pub fn result(&self, operation: OperationId, number: u32) -> Option<ValueId> {
        (operation == self.id && number < self.result_types.len).then_some(
            ValueId::OperationResult {
                operation,
                result: number,
            },
        )
    }
}
impl Region {
    pub fn parent_operation(&self) -> OperationId {
        self.parent
    }
    pub fn blocks<'a>(&self, document: &'a Document) -> Option<&'a [BlockId]> {
        if self.generation != document.generation {
            return None;
        }
        document.block_lists.get(self.blocks)
    }
}
impl Block {
    pub fn parent_region(&self) -> RegionId {
        self.parent
    }
}

impl Successor {
    pub fn block(&self) -> BlockId {
        self.block
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeSpec {
    pub spelling: String,
    pub value: TypeValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributeSpec {
    pub name: String,
    pub spelling: String,
    pub value: AttributeValue,
}

/// Arena-independent description of a regionless operation to insert.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationSpec {
    pub name: String,
    pub operands: Vec<ValueId>,
    pub result_types: Vec<TypeSpec>,
    pub function_type: TypeSpec,
    pub attributes: Vec<AttributeSpec>,
    pub properties: Vec<AttributeSpec>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertionPoint {
    Root(usize),
    Block { block: BlockId, index: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditError {
    IncompleteDocument,
    StaleOperation(OperationId),
    InvalidOperation(OperationId),
    ForeignOperation(OperationId),
    StaleBlock(BlockId),
    ForeignBlock(BlockId),
    StaleValue(ValueId),
    InvalidValue(ValueId),
    ForeignValue(ValueId),
    InvalidPosition,
    InvalidOperandIndex,
    InvalidSuccessorIndex,
    InvalidSuccessorArgumentIndex,
    ResultCountChange,
    TypeMismatch,
    OwnedRegionsUnsupported,
    LiveUses(OperationId),
    Structural(ValidationError),
    Semantic(SemanticVerificationError),
}
impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignOperation(id) => {
                write!(f, "operation handle {id:?} belongs to another document")
            }
            Self::ForeignBlock(id) => {
                write!(f, "block handle {id:?} belongs to another document")
            }
            Self::ForeignValue(id) => {
                write!(f, "value handle {id:?} belongs to another document")
            }
            Self::StaleOperation(id) => write!(f, "operation handle {id:?} is stale"),
            Self::InvalidOperation(id) => write!(f, "operation handle {id:?} is invalid"),
            Self::StaleBlock(id) => write!(f, "block handle {id:?} is stale"),
            Self::StaleValue(id) => write!(f, "value handle {id:?} is stale"),
            Self::InvalidValue(id) => write!(f, "value handle {id:?} is invalid"),
            Self::Structural(error) => {
                write!(f, "edited document is structurally invalid: {error}")
            }
            Self::Semantic(error) => write!(f, "edited document is semantically invalid: {error}"),
            Self::IncompleteDocument => f.write_str("cannot edit an incomplete document"),
            Self::InvalidPosition => f.write_str("edit position is invalid"),
            Self::InvalidOperandIndex => f.write_str("operand index is invalid"),
            Self::InvalidSuccessorIndex => f.write_str("successor index is invalid"),
            Self::InvalidSuccessorArgumentIndex => {
                f.write_str("successor argument index is invalid")
            }
            Self::ResultCountChange => {
                f.write_str("editing cannot change an operation's result count")
            }
            Self::TypeMismatch => f.write_str("edited value has an incompatible type"),
            Self::OwnedRegionsUnsupported => {
                f.write_str("editing operations with owned regions is unsupported")
            }
            Self::LiveUses(id) => write!(f, "operation {id:?} still has live uses"),
        }
    }
}
impl std::error::Error for EditError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structural(error) => Some(error),
            Self::Semantic(error) => Some(error),
            _ => None,
        }
    }
}

/// A private-copy transaction. Dropping it or returning an error never changes the document.
pub struct DocumentEditor<'a> {
    original: &'a mut Document,
    working: Document,
    registry: &'a DialectRegistry,
}

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
    fn intern_type_spec(&mut self, spec: &TypeSpec) -> TypeId {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    InvalidOperationStorage,
    InvalidList,
    StaleOperation(OperationId),
    StaleBlock(BlockId),
    StaleRegion(RegionId),
    StaleType(TypeId),
    StaleAttribute(AttributeId),
    InvalidString,
    ParentChildMismatch,
    InvalidValue,
    InvalidSuccessor,
    InvalidLocation,
    InvalidSentinel,
    InvalidRetention,
}
impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = match self {
            Self::InvalidOperationStorage => "operation storage is inconsistent",
            Self::InvalidList => "an arena list reference is invalid",
            Self::StaleOperation(_) => "an operation reference is stale",
            Self::StaleBlock(_) => "a block reference is stale",
            Self::StaleRegion(_) => "a region reference is stale",
            Self::StaleType(_) => "a type reference is stale",
            Self::StaleAttribute(_) => "an attribute reference is stale",
            Self::InvalidString => "a string reference is invalid",
            Self::ParentChildMismatch => "parent and child references disagree",
            Self::InvalidValue => "a value reference is invalid",
            Self::InvalidSuccessor => "a successor reference is invalid",
            Self::InvalidLocation => "a location reference is invalid",
            Self::InvalidSentinel => "an invalid sentinel appears in a complete document",
            Self::InvalidRetention => "retained source, syntax, or mappings are inconsistent",
        };
        write!(f, "invalid semantic document: {detail}")
    }
}
impl std::error::Error for ValidationError {}

struct Interner<T = String> {
    values: Vec<T>,
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
    fn intern(&mut self, value: &str) -> u32 {
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
    fn intern_value(&mut self, value: T) -> u32 {
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

pub fn lower_proving_fixture(
    file: &ParsedFile,
    mode: LoweringMode,
    _registry: &SharedRegistry,
) -> LoweringResult {
    lower_proving_fixture_with_retention(file, mode, RetentionProfile::SemanticOnly, _registry)
}

pub fn lower_proving_fixture_with_retention(
    file: &ParsedFile,
    mode: LoweringMode,
    retention_profile: RetentionProfile,
    _registry: &SharedRegistry,
) -> LoweringResult {
    lower_with_registry(file, mode, retention_profile, &DialectRegistry::EMPTY)
}

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
            let operation_spelling = if op.results().next().is_some() {
                spelling.split_once('=')?.1
            } else {
                spelling
            };
            let mnemonic = operation_spelling
                .split_ascii_whitespace()
                .next()?
                .trim_matches('"');
            if let Some(shape) = registry.operation_shape(mnemonic) {
                return lower_operation_shape(
                    shape,
                    mnemonic,
                    &RegisteredLoweringContext { spelling },
                )
                .map(|lowering| MatchedLowering {
                    name: mnemonic.to_owned(),
                    shape: Some(shape),
                    lowering,
                });
            }
            let descriptor = registry.custom_operation(mnemonic)?;
            descriptor
                .assembly
                .and_then(|program| program.lower(&RegisteredLoweringContext { spelling }))
                .or_else(|| {
                    descriptor
                        .lower
                        .and_then(|lower| lower(&RegisteredLoweringContext { spelling }))
                })
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
        let name = registered[i]
            .as_ref()
            .map(|matched| matched.name.as_str())
            .or_else(|| operation_name(source.bytes(), range))
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

    let mut labels_by_region = HashMap::<RegionId, HashMap<String, BlockId>>::new();
    for (i, block) in blocks.iter().enumerate() {
        let parent = block_regions[&block_ids[&block.id()]];
        let label = block.label().and_then(|label| {
            let name = first_identifier(
                text(source.bytes(), syntax.tree().text_range(label).unwrap()),
                b'^',
            )?;
            labels_by_region
                .entry(parent)
                .or_default()
                .insert(name.clone(), BlockId::new(i, generation));
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
                let range = successor.tree().text_range(successor.id()).unwrap();
                let spelling = text(source.bytes(), range);
                let label = first_identifier(spelling, b'^').unwrap_or_default();
                let block = parent_region
                    .and_then(|region| labels_by_region.get(&region))
                    .and_then(|labels| labels.get(&label))
                    .copied();
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
                        let range = argument.tree().text_range(argument.id()).unwrap();
                        resolve_value(
                            text(source.bytes(), range),
                            range,
                            parent_region,
                            doc.operations[i].parent,
                            &region_definitions,
                            &block_definitions,
                            &region_outer,
                            &mut doc,
                        )
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

fn text(bytes: &[u8], range: TextRange) -> &str {
    std::str::from_utf8(&bytes[range.start() as usize..range.end() as usize]).unwrap_or("")
}

fn first_identifier(spelling: &str, sigil: u8) -> Option<String> {
    let bytes = spelling.as_bytes();
    let start = bytes.iter().position(|byte| *byte == sigil)?;
    let end = bytes[start + 1..]
        .iter()
        .position(|byte| byte.is_ascii_whitespace() || b"#: ,()={}[]".contains(byte))
        .map_or(bytes.len(), |end| start + 1 + end);
    Some(spelling[start + 1..end].to_owned())
}

fn argument_type(spelling: &str) -> &str {
    let Some((_, tail)) = spelling.split_once(':') else {
        return "<invalid>";
    };
    tail.trim().split("loc(").next().unwrap_or(tail).trim()
}

fn operation_output_types(op: crate::parser::OperationSyntax<'_>, bytes: &[u8]) -> Vec<String> {
    let function = op
        .tree()
        .children(op.id())
        .into_iter()
        .flatten()
        .find(|child| op.tree().kind(*child) == Some(SyntaxKind::FunctionType));
    let Some(range) = function.and_then(|node| op.tree().text_range(node)) else {
        return Vec::new();
    };
    let spelling = text(bytes, range);
    let output = spelling
        .split_once("->")
        .map(|(_, output)| output.trim())
        .unwrap_or("");
    split_types(output)
}

fn split_types(spelling: &str) -> Vec<String> {
    let spelling = spelling.trim();
    let inner = spelling
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .unwrap_or(spelling);
    if inner.trim().is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in inner.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        if byte == b'"' {
            quoted = true;
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'>' if index == 0 || inner.as_bytes()[index - 1] != b'-' => depth -= 1,
            b',' if depth == 0 => {
                result.push(inner[start..index].trim().to_owned());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(inner[start..].trim().to_owned());
    result
}

pub(crate) fn split_registered_types(spelling: &str) -> Vec<String> {
    split_types(spelling)
}

// These parameters keep type interning explicit at the lowering boundary instead of
// hiding mutable lowering state in a broader context object.
#[allow(clippy::too_many_arguments)]
fn intern_type(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    types: &mut Interner<TypeValue>,
    spellings: &mut Vec<String>,
    generation: u128,
    doc: &mut Document,
) -> TypeId {
    let value = lower_type_value(spelling, range, type_aliases, attribute_aliases, doc);
    let index = types.intern_value(value);
    if index as usize == spellings.len() {
        spellings.push(spelling.trim().to_owned());
    }
    TypeId::new(index as usize, generation)
}

fn lower_type_value(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    doc: &mut Document,
) -> TypeValue {
    let mut expansion = AliasExpansionState::new(doc.alias_expansion_depth_limit);
    lower_type_value_with_stack(
        spelling,
        range,
        type_aliases,
        attribute_aliases,
        &mut expansion,
        doc,
    )
}

fn lower_type_value_with_stack(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    alias_stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> TypeValue {
    let spelling = spelling.trim();
    if let Some((target, _)) = type_aliases.get(spelling) {
        if target != spelling {
            if let Err(message) = alias_stack.enter(spelling, "type") {
                return TypeValue::Invalid(push_diagnostic(doc, range, message));
            }
            let value = lower_type_value_with_stack(
                target,
                range,
                type_aliases,
                attribute_aliases,
                alias_stack,
                doc,
            );
            alias_stack.exit(spelling);
            return value;
        }
    }
    if !is_composite_type(spelling) {
        if let Ok(value) = resolve_type(spelling, type_aliases, attribute_aliases, alias_stack) {
            return value;
        }
    }
    if let Some((inputs, results)) = split_arrow(spelling) {
        return TypeValue::Function {
            inputs: split_types(inputs)
                .iter()
                .map(|value| {
                    lower_type_value_with_stack(
                        value,
                        range,
                        type_aliases,
                        attribute_aliases,
                        alias_stack,
                        doc,
                    )
                })
                .collect(),
            results: split_types(results)
                .iter()
                .map(|value| {
                    lower_type_value_with_stack(
                        value,
                        range,
                        type_aliases,
                        attribute_aliases,
                        alias_stack,
                        doc,
                    )
                })
                .collect(),
        };
    }
    if let Some(inner) = angle_inner(spelling, "tuple") {
        return TypeValue::Tuple(
            split_types(inner)
                .iter()
                .map(|value| {
                    lower_type_value_with_stack(
                        value,
                        range,
                        type_aliases,
                        attribute_aliases,
                        alias_stack,
                        doc,
                    )
                })
                .collect(),
        );
    }
    for (prefix, constructor) in [("tensor", 0u8), ("vector", 1), ("memref", 2)] {
        if let Some(inner) = angle_inner(spelling, prefix) {
            let parts = split_top_level_commas(inner);
            let shape = parts.first().copied().unwrap_or("");
            let shape_parts = split_top_level_x(shape);
            if let Some(element) = shape_parts.last() {
                let dimensions = shape_parts[..shape_parts.len().saturating_sub(1)]
                    .iter()
                    .map(|dimension| {
                        let scalable = prefix == "vector" && dimension.starts_with('[');
                        let (size, invalid) = match *dimension {
                            "?" | "*" => (None, None),
                            _ => match dimension.trim_matches(&['[', ']'][..]).parse() {
                                Ok(size) => (Some(size), None),
                                Err(_) => (
                                    None,
                                    Some(push_diagnostic(
                                        doc,
                                        range,
                                        format!("invalid {prefix} dimension `{dimension}`"),
                                    )),
                                ),
                            },
                        };
                        ShapedDimension {
                            size,
                            scalable,
                            invalid,
                        }
                    })
                    .collect::<Vec<_>>();
                let element = Box::new(lower_type_value_with_stack(
                    element,
                    range,
                    type_aliases,
                    attribute_aliases,
                    alias_stack,
                    doc,
                ));
                return match constructor {
                    0 => TypeValue::Tensor {
                        dimensions,
                        element,
                        encoding: parts.get(1).map(|value| {
                            Box::new(lower_attribute_value(
                                value,
                                range,
                                type_aliases,
                                attribute_aliases,
                                alias_stack,
                                doc,
                            ))
                        }),
                        unranked: shape_parts.first().copied() == Some("*"),
                    },
                    1 => TypeValue::Vector {
                        scalable: shape_parts[..shape_parts.len().saturating_sub(1)]
                            .iter()
                            .map(|dimension| dimension.starts_with('['))
                            .collect(),
                        dimensions,
                        element,
                    },
                    _ => TypeValue::MemRef {
                        dimensions,
                        element,
                        layout: parts.get(1).map(|value| {
                            lower_memref_layout(
                                value,
                                range,
                                type_aliases,
                                attribute_aliases,
                                alias_stack,
                                doc,
                            )
                        }),
                        memory_space: parts.get(2).map(|value| {
                            Box::new(lower_memref_memory_space(
                                value,
                                range,
                                type_aliases,
                                attribute_aliases,
                                alias_stack,
                                doc,
                            ))
                        }),
                    },
                };
            }
        }
    }
    let message = match resolve_type(spelling, type_aliases, attribute_aliases, alias_stack) {
        Err(message) => message,
        Ok(_) => format!("unsupported or malformed type `{spelling}`"),
    };
    TypeValue::Invalid(push_diagnostic(doc, range, message))
}

fn lower_memref_layout(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
    doc: &mut Document,
) -> MemRefLayout {
    if spelling.trim().starts_with("affine_map<") {
        return match lower_affine_attribute(spelling.trim(), range, doc) {
            AttributeValue::AffineMap(map) => MemRefLayout::AffineMap(map),
            AttributeValue::Invalid(diagnostic) => MemRefLayout::Invalid(diagnostic),
            _ => MemRefLayout::Invalid(push_diagnostic(
                doc,
                range,
                "memref affine layout has wrong kind".into(),
            )),
        };
    }
    if let Some(affine_spelling) =
        match resolve_affine_alias(spelling.trim(), attribute_aliases, expansion) {
            Ok(value) => value,
            Err(message) => return MemRefLayout::Invalid(push_diagnostic(doc, range, message)),
        }
    {
        return match lower_affine_attribute(affine_spelling, range, doc) {
            AttributeValue::AffineMap(map) => MemRefLayout::AffineMap(map),
            AttributeValue::Invalid(diagnostic) => MemRefLayout::Invalid(diagnostic),
            AttributeValue::IntegerSet(_) => MemRefLayout::Invalid(push_diagnostic(
                doc,
                range,
                "integer set has wrong kind for memref affine layout".into(),
            )),
            AttributeValue::Opaque(_)
            | AttributeValue::Large(_)
            | AttributeValue::WideNumber(_)
            | AttributeValue::Type(_)
            | AttributeValue::Boolean(_)
            | AttributeValue::Integer(_)
            | AttributeValue::Float(_)
            | AttributeValue::String(_)
            | AttributeValue::Symbol(_)
            | AttributeValue::Array(_)
            | AttributeValue::Dictionary(_)
            | AttributeValue::Location(_) => MemRefLayout::Invalid(push_diagnostic(
                doc,
                range,
                "affine alias has wrong kind for memref affine layout".into(),
            )),
        };
    }
    match resolve_memref_layout(spelling, type_aliases, attribute_aliases, expansion) {
        Ok(MemRefLayout::Opaque { spelling, .. }) => {
            let parameters = lower_memref_alias_parameters(
                &spelling,
                range,
                type_aliases,
                attribute_aliases,
                expansion,
                doc,
            );
            MemRefLayout::Opaque {
                spelling,
                parameters,
            }
        }
        Ok(layout) => layout,
        Err(message) => {
            if spelling.trim().starts_with("strided<") || spelling.trim().starts_with("affine_map<")
            {
                MemRefLayout::Opaque {
                    spelling: compact(spelling),
                    parameters: lower_memref_alias_parameters(
                        spelling,
                        range,
                        type_aliases,
                        attribute_aliases,
                        expansion,
                        doc,
                    ),
                }
            } else {
                MemRefLayout::Invalid(push_diagnostic(doc, range, message))
            }
        }
    }
}

fn resolve_affine_alias<'a>(
    spelling: &'a str,
    attribute_aliases: &'a HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<Option<&'a str>, String> {
    let spelling = spelling.trim();
    if !spelling.starts_with('#') || spelling.contains('<') {
        return Ok(None);
    }
    let Some((target, _)) = attribute_aliases.get(spelling) else {
        return Ok(None);
    };
    stack.enter(spelling, "attribute")?;
    let result = if target.trim().starts_with('!') {
        Err(format!(
            "alias `{spelling}` has type kind, expected attribute"
        ))
    } else if target.trim().starts_with("affine_map<") || target.trim().starts_with("affine_set<") {
        Ok(Some(target.trim()))
    } else if target.trim().starts_with('#') {
        resolve_affine_alias(target, attribute_aliases, stack)
    } else {
        Ok(None)
    };
    stack.exit(spelling);
    result
}

fn resolve_memref_layout(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
) -> Result<MemRefLayout, String> {
    let spelling = spelling.trim();
    if spelling.starts_with("strided<") || spelling.starts_with("affine_map<") {
        if let Some(message) =
            first_invalid_memref_alias(spelling, type_aliases, attribute_aliases, expansion)
        {
            return Err(message);
        }
        return Ok(MemRefLayout::Opaque {
            spelling: compact(spelling),
            parameters: Vec::new(),
        });
    }
    resolve_attribute(spelling, type_aliases, attribute_aliases, expansion)
        .and_then(|value| match value {
            AttributeValue::Type(_) => Err("type value has wrong kind".into()),
            value => Ok(MemRefLayout::Attribute(Box::new(value))),
        })
        .map_err(|message| format!("invalid memref layout `{spelling}`: {message}"))
}

fn is_composite_type(spelling: &str) -> bool {
    let spelling = spelling.trim();
    spelling.contains("->")
        || ["tuple<", "tensor<", "vector<", "memref<"]
            .iter()
            .any(|prefix| spelling.starts_with(prefix))
}

fn lower_memref_alias_parameters(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
    doc: &mut Document,
) -> Vec<AttributeValue> {
    alias_spellings(spelling)
        .into_iter()
        .map(|alias| {
            let value = if alias.starts_with('!') {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    range,
                    format!("memref layout alias `{alias}` has type kind, expected attribute"),
                ))
            } else {
                lower_attribute_value(
                    &alias,
                    range,
                    type_aliases,
                    attribute_aliases,
                    expansion,
                    doc,
                )
            };
            if matches!(value, AttributeValue::Type(_)) {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    range,
                    format!("memref layout alias `{alias}` has type kind, expected attribute"),
                ))
            } else {
                value
            }
        })
        .collect()
}

fn first_invalid_memref_alias(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
) -> Option<String> {
    alias_spellings(spelling).into_iter().find_map(|alias| {
        if alias.starts_with('!') {
            return Some(format!(
                "alias `{alias}` has type kind, expected memref layout"
            ));
        }
        match resolve_attribute(&alias, type_aliases, attribute_aliases, expansion) {
            Ok(AttributeValue::Type(_)) => Some(format!(
                "memref layout alias `{alias}` has type kind, expected attribute"
            )),
            Ok(_) => None,
            Err(message) => Some(format!(
                "invalid memref layout parameter `{alias}`: {message}"
            )),
        }
    })
}

fn alias_spellings(spelling: &str) -> Vec<String> {
    let bytes = spelling.as_bytes();
    let mut aliases = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'#' || bytes[index] == b'!' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric()
                    || matches!(bytes[index], b'_' | b'$' | b'.' | b'-'))
            {
                index += 1;
            }
            aliases.push(spelling[start..index].to_owned());
        } else {
            index += 1;
        }
    }
    aliases
}

fn lower_memref_memory_space(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
    doc: &mut Document,
) -> AttributeValue {
    if spelling.trim().starts_with('!') {
        return AttributeValue::Invalid(push_diagnostic(
            doc,
            range,
            format!(
                "memref memory space `{}` has type kind, expected attribute",
                spelling.trim()
            ),
        ));
    }
    let value = lower_attribute_value(
        spelling,
        range,
        type_aliases,
        attribute_aliases,
        expansion,
        doc,
    );
    if matches!(value, AttributeValue::Type(_)) {
        AttributeValue::Invalid(push_diagnostic(
            doc,
            range,
            format!(
                "memref memory space `{}` has type kind, expected attribute",
                spelling.trim()
            ),
        ))
    } else {
        value
    }
}

fn resolve_memref_memory_space(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    expansion: &mut AliasExpansionState,
) -> Result<AttributeValue, String> {
    if spelling.trim().starts_with('!') {
        return Err(format!(
            "memref memory space `{}` has type kind, expected attribute",
            spelling.trim()
        ));
    }
    resolve_attribute(spelling, type_aliases, attribute_aliases, expansion).and_then(|value| {
        match value {
            AttributeValue::Type(_) => Err(format!(
                "memref memory space `{}` has type kind, expected attribute",
                spelling.trim()
            )),
            value => Ok(value),
        }
    })
}

fn resolve_type(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<TypeValue, String> {
    let spelling = spelling.trim();
    if spelling.starts_with('!') && !spelling.contains('<') {
        let other = format!("#{}", &spelling[1..]);
        if attribute_aliases.contains_key(&other) {
            return Err(format!(
                "alias `{spelling}` has attribute kind, expected type"
            ));
        }
        let Some((target, _)) = type_aliases.get(spelling) else {
            return Err(format!("unresolved type alias `{spelling}`"));
        };
        stack.enter(spelling, "type")?;
        let result = resolve_type(target, type_aliases, attribute_aliases, stack);
        stack.exit(spelling);
        return result;
    }
    if spelling == "index" {
        return Ok(TypeValue::Index);
    }
    if let Some(width) = spelling.strip_prefix('i').and_then(parse_width) {
        return Ok(TypeValue::Integer {
            width,
            signedness: None,
        });
    }
    if let Some(width) = spelling.strip_prefix("si").and_then(parse_width) {
        return Ok(TypeValue::Integer {
            width,
            signedness: Some(true),
        });
    }
    if let Some(width) = spelling.strip_prefix("ui").and_then(parse_width) {
        return Ok(TypeValue::Integer {
            width,
            signedness: Some(false),
        });
    }
    if is_float_spelling(spelling) {
        return Ok(TypeValue::Float(spelling.to_owned()));
    }
    if let Some((inputs, results)) = split_arrow(spelling) {
        return Ok(TypeValue::Function {
            inputs: split_types(inputs)
                .iter()
                .map(|s| resolve_type(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
            results: split_types(results)
                .iter()
                .map(|s| resolve_type(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
        });
    }
    if let Some(inner) = angle_inner(spelling, "tuple") {
        return Ok(TypeValue::Tuple(
            split_types(inner)
                .iter()
                .map(|s| resolve_type(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
        ));
    }
    for (prefix, constructor) in [("tensor", 0u8), ("vector", 1), ("memref", 2)] {
        if let Some(inner) = angle_inner(spelling, prefix) {
            let parts = split_top_level_commas(inner);
            let shape = parts.first().copied().unwrap_or("");
            let shape_parts = split_top_level_x(shape);
            let Some(element) = shape_parts.last() else {
                return Err(format!("malformed {prefix} type"));
            };
            let unranked = prefix == "tensor" && shape_parts.first().copied() == Some("*");
            let dimensions = shape_parts[..shape_parts.len() - 1]
                .iter()
                .map(|d| {
                    let scalable = prefix == "vector" && d.starts_with('[');
                    let size = if *d == "?" || *d == "*" {
                        None
                    } else {
                        Some(
                            d.trim_matches(&['[', ']'][..])
                                .parse::<u64>()
                                .map_err(|_| format!("invalid {prefix} dimension `{d}`"))?,
                        )
                    };
                    Ok(ShapedDimension {
                        size,
                        scalable,
                        invalid: None,
                    })
                })
                .collect::<Result<Vec<ShapedDimension>, String>>()?;
            let element = Box::new(resolve_type(
                element,
                type_aliases,
                attribute_aliases,
                stack,
            )?);
            return Ok(match constructor {
                0 => TypeValue::Tensor {
                    dimensions,
                    element,
                    encoding: match parts.get(1) {
                        Some(encoding) => Some(Box::new(resolve_attribute(
                            encoding,
                            type_aliases,
                            attribute_aliases,
                            stack,
                        )?)),
                        None => None,
                    },
                    unranked,
                },
                1 => TypeValue::Vector {
                    dimensions,
                    element,
                    scalable: shape_parts[..shape_parts.len() - 1]
                        .iter()
                        .map(|d| d.starts_with('['))
                        .collect(),
                },
                _ => TypeValue::MemRef {
                    dimensions,
                    element,
                    layout: parts
                        .get(1)
                        .map(|value| {
                            resolve_memref_layout(value, type_aliases, attribute_aliases, stack)
                        })
                        .transpose()?,
                    memory_space: parts
                        .get(2)
                        .map(|value| {
                            resolve_memref_memory_space(
                                value,
                                type_aliases,
                                attribute_aliases,
                                stack,
                            )
                            .map(Box::new)
                        })
                        .transpose()?,
                },
            });
        }
    }
    if spelling.starts_with('!') {
        if spelling.contains('<') {
            let Some(open) = spelling.find('<') else {
                unreachable!()
            };
            let Some(close) = matching_delimiter(&spelling[open + 1..], '<', '>') else {
                return Err(format!("malformed opaque type `{spelling}`"));
            };
            if !spelling[open + 1 + close + 1..].trim().is_empty() {
                return Err(format!("trailing garbage after opaque type `{spelling}`"));
            }
        }
        return Ok(TypeValue::Opaque(Arc::from(spelling.as_bytes())));
    }
    Err(format!("unsupported or malformed type `{spelling}`"))
}

fn resolve_attribute(
    spelling: &str,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
) -> Result<AttributeValue, String> {
    let spelling = spelling.trim();
    if spelling == "true" || spelling == "false" {
        return Ok(AttributeValue::Boolean(spelling == "true"));
    }
    if spelling.starts_with("dense_resource<") {
        return balanced_large_attribute(spelling, "dense_resource", LargeAttributeValue::Resource);
    }
    if spelling.starts_with("dense<") {
        return balanced_large_attribute(spelling, "dense", LargeAttributeValue::Dense);
    }
    if spelling.starts_with("sparse<") {
        return balanced_large_attribute(spelling, "sparse", LargeAttributeValue::Sparse);
    }
    if spelling.starts_with('#') && !spelling.contains('<') {
        let other = format!("!{}", &spelling[1..]);
        if type_aliases.contains_key(&other) {
            return Err(format!(
                "alias `{spelling}` has type kind, expected attribute"
            ));
        }
        let Some((target, _)) = attribute_aliases.get(spelling) else {
            return Err(format!("unresolved attribute alias `{spelling}`"));
        };
        stack.enter(spelling, "attribute")?;
        let result = resolve_attribute(target, type_aliases, attribute_aliases, stack);
        stack.exit(spelling);
        return result;
    }
    if spelling.starts_with('@') {
        return Ok(AttributeValue::Symbol(
            spelling
                .split("::")
                .map(|s| s.trim_start_matches('@').to_owned())
                .collect(),
        ));
    }
    if spelling.starts_with('"') {
        return Ok(AttributeValue::String(spelling.to_owned()));
    }
    if let Some(inner) = bracket_inner(spelling, '[', ']') {
        return Ok(AttributeValue::Array(
            split_types(inner)
                .iter()
                .map(|s| resolve_attribute(s, type_aliases, attribute_aliases, stack))
                .collect::<Result<_, _>>()?,
        ));
    }
    if let Some(inner) = bracket_inner(spelling, '{', '}') {
        let mut entries = split_types(inner)
            .into_iter()
            .map(|entry| {
                let (name, value) = entry
                    .split_once('=')
                    .ok_or_else(|| format!("malformed dictionary entry `{entry}`"))?;
                Ok((
                    name.trim().to_owned(),
                    resolve_attribute(value, type_aliases, attribute_aliases, stack)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        return Ok(AttributeValue::Dictionary(entries));
    }
    if spelling.starts_with("loc(") {
        return parse_location(spelling)
            .map(AttributeValue::Location)
            .ok_or_else(|| "invalid semantic location".into());
    }
    if let Ok(ty) = resolve_type(spelling, type_aliases, attribute_aliases, stack) {
        return Ok(AttributeValue::Type(ty));
    }
    let literal = spelling.split(':').next().unwrap_or(spelling).trim();
    if is_valid_wide_number(spelling) {
        return Ok(AttributeValue::WideNumber(Arc::from(spelling.as_bytes())));
    }
    if literal.parse::<i128>().is_ok() {
        return Ok(AttributeValue::Integer(compact(spelling)));
    }
    if literal.parse::<f64>().is_ok() {
        return Ok(AttributeValue::Float(compact(spelling)));
    }
    if spelling.starts_with('#') {
        let Some(open) = spelling.find('<') else {
            return Err(format!("malformed opaque attribute `{spelling}`"));
        };
        let Some(close) = matching_delimiter(&spelling[open + 1..], '<', '>') else {
            return Err(format!("malformed opaque attribute `{spelling}`"));
        };
        if !spelling[open + 1 + close + 1..].trim().is_empty() {
            return Err(format!(
                "trailing garbage after opaque attribute `{spelling}`"
            ));
        }
        return Ok(AttributeValue::Opaque(Arc::from(spelling.as_bytes())));
    }
    Err(format!("unsupported or malformed attribute `{spelling}`"))
}

fn balanced_large_attribute(
    spelling: &str,
    prefix: &str,
    wrap: impl FnOnce(Arc<[u8]>) -> LargeAttributeValue,
) -> Result<AttributeValue, String> {
    let rest = spelling
        .strip_prefix(prefix)
        .and_then(|value| value.strip_prefix('<'))
        .ok_or_else(|| format!("malformed {prefix} payload"))?;
    let close =
        matching_delimiter(rest, '<', '>').ok_or_else(|| format!("malformed {prefix} payload"))?;
    let suffix = rest[close + 1..].trim();
    let Some(suffix) = suffix.strip_prefix(':').map(str::trim) else {
        return Err(format!("malformed {prefix} payload suffix"));
    };
    if suffix.is_empty()
        || resolve_type(
            suffix,
            &HashMap::new(),
            &HashMap::new(),
            &mut AliasExpansionState::new(64),
        )
        .is_err()
    {
        return Err(format!("malformed {prefix} payload suffix"));
    }
    Ok(AttributeValue::Large(wrap(Arc::from(spelling.as_bytes()))))
}

fn is_valid_wide_number(value: &str) -> bool {
    let Some((literal, suffix)) = value.split_once(':') else {
        return false;
    };
    let literal = literal.trim().trim_start_matches(['+', '-']);
    let suffix = suffix.trim();
    if literal.is_empty() || suffix.len() < 2 {
        return false;
    }
    let digits = literal.strip_prefix("0x").unwrap_or(literal);
    let is_hex = literal.starts_with("0x");
    if digits.is_empty()
        || (!is_hex && !digits.bytes().all(|byte| byte.is_ascii_digit()))
        || (is_hex && !digits.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return false;
    }
    let width = suffix
        .strip_prefix('i')
        .or_else(|| suffix.strip_prefix("si"))
        .or_else(|| suffix.strip_prefix("ui"));
    width.is_some_and(|width| parse_width(width).is_some())
}

fn parse_location(spelling: &str) -> Option<LocationValue> {
    let inner = spelling
        .trim()
        .strip_prefix("loc(")?
        .strip_suffix(')')?
        .trim();
    if inner == "unknown" {
        return Some(LocationValue::Unknown);
    }
    if let Some(fused) = inner.strip_prefix("fused") {
        let (metadata, values) = if let Some(rest) = fused.strip_prefix('<') {
            let end = matching_delimiter(rest, '<', '>')?;
            (Some(rest[..end].trim().to_owned()), rest[end + 1..].trim())
        } else {
            (None, fused.trim())
        };
        let values = bracket_inner(values, '[', ']')?;
        return Some(LocationValue::Fused {
            metadata,
            locations: split_top_level_commas(values)
                .iter()
                .map(|value| parse_location_detail(value))
                .collect::<Option<Vec<_>>>()?,
        });
    }
    if let Some(callsite) = inner
        .strip_prefix("callsite(")
        .and_then(|value| value.strip_suffix(')'))
    {
        let (callee, caller) = split_at_keyword(callsite, " at ")?;
        return Some(LocationValue::CallSite {
            callee: Box::new(parse_location_detail(callee)?),
            caller: Box::new(parse_location_detail(caller)?),
        });
    }
    parse_location_detail(inner)
}

fn lower_location_value(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> LocationValue {
    let invalid = |doc: &mut Document, message: String| {
        LocationValue::Invalid(push_diagnostic(doc, range, message))
    };
    let spelling = spelling.trim();
    if spelling.starts_with('#') {
        let alias = spelling.to_owned();
        let Some((target, _)) = attribute_aliases.get(&alias) else {
            let message = if type_aliases.contains_key(&format!("!{}", &alias[1..])) {
                format!("alias `{alias}` has type kind, expected location")
            } else {
                format!("unresolved location alias `{alias}`")
            };
            return invalid(doc, message);
        };
        if let Err(message) = stack.enter(&alias, "location") {
            return invalid(doc, message);
        }
        let wrapped;
        let target = if target.starts_with("loc(") {
            target.as_str()
        } else {
            wrapped = format!("loc({target})");
            wrapped.as_str()
        };
        let result =
            lower_location_value(target, range, type_aliases, attribute_aliases, stack, doc);
        stack.exit(&alias);
        return result;
    }
    let Some(inner) = spelling
        .strip_prefix("loc(")
        .and_then(|value| value.strip_suffix(')'))
        .map(str::trim)
    else {
        return invalid(doc, "invalid semantic location".into());
    };
    if inner.starts_with('#') {
        return lower_location_value(inner, range, type_aliases, attribute_aliases, stack, doc);
    }
    if let Some(fused) = inner.strip_prefix("fused") {
        let (metadata, values) = match fused_parts(fused) {
            Some(parts) => parts,
            None => return invalid(doc, "malformed fused location".into()),
        };
        let locations = split_top_level_commas(values)
            .iter()
            .map(|value| {
                lower_location_detail(value, range, type_aliases, attribute_aliases, stack, doc)
            })
            .collect();
        return LocationValue::Fused {
            metadata,
            locations,
        };
    }
    if let Some(callsite) = inner
        .strip_prefix("callsite(")
        .and_then(|value| value.strip_suffix(')'))
    {
        let Some((callee, caller)) = split_at_keyword(callsite, " at ") else {
            return invalid(doc, "malformed callsite location".into());
        };
        return LocationValue::CallSite {
            callee: Box::new(lower_location_detail(
                callee,
                range,
                type_aliases,
                attribute_aliases,
                stack,
                doc,
            )),
            caller: Box::new(lower_location_detail(
                caller,
                range,
                type_aliases,
                attribute_aliases,
                stack,
                doc,
            )),
        };
    }
    lower_location_detail(inner, range, type_aliases, attribute_aliases, stack, doc)
}

fn lower_location_detail(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> LocationValue {
    let spelling = spelling.trim();
    if spelling.starts_with("loc(") || spelling.starts_with('#') {
        lower_location_value(spelling, range, type_aliases, attribute_aliases, stack, doc)
    } else if spelling.starts_with("callsite(") || spelling.starts_with("fused") {
        let wrapped = format!("loc({spelling})");
        lower_location_value(&wrapped, range, type_aliases, attribute_aliases, stack, doc)
    } else if let Some(stripped) = spelling.strip_prefix('"') {
        let Some(quote_end) = stripped.find('"').map(|index| index + 1) else {
            return LocationValue::Invalid(push_diagnostic(
                doc,
                range,
                format!("invalid nested location `{spelling}`"),
            ));
        };
        let name = spelling[..=quote_end].to_owned();
        let rest = spelling[quote_end + 1..].trim();
        if rest.starts_with(':') {
            return parse_location_detail(spelling).unwrap_or_else(|| {
                LocationValue::Invalid(push_diagnostic(
                    doc,
                    range,
                    format!("invalid nested location `{spelling}`"),
                ))
            });
        }
        let child = if rest.starts_with('(') && rest.ends_with(')') {
            Some(Box::new(lower_location_detail(
                &rest[1..rest.len() - 1],
                range,
                type_aliases,
                attribute_aliases,
                stack,
                doc,
            )))
        } else {
            None
        };
        let has_child = child.is_some();
        LocationValue::Name {
            name,
            child,
            metadata: (!rest.is_empty() && !has_child).then(|| compact(rest)),
        }
    } else {
        parse_location_detail(spelling).unwrap_or_else(|| {
            LocationValue::Invalid(push_diagnostic(
                doc,
                range,
                format!("invalid nested location `{spelling}`"),
            ))
        })
    }
}

fn fused_parts(value: &str) -> Option<(Option<String>, &str)> {
    let value = value.trim();
    if let Some(rest) = value.strip_prefix('<') {
        let end = matching_delimiter(rest, '<', '>')?;
        let metadata = Some(rest[..end].trim().to_owned());
        let values = rest[end + 1..].trim();
        Some((metadata, bracket_inner(values, '[', ']')?))
    } else {
        Some((None, bracket_inner(value, '[', ']')?))
    }
}

fn parse_location_detail(value: &str) -> Option<LocationValue> {
    let value = value.trim();
    if let Some(stripped) = value.strip_prefix('"') {
        let quote_end = stripped.find('"')? + 1;
        let name = value[..=quote_end].to_owned();
        let rest = value[quote_end + 1..].trim();
        if rest.starts_with(':') {
            return parse_file_line_column(value);
        }
        let child = if rest.starts_with('(') && rest.ends_with(')') {
            Some(Box::new(parse_location_detail(&rest[1..rest.len() - 1])?))
        } else {
            None
        };
        let has_child = child.is_some();
        return Some(LocationValue::Name {
            name,
            child,
            metadata: (!rest.is_empty() && !has_child).then(|| compact(rest)),
        });
    }
    parse_file_line_column(value)
}

fn valid_diagnostic(document: &Document, diagnostic: DiagnosticId) -> bool {
    !document.complete
        && document.valid(
            diagnostic.index,
            diagnostic.generation,
            document.diagnostics.len(),
        )
}

fn valid_type_value(document: &Document, value: &TypeValue) -> bool {
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

fn valid_attribute_value(document: &Document, value: &AttributeValue) -> bool {
    match value {
        AttributeValue::Type(value) => valid_type_value(document, value),
        AttributeValue::Array(values) => values
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

fn valid_affine_storage(document: &Document) -> bool {
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

fn valid_location_value(document: &Document, value: &LocationValue) -> bool {
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

fn parse_file_line_column(value: &str) -> Option<LocationValue> {
    let mut parts = value.rsplitn(3, ':');
    let column = parts.next()?.trim().parse().ok()?;
    let line = parts.next()?.trim().parse().ok()?;
    Some(LocationValue::FileLineColumn {
        file: parts.next()?.trim().to_owned(),
        line,
        column,
    })
}

fn split_at_keyword<'a>(value: &'a str, keyword: &str) -> Option<(&'a str, &'a str)> {
    let index = value.find(keyword)?;
    Some((&value[..index], &value[index + keyword.len()..]))
}

fn matching_delimiter(value: &str, _open: char, close: char) -> Option<usize> {
    let mut expected = vec![close];
    let mut quoted = false;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
            continue;
        }
        if character == '"' {
            quoted = true;
            continue;
        }
        if let Some(nested_close) = delimiter_close(character) {
            expected.push(nested_close);
            continue;
        }
        if matches!(character, ')' | ']' | '}' | '>') {
            if expected.pop() != Some(character) {
                return None;
            }
            if expected.is_empty() {
                return Some(index);
            }
        }
    }
    None
}

fn delimiter_close(character: char) -> Option<char> {
    match character {
        '(' => Some(')'),
        '[' => Some(']'),
        '{' => Some('}'),
        '<' => Some('>'),
        _ => None,
    }
}

fn compact(value: &str) -> String {
    value.chars().filter(|c| !c.is_whitespace()).collect()
}
fn angle_inner<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    value
        .strip_prefix(prefix)?
        .trim()
        .strip_prefix('<')?
        .strip_suffix('>')
}
fn bracket_inner(value: &str, open: char, close: char) -> Option<&str> {
    value.trim().strip_prefix(open)?.strip_suffix(close)
}
fn split_arrow(value: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    let mut quoted = false;
    let mut escaped = false;
    let bytes = value.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        let byte = bytes[i];
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        if byte == b'"' {
            quoted = true;
            continue;
        }
        match byte {
            b'(' | b'<' | b'[' | b'{' => depth += 1,
            b')' | b'>' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if &bytes[i..i + 2] == b"->" && depth == 0 {
            return Some((&value[..i], &value[i + 2..]));
        }
    }
    None
}
fn split_top_level_x(value: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    for (i, byte) in value.bytes().enumerate() {
        match byte {
            b'<' | b'(' | b'[' | b'{' => depth += 1,
            b'>' | b')' | b']' | b'}' => depth -= 1,
            b'x' if depth == 0 => {
                result.push(value[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(value[start..].trim());
    result
}

fn split_top_level_commas(value: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'<' | b'(' | b'[' | b'{' => depth += 1,
            b'>' | b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                result.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(value[start..].trim());
    result
}

fn parse_width(value: &str) -> Option<u32> {
    (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

fn is_float_spelling(value: &str) -> bool {
    matches!(
        value,
        "bf16"
            | "tf32"
            | "f16"
            | "f32"
            | "f64"
            | "f80"
            | "f128"
            | "f8E4M3"
            | "f8E5M2"
            | "f8E4M3FN"
            | "f8E5M2FNUZ"
            | "f8E4M3FNUZ"
            | "f8E4M3B11FNUZ"
            | "f8E3M4"
            | "f8E8M0FNU"
            | "f4E2M1FN"
            | "f6E2M3FN"
            | "f6E3M2FN"
    )
}

// Attribute and property dictionaries share this lowering path, whose inputs are
// deliberately passed separately to keep the surrounding lowering state local.
#[allow(clippy::too_many_arguments)]
fn lower_dictionary(
    dictionary: Option<crate::representation::NodeId>,
    tree: &crate::representation::SyntaxTree,
    bytes: &[u8],
    strings: &mut Interner,
    attributes: &mut Interner<AttributeValue>,
    attribute_spellings: &mut Vec<String>,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    generation: u128,
    kind: &str,
    doc: &mut Document,
) -> Vec<(u32, AttributeId)> {
    let Some(dictionary) = dictionary else {
        return Vec::new();
    };
    let mut seen = HashMap::<u32, TextRange>::new();
    let mut result = tree
        .children(dictionary)
        .into_iter()
        .flatten()
        .filter(|child| tree.kind(*child) == Some(SyntaxKind::Attribute))
        .filter_map(|attribute| {
            let attribute_range = tree.text_range(attribute)?;
            let spelling = text(bytes, attribute_range);
            let (name, value) = spelling.split_once('=').unwrap_or((spelling, ""));
            let name_id = strings.intern(name.trim());
            let duplicate = seen.insert(name_id, attribute_range).map(|previous| {
                push_diagnostic(
                    doc,
                    attribute_range,
                    format!(
                        "duplicate {kind} key `{}` (previous at {})",
                        name.trim(),
                        previous.start()
                    ),
                )
            });
            let value_spelling = value.trim();
            let malformed_numeric_prefix = value_spelling == "0"
                && bytes
                    .get(attribute_range.end() as usize)
                    .is_some_and(|byte| *byte == b'x');
            let integer_suffix = value_spelling
                .split_once(':')
                .map(|(_, suffix)| suffix.trim())
                .and_then(|suffix| {
                    suffix
                        .strip_prefix('i')
                        .or_else(|| suffix.strip_prefix("si"))
                        .or_else(|| suffix.strip_prefix("ui"))
                })
                .is_some_and(|width| parse_width(width).is_some());
            let numeric_payload_candidate = malformed_numeric_prefix
                || (value_spelling
                    .trim_start_matches(['+', '-'])
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_digit())
                    && value_spelling.contains(':'))
                || integer_suffix;
            let affine_value = value_spelling.starts_with("affine_map<")
                || value_spelling.starts_with("affine_set<");
            let owned_payload_candidate = !affine_value
                && (value_spelling.starts_with("dense<")
                    || value_spelling.starts_with("sparse<")
                    || value_spelling.starts_with("dense_resource<")
                    || (value_spelling.starts_with('#') && value_spelling.contains('<'))
                    || numeric_payload_candidate);
            let semantic = if name.trim() == "no_inline" && value_spelling.is_empty() {
                AttributeValue::Opaque(Arc::from(b"unit".as_slice()))
            } else if malformed_numeric_prefix
                || (owned_payload_candidate && tree.has_error(attribute).unwrap_or(false))
            {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    attribute_range,
                    "malformed attribute value".into(),
                ))
            } else {
                let mut expansion = AliasExpansionState::new(doc.alias_expansion_depth_limit);
                lower_attribute_value(
                    value_spelling,
                    attribute_range,
                    type_aliases,
                    attribute_aliases,
                    &mut expansion,
                    doc,
                )
            };
            let semantic = if let Some(diagnostic) = duplicate {
                AttributeValue::Invalid(diagnostic)
            } else if value_spelling.is_empty() && name.trim() != "no_inline" {
                AttributeValue::Invalid(push_diagnostic(
                    doc,
                    attribute_range,
                    "malformed dictionary entry".into(),
                ))
            } else {
                semantic
            };
            let index = attributes.intern_value(semantic);
            if index as usize == attribute_spellings.len() {
                attribute_spellings.push(value_spelling.to_owned());
            }
            Some((name_id, AttributeId::new(index as usize, generation)))
        })
        .collect::<Vec<_>>();
    result.sort_by_key(|(name, _)| strings.values[*name as usize].clone());
    result
}

fn lower_attribute_value(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
) -> AttributeValue {
    lower_attribute_value_with_depth(
        spelling,
        range,
        type_aliases,
        attribute_aliases,
        stack,
        doc,
        0,
    )
}

fn lower_attribute_value_with_depth(
    spelling: &str,
    range: TextRange,
    type_aliases: &HashMap<String, (String, TextRange)>,
    attribute_aliases: &HashMap<String, (String, TextRange)>,
    stack: &mut AliasExpansionState,
    doc: &mut Document,
    depth: usize,
) -> AttributeValue {
    let spelling = spelling.trim();
    if spelling == "true" || spelling == "false" {
        return AttributeValue::Boolean(spelling == "true");
    }
    if spelling == "unit" {
        return AttributeValue::Opaque(Arc::from(b"unit".as_slice()));
    }
    if let Some(inner) = angle_inner(spelling, "type") {
        return AttributeValue::Type(lower_type_value_with_stack(
            inner,
            range,
            type_aliases,
            attribute_aliases,
            stack,
            doc,
        ));
    }
    if spelling.starts_with("affine_map<") || spelling.starts_with("affine_set<") {
        return lower_affine_attribute(spelling, range, doc);
    }
    if spelling.starts_with('#') && !spelling.contains('<') {
        match resolve_affine_alias(spelling, attribute_aliases, stack) {
            Ok(Some(target)) => return lower_affine_attribute(target, range, doc),
            Err(message) => {
                return AttributeValue::Invalid(push_diagnostic(doc, range, message));
            }
            Ok(None) => {}
        }
    }
    if let Some(inner) = bracket_inner(spelling, '[', ']') {
        if depth >= doc.attribute_depth_limit {
            return AttributeValue::Invalid(push_diagnostic(
                doc,
                range,
                "attribute nesting depth limit exceeded".into(),
            ));
        }
        return AttributeValue::Array(
            split_types(inner)
                .iter()
                .map(|item| {
                    lower_attribute_value_with_depth(
                        item,
                        range,
                        type_aliases,
                        attribute_aliases,
                        stack,
                        doc,
                        depth + 1,
                    )
                })
                .collect(),
        );
    }
    if let Some(inner) = bracket_inner(spelling, '{', '}') {
        if depth >= doc.attribute_depth_limit {
            return AttributeValue::Invalid(push_diagnostic(
                doc,
                range,
                "attribute nesting depth limit exceeded".into(),
            ));
        }
        let mut seen = HashMap::new();
        let mut entries = split_types(inner)
            .into_iter()
            .map(|entry| {
                let Some((name, value)) = entry.split_once('=') else {
                    push_diagnostic(doc, range, format!("malformed dictionary entry `{entry}`"));
                    return (
                        "<invalid>".into(),
                        AttributeValue::Invalid(push_diagnostic(
                            doc,
                            range,
                            "malformed dictionary entry".into(),
                        )),
                    );
                };
                let name = name.trim().to_owned();
                let duplicate = seen.insert(name.clone(), ()).is_some();
                let value = lower_attribute_value_with_depth(
                    value,
                    range,
                    type_aliases,
                    attribute_aliases,
                    stack,
                    doc,
                    depth + 1,
                );
                let value = if duplicate {
                    AttributeValue::Invalid(push_diagnostic(
                        doc,
                        range,
                        format!("duplicate dictionary key `{name}`"),
                    ))
                } else {
                    value
                };
                (name, value)
            })
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        return AttributeValue::Dictionary(entries);
    }
    if spelling.starts_with("loc(") {
        return AttributeValue::Location(lower_location_value(
            spelling,
            range,
            type_aliases,
            attribute_aliases,
            stack,
            doc,
        ));
    }
    if spelling.starts_with('!')
        || spelling.starts_with('i')
        || spelling.starts_with("si")
        || spelling.starts_with("ui")
        || spelling.starts_with('f')
        || spelling.starts_with('b')
        || spelling.starts_with('t')
        || spelling == "index"
        || spelling.starts_with("tensor<")
        || spelling.starts_with("vector<")
        || spelling.starts_with("memref<")
        || spelling.starts_with("tuple<")
        || spelling.contains("->")
    {
        return AttributeValue::Type(lower_type_value_with_stack(
            spelling,
            range,
            type_aliases,
            attribute_aliases,
            stack,
            doc,
        ));
    }
    match resolve_attribute(spelling, type_aliases, attribute_aliases, stack) {
        Ok(value) => value,
        Err(message) => AttributeValue::Invalid(push_diagnostic(doc, range, message)),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AffineToken {
    Identifier(String),
    Integer(i64),
    InvalidInteger(String),
    InvalidOperator(String),
    Plus,
    Minus,
    Star,
    FloorDiv,
    CeilDiv,
    Mod,
    LParen,
    RParen,
}

fn lower_affine_attribute(spelling: &str, range: TextRange, doc: &mut Document) -> AttributeValue {
    let (kind, inner) = if let Some(inner) = angle_inner(spelling, "affine_map") {
        (SyntaxKind::AffineMap, inner)
    } else if let Some(inner) = angle_inner(spelling, "affine_set") {
        (SyntaxKind::IntegerSet, inner)
    } else {
        let kind = if spelling.trim().starts_with("affine_map") {
            SyntaxKind::AffineMap
        } else {
            SyntaxKind::IntegerSet
        };
        return invalid_affine_attribute(kind, range, doc, "malformed affine value");
    };
    let Some(after_open) = inner.strip_prefix('(') else {
        return invalid_affine_attribute(kind, range, doc, "malformed affine dimension arity");
    };
    let Some(dim_tail_end) = matching_delimiter(after_open, '(', ')') else {
        return invalid_affine_attribute(kind, range, doc, "malformed affine dimension arity");
    };
    let dim_end = dim_tail_end + 1;
    let dimensions = parse_affine_names(&inner[1..dim_end], "dimension", range, doc);
    let mut rest = inner.get(dim_end + 1..).unwrap_or("").trim();
    let symbols = if rest.starts_with('[') {
        let Some(after_open) = rest.strip_prefix('[') else {
            return invalid_affine_attribute(kind, range, doc, "malformed affine symbol arity");
        };
        let Some(tail_end) = matching_delimiter(after_open, '[', ']') else {
            return invalid_affine_attribute(kind, range, doc, "malformed affine symbol arity");
        };
        let end = tail_end + 1;
        let names = parse_affine_names(&rest[1..end], "symbol", range, doc);
        rest = rest.get(end + 1..).unwrap_or("").trim();
        names
    } else {
        Vec::new()
    };
    let separator = if kind == SyntaxKind::AffineMap {
        "->"
    } else {
        ":"
    };
    let Some(body) = rest.strip_prefix(separator).map(str::trim) else {
        return invalid_affine_attribute(
            kind,
            range,
            doc,
            &format!("malformed affine {separator} separator"),
        );
    };
    let Some(body) = bracket_inner(body, '(', ')') else {
        return invalid_affine_attribute(
            kind,
            range,
            doc,
            "malformed affine result or constraint list",
        );
    };
    if kind == SyntaxKind::AffineMap {
        let results = split_affine_items(body)
            .into_iter()
            .map(|expression| {
                lower_affine_expression(expression, &dimensions, &symbols, range, doc)
            })
            .collect::<Vec<_>>();
        let value = AffineMapValue {
            dimensions: dimensions.len() as u32,
            symbols: symbols.len() as u32,
            results,
        };
        let index = intern_affine_map(doc, value);
        AttributeValue::AffineMap(AffineMapId::new(index, doc.generation))
    } else {
        let constraints = split_affine_items(body)
            .into_iter()
            .map(|constraint| {
                lower_integer_constraint(constraint, &dimensions, &symbols, range, doc)
            })
            .collect::<Vec<_>>();
        let value = IntegerSetValue {
            dimensions: dimensions.len() as u32,
            symbols: symbols.len() as u32,
            constraints,
        };
        let index = intern_integer_set(doc, value);
        AttributeValue::IntegerSet(IntegerSetId::new(index, doc.generation))
    }
}

fn invalid_affine_attribute(
    kind: SyntaxKind,
    range: TextRange,
    doc: &mut Document,
    message: &str,
) -> AttributeValue {
    let expression = invalid_affine_expression(doc, range, message);
    if kind == SyntaxKind::AffineMap {
        let index = intern_affine_map(
            doc,
            AffineMapValue {
                dimensions: 0,
                symbols: 0,
                results: vec![expression],
            },
        );
        AttributeValue::AffineMap(AffineMapId::new(index, doc.generation))
    } else {
        let diagnostic = push_diagnostic(doc, range, message.to_owned());
        let right = invalid_affine_expression(doc, range, message);
        let index = intern_integer_set(
            doc,
            IntegerSetValue {
                dimensions: 0,
                symbols: 0,
                constraints: vec![IntegerSetConstraint {
                    left: expression,
                    relation: IntegerSetRelation::Invalid(diagnostic),
                    right,
                }],
            },
        );
        AttributeValue::IntegerSet(IntegerSetId::new(index, doc.generation))
    }
}

fn invalid_affine_expression(doc: &mut Document, range: TextRange, message: &str) -> AffineExprId {
    let diagnostic = push_diagnostic(doc, range, message.to_owned());
    let index = intern_affine_expression(doc, AffineExprValue::Invalid(diagnostic));
    AffineExprId::new(index, doc.generation)
}

fn split_affine_items(value: &str) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut result = Vec::new();
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                result.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(value[start..].trim());
    result
}

fn parse_affine_names(
    value: &str,
    kind: &str,
    range: TextRange,
    doc: &mut Document,
) -> Vec<String> {
    let mut names = Vec::new();
    for name in value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        if !name.bytes().enumerate().all(|(i, byte)| {
            byte == b'_' || byte.is_ascii_alphanumeric() && (i > 0 || !byte.is_ascii_digit())
        }) {
            push_diagnostic(
                doc,
                range,
                format!("malformed affine {kind} identifier `{name}`"),
            );
        }
        if names.iter().any(|existing| existing == name) {
            push_diagnostic(
                doc,
                range,
                format!("duplicate affine {kind} identifier `{name}`"),
            );
        }
        names.push(name.to_owned());
    }
    names
}

fn lower_integer_constraint(
    value: &str,
    dimensions: &[String],
    symbols: &[String],
    range: TextRange,
    doc: &mut Document,
) -> IntegerSetConstraint {
    let mut depth = 0usize;
    let mut found = None;
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b'=' | b'>' | b'<' if depth == 0 => {
                found = Some((index, byte));
                break;
            }
            _ => {}
        }
    }
    let (left, relation, right) = match found {
        Some((index, b'=')) if value[index..].starts_with("==") => (
            &value[..index],
            IntegerSetRelation::Equal,
            &value[index + 2..],
        ),
        Some((index, b'>')) if value[index..].starts_with(">=") => (
            &value[..index],
            IntegerSetRelation::GreaterEqual,
            &value[index + 2..],
        ),
        Some((index, b'<')) if value[index..].starts_with("<=") => (
            &value[..index],
            IntegerSetRelation::LessEqual,
            &value[index + 2..],
        ),
        Some((index, _)) => {
            let diagnostic = push_diagnostic(
                doc,
                range,
                format!("invalid affine constraint operator in `{value}`"),
            );
            (
                &value[..index],
                IntegerSetRelation::Invalid(diagnostic),
                &value[index + 1..],
            )
        }
        None => {
            let diagnostic = push_diagnostic(
                doc,
                range,
                format!("missing affine constraint operator in `{value}`"),
            );
            (value, IntegerSetRelation::Invalid(diagnostic), "")
        }
    };
    IntegerSetConstraint {
        left: lower_affine_expression(left, dimensions, symbols, range, doc),
        relation,
        right: lower_affine_expression(right, dimensions, symbols, range, doc),
    }
}

fn lower_affine_expression(
    value: &str,
    dimensions: &[String],
    symbols: &[String],
    range: TextRange,
    doc: &mut Document,
) -> AffineExprId {
    let tokens = tokenize_affine(value);
    let mut parser = AffineExpressionParser {
        tokens: &tokens,
        position: 0,
        dimensions,
        symbols,
        range,
        doc,
    };
    let expression = parser.expression(0);
    if parser.position != tokens.len() {
        return parser.invalid(format!("malformed affine expression `{}`", value.trim()));
    }
    expression
}

fn tokenize_affine(value: &str) -> Vec<AffineToken> {
    let bytes = value.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let token = match bytes[i] {
            b'+' => {
                i += 1;
                AffineToken::Plus
            }
            b'-' => {
                i += 1;
                AffineToken::Minus
            }
            b'*' => {
                i += 1;
                AffineToken::Star
            }
            b'(' => {
                i += 1;
                AffineToken::LParen
            }
            b')' => {
                i += 1;
                AffineToken::RParen
            }
            b'0'..=b'9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let literal = &value[start..i];
                match literal.parse() {
                    Ok(value) => AffineToken::Integer(value),
                    Err(_) => AffineToken::InvalidInteger(literal.to_owned()),
                }
            }
            b'/' | b'%' => {
                let operator = bytes[i] as char;
                i += 1;
                AffineToken::InvalidOperator(operator.to_string())
            }
            _ => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &value[start..i.max(start + 1)];
                if i == start {
                    i += 1;
                }
                match word {
                    "floordiv" => AffineToken::FloorDiv,
                    "ceildiv" => AffineToken::CeilDiv,
                    "mod" => AffineToken::Mod,
                    _ => AffineToken::Identifier(word.to_owned()),
                }
            }
        };
        tokens.push(token);
    }
    tokens
}

struct AffineExpressionParser<'a> {
    tokens: &'a [AffineToken],
    position: usize,
    dimensions: &'a [String],
    symbols: &'a [String],
    range: TextRange,
    doc: &'a mut Document,
}

impl AffineExpressionParser<'_> {
    fn expression(&mut self, minimum: u8) -> AffineExprId {
        let mut left = self.primary();
        loop {
            let (precedence, operator) = match self.tokens.get(self.position) {
                Some(AffineToken::Plus) => (1, AffineBinaryOperator::Add),
                Some(AffineToken::Minus) => (1, AffineBinaryOperator::Subtract),
                Some(AffineToken::Star) => (2, AffineBinaryOperator::Multiply),
                Some(AffineToken::FloorDiv) => (2, AffineBinaryOperator::FloorDiv),
                Some(AffineToken::CeilDiv) => (2, AffineBinaryOperator::CeilDiv),
                Some(AffineToken::Mod) => (2, AffineBinaryOperator::Mod),
                _ => break,
            };
            if precedence < minimum {
                break;
            }
            self.position += 1;
            let right = self.expression(precedence + 1);
            left = self.intern(AffineExprValue::Binary {
                operator,
                left,
                right,
            });
        }
        left
    }
    fn primary(&mut self) -> AffineExprId {
        match self.tokens.get(self.position).cloned() {
            Some(AffineToken::Integer(value)) => {
                self.position += 1;
                self.intern(AffineExprValue::Constant(value))
            }
            Some(AffineToken::InvalidInteger(literal)) => {
                self.position += 1;
                self.invalid(format!(
                    "affine integer literal `{literal}` is out of range for i64"
                ))
            }
            Some(AffineToken::InvalidOperator(operator)) => {
                self.position += 1;
                self.invalid(format!("unsupported affine operator `{operator}`"))
            }
            Some(AffineToken::Identifier(name)) => {
                self.position += 1;
                if let Some(index) = self.dimensions.iter().position(|value| value == &name) {
                    self.intern(AffineExprValue::Dimension(index as u32))
                } else if let Some(index) = self.symbols.iter().position(|value| value == &name) {
                    self.intern(AffineExprValue::Symbol(index as u32))
                } else {
                    self.invalid(format!(
                        "affine identifier `{name}` exceeds declared dimension/symbol arity"
                    ))
                }
            }
            Some(AffineToken::Minus) => {
                self.position += 1;
                let right = self.primary();
                let zero = self.intern(AffineExprValue::Constant(0));
                self.intern(AffineExprValue::Binary {
                    operator: AffineBinaryOperator::Subtract,
                    left: zero,
                    right,
                })
            }
            Some(AffineToken::LParen) => {
                self.position += 1;
                let value = self.expression(0);
                if self.tokens.get(self.position) == Some(&AffineToken::RParen) {
                    self.position += 1;
                    value
                } else {
                    self.invalid("unclosed affine expression".into())
                }
            }
            _ => self.invalid("missing affine expression operand".into()),
        }
    }
    fn intern(&mut self, value: AffineExprValue) -> AffineExprId {
        let index = intern_affine_expression(self.doc, value);
        AffineExprId::new(index, self.doc.generation)
    }
    fn invalid(&mut self, message: String) -> AffineExprId {
        let diagnostic = push_diagnostic(self.doc, self.range, message);
        self.intern(AffineExprValue::Invalid(diagnostic))
    }
}

fn intern_affine_expression(doc: &mut Document, value: AffineExprValue) -> usize {
    if let Some(index) = doc
        .affine_expressions
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.affine_expressions.len();
        doc.affine_expressions.push(value);
        index
    }
}
fn intern_affine_map(doc: &mut Document, value: AffineMapValue) -> usize {
    if let Some(index) = doc
        .affine_maps
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.affine_maps.len();
        doc.affine_maps.push(value);
        index
    }
}
fn intern_integer_set(doc: &mut Document, value: IntegerSetValue) -> usize {
    if let Some(index) = doc
        .integer_sets
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.integer_sets.len();
        doc.integer_sets.push(value);
        index
    }
}

// Resolution needs both region- and block-scoped definition maps plus the region
// ancestry; keeping them explicit makes the lookup rules visible at each call site.
#[allow(clippy::too_many_arguments)]
fn resolve_value(
    spelling: &str,
    range: TextRange,
    mut region: Option<RegionId>,
    block: Option<BlockId>,
    region_definitions: &HashMap<(Option<RegionId>, String), Vec<ValueId>>,
    block_definitions: &HashMap<(BlockId, String), Vec<ValueId>>,
    region_outer: &HashMap<RegionId, Option<RegionId>>,
    doc: &mut Document,
) -> ValueReference {
    let name = first_identifier(spelling, b'%').unwrap_or_default();
    let number = spelling
        .split_once('#')
        .and_then(|(_, number)| number.trim().split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|number| number.parse::<usize>().ok())
        .unwrap_or(0);
    if let Some(block) = block {
        if let Some(value) = block_definitions
            .get(&(block, name.clone()))
            .and_then(|values| values.get(number))
            .copied()
        {
            return ValueReference::Resolved(value);
        }
    }
    loop {
        if let Some(value) = region_definitions
            .get(&(region, name.clone()))
            .and_then(|values| values.get(number))
            .copied()
        {
            return ValueReference::Resolved(value);
        }
        let Some(current) = region else { break };
        region = region_outer.get(&current).copied().flatten();
    }
    let display = if spelling.contains('#') {
        format!("%{name}#{number}")
    } else {
        format!("%{name}")
    };
    let diagnostic = push_diagnostic(doc, range, format!("unresolved SSA value `{display}`"));
    ValueReference::Invalid(diagnostic)
}

fn value_type(doc: &Document, reference: ValueReference) -> Option<&str> {
    let value = match reference {
        ValueReference::Resolved(value) => value,
        ValueReference::Invalid(_) => return None,
    };
    let type_id = match value {
        ValueId::OperationResult { operation, result } => {
            let operation = doc.operation(operation)?;
            *doc.types_lists
                .get(operation.result_types)?
                .get(result as usize)?
        }
        ValueId::BlockArgument { block, argument } => {
            let block = doc.block(block)?;
            *doc.types_lists
                .get(block.argument_types)?
                .get(argument as usize)?
        }
    };
    doc.type_spelling(type_id)
}

fn push_diagnostic(doc: &mut Document, range: TextRange, message: String) -> DiagnosticId {
    let id = DiagnosticId::new(doc.diagnostics.len(), doc.generation);
    doc.diagnostics.push(SemanticDiagnostic { range, message });
    doc.complete = false;
    id
}

fn operation_name(bytes: &[u8], range: TextRange) -> Option<&str> {
    let mut text = bytes.get(range.start() as usize..range.end() as usize)?;
    let first = text.iter().position(|byte| !byte.is_ascii_whitespace())?;
    text = &text[first..];
    if text.first() == Some(&b'%') {
        let equal = text.iter().position(|byte| *byte == b'=')?;
        text = &text[equal + 1..];
        let first = text.iter().position(|byte| !byte.is_ascii_whitespace())?;
        text = &text[first..];
    }
    if text.first() != Some(&b'"') {
        return bare_operation_name(text);
    }
    let start = text.iter().position(|b| *b == b'"')? + 1;
    let end = start + text[start..].iter().position(|b| *b == b'"')?;
    std::str::from_utf8(&text[start..end]).ok()
}

fn bare_operation_name(text: &[u8]) -> Option<&str> {
    let start = text.iter().position(|byte| !byte.is_ascii_whitespace())?;
    let end = text[start..]
        .iter()
        .position(|byte| byte.is_ascii_whitespace() || b"@({[".contains(byte))
        .map_or(text.len(), |end| start + end);
    std::str::from_utf8(&text[start..end]).ok()
}

fn leading_symbol(spelling: &str) -> Option<&str> {
    let spelling = spelling.trim_start();
    let operation = if spelling.starts_with('%') {
        spelling.split_once('=')?.1
    } else {
        spelling
    };
    let mut parts = operation.split_ascii_whitespace();
    parts.next()?;
    let symbol = parts.next()?.strip_prefix('@')?;
    let end = symbol
        .bytes()
        .position(|byte| b"#: ,()={}[]".contains(&byte))
        .unwrap_or(symbol.len());
    Some(&symbol[..end])
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_identity_allocation_skips_used_values_across_wrap() {
        let mut identities = OperationIdentityState {
            next: u32::MAX - 1,
            allocated: HashSet::from([u32::MAX]),
        };
        assert_eq!(identities.allocate(), 1);
        assert_eq!(identities.allocate(), 2);
    }

    #[test]
    fn dropped_document_handles_stay_foreign_across_identity_counter_boundary() {
        let next = NEXT_DOCUMENT_IDENTITY.get_or_init(|| Mutex::new(1));
        *next
            .lock()
            .expect("document identity allocator is not poisoned") = u32::MAX as u128;

        let (old_operation, old_value, old_identity) = {
            let document = lower(b"%x = \"value\"() : () -> i32");
            let operation = document.root_operations()[0];
            let value = document
                .operation(operation)
                .unwrap()
                .result(operation, 0)
                .unwrap();
            (operation, value, document.identity.0)
        };
        let document = lower(b"%x = \"value\"() : () -> i32");
        assert_ne!(old_identity, document.identity.0);
        assert!(matches!(
            document.check_operation(old_operation),
            Err(EditError::ForeignOperation(_))
        ));
        assert!(matches!(
            document.check_value(old_value),
            Err(EditError::ForeignValue(_))
        ));
    }

    impl Document {
        fn corrupt_root_generation(&mut self) {
            self.operation_lists.0[self.roots.start as usize].generation =
                self.operation_generations[0].wrapping_add(1);
        }
        fn corrupt_parent(&mut self) {
            self.operations[1].parent = Some(BlockId::new(9, self.generation));
        }
        fn corrupt_operand_list_range(&mut self) {
            self.operations[2].operands.len = u32::MAX;
        }
        fn corrupt_invalid_result_reference(&mut self) {
            let operation =
                OperationId::with_owner(1, self.operation_generations[1], self.identity.0);
            self.values.0[self.operations[2].operands.start as usize] =
                ValueReference::Resolved(ValueId::OperationResult {
                    operation,
                    result: 1,
                });
        }
        fn corrupt_stale_result_reference(&mut self) {
            let operation = OperationId::with_owner(
                1,
                self.operation_generations[1].wrapping_add(1),
                self.identity.0,
            );
            self.values.0[self.operations[2].operands.start as usize] =
                ValueReference::Resolved(ValueId::OperationResult {
                    operation,
                    result: 0,
                });
        }
        fn remove_region_from_parent(&mut self) {
            self.operations[0].regions.len = 0;
        }
        fn remove_block_from_parent(&mut self) {
            self.regions[0].blocks.len = 0;
        }
        fn duplicate_root_membership(&mut self) {
            let root = self.operation_lists.get(self.roots).unwrap()[0];
            self.roots = self.operation_lists.push(&[root, root]);
        }
        fn remove_root_membership(&mut self) {
            self.roots = self.operation_lists.push(&[]);
        }
        fn mismatch_root_parent(&mut self) {
            self.operations[0].parent = Some(BlockId::new(0, self.generation));
        }
    }
    fn lower(bytes: &[u8]) -> Document {
        let parsed = ParsedFile::parse(bytes.to_vec()).unwrap();
        lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
            .document
            .unwrap()
    }
    #[test]
    fn validator_rejects_stale_ids_and_relationships() {
        let bytes = include_bytes!("../../../tests/corpus/mlir-22.1/semantic-proving/valid.mlir");
        let mut stale = lower(bytes);
        stale.corrupt_root_generation();
        assert!(matches!(
            stale.validate(),
            Err(ValidationError::StaleOperation(_))
        ));
        let mut relationship = lower(bytes);
        relationship.corrupt_parent();
        assert!(matches!(
            relationship.validate(),
            Err(ValidationError::StaleBlock(_))
        ));
        let mut malformed_list = lower(bytes);
        assert_eq!(
            malformed_list.validate(),
            Ok(()),
            "the uncorrupted fixture remains valid"
        );
        malformed_list.corrupt_operand_list_range();
        assert_eq!(malformed_list.validate(), Err(ValidationError::InvalidList));
        let mut invalid_result = lower(bytes);
        invalid_result.corrupt_invalid_result_reference();
        assert_eq!(
            invalid_result.validate(),
            Err(ValidationError::InvalidValue)
        );
        let mut stale_result = lower(bytes);
        stale_result.corrupt_stale_result_reference();
        assert_eq!(stale_result.validate(), Err(ValidationError::InvalidValue));

        let mut missing_region_backlink = lower(bytes);
        missing_region_backlink.remove_region_from_parent();
        assert_eq!(
            missing_region_backlink.validate(),
            Err(ValidationError::ParentChildMismatch)
        );
        let mut missing_block_backlink = lower(bytes);
        missing_block_backlink.remove_block_from_parent();
        assert_eq!(
            missing_block_backlink.validate(),
            Err(ValidationError::ParentChildMismatch)
        );
        let mut duplicate_root = lower(bytes);
        duplicate_root.duplicate_root_membership();
        assert_eq!(
            duplicate_root.validate(),
            Err(ValidationError::ParentChildMismatch)
        );
        let mut missing_root = lower(bytes);
        missing_root.remove_root_membership();
        assert_eq!(
            missing_root.validate(),
            Err(ValidationError::ParentChildMismatch)
        );
        let mut mismatched_root = lower(bytes);
        mismatched_root.mismatch_root_parent();
        assert_eq!(
            mismatched_root.validate(),
            Err(ValidationError::ParentChildMismatch)
        );
    }
}
