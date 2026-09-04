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

mod edit;
mod lowering;
mod values;
mod verify;

pub use lowering::{lower_with_dialect_registry, lower_with_dialect_registry_and_retention};
pub(crate) use values::split_registered_types;
pub(crate) use verify::{
    verify_builtin_module, verify_cf_br, verify_cf_cond_br, verify_func_call, verify_func_func,
    verify_func_return,
};

use values::*;
use verify::*;

static NEXT_DOCUMENT_IDENTITY: OnceLock<Mutex<u128>> = OnceLock::new();
static LIVE_DOCUMENT_IDENTITIES: OnceLock<Mutex<HashMap<u128, Weak<DocumentIdentity>>>> =
    OnceLock::new();

#[derive(Debug)]
struct DocumentIdentity(u128);

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
    mnemonic: &'a str,
    leading_symbol: Option<&'a str>,
    visibility: Option<&'a str>,
    arguments: Vec<RegisteredArgument<'a>>,
    function_results: Option<&'a str>,
    function_type: Option<&'a str>,
}

pub struct RegisteredArgument<'a> {
    spelling: &'a str,
    attributes: Option<&'a str>,
}

impl<'a> RegisteredLoweringContext<'a> {
    pub fn spelling(&self) -> &'a str {
        self.spelling
    }
    pub fn mnemonic(&self) -> &'a str {
        self.mnemonic
    }
    pub fn leading_symbol(&self) -> Option<&'a str> {
        self.leading_symbol
    }
    pub fn visibility(&self) -> Option<&'a str> {
        self.visibility
    }
    pub fn arguments(&self) -> impl Iterator<Item = (&'a str, Option<&'a str>)> + '_ {
        self.arguments
            .iter()
            .map(|argument| (argument.spelling, argument.attributes))
    }
    pub fn function_results(&self) -> Option<&'a str> {
        self.function_results
    }
    pub fn function_type(&self) -> Option<&'a str> {
        self.function_type
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
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY)
            .document
            .unwrap()
    }

    #[test]
    fn successor_rewiring_rejects_a_missing_target_type_slot() {
        let mut document = lower(
            br#""outer"() ({
^entry(%value: i32):
  "branch"() [^target : (%value : i32)] : () -> ()
^target(%argument: i32):
  "sink"() : () -> ()
}) : () -> ()"#,
        );
        let outer = document.root_operations()[0];
        let blocks = document
            .region(document.operation_regions(outer).unwrap()[0])
            .unwrap()
            .blocks(&document)
            .unwrap()
            .to_vec();
        let branch = document.block_operations(blocks[0]).unwrap()[0];
        let value = ValueId::BlockArgument {
            block: blocks[0],
            argument: 0,
        };
        let registry = DialectRegistry::EMPTY;
        let mut editor = document.edit(&registry).unwrap();
        editor.working.blocks[blocks[1].index()].argument_types =
            editor.working.types_lists.push(&[]);
        assert_eq!(
            editor.rewire_successor_argument(branch, 0, 0, value),
            Err(EditError::TypeMismatch)
        );
    }

    #[test]
    fn commit_checks_successor_types_after_bulk_use_replacement() {
        let mut document = lower(
            br#""outer"() ({
^entry:
  %from = "from"() : () -> i64
  %to = "to"() : () -> i64
  "branch"() [^target : (%from : i64)] : () -> ()
^target(%argument: i64):
  "sink"() : () -> ()
}) : () -> ()"#,
        );
        let operations = document.operations().collect::<Vec<_>>();
        let from = document
            .operation(operations[1])
            .unwrap()
            .result(operations[1], 0)
            .unwrap();
        let to = document
            .operation(operations[2])
            .unwrap()
            .result(operations[2], 0)
            .unwrap();
        let outer = document.root_operations()[0];
        let target = document
            .region(document.operation_regions(outer).unwrap()[0])
            .unwrap()
            .blocks(&document)
            .unwrap()[1];
        let revision = document.revision();
        let registry = DialectRegistry::EMPTY;
        let mut editor = document.edit(&registry).unwrap();
        let i32_type = editor.intern_type_spec(&TypeSpec {
            spelling: "i32".into(),
            value: TypeValue::Integer {
                width: 32,
                signedness: None,
            },
        });
        editor.working.blocks[target.index()].argument_types =
            editor.working.types_lists.push(&[i32_type]);
        assert_eq!(editor.replace_all_uses(from, to).unwrap(), 1);
        assert!(matches!(
            editor.commit(),
            Err(EditError::Semantic(SemanticVerificationError::Operation {
                message,
                ..
            })) if message == "successor argument types do not match the target block arguments"
        ));
        assert_eq!(document.revision(), revision);
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
