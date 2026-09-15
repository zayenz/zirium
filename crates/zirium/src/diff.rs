//! Structural and dataflow comparison for two semantic MLIR documents.
//!
//! A [`Diff`] borrows both immutable documents and the registry used to
//! interpret them. Callers must use the same registry semantics that were used
//! while lowering both documents.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use serde::Serialize;

use crate::{
    dialect::DialectRegistry,
    printer::compare::Correspondence,
    semantic::{BlockId, Document, OperationId, RegionId},
    source::TextRange,
};

static NEXT_DIFF_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiffOptions {
    pub compare_locations: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiffLimits {
    pub max_work: usize,
    pub max_changes: usize,
}

impl Default for DiffLimits {
    fn default() -> Self {
        Self {
            max_work: 10_000_000,
            max_changes: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffSide {
    Before,
    After,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChangeKind {
    Added,
    Removed,
    Modified,
    Moved,
}

impl ChangeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Removed => "removed",
            Self::Modified => "modified",
            Self::Moved => "moved",
        }
    }
}

impl std::str::FromStr for ChangeKind {
    type Err = DiffError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "added" => Ok(Self::Added),
            "removed" => Ok(Self::Removed),
            "modified" => Ok(Self::Modified),
            "moved" => Ok(Self::Moved),
            _ => Err(DiffError::InvalidChangeKind(value.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeField {
    Name,
    Operands,
    ResultTypes,
    FunctionType,
    Attributes,
    Properties,
    Successors,
    Regions,
    Location,
    Position,
}

impl ChangeField {
    pub const ALL: [Self; 10] = [
        Self::Name,
        Self::Operands,
        Self::ResultTypes,
        Self::FunctionType,
        Self::Attributes,
        Self::Properties,
        Self::Successors,
        Self::Regions,
        Self::Location,
        Self::Position,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Operands => "operands",
            Self::ResultTypes => "result_types",
            Self::FunctionType => "function_type",
            Self::Attributes => "attributes",
            Self::Properties => "properties",
            Self::Successors => "successors",
            Self::Regions => "regions",
            Self::Location => "location",
            Self::Position => "position",
        }
    }
}

impl std::str::FromStr for ChangeField {
    type Err = DiffError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|field| field.as_str() == value)
            .ok_or_else(|| DiffError::InvalidField(value.to_owned()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChangeId {
    index: u32,
    diff: u64,
}

impl ChangeId {
    pub fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DiffOperation {
    operation: OperationId,
    side: DiffSide,
    diff: u64,
}

impl DiffOperation {
    pub fn side(self) -> DiffSide {
        self.side
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Endpoint {
    pub name: String,
    pub symbol_path: Vec<String>,
    pub path: String,
    #[serde(serialize_with = "serialize_range")]
    pub range: Option<TextRange>,
}

fn serialize_range<S>(range: &Option<TextRange>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    range
        .map(|range| [range.start(), range.end()])
        .serialize(serializer)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FieldValue {
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FieldDifference {
    pub path: String,
    pub before: FieldValue,
    pub after: FieldValue,
    pub comparison: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    id: ChangeId,
    kind: ChangeKind,
    moved: bool,
    before: Option<OperationId>,
    after: Option<OperationId>,
    fields: Vec<ChangeField>,
    details: Vec<FieldDifference>,
}

impl Change {
    pub fn id(&self) -> ChangeId {
        self.id
    }
    pub fn kind(&self) -> ChangeKind {
        self.kind
    }
    pub fn moved(&self) -> bool {
        self.moved
    }
    pub fn fields(&self) -> &[ChangeField] {
        &self.fields
    }
    pub fn details(&self) -> &[FieldDifference] {
        &self.details
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DiffStatistics {
    pub before_operations: usize,
    pub after_operations: usize,
    pub matched_operations: usize,
    pub ambiguous_groups: usize,
    pub bounded_fallback_groups: usize,
    pub opaque_before: usize,
    pub opaque_after: usize,
    pub work_units: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffError {
    InvalidLimits,
    IncompleteInput { side: DiffSide },
    InvalidStructure { side: DiffSide, message: String },
    UnsupportedResource { side: DiffSide },
    WorkLimitExceeded,
    ChangeLimitExceeded,
    InvalidHandle,
    InvalidField(String),
    InvalidChangeKind(String),
}

impl fmt::Display for DiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => f.write_str("diff limits must be positive"),
            Self::IncompleteInput { side } => {
                write!(f, "{side:?} input is semantically incomplete")
            }
            Self::InvalidStructure { side, message } => {
                write!(f, "{side:?} input has invalid structure: {message}")
            }
            Self::UnsupportedResource { side } => {
                write!(f, "{side:?} input contains a resource-backed attribute")
            }
            Self::WorkLimitExceeded => f.write_str("semantic diff work limit exceeded"),
            Self::ChangeLimitExceeded => f.write_str("semantic diff change limit exceeded"),
            Self::InvalidHandle => f.write_str("diff handle belongs to another comparison"),
            Self::InvalidField(field) => write!(f, "unknown changed field `{field}`"),
            Self::InvalidChangeKind(kind) => write!(f, "unknown change kind `{kind}`"),
        }
    }
}

impl std::error::Error for DiffError {}

pub struct Diff<'a> {
    before: &'a Document,
    after: &'a Document,
    registry: &'a DialectRegistry,
    identity: u64,
    correspondence: Correspondence,
    changes: Vec<Change>,
    before_paths: HashMap<OperationId, String>,
    after_paths: HashMap<OperationId, String>,
    options: DiffOptions,
    statistics: DiffStatistics,
    diagnostics: Vec<String>,
}

pub fn compare<'a>(
    before: &'a Document,
    after: &'a Document,
    registry: &'a DialectRegistry,
    options: DiffOptions,
    limits: DiffLimits,
) -> Result<Diff<'a>, DiffError> {
    if limits.max_work == 0 || limits.max_changes == 0 {
        return Err(DiffError::InvalidLimits);
    }
    validate_input(before, DiffSide::Before)?;
    validate_input(after, DiffSide::After)?;

    let identity = NEXT_DIFF_ID.fetch_add(1, Ordering::Relaxed).max(1);
    let mut builder = Matcher::new(before, after, registry, limits.max_work);
    builder.match_list(before.root_operations(), after.root_operations())?;
    let (operations, regions, blocks, work, ambiguous_groups, bounded_fallback_groups, diagnostics) =
        builder.finish();
    let correspondence = Correspondence::from_maps(operations, regions, blocks);
    let before_paths = operation_paths(before);
    let after_paths = operation_paths(after);
    let mut diff = Diff {
        before,
        after,
        registry,
        identity,
        correspondence,
        changes: Vec::new(),
        before_paths,
        after_paths,
        options,
        statistics: DiffStatistics {
            before_operations: before.operations().count(),
            after_operations: after.operations().count(),
            matched_operations: 0,
            ambiguous_groups,
            bounded_fallback_groups,
            work_units: work,
            ..DiffStatistics::default()
        },
        diagnostics,
    };
    diff.statistics.matched_operations = diff.correspondence.operations.len();
    diff.construct_changes(limits.max_changes)?;
    Ok(diff)
}

fn validate_input(document: &Document, side: DiffSide) -> Result<(), DiffError> {
    if !document.is_semantically_complete() {
        return Err(DiffError::IncompleteInput { side });
    }
    document
        .validate_structure()
        .map_err(|error| DiffError::InvalidStructure {
            side,
            message: error.to_string(),
        })?;
    if document
        .comparison_coverage()
        .unrepresented_file_metadata
        .is_some()
    {
        return Err(DiffError::UnsupportedResource { side });
    }
    if document_has_resource(document) {
        return Err(DiffError::UnsupportedResource { side });
    }
    Ok(())
}

fn document_has_resource(document: &Document) -> bool {
    document.operations().any(|operation| {
        document
            .attribute_entries(operation)
            .into_iter()
            .flatten()
            .any(|(_, id)| attribute_has_resource(document.attribute_value(id)))
            || document
                .operation_properties(operation)
                .into_iter()
                .flatten()
                .any(|(_, id)| attribute_has_resource(document.attribute_value(*id)))
            || document
                .result_types(operation)
                .into_iter()
                .flatten()
                .any(|id| type_has_resource(document.type_value(*id)))
            || document
                .function_type(operation)
                .is_some_and(|id| type_has_resource(document.type_value(id)))
    })
}

fn type_has_resource(value: Option<&crate::semantic::TypeValue>) -> bool {
    use crate::semantic::{MemRefLayout, TypeValue};
    value.is_some_and(|value| match value {
        TypeValue::Complex(value) => type_has_resource(Some(value)),
        TypeValue::Tuple(values) => values.iter().any(|value| type_has_resource(Some(value))),
        TypeValue::Tensor {
            element, encoding, ..
        } => {
            type_has_resource(Some(element))
                || encoding
                    .as_deref()
                    .is_some_and(|value| attribute_has_resource(Some(value)))
        }
        TypeValue::Vector { element, .. } => type_has_resource(Some(element)),
        TypeValue::MemRef {
            element,
            layout,
            memory_space,
            ..
        } => {
            type_has_resource(Some(element))
                || memory_space
                    .as_deref()
                    .is_some_and(|value| attribute_has_resource(Some(value)))
                || layout.as_ref().is_some_and(|layout| match layout {
                    MemRefLayout::Opaque { parameters, .. } => parameters
                        .iter()
                        .any(|value| attribute_has_resource(Some(value))),
                    MemRefLayout::Attribute(value) => attribute_has_resource(Some(value)),
                    _ => false,
                })
        }
        TypeValue::Function { inputs, results } => inputs
            .iter()
            .chain(results)
            .any(|value| type_has_resource(Some(value))),
        _ => false,
    })
}

fn attribute_has_resource(value: Option<&crate::semantic::AttributeValue>) -> bool {
    use crate::semantic::{AttributeValue as A, LargeAttributeValue as L, MemRefLayout, TypeValue};
    fn ty(value: &TypeValue) -> bool {
        match value {
            TypeValue::Complex(value) => ty(value),
            TypeValue::Tuple(values) => values.iter().any(ty),
            TypeValue::Tensor {
                element, encoding, ..
            } => ty(element) || encoding.as_deref().is_some_and(attr),
            TypeValue::Vector { element, .. } => ty(element),
            TypeValue::MemRef {
                element,
                layout,
                memory_space,
                ..
            } => {
                ty(element)
                    || memory_space.as_deref().is_some_and(attr)
                    || layout.as_ref().is_some_and(|layout| match layout {
                        MemRefLayout::Opaque { parameters, .. } => parameters.iter().any(attr),
                        MemRefLayout::Attribute(value) => attr(value),
                        _ => false,
                    })
            }
            TypeValue::Function { inputs, results } => inputs.iter().chain(results).any(ty),
            _ => false,
        }
    }
    fn attr(value: &A) -> bool {
        match value {
            A::Large(L::Resource(_)) => true,
            A::Type(value) => ty(value),
            A::Array(values)
            | A::DenseArray {
                elements: values, ..
            } => values.iter().any(attr),
            A::Dictionary(values) => values.iter().any(|(_, value)| attr(value)),
            _ => false,
        }
    }
    value.is_some_and(attr)
}

impl Diff<'_> {
    pub fn len(&self) -> usize {
        self.changes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
    pub fn change_ids(&self) -> impl ExactSizeIterator<Item = ChangeId> + '_ {
        self.changes.iter().map(Change::id)
    }
    pub fn change(&self, id: ChangeId) -> Result<&Change, DiffError> {
        if id.diff != self.identity {
            return Err(DiffError::InvalidHandle);
        }
        self.changes.get(id.index()).ok_or(DiffError::InvalidHandle)
    }
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }
    pub fn options(&self) -> DiffOptions {
        self.options
    }
    pub fn statistics(&self) -> &DiffStatistics {
        &self.statistics
    }
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }
    pub fn registry(&self) -> &DialectRegistry {
        self.registry
    }
    pub fn document(&self, side: DiffSide) -> &Document {
        match side {
            DiffSide::Before => self.before,
            DiffSide::After => self.after,
        }
    }
    pub fn endpoint_id(
        &self,
        id: ChangeId,
        side: DiffSide,
    ) -> Result<Option<OperationId>, DiffError> {
        let change = self.change(id)?;
        Ok(match side {
            DiffSide::Before => change.before,
            DiffSide::After => change.after,
        })
    }
    pub fn representative_id(&self, id: ChangeId) -> Result<(DiffSide, OperationId), DiffError> {
        let change = self.change(id)?;
        change
            .after
            .map(|operation| (DiffSide::After, operation))
            .or_else(|| change.before.map(|operation| (DiffSide::Before, operation)))
            .ok_or(DiffError::InvalidHandle)
    }
    pub fn before_operation(&self, id: ChangeId) -> Result<Option<DiffOperation>, DiffError> {
        Ok(self.change(id)?.before.map(|operation| DiffOperation {
            operation,
            side: DiffSide::Before,
            diff: self.identity,
        }))
    }
    pub fn after_operation(&self, id: ChangeId) -> Result<Option<DiffOperation>, DiffError> {
        Ok(self.change(id)?.after.map(|operation| DiffOperation {
            operation,
            side: DiffSide::After,
            diff: self.identity,
        }))
    }
    pub fn operation_id(&self, operation: DiffOperation) -> Result<OperationId, DiffError> {
        (operation.diff == self.identity)
            .then_some(operation.operation)
            .ok_or(DiffError::InvalidHandle)
    }
    pub fn scoped_operation(
        &self,
        side: DiffSide,
        operation: OperationId,
    ) -> Result<DiffOperation, DiffError> {
        if self.document(side).operation(operation).is_none() {
            return Err(DiffError::InvalidHandle);
        }
        Ok(DiffOperation {
            operation,
            side,
            diff: self.identity,
        })
    }
    pub fn paired_operation(
        &self,
        operation: DiffOperation,
    ) -> Result<Option<DiffOperation>, DiffError> {
        if operation.diff != self.identity {
            return Err(DiffError::InvalidHandle);
        }
        let (side, paired) = match operation.side {
            DiffSide::Before => (
                DiffSide::After,
                self.correspondence
                    .operations
                    .get(&operation.operation)
                    .copied(),
            ),
            DiffSide::After => (
                DiffSide::Before,
                self.correspondence
                    .operations
                    .iter()
                    .find_map(|(before, after)| (*after == operation.operation).then_some(*before)),
            ),
        };
        Ok(paired.map(|operation| DiffOperation {
            operation,
            side,
            diff: self.identity,
        }))
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.serializable_changes())
    }

    pub fn selection_to_json(
        &self,
        ids: &[ChangeId],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut changes = Vec::with_capacity(ids.len());
        for &id in ids {
            changes.push(self.serializable_change(self.change(id)?));
        }
        Ok(serde_json::to_string_pretty(&changes)?)
    }

    pub fn to_text(&self) -> String {
        if self.changes.is_empty() {
            return "No structural changes.\n".to_owned();
        }
        let mut counts = [0usize; 4];
        for change in &self.changes {
            counts[match change.kind {
                ChangeKind::Added => 0,
                ChangeKind::Removed => 1,
                ChangeKind::Modified => 2,
                ChangeKind::Moved => 3,
            }] += 1;
        }
        let mut output = format!(
            "{} modified, {} added, {} removed, {} moved\n\n",
            counts[2], counts[0], counts[1], counts[3]
        );
        for change in &self.changes {
            let endpoint = change.after.or(change.before).expect("change has endpoint");
            let document = if change.after.is_some() {
                self.after
            } else {
                self.before
            };
            output.push_str(change.kind.as_str());
            output.push(' ');
            output.push_str(document.operation_name(endpoint).unwrap_or("<invalid>"));
            if change.moved && change.kind != ChangeKind::Moved {
                output.push_str(" (moved)");
            }
            output.push('\n');
            for detail in &change.details {
                output.push_str("  ");
                output.push_str(
                    detail
                        .path
                        .trim_start_matches('/')
                        .replace('/', ".")
                        .as_str(),
                );
                output.push_str(": ");
                output.push_str(detail.before.value.as_deref().unwrap_or("<absent>"));
                output.push_str(" -> ");
                output.push_str(detail.after.value.as_deref().unwrap_or("<absent>"));
                output.push('\n');
            }
        }
        output
    }

    pub fn selection_to_text(&self, ids: &[ChangeId]) -> Result<String, DiffError> {
        if ids.is_empty() {
            return Ok(if self.is_empty() {
                "No structural changes.\n".to_owned()
            } else {
                "No selected changes.\n".to_owned()
            });
        }
        let mut counts = [0usize; 4];
        let mut records = String::new();
        for &id in ids {
            let change = self.change(id)?;
            counts[match change.kind {
                ChangeKind::Added => 0,
                ChangeKind::Removed => 1,
                ChangeKind::Modified => 2,
                ChangeKind::Moved => 3,
            }] += 1;
            let operation = change.after.or(change.before).expect("change endpoint");
            let document = if change.after.is_some() {
                self.after
            } else {
                self.before
            };
            records.push_str(change.kind.as_str());
            records.push(' ');
            records.push_str(document.operation_name(operation).unwrap_or("<invalid>"));
            records.push('\n');
            for detail in &change.details {
                records.push_str("  ");
                records.push_str(detail.path.trim_start_matches('/'));
                records.push_str(": ");
                records.push_str(detail.before.value.as_deref().unwrap_or("<absent>"));
                records.push_str(" -> ");
                records.push_str(detail.after.value.as_deref().unwrap_or("<absent>"));
                records.push('\n');
            }
        }
        Ok(format!(
            "{} modified, {} added, {} removed, {} moved\n\n{records}",
            counts[2], counts[0], counts[1], counts[3]
        ))
    }

    fn serializable_changes(&self) -> Vec<SerializableChange> {
        self.changes
            .iter()
            .map(|change| self.serializable_change(change))
            .collect()
    }

    fn serializable_change(&self, change: &Change) -> SerializableChange {
        SerializableChange {
            id: change.id.index(),
            kind: change.kind,
            moved: change.moved,
            before: change
                .before
                .map(|op| self.endpoint(self.before, op, DiffSide::Before)),
            after: change
                .after
                .map(|op| self.endpoint(self.after, op, DiffSide::After)),
            fields: change.fields.clone(),
            details: change.details.clone(),
        }
    }

    fn construct_changes(&mut self, max_changes: usize) -> Result<(), DiffError> {
        let reverse: HashMap<_, _> = self
            .correspondence
            .operations
            .iter()
            .map(|(&a, &b)| (b, a))
            .collect();
        let moved = moved_operations(
            self.before,
            self.after,
            &self.correspondence.operations,
            &self.correspondence.blocks,
        );
        for after in structural_preorder(self.after) {
            let before = reverse.get(&after).copied();
            let (kind, fields, details, did_move) = if let Some(before) = before {
                let mut fields = Vec::new();
                let equal = self.correspondence.equal_operation_fields(
                    before,
                    after,
                    self.before,
                    self.after,
                );
                for (field, same) in ChangeField::ALL[..8].iter().zip(equal) {
                    if !same {
                        fields.push(*field);
                    }
                }
                if self.options.compare_locations
                    && !self
                        .correspondence
                        .equal_locations(before, after, self.before, self.after)
                {
                    fields.push(ChangeField::Location);
                }
                let did_move = moved.contains(&before);
                if did_move {
                    fields.push(ChangeField::Position);
                }
                if fields.is_empty() {
                    continue;
                }
                let non_position = fields.iter().any(|field| *field != ChangeField::Position);
                let kind = if non_position {
                    ChangeKind::Modified
                } else {
                    ChangeKind::Moved
                };
                let details = fields
                    .iter()
                    .map(|field| self.field_detail(*field, before, after))
                    .collect();
                (kind, fields, details, did_move)
            } else {
                (ChangeKind::Added, Vec::new(), Vec::new(), false)
            };
            self.push_change(
                max_changes,
                kind,
                did_move,
                before,
                Some(after),
                fields,
                details,
            )?;
        }
        for before in structural_preorder(self.before) {
            if !self.correspondence.operations.contains_key(&before) {
                self.push_change(
                    max_changes,
                    ChangeKind::Removed,
                    false,
                    Some(before),
                    None,
                    Vec::new(),
                    Vec::new(),
                )?;
            }
        }
        Ok(())
    }

    fn push_change(
        &mut self,
        max: usize,
        kind: ChangeKind,
        moved: bool,
        before: Option<OperationId>,
        after: Option<OperationId>,
        fields: Vec<ChangeField>,
        details: Vec<FieldDifference>,
    ) -> Result<(), DiffError> {
        if self.changes.len() >= max {
            return Err(DiffError::ChangeLimitExceeded);
        }
        let id = ChangeId {
            index: self.changes.len() as u32,
            diff: self.identity,
        };
        self.changes.push(Change {
            id,
            kind,
            moved,
            before,
            after,
            fields,
            details,
        });
        Ok(())
    }

    fn field_detail(
        &self,
        field: ChangeField,
        before: OperationId,
        after: OperationId,
    ) -> FieldDifference {
        let display = |document: &Document, operation: OperationId, field| -> String {
            match field {
                ChangeField::Name => document
                    .operation_name(operation)
                    .unwrap_or("<invalid>")
                    .to_owned(),
                ChangeField::Operands => {
                    format!("{:?}", document.operands(operation).unwrap_or(&[]))
                }
                ChangeField::ResultTypes => document
                    .result_types(operation)
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|id| document.type_spelling(*id))
                    .collect::<Vec<_>>()
                    .join(", "),
                ChangeField::FunctionType => document
                    .function_type(operation)
                    .and_then(|id| document.type_spelling(id))
                    .unwrap_or("")
                    .to_owned(),
                ChangeField::Attributes => format_entries(document.attributes(operation)),
                ChangeField::Properties => format_entries(document.properties(operation)),
                ChangeField::Successors => {
                    format!("{:?}", document.successors(operation).unwrap_or(&[]))
                }
                ChangeField::Regions => format!(
                    "{} region(s)",
                    document.operation_regions(operation).map_or(0, <[_]>::len)
                ),
                ChangeField::Location => document
                    .operation_location(operation)
                    .flatten()
                    .unwrap_or("<none>")
                    .to_owned(),
                ChangeField::Position => {
                    format!("{}", sibling_position(document, operation).unwrap_or(0))
                }
            }
        };
        FieldDifference {
            path: format!("/{}", field.as_str()),
            before: FieldValue {
                present: true,
                value: Some(display(self.before, before, field)),
            },
            after: FieldValue {
                present: true,
                value: Some(display(self.after, after, field)),
            },
            comparison: "represented_value",
        }
    }

    fn endpoint(&self, document: &Document, operation: OperationId, side: DiffSide) -> Endpoint {
        Endpoint {
            name: document
                .operation_name(operation)
                .unwrap_or("<invalid>")
                .to_owned(),
            symbol_path: symbol_path(document, operation),
            path: match side {
                DiffSide::Before => &self.before_paths,
                DiffSide::After => &self.after_paths,
            }
            .get(&operation)
            .cloned()
            .unwrap_or_default(),
            range: document.operation_source_range(operation),
        }
    }
}

#[derive(Serialize)]
struct SerializableChange {
    id: usize,
    kind: ChangeKind,
    moved: bool,
    before: Option<Endpoint>,
    after: Option<Endpoint>,
    fields: Vec<ChangeField>,
    details: Vec<FieldDifference>,
}

fn format_entries<'a>(entries: Option<impl Iterator<Item = (&'a str, &'a str)>>) -> String {
    entries
        .into_iter()
        .flatten()
        .map(|(name, value)| format!("{name} = {value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn symbol_path(document: &Document, mut operation: OperationId) -> Vec<String> {
    let mut result = Vec::new();
    loop {
        if let Some(name) = document.operation_symbol_name(operation) {
            result.push(name);
        }
        let Some(block) = document
            .operation(operation)
            .and_then(|operation| operation.parent_block())
        else {
            break;
        };
        let Some(region) = document.block(block).map(|block| block.parent_region()) else {
            break;
        };
        let Some(parent) = document
            .region(region)
            .map(|region| region.parent_operation())
        else {
            break;
        };
        operation = parent;
    }
    result.reverse();
    result
}

fn operation_paths(document: &Document) -> HashMap<OperationId, String> {
    fn visit(
        document: &Document,
        operation: OperationId,
        path: String,
        out: &mut HashMap<OperationId, String>,
    ) {
        out.insert(operation, path.clone());
        for (ri, &region) in document
            .operation_regions(operation)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            for (bi, &block) in document
                .region(region)
                .and_then(|region| region.blocks(document))
                .unwrap_or(&[])
                .iter()
                .enumerate()
            {
                for (oi, &child) in document
                    .block_operations(block)
                    .unwrap_or(&[])
                    .iter()
                    .enumerate()
                {
                    visit(
                        document,
                        child,
                        format!("{path}/regions/{ri}/blocks/{bi}/operations/{oi}"),
                        out,
                    );
                }
            }
        }
    }
    let mut out = HashMap::new();
    for (index, &root) in document.root_operations().iter().enumerate() {
        visit(document, root, format!("/operations/{index}"), &mut out);
    }
    out
}

fn structural_preorder(document: &Document) -> Vec<OperationId> {
    let paths = operation_paths(document);
    let mut entries: Vec<_> = paths.into_iter().collect();
    entries.sort_by(|a, b| path_indices(&a.1).cmp(&path_indices(&b.1)));
    entries
        .into_iter()
        .map(|(operation, _)| operation)
        .collect()
}

fn path_indices(path: &str) -> Vec<usize> {
    path.split('/')
        .filter_map(|part| part.parse().ok())
        .collect()
}

fn sibling_position(document: &Document, operation: OperationId) -> Option<usize> {
    let block = document.operation(operation)?.parent_block()?;
    document
        .block_operations(block)?
        .iter()
        .position(|item| *item == operation)
}

fn moved_operations(
    before: &Document,
    after: &Document,
    map: &HashMap<OperationId, OperationId>,
    block_map: &HashMap<BlockId, BlockId>,
) -> HashSet<OperationId> {
    let mut moved = HashSet::new();
    fn check(
        before_list: &[OperationId],
        after_list: &[OperationId],
        map: &HashMap<OperationId, OperationId>,
        moved: &mut HashSet<OperationId>,
    ) {
        let after_positions: HashMap<_, _> = after_list
            .iter()
            .enumerate()
            .map(|(i, op)| (*op, i))
            .collect();
        let matched: Vec<_> = before_list
            .iter()
            .filter_map(|before| Some((*before, *after_positions.get(map.get(before)?)?)))
            .collect();
        for (i, &(operation, position)) in matched.iter().enumerate() {
            if matched[..i].iter().any(|(_, earlier)| *earlier > position)
                || matched[i + 1..].iter().any(|(_, later)| *later < position)
            {
                moved.insert(operation);
            }
        }
    }
    check(
        before.root_operations(),
        after.root_operations(),
        map,
        &mut moved,
    );
    for (&before_block, &after_block) in block_map {
        check(
            before.block_operations(before_block).unwrap_or(&[]),
            after.block_operations(after_block).unwrap_or(&[]),
            map,
            &mut moved,
        );
    }
    moved
}

struct Matcher<'a> {
    before: &'a Document,
    after: &'a Document,
    registry: &'a DialectRegistry,
    operations: HashMap<OperationId, OperationId>,
    regions: HashMap<RegionId, RegionId>,
    blocks: HashMap<BlockId, BlockId>,
    remaining: usize,
    initial: usize,
    ambiguous: usize,
    bounded_fallback: usize,
    diagnostics: Vec<String>,
}

impl<'a> Matcher<'a> {
    fn new(
        before: &'a Document,
        after: &'a Document,
        registry: &'a DialectRegistry,
        work: usize,
    ) -> Self {
        Self {
            before,
            after,
            registry,
            operations: HashMap::new(),
            regions: HashMap::new(),
            blocks: HashMap::new(),
            remaining: work,
            initial: work,
            ambiguous: 0,
            bounded_fallback: 0,
            diagnostics: Vec::new(),
        }
    }
    fn charge(&mut self, amount: usize) -> Result<(), DiffError> {
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or(DiffError::WorkLimitExceeded)?;
        Ok(())
    }
    fn finish(
        self,
    ) -> (
        HashMap<OperationId, OperationId>,
        HashMap<RegionId, RegionId>,
        HashMap<BlockId, BlockId>,
        usize,
        usize,
        usize,
        Vec<String>,
    ) {
        (
            self.operations,
            self.regions,
            self.blocks,
            self.initial - self.remaining,
            self.ambiguous,
            self.bounded_fallback,
            self.diagnostics,
        )
    }

    fn match_list(
        &mut self,
        before: &[OperationId],
        after: &[OperationId],
    ) -> Result<(), DiffError> {
        self.charge(before.len() + after.len())?;
        let mut before_left: HashSet<_> = before.iter().copied().collect();
        let mut after_left: HashSet<_> = after.iter().copied().collect();

        self.match_symbol_anchors(before, after, &mut before_left, &mut after_left)?;

        let before_keys = grouped(before.iter().copied(), |op| exact_key(self.before, op));
        let after_keys = grouped(after.iter().copied(), |op| exact_key(self.after, op));
        for (key, left) in &before_keys {
            if left.len() == 1 && after_keys.get(key).is_some_and(|right| right.len() == 1) {
                let right = after_keys[key][0];
                self.pair(left[0], right);
                before_left.remove(&left[0]);
                after_left.remove(&right);
            }
        }
        let before_remaining: Vec<_> = before
            .iter()
            .copied()
            .filter(|op| before_left.contains(op))
            .collect();
        let after_remaining: Vec<_> = after
            .iter()
            .copied()
            .filter(|op| after_left.contains(op))
            .collect();
        if before_remaining.len() == after_remaining.len()
            && before_remaining
                .iter()
                .zip(&after_remaining)
                .all(|(&a, &b)| exact_key(self.before, a) == exact_key(self.after, b))
        {
            for (&a, &b) in before_remaining.iter().zip(&after_remaining) {
                self.pair(a, b);
                before_left.remove(&a);
                after_left.remove(&b);
            }
        } else {
            let before_shapes = grouped(before_remaining.iter().copied(), |op| {
                shape_key(self.before, op)
            });
            let after_shapes = grouped(after_remaining.iter().copied(), |op| {
                shape_key(self.after, op)
            });
            for (key, left) in before_shapes {
                let Some(right) = after_shapes.get(&key) else {
                    continue;
                };
                let candidates = left.len().saturating_mul(right.len());
                if candidates > 65_536 {
                    self.bounded_fallback += 1;
                    continue;
                }
                self.charge(candidates)?;
                if left.len() == 1 && right.len() == 1 {
                    self.pair(left[0], right[0]);
                    before_left.remove(&left[0]);
                    after_left.remove(&right[0]);
                } else if !left.is_empty() {
                    self.ambiguous += 1;
                }
            }
        }
        let pairs: Vec<_> = self
            .operations
            .iter()
            .filter_map(|(&a, &b)| (before.contains(&a) && after.contains(&b)).then_some((a, b)))
            .collect();
        for (before_op, after_op) in pairs {
            self.match_children(before_op, after_op)?;
        }
        Ok(())
    }
    fn pair(&mut self, before: OperationId, after: OperationId) {
        self.operations.insert(before, after);
    }

    fn match_symbol_anchors(
        &mut self,
        before: &[OperationId],
        after: &[OperationId],
        before_left: &mut HashSet<OperationId>,
        after_left: &mut HashSet<OperationId>,
    ) -> Result<(), DiffError> {
        let symbols = |document: &Document, operations: &[OperationId]| {
            let mut result: BTreeMap<String, Vec<OperationId>> = BTreeMap::new();
            for &operation in operations {
                let name = document.operation_name(operation).unwrap_or("");
                if !self.registry.symbols(name).defines_symbol {
                    continue;
                }
                if let Some(symbol) = document.operation_symbol_name(operation) {
                    result.entry(symbol).or_default().push(operation);
                }
            }
            result
        };
        let before_symbols = symbols(self.before, before);
        let after_symbols = symbols(self.after, after);
        for (name, left) in &before_symbols {
            let Some(right) = after_symbols.get(name) else {
                continue;
            };
            if left.len() == 1 && right.len() == 1 {
                self.charge(1)?;
                self.pair(left[0], right[0]);
                before_left.remove(&left[0]);
                after_left.remove(&right[0]);
            } else {
                self.ambiguous += 1;
                self.diagnostics.push(format!(
                    "duplicate registered symbol `{name}` disabled a diff anchor"
                ));
            }
        }
        Ok(())
    }
    fn match_children(
        &mut self,
        before_op: OperationId,
        after_op: OperationId,
    ) -> Result<(), DiffError> {
        let before_regions = self.before.operation_regions(before_op).unwrap_or(&[]);
        let after_regions = self.after.operation_regions(after_op).unwrap_or(&[]);
        for (&before_region, &after_region) in before_regions.iter().zip(after_regions) {
            self.regions.insert(before_region, after_region);
            let before_blocks = self
                .before
                .region(before_region)
                .and_then(|r| r.blocks(self.before))
                .unwrap_or(&[]);
            let after_blocks = self
                .after
                .region(after_region)
                .and_then(|r| r.blocks(self.after))
                .unwrap_or(&[]);
            self.match_blocks(before_blocks, after_blocks)?;
        }
        Ok(())
    }

    fn match_blocks(&mut self, before: &[BlockId], after: &[BlockId]) -> Result<(), DiffError> {
        self.charge(before.len() + after.len())?;
        let mut pairs = Vec::new();
        let mut before_left: HashSet<_> = before.iter().copied().collect();
        let mut after_left: HashSet<_> = after.iter().copied().collect();

        // Entry blocks are defined by their role in CFG regions, and sole blocks
        // are the natural anchor for graph and single-block regions.
        if let (Some(&left), Some(&right)) = (before.first(), after.first()) {
            pairs.push((left, right));
            before_left.remove(&left);
            after_left.remove(&right);
        }

        let before_keys = grouped(before_left.iter().copied(), |block| {
            block_key(self.before, block)
        });
        let after_keys = grouped(after_left.iter().copied(), |block| {
            block_key(self.after, block)
        });
        for (key, left) in before_keys {
            let Some(right) = after_keys.get(&key) else {
                continue;
            };
            let candidates = left.len().saturating_mul(right.len());
            if candidates > 65_536 {
                self.bounded_fallback += 1;
                continue;
            }
            self.charge(candidates)?;
            if left.len() == 1 && right.len() == 1 {
                pairs.push((left[0], right[0]));
                before_left.remove(&left[0]);
                after_left.remove(&right[0]);
            } else {
                self.ambiguous += 1;
            }
        }

        for (left, right) in pairs {
            self.blocks.insert(left, right);
            self.match_list(
                self.before.block_operations(left).unwrap_or(&[]),
                self.after.block_operations(right).unwrap_or(&[]),
            )?;
        }
        Ok(())
    }
}

fn grouped<T: Copy, K: Ord>(
    items: impl Iterator<Item = T>,
    key: impl Fn(T) -> K,
) -> BTreeMap<K, Vec<T>> {
    let mut result = BTreeMap::new();
    for item in items {
        result.entry(key(item)).or_insert_with(Vec::new).push(item);
    }
    result
}

fn shape_key(document: &Document, operation: OperationId) -> String {
    format!(
        "{}|{}|{}|{}|{}",
        document.operation_name(operation).unwrap_or(""),
        document
            .operation_symbol_name(operation)
            .unwrap_or_default(),
        document.operands(operation).map_or(0, <[_]>::len),
        document.result_types(operation).map_or(0, <[_]>::len),
        document.operation_regions(operation).map_or(0, <[_]>::len)
    )
}

fn exact_key(document: &Document, operation: OperationId) -> String {
    let types = document
        .result_types(operation)
        .unwrap_or(&[])
        .iter()
        .filter_map(|id| document.type_spelling(*id))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}|{types}|{}|{}",
        shape_key(document, operation),
        format_entries(document.attributes(operation)),
        format_entries(document.properties(operation))
    )
}

fn block_key(document: &Document, block: BlockId) -> String {
    let operations = document
        .block_operations(block)
        .unwrap_or(&[])
        .iter()
        .map(|&operation| exact_key(document, operation))
        .collect::<Vec<_>>()
        .join("\u{1f}");
    format!(
        "{}|{operations}",
        document.block_argument_types(block).map_or(0, <[_]>::len)
    )
}
