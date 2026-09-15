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
    pub value: Option<serde_json::Value>,
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
    builder.charge(comparison_input_work(before).saturating_add(comparison_input_work(after)))?;
    builder.match_list(before.root_operations(), after.root_operations())?;
    let matched = builder.finish();
    let correspondence =
        Correspondence::from_maps(matched.operations, matched.regions, matched.blocks);
    let before_paths = operation_paths(before);
    let after_paths = operation_paths(after);
    let opaque_before = opaque_values(before).len();
    let opaque_after = opaque_values(after).len();
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
            ambiguous_groups: matched.ambiguous_groups,
            bounded_fallback_groups: matched.bounded_fallback_groups,
            opaque_before,
            opaque_after,
            work_units: matched.work_units,
        },
        diagnostics: matched.diagnostics,
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
                output.push_str(&display_field_value(&detail.before));
                output.push_str(" -> ");
                output.push_str(&display_field_value(&detail.after));
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
                records.push_str(&display_field_value(&detail.before));
                records.push_str(" -> ");
                records.push_str(&display_field_value(&detail.after));
                records.push('\n');
            }
        }
        Ok(format!(
            "{} modified, {} added, {} removed, {} moved\n\n{records}",
            counts[2], counts[0], counts[1], counts[3]
        ))
    }

    pub fn selection_to_markdown(&self, ids: &[ChangeId]) -> Result<String, DiffError> {
        fn cell(value: &str) -> String {
            value
                .replace('\\', "\\\\")
                .replace('|', "\\|")
                .replace('\n', "<br>")
        }
        let mut output = String::from(
            "| Kind | Before context | After context | Changed fields |\n| --- | --- | --- | --- |\n",
        );
        for &id in ids {
            let change = self.change(id)?;
            let context = |document: &Document, operation: OperationId, side| {
                let endpoint = self.endpoint(document, operation, side);
                let symbols = endpoint.symbol_path.join("::");
                let value = if symbols.is_empty() {
                    format!("{} `{}`", endpoint.name, endpoint.path)
                } else {
                    format!("{symbols}: {} `{}`", endpoint.name, endpoint.path)
                };
                cell(&value)
            };
            let before = change
                .before
                .map(|operation| context(self.before, operation, DiffSide::Before))
                .unwrap_or_default();
            let after = change
                .after
                .map(|operation| context(self.after, operation, DiffSide::After))
                .unwrap_or_default();
            let fields = change
                .fields
                .iter()
                .map(|field| field.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            output.push_str(&format!(
                "| {} | {before} | {after} | {} |\n",
                change.kind.as_str(),
                cell(&fields)
            ));
        }
        Ok(output)
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
                    .flat_map(|field| self.field_details(*field, before, after))
                    .collect();
                (kind, fields, details, did_move)
            } else {
                (ChangeKind::Added, Vec::new(), Vec::new(), false)
            };
            self.push_change(
                max_changes,
                kind,
                did_move,
                (before, Some(after)),
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
                    (Some(before), None),
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
        endpoints: (Option<OperationId>, Option<OperationId>),
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
            before: endpoints.0,
            after: endpoints.1,
            fields,
            details,
        });
        Ok(())
    }

    fn field_details(
        &self,
        field: ChangeField,
        before: OperationId,
        after: OperationId,
    ) -> Vec<FieldDifference> {
        match field {
            ChangeField::Operands => self.sequence_details(
                "/operands",
                self.before.operands(before).unwrap_or(&[]),
                self.after.operands(after).unwrap_or(&[]),
                |left, right| self.correspondence.equal_value_references(*left, *right),
                |value| value_reference_json(self.before, *value, &self.before_paths),
                |value| value_reference_json(self.after, *value, &self.after_paths),
            ),
            ChangeField::ResultTypes => self.sequence_details(
                "/result_types",
                self.before.result_types(before).unwrap_or(&[]),
                self.after.result_types(after).unwrap_or(&[]),
                |left, right| {
                    self.correspondence.equal_type_ids(
                        self.before,
                        self.after,
                        Some(*left),
                        Some(*right),
                    )
                },
                |value| serde_json::json!(self.before.type_spelling(*value)),
                |value| serde_json::json!(self.after.type_spelling(*value)),
            ),
            ChangeField::Attributes => self.entry_details(
                "/attributes",
                self.before.operation_attributes(before),
                self.after.operation_attributes(after),
            ),
            ChangeField::Properties => self.entry_details(
                "/properties",
                self.before.operation_properties(before),
                self.after.operation_properties(after),
            ),
            ChangeField::Successors => self.sequence_details(
                "/successors",
                self.before.successors(before).unwrap_or(&[]),
                self.after.successors(after).unwrap_or(&[]),
                |left, right| {
                    self.correspondence
                        .equal_successors(self.before, self.after, *left, *right)
                },
                |value| successor_json(self.before, *value, &self.before_paths),
                |value| successor_json(self.after, *value, &self.after_paths),
            ),
            ChangeField::Regions => vec![FieldDifference {
                path: "/regions".to_owned(),
                before: present(region_shell_json(self.before, before, &self.before_paths)),
                after: present(region_shell_json(self.after, after, &self.after_paths)),
                comparison: "represented_value",
            }],
            _ => vec![FieldDifference {
                path: format!("/{}", field.as_str()),
                before: self.field_value(self.before, before, field),
                after: self.field_value(self.after, after, field),
                comparison: "represented_value",
            }],
        }
    }

    fn sequence_details<T>(
        &self,
        path: &str,
        before: &[T],
        after: &[T],
        equal: impl Fn(&T, &T) -> bool,
        before_value: impl Fn(&T) -> serde_json::Value,
        after_value: impl Fn(&T) -> serde_json::Value,
    ) -> Vec<FieldDifference> {
        if before.len() != after.len() {
            return vec![FieldDifference {
                path: path.to_owned(),
                before: present(serde_json::Value::Array(
                    before.iter().map(before_value).collect(),
                )),
                after: present(serde_json::Value::Array(
                    after.iter().map(after_value).collect(),
                )),
                comparison: "represented_value",
            }];
        }
        before
            .iter()
            .zip(after)
            .enumerate()
            .filter(|(_, (left, right))| !equal(left, right))
            .map(|(index, (left, right))| FieldDifference {
                path: format!("{path}/{index}"),
                before: present(before_value(left)),
                after: present(after_value(right)),
                comparison: "represented_value",
            })
            .collect()
    }

    fn entry_details(
        &self,
        path: &str,
        before: Option<&[(u32, crate::semantic::AttributeId)]>,
        after: Option<&[(u32, crate::semantic::AttributeId)]>,
    ) -> Vec<FieldDifference> {
        let before: BTreeMap<_, _> = before
            .unwrap_or(&[])
            .iter()
            .filter_map(|(name, value)| Some((self.before.string(*name)?.to_owned(), *value)))
            .collect();
        let after: BTreeMap<_, _> = after
            .unwrap_or(&[])
            .iter()
            .filter_map(|(name, value)| Some((self.after.string(*name)?.to_owned(), *value)))
            .collect();
        before
            .keys()
            .chain(after.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter_map(|name| {
                let left = before.get(name).copied();
                let right = after.get(name).copied();
                (!self
                    .correspondence
                    .equal_attribute_ids(self.before, self.after, left, right))
                .then(|| FieldDifference {
                    path: format!("{path}/{}", json_pointer_escape(name)),
                    before: left.map_or_else(absent, |id| {
                        present(serde_json::json!(self.before.attribute_spelling_value(id)))
                    }),
                    after: right.map_or_else(absent, |id| {
                        present(serde_json::json!(self.after.attribute_spelling_value(id)))
                    }),
                    comparison: if left
                        .and_then(|id| self.before.attribute_value(id))
                        .is_some_and(attribute_contains_opaque)
                        || right
                            .and_then(|id| self.after.attribute_value(id))
                            .is_some_and(attribute_contains_opaque)
                    {
                        "opaque_bytes"
                    } else {
                        "represented_value"
                    },
                })
            })
            .collect()
    }

    fn field_value(
        &self,
        document: &Document,
        operation: OperationId,
        field: ChangeField,
    ) -> FieldValue {
        let value = match field {
            ChangeField::Name => serde_json::json!(document.operation_name(operation)),
            ChangeField::FunctionType => serde_json::json!(
                document
                    .function_type(operation)
                    .and_then(|id| document.type_spelling(id))
            ),
            ChangeField::Location => {
                serde_json::json!(document.operation_location(operation).flatten())
            }
            ChangeField::Position => serde_json::json!({
                "container": containing_list_path(document, operation),
                "ordinal": sibling_position(document, operation).unwrap_or(0),
            }),
            ChangeField::Operands
            | ChangeField::ResultTypes
            | ChangeField::Attributes
            | ChangeField::Properties
            | ChangeField::Successors
            | ChangeField::Regions => unreachable!("handled separately"),
        };
        present(value)
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

fn present(value: serde_json::Value) -> FieldValue {
    FieldValue {
        present: true,
        value: Some(value),
    }
}

fn absent() -> FieldValue {
    FieldValue {
        present: false,
        value: None,
    }
}

fn display_field_value(value: &FieldValue) -> String {
    if !value.present {
        return "<absent>".to_owned();
    }
    match value.value.as_ref() {
        Some(serde_json::Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
        None => "<absent>".to_owned(),
    }
}

fn json_pointer_escape(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn value_reference_json(
    document: &Document,
    value: crate::semantic::ValueReference,
    operation_paths: &HashMap<OperationId, String>,
) -> serde_json::Value {
    use crate::semantic::{ValueId, ValueReference};
    match value {
        ValueReference::Resolved(ValueId::OperationResult { operation, result }) => {
            serde_json::json!({
                "definition": operation_paths.get(&operation),
                "result": result,
            })
        }
        ValueReference::Resolved(ValueId::BlockArgument { block, argument }) => {
            serde_json::json!({
                "block": block_path(document, block, operation_paths),
                "argument": argument,
            })
        }
        ValueReference::Invalid(_) => serde_json::json!({"invalid": true}),
    }
}

fn successor_json(
    document: &Document,
    successor: crate::semantic::Successor,
    operation_paths: &HashMap<OperationId, String>,
) -> serde_json::Value {
    serde_json::json!({
        "block": block_path(document, successor.block, operation_paths),
        "arguments": document
            .successor_arguments(successor)
            .unwrap_or(&[])
            .iter()
            .map(|value| value_reference_json(document, *value, operation_paths))
            .collect::<Vec<_>>(),
    })
}

fn region_shell_json(
    document: &Document,
    operation: OperationId,
    operation_paths: &HashMap<OperationId, String>,
) -> serde_json::Value {
    serde_json::Value::Array(
        document
            .operation_regions(operation)
            .unwrap_or(&[])
            .iter()
            .map(|region| {
                serde_json::json!({
                    "blocks": document
                        .region(*region)
                        .and_then(|region| region.blocks(document))
                        .unwrap_or(&[])
                        .iter()
                        .map(|block| serde_json::json!({
                            "path": block_path(document, *block, operation_paths),
                            "argument_types": document
                                .block_argument_types(*block)
                                .unwrap_or(&[])
                                .iter()
                                .map(|value| document.type_spelling(*value))
                                .collect::<Vec<_>>(),
                        }))
                        .collect::<Vec<_>>(),
                })
            })
            .collect(),
    )
}

fn block_path(
    document: &Document,
    target: BlockId,
    operation_paths: &HashMap<OperationId, String>,
) -> Option<String> {
    for (&operation, path) in operation_paths {
        for (region_index, &region) in document
            .operation_regions(operation)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            for (block_index, &block) in document
                .region(region)
                .and_then(|region| region.blocks(document))
                .unwrap_or(&[])
                .iter()
                .enumerate()
            {
                if block == target {
                    return Some(format!(
                        "{path}/regions/{region_index}/blocks/{block_index}"
                    ));
                }
            }
        }
    }
    None
}

fn containing_list_path(document: &Document, operation: OperationId) -> String {
    let paths = operation_paths(document);
    document
        .operation(operation)
        .and_then(|operation| operation.parent_block())
        .and_then(|block| block_path(document, block, &paths))
        .map_or_else(
            || "/operations".to_owned(),
            |path| format!("{path}/operations"),
        )
}

fn attribute_contains_opaque(value: &crate::semantic::AttributeValue) -> bool {
    use crate::semantic::{AttributeValue, LargeAttributeValue};
    match value {
        AttributeValue::Type(value) => type_contains_opaque(value),
        AttributeValue::Array(values)
        | AttributeValue::DenseArray {
            elements: values, ..
        } => values.iter().any(attribute_contains_opaque),
        AttributeValue::Dictionary(values) => values
            .iter()
            .any(|(_, value)| attribute_contains_opaque(value)),
        AttributeValue::Large(LargeAttributeValue::Dense(_) | LargeAttributeValue::Sparse(_))
        | AttributeValue::WideNumber(_)
        | AttributeValue::Opaque(_) => true,
        _ => false,
    }
}

fn type_contains_opaque(value: &crate::semantic::TypeValue) -> bool {
    use crate::semantic::{MemRefLayout, TypeValue};
    match value {
        TypeValue::Opaque(_) => true,
        TypeValue::Complex(value) => type_contains_opaque(value),
        TypeValue::Tuple(values) => values.iter().any(type_contains_opaque),
        TypeValue::Tensor {
            element, encoding, ..
        } => {
            type_contains_opaque(element)
                || encoding.as_deref().is_some_and(attribute_contains_opaque)
        }
        TypeValue::Vector { element, .. } => type_contains_opaque(element),
        TypeValue::MemRef {
            element,
            layout,
            memory_space,
            ..
        } => {
            type_contains_opaque(element)
                || memory_space
                    .as_deref()
                    .is_some_and(attribute_contains_opaque)
                || layout.as_ref().is_some_and(|layout| match layout {
                    MemRefLayout::Opaque { .. } => true,
                    MemRefLayout::Attribute(value) => attribute_contains_opaque(value),
                    _ => false,
                })
        }
        TypeValue::Function { inputs, results } => {
            inputs.iter().chain(results).any(type_contains_opaque)
        }
        _ => false,
    }
}

fn opaque_values(document: &Document) -> HashSet<Vec<u8>> {
    use crate::semantic::{AttributeValue, LargeAttributeValue, MemRefLayout, TypeValue};
    fn collect_type(value: &TypeValue, values: &mut HashSet<Vec<u8>>) {
        match value {
            TypeValue::Opaque(value) => {
                let mut key = b"type:".to_vec();
                key.extend_from_slice(value);
                values.insert(key);
            }
            TypeValue::Complex(value) => collect_type(value, values),
            TypeValue::Tuple(items) => items.iter().for_each(|item| collect_type(item, values)),
            TypeValue::Tensor {
                element, encoding, ..
            } => {
                collect_type(element, values);
                if let Some(value) = encoding.as_deref() {
                    collect_attribute(value, values);
                }
            }
            TypeValue::Vector { element, .. } => collect_type(element, values),
            TypeValue::MemRef {
                element,
                layout,
                memory_space,
                ..
            } => {
                collect_type(element, values);
                if let Some(value) = memory_space.as_deref() {
                    collect_attribute(value, values);
                }
                if let Some(layout) = layout {
                    match layout {
                        MemRefLayout::Opaque {
                            spelling,
                            parameters,
                        } => {
                            let mut key = b"layout:".to_vec();
                            key.extend_from_slice(spelling.as_bytes());
                            values.insert(key);
                            parameters
                                .iter()
                                .for_each(|value| collect_attribute(value, values));
                        }
                        MemRefLayout::Attribute(value) => collect_attribute(value, values),
                        _ => {}
                    }
                }
            }
            TypeValue::Function { inputs, results } => inputs
                .iter()
                .chain(results)
                .for_each(|item| collect_type(item, values)),
            _ => {}
        }
    }
    fn collect_attribute(value: &AttributeValue, values: &mut HashSet<Vec<u8>>) {
        match value {
            AttributeValue::Type(value) => collect_type(value, values),
            AttributeValue::Array(items)
            | AttributeValue::DenseArray {
                elements: items, ..
            } => items
                .iter()
                .for_each(|item| collect_attribute(item, values)),
            AttributeValue::Dictionary(items) => items
                .iter()
                .for_each(|(_, item)| collect_attribute(item, values)),
            AttributeValue::Large(
                LargeAttributeValue::Dense(value) | LargeAttributeValue::Sparse(value),
            )
            | AttributeValue::WideNumber(value)
            | AttributeValue::Opaque(value) => {
                values.insert(value.to_vec());
            }
            _ => {}
        }
    }

    let mut values = HashSet::new();
    for operation in document.operations() {
        for (_, value) in document.attribute_entries(operation).into_iter().flatten() {
            if let Some(value) = document.attribute_value(value) {
                collect_attribute(value, &mut values);
            }
        }
        for (_, value) in document
            .operation_properties(operation)
            .into_iter()
            .flatten()
        {
            if let Some(value) = document.attribute_value(*value) {
                collect_attribute(value, &mut values);
            }
        }
        for value in document.result_types(operation).into_iter().flatten() {
            if let Some(value) = document.type_value(*value) {
                collect_type(value, &mut values);
            }
        }
        if let Some(value) = document
            .function_type(operation)
            .and_then(|value| document.type_value(value))
        {
            collect_type(value, &mut values);
        }
    }
    values
}

fn comparison_input_work(document: &Document) -> usize {
    fn chunks(bytes: usize) -> usize {
        bytes.saturating_add(63) / 64
    }
    fn type_work(value: &crate::semantic::TypeValue) -> usize {
        use crate::semantic::{MemRefLayout, TypeValue};
        1 + match value {
            TypeValue::Opaque(value) => chunks(value.len()),
            TypeValue::Complex(value) => type_work(value),
            TypeValue::Tuple(values) => values.iter().map(type_work).sum(),
            TypeValue::Tensor {
                element, encoding, ..
            } => type_work(element) + encoding.as_deref().map_or(0, attribute_work),
            TypeValue::Vector { element, .. } => type_work(element),
            TypeValue::MemRef {
                element,
                layout,
                memory_space,
                ..
            } => {
                type_work(element)
                    + memory_space.as_deref().map_or(0, attribute_work)
                    + layout.as_ref().map_or(0, |layout| match layout {
                        MemRefLayout::Opaque {
                            spelling,
                            parameters,
                        } => {
                            chunks(spelling.len())
                                + parameters.iter().map(attribute_work).sum::<usize>()
                        }
                        MemRefLayout::Attribute(value) => attribute_work(value),
                        _ => 1,
                    })
            }
            TypeValue::Function { inputs, results } => {
                inputs.iter().chain(results).map(type_work).sum()
            }
            _ => 0,
        }
    }
    fn attribute_work(value: &crate::semantic::AttributeValue) -> usize {
        use crate::semantic::{AttributeValue, LargeAttributeValue};
        1 + match value {
            AttributeValue::Type(value) => type_work(value),
            AttributeValue::Array(values)
            | AttributeValue::DenseArray {
                elements: values, ..
            } => values.iter().map(attribute_work).sum(),
            AttributeValue::Dictionary(values) => {
                values.iter().map(|(_, value)| attribute_work(value)).sum()
            }
            AttributeValue::Large(
                LargeAttributeValue::Dense(value)
                | LargeAttributeValue::Sparse(value)
                | LargeAttributeValue::Resource(value),
            )
            | AttributeValue::WideNumber(value)
            | AttributeValue::Opaque(value) => chunks(value.len()),
            _ => 0,
        }
    }

    document
        .operations()
        .map(|operation| {
            1 + document.operands(operation).map_or(0, <[_]>::len)
                + document.successors(operation).map_or(0, <[_]>::len)
                + document
                    .successors(operation)
                    .unwrap_or(&[])
                    .iter()
                    .map(|successor| {
                        document
                            .successor_arguments(*successor)
                            .map_or(0, <[_]>::len)
                    })
                    .sum::<usize>()
                + document
                    .attribute_entries(operation)
                    .into_iter()
                    .flatten()
                    .filter_map(|(_, id)| document.attribute_value(id))
                    .map(attribute_work)
                    .sum::<usize>()
                + document
                    .operation_properties(operation)
                    .into_iter()
                    .flatten()
                    .filter_map(|(_, id)| document.attribute_value(*id))
                    .map(attribute_work)
                    .sum::<usize>()
                + document
                    .result_types(operation)
                    .into_iter()
                    .flatten()
                    .filter_map(|id| document.type_value(*id))
                    .map(type_work)
                    .sum::<usize>()
        })
        .sum()
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
    entries.sort_by_key(|entry| path_indices(&entry.1));
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

struct MatcherResult {
    operations: HashMap<OperationId, OperationId>,
    regions: HashMap<RegionId, RegionId>,
    blocks: HashMap<BlockId, BlockId>,
    work_units: usize,
    ambiguous_groups: usize,
    bounded_fallback_groups: usize,
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
    fn finish(self) -> MatcherResult {
        MatcherResult {
            operations: self.operations,
            regions: self.regions,
            blocks: self.blocks,
            work_units: self.initial - self.remaining,
            ambiguous_groups: self.ambiguous,
            bounded_fallback_groups: self.bounded_fallback,
            diagnostics: self.diagnostics,
        }
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
        self.refine_by_connections(&mut before_left, &mut after_left)?;
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

    fn refine_by_connections(
        &mut self,
        before_left: &mut HashSet<OperationId>,
        after_left: &mut HashSet<OperationId>,
    ) -> Result<(), DiffError> {
        for _ in 0..4 {
            let candidates = before_left.len().saturating_mul(after_left.len());
            if candidates > 65_536 {
                self.bounded_fallback += 1;
                return Ok(());
            }
            self.charge(candidates)?;
            let mut scores = Vec::new();
            for &before in before_left.iter() {
                for &after in after_left.iter() {
                    if !candidate_shapes_compatible(self.before, before, self.after, after)
                        || registered_symbol_names_differ(
                            self.registry,
                            self.before,
                            before,
                            self.after,
                            after,
                        )
                    {
                        continue;
                    }
                    let score = connection_evidence(
                        self.before,
                        before,
                        self.after,
                        after,
                        &self.operations,
                        &self.blocks,
                    );
                    if score > 0 {
                        scores.push((before, after, score));
                    }
                }
            }
            let mut accepted = Vec::new();
            for &(before, after, score) in &scores {
                let left_best = scores
                    .iter()
                    .filter(|(candidate, _, _)| *candidate == before)
                    .map(|(_, _, score)| *score)
                    .max();
                let right_best = scores
                    .iter()
                    .filter(|(_, candidate, _)| *candidate == after)
                    .map(|(_, _, score)| *score)
                    .max();
                let left_unique = scores
                    .iter()
                    .filter(|(candidate, _, candidate_score)| {
                        *candidate == before && *candidate_score == score
                    })
                    .count()
                    == 1;
                let right_unique = scores
                    .iter()
                    .filter(|(_, candidate, candidate_score)| {
                        *candidate == after && *candidate_score == score
                    })
                    .count()
                    == 1;
                if left_best == Some(score)
                    && right_best == Some(score)
                    && left_unique
                    && right_unique
                {
                    accepted.push((before, after));
                }
            }
            if accepted.is_empty() {
                break;
            }
            for (before, after) in accepted {
                if before_left.remove(&before) && after_left.remove(&after) {
                    self.pair(before, after);
                }
            }
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

fn candidate_shapes_compatible(
    before: &Document,
    before_op: OperationId,
    after: &Document,
    after_op: OperationId,
) -> bool {
    before.operands(before_op).map_or(0, <[_]>::len)
        == after.operands(after_op).map_or(0, <[_]>::len)
        && before.result_types(before_op).map_or(0, <[_]>::len)
            == after.result_types(after_op).map_or(0, <[_]>::len)
        && before.operation_regions(before_op).map_or(0, <[_]>::len)
            == after.operation_regions(after_op).map_or(0, <[_]>::len)
}

fn registered_symbol_names_differ(
    registry: &DialectRegistry,
    before: &Document,
    before_op: OperationId,
    after: &Document,
    after_op: OperationId,
) -> bool {
    let before_name = before.operation_name(before_op).unwrap_or("");
    let after_name = after.operation_name(after_op).unwrap_or("");
    (registry.symbols(before_name).defines_symbol || registry.symbols(after_name).defines_symbol)
        && before.operation_symbol_name(before_op) != after.operation_symbol_name(after_op)
}

fn connection_evidence(
    before: &Document,
    before_op: OperationId,
    after: &Document,
    after_op: OperationId,
    operations: &HashMap<OperationId, OperationId>,
    blocks: &HashMap<BlockId, BlockId>,
) -> usize {
    use crate::semantic::{UseSite, ValueId, ValueReference};

    fn mapped_reference(
        before: ValueReference,
        after: ValueReference,
        operations: &HashMap<OperationId, OperationId>,
        blocks: &HashMap<BlockId, BlockId>,
    ) -> bool {
        match (before, after) {
            (
                ValueReference::Resolved(ValueId::OperationResult {
                    operation: left,
                    result: left_result,
                }),
                ValueReference::Resolved(ValueId::OperationResult {
                    operation: right,
                    result: right_result,
                }),
            ) => operations.get(&left) == Some(&right) && left_result == right_result,
            (
                ValueReference::Resolved(ValueId::BlockArgument {
                    block: left,
                    argument: left_argument,
                }),
                ValueReference::Resolved(ValueId::BlockArgument {
                    block: right,
                    argument: right_argument,
                }),
            ) => blocks.get(&left) == Some(&right) && left_argument == right_argument,
            _ => false,
        }
    }

    let mut score = before
        .operands(before_op)
        .unwrap_or(&[])
        .iter()
        .zip(after.operands(after_op).unwrap_or(&[]))
        .filter(|(left, right)| mapped_reference(**left, **right, operations, blocks))
        .count();

    let result_count = before.result_types(before_op).map_or(0, <[_]>::len);
    for result in 0..result_count {
        let before_uses = before.uses(ValueId::OperationResult {
            operation: before_op,
            result: result as u32,
        });
        let after_uses = after.uses(ValueId::OperationResult {
            operation: after_op,
            result: result as u32,
        });
        for left in before_uses {
            let matches = after_uses.iter().any(|right| match (left, *right) {
                (
                    UseSite::Operand {
                        operation: left_op,
                        index: left_index,
                    },
                    UseSite::Operand {
                        operation: right_op,
                        index: right_index,
                    },
                ) => operations.get(&left_op) == Some(&right_op) && left_index == right_index,
                (
                    UseSite::SuccessorArgument {
                        operation: left_op,
                        successor: left_successor,
                        argument: left_argument,
                    },
                    UseSite::SuccessorArgument {
                        operation: right_op,
                        successor: right_successor,
                        argument: right_argument,
                    },
                ) => {
                    operations.get(&left_op) == Some(&right_op)
                        && left_successor == right_successor
                        && left_argument == right_argument
                }
                _ => false,
            });
            score += usize::from(matches);
        }
    }
    score
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
