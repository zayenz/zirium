//! Shared JSON registry configuration for library callers and the CLI.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs, io,
    io::Read,
    path::{Path, PathBuf},
};

use super::{DeclarativeRegistryError, DialectRegistry, OperationShape};
use serde::Deserialize;

/// A complete registry: selected built-ins plus caller-named operation shapes.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryConfig {
    #[serde(default, deserialize_with = "deserialize_imports")]
    pub imports: Vec<String>,
    #[serde(default)]
    pub presets: Vec<String>,
    pub builtins: Vec<String>,
    pub operation_shapes: Vec<OperationShapeConfig>,
    #[serde(default)]
    pub operation_formats: Vec<OperationFormatConfig>,
    #[serde(default)]
    pub operation_alternatives: Vec<OperationAlternativesConfig>,
}

/// Resource limits for loading a filesystem registry graph.
#[derive(Clone, Copy, Debug)]
pub struct RegistryLoadOptions {
    pub max_depth: usize,
    pub max_files: usize,
    pub max_edges: usize,
    pub max_bytes: usize,
}

impl Default for RegistryLoadOptions {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_files: 1024,
            max_edges: 4096,
            max_bytes: 16 * 1024 * 1024,
        }
    }
}

fn deserialize_imports<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    let imports = Vec::<String>::deserialize(deserializer)?;
    for import in &imports {
        if import.is_empty() || Path::new(import).is_absolute() {
            return Err(D::Error::custom(
                "registry imports must be non-empty relative paths",
            ));
        }
    }
    Ok(imports)
}

/// One named operation using an existing custom grammar.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationShapeConfig {
    pub name: String,
    pub shape: OperationShape,
    #[serde(default)]
    pub callee_attribute: Option<String>,
}

/// One named operation using a validated format description.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationFormatConfig {
    pub name: String,
    pub format: String,
    #[serde(default)]
    pub callee_attribute: Option<String>,
}

#[derive(Clone, Debug, serde::Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OperationAlternativesConfig {
    pub name: String,
    pub alternatives: Vec<OperationGrammarConfig>,
    #[serde(default)]
    pub callee_attribute: Option<String>,
}

#[derive(Clone, Debug, serde::Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OperationGrammarConfig {
    #[serde(default)]
    pub shape: Option<OperationShape>,
    #[serde(default)]
    pub format: Option<String>,
}

type ExpandedRegistry = (
    BTreeSet<String>,
    BTreeMap<String, OperationShape>,
    BTreeMap<String, String>,
    BTreeMap<String, Vec<OperationGrammarConfig>>,
    BTreeMap<String, String>,
);

const PRESET_NAMES: &[&str] = &[
    "stablehlo",
    "tosa",
    "scf",
    "linalg",
    "acc",
    "affine",
    "amdgpu",
    "amx",
    "arith",
    "arm_neon",
    "arm_sme",
    "arm_sve",
    "async",
    "bufferization",
    "cf",
    "complex",
    "dlti",
    "emitc",
    "func",
    "gpu",
    "index",
    "irdl",
    "llvm",
    "math",
    "memref",
    "ml_program",
    "mpi",
    "shard",
    "nvgpu",
    "nvvm",
    "omp",
    "pdl",
    "pdl_interp",
    "ptr",
    "quant",
    "rocdl",
    "shape",
    "sparse_tensor",
    "smt",
    "spirv",
    "tensor",
    "transform",
    "ub",
    "vector",
    "wasmssa",
    "x86vector",
    "xegpu",
    "xevm",
];

fn preset_json(name: &str) -> Option<&'static str> {
    match name {
        "stablehlo" => Some(include_str!("../../registries/stablehlo.json")),
        "tosa" => Some(include_str!("../../registries/tosa.json")),
        "scf" => Some(include_str!("../../registries/scf.json")),
        "linalg" => Some(include_str!("../../registries/linalg.json")),
        "acc" => Some(include_str!("../../registries/acc.json")),
        "affine" => Some(include_str!("../../registries/affine.json")),
        "amdgpu" => Some(include_str!("../../registries/amdgpu.json")),
        "amx" => Some(include_str!("../../registries/amx.json")),
        "arith" => Some(include_str!("../../registries/arith.json")),
        "arm_neon" => Some(include_str!("../../registries/arm_neon.json")),
        "arm_sme" => Some(include_str!("../../registries/arm_sme.json")),
        "arm_sve" => Some(include_str!("../../registries/arm_sve.json")),
        "async" => Some(include_str!("../../registries/async.json")),
        "bufferization" => Some(include_str!("../../registries/bufferization.json")),
        "cf" => Some(include_str!("../../registries/cf.json")),
        "complex" => Some(include_str!("../../registries/complex.json")),
        "dlti" => Some(include_str!("../../registries/dlti.json")),
        "emitc" => Some(include_str!("../../registries/emitc.json")),
        "func" => Some(include_str!("../../registries/func.json")),
        "gpu" => Some(include_str!("../../registries/gpu.json")),
        "index" => Some(include_str!("../../registries/index.json")),
        "irdl" => Some(include_str!("../../registries/irdl.json")),
        "llvm" => Some(include_str!("../../registries/llvm.json")),
        "math" => Some(include_str!("../../registries/math.json")),
        "memref" => Some(include_str!("../../registries/memref.json")),
        "ml_program" => Some(include_str!("../../registries/ml_program.json")),
        "mpi" => Some(include_str!("../../registries/mpi.json")),
        "shard" => Some(include_str!("../../registries/shard.json")),
        "nvgpu" => Some(include_str!("../../registries/nvgpu.json")),
        "nvvm" => Some(include_str!("../../registries/nvvm.json")),
        "omp" => Some(include_str!("../../registries/omp.json")),
        "pdl" => Some(include_str!("../../registries/pdl.json")),
        "pdl_interp" => Some(include_str!("../../registries/pdl_interp.json")),
        "ptr" => Some(include_str!("../../registries/ptr.json")),
        "quant" => Some(include_str!("../../registries/quant.json")),
        "rocdl" => Some(include_str!("../../registries/rocdl.json")),
        "shape" => Some(include_str!("../../registries/shape.json")),
        "sparse_tensor" => Some(include_str!("../../registries/sparse_tensor.json")),
        "smt" => Some(include_str!("../../registries/smt.json")),
        "spirv" => Some(include_str!("../../registries/spirv.json")),
        "tensor" => Some(include_str!("../../registries/tensor.json")),
        "transform" => Some(include_str!("../../registries/transform.json")),
        "ub" => Some(include_str!("../../registries/ub.json")),
        "vector" => Some(include_str!("../../registries/vector.json")),
        "wasmssa" => Some(include_str!("../../registries/wasmssa.json")),
        "x86vector" => Some(include_str!("../../registries/x86vector.json")),
        "xegpu" => Some(include_str!("../../registries/xegpu.json")),
        "xevm" => Some(include_str!("../../registries/xevm.json")),
        _ => None,
    }
}

impl RegistryConfig {
    /// Reads JSON, rejecting missing/unknown fields and unsupported shapes.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        // Deserialize first so duplicate fields and normal schema errors retain
        // their source locations. Serde structs also accept arrays; this file
        // format deliberately requires JSON objects for its records.
        let config = serde_json::from_str(json)?;
        let value: serde_json::Value = serde_json::from_str(json)?;
        if !value.is_object()
            || value["operation_shapes"]
                .as_array()
                .is_some_and(|entries| entries.iter().any(|entry| !entry.is_object()))
            || value["operation_formats"]
                .as_array()
                .is_some_and(|entries| entries.iter().any(|entry| !entry.is_object()))
            || value["operation_alternatives"]
                .as_array()
                .is_some_and(|entries| {
                    entries.iter().any(|entry| {
                        !entry.is_object()
                            || entry["alternatives"].as_array().is_some_and(|grammars| {
                                grammars.iter().any(|grammar| !grammar.is_object())
                            })
                    })
                })
        {
            return Err(<serde_json::Error as serde::de::Error>::custom(
                "registry and operation records must be JSON objects",
            ));
        }
        Ok(config)
    }

    /// Combines complete configurations. Identical entries across configurations
    /// are shared; duplicate entries within a configuration still fail validation.
    pub fn build_many(configs: &[Self]) -> Result<DialectRegistry, DeclarativeRegistryError> {
        if configs.iter().any(|config| !config.imports.is_empty()) {
            return Err(DeclarativeRegistryError::UnresolvedImports);
        }
        let mut builtins = BTreeSet::new();
        let mut shapes = BTreeMap::new();
        let mut formats = BTreeMap::new();
        let mut alternatives = BTreeMap::new();
        let mut call_target_attributes = BTreeMap::new();
        for config in configs {
            let (
                config_builtins,
                config_shapes,
                config_formats,
                config_alternatives,
                config_call_target_attributes,
            ) = config.expanded()?;
            builtins.extend(config_builtins);
            for (name, shape) in config_shapes {
                if let Some(previous) = shapes.insert(name.clone(), shape)
                    && previous != shape
                {
                    return Err(DeclarativeRegistryError::ConflictingShape(name));
                }
            }
            for (name, format) in config_formats {
                if let Some(previous) = formats.insert(name.clone(), format.clone())
                    && previous != format
                {
                    return Err(DeclarativeRegistryError::ConflictingFormat(name));
                }
            }
            for (name, grammars) in config_alternatives {
                if let Some(previous) = alternatives.insert(name.clone(), grammars.clone())
                    && previous != grammars
                {
                    return Err(DeclarativeRegistryError::ConflictingAlternatives(name));
                }
            }
            for (name, attribute) in config_call_target_attributes {
                if let Some(previous) =
                    call_target_attributes.insert(name.clone(), attribute.clone())
                    && previous != attribute
                {
                    return Err(DeclarativeRegistryError::ConflictingCallTargetAttribute(
                        name,
                    ));
                }
            }
        }
        Self::build_entries(
            builtins,
            shapes,
            formats,
            alternatives,
            call_target_attributes,
        )
    }

    /// Validates all registrations and constructs an owned registry.
    pub fn build(&self) -> Result<DialectRegistry, DeclarativeRegistryError> {
        if !self.imports.is_empty() {
            return Err(DeclarativeRegistryError::UnresolvedImports);
        }
        let (builtins, shapes, formats, alternatives, call_target_attributes) = self.expanded()?;
        Self::build_entries(
            builtins,
            shapes,
            formats,
            alternatives,
            call_target_attributes,
        )
    }

    fn expanded(&self) -> Result<ExpandedRegistry, DeclarativeRegistryError> {
        let mut seen_presets = BTreeSet::new();
        let mut builtins = BTreeSet::new();
        let mut shapes = BTreeMap::new();
        let mut formats = BTreeMap::new();
        let mut alternatives = BTreeMap::new();
        let mut call_target_attributes = BTreeMap::new();
        for name in &self.presets {
            if !seen_presets.insert(name.as_str()) {
                return Err(DeclarativeRegistryError::DuplicatePreset(name.clone()));
            }
            let json = preset_json(name)
                .ok_or_else(|| DeclarativeRegistryError::UnknownPreset(name.clone()))?;
            let preset = Self::from_json(json).expect("bundled registry preset must be valid");
            let (
                preset_builtins,
                preset_shapes,
                preset_formats,
                preset_alternatives,
                preset_call_target_attributes,
            ) = preset.expanded()?;
            builtins.extend(preset_builtins);
            for (name, shape) in preset_shapes {
                if let Some(previous) = shapes.insert(name.clone(), shape)
                    && previous != shape
                {
                    return Err(DeclarativeRegistryError::ConflictingShape(name));
                }
            }
            for (name, format) in preset_formats {
                formats.insert(name, format);
            }
            alternatives.extend(preset_alternatives);
            for (name, attribute) in preset_call_target_attributes {
                if let Some(previous) =
                    call_target_attributes.insert(name.clone(), attribute.clone())
                    && previous != attribute
                {
                    return Err(DeclarativeRegistryError::ConflictingCallTargetAttribute(
                        name,
                    ));
                }
            }
        }

        let mut explicit_builtins = BTreeSet::new();
        for name in &self.builtins {
            if !explicit_builtins.insert(name.as_str()) {
                return Err(DeclarativeRegistryError::DuplicateOperation(name.clone()));
            }
            builtins.insert(name.clone());
        }
        let mut explicit_shapes = BTreeSet::new();
        for operation in &self.operation_shapes {
            if !explicit_shapes.insert(operation.name.as_str()) {
                return Err(DeclarativeRegistryError::DuplicateOperation(
                    operation.name.clone(),
                ));
            }
            if let Some(previous) = shapes.insert(operation.name.clone(), operation.shape)
                && previous != operation.shape
            {
                return Err(DeclarativeRegistryError::ConflictingShape(
                    operation.name.clone(),
                ));
            }
            if let Some(attribute) = operation.callee_attribute.as_deref()
                && (operation.shape != OperationShape::CallLike || !valid_attribute_name(attribute))
            {
                return Err(DeclarativeRegistryError::InvalidCallTargetAttribute(
                    operation.name.clone(),
                ));
            }
            if operation.shape == OperationShape::CallLike {
                let attribute = operation.callee_attribute.as_deref().unwrap_or("callee");
                if let Some(previous) =
                    call_target_attributes.insert(operation.name.clone(), attribute.to_owned())
                    && previous != attribute
                {
                    return Err(DeclarativeRegistryError::ConflictingCallTargetAttribute(
                        operation.name.clone(),
                    ));
                }
            }
        }
        let mut explicit_formats = BTreeSet::new();
        for operation in &self.operation_formats {
            if !explicit_formats.insert(operation.name.as_str())
                || explicit_shapes.contains(operation.name.as_str())
            {
                return Err(DeclarativeRegistryError::DuplicateOperation(
                    operation.name.clone(),
                ));
            }
            if let Some(previous) = formats.insert(operation.name.clone(), operation.format.clone())
                && previous != operation.format
            {
                return Err(DeclarativeRegistryError::ConflictingFormat(
                    operation.name.clone(),
                ));
            }
            if let Some(attribute) = operation.callee_attribute.as_deref() {
                let direct_call = super::OperationFormat::parse(&operation.format)
                    .is_ok_and(|format| format.captures_callee());
                if !valid_attribute_name(attribute) || !direct_call {
                    return Err(DeclarativeRegistryError::InvalidCallTargetAttribute(
                        operation.name.clone(),
                    ));
                }
                if let Some(previous) =
                    call_target_attributes.insert(operation.name.clone(), attribute.to_owned())
                    && previous != attribute
                {
                    return Err(DeclarativeRegistryError::ConflictingCallTargetAttribute(
                        operation.name.clone(),
                    ));
                }
            }
        }
        let mut explicit_alternatives = BTreeSet::new();
        for operation in &self.operation_alternatives {
            if !explicit_alternatives.insert(operation.name.as_str())
                || explicit_shapes.contains(operation.name.as_str())
                || explicit_formats.contains(operation.name.as_str())
            {
                return Err(DeclarativeRegistryError::DuplicateOperation(
                    operation.name.clone(),
                ));
            }
            if operation.alternatives.len() < 2
                || operation.alternatives.iter().any(|grammar| {
                    matches!(
                        (&grammar.shape, &grammar.format),
                        (None, None) | (Some(_), Some(_))
                    )
                })
            {
                return Err(DeclarativeRegistryError::InvalidOperationAlternatives(
                    operation.name.clone(),
                ));
            }
            if let Some(previous) =
                alternatives.insert(operation.name.clone(), operation.alternatives.clone())
                && previous != operation.alternatives
            {
                return Err(DeclarativeRegistryError::ConflictingAlternatives(
                    operation.name.clone(),
                ));
            }
            if let Some(attribute) = operation.callee_attribute.as_deref() {
                let direct_calls = operation.alternatives.iter().all(|grammar| {
                    grammar.shape == Some(OperationShape::CallLike)
                        || grammar.format.as_deref().is_some_and(|description| {
                            super::OperationFormat::parse(description)
                                .is_ok_and(|format| format.captures_callee())
                        })
                });
                if !valid_attribute_name(attribute) || !direct_calls {
                    return Err(DeclarativeRegistryError::InvalidCallTargetAttribute(
                        operation.name.clone(),
                    ));
                }
                if let Some(previous) =
                    call_target_attributes.insert(operation.name.clone(), attribute.to_owned())
                    && previous != attribute
                {
                    return Err(DeclarativeRegistryError::ConflictingCallTargetAttribute(
                        operation.name.clone(),
                    ));
                }
            }
        }
        Ok((
            builtins,
            shapes,
            formats,
            alternatives,
            call_target_attributes,
        ))
    }

    fn build_entries(
        builtins: BTreeSet<String>,
        shapes: BTreeMap<String, OperationShape>,
        formats: BTreeMap<String, String>,
        alternatives: BTreeMap<String, Vec<OperationGrammarConfig>>,
        call_target_attributes: BTreeMap<String, String>,
    ) -> Result<DialectRegistry, DeclarativeRegistryError> {
        let builtins = builtins.iter().map(String::as_str).collect::<Vec<_>>();
        let shapes = shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), *shape))
            .collect::<Vec<_>>();
        let formats = formats
            .iter()
            .map(|(name, format)| (name.as_str(), format.as_str()))
            .collect::<Vec<_>>();
        DialectRegistry::declarative(&builtins)?
            .extend_operation_shapes(&shapes)?
            .extend_operation_formats(&formats)
            .and_then(|registry| {
                registry
                    .extend_operation_alternatives(&alternatives.into_iter().collect::<Vec<_>>())
            })
            .map(|registry| {
                registry.with_call_target_attributes(call_target_attributes.into_iter().collect())
            })
    }
}

fn valid_attribute_name(name: &str) -> bool {
    !name.is_empty()
        && name.split('.').all(|component| {
            let mut chars = component.chars();
            chars
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
                && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        })
}

/// Reading, deserializing, or constructing a registry failed.
#[derive(Debug)]
pub enum RegistryConfigError {
    Io {
        path: PathBuf,
        error: io::Error,
    },
    Json {
        path: PathBuf,
        error: serde_json::Error,
    },
    IoInGraph {
        path: PathBuf,
        error: io::Error,
        chain: Vec<PathBuf>,
    },
    JsonInGraph {
        path: PathBuf,
        error: serde_json::Error,
        chain: Vec<PathBuf>,
    },
    Registry(DeclarativeRegistryError),
    Graph(String),
    Limit(String),
}

impl fmt::Display for RegistryConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Json { path, error } => write!(f, "{}: {error}", path.display()),
            Self::IoInGraph { path, error, chain } => {
                write!(f, "{}: {error}{}", path.display(), display_chain(chain))
            }
            Self::JsonInGraph { path, error, chain } => {
                write!(f, "{}: {error}{}", path.display(), display_chain(chain))
            }
            Self::Registry(error) => error.fmt(f),
            Self::Graph(message) | Self::Limit(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for RegistryConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error, .. } | Self::IoInGraph { error, .. } => Some(error),
            Self::Json { error, .. } | Self::JsonInGraph { error, .. } => Some(error),
            Self::Registry(error) => Some(error),
            Self::Graph(_) | Self::Limit(_) => None,
        }
    }
}

fn io_error(path: PathBuf, error: io::Error, chain: Vec<PathBuf>) -> RegistryConfigError {
    if chain.len() < 2 {
        RegistryConfigError::Io { path, error }
    } else {
        RegistryConfigError::IoInGraph { path, error, chain }
    }
}

fn json_error(path: PathBuf, error: serde_json::Error, chain: Vec<PathBuf>) -> RegistryConfigError {
    if chain.len() < 2 {
        RegistryConfigError::Json { path, error }
    } else {
        RegistryConfigError::JsonInGraph { path, error, chain }
    }
}

fn display_chain(chain: &[PathBuf]) -> String {
    if chain.len() < 2 {
        String::new()
    } else {
        format!(
            " (import chain: {})",
            chain
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(" -> ")
        )
    }
}

#[derive(Clone)]
struct SourcedConfig {
    config: RegistryConfig,
    origin: PathBuf,
    chain: Vec<PathBuf>,
}

trait RegistryResolver {
    type Id: Clone + Ord + Eq;

    fn read(&mut self, id: &Self::Id, limit: usize) -> io::Result<Vec<u8>>;
    fn resolve(&mut self, parent: &Self::Id, import: &str) -> io::Result<Self::Id>;
    fn display(&self, id: &Self::Id) -> PathBuf;
}

struct FilesystemResolver;

impl RegistryResolver for FilesystemResolver {
    type Id = PathBuf;

    fn read(&mut self, id: &Self::Id, limit: usize) -> io::Result<Vec<u8>> {
        let mut json = Vec::new();
        fs::File::open(id)?
            .take(limit as u64)
            .read_to_end(&mut json)?;
        Ok(json)
    }

    fn resolve(&mut self, parent: &Self::Id, import: &str) -> io::Result<Self::Id> {
        fs::canonicalize(
            parent
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(import),
        )
    }

    fn display(&self, id: &Self::Id) -> PathBuf {
        id.clone()
    }
}

struct ResourceResolver<F> {
    reader: F,
}

impl<F> RegistryResolver for ResourceResolver<F>
where
    F: FnMut(&str, usize) -> io::Result<Vec<u8>>,
{
    type Id = String;

    fn read(&mut self, id: &Self::Id, limit: usize) -> io::Result<Vec<u8>> {
        (self.reader)(id, limit)
    }

    fn resolve(&mut self, parent: &Self::Id, import: &str) -> io::Result<Self::Id> {
        let parent = parent.rsplit_once('/').map_or("", |(parent, _)| parent);
        if parent.is_empty() {
            normalize_resource_id(import)
        } else {
            normalize_resource_id(&format!("{parent}/{import}"))
        }
    }

    fn display(&self, id: &Self::Id) -> PathBuf {
        PathBuf::from(id)
    }
}

fn normalize_resource_id(identifier: &str) -> io::Result<String> {
    if identifier.is_empty() || identifier.starts_with('/') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "resource identifiers must be non-empty relative paths",
        ));
    }
    let mut components = Vec::new();
    for component in identifier.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.pop().is_none() {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "resource import escapes the package boundary",
                    ));
                }
            }
            component => components.push(component),
        }
    }
    if components.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "resource identifier does not name a file",
        ));
    }
    Ok(components.join("/"))
}

struct RegistryGraphLoader<R: RegistryResolver> {
    resolver: R,
    options: RegistryLoadOptions,
    files: usize,
    edges: usize,
    bytes: usize,
    active: Vec<R::Id>,
    completed: BTreeSet<R::Id>,
    configs: Vec<SourcedConfig>,
}

impl<R: RegistryResolver> RegistryGraphLoader<R> {
    fn new(options: RegistryLoadOptions, resolver: R) -> Self {
        Self {
            resolver,
            options,
            files: 0,
            edges: 0,
            bytes: 0,
            active: Vec::new(),
            completed: BTreeSet::new(),
            configs: Vec::new(),
        }
    }

    fn load_root(&mut self, id: R::Id) -> Result<(), RegistryConfigError> {
        self.load(id, 0, Vec::new())
    }

    fn load(
        &mut self,
        id: R::Id,
        depth: usize,
        mut parent_chain: Vec<PathBuf>,
    ) -> Result<(), RegistryConfigError> {
        let path = self.resolver.display(&id);
        parent_chain.push(path.clone());
        if depth > self.options.max_depth {
            return Err(RegistryConfigError::Limit(format!(
                "registry import depth {depth} exceeds limit {}{}",
                self.options.max_depth,
                display_chain(&parent_chain)
            )));
        }
        if let Some(cycle_start) = self.active.iter().position(|active| active == &id) {
            let mut cycle = self.active[cycle_start..]
                .iter()
                .map(|id| self.resolver.display(id))
                .collect::<Vec<_>>();
            cycle.push(path.clone());
            return Err(RegistryConfigError::Graph(format!(
                "registry import cycle in chain {} (cycle: {})",
                parent_chain
                    .iter()
                    .map(|entry| entry.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" -> "),
                cycle
                    .iter()
                    .map(|entry| entry.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            )));
        }
        if self.completed.contains(&id) {
            return Ok(());
        }
        if self.files >= self.options.max_files {
            return Err(RegistryConfigError::Limit(format!(
                "registry file count exceeds limit {}{}",
                self.options.max_files,
                display_chain(&parent_chain)
            )));
        }

        let remaining = self.options.max_bytes.saturating_sub(self.bytes);
        let json = self
            .resolver
            .read(&id, remaining.saturating_add(1))
            .map_err(|error| io_error(path.clone(), error, parent_chain.clone()))?;
        if json.len() > remaining {
            return Err(RegistryConfigError::Limit(format!(
                "registry bytes exceed limit {}{}",
                self.options.max_bytes,
                display_chain(&parent_chain)
            )));
        }
        let json = String::from_utf8(json).map_err(|error| {
            io_error(
                path.clone(),
                io::Error::new(io::ErrorKind::InvalidData, error),
                parent_chain.clone(),
            )
        })?;
        let config = RegistryConfig::from_json(&json)
            .map_err(|error| json_error(path.clone(), error, parent_chain.clone()))?;
        self.files += 1;
        self.bytes += json.len();
        self.active.push(id.clone());

        let mut siblings = BTreeMap::<R::Id, String>::new();
        for import in &config.imports {
            self.edges = self.edges.saturating_add(1);
            if self.edges > self.options.max_edges {
                return Err(RegistryConfigError::Limit(format!(
                    "registry import edges exceed limit {}{}",
                    self.options.max_edges,
                    display_chain(&parent_chain)
                )));
            }
            let child = self
                .resolver
                .resolve(&id, import)
                .map_err(|error| io_error(PathBuf::from(import), error, parent_chain.clone()))?;
            if let Some(first) = siblings.insert(child.clone(), import.clone()) {
                let child_path = self.resolver.display(&child);
                return Err(RegistryConfigError::Graph(format!(
                    "{} imports the same canonical child {} twice as {first:?} and {import:?}{}",
                    path.display(),
                    child_path.display(),
                    display_chain(&parent_chain)
                )));
            }
            self.load(child, depth + 1, parent_chain.clone())?;
        }

        self.active.pop();
        self.completed.insert(id);
        self.configs.push(SourcedConfig {
            config,
            origin: path,
            chain: parent_chain,
        });
        Ok(())
    }
}

fn source_description(source: &SourcedConfig) -> String {
    format!(
        "{}{}",
        source.origin.display(),
        display_chain(&source.chain)
    )
}

fn conflicting_sources(
    what: &str,
    name: &str,
    first: &SourcedConfig,
    second: &SourcedConfig,
) -> RegistryConfigError {
    RegistryConfigError::Graph(format!(
        "conflicting {what} for {name}: {} and {}",
        source_description(first),
        source_description(second)
    ))
}

fn build_sourced(configs: &[SourcedConfig]) -> Result<DialectRegistry, RegistryConfigError> {
    let mut builtins = BTreeSet::new();
    let mut shapes = BTreeMap::new();
    let mut formats = BTreeMap::new();
    let mut alternatives = BTreeMap::new();
    let mut call_target_attributes = BTreeMap::new();
    let mut shape_sources = BTreeMap::<String, usize>::new();
    let mut format_sources = BTreeMap::<String, usize>::new();
    let mut alternative_sources = BTreeMap::<String, usize>::new();
    let mut call_sources = BTreeMap::<String, usize>::new();
    let mut definition_sources = BTreeMap::<String, (&'static str, usize)>::new();

    for (index, source) in configs.iter().enumerate() {
        let (source_builtins, source_shapes, source_formats, source_alternatives, source_calls) =
            source.config.expanded().map_err(|error| {
                RegistryConfigError::Graph(format!("{error} in {}", source_description(source)))
            })?;

        for name in &source_builtins {
            record_definition(
                &mut definition_sources,
                configs,
                name,
                "built-in operation",
                index,
            )?;
        }
        for name in source_shapes.keys() {
            record_definition(
                &mut definition_sources,
                configs,
                name,
                "operation shape",
                index,
            )?;
        }
        for name in source_formats.keys() {
            record_definition(
                &mut definition_sources,
                configs,
                name,
                "operation format",
                index,
            )?;
        }
        for name in source_alternatives.keys() {
            record_definition(
                &mut definition_sources,
                configs,
                name,
                "operation alternatives",
                index,
            )?;
        }

        builtins.extend(source_builtins);
        for (name, shape) in source_shapes {
            if let Some(previous) = shapes.insert(name.clone(), shape)
                && previous != shape
            {
                return Err(conflicting_sources(
                    "operation shapes",
                    &name,
                    &configs[shape_sources[&name]],
                    source,
                ));
            }
            shape_sources.entry(name).or_insert(index);
        }
        for (name, format) in source_formats {
            if let Some(previous) = formats.insert(name.clone(), format.clone())
                && previous != format
            {
                return Err(conflicting_sources(
                    "operation formats",
                    &name,
                    &configs[format_sources[&name]],
                    source,
                ));
            }
            format_sources.entry(name).or_insert(index);
        }
        for (name, grammars) in source_alternatives {
            if let Some(previous) = alternatives.insert(name.clone(), grammars.clone())
                && previous != grammars
            {
                return Err(conflicting_sources(
                    "operation alternatives",
                    &name,
                    &configs[alternative_sources[&name]],
                    source,
                ));
            }
            alternative_sources.entry(name).or_insert(index);
        }
        for (name, attribute) in source_calls {
            if let Some(previous) = call_target_attributes.insert(name.clone(), attribute.clone())
                && previous != attribute
            {
                return Err(conflicting_sources(
                    "call-target attributes",
                    &name,
                    &configs[call_sources[&name]],
                    source,
                ));
            }
            call_sources.entry(name).or_insert(index);
        }
    }

    RegistryConfig::build_entries(
        builtins,
        shapes,
        formats,
        alternatives,
        call_target_attributes,
    )
    .map_err(|error| {
        RegistryConfigError::Graph(format!(
            "{error} while composing {}",
            configs
                .iter()
                .map(source_description)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })
}

fn record_definition(
    definitions: &mut BTreeMap<String, (&'static str, usize)>,
    configs: &[SourcedConfig],
    name: &str,
    kind: &'static str,
    index: usize,
) -> Result<(), RegistryConfigError> {
    if let Some(&(previous_kind, previous_index)) = definitions.get(name) {
        if previous_kind != kind {
            return Err(conflicting_sources(
                "registry definitions",
                name,
                &configs[previous_index],
                &configs[index],
            ));
        }
    } else {
        definitions.insert(name.to_owned(), (kind, index));
    }
    Ok(())
}

impl DialectRegistry {
    /// Names of the registry presets bundled with this Zirium release.
    pub const fn preset_names() -> &'static [&'static str] {
        PRESET_NAMES
    }

    /// Builds one registry preset bundled with this Zirium release.
    pub fn from_name(name: &str) -> Result<Self, DeclarativeRegistryError> {
        RegistryConfig {
            imports: Vec::new(),
            presets: vec![name.to_owned()],
            builtins: Vec::new(),
            operation_shapes: Vec::new(),
            operation_formats: Vec::new(),
            operation_alternatives: Vec::new(),
        }
        .build()
    }

    /// Loads a complete registry from a UTF-8 JSON file.
    ///
    /// No default operations are implicitly added. I/O, JSON, and registration
    /// failures are distinguished in the returned error.
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self, RegistryConfigError> {
        Self::from_config_files([path])
    }

    /// Combines JSON registry files, rejecting conflicting definitions.
    pub fn from_config_files<P: AsRef<Path>>(
        paths: impl IntoIterator<Item = P>,
    ) -> Result<Self, RegistryConfigError> {
        Self::from_config_files_with_options(paths, RegistryLoadOptions::default())
    }

    /// Combines filesystem registry graphs with explicit loader limits.
    pub fn from_config_files_with_options<P: AsRef<Path>>(
        paths: impl IntoIterator<Item = P>,
        options: RegistryLoadOptions,
    ) -> Result<Self, RegistryConfigError> {
        Self::load_config_files(paths, None, options)
    }

    /// Loads path-like resource identifiers through a caller-supplied reader.
    ///
    /// Imports are resolved lexically relative to their parent identifier.
    /// Normalized identifiers establish identity for deduplication and cycle
    /// checks. `..` is accepted within the package but rejected when it would
    /// escape the package boundary. The reader receives the maximum number of
    /// bytes it should return; returned bytes count toward `max_bytes`.
    pub fn from_config_resources_with_options<S, F>(
        roots: impl IntoIterator<Item = S>,
        reader: F,
        options: RegistryLoadOptions,
    ) -> Result<Self, RegistryConfigError>
    where
        S: AsRef<str>,
        F: FnMut(&str, usize) -> io::Result<Vec<u8>>,
    {
        let mut loader = RegistryGraphLoader::new(options, ResourceResolver { reader });
        for root in roots {
            let root = root.as_ref();
            let normalized =
                normalize_resource_id(root).map_err(|error| RegistryConfigError::Io {
                    path: PathBuf::from(root),
                    error,
                })?;
            loader.load_root(normalized)?;
        }
        build_sourced(&loader.configs)
    }

    /// Loads resource identifiers with the default graph limits.
    pub fn from_config_resources<S, F>(
        roots: impl IntoIterator<Item = S>,
        reader: F,
    ) -> Result<Self, RegistryConfigError>
    where
        S: AsRef<str>,
        F: FnMut(&str, usize) -> io::Result<Vec<u8>>,
    {
        Self::from_config_resources_with_options(roots, reader, RegistryLoadOptions::default())
    }

    /// Combines filesystem roots and trailing bundled presets.
    pub fn from_config_files_with_options_and_presets<P: AsRef<Path>>(
        paths: impl IntoIterator<Item = P>,
        presets: Vec<String>,
        options: RegistryLoadOptions,
    ) -> Result<Self, RegistryConfigError> {
        let trailing = (!presets.is_empty()).then_some(RegistryConfig {
            imports: Vec::new(),
            presets,
            builtins: Vec::new(),
            operation_shapes: Vec::new(),
            operation_formats: Vec::new(),
            operation_alternatives: Vec::new(),
        });
        Self::load_config_files(paths, trailing, options)
    }

    fn load_config_files<P: AsRef<Path>>(
        paths: impl IntoIterator<Item = P>,
        trailing: Option<RegistryConfig>,
        options: RegistryLoadOptions,
    ) -> Result<Self, RegistryConfigError> {
        let mut loader = RegistryGraphLoader::new(options, FilesystemResolver);
        for path in paths {
            let path = path.as_ref();
            let canonical = fs::canonicalize(path).map_err(|error| RegistryConfigError::Io {
                path: path.to_owned(),
                error,
            })?;
            loader.load_root(canonical)?;
        }
        if let Some(config) = trailing {
            loader.configs.push(SourcedConfig {
                config,
                origin: PathBuf::from("<command-line presets>"),
                chain: Vec::new(),
            });
        }
        build_sourced(&loader.configs)
    }
}
