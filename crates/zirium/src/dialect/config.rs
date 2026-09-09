//! Shared JSON registry configuration for library callers and the CLI.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs, io,
    path::{Path, PathBuf},
};

use super::{DeclarativeRegistryError, DialectRegistry, OperationShape};

/// A complete registry: selected built-ins plus caller-named operation shapes.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryConfig {
    #[serde(default)]
    pub presets: Vec<String>,
    pub builtins: Vec<String>,
    pub operation_shapes: Vec<OperationShapeConfig>,
}

/// One named operation using an existing custom grammar.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationShapeConfig {
    pub name: String,
    pub shape: OperationShape,
}

const PRESET_NAMES: &[&str] = &["stablehlo"];

fn preset_json(name: &str) -> Option<&'static str> {
    match name {
        name if name == PRESET_NAMES[0] => Some(include_str!("../../registries/stablehlo.json")),
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
        {
            return Err(<serde_json::Error as serde::de::Error>::custom(
                "registry and operation shape records must be JSON objects",
            ));
        }
        Ok(config)
    }

    /// Combines complete configurations. Identical entries across configurations
    /// are shared; duplicate entries within a configuration still fail validation.
    pub fn build_many(configs: &[Self]) -> Result<DialectRegistry, DeclarativeRegistryError> {
        let mut builtins = BTreeSet::new();
        let mut shapes = BTreeMap::new();
        for config in configs {
            let (config_builtins, config_shapes) = config.expanded()?;
            builtins.extend(config_builtins);
            for (name, shape) in config_shapes {
                if let Some(previous) = shapes.insert(name.clone(), shape)
                    && previous != shape
                {
                    return Err(DeclarativeRegistryError::ConflictingShape(name));
                }
            }
        }
        Self::build_entries(builtins, shapes)
    }

    /// Validates all registrations and constructs an owned registry.
    pub fn build(&self) -> Result<DialectRegistry, DeclarativeRegistryError> {
        let (builtins, shapes) = self.expanded()?;
        Self::build_entries(builtins, shapes)
    }

    fn expanded(
        &self,
    ) -> Result<(BTreeSet<String>, BTreeMap<String, OperationShape>), DeclarativeRegistryError>
    {
        let mut seen_presets = BTreeSet::new();
        let mut builtins = BTreeSet::new();
        let mut shapes = BTreeMap::new();
        for name in &self.presets {
            if !seen_presets.insert(name.as_str()) {
                return Err(DeclarativeRegistryError::DuplicatePreset(name.clone()));
            }
            let json = preset_json(name)
                .ok_or_else(|| DeclarativeRegistryError::UnknownPreset(name.clone()))?;
            let preset = Self::from_json(json).expect("bundled registry preset must be valid");
            let (preset_builtins, preset_shapes) = preset.expanded()?;
            builtins.extend(preset_builtins);
            for (name, shape) in preset_shapes {
                if let Some(previous) = shapes.insert(name.clone(), shape)
                    && previous != shape
                {
                    return Err(DeclarativeRegistryError::ConflictingShape(name));
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
        }
        Ok((builtins, shapes))
    }

    fn build_entries(
        builtins: BTreeSet<String>,
        shapes: BTreeMap<String, OperationShape>,
    ) -> Result<DialectRegistry, DeclarativeRegistryError> {
        let builtins = builtins.iter().map(String::as_str).collect::<Vec<_>>();
        let shapes = shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), *shape))
            .collect::<Vec<_>>();
        DialectRegistry::declarative(&builtins)?.extend_operation_shapes(&shapes)
    }
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
    Registry(DeclarativeRegistryError),
}

impl fmt::Display for RegistryConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Json { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Registry(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for RegistryConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Io { error, .. } => error,
            Self::Json { error, .. } => error,
            Self::Registry(error) => error,
        })
    }
}

impl DialectRegistry {
    /// Names of the registry presets bundled with this Zirium release.
    pub const fn preset_names() -> &'static [&'static str] {
        PRESET_NAMES
    }

    /// Builds one registry preset bundled with this Zirium release.
    pub fn from_name(name: &str) -> Result<Self, DeclarativeRegistryError> {
        RegistryConfig {
            presets: vec![name.to_owned()],
            builtins: Vec::new(),
            operation_shapes: Vec::new(),
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
        let mut configs = Vec::new();
        for path in paths {
            let path = path.as_ref();
            let json = fs::read_to_string(path).map_err(|error| RegistryConfigError::Io {
                path: path.to_owned(),
                error,
            })?;
            configs.push(RegistryConfig::from_json(&json).map_err(|error| {
                RegistryConfigError::Json {
                    path: path.to_owned(),
                    error,
                }
            })?);
        }
        RegistryConfig::build_many(&configs).map_err(RegistryConfigError::Registry)
    }
}
