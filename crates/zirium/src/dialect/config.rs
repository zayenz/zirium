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
            config.build()?;
            builtins.extend(config.builtins.iter().cloned());
            for operation in &config.operation_shapes {
                if let Some(previous) = shapes.insert(operation.name.clone(), operation.shape)
                    && previous != operation.shape
                {
                    return Err(DeclarativeRegistryError::ConflictingShape(
                        operation.name.clone(),
                    ));
                }
            }
        }
        Self {
            builtins: builtins.into_iter().collect(),
            operation_shapes: shapes
                .into_iter()
                .map(|(name, shape)| OperationShapeConfig { name, shape })
                .collect(),
        }
        .build()
    }

    /// Validates all registrations and constructs an owned registry.
    pub fn build(&self) -> Result<DialectRegistry, DeclarativeRegistryError> {
        let builtins = self.builtins.iter().map(String::as_str).collect::<Vec<_>>();
        let shapes = self
            .operation_shapes
            .iter()
            .map(|operation| (operation.name.as_str(), operation.shape))
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
