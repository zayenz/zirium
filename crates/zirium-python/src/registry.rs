use super::*;
use pyo3::types::{PyDict, PyMapping, PyMappingMethods, PyTuple};
use zirium::dialect::{RegistryConfig, RegistryConfigError};

#[derive(Clone)]
pub(super) enum RegistryKind {
    Empty,
    Core,
    Proving,
    Declarative(Arc<DialectRegistry>),
}

pub(super) static EMPTY_REGISTRY: DialectRegistry = DialectRegistry::EMPTY;

impl RegistryKind {
    pub(super) fn registry(&self) -> &DialectRegistry {
        match self {
            Self::Empty => &EMPTY_REGISTRY,
            Self::Core => DialectRegistry::core(),
            Self::Proving => DialectRegistry::proving(),
            Self::Declarative(registry) => registry,
        }
    }
}

#[pyclass(name = "DialectRegistry", frozen, module = "zirium._zirium")]
pub(super) struct DialectRegistryHandle {
    pub(super) kind: RegistryKind,
}

#[pyclass(name = "OperationShape", frozen, module = "zirium._zirium")]
#[derive(Clone)]
pub(super) struct OperationShape {
    shape: CoreOperationShape,
}

#[pymethods]
impl OperationShape {
    #[classattr]
    const FUNC_LIKE: Self = Self {
        shape: CoreOperationShape::FuncLike,
    };

    #[classattr]
    const CALL_LIKE: Self = Self {
        shape: CoreOperationShape::CallLike,
    };

    #[classattr]
    const BINARY_OPERANDS: Self = Self {
        shape: CoreOperationShape::BinaryOperands,
    };

    #[classattr]
    const OPTIONAL_TYPED_OPERANDS: Self = Self {
        shape: CoreOperationShape::OptionalTypedOperands,
    };

    #[classattr]
    const UNARY_OPERAND: Self = Self {
        shape: CoreOperationShape::UnaryOperand,
    };

    #[classattr]
    const VARIADIC_OPERANDS: Self = Self {
        shape: CoreOperationShape::VariadicOperands,
    };

    #[classattr]
    const LITERAL_ATTRIBUTE: Self = Self {
        shape: CoreOperationShape::LiteralAttribute,
    };
}

#[pymethods]
impl DialectRegistryHandle {
    #[staticmethod]
    fn preset_names<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, DialectRegistry::preset_names())
    }

    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        let registry = DialectRegistry::from_name(name).map_err(py_error)?;
        Ok(Self {
            kind: RegistryKind::Declarative(Arc::new(registry)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, *additional_paths))]
    fn from_file(
        path: PathBuf,
        additional_paths: &Bound<'_, PyTuple>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let mut paths = vec![path];
        paths.extend(additional_paths.extract::<Vec<PathBuf>>()?);
        py.detach(move || {
            let registry =
                DialectRegistry::from_config_files(paths).map_err(|error| match error {
                    RegistryConfigError::Io { .. } => PyIOError::new_err(error.to_string()),
                    _ => PyValueError::new_err(error.to_string()),
                })?;
            Ok(Self {
                kind: RegistryKind::Declarative(Arc::new(registry)),
            })
        })
    }

    #[staticmethod]
    #[pyo3(signature = (config, *additional_configs))]
    fn from_config(
        config: &Bound<'_, PyAny>,
        additional_configs: &Bound<'_, PyTuple>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let model_type = py.import("zirium.config")?.getattr("RegistryConfig")?;
        let dumps = py.import("json")?.getattr("dumps")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("allow_nan", false)?;
        let mut json_configs = Vec::new();
        for config in std::iter::once(config.clone()).chain(additional_configs.iter()) {
            let config = if config.is_instance(&model_type)? {
                config.call_method0("model_dump")?
            } else {
                config
            };
            json_configs.push(dumps.call((config,), Some(&kwargs))?.extract::<String>()?);
        }
        py.detach(move || {
            let configs = json_configs
                .iter()
                .map(|json| RegistryConfig::from_json(json).map_err(py_error))
                .collect::<PyResult<Vec<_>>>()?;
            let registry = RegistryConfig::build_many(&configs).map_err(py_error)?;
            Ok(Self {
                kind: RegistryKind::Declarative(Arc::new(registry)),
            })
        })
    }

    #[staticmethod]
    fn empty() -> Self {
        Self {
            kind: RegistryKind::Empty,
        }
    }

    #[staticmethod]
    fn proving() -> Self {
        Self {
            kind: RegistryKind::Proving,
        }
    }

    #[staticmethod]
    fn core() -> Self {
        Self {
            kind: RegistryKind::Core,
        }
    }

    #[staticmethod]
    fn declarative(operations: Vec<String>) -> PyResult<Self> {
        let names = operations.iter().map(String::as_str).collect::<Vec<_>>();
        let registry = DialectRegistry::declarative(&names).map_err(py_error)?;
        Ok(Self {
            kind: RegistryKind::Declarative(Arc::new(registry)),
        })
    }

    #[staticmethod]
    fn with_operation_shapes(operation_shapes: &Bound<'_, PyMapping>) -> PyResult<Self> {
        let operation_shapes = operation_shapes
            .items()?
            .extract::<Vec<(String, PyRef<'_, OperationShape>)>>()?;
        let owned = operation_shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.shape))
            .collect::<Vec<_>>();
        let registry = DialectRegistry::with_operation_shapes(&owned).map_err(py_error)?;
        Ok(Self {
            kind: RegistryKind::Declarative(Arc::new(registry)),
        })
    }

    fn extend_operation_shapes(&self, operation_shapes: &Bound<'_, PyMapping>) -> PyResult<Self> {
        let operation_shapes = operation_shapes
            .items()?
            .extract::<Vec<(String, PyRef<'_, OperationShape>)>>()?;
        let owned = operation_shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.shape))
            .collect::<Vec<_>>();
        let registry = self
            .kind
            .registry()
            .extend_operation_shapes(&owned)
            .map_err(py_error)?;
        Ok(Self {
            kind: RegistryKind::Declarative(Arc::new(registry)),
        })
    }
}
