use super::*;

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
}

#[pymethods]
impl DialectRegistryHandle {
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
    fn with_operation_shapes(
        operation_shapes: HashMap<String, PyRef<'_, OperationShape>>,
    ) -> PyResult<Self> {
        let owned = operation_shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.shape))
            .collect::<Vec<_>>();
        let registry = DialectRegistry::with_operation_shapes(&owned).map_err(py_error)?;
        Ok(Self {
            kind: RegistryKind::Declarative(Arc::new(registry)),
        })
    }
}
