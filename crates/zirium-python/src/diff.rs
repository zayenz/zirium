use super::*;
use pyo3::types::{PyBool, PyDict, PyList};
use zirium::diff::{DiffLimits, DiffOptions, DiffSide, compare as compare_documents};
use zirium::query as core_query;

#[derive(Clone)]
struct OwnedChange {
    kind: String,
    moved: bool,
    before: Option<OperationId>,
    after: Option<OperationId>,
    fields: Vec<String>,
    details_json: String,
}

#[pyclass(name = "Change", frozen, module = "zirium._zirium")]
#[derive(Clone)]
pub(super) struct Change {
    before_state: SharedDocument,
    after_state: SharedDocument,
    value: OwnedChange,
}

#[pymethods]
impl Change {
    #[getter]
    fn kind(&self) -> &str {
        &self.value.kind
    }

    #[getter]
    fn moved(&self) -> bool {
        self.value.moved
    }

    #[getter]
    fn before(&self) -> Option<SemanticOperation> {
        self.value
            .before
            .map(|id| SemanticOperation::new(self.before_state.clone(), id))
    }

    #[getter]
    fn after(&self) -> Option<SemanticOperation> {
        self.value
            .after
            .map(|id| SemanticOperation::new(self.after_state.clone(), id))
    }

    #[getter]
    fn fields(&self) -> Vec<String> {
        self.value.fields.clone()
    }

    #[getter]
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json = py.import("json")?;
        json.call_method1("loads", (&self.value.details_json,))
    }
}

#[pyclass(name = "Diff", frozen, module = "zirium._zirium")]
pub(super) struct Diff {
    before_state: SharedDocument,
    after_state: SharedDocument,
    changes: Vec<OwnedChange>,
    json: String,
    text: String,
    compare_locations: bool,
    statistics: Vec<(String, usize)>,
    registry: RegistryKind,
    limits: DiffLimits,
}

#[pymethods]
impl Diff {
    #[getter]
    fn changes(&self) -> Vec<Change> {
        self.changes
            .iter()
            .cloned()
            .map(|value| Change {
                before_state: self.before_state.clone(),
                after_state: self.after_state.clone(),
                value,
            })
            .collect()
    }

    #[getter]
    fn compare_locations(&self) -> bool {
        self.compare_locations
    }

    #[getter]
    fn statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (name, value) in &self.statistics {
            result.set_item(name, value)?;
        }
        Ok(result)
    }

    fn to_json(&self) -> &str {
        &self.json
    }

    fn to_text(&self) -> &str {
        &self.text
    }

    fn __len__(&self) -> usize {
        self.changes.len()
    }

    #[pyo3(signature = (expression, *, max_work=None, max_items=None, strict=false))]
    fn query(
        &self,
        py: Python<'_>,
        expression: &Bound<'_, PyAny>,
        max_work: Option<&Bound<'_, PyAny>>,
        max_items: Option<&Bound<'_, PyAny>>,
        strict: bool,
    ) -> PyResult<Py<PyAny>> {
        let options = core_query::EvaluationOptions {
            strict_unknown_references: strict,
        };
        let defaults = core_query::EvaluationLimits::default();
        let limits = core_query::EvaluationLimits {
            max_work: positive_limit(max_work, "max_work")?.unwrap_or(defaults.max_work),
            max_items: positive_limit(max_items, "max_items")?.unwrap_or(defaults.max_items),
        };
        let before = read_document(&self.before_state)?;
        let after = read_document(&self.after_state)?;
        let comparison = compare_documents(
            &before,
            &after,
            self.registry.registry(),
            DiffOptions {
                compare_locations: self.compare_locations,
            },
            self.limits,
        )
        .map_err(diff_error)?;
        if let Ok(query) = expression.extract::<PyRef<'_, ChangeQuery>>() {
            let ids = query
                .inner
                .evaluate_with_options(&comparison, limits, options)
                .map_err(py_error)?;
            let values = ids
                .into_iter()
                .map(|id| self.change_wrapper(id.index()))
                .collect::<Vec<_>>();
            return Ok(PyList::new(py, values)?.into_any().unbind());
        }
        if let Ok(query) = expression.extract::<PyRef<'_, DiffOpQuery>>() {
            let selection = query
                .inner
                .evaluate_with_options(&comparison, limits, options)
                .map_err(py_error)?;
            let state = if selection.side() == DiffSide::Before {
                &self.before_state
            } else {
                &self.after_state
            };
            let values = selection
                .operations()
                .iter()
                .map(|operation| {
                    SemanticOperation::new(
                        state.clone(),
                        comparison.operation_id(*operation).expect("own operation"),
                    )
                })
                .collect::<Vec<_>>();
            return Ok(PyList::new(py, values)?.into_any().unbind());
        }
        if let Ok(query) = expression.extract::<PyRef<'_, DiffStringQuery>>() {
            let values = query
                .inner
                .evaluate_with_options(&comparison, limits, options)
                .map_err(py_error)?;
            return Ok(PyList::new(py, values)?.into_any().unbind());
        }
        if let Ok(query) = expression.extract::<PyRef<'_, DiffCountQuery>>() {
            let value = query
                .inner
                .evaluate_with_options(&comparison, limits, options)
                .map_err(py_error)?;
            return Ok(value.into_pyobject(py)?.into_any().unbind());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a diff query expression",
        ))
    }
}

impl Diff {
    fn change_wrapper(&self, index: usize) -> Change {
        Change {
            before_state: self.before_state.clone(),
            after_state: self.after_state.clone(),
            value: self.changes[index].clone(),
        }
    }
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ChangePredicate {
    inner: core_query::ChangePredicate,
}

#[pymethods]
impl ChangePredicate {
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other) = other.extract::<PyRef<'_, Self>>() {
            return Ok(Self {
                inner: self.inner.clone() & other.inner.clone(),
            });
        }
        if let Ok(other) = other.extract::<PyRef<'_, super::query::Predicate>>() {
            return Ok(Self {
                inner: self.inner.clone() & other.inner.clone(),
            });
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a predicate",
        ))
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other) = other.extract::<PyRef<'_, Self>>() {
            return Ok(Self {
                inner: self.inner.clone() | other.inner.clone(),
            });
        }
        if let Ok(other) = other.extract::<PyRef<'_, super::query::Predicate>>() {
            return Ok(Self {
                inner: self.inner.clone() | other.inner.clone(),
            });
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a predicate",
        ))
    }
    fn __invert__(&self) -> Self {
        Self {
            inner: !self.inner.clone(),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "query predicates have no truth value",
        ))
    }
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ChangeQuery {
    inner: core_query::ChangeQuery,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct DiffOpQuery {
    inner: core_query::DiffOpQuery,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct DiffStringQuery {
    inner: core_query::DiffStringQuery,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct DiffCountQuery {
    inner: core_query::DiffCountQuery,
}

#[pymethods]
impl ChangeQuery {
    fn filter(&self, predicate: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(predicate) = predicate.extract::<PyRef<'_, ChangePredicate>>() {
            return Ok(Self {
                inner: self.inner.filter(predicate.inner.clone()),
            });
        }
        if let Ok(predicate) = predicate.extract::<PyRef<'_, super::query::Predicate>>() {
            return Ok(Self {
                inner: self.inner.filter(predicate.inner.clone()),
            });
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a predicate",
        ))
    }
    fn before(&self) -> DiffOpQuery {
        DiffOpQuery {
            inner: self.inner.before(),
        }
    }
    fn after(&self) -> DiffOpQuery {
        DiffOpQuery {
            inner: self.inner.after(),
        }
    }
    fn unique(&self) -> Self {
        Self {
            inner: self.inner.unique(),
        }
    }
    fn reverse(&self) -> Self {
        Self {
            inner: self.inner.reverse(),
        }
    }
    fn head(&self, count: usize) -> Self {
        Self {
            inner: self.inner.head(count),
        }
    }
    fn tail(&self, count: usize) -> Self {
        Self {
            inner: self.inner.tail(count),
        }
    }
    fn names(&self) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.names(),
        }
    }
    fn attr(&self, name: &str) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.attr(name),
        }
    }
    fn result_types(&self) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.result_types(),
        }
    }
    fn operand_types(&self) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.operand_types(),
        }
    }
    fn count(&self) -> DiffCountQuery {
        DiffCountQuery {
            inner: self.inner.count(),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "query expressions have no truth value",
        ))
    }
}

#[pymethods]
impl DiffOpQuery {
    fn filter(&self, predicate: &super::query::Predicate) -> Self {
        Self {
            inner: self.inner.filter(predicate.inner.clone()),
        }
    }
    #[pyo3(signature=(index=None))]
    fn users(&self, index: Option<usize>) -> Self {
        Self {
            inner: index.map_or_else(|| self.inner.users(), |i| self.inner.users_at(i)),
        }
    }
    #[pyo3(signature=(index=None))]
    fn defs(&self, index: Option<usize>) -> Self {
        Self {
            inner: index.map_or_else(|| self.inner.defs(), |i| self.inner.defs_at(i)),
        }
    }
    fn parent(&self) -> Self {
        Self {
            inner: self.inner.parent(),
        }
    }
    fn children(&self) -> Self {
        Self {
            inner: self.inner.children(),
        }
    }
    fn root(&self, predicate: &super::query::Predicate) -> Self {
        Self {
            inner: self.inner.root(predicate.inner.clone()),
        }
    }
    fn subtree(&self) -> Self {
        Self {
            inner: self.inner.subtree(),
        }
    }
    fn reachable(&self) -> Self {
        Self {
            inner: self.inner.reachable(),
        }
    }
    fn closure(&self) -> Self {
        Self {
            inner: self.inner.closure(),
        }
    }
    fn slice(&self) -> Self {
        Self {
            inner: self.inner.slice(),
        }
    }
    fn backward_slice(&self) -> Self {
        Self {
            inner: self.inner.backward_slice(),
        }
    }
    fn forward_slice(&self) -> Self {
        Self {
            inner: self.inner.forward_slice(),
        }
    }
    fn fixpoint(&self, body: &DiffOpQuery) -> Self {
        Self {
            inner: self.inner.fixpoint(&body.inner),
        }
    }
    fn unique(&self) -> Self {
        Self {
            inner: self.inner.unique(),
        }
    }
    fn reverse(&self) -> Self {
        Self {
            inner: self.inner.reverse(),
        }
    }
    fn head(&self, count: usize) -> Self {
        Self {
            inner: self.inner.head(count),
        }
    }
    fn tail(&self, count: usize) -> Self {
        Self {
            inner: self.inner.tail(count),
        }
    }
    fn names(&self) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.names(),
        }
    }
    fn attr(&self, name: &str) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.attr(name),
        }
    }
    fn result_types(&self) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.result_types(),
        }
    }
    fn operand_types(&self) -> DiffStringQuery {
        DiffStringQuery {
            inner: self.inner.operand_types(),
        }
    }
    fn count(&self) -> DiffCountQuery {
        DiffCountQuery {
            inner: self.inner.count(),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "query expressions have no truth value",
        ))
    }
}

#[pymethods]
impl DiffStringQuery {
    fn sort(&self) -> Self {
        Self {
            inner: self.inner.sort(),
        }
    }
    fn min(&self) -> Self {
        Self {
            inner: self.inner.min(),
        }
    }
    fn min_all(&self) -> Self {
        Self {
            inner: self.inner.min_all(),
        }
    }
    fn max(&self) -> Self {
        Self {
            inner: self.inner.max(),
        }
    }
    fn max_all(&self) -> Self {
        Self {
            inner: self.inner.max_all(),
        }
    }
    fn unique(&self) -> Self {
        Self {
            inner: self.inner.unique(),
        }
    }
    fn count(&self) -> DiffCountQuery {
        DiffCountQuery {
            inner: self.inner.count(),
        }
    }
    fn reverse(&self) -> Self {
        Self {
            inner: self.inner.reverse(),
        }
    }
    fn head(&self, count: usize) -> Self {
        Self {
            inner: self.inner.head(count),
        }
    }
    fn tail(&self, count: usize) -> Self {
        Self {
            inner: self.inner.tail(count),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "query expressions have no truth value",
        ))
    }
}

#[pymethods]
impl DiffCountQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "query expressions have no truth value",
        ))
    }
}

#[pyfunction]
fn changes() -> ChangeQuery {
    ChangeQuery {
        inner: core_query::changes(),
    }
}
#[pyfunction]
fn change_input() -> ChangeQuery {
    ChangeQuery {
        inner: core_query::change_input(),
    }
}
#[pyfunction]
fn diff_op_input() -> DiffOpQuery {
    DiffOpQuery {
        inner: core_query::diff_op_input(),
    }
}
#[pyfunction]
fn change(kind: &str) -> PyResult<ChangePredicate> {
    Ok(ChangePredicate {
        inner: core_query::change(kind.parse().map_err(py_error)?),
    })
}
#[pyfunction]
fn changed(field: &str) -> PyResult<ChangePredicate> {
    Ok(ChangePredicate {
        inner: core_query::changed(field.parse().map_err(py_error)?),
    })
}

#[pyfunction(name = "diff", signature = (before, after, *, compare_locations=false, max_work=None, max_changes=None))]
fn semantic_diff(
    before: &Document,
    after: &Document,
    compare_locations: bool,
    max_work: Option<&Bound<'_, PyAny>>,
    max_changes: Option<&Bound<'_, PyAny>>,
    py: Python<'_>,
) -> PyResult<Diff> {
    if !before.registry.same_context(&after.registry) {
        return Err(PyValueError::new_err(
            "parse and lower both documents with the same registry instance",
        ));
    }
    let defaults = DiffLimits::default();
    let limits = DiffLimits {
        max_work: positive_limit(max_work, "max_work")?.unwrap_or(defaults.max_work),
        max_changes: positive_limit(max_changes, "max_changes")?.unwrap_or(defaults.max_changes),
    };
    if limits.max_work == 0 || limits.max_changes == 0 {
        return Err(PyValueError::new_err("diff limits must be positive"));
    }
    let before_input = before.state.clone();
    let after_input = after.state.clone();
    let registry = before.registry.clone();
    py.detach(move || {
        let (before_snapshot, after_snapshot) = if Arc::ptr_eq(&before_input, &after_input) {
            let snapshot = read_document(&before_input)?.clone();
            (snapshot.clone(), snapshot)
        } else if Arc::as_ptr(&before_input) as usize <= Arc::as_ptr(&after_input) as usize {
            let left = read_document(&before_input)?;
            let right = read_document(&after_input)?;
            (left.clone(), right.clone())
        } else {
            let right = read_document(&after_input)?;
            let left = read_document(&before_input)?;
            (left.clone(), right.clone())
        };
        let before_state = Arc::new(RwLock::new(before_snapshot));
        let after_state = Arc::new(RwLock::new(after_snapshot));
        let before_guard = read_document(&before_state)?;
        let after_guard = read_document(&after_state)?;
        let comparison = compare_documents(
            &before_guard,
            &after_guard,
            registry.registry(),
            DiffOptions { compare_locations },
            limits,
        )
        .map_err(diff_error)?;
        let changes = comparison
            .change_ids()
            .map(|id| {
                let change = comparison.change(id).expect("own change ID");
                OwnedChange {
                    kind: change.kind().as_str().to_owned(),
                    moved: change.moved(),
                    before: comparison
                        .endpoint_id(id, DiffSide::Before)
                        .expect("own ID"),
                    after: comparison.endpoint_id(id, DiffSide::After).expect("own ID"),
                    fields: change
                        .fields()
                        .iter()
                        .map(|field| field.as_str().to_owned())
                        .collect(),
                    details_json: serde_json::to_string(change.details()).expect("serializable"),
                }
            })
            .collect();
        let statistics = comparison.statistics();
        let statistics = vec![
            ("before_operations".into(), statistics.before_operations),
            ("after_operations".into(), statistics.after_operations),
            ("matched_operations".into(), statistics.matched_operations),
            ("ambiguous_groups".into(), statistics.ambiguous_groups),
            (
                "bounded_fallback_groups".into(),
                statistics.bounded_fallback_groups,
            ),
            ("opaque_before".into(), statistics.opaque_before),
            ("opaque_after".into(), statistics.opaque_after),
            ("work_units".into(), statistics.work_units),
        ];
        let result = Diff {
            json: comparison.to_json().map_err(py_error)?,
            text: comparison.to_text(),
            before_state: before_state.clone(),
            after_state: after_state.clone(),
            changes,
            compare_locations,
            statistics,
            registry: registry.clone(),
            limits,
        };
        drop(comparison);
        drop(before_guard);
        drop(after_guard);
        Ok(result)
    })
}

fn positive_limit(value: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a positive integer, not bool"
        )));
    }
    let value = value
        .extract::<i128>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a positive integer")))?;
    if value <= 0 || value > usize::MAX as i128 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a positive integer"
        )));
    }
    Ok(Some(value as usize))
}

fn diff_error(error: zirium::diff::DiffError) -> PyErr {
    use zirium::diff::DiffError;
    match error {
        DiffError::WorkLimitExceeded | DiffError::ChangeLimitExceeded => {
            ResourceLimitError::new_err(error.to_string())
        }
        DiffError::InvalidStructure { .. } => {
            StructuralVerificationError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Diff>()?;
    module.add_class::<Change>()?;
    module.add_function(wrap_pyfunction!(semantic_diff, module)?)?;
    module.add_class::<ChangePredicate>()?;
    module.add_class::<ChangeQuery>()?;
    module.add_class::<DiffOpQuery>()?;
    module.add_class::<DiffStringQuery>()?;
    module.add_class::<DiffCountQuery>()?;
    module.add_function(wrap_pyfunction!(changes, module)?)?;
    module.add_function(wrap_pyfunction!(change_input, module)?)?;
    module.add_function(wrap_pyfunction!(diff_op_input, module)?)?;
    module.add_function(wrap_pyfunction!(change, module)?)?;
    module.add_function(wrap_pyfunction!(changed, module)?)?;
    Ok(())
}
