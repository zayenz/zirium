//! Python builders own Rust expressions, never query source strings.
use super::*;
use pyo3::{
    exceptions::PyTypeError,
    types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString},
};
use zirium::query as core;

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportExpr {
    inner: core::ReportExpr,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportQuery {
    inner: core::ReportQuery,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportGenerator {
    name: String,
    source: core::ReportExpr,
    filter: Option<core::ReportExpr>,
    bind: bool,
}
impl ReportGenerator {
    fn clause(&self) -> core::ReportGenerator {
        if self.bind {
            core::ReportGenerator::bind(self.name.clone(), self.source.clone())
        } else {
            core::ReportGenerator::in_values(
                self.name.clone(),
                self.source.clone(),
                self.filter.clone(),
            )
        }
    }
}
fn report_expression(value: &Bound<'_, PyAny>) -> PyResult<core::ReportExpr> {
    if let Ok(value) = value.extract::<PyRef<'_, ReportExpr>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, StringQuery>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, ScalarStringQuery>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, CountQuery>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, ReportQuery>>() {
        return Ok(value.inner.clone().into());
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(core::ReportExpr::boolean(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(core::ReportExpr::integer(value.extract()?));
    }
    if value.is_instance_of::<PyFloat>() {
        return core::ReportExpr::float(value.extract()?).map_err(py_error);
    }
    if value.is_instance_of::<PyString>() {
        return Ok(core::ReportExpr::string(value.extract::<String>()?));
    }
    if let Ok(values) = value.cast::<PyList>() {
        return Ok(core::ReportExpr::array(
            values
                .iter()
                .map(|value| report_expression(&value))
                .collect::<PyResult<Vec<_>>>()?,
        ));
    }
    if let Ok(fields) = value.cast::<PyDict>() {
        return Ok(core::ReportExpr::record(
            fields
                .iter()
                .map(|(key, value)| Ok((key.extract::<String>()?, report_expression(&value)?)))
                .collect::<PyResult<Vec<_>>>()?,
        ));
    }
    Err(PyTypeError::new_err(
        "expected a report scalar, list, string-keyed dict, or scalar report query",
    ))
}
#[pymethods]
impl ReportExpr {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: report_expression(value)?,
        })
    }
    #[staticmethod]
    fn name(name: String) -> Self {
        Self {
            inner: core::ReportExpr::name(name),
        }
    }
    #[staticmethod]
    fn comprehension(
        head: &Bound<'_, PyAny>,
        clauses: Vec<PyRef<'_, ReportGenerator>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core::ReportExpr::comprehension(
                report_expression(head)?,
                clauses.iter().map(|clause| clause.clause()),
            ),
        })
    }
    #[staticmethod]
    fn scope(bindings: &Bound<'_, PyDict>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: core::ReportExpr::scope(
                bindings
                    .iter()
                    .map(|(key, value)| Ok((key.extract::<String>()?, report_expression(&value)?)))
                    .collect::<PyResult<Vec<_>>>()?,
                report_expression(value)?,
            ),
        })
    }
    fn field(&self, name: String) -> Self {
        Self {
            inner: self.inner.clone().field(name),
        }
    }
    fn at(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().at(report_expression(index)?),
        })
    }
    fn one(&self) -> Self {
        Self {
            inner: self.inner.clone().one(),
        }
    }
    fn length(&self) -> Self {
        Self {
            inner: self.inner.clone().length(),
        }
    }
    fn sum(&self) -> Self {
        Self {
            inner: self.inner.clone().sum(),
        }
    }
    fn forall(&self) -> Self {
        Self {
            inner: self.inner.clone().forall(),
        }
    }
    fn exists(&self) -> Self {
        Self {
            inner: self.inner.clone().exists(),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "report expressions have no truth value; evaluate with document.query(report(expression))",
        ))
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Add, other)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Subtract, other)
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Multiply, other)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Divide, other)
    }
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::GreaterEqual, other)
    }
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Greater, other)
    }
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::LessEqual, other)
    }
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Less, other)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Equal, other)
    }
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::NotEqual, other)
    }
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::And, other)
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Or, other)
    }
    fn __invert__(&self) -> Self {
        Self {
            inner: self.inner.clone().unary(core::ValueUnaryOperator::Not),
        }
    }
}
impl ReportExpr {
    fn binary(
        &self,
        operator: core::ValueBinaryOperator,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .binary(operator, report_expression(other)?),
        })
    }
}
#[pymethods]
impl ReportQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
    }
}
#[pymethods]
impl ReportGenerator {
    #[staticmethod]
    #[pyo3(signature=(name,source,filter=None))]
    fn in_values(
        name: String,
        source: &Bound<'_, PyAny>,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            name,
            source: report_expression(source)?,
            filter: filter.map(report_expression).transpose()?,
            bind: false,
        })
    }
    #[staticmethod]
    fn bind(name: String, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            name,
            source: report_expression(value)?,
            filter: None,
            bind: true,
        })
    }
}
#[pyfunction]
fn bound_ops(name: String) -> OpQuery {
    OpQuery {
        inner: core::bound_ops(name),
    }
}
#[pyfunction]
fn report(value: &Bound<'_, PyAny>) -> PyResult<ReportQuery> {
    Ok(ReportQuery {
        inner: core::report(report_expression(value)?),
    })
}
#[pyfunction]
fn require(
    condition: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    message: String,
) -> PyResult<ReportExpr> {
    Ok(ReportExpr {
        inner: core::ReportExpr::require(
            report_expression(condition)?,
            report_expression(value)?,
            message,
        ),
    })
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct OpQuery {
    inner: core::OpQuery,
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct StringQuery {
    inner: core::StringQuery,
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct CountQuery {
    inner: core::CountQuery,
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ScalarStringQuery {
    inner: core::QueryExpr<String>,
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct TypeQuery {
    inner: core::TypeQuery,
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct AttributeQuery {
    inner: core::AttributeQuery,
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct MapQuery {
    inner: core::MapQuery<core::QueryOutput>,
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct Predicate {
    pub(super) inner: core::Predicate,
}

#[pymethods]
impl Predicate {
    fn __and__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() & other.inner.clone(),
        }
    }
    fn __or__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() | other.inner.clone(),
        }
    }
    fn __invert__(&self) -> Self {
        Self {
            inner: !self.inner.clone(),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query predicates have no truth value; use &, |, and ~ instead of and, or, and not",
        ))
    }
}

#[pymethods]
impl OpQuery {
    fn project(&self, subject: String, fields: &Bound<'_, PyDict>) -> PyResult<ReportQuery> {
        Ok(ReportQuery {
            inner: self.inner.project(
                subject,
                fields
                    .iter()
                    .map(|(key, value)| Ok((key.extract::<String>()?, report_expression(&value)?)))
                    .collect::<PyResult<Vec<_>>>()?,
            ),
        })
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
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
    fn filter(&self, predicate: &Predicate) -> Self {
        Self {
            inner: self.inner.filter(predicate.inner.clone()),
        }
    }
    fn root(&self, predicate: &Predicate) -> Self {
        Self {
            inner: self.inner.root(predicate.inner.clone()),
        }
    }
    #[pyo3(signature = (index=None))]
    fn defs(&self, index: Option<usize>) -> Self {
        Self {
            inner: match index {
                Some(index) => self.inner.defs_at(index),
                None => self.inner.defs(),
            },
        }
    }
    #[pyo3(signature = (index=None))]
    fn users(&self, index: Option<usize>) -> Self {
        Self {
            inner: match index {
                Some(index) => self.inner.users_at(index),
                None => self.inner.users(),
            },
        }
    }
    fn where_exists(&self, query: &Self) -> Self {
        Self {
            inner: self.inner.where_exists(&query.inner),
        }
    }
    fn fixpoint(&self, query: &Self) -> Self {
        Self {
            inner: self.inner.fixpoint(&query.inner),
        }
    }
    fn string_attr(&self, name: String) -> StringQuery {
        StringQuery {
            inner: self.inner.string_attr(name),
        }
    }
    fn attr(&self, name: String) -> StringQuery {
        StringQuery {
            inner: self.inner.attr(name),
        }
    }
    fn attributes(&self, name: String) -> AttributeQuery {
        AttributeQuery {
            inner: self.inner.attributes(name),
        }
    }
    fn map_by(&self, key: &ScalarStringQuery, value: &Bound<'_, PyAny>) -> PyResult<MapQuery> {
        Ok(MapQuery {
            inner: self.inner.map_by(&key.inner, &extract(value)?),
        })
    }
    fn sort_by(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(key) = key.extract::<PyRef<'_, ScalarStringQuery>>() {
            return Ok(Self {
                inner: self.inner.sort_by(&key.inner),
            });
        }
        if let Ok(key) = key.extract::<PyRef<'_, CountQuery>>() {
            return Ok(Self {
                inner: self.inner.sort_by(&key.inner),
            });
        }
        Err(PyTypeError::new_err(
            "ordering key must be a scalar string query (.one()) or a count query",
        ))
    }
    fn min_by(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(key) = key.extract::<PyRef<'_, ScalarStringQuery>>() {
            return Ok(Self {
                inner: self.inner.min_by(&key.inner),
            });
        }
        if let Ok(key) = key.extract::<PyRef<'_, CountQuery>>() {
            return Ok(Self {
                inner: self.inner.min_by(&key.inner),
            });
        }
        Err(PyTypeError::new_err(
            "ordering key must be a scalar string query (.one()) or a count query",
        ))
    }
    fn max_by(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(key) = key.extract::<PyRef<'_, ScalarStringQuery>>() {
            return Ok(Self {
                inner: self.inner.max_by(&key.inner),
            });
        }
        if let Ok(key) = key.extract::<PyRef<'_, CountQuery>>() {
            return Ok(Self {
                inner: self.inner.max_by(&key.inner),
            });
        }
        Err(PyTypeError::new_err(
            "ordering key must be a scalar string query (.one()) or a count query",
        ))
    }
    fn min_all_by(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(key) = key.extract::<PyRef<'_, ScalarStringQuery>>() {
            return Ok(Self {
                inner: self.inner.min_all_by(&key.inner),
            });
        }
        if let Ok(key) = key.extract::<PyRef<'_, CountQuery>>() {
            return Ok(Self {
                inner: self.inner.min_all_by(&key.inner),
            });
        }
        Err(PyTypeError::new_err(
            "ordering key must be a scalar string query (.one()) or a count query",
        ))
    }
    fn max_all_by(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(key) = key.extract::<PyRef<'_, ScalarStringQuery>>() {
            return Ok(Self {
                inner: self.inner.max_all_by(&key.inner),
            });
        }
        if let Ok(key) = key.extract::<PyRef<'_, CountQuery>>() {
            return Ok(Self {
                inner: self.inner.max_all_by(&key.inner),
            });
        }
        Err(PyTypeError::new_err(
            "ordering key must be a scalar string query (.one()) or a count query",
        ))
    }
    fn union(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.union(&other.inner),
        }
    }
    fn intersect(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.intersect(&other.inner),
        }
    }
    fn difference(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.difference(&other.inner),
        }
    }
    fn unique(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.unique(),
        }
    }
    fn reverse(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.reverse(),
        }
    }
    fn count(&self) -> CountQuery {
        CountQuery {
            inner: self.inner.count(),
        }
    }
    fn parent(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.parent(),
        }
    }
    fn children(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.children(),
        }
    }
    fn subtree(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.subtree(),
        }
    }
    fn closure(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.closure(),
        }
    }
    /// Follow transitive SSA definitions from the current selection.
    ///
    /// This only builds an immutable expression; work happens in
    /// `Document.query()`. Cycles terminate and the evaluation budget applies.
    /// Use `reachable()` when control-flow reachability is also wanted.
    fn slice(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.slice(),
        }
    }
    /// Explicit alias for `slice()`.
    fn backward_slice(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.backward_slice(),
        }
    }
    /// Follow transitive SSA users from the current selection.
    fn forward_slice(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.forward_slice(),
        }
    }
    fn reachable(&self) -> OpQuery {
        OpQuery {
            inner: self.inner.reachable(),
        }
    }
    fn names(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.names(),
        }
    }
    fn result_types(&self) -> TypeQuery {
        TypeQuery {
            inner: self.inner.result_types(),
        }
    }
    fn operand_types(&self) -> TypeQuery {
        TypeQuery {
            inner: self.inner.operand_types(),
        }
    }
}

#[pymethods]
impl StringQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
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
    fn union(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.union(&other.inner),
        }
    }
    fn intersect(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.intersect(&other.inner),
        }
    }
    fn difference(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.difference(&other.inner),
        }
    }
    fn tally(&self) -> MapQuery {
        MapQuery {
            inner: self.inner.tally().map_values_erased(),
        }
    }
    fn unique(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.unique(),
        }
    }
    fn reverse(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.reverse(),
        }
    }
    fn count(&self) -> CountQuery {
        CountQuery {
            inner: self.inner.count(),
        }
    }
    fn sort(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.sort(),
        }
    }
    fn min(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.min(),
        }
    }
    fn max(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.max(),
        }
    }
    fn min_all(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.min_all(),
        }
    }
    fn max_all(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.max_all(),
        }
    }
    fn one(&self) -> ScalarStringQuery {
        ScalarStringQuery {
            inner: self.inner.one(),
        }
    }
}

#[pymethods]
impl CountQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
    }
}

#[pymethods]
impl ScalarStringQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
    }
}

#[pymethods]
impl TypeQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
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
    fn unique(&self) -> TypeQuery {
        TypeQuery {
            inner: self.inner.unique(),
        }
    }
    fn reverse(&self) -> TypeQuery {
        TypeQuery {
            inner: self.inner.reverse(),
        }
    }
    fn count(&self) -> CountQuery {
        CountQuery {
            inner: self.inner.count(),
        }
    }
    fn spellings(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.spellings(),
        }
    }
}

#[pymethods]
impl AttributeQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
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
    fn unique(&self) -> AttributeQuery {
        AttributeQuery {
            inner: self.inner.unique(),
        }
    }
    fn reverse(&self) -> AttributeQuery {
        AttributeQuery {
            inner: self.inner.reverse(),
        }
    }
    fn count(&self) -> CountQuery {
        CountQuery {
            inner: self.inner.count(),
        }
    }
    fn spellings(&self) -> StringQuery {
        StringQuery {
            inner: self.inner.spellings(),
        }
    }
}

#[pymethods]
impl MapQuery {
    #[classmethod]
    fn __class_getitem__(
        class: &Bound<'_, pyo3::types::PyType>,
        item: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(class
            .py()
            .import("types")?
            .getattr("GenericAlias")?
            .call1((class, item))?
            .unbind())
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "query expressions have no truth value; evaluate with document.query(expression)",
        ))
    }
    fn count(&self) -> CountQuery {
        CountQuery {
            inner: self.inner.count(),
        }
    }
}

fn extract(value: &Bound<'_, PyAny>) -> PyResult<core::QueryExpr<core::QueryOutput>> {
    if let Ok(value) = value.extract::<PyRef<'_, ReportQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, OpQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, StringQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, CountQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, ScalarStringQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, TypeQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, AttributeQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, MapQuery>>() {
        return Ok(value.inner.erased());
    }
    Err(PyTypeError::new_err("expected a zirium.query expression"))
}
#[pyfunction]
fn ops() -> OpQuery {
    OpQuery { inner: core::ops() }
}
#[pyfunction]
fn input() -> OpQuery {
    OpQuery {
        inner: core::input(),
    }
}
#[pyfunction]
fn op(name: String) -> Predicate {
    Predicate {
        inner: core::op(name),
    }
}
#[pyfunction]
fn op_expr(name: &Bound<'_, PyAny>) -> PyResult<Predicate> {
    Ok(Predicate {
        inner: core::op_expr(report_expression(name)?),
    })
}
#[pyfunction]
fn dialect(name: String) -> Predicate {
    Predicate {
        inner: core::dialect(name),
    }
}
#[pyfunction]
fn has_attr(name: String) -> Predicate {
    Predicate {
        inner: core::has_attr(name),
    }
}
#[pyfunction]
fn result_type(spelling: String) -> Predicate {
    Predicate {
        inner: core::result_type(spelling),
    }
}
#[pyfunction]
fn string_attr_eq(name: String, value: String) -> Predicate {
    Predicate {
        inner: core::string_attr_eq(name, value),
    }
}
#[pyfunction]
fn always(value: bool) -> Predicate {
    Predicate {
        inner: core::always(value),
    }
}

pub(super) fn evaluate(
    py: Python<'_>,
    document: &Document,
    expression: &Bound<'_, PyAny>,
    max_work: Option<usize>,
    max_items: Option<usize>,
    strict: bool,
) -> PyResult<Py<PyAny>> {
    let query = extract(expression)?;
    let state = document.state.clone();
    let registry = document.registry.clone();
    let defaults = core::EvaluationLimits::default();
    let limits = core::EvaluationLimits {
        max_work: max_work.unwrap_or(defaults.max_work),
        max_items: max_items.unwrap_or(defaults.max_items),
    };
    let mut budget = query.new_budget(limits);
    let (output, mut budget) = py.detach(move || {
        let document = read_document(&state)?;
        let output = query
            .evaluate_output_with_budget(
                &document,
                registry.registry(),
                core::EvaluationOptions {
                    strict_unknown_references: strict,
                },
                &mut budget,
            )
            .map_err(py_error)?;
        Ok::<_, PyErr>((output, budget))
    })?;
    native(py, &document.state, output, &mut budget)
}

fn native(
    py: Python<'_>,
    state: &SharedDocument,
    output: core::QueryOutput,
    budget: &mut core::Budget,
) -> PyResult<Py<PyAny>> {
    use core::{NativeValue, QueryOutput};
    Ok(match output {
        QueryOutput::Operations(ids) => {
            let list = PyList::empty(py);
            for id in ids {
                list.append(Py::new(
                    py,
                    SemanticOperation {
                        state: state.clone(),
                        id,
                    },
                )?)?;
            }
            list.into_any().unbind()
        }
        QueryOutput::Values(values) => values.into_pyobject(py)?.into_any().unbind(),
        QueryOutput::Count(count) => count.into_pyobject(py)?.into_any().unbind(),
        QueryOutput::Native(NativeValue::Constructed(value)) => {
            core::constructed_value_type(
                value.value(),
                budget,
                core::ValueLocation::Query(value.range()),
            )
            .map_err(py_error)?;
            constructed(py, value.value(), budget, &value.range())?
        }
        QueryOutput::Native(NativeValue::String(value)) => {
            value.into_pyobject(py)?.into_any().unbind()
        }
        QueryOutput::Native(NativeValue::Types(ids)) => {
            let list = PyList::empty(py);
            for id in ids {
                list.append(Py::new(
                    py,
                    SemanticType {
                        state: state.clone(),
                        id: Some(id),
                        owned: None,
                    },
                )?)?;
            }
            list.into_any().unbind()
        }
        QueryOutput::Native(NativeValue::Attributes(values)) => {
            let list = PyList::empty(py);
            for (name, id) in values {
                list.append(Py::new(
                    py,
                    SemanticAttribute {
                        state: state.clone(),
                        id: Some(id),
                        name,
                        owned: None,
                        owned_spelling: None,
                        element_spellings: Arc::new(OnceLock::new()),
                    },
                )?)?;
            }
            list.into_any().unbind()
        }
        QueryOutput::Native(NativeValue::Map(entries)) => {
            let dict = PyDict::new(py);
            for (key, value) in entries {
                dict.set_item(key, native(py, state, value, budget)?)?;
            }
            dict.into_any().unbind()
        }
        QueryOutput::Map(entries) => {
            let dict = PyDict::new(py);
            for (key, value) in entries {
                dict.set_item(
                    key,
                    value
                        .as_u64()
                        .ok_or_else(|| py_error("invalid tally result"))?,
                )?;
            }
            dict.into_any().unbind()
        }
        _ => return Err(py_error("unexpected structured query output")),
    })
}

fn constructed(
    py: Python<'_>,
    value: &core::ConstructedValue,
    budget: &mut core::Budget,
    range: &std::ops::Range<usize>,
) -> PyResult<Py<PyAny>> {
    use core::ConstructedValue as V;
    budget.charge(1, zirql_location(range)).map_err(py_error)?;
    budget
        .charge_results(1, zirql_location(range))
        .map_err(py_error)?;
    Ok(match value {
        V::Boolean(value) => value.into_pyobject(py)?.to_owned().into_any().unbind(),
        V::Integer(value) => value.into_pyobject(py)?.into_any().unbind(),
        V::Float(value) => value.into_pyobject(py)?.into_any().unbind(),
        V::String(value) => {
            budget
                .charge(value.len(), zirql_location(range))
                .map_err(py_error)?;
            value.into_pyobject(py)?.into_any().unbind()
        }
        V::Array(values) => {
            budget
                .charge(values.len(), zirql_location(range))
                .map_err(py_error)?;
            budget
                .charge_results(values.len(), zirql_location(range))
                .map_err(py_error)?;
            let result = PyList::empty(py);
            for value in values {
                result.append(constructed(py, value, budget, range)?)?;
            }
            result.into_any().unbind()
        }
        V::Record(fields) => {
            budget
                .charge(fields.len(), zirql_location(range))
                .map_err(py_error)?;
            budget
                .charge_results(fields.len(), zirql_location(range))
                .map_err(py_error)?;
            let result = PyDict::new(py);
            for (name, value) in fields {
                budget
                    .charge(name.len(), zirql_location(range))
                    .map_err(py_error)?;
                result.set_item(name, constructed(py, value, budget, range)?)?;
            }
            result.into_any().unbind()
        }
        V::Member(_) | V::KeyedArray { .. } | V::Tuple(_) | V::Set { .. } => {
            return Err(PyTypeError::new_err("unsupported constructed report kind"));
        }
    })
}
fn zirql_location(range: &std::ops::Range<usize>) -> core::ValueLocation {
    core::ValueLocation::Query(range.clone())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ReportExpr>()?;
    module.add_class::<ReportQuery>()?;
    module.add_class::<ReportGenerator>()?;
    module.add_function(wrap_pyfunction!(bound_ops, module)?)?;
    module.add_function(wrap_pyfunction!(report, module)?)?;
    module.add_function(wrap_pyfunction!(require, module)?)?;
    module.add_class::<OpQuery>()?;
    module.add_class::<StringQuery>()?;
    module.add_class::<CountQuery>()?;
    module.add_class::<ScalarStringQuery>()?;
    module.add_class::<TypeQuery>()?;
    module.add_class::<AttributeQuery>()?;
    module.add_class::<MapQuery>()?;
    module.add_class::<Predicate>()?;
    module.add_function(wrap_pyfunction!(ops, module)?)?;
    module.add_function(wrap_pyfunction!(input, module)?)?;
    module.add_function(wrap_pyfunction!(op, module)?)?;
    module.add_function(wrap_pyfunction!(op_expr, module)?)?;
    module.add_function(wrap_pyfunction!(dialect, module)?)?;
    module.add_function(wrap_pyfunction!(has_attr, module)?)?;
    module.add_function(wrap_pyfunction!(result_type, module)?)?;
    module.add_function(wrap_pyfunction!(string_attr_eq, module)?)?;
    module.add_function(wrap_pyfunction!(always, module)?)?;
    Ok(())
}
