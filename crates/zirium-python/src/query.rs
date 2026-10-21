//! Python builders own Rust expressions, never query source strings.
use super::*;
use pyo3::{
    exceptions::PyTypeError,
    types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString},
};
use zirium::query as core;

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct SsaQuery {
    inner: core::SsaQuery,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct CaptureQuery {
    inner: core::CaptureQuery,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct MatchBuilder {
    inner: core::MatchBuilder,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct MatchPredicate {
    inner: core::MatchPredicate,
}
#[pymethods]
impl MatchBuilder {
    fn from_(&self, name: String, source: &OpQuery) -> Self {
        Self {
            inner: self.inner.clone().from(name, &source.inner),
        }
    }
    fn bind(&self, name: String, value: &SsaQuery) -> Self {
        Self {
            inner: self.inner.clone().bind(name, &value.inner),
        }
    }
    fn where_(&self, predicate: &MatchPredicate) -> Self {
        Self {
            inner: self.inner.clone().where_(&predicate.inner),
        }
    }
    fn build(&self) -> CaptureQuery {
        CaptureQuery {
            inner: self.inner.clone().build(),
        }
    }
}
#[pymethods]
impl MatchPredicate {
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
            "match predicates have no truth value; use &, |, and ~",
        ))
    }
}
#[pymethods]
impl SsaQuery {
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "evaluate SSA queries with document.query(expression)",
        ))
    }
}
#[pymethods]
impl CaptureQuery {
    fn with_parameters(&self, declarations: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut budget = self.inner.new_budget(core::EvaluationLimits::default());
        Ok(Self {
            inner: self
                .inner
                .with_parameters(query_parameters(declarations, &mut budget)?),
        })
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "evaluate capture queries with document.query(expression)",
        ))
    }
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
}
#[pyfunction]
fn matches() -> MatchBuilder {
    MatchBuilder {
        inner: core::matches(),
    }
}
#[pyfunction]
fn bound_ssa(name: String) -> SsaQuery {
    SsaQuery {
        inner: core::bound_ssa(name),
    }
}
#[pyfunction]
fn capture_ops(subject: String, field: String) -> OpQuery {
    OpQuery {
        inner: core::capture_ops(subject, field),
    }
}
#[pyfunction]
fn capture_ssa(subject: String, field: String) -> SsaQuery {
    SsaQuery {
        inner: core::capture_ssa(subject, field),
    }
}
#[pyfunction]
fn same_value(left: &SsaQuery, right: &SsaQuery) -> MatchPredicate {
    MatchPredicate {
        inner: core::same_value(&left.inner, &right.inner),
    }
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportExpr {
    inner: core::ReportExpr,
}
/// Python owners retain reusable common patterns; builders copy only metadata.
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportPattern {
    inner: std::sync::Arc<core::Pattern>,
}
fn pattern_budget() -> core::Budget {
    let limits = core::EvaluationLimits::default();
    core::Budget::for_program(core::ProgramLimits {
        work: limits.max_work,
        results: 1_000_000,
        nesting: core::parser::DEFAULT_NESTING_LIMIT,
        stream_items: limits.max_items,
    })
}
impl ReportPattern {
    fn pattern(&self, budget: &mut core::Budget) -> PyResult<core::Pattern> {
        core::copy_pattern(&self.inner, budget).map_err(|e| PyTypeError::new_err(e.to_string()))
    }
    fn node(kind: core::PatternKind) -> Self {
        Self {
            inner: std::sync::Arc::new(core::Located::new(kind, 0..0)),
        }
    }
}
#[pymethods]
impl ReportPattern {
    #[staticmethod]
    fn name(name: String) -> Self {
        Self::node(core::PatternKind::Name(core::Located::new(name, 0..0)))
    }
    #[staticmethod]
    fn wildcard() -> Self {
        Self::node(core::PatternKind::Wildcard)
    }
    #[staticmethod]
    fn absent() -> Self {
        Self::node(core::PatternKind::Absent)
    }
    #[staticmethod]
    fn present(pattern: &ReportPattern) -> PyResult<Self> {
        Ok(Self::node(core::PatternKind::Present(Box::new(
            pattern.pattern(&mut pattern_budget())?,
        ))))
    }
    #[staticmethod]
    fn tuple(patterns: Vec<PyRef<'_, ReportPattern>>) -> PyResult<Self> {
        let mut budget = pattern_budget();
        Ok(Self::node(core::PatternKind::Tuple(
            patterns
                .iter()
                .map(|p| p.pattern(&mut budget))
                .collect::<PyResult<_>>()?,
        )))
    }
    #[staticmethod]
    #[pyo3(signature = (fields, *, ignore_rest=false))]
    fn record(fields: &Bound<'_, PyDict>, ignore_rest: bool) -> PyResult<Self> {
        let mut budget = pattern_budget();
        let fields = fields
            .iter()
            .map(|(label, pattern)| {
                let name = label.extract::<String>()?;
                let pattern = pattern.extract::<PyRef<'_, ReportPattern>>()?;
                Ok((
                    core::Located::new(name, 0..0),
                    pattern.pattern(&mut budget)?,
                ))
            })
            .collect::<PyResult<_>>()?;
        Ok(Self::node(core::PatternKind::Record {
            fields,
            ignore_rest,
        }))
    }
    #[staticmethod]
    fn literal(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = if value.is_instance_of::<PyBool>() {
            core::ConstructedValue::Boolean(value.extract()?)
        } else if value.is_instance_of::<PyInt>() {
            core::ConstructedValue::Integer(value.extract()?)
        } else if value.is_instance_of::<PyFloat>() {
            let value: f64 = value.extract()?;
            if !value.is_finite() {
                return Err(PyTypeError::new_err("pattern literals must be finite"));
            }
            core::ConstructedValue::Float(value)
        } else if value.is_instance_of::<PyString>() {
            core::ConstructedValue::String(value.extract()?)
        } else {
            return Err(PyTypeError::new_err(
                "pattern literals require a present scalar",
            ));
        };
        Ok(Self::node(core::PatternKind::Literal(value)))
    }
    fn located(&self, start: usize, end: usize) -> PyResult<Self> {
        if start > end {
            return Err(PyTypeError::new_err("pattern range must be ordered"));
        }
        let mut pattern = self.pattern(&mut pattern_budget())?;
        pattern.range = start..end;
        if let core::PatternKind::Name(name) = &mut pattern.value {
            name.range = start..end;
        }
        Ok(Self {
            inner: std::sync::Arc::new(pattern),
        })
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err("patterns have no Python truth value"))
    }
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportMatchArm {
    pattern: ReportPattern,
    guard: Option<core::ReportExpr>,
    value: core::ReportExpr,
}
#[pymethods]
impl ReportMatchArm {
    #[new]
    #[pyo3(signature = (pattern, value, *, guard=None))]
    fn new(
        pattern: &ReportPattern,
        value: &Bound<'_, PyAny>,
        guard: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            pattern: pattern.clone(),
            value: report_expression(value)?,
            guard: guard.map(report_expression).transpose()?,
        })
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "match arms have no Python truth value",
        ))
    }
}

#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportQuery {
    inner: core::ReportQuery,
}
#[pyclass(frozen, module = "zirium.query")]
#[derive(Clone)]
pub(super) struct ReportGenerator {
    pattern: ReportPattern,
    case: bool,
    source: core::ReportExpr,
    filter: Option<core::ReportExpr>,
    bind: bool,
}
impl ReportGenerator {
    fn clause(&self) -> PyResult<core::ReportGenerator> {
        Ok(if self.bind {
            core::ReportGenerator::bind(
                match &self.pattern.inner.value {
                    core::PatternKind::Name(name) => name.value.clone(),
                    _ => unreachable!("bind stores a name pattern"),
                },
                self.source.clone(),
            )
        } else {
            core::ReportGenerator::in_pattern(
                self.pattern.pattern(&mut pattern_budget())?,
                self.source.clone(),
                self.filter.clone(),
                self.case,
            )
        })
    }
}
fn report_expression(value: &Bound<'_, PyAny>) -> PyResult<core::ReportExpr> {
    if value.is_none() {
        return Ok(core::ReportExpr::null());
    }
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
    /// Retain a native selection or handle without projecting it to a scalar.
    #[staticmethod]
    fn native(query: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(query) = query.extract::<PyRef<'_, OpQuery>>() {
            return Ok(Self {
                inner: core::ReportExpr::native(&query.inner),
            });
        }
        if let Ok(query) = query.extract::<PyRef<'_, SsaQuery>>() {
            return Ok(Self {
                inner: core::ReportExpr::native(&query.inner),
            });
        }
        if let Ok(query) = query.extract::<PyRef<'_, CaptureQuery>>() {
            return Ok(Self {
                inner: core::ReportExpr::native(&query.inner),
            });
        }
        Err(PyTypeError::new_err(
            "expected an operation, SSA value or capture query",
        ))
    }
    #[staticmethod]
    fn comprehension(
        head: &Bound<'_, PyAny>,
        clauses: Vec<PyRef<'_, ReportGenerator>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core::ReportExpr::comprehension(
                report_expression(head)?,
                clauses
                    .iter()
                    .map(|clause| clause.clause())
                    .collect::<PyResult<Vec<_>>>()?,
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
    #[staticmethod]
    fn tuple(values: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: core::ReportExpr::tuple(
                values
                    .iter()
                    .map(report_expression)
                    .collect::<PyResult<Vec<_>>>()?,
            ),
        })
    }
    #[staticmethod]
    fn block(
        bindings: Vec<(PyRef<'_, ReportPattern>, Bound<'_, PyAny>)>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let mut budget = pattern_budget();
        let bindings = bindings
            .iter()
            .map(|(pattern, value)| Ok((pattern.pattern(&mut budget)?, report_expression(value)?)))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            inner: core::ReportExpr::block(bindings, report_expression(value)?)
                .map_err(py_error)?,
        })
    }
    #[staticmethod]
    fn conditional(
        condition: &Bound<'_, PyAny>,
        selected: &Bound<'_, PyAny>,
        otherwise: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core::ReportExpr::conditional(
                report_expression(condition)?,
                report_expression(selected)?,
                report_expression(otherwise)?,
            ),
        })
    }
    #[staticmethod]
    fn value_match(
        subject: &Bound<'_, PyAny>,
        arms: Vec<PyRef<'_, ReportMatchArm>>,
    ) -> PyResult<Self> {
        let mut budget = pattern_budget();
        let arms = arms
            .iter()
            .map(|arm| {
                Ok(core::ReportMatchArm {
                    pattern: arm.pattern.pattern(&mut budget)?,
                    guard: arm.guard.clone(),
                    value: arm.value.clone(),
                })
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            inner: core::ReportExpr::value_match(report_expression(subject)?, arms)
                .map_err(py_error)?,
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
    fn occurs(&self) -> Self {
        Self {
            inner: self.inner.clone().occurs(),
        }
    }
    fn absent(&self) -> Self {
        Self {
            inner: self.inner.clone().absent(),
        }
    }
    fn lookup(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().lookup(report_expression(key)?),
        })
    }
    fn has_key(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().has_key(report_expression(key)?),
        })
    }
    fn default(&self, fallback: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().default(report_expression(fallback)?),
        })
    }
    fn contains(&self, fragment: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::Contains, fragment)
    }
    fn starts_with(&self, prefix: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::StartsWith, prefix)
    }
    fn ends_with(&self, suffix: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(core::ValueBinaryOperator::EndsWith, suffix)
    }
    fn parse_int(&self) -> Self {
        Self {
            inner: self.inner.clone().parse_int(),
        }
    }
    fn parse_float(&self) -> Self {
        Self {
            inner: self.inner.clone().parse_float(),
        }
    }
    fn to_string(&self) -> Self {
        Self {
            inner: self.inner.clone().to_string(),
        }
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
    fn count(&self) -> Self {
        Self {
            inner: self.inner.clone().count(),
        }
    }
    fn filter(&self, pattern: &ReportPattern, body: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().filter(
                pattern.pattern(&mut pattern_budget())?,
                report_expression(body)?,
            ),
        })
    }
    fn head(&self, count: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().head(report_expression(count)?),
        })
    }
    fn tail(&self, count: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().tail(report_expression(count)?),
        })
    }
    fn reverse(&self) -> Self {
        Self {
            inner: self.inner.clone().reverse(),
        }
    }
    #[pyo3(signature = (operand=None, message=None))]
    fn check(
        &self,
        operand: Option<&Bound<'_, PyAny>>,
        message: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().check(
                operand.map(report_expression).transpose()?,
                message.map(report_expression).transpose()?,
            ),
        })
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
    #[pyo3(signature = (*, descending=false))]
    fn sort(&self, descending: bool) -> Self {
        Self {
            inner: self.inner.clone().sort(descending),
        }
    }
    #[pyo3(signature = (field, *, descending=false))]
    fn sort_by(&self, field: String, descending: bool) -> Self {
        Self {
            inner: self.inner.clone().sort_by(field, descending),
        }
    }
    #[pyo3(signature = (*, descending = false))]
    fn sort_by_value(&self, descending: bool) -> Self {
        Self {
            inner: self.inner.clone().sort_by_value(descending),
        }
    }

    fn min(&self) -> Self {
        Self {
            inner: self.inner.clone().min(),
        }
    }
    fn max(&self) -> Self {
        Self {
            inner: self.inner.clone().max(),
        }
    }
    fn min_by(&self, field: String) -> Self {
        Self {
            inner: self.inner.clone().min_by(field),
        }
    }
    fn max_by(&self, field: String) -> Self {
        Self {
            inner: self.inner.clone().max_by(field),
        }
    }
    fn group_by(&self, field: String) -> Self {
        Self {
            inner: self.inner.clone().group_by(field),
        }
    }
    fn map(&self, pattern: &ReportPattern, body: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().map(
                pattern.pattern(&mut pattern_budget())?,
                report_expression(body)?,
            ),
        })
    }
    fn flat_map(&self, pattern: &ReportPattern, body: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().flat_map(
                pattern.pattern(&mut pattern_budget())?,
                report_expression(body)?,
            ),
        })
    }
    fn windows(&self, size: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().windows(report_expression(size)?),
        })
    }
    fn runs_by(&self, field: String) -> Self {
        Self {
            inner: self.inner.clone().runs_by(field),
        }
    }
    fn with_fields(&self, fields: &Bound<'_, PyDict>) -> PyResult<Self> {
        let fields = fields
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, report_expression(&value)?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: self.inner.clone().with_fields(fields),
        })
    }
    fn align(&self, after: &Bound<'_, PyAny>, field: String) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().align(report_expression(after)?, field),
        })
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
fn query_parameters(
    declarations: &Bound<'_, PyDict>,
    budget: &mut core::Budget,
) -> PyResult<Vec<core::ParameterDeclaration>> {
    budget
        .charge(declarations.len(), core::ValueLocation::Query(0..0))
        .map_err(py_error)?;
    budget
        .charge_results(declarations.len() + 1, core::ValueLocation::Query(0..0))
        .map_err(py_error)?;
    let mut parameters = Vec::with_capacity(declarations.len());
    for (name, ty) in declarations.iter() {
        let name = parameter_string(&name, budget)?;
        let ty = parameter_string(&ty, budget)?;
        let ty = match ty.as_str() {
            "int" => core::ParameterType::Int,
            "string" => core::ParameterType::String,
            "bool" => core::ParameterType::Bool,
            _ => {
                return Err(PyTypeError::new_err(
                    "parameter type must be int, string or bool",
                ));
            }
        };
        parameters.push(core::ParameterDeclaration {
            name: core::Located::new(name, 0..0),
            ty: core::Located::new(ty, 0..0),
        });
    }
    Ok(parameters)
}
#[pymethods]
impl ReportQuery {
    /// Attach required int/string/bool input declarations to this immutable request.
    fn with_parameters(&self, declarations: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut budget = self.inner.new_budget(core::EvaluationLimits::default());
        let parameters = query_parameters(declarations, &mut budget)?;
        Ok(Self {
            inner: self.inner.with_parameters(parameters),
        })
    }
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
            pattern: ReportPattern::name(name),
            case: false,
            source: report_expression(source)?,
            filter: filter.map(report_expression).transpose()?,
            bind: false,
        })
    }
    #[staticmethod]
    #[pyo3(signature=(pattern, source, filter=None, *, case=false))]
    fn in_pattern(
        pattern: &ReportPattern,
        source: &Bound<'_, PyAny>,
        filter: Option<&Bound<'_, PyAny>>,
        case: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            pattern: pattern.clone(),
            case,
            source: report_expression(source)?,
            filter: filter.map(report_expression).transpose()?,
            bind: false,
        })
    }
    #[staticmethod]
    fn bind(name: String, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            pattern: ReportPattern::name(name),
            case: false,
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
    fn operand(&self, index: usize) -> SsaQuery {
        SsaQuery {
            inner: self.inner.operand(index),
        }
    }
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
    #[pyo3(signature = (edge, *, to, max_depth))]
    fn paths(&self, edge: &str, to: &Predicate, max_depth: usize) -> PyResult<CaptureQuery> {
        if edge != "users" {
            return Err(PyTypeError::new_err("path edge must be 'users'"));
        }
        Ok(CaptureQuery {
            inner: self
                .inner
                .paths(core::PathEdge::Users, to.inner.clone(), max_depth),
        })
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
    fn sort(&self) -> Self {
        Self {
            inner: self.inner.sort(),
        }
    }
    fn sort_by_value(&self) -> Self {
        Self {
            inner: self.inner.sort_by_value(),
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
    fn head_expr(&self, count: &ReportExpr) -> Self {
        Self {
            inner: self.inner.head_expr(count.inner.clone()),
        }
    }
    fn tail_expr(&self, count: &ReportExpr) -> Self {
        Self {
            inner: self.inner.tail_expr(count.inner.clone()),
        }
    }
    fn count(&self) -> CountQuery {
        CountQuery {
            inner: self.inner.count(),
        }
    }
}

fn extract(value: &Bound<'_, PyAny>) -> PyResult<core::QueryExpr<core::QueryOutput>> {
    if let Ok(value) = value.extract::<PyRef<'_, SsaQuery>>() {
        return Ok(value.inner.erased());
    }
    if let Ok(value) = value.extract::<PyRef<'_, CaptureQuery>>() {
        return Ok(value.inner.erased());
    }
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

fn parameter_string(value: &Bound<'_, PyAny>, budget: &mut core::Budget) -> PyResult<String> {
    let value = value.cast::<PyString>().map_err(|_| {
        PyTypeError::new_err("parameter names, types and string values must be str")
    })?;
    let bytes = value
        .len()?
        .checked_mul(4)
        .ok_or_else(|| PyTypeError::new_err("parameter string size overflow"))?;
    budget
        .charge(bytes, core::ValueLocation::Query(0..0))
        .map_err(py_error)?;
    let text = value.to_str()?;
    budget
        .charge(text.len(), core::ValueLocation::Query(0..0))
        .map_err(py_error)?;
    Ok(text.to_owned())
}
fn parameter_values(
    parameters: Option<&Bound<'_, PyDict>>,
    budget: &mut core::Budget,
) -> PyResult<Vec<(String, core::ConstructedValue)>> {
    let Some(parameters) = parameters else {
        return Ok(Vec::new());
    };
    budget
        .charge(parameters.len(), core::ValueLocation::Query(0..0))
        .map_err(py_error)?;
    budget
        .charge_results(parameters.len() + 1, core::ValueLocation::Query(0..0))
        .map_err(py_error)?;
    let mut values = Vec::with_capacity(parameters.len());
    for (name, value) in parameters.iter() {
        let name = parameter_string(&name, budget)?;
        budget
            .charge_results(1, core::ValueLocation::Query(0..0))
            .map_err(py_error)?;
        let value = if value.is_instance_of::<PyBool>() {
            core::ConstructedValue::Boolean(value.extract()?)
        } else if value.is_instance_of::<PyInt>() {
            core::ConstructedValue::Integer(value.extract::<i64>()?)
        } else if value.is_instance_of::<PyString>() {
            core::ConstructedValue::String(parameter_string(&value, budget)?)
        } else {
            return Err(PyTypeError::new_err(
                "parameter values must be bool, int fitting i64 or str",
            ));
        };
        values.push((name, value));
    }
    Ok(values)
}

pub(super) fn evaluate(
    py: Python<'_>,
    document: &Document,
    expression: &Bound<'_, PyAny>,
    parameters: Option<&Bound<'_, PyDict>>,
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
    let parameters = parameter_values(parameters, &mut budget)?;
    let (output, mut budget) = py.detach(move || {
        let document = read_document(&state)?;
        let output = query
            .evaluate_output_with_parameters_and_budget(
                &document,
                registry.registry(),
                core::EvaluationOptions {
                    strict_unknown_references: strict,
                },
                core::ParameterInputs::Typed(&parameters),
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
    native_ref(py, state, &output, budget)
}

fn native_ref(
    py: Python<'_>,
    state: &SharedDocument,
    output: &core::QueryOutput,
    budget: &mut core::Budget,
) -> PyResult<Py<PyAny>> {
    use core::{NativeValue, QueryOutput};
    Ok(match output {
        QueryOutput::Native(NativeValue::Retained(value)) => {
            budget
                .charge(1, zirql_location(&(0..0)))
                .map_err(py_error)?;
            return native_ref(py, state, value, budget);
        }
        QueryOutput::Native(NativeValue::Computational(value)) => {
            return Err(py_error(format!(
                "computational values cannot be converted to Python at query bytes {:?}",
                value.range()
            )));
        }
        QueryOutput::Native(NativeValue::Tuple(value)) => {
            positional_ref(py, state, value.values(), budget, &value.range())?
        }
        QueryOutput::Native(NativeValue::Sequence(value)) => {
            positional_ref(py, state, value.values(), budget, &value.range())?
        }
        QueryOutput::Native(NativeValue::Ssa(value)) => {
            budget
                .charge(1, zirql_location(&(0..0)))
                .map_err(py_error)?;
            budget
                .charge_results(1, zirql_location(&(0..0)))
                .map_err(py_error)?;
            read_document(state)?
                .check_value(*value)
                .map_err(edit_error)?;
            Py::new(
                py,
                SemanticValue {
                    state: state.clone(),
                    value: ValueReference::Resolved(*value),
                },
            )?
            .into_any()
        }
        QueryOutput::Native(NativeValue::Capture(record)) => {
            capture_record_ref(py, state, record, budget, &(0..0))?
        }
        QueryOutput::Native(NativeValue::Captures(value)) => {
            let range = value.range();
            budget
                .charge(value.len() + 1, zirql_location(&range))
                .map_err(py_error)?;
            budget
                .charge_results(value.len() + 1, zirql_location(&range))
                .map_err(py_error)?;
            let list = PyList::empty(py);
            for row in value.rows() {
                // Rust captures were already admitted. Convert borrowed fields directly
                // so native handles do not pass through an uncharged Rust snapshot.
                list.append(capture_record_ref(py, state, row, budget, &range)?)?;
            }
            list.into_any().unbind()
        }
        QueryOutput::Operations(ids) | QueryOutput::Native(NativeValue::OperationSequence(ids)) => {
            let list = PyList::empty(py);
            for &id in ids {
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
        QueryOutput::Native(NativeValue::JsonValue(value)) => {
            json_to_python(py, value, budget, &(0..0))?
        }
        QueryOutput::Native(NativeValue::Constructed(value)) => {
            let json = value.to_json_with_budget(budget).map_err(py_error)?;
            json_to_python(py, &json, budget, &value.range())?
        }
        QueryOutput::Native(NativeValue::String(value)) => {
            value.into_pyobject(py)?.into_any().unbind()
        }
        QueryOutput::Native(NativeValue::Types(ids)) => {
            let list = PyList::empty(py);
            for &id in ids {
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
                        id: Some(*id),
                        name: name.clone(),
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
                dict.set_item(key, native_ref(py, state, value, budget)?)?;
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

fn positional_ref(
    py: Python<'_>,
    state: &SharedDocument,
    values: &[core::QueryOutput],
    budget: &mut core::Budget,
    range: &std::ops::Range<usize>,
) -> PyResult<Py<PyAny>> {
    budget
        .charge(values.len() + 1, zirql_location(range))
        .map_err(py_error)?;
    budget
        .charge_results(values.len() + 1, zirql_location(range))
        .map_err(py_error)?;
    let list = PyList::empty(py);
    for value in values {
        list.append(native_ref(py, state, value, budget)?)?;
    }
    Ok(list.into_any().unbind())
}

fn capture_record_ref(
    py: Python<'_>,
    state: &SharedDocument,
    record: &core::NativeCapture<core::QueryOutput>,
    budget: &mut core::Budget,
    range: &std::ops::Range<usize>,
) -> PyResult<Py<PyAny>> {
    budget
        .charge(record.fields.len() + 1, zirql_location(range))
        .map_err(py_error)?;
    budget
        .charge_results(record.fields.len() + 1, zirql_location(range))
        .map_err(py_error)?;
    let dict = PyDict::new(py);
    for (name, value) in &record.fields {
        budget
            .charge(name.len(), zirql_location(range))
            .map_err(py_error)?;
        budget
            .charge_results(1, zirql_location(range))
            .map_err(py_error)?;
        let mut value = value;
        while let core::QueryOutput::Native(core::NativeValue::Retained(shared)) = value {
            budget.charge(1, zirql_location(range)).map_err(py_error)?;
            value = shared;
        }
        let value = match value {
            core::QueryOutput::Operations(ids) if ids.len() == 1 => {
                budget.charge(1, zirql_location(range)).map_err(py_error)?;
                budget
                    .charge_results(1, zirql_location(range))
                    .map_err(py_error)?;
                read_document(state)?
                    .check_operation(ids[0])
                    .map_err(edit_error)?;
                Py::new(
                    py,
                    SemanticOperation {
                        state: state.clone(),
                        id: ids[0],
                    },
                )?
                .into_any()
            }
            core::QueryOutput::Native(core::NativeValue::OperationSequence(ids)) => {
                budget
                    .charge(ids.len() + 1, zirql_location(range))
                    .map_err(py_error)?;
                budget
                    .charge_results(ids.len() + 1, zirql_location(range))
                    .map_err(py_error)?;
                let list = PyList::empty(py);
                for &id in ids {
                    read_document(state)?
                        .check_operation(id)
                        .map_err(edit_error)?;
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
            core::QueryOutput::Native(core::NativeValue::Ssa(value)) => {
                budget.charge(1, zirql_location(range)).map_err(py_error)?;
                budget
                    .charge_results(1, zirql_location(range))
                    .map_err(py_error)?;
                read_document(state)?
                    .check_value(*value)
                    .map_err(edit_error)?;
                Py::new(
                    py,
                    SemanticValue {
                        state: state.clone(),
                        value: ValueReference::Resolved(*value),
                    },
                )?
                .into_any()
            }
            core::QueryOutput::Native(core::NativeValue::Capture(record)) => {
                capture_record_ref(py, state, record, budget, range)?
            }
            _ => native_ref(py, state, value, budget)?,
        };
        dict.set_item(name, value)?;
    }
    Ok(dict.into_any().unbind())
}

fn json_to_python(
    py: Python<'_>,
    value: &serde_json::Value,
    budget: &mut core::Budget,
    range: &std::ops::Range<usize>,
) -> PyResult<Py<PyAny>> {
    // Python failures remain Python errors; the outer result carries the
    // request's nesting guard without translating those failures.
    budget
        .within_evaluation::<_, core::EvaluationError>(1, zirql_location(range), |budget| {
            Ok(json_node_to_python(py, value, budget, range))
        })
        .map_err(py_error)?
}

fn json_node_to_python(
    py: Python<'_>,
    value: &serde_json::Value,
    budget: &mut core::Budget,
    range: &std::ops::Range<usize>,
) -> PyResult<Py<PyAny>> {
    use serde_json::Value as V;
    let slots = match value {
        V::Array(values) => values.len(),
        V::Object(fields) => fields.len(),
        _ => 0,
    };
    budget
        .charge(slots + 1, zirql_location(range))
        .map_err(py_error)?;
    budget
        .charge_results(slots + 1, zirql_location(range))
        .map_err(py_error)?;
    Ok(match value {
        V::Null => py.None(),
        V::Bool(value) => value.into_pyobject(py)?.to_owned().into_any().unbind(),
        V::Number(value) => {
            if let Some(value) = value.as_i64() {
                value.into_pyobject(py)?.into_any().unbind()
            } else if let Some(value) = value.as_u64() {
                value.into_pyobject(py)?.into_any().unbind()
            } else {
                value
                    .as_f64()
                    .expect("finite JSON number")
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        V::String(value) => {
            charge_python_text(value, budget, range)?;
            value.into_pyobject(py)?.into_any().unbind()
        }
        V::Array(values) => {
            let result = PyList::empty(py);
            for value in values {
                result.append(json_to_python(py, value, budget, range)?)?;
            }
            result.into_any().unbind()
        }
        V::Object(fields) => {
            let result = PyDict::new(py);
            for (label, value) in fields {
                charge_python_text(label, budget, range)?;
                result.set_item(label, json_to_python(py, value, budget, range)?)?;
            }
            result.into_any().unbind()
        }
    })
}

fn charge_python_text(
    text: &str,
    budget: &mut core::Budget,
    range: &std::ops::Range<usize>,
) -> PyResult<()> {
    budget
        .charge(text.len(), zirql_location(range))
        .map_err(py_error)?;
    budget
        .charge_results(text.len(), zirql_location(range))
        .map_err(py_error)
}

fn zirql_location(range: &std::ops::Range<usize>) -> core::ValueLocation {
    core::ValueLocation::Query(range.clone())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SsaQuery>()?;
    module.add_class::<CaptureQuery>()?;
    module.add_class::<MatchBuilder>()?;
    module.add_class::<MatchPredicate>()?;
    module.add_function(wrap_pyfunction!(matches, module)?)?;
    module.add_function(wrap_pyfunction!(bound_ssa, module)?)?;
    module.add_function(wrap_pyfunction!(capture_ops, module)?)?;
    module.add_function(wrap_pyfunction!(capture_ssa, module)?)?;
    module.add_function(wrap_pyfunction!(same_value, module)?)?;
    module.add_class::<ReportExpr>()?;
    module.add_class::<ReportPattern>()?;
    module.add_class::<ReportMatchArm>()?;
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
