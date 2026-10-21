use std::borrow::Cow;
use std::ops::Range;

use super::{EvaluationError, QueryOutput, parser::PrintPart};

pub(super) fn interpolate(
    parts: &[PrintPart],
    bindings: &zirql::Bindings<QueryOutput>,
    state: &mut zirql::Budget,
    range: &Range<usize>,
) -> Result<String, EvaluationError> {
    zirql::expand_fragments(
        parts,
        &mut (),
        state,
        zirql::Location::Query(range.clone()),
        |part, _, budget| {
            Ok(match part {
                PrintPart::Literal(value) => Cow::Borrowed(value.as_str()),
                PrintPart::Binding(name) => match bindings.get(name, range.clone()).map_err(|_| {
                    zirql::EvaluationError::Host(EvaluationError::new(format!(
                        "binding `{name}` is unavailable in this evaluation context"
                    )).at(range.clone()))
                })? {
                    QueryOutput::Count(count) => {
                        let bytes = usize::try_from(count.checked_ilog10().unwrap_or(0))
                            .expect("usize represents the number of decimal digits") + 1;
                        budget.charge(bytes, zirql::Location::Query(range.clone()))?;
                        budget.charge_results(bytes, zirql::Location::Query(range.clone()))?;
                        Cow::Owned(count.to_string())
                    }
                    QueryOutput::Values(values) if values.len() == 1 => {
                        Cow::Borrowed(values[0].as_str())
                    }
                    QueryOutput::Array(values) if values.len() == 1 && values[0].is_string() => {
                        Cow::Borrowed(values[0].as_str().expect("one JSON string"))
                    }
                    _ => {
                        return Err(zirql::EvaluationError::Host(EvaluationError::new(format!(
                            "string interpolation `{{{name}}}` requires a count or exactly one string; project or count the binding first"
                        )).at(range.clone())));
                    }
                },
            })
        },
    ).map_err(super::program::evaluation_error)
}
