use crate::SyntaxKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FormatBinding {
    Operands,
    Value,
    Results,
    Result,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FormatLiteral {
    Colon,
    To,
    Into,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FormatStep {
    Begin(SyntaxKind),
    Capture(FormatBinding),
    AttributeDictionary,
    Literal(FormatLiteral),
    Type(FormatBinding),
    End(SyntaxKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OperationFormat {
    steps: Box<[FormatStep]>,
}

impl OperationFormat {
    pub(crate) fn parse(description: &str) -> Option<Self> {
        let directives = scan(description)?;
        validate(&directives)?;
        Some(Self {
            steps: directives.into_boxed_slice(),
        })
    }

    pub(crate) fn steps(&self) -> &[FormatStep] {
        &self.steps
    }

    pub(crate) fn captures(&self, binding: FormatBinding) -> bool {
        self.steps.contains(&FormatStep::Capture(binding))
    }

    pub(crate) fn identity_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.steps.iter().flat_map(|step| match step {
            FormatStep::Begin(kind) => [0, *kind as u8],
            FormatStep::Capture(binding) => [1, *binding as u8],
            FormatStep::AttributeDictionary => [2, 0],
            FormatStep::Literal(literal) => [3, *literal as u8],
            FormatStep::Type(binding) => [4, *binding as u8],
            FormatStep::End(kind) => [5, *kind as u8],
        })
    }
}

fn scan(description: &str) -> Option<Vec<FormatStep>> {
    let mut raw = Vec::new();
    for word in description.split_ascii_whitespace() {
        let step = match word {
            "$operands" => FormatStep::Capture(FormatBinding::Operands),
            "$value" => FormatStep::Capture(FormatBinding::Value),
            "attr-dict" => FormatStep::AttributeDictionary,
            "`:`" => FormatStep::Literal(FormatLiteral::Colon),
            "`to`" => FormatStep::Literal(FormatLiteral::To),
            "`into`" => FormatStep::Literal(FormatLiteral::Into),
            "type($operands)" => FormatStep::Type(FormatBinding::Operands),
            "type($value)" => FormatStep::Type(FormatBinding::Value),
            "type($results)" => FormatStep::Type(FormatBinding::Results),
            "type($result)" => FormatStep::Type(FormatBinding::Result),
            _ => return None,
        };
        raw.push(step);
    }

    let operand_program = raw.contains(&FormatStep::Capture(FormatBinding::Operands));
    let mut steps = Vec::with_capacity(raw.len() + 4);
    for step in raw {
        if step == FormatStep::Capture(FormatBinding::Value) {
            steps.push(FormatStep::Begin(SyntaxKind::ArithConstantValue));
        }
        if step == FormatStep::Type(FormatBinding::Operands) {
            steps.push(FormatStep::Begin(SyntaxKind::FunctionType));
        }
        if !operand_program && step == FormatStep::Type(FormatBinding::Result) {
            steps.push(FormatStep::Begin(SyntaxKind::FunctionType));
        }
        steps.push(step);
        if step == FormatStep::Type(FormatBinding::Value) {
            steps.push(FormatStep::End(SyntaxKind::ArithConstantValue));
        }
        if matches!(
            step,
            FormatStep::Type(FormatBinding::Results | FormatBinding::Result)
        ) && steps.contains(&FormatStep::Begin(SyntaxKind::FunctionType))
        {
            steps.push(FormatStep::End(SyntaxKind::FunctionType));
        }
    }
    Some(steps)
}

fn validate(steps: &[FormatStep]) -> Option<()> {
    let directive_count = steps
        .iter()
        .filter(|step| !matches!(step, FormatStep::Begin(_) | FormatStep::End(_)))
        .count();
    if directive_count != 6 {
        return None;
    }
    let operand_capture = steps
        .iter()
        .position(|step| *step == FormatStep::Capture(FormatBinding::Operands));
    let value_capture = steps
        .iter()
        .position(|step| *step == FormatStep::Capture(FormatBinding::Value));
    if operand_capture.is_some() == value_capture.is_some() {
        return None;
    }
    if steps
        .iter()
        .filter(|step| matches!(step, FormatStep::AttributeDictionary))
        .count()
        != 1
    {
        return None;
    }

    if let Some(capture) = operand_capture {
        let attributes = position(steps, FormatStep::AttributeDictionary)?;
        let operand_type = position(steps, FormatStep::Type(FormatBinding::Operands))?;
        let result_type = position(steps, FormatStep::Type(FormatBinding::Results))?;
        let colon = position(steps, FormatStep::Literal(FormatLiteral::Colon))?;
        let separators = steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| {
                matches!(
                    step,
                    FormatStep::Literal(FormatLiteral::To | FormatLiteral::Into)
                )
                .then_some(index)
            })
            .collect::<Vec<_>>();
        (capture < attributes
            && attributes < colon
            && colon < operand_type
            && separators.len() == 1
            && operand_type < separators[0]
            && separators[0] < result_type)
            .then_some(())
    } else {
        let capture = value_capture?;
        let attributes = position(steps, FormatStep::AttributeDictionary)?;
        let value_type = position(steps, FormatStep::Type(FormatBinding::Value))?;
        let result_type = position(steps, FormatStep::Type(FormatBinding::Result))?;
        let colons = steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| {
                (*step == FormatStep::Literal(FormatLiteral::Colon)).then_some(index)
            })
            .collect::<Vec<_>>();
        (colons.len() == 2
            && capture < colons[0]
            && colons[0] < value_type
            && value_type < attributes
            && attributes < colons[1]
            && colons[1] < result_type)
            .then_some(())
    }
}

fn position(steps: &[FormatStep], needle: FormatStep) -> Option<usize> {
    let mut positions = steps
        .iter()
        .enumerate()
        .filter_map(|(index, step)| (*step == needle).then_some(index));
    let position = positions.next()?;
    positions.next().is_none().then_some(position)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_supported_formats_and_rejects_invalid_bindings() {
        let to =
            OperationFormat::parse("$operands attr-dict `:` type($operands) `to` type($results)")
                .unwrap();
        let into =
            OperationFormat::parse("$operands attr-dict `:` type($operands) `into` type($results)")
                .unwrap();
        assert_ne!(to, into);
        assert_ne!(
            to.identity_bytes().collect::<Vec<_>>(),
            into.identity_bytes().collect::<Vec<_>>()
        );
        assert!(
            OperationFormat::parse("$value `:` type($value) attr-dict `:` type($result)").is_some()
        );
        assert!(OperationFormat::parse("$value `:` type($operands)").is_none());
        assert!(OperationFormat::parse("$operands `to` type($results)").is_none());
        assert!(
            OperationFormat::parse(
                "$operands attr-dict `:` type($operands) `to` `into` type($results)"
            )
            .is_none()
        );
    }
}
