use crate::{SyntaxKind, lexer, source::Source};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FormatCapture {
    Operands,
    Operand(usize),
    Value,
    Callee,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FormatTarget {
    Operands,
    Operand(usize),
    Results,
    Result,
    Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FormatTypeCapture {
    targets: Box<[FormatTarget]>,
    per_operand_list: bool,
}

impl FormatTypeCapture {
    pub(crate) fn targets(&self) -> &[FormatTarget] {
        &self.targets
    }
    pub(crate) fn is_per_operand_list(&self) -> bool {
        self.per_operand_list
    }
    pub(crate) fn accepts_parenthesized_list(&self) -> bool {
        !self.per_operand_list
            && self.targets.len() == 1
            && matches!(
                self.targets[0],
                FormatTarget::Operands | FormatTarget::Results
            )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FormatStep {
    Begin(SyntaxKind),
    Capture(FormatCapture),
    AttributeDictionary,
    Literal(String),
    Type(FormatTypeCapture),
    End(SyntaxKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OperationFormat {
    description: String,
    steps: Box<[FormatStep]>,
    types: Box<[FormatTypeCapture]>,
}

impl OperationFormat {
    pub(crate) fn parse(description: &str) -> Result<Self, String> {
        let raw = scan(description)?;
        validate(&raw)?;
        let types = raw
            .iter()
            .filter_map(|step| match step {
                FormatStep::Type(capture) => Some(capture.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let typed_value = types
            .iter()
            .any(|capture| capture.targets().contains(&FormatTarget::Value));
        let mut steps = Vec::with_capacity(raw.len() + types.len() * 2 + 2);
        for step in raw {
            match step {
                FormatStep::Capture(FormatCapture::Value) => {
                    steps.push(FormatStep::Begin(SyntaxKind::ArithConstantValue));
                    steps.push(FormatStep::Capture(FormatCapture::Value));
                    if !typed_value {
                        steps.push(FormatStep::End(SyntaxKind::ArithConstantValue));
                    }
                }
                FormatStep::Type(capture) => {
                    let closes_value = capture.targets().contains(&FormatTarget::Value);
                    steps.push(FormatStep::Begin(SyntaxKind::FunctionType));
                    steps.push(FormatStep::Type(capture));
                    steps.push(FormatStep::End(SyntaxKind::FunctionType));
                    if closes_value {
                        steps.push(FormatStep::End(SyntaxKind::ArithConstantValue));
                    }
                }
                step => steps.push(step),
            }
        }
        Ok(Self {
            description: description.to_owned(),
            steps: steps.into_boxed_slice(),
            types,
        })
    }

    pub(crate) fn description(&self) -> &str {
        &self.description
    }
    pub(crate) fn steps(&self) -> &[FormatStep] {
        &self.steps
    }
    pub(crate) fn types(&self) -> &[FormatTypeCapture] {
        &self.types
    }
    pub(crate) fn captures_value(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step, FormatStep::Capture(FormatCapture::Value)))
    }
    pub(crate) fn captures_callee(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step, FormatStep::Capture(FormatCapture::Callee)))
    }
    pub(crate) fn identity_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.description.bytes()
    }
}

fn scan(description: &str) -> Result<Vec<FormatStep>, String> {
    let mut elements = Vec::new();
    let mut offset = 0;
    while offset < description.len() {
        offset += description[offset..]
            .chars()
            .take_while(|c| c.is_ascii_whitespace())
            .map(char::len_utf8)
            .sum::<usize>();
        if offset == description.len() {
            break;
        }
        let rest = &description[offset..];
        if let Some(literal_body) = rest.strip_prefix('`') {
            let end = literal_body
                .find('`')
                .map(|i| i + 1)
                .ok_or_else(|| format!("unterminated literal at byte {offset}"))?;
            let literal = &rest[1..end];
            validate_literal(literal, offset + 1)?;
            elements.push(FormatStep::Literal(literal.to_owned()));
            offset += end + 1;
            continue;
        }
        if rest.starts_with("type(") || rest.starts_with("types(") {
            let per_operand_list = rest.starts_with("types(");
            let open = if per_operand_list { 5 } else { 4 };
            let close = rest[open + 1..]
                .find(')')
                .map(|i| open + 1 + i)
                .ok_or_else(|| format!("unterminated type directive at byte {offset}"))?;
            let targets = parse_targets(&rest[open + 1..close], offset + open + 1)?;
            if per_operand_list && targets.as_slice() != [FormatTarget::Operands] {
                return Err(format!(
                    "types(...) only accepts $operands at byte {offset}"
                ));
            }
            elements.push(FormatStep::Type(FormatTypeCapture {
                targets: targets.into_boxed_slice(),
                per_operand_list,
            }));
            offset += close + 1;
            continue;
        }
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let word = &rest[..end];
        let step = if word == "$operands" {
            FormatStep::Capture(FormatCapture::Operands)
        } else if let Some(index) = parse_operand(word) {
            FormatStep::Capture(FormatCapture::Operand(index))
        } else {
            match word {
                "$value" => FormatStep::Capture(FormatCapture::Value),
                "$callee" => FormatStep::Capture(FormatCapture::Callee),
                "attr-dict" => FormatStep::AttributeDictionary,
                _ => return Err(format!("unknown directive {word:?} at byte {offset}")),
            }
        };
        elements.push(step);
        offset += end;
    }
    if elements.is_empty() {
        return Err("operation format must not be empty".into());
    }
    Ok(elements)
}

fn validate_literal(literal: &str, offset: usize) -> Result<(), String> {
    if literal.is_empty() || literal.bytes().any(|b| b.is_ascii_whitespace()) {
        return Err(format!(
            "invalid empty or whitespace literal at byte {offset}"
        ));
    }
    let source = Source::new(literal.as_bytes())
        .map_err(|_| format!("invalid literal {literal:?} at byte {offset}"))?;
    let lexed = lexer::lex(&source);
    if !lexed.diagnostics().is_empty() || lexed.tokens().len() != 2 {
        return Err(format!(
            "literal {literal:?} must be exactly one MLIR token at byte {offset}"
        ));
    }
    Ok(())
}

fn parse_targets(text: &str, offset: usize) -> Result<Vec<FormatTarget>, String> {
    let mut targets = Vec::new();
    for raw in text.split(',') {
        let target = raw.trim();
        targets.push(match target {
            "$operands" => FormatTarget::Operands,
            "$results" => FormatTarget::Results,
            "$result" => FormatTarget::Result,
            "$value" => FormatTarget::Value,
            _ => parse_operand(target)
                .map(FormatTarget::Operand)
                .ok_or_else(|| format!("unknown type target {target:?} at byte {offset}"))?,
        });
    }
    if targets.is_empty()
        || targets
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != targets.len()
    {
        return Err(format!("empty or duplicate type targets at byte {offset}"));
    }
    Ok(targets)
}

fn parse_operand(text: &str) -> Option<usize> {
    text.strip_prefix("$operands[")?
        .strip_suffix(']')?
        .parse()
        .ok()
}

fn validate(steps: &[FormatStep]) -> Result<(), String> {
    let mut variadic_operands = false;
    let mut indexed_operands = Vec::new();
    let mut value = false;
    let mut callee = false;
    let mut value_count = 0;
    let mut callee_count = 0;
    let mut attributes = 0;
    for step in steps {
        match step {
            FormatStep::Capture(FormatCapture::Operands) => variadic_operands = true,
            FormatStep::Capture(FormatCapture::Operand(index)) => indexed_operands.push(*index),
            FormatStep::Capture(FormatCapture::Value) => {
                value = true;
                value_count += 1;
            }
            FormatStep::Capture(FormatCapture::Callee) => {
                callee = true;
                callee_count += 1;
            }
            FormatStep::AttributeDictionary => attributes += 1,
            _ => {}
        }
    }
    if attributes > 1 {
        return Err("attr-dict may appear at most once".into());
    }
    if value_count > 1 || callee_count > 1 {
        return Err("$value and $callee may each appear at most once".into());
    }
    if variadic_operands && !indexed_operands.is_empty() {
        return Err("$operands cannot be mixed with indexed operands".into());
    }
    if variadic_operands
        && steps
            .iter()
            .filter(|s| matches!(s, FormatStep::Capture(FormatCapture::Operands)))
            .count()
            != 1
    {
        return Err("$operands may appear only once".into());
    }
    if !indexed_operands.is_empty()
        && indexed_operands != (0..indexed_operands.len()).collect::<Vec<_>>()
    {
        return Err("indexed operands must appear once in order starting at zero".into());
    }
    if value && (variadic_operands || !indexed_operands.is_empty() || callee) {
        return Err("$value cannot be combined with SSA operands or $callee".into());
    }
    if value {
        let value_position = steps
            .iter()
            .position(|step| matches!(step, FormatStep::Capture(FormatCapture::Value)))
            .expect("value capture was observed");
        if let Some(type_position) = steps.iter().position(|step| {
            matches!(step, FormatStep::Type(capture) if capture.targets().contains(&FormatTarget::Value))
        }) && (type_position <= value_position
            || steps[value_position + 1..type_position]
                .iter()
                .any(|step| !matches!(step, FormatStep::Literal(_))))
        {
            return Err("type($value) must follow $value with only literals between them".into());
        }
    }
    let has_operands = variadic_operands || !indexed_operands.is_empty();
    let mut assigned = std::collections::BTreeSet::new();
    for capture in steps.iter().filter_map(|s| match s {
        FormatStep::Type(c) => Some(c),
        _ => None,
    }) {
        for target in capture.targets() {
            match target {
                FormatTarget::Operands if !has_operands => {
                    return Err("type($operands) requires an operand capture".into());
                }
                FormatTarget::Operand(index)
                    if !variadic_operands && !indexed_operands.contains(index) =>
                {
                    return Err(format!("type target $operands[{index}] is not captured"));
                }
                FormatTarget::Value if !value => {
                    return Err("type($value) requires a $value capture".into());
                }
                _ => {}
            }
            let overlaps = match target {
                FormatTarget::Operands => assigned.iter().any(|target| {
                    matches!(target, FormatTarget::Operands | FormatTarget::Operand(_))
                }),
                FormatTarget::Operand(_) => assigned.contains(&FormatTarget::Operands),
                FormatTarget::Results | FormatTarget::Result => assigned
                    .iter()
                    .any(|target| matches!(target, FormatTarget::Results | FormatTarget::Result)),
                FormatTarget::Value => assigned.contains(&FormatTarget::Value),
            };
            if overlaps {
                return Err(format!(
                    "type target {target:?} overlaps an earlier assignment"
                ));
            }
            if !assigned.insert(target.clone()) {
                return Err(format!("type target {target:?} is assigned more than once"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiles_legacy_and_composed_formats() {
        for format in [
            "$operands attr-dict `:` type($operands) `to` type($results)",
            "$operands attr-dict `:` type($operands) `into` type($results)",
            "$operands attr-dict `:` type($operands) `->` type($results)",
            "$value `:` type($value) attr-dict `:` type($result)",
            "$operands[0] `,` $operands[1] `:` type($operands) `->` type($results)",
            "$callee attr-dict",
            "$operands attr-dict `:` types($operands) `->` type($results)",
        ] {
            OperationFormat::parse(format).unwrap();
        }
    }
    #[test]
    fn rejects_inconsistent_bindings() {
        for format in [
            "",
            "$value type($operands)",
            "$operands $operands",
            "$operands $operands[0]",
            "$operands[1]",
            "$operands attr-dict attr-dict",
            "$callee type($value)",
            "$operands types($results)",
        ] {
            assert!(OperationFormat::parse(format).is_err(), "accepted {format}");
        }
    }
}
