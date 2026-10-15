//! The existing diff surface frontend, lowered once into shared expressions.
use super::diff_program::{
    DiffProgram, DiffStage, Expression, Leaf, Order, OwnedExpression, SetOperator,
    dispose_statements, sequence, stream,
};
use super::{EvaluationError, OperationTraversal};
use std::ops::Range;
use zirql::{ExpressionKind, Located, PredicateKind, Stage};

type DiffSetOperator = SetOperator;

pub(super) fn parse(source: &str) -> Result<DiffProgram, EvaluationError> {
    DiffProgram::check_read_only(source)?;
    let mut statements = Vec::new();
    if let Err(error) = parse_statements(source, &mut statements) {
        dispose_statements(statements);
        return Err(error);
    }
    Ok(DiffProgram {
        program: zirql::Program::new(statements, 0..source.len()),
    })
}

fn parse_statements(
    source: &str,
    statements: &mut Vec<Located<zirql::Statement<DiffStage, SetOperator, ()>>>,
) -> Result<(), EvaluationError> {
    for statement in split_diff_statements(source).map_err(EvaluationError::new)? {
        let statement = statement.trim();
        if statement.is_empty() {
            continue;
        }
        let range = location(source, statement);
        let value = if let Some(expression) = statement.strip_prefix("do ") {
            zirql::Statement::Evaluate {
                expression: parse_expression(source, expression, 0)?,
                implicit_output: None,
            }
        } else if let Some((name, expression)) = split_diff_binding(statement) {
            zirql::Statement::Bind {
                name: Located::new(name.to_owned(), location(source, name)),
                expression: parse_expression(source, expression, 0)?,
            }
        } else {
            zirql::Statement::Evaluate {
                expression: parse_expression(source, statement, 0)?,
                implicit_output: Some(()),
            }
        };
        statements.push(Located::new(value, range));
    }
    Ok(())
}

fn location(source: &str, part: &str) -> Range<usize> {
    let start = part.as_ptr() as usize - source.as_ptr() as usize;
    start..start + part.len()
}
fn parse_expression(source: &str, part: &str, depth: usize) -> Result<Expression, EvaluationError> {
    let range = location(source, part);
    if depth > super::parser::DEFAULT_NESTING_LIMIT {
        return Err(EvaluationError::new("query nesting limit exceeded").at(range));
    }
    let mut sets = split_diff_sets(part)
        .map_err(|error| EvaluationError::new(error).at(range.clone()))?
        .into_iter();
    let (_, first) = sets
        .next()
        .ok_or_else(|| EvaluationError::new("empty diff expression").at(range.clone()))?;
    let mut expression = OwnedExpression::new(parse_pipeline(source, first, depth)?);
    for (operator, pipeline) in sets {
        let right = parse_pipeline(source, pipeline, depth)?;
        expression = OwnedExpression::new(Located::new(
            ExpressionKind::Set {
                operator: operator.expect("later set part"),
                left: Box::new(expression.into_inner()),
                right: Box::new(right),
            },
            range.clone(),
        ));
    }
    Ok(expression.into_inner())
}

fn parse_pipeline(source: &str, part: &str, depth: usize) -> Result<Expression, EvaluationError> {
    let mut stages = Vec::new();
    for stage in split_diff_pipeline(part).map_err(EvaluationError::new)? {
        let stage = stage.trim();
        let range = location(source, stage);
        let compiled = parse_stage(source, stage, depth, &range).map_err(|mut error| {
            if error.range.is_none() {
                error.range = Some(range.clone());
            }
            error
        })?;
        stages.push(Located::new(compiled, range));
    }
    Ok(sequence(stages, location(source, part)))
}
fn parse_stage(
    source: &str,
    stage: &str,
    depth: usize,
    range: &Range<usize>,
) -> Result<DiffStage, EvaluationError> {
    let error = |message: &str| EvaluationError::new(message).at(range.clone());
    let arguments =
        |name: &str| split_diff_call_arguments(stage, name).map_err(EvaluationError::new);
    let one = |name: &str| -> Result<&str, EvaluationError> {
        let arguments = arguments(name)?;
        if arguments.len() != 1 || arguments[0].is_empty() {
            return Err(error(&format!("{name} requires one argument")));
        }
        Ok(arguments[0])
    };
    let number = |name: &str| {
        one(name)?
            .parse::<usize>()
            .map_err(|_| error(&format!("{name} requires a non-negative item count")))
    };
    if let Some(group) = stage
        .strip_prefix('(')
        .and_then(|stage| stage.strip_suffix(')'))
    {
        return Ok(DiffStage::Nested(Box::new(OwnedExpression::new(
            Located::new(
                ExpressionKind::Group(Box::new(parse_expression(source, group, depth + 1)?)),
                range.clone(),
            ),
        ))));
    }
    if stage.starts_with("filter(") {
        let predicate = DiffPredicateParser::parse(one("filter")?, range.clone())?;
        return stream(Stage::Filter(predicate), range.clone());
    }
    Ok(match stage {
        "input" => DiffStage::Input,
        "before" => DiffStage::Project(crate::diff::DiffSide::Before),
        "after" => DiffStage::Project(crate::diff::DiffSide::After),
        "users" => DiffStage::Users(None),
        "defs" => DiffStage::Defs(None),
        "parent" => DiffStage::Parent,
        "children" => DiffStage::Children,
        "subtree" => DiffStage::Subtree,
        "reachable" => DiffStage::Graph(OperationTraversal::Reachable),
        "closure" => DiffStage::Graph(OperationTraversal::Closure),
        "slice" | "backward_slice" => DiffStage::Graph(OperationTraversal::Slice),
        "forward_slice" => DiffStage::Graph(OperationTraversal::ForwardSlice),
        "unique" => {
            let DiffStage::Stream(pipeline) = stream(Stage::Unique, range.clone())? else {
                unreachable!("stream builds a native pipeline")
            };
            DiffStage::SelectionUnique(pipeline)
        }
        "count" => DiffStage::Count,
        "tally" => return stream(Stage::Tally, range.clone()),
        "names" => DiffStage::Names,
        "result_types" => DiffStage::ResultTypes,
        "operand_types" => DiffStage::OperandTypes,
        "reverse" => DiffStage::Reverse,
        "sort" => DiffStage::Sort,
        "min" | "max" | "min_all" | "max_all" => DiffStage::Extreme {
            maximum: stage.starts_with("max"),
            all: stage.ends_with("_all"),
        },
        "check" => DiffStage::Check(None),
        "emit" => DiffStage::Noop,
        "json" => DiffStage::Json,
        "markdown" => DiffStage::Markdown,
        _ if stage.starts_with("users(") => DiffStage::Users(Some(number("users")?)),
        _ if stage.starts_with("defs(") => DiffStage::Defs(Some(number("defs")?)),
        _ if stage.starts_with("head(") => DiffStage::Head(number("head")?),
        _ if stage.starts_with("tail(") => DiffStage::Tail(number("tail")?),
        _ if stage.starts_with("check(") => DiffStage::Check(Some(number("check")?)),
        _ if stage.starts_with("attr(") => {
            let names = extract_string_calls(stage, "attr");
            if names.len() != 1 {
                return Err(error("attr requires one string argument"));
            }
            DiffStage::Attr(names.into_iter().next().unwrap())
        }
        _ if stage.starts_with("root(") => {
            let names = extract_string_calls(stage, "op");
            if names.len() != 1 {
                return Err(error("root in diff mode requires op(\"name\")"));
            }
            let predicate = Located::new(
                PredicateKind::Leaf(Leaf::Op(names.into_iter().next().unwrap())),
                range.clone(),
            );
            let DiffStage::Stream(filter) = stream(Stage::Filter(predicate), range.clone())? else {
                unreachable!("stream builds a native pipeline")
            };
            DiffStage::Root(filter)
        }
        _ if stage.starts_with("where_exists(") => DiffStage::WhereExists(
            super::diff_program::nested(parse_expression(source, one("where_exists")?, depth + 1)?),
        ),
        _ if stage.starts_with("fixpoint(") => DiffStage::Fixpoint(super::diff_program::nested(
            parse_expression(source, one("fixpoint")?, depth + 1)?,
        )),
        _ if [
            "sort_by(",
            "min_by(",
            "max_by(",
            "min_all_by(",
            "max_all_by(",
        ]
        .iter()
        .any(|prefix| stage.starts_with(prefix)) =>
        {
            let name = stage.split_once('(').unwrap().0;
            let kind = match name {
                "sort_by" => Order::Sort,
                "min_by" => Order::Min,
                "max_by" => Order::Max,
                "min_all_by" => Order::MinAll,
                "max_all_by" => Order::MaxAll,
                _ => unreachable!("matched ordering stage"),
            };
            if name == "sort_by" && one(name)?.trim() == "value" {
                DiffStage::SortMapValue
            } else {
                DiffStage::Order {
                    kind,
                    selector: super::diff_program::nested(parse_expression(
                        source,
                        one(name)?,
                        depth + 1,
                    )?),
                }
            }
        }
        _ if stage.starts_with("map_by(") => {
            let arguments = arguments("map_by")?;
            if arguments.len() != 2 {
                return Err(error("map_by requires a key and value expression"));
            }
            DiffStage::MapBy {
                key: super::diff_program::nested(parse_expression(
                    source,
                    arguments[0],
                    depth + 1,
                )?),
                value: super::diff_program::nested(parse_expression(
                    source,
                    arguments[1],
                    depth + 1,
                )?),
            }
        }
        _ if stage.starts_with("paths(") => {
            return Err(error("native paths are unavailable on diff queries"));
        }
        _ if stage.starts_with("print(") => {
            DiffStage::Print(parse_diff_string(one("print")?).map_err(EvaluationError::new)?)
        }
        _ if !stage.is_empty()
            && stage.bytes().enumerate().all(|(i, b)| {
                b == b'_' || b.is_ascii_alphabetic() || i > 0 && b.is_ascii_digit()
            }) =>
        {
            DiffStage::Nested(Box::new(OwnedExpression::new(Located::new(
                ExpressionKind::Binding(stage.to_owned()),
                range.clone(),
            ))))
        }
        _ => {
            return Err(error(&format!(
                "unknown or unsupported diff query stage `{stage}`"
            )));
        }
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DiffPredicateToken {
    Identifier(String),
    String(String),
    Left,
    Right,
    Comma,
}

struct DiffPredicateParser {
    tokens: Vec<DiffPredicateToken>,
    cursor: usize,
    range: Range<usize>,
}
impl DiffPredicateParser {
    fn parse(source: &str, range: Range<usize>) -> Result<zirql::Predicate<Leaf>, EvaluationError> {
        let mut parser = Self {
            tokens: lex_diff_predicate(source).map_err(EvaluationError::new)?,
            cursor: 0,
            range,
        };
        let predicate = parser.or(0)?;
        if parser.cursor != parser.tokens.len() {
            return Err(parser.error("unexpected token in diff predicate"));
        }
        Ok(predicate)
    }
    fn error(&self, message: &str) -> EvaluationError {
        EvaluationError::new(message).at(self.range.clone())
    }
    fn predicate(&self, kind: PredicateKind<Leaf>) -> zirql::Predicate<Leaf> {
        Located::new(kind, self.range.clone())
    }
    fn or(&mut self, depth: usize) -> Result<zirql::Predicate<Leaf>, EvaluationError> {
        let mut predicates = vec![self.and(depth)?];
        while self.take_identifier("or") {
            predicates.push(self.and(depth)?);
        }
        Ok(if predicates.len() == 1 {
            predicates.pop().unwrap()
        } else {
            self.predicate(PredicateKind::Or(predicates))
        })
    }
    fn and(&mut self, depth: usize) -> Result<zirql::Predicate<Leaf>, EvaluationError> {
        let mut predicates = vec![self.unary(depth)?];
        while self.take_identifier("and") {
            predicates.push(self.unary(depth)?);
        }
        Ok(if predicates.len() == 1 {
            predicates.pop().unwrap()
        } else {
            self.predicate(PredicateKind::And(predicates))
        })
    }
    fn unary(&mut self, depth: usize) -> Result<zirql::Predicate<Leaf>, EvaluationError> {
        if depth >= super::parser::DEFAULT_NESTING_LIMIT {
            return Err(self.error("predicate nesting limit exceeded"));
        }
        if self.take_identifier("not") {
            let inner = self.unary(depth + 1)?;
            return Ok(self.predicate(PredicateKind::Not(Box::new(inner))));
        }
        if self.take(&DiffPredicateToken::Left) {
            let inner = self.or(depth + 1)?;
            self.expect(DiffPredicateToken::Right)?;
            return Ok(self.predicate(PredicateKind::Group(Box::new(inner))));
        }
        let name = match self.tokens.get(self.cursor).cloned() {
            Some(DiffPredicateToken::Identifier(name)) => name,
            _ => return Err(self.error("expected a diff predicate")),
        };
        self.cursor += 1;
        if name == "true" || name == "false" {
            return Ok(self.predicate(PredicateKind::Leaf(Leaf::Bool(name == "true"))));
        }
        self.expect(DiffPredicateToken::Left)?;
        let first = self.string()?;
        let second = if self.take(&DiffPredicateToken::Comma) {
            Some(self.string()?)
        } else {
            None
        };
        self.expect(DiffPredicateToken::Right)?;
        let leaf = match (name.as_str(), second) {
            ("change", None) => Leaf::Change(
                first
                    .parse()
                    .map_err(|error: crate::diff::DiffError| self.error(&error.to_string()))?,
            ),
            ("changed", None) => Leaf::Changed(
                first
                    .parse()
                    .map_err(|error: crate::diff::DiffError| self.error(&error.to_string()))?,
            ),
            ("op", None) => Leaf::Op(first),
            ("dialect", None) => Leaf::Dialect(first),
            ("result_type", None) => Leaf::ResultType(first),
            ("has_attr", None) => Leaf::HasAttr(first),
            ("attr", Some(second)) => Leaf::Attr(first, second),
            _ => return Err(self.error(&format!("unsupported diff predicate `{name}`"))),
        };
        Ok(self.predicate(PredicateKind::Leaf(leaf)))
    }
    fn string(&mut self) -> Result<String, EvaluationError> {
        match self.tokens.get(self.cursor).cloned() {
            Some(DiffPredicateToken::String(value)) => {
                self.cursor += 1;
                Ok(value)
            }
            _ => Err(self.error("expected a string argument in diff predicate")),
        }
    }
    fn take_identifier(&mut self, expected: &str) -> bool {
        if matches!(self.tokens.get(self.cursor),Some(DiffPredicateToken::Identifier(value)) if value==expected)
        {
            self.cursor += 1;
            true
        } else {
            false
        }
    }
    fn take(&mut self, expected: &DiffPredicateToken) -> bool {
        if self.tokens.get(self.cursor) == Some(expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, expected: DiffPredicateToken) -> Result<(), EvaluationError> {
        if self.take(&expected) {
            Ok(())
        } else {
            Err(self.error("malformed diff predicate"))
        }
    }
}

fn split_diff_sets(source: &str) -> Result<Vec<(Option<DiffSetOperator>, &str)>, String> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    let mut start = 0usize;
    let mut operator = None;
    let mut result = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            cursor += 1;
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'(' | b'{' | b'[' => {
                depth += 1;
                if depth > super::parser::DEFAULT_NESTING_LIMIT {
                    return Err("query nesting limit exceeded".into());
                }
            }
            b')' | b'}' | b']' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or("unmatched delimiter in diff query")?;
            }
            byte if depth == 0 && (byte.is_ascii_alphabetic() || byte == b'_') => {
                let word_start = cursor;
                cursor += 1;
                while cursor < bytes.len()
                    && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
                {
                    cursor += 1;
                }
                let next = DiffSetOperator::from_keyword(&source[word_start..cursor]);
                if let Some(next) = next {
                    let pipeline = source[start..word_start].trim();
                    if pipeline.is_empty() {
                        return Err("set operator requires a left selection".into());
                    }
                    result.push((operator, pipeline));
                    operator = Some(next);
                    start = cursor;
                }
                continue;
            }
            _ => {}
        }
        cursor += 1;
    }
    if quoted || depth != 0 {
        return Err("unterminated string or group in diff query".into());
    }
    let pipeline = source[start..].trim();
    if pipeline.is_empty() {
        return Err("set operator requires a right selection".into());
    }
    result.push((operator, pipeline));
    Ok(result)
}

fn split_diff_call_arguments<'a>(stage: &'a str, name: &str) -> Result<Vec<&'a str>, String> {
    let prefix = format!("{name}(");
    let inner = stage
        .strip_prefix(&prefix)
        .and_then(|value| value.strip_suffix(')'))
        .ok_or_else(|| format!("invalid {name} stage"))?;
    let bytes = inner.as_bytes();
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    let mut start = 0usize;
    let mut arguments = Vec::new();
    for (cursor, byte) in bytes.iter().copied().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'(' | b'{' | b'[' => {
                depth += 1;
                if depth > super::parser::DEFAULT_NESTING_LIMIT {
                    return Err("query nesting limit exceeded".into());
                }
            }
            b')' | b'}' | b']' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| format!("unmatched delimiter in {name}"))?;
            }
            b',' if depth == 0 => {
                arguments.push(inner[start..cursor].trim());
                start = cursor + 1;
            }
            _ => {}
        }
    }
    if quoted || depth != 0 {
        return Err(format!("unterminated string or group in {name}"));
    }
    arguments.push(inner[start..].trim());
    Ok(arguments)
}

fn parse_diff_string(source: &str) -> Result<String, String> {
    serde_json::from_str(source).map_err(|_| "expected a quoted UTF-8 string".into())
}

pub(super) fn reject_diff_mutations(source: &str) -> Result<(), String> {
    let mut quoted = false;
    let mut escaped = false;
    let bytes = source.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            cursor += 1;
            continue;
        }
        if byte == b'"' {
            quoted = true;
            cursor += 1;
            continue;
        }
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
            {
                cursor += 1;
            }
            if matches!(&source[start..cursor], "set_attr" | "remove_attr") {
                return Err("diff queries are read-only".into());
            }
        } else {
            cursor += 1;
        }
    }
    Ok(())
}

fn split_diff_statements(source: &str) -> Result<Vec<&str>, String> {
    split_diff_top_level(source, b';')
}

fn split_diff_binding(source: &str) -> Option<(&str, &str)> {
    let parts = split_diff_top_level(source, b'=').ok()?;
    if parts.len() != 2 {
        return None;
    }
    let name = parts[0].trim();
    if name.is_empty()
        || !name.bytes().enumerate().all(|(index, byte)| {
            byte == b'_' || byte.is_ascii_alphabetic() || (index > 0 && byte.is_ascii_digit())
        })
    {
        return None;
    }
    Some((name, parts[1].trim()))
}

fn split_diff_top_level(source: &str, delimiter: u8) -> Result<Vec<&str>, String> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    let mut start = 0usize;
    let mut result = Vec::new();
    for (index, byte) in source.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'(' | b'{' | b'[' => {
                depth += 1;
                if depth > super::parser::DEFAULT_NESTING_LIMIT {
                    return Err("query nesting limit exceeded".into());
                }
            }
            b')' | b'}' | b']' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or("unmatched closing delimiter in diff query")?;
            }
            byte if byte == delimiter && depth == 0 => {
                result.push(&source[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if quoted || depth != 0 {
        return Err("unterminated string or group in diff query".into());
    }
    result.push(&source[start..]);
    Ok(result)
}

fn split_diff_pipeline(source: &str) -> Result<Vec<&str>, String> {
    split_diff_top_level(source, b'|')
}

fn lex_diff_predicate(source: &str) -> Result<Vec<DiffPredicateToken>, String> {
    let bytes = source.as_bytes();
    let mut cursor = 0;
    let mut tokens = Vec::new();
    while cursor < bytes.len() {
        match bytes[cursor] {
            byte if byte.is_ascii_whitespace() => cursor += 1,
            b'(' => {
                tokens.push(DiffPredicateToken::Left);
                cursor += 1;
            }
            b')' => {
                tokens.push(DiffPredicateToken::Right);
                cursor += 1;
            }
            b',' => {
                tokens.push(DiffPredicateToken::Comma);
                cursor += 1;
            }
            b'"' => {
                let start = cursor;
                cursor += 1;
                let mut escaped = false;
                while cursor < bytes.len() {
                    let byte = bytes[cursor];
                    cursor += 1;
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == b'"' {
                        break;
                    }
                }
                if cursor > bytes.len() || bytes.get(cursor.saturating_sub(1)) != Some(&b'"') {
                    return Err("unterminated string in diff predicate".into());
                }
                let value: String = serde_json::from_str(&source[start..cursor])
                    .map_err(|_| "invalid string in diff predicate")?;
                tokens.push(DiffPredicateToken::String(value));
            }
            byte if byte.is_ascii_alphabetic() || byte == b'_' => {
                let start = cursor;
                cursor += 1;
                while cursor < bytes.len()
                    && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
                {
                    cursor += 1;
                }
                tokens.push(DiffPredicateToken::Identifier(
                    source[start..cursor].to_owned(),
                ));
            }
            _ => return Err("invalid token in diff predicate".into()),
        }
    }
    Ok(tokens)
}

fn extract_string_calls(source: &str, name: &str) -> Vec<String> {
    let prefix = format!("{name}(\"");
    let mut rest = source;
    let mut values = Vec::new();
    while let Some(start) = rest.find(&prefix) {
        rest = &rest[start + prefix.len()..];
        let Some(end) = rest.find("\")") else {
            break;
        };
        values.push(rest[..end].to_owned());
        rest = &rest[end + 2..];
    }
    values
}
