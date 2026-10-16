//! The existing diff surface frontend, lowered once into shared expressions.
use super::diff_program::{
    DiffProgram, DiffStage, Expression, Leaf, Order, OwnedExpression, SetOperator,
    dispose_statements, stream,
};
use super::{EvaluationError, OperationTraversal};
use std::ops::Range;
use zirql::{ExpressionKind, Located, Stage};

pub(super) fn parse(source: &str) -> Result<DiffProgram, EvaluationError> {
    DiffProgram::check_read_only(source)?;
    let limits = super::comprehension::default_limits();
    let mut budget = zirql::Budget::for_program(limits);
    let mut statements = Vec::new();
    if let Err(error) = parse_statements(source, &mut statements, &mut budget) {
        dispose_statements(statements);
        return Err(error);
    }
    Ok(DiffProgram {
        program: zirql::Program::new(statements, 0..source.len()),
        parse_work: limits.work - budget.remaining_work(),
        parse_results: limits.results - budget.remaining_results().unwrap_or(limits.results),
    })
}

fn parse_statements(
    source: &str,
    statements: &mut Vec<Located<zirql::Statement<DiffStage, SetOperator, ()>>>,
    budget: &mut zirql::Budget,
) -> Result<(), EvaluationError> {
    let mut bindings = vec!["before_document", "after_document", "input"];
    for statement in split_diff_statements(source).map_err(EvaluationError::new)? {
        let statement = statement.trim();
        if statement.is_empty() {
            continue;
        }
        let range = location(source, statement);
        let value = if let Some(expression) = statement.strip_prefix("do ") {
            zirql::Statement::Evaluate {
                expression: parse_expression(source, expression, &bindings, budget)?,
                implicit_output: None,
            }
        } else if let Some((name, expression)) = split_diff_binding(statement) {
            zirql::Statement::Bind {
                name: Located::new(name.to_owned(), location(source, name)),
                expression: parse_expression(source, expression, &bindings, budget)?,
            }
        } else {
            zirql::Statement::Evaluate {
                expression: parse_expression(source, statement, &bindings, budget)?,
                implicit_output: Some(()),
            }
        };
        if let Some((name, _)) = split_diff_binding(statement) {
            bindings.push(name);
        }
        statements.push(Located::new(value, range));
    }
    Ok(())
}

fn location(source: &str, part: &str) -> Range<usize> {
    let start = part.as_ptr() as usize - source.as_ptr() as usize;
    start..start + part.len()
}
fn parse_expression<'a>(
    source: &'a str,
    part: &str,
    bindings: &[&'a str],
    budget: &mut zirql::Budget,
) -> Result<Expression, EvaluationError> {
    let mut reader = zirql::ArgumentReader::new(
        &source[..location(source, part).end],
        location(source, part).start,
    )
    .map_err(super::core_error)?;
    budget
        .charge(
            bindings.iter().map(|name| name.len() + 1).sum(),
            zirql::Location::Query(location(source, part)),
        )
        .map_err(super::core_error)?;
    budget
        .charge_results(
            bindings.len(),
            zirql::Location::Query(location(source, part)),
        )
        .map_err(super::core_error)?;
    let mut host = DiffReader::new(
        source,
        bindings.to_vec(),
        super::parser::DEFAULT_NESTING_LIMIT,
    );
    let nesting = host.nesting;
    let expression = OwnedExpression::new(
        zirql::read_native_expression(&mut reader, &mut host, budget, nesting)
            .map_err(super::core_error)?,
    );
    reader.skip_whitespace();
    if !reader.remaining().is_empty() {
        return Err(super::core_error(
            reader.error("unexpected token in diff expression"),
        ));
    }
    Ok(expression.into_inner())
}

fn parse_stage(stage: &str, range: &Range<usize>) -> Result<DiffStage, EvaluationError> {
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
        _ if stage.starts_with("paths(") => {
            return Err(error("native paths are unavailable on diff queries"));
        }
        _ if stage.starts_with("print(") => {
            DiffStage::Print(parse_diff_string(one("print")?).map_err(EvaluationError::new)?)
        }
        _ => {
            return Err(error(&format!(
                "unknown or unsupported diff query stage `{stage}`"
            )));
        }
    })
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

struct DiffReader<'a> {
    source: &'a str,
    bindings: Vec<&'a str>,
    nesting: usize,
}
impl<'a> DiffReader<'a> {
    fn new(source: &'a str, bindings: Vec<&'a str>, nesting: usize) -> Self {
        Self {
            source,
            bindings,
            nesting,
        }
    }
    fn leaf(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        argument: Option<&zirql::Pattern>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Leaf, zirql::QueryError> {
        let name = reader.word()?;
        let name = name.value.to_owned();
        if name == "true" || name == "false" {
            return Ok(Leaf::Bool(name == "true"));
        }
        reader.expect('(')?;
        let mut arguments = Vec::new();
        let count = if name == "attr" { 2 } else { 1 };
        for index in 0..count {
            if index != 0 {
                reader.expect(',')?;
            }
            let value =
                zirql::read_value_expression_in_scope(reader, self, locals, budget, nesting)?;
            let pattern = argument
                .map(|pattern| zirql::copy_pattern(pattern, budget))
                .transpose()?;
            let range = value.range.clone();
            let subject = Located::new(
                zirql::ValueExpressionKind::NativeQuery(Box::new(Located::new(
                    ExpressionKind::Input,
                    range.clone(),
                ))),
                range,
            );
            let bound = zirql::PredicateArgument {
                argument: pattern,
                body: Box::new(value),
            }
            .bind_value(subject);
            budget.charge_results(1, zirql::Location::Query(bound.range.clone()))?;
            arguments.push(super::diff_program::SharedValue(std::sync::Arc::new(bound)));
        }
        reader.expect(')')?;
        Ok(Leaf::Dynamic { name, arguments })
    }
}
impl zirql::ValueReaderHost for DiffReader<'_> {
    type Stage = DiffStage;
    type SetOperator = SetOperator;
    type Emitter = ();
    type Predicate = Leaf;
    fn field_pipeline_source(&self) -> zirql::FieldPipelineSource {
        zirql::FieldPipelineSource::Constructed
    }
    fn is_binding(&self, name: &str) -> bool {
        name == "input" || self.bindings.contains(&name)
    }
    fn is_stage(&self, name: &str) -> bool {
        matches!(
            name,
            "filter"
                | "unique"
                | "where_exists"
                | "fixpoint"
                | "sort_by"
                | "min_by"
                | "max_by"
                | "min_all_by"
                | "max_all_by"
                | "map_by"
                | "project"
                | "before"
                | "after"
                | "users"
                | "defs"
                | "parent"
                | "children"
                | "subtree"
                | "root"
                | "reachable"
                | "closure"
                | "slice"
                | "backward_slice"
                | "forward_slice"
                | "reverse"
                | "head"
                | "tail"
                | "count"
                | "tally"
                | "names"
                | "attr"
                | "result_types"
                | "operand_types"
                | "sort"
                | "min"
                | "max"
                | "min_all"
                | "max_all"
                | "check"
                | "emit"
                | "json"
                | "markdown"
                | "print"
        )
    }
    fn parse_stage(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<zirql::ExpressionStep<DiffStage, ()>, zirql::QueryError> {
        self.parse_stage_in_scope(reader, &[], budget, self.nesting)
    }
    fn parse_stage_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<zirql::ExpressionStep<DiffStage, ()>, zirql::QueryError> {
        let start = reader.position();
        if reader.consume('(') {
            let value =
                zirql::read_value_expression_in_scope(reader, self, locals, budget, nesting)?;
            reader.expect(')')?;
            let stage = match value.value {
                zirql::ValueExpressionKind::NativeQuery(expression)
                | zirql::ValueExpressionKind::NativeValue(expression) => {
                    DiffStage::Nested(super::diff_program::nested(*expression))
                }
                kind => DiffStage::Value(super::diff_program::SharedValue(std::sync::Arc::new(
                    Located::new(kind, value.range),
                ))),
            };
            return Ok(zirql::ExpressionStep::Stage(stage));
        }
        let name = reader.word()?;
        if name.value == "root" {
            reader.expect('(')?;
            let predicate =
                zirql::read_native_predicate_argument(reader, self, locals, budget, nesting)?;
            reader.expect(')')?;
            let DiffStage::Stream(pipeline) =
                stream(Stage::Filter(predicate), start..reader.position()).map_err(|error| {
                    zirql::QueryError::query(
                        error.range.unwrap_or(start..reader.position()),
                        error.message,
                    )
                })?
            else {
                unreachable!("stream filter")
            };
            return Ok(zirql::ExpressionStep::Stage(DiffStage::Root(pipeline)));
        }
        reader.skip_whitespace();
        if reader.consume('(') {
            let mut depth = 1usize;
            let lexed = super::lexer::lex(reader.remaining());
            for token in lexed.tokens() {
                match token.kind() {
                    super::lexer::TokenKind::LParen => depth += 1,
                    super::lexer::TokenKind::RParen => {
                        depth -= 1;
                        if depth == 0 {
                            reader.advance(token.range().end() as usize)?;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth != 0 {
                return Err(reader.error("expected `)` after diff stage"));
            }
        }
        budget.charge(
            reader.position() - start,
            zirql::Location::Query(start..reader.position()),
        )?;
        let part = self.source[start..reader.position()].trim_end();
        let stage = parse_stage(part, &(start..reader.position())).map_err(|error| {
            zirql::QueryError::query(
                error.range.unwrap_or(start..reader.position()),
                error.message,
            )
        })?;
        Ok(zirql::ExpressionStep::Stage(stage))
    }
    fn parse_predicate_leaf(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<Leaf, zirql::QueryError> {
        self.leaf(reader, None, &[], budget, self.nesting)
    }
    fn parse_predicate_leaf_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Leaf, zirql::QueryError> {
        self.leaf(reader, None, locals, budget, nesting)
    }
    fn parse_predicate_leaf_in_argument(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        argument: Option<&zirql::Pattern>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Leaf, zirql::QueryError> {
        self.leaf(reader, argument, locals, budget, nesting)
    }
    fn is_predicate_leaf(&self, name: &str) -> bool {
        matches!(
            name,
            "true"
                | "false"
                | "change"
                | "changed"
                | "op"
                | "dialect"
                | "result_type"
                | "has_attr"
                | "attr"
        )
    }
    fn predicate_expression(
        &mut self,
        predicate: zirql::PredicateArgument<Expression>,
        range: Range<usize>,
    ) -> Result<Leaf, zirql::QueryError> {
        let subject = Located::new(
            zirql::ValueExpressionKind::NativeQuery(Box::new(Located::new(
                ExpressionKind::Input,
                range.clone(),
            ))),
            range,
        );
        Ok(Leaf::Expression(super::diff_program::SharedValue(
            std::sync::Arc::new(predicate.bind(subject)),
        )))
    }
    fn filter(&mut self, predicate: zirql::Predicate<Leaf>, range: Range<usize>) -> DiffStage {
        stream(Stage::Filter(predicate), range).expect("shared reader bounded the filter")
    }
    fn projection(
        &mut self,
        projection: zirql::Projection<Expression>,
        _: Range<usize>,
    ) -> DiffStage {
        for (_, value) in projection.fields {
            zirql::dispose_native_value(value);
        }
        DiffStage::UnsupportedProjection
    }
    fn value_source(
        &mut self,
        value: zirql::ValueExpression<Expression>,
        _: Range<usize>,
    ) -> Result<DiffStage, zirql::QueryError> {
        Ok(DiffStage::Value(super::diff_program::SharedValue(
            std::sync::Arc::new(value),
        )))
    }
    fn ordering(
        &mut self,
        ordering: zirql::NativeOrdering<Expression>,
        _range: Range<usize>,
    ) -> Result<DiffStage, zirql::QueryError> {
        use zirql::NativeOrder as O;
        Ok(match ordering.selector {
            Some(selector) => DiffStage::Order {
                kind: match ordering.order {
                    O::Sort => Order::Sort,
                    O::Min => Order::Min,
                    O::Max => Order::Max,
                    O::MinAll => Order::MinAll,
                    O::MaxAll => Order::MaxAll,
                },
                selector: super::diff_program::nested(selector),
            },
            None => match ordering.order {
                O::Sort => DiffStage::Sort,
                O::Min | O::Max | O::MinAll | O::MaxAll => DiffStage::Extreme {
                    maximum: matches!(ordering.order, O::Max | O::MaxAll),
                    all: matches!(ordering.order, O::MinAll | O::MaxAll),
                },
            },
        })
    }
    fn map_sorting(&mut self, _: Range<usize>) -> Result<DiffStage, zirql::QueryError> {
        Ok(DiffStage::SortMapValue)
    }
    fn mapping(
        &mut self,
        mapping: zirql::NativeMap<Expression>,
        _: Range<usize>,
    ) -> Result<DiffStage, zirql::QueryError> {
        Ok(DiffStage::MapBy {
            key: super::diff_program::nested(mapping.key),
            value: super::diff_program::nested(mapping.value),
        })
    }
    fn fixed_point(
        &mut self,
        query: Expression,
        _: Range<usize>,
    ) -> Result<DiffStage, zirql::QueryError> {
        Ok(DiffStage::Fixpoint(super::diff_program::nested(query)))
    }
    fn where_exists(
        &mut self,
        query: Expression,
        _: Range<usize>,
    ) -> Result<DiffStage, zirql::QueryError> {
        Ok(DiffStage::WhereExists(super::diff_program::nested(query)))
    }
    fn set_operator(&self, word: &str) -> Option<SetOperator> {
        match word {
            "union" => Some(SetOperator::Union),
            "intersect" => Some(SetOperator::Intersect),
            "except" => Some(SetOperator::Except),
            _ => None,
        }
    }
}
