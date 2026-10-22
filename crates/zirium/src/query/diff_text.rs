//! The diff surface frontend, lowered once into shared expressions.
use super::diff_program::{DiffProgram, DiffStage, Expression, Leaf, SetOperator};
use super::{EvaluationError, OperationTraversal};
use std::{collections::HashSet, ops::Range, sync::Arc};
use zirql::{ExpressionKind, Located, Stage};

pub(super) fn parse(source: &str) -> Result<DiffProgram, EvaluationError> {
    DiffProgram::check_read_only(source)?;
    let limits = super::comprehension::default_limits();
    let mut budget = zirql::Budget::for_program(limits);
    let nesting = super::parser::DEFAULT_NESTING_LIMIT;
    let parsed = zirql::read_program(
        source,
        &["before_document", "after_document", "input"],
        &mut DiffReader {
            nesting,
            function_scope: false,
            constructed_functions: HashSet::new(),
            suppressed_output: false,
        },
        &mut budget,
        nesting,
    )
    .map_err(super::core_error)?;
    Ok(DiffProgram {
        program: zirql::Program::new(parsed.statements, parsed.range),
        functions: parsed.functions,
        parameters: parsed
            .parameters
            .into_iter()
            .map(|parameter| parameter.declaration)
            .collect(),
        parse_work: limits.work - budget.remaining_work(),
        parse_results: limits.results - budget.remaining_results().unwrap_or(limits.results),
    })
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

struct DiffReader {
    nesting: usize,
    function_scope: bool,
    suppressed_output: bool,
    constructed_functions: HashSet<String>,
}
impl zirql::ProgramReaderHost for DiffReader {
    fn function_scope(&mut self, active: bool) {
        self.function_scope = active;
    }
    fn function_result(
        &mut self,
        name: Located<&str>,
        result: &Located<zirql::SignatureType>,
        budget: &mut zirql::Budget,
    ) -> Result<(), zirql::QueryError> {
        if !matches!(&result.value, zirql::SignatureType::Named(name) if matches!(name.value.as_str(), "Change" | "BeforeOperation" | "AfterOperation"))
        {
            budget.charge(
                name.value.len() + 1,
                zirql::Location::Query(name.range.clone()),
            )?;
            budget.charge_results(1, zirql::Location::Query(name.range))?;
            self.constructed_functions.insert(name.value.to_owned());
        }
        Ok(())
    }
    fn admit_name(&mut self, name: Located<&str>, _: bool) -> Result<(), zirql::QueryError> {
        if super::diff_program::is_reserved_name(name.value) {
            return Err(zirql::QueryError::query(
                name.range,
                "binding name is reserved in diff mode",
            ));
        }
        Ok(())
    }
    fn read_binding(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<Option<zirql::Pattern>, zirql::QueryError> {
        let mut probe = zirql::ArgumentReader::new(reader.source(), reader.position())?;
        let Ok(name) = probe.word() else {
            return Ok(None);
        };
        if !probe.consume('=') || probe.remaining().starts_with('=') {
            return Ok(None);
        }
        budget.charge(
            name.value.len() + 1,
            zirql::Location::Query(name.range.clone()),
        )?;
        budget.charge_results(2, zirql::Location::Query(name.range.clone()))?;
        let range = name.range.clone();
        let name = Located::new(name.value.to_owned(), name.range);
        reader.advance(probe.position() - reader.position())?;
        Ok(Some(Located::new(zirql::PatternKind::Name(name), range)))
    }
    fn allows_unterminated_host_binding(&self) -> bool {
        true
    }
    fn read_host_statement(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        _: &[&str],
        budget: &mut zirql::Budget,
        _: usize,
    ) -> Result<Option<Vec<Located<zirql::ReadStatement<Self>>>>, zirql::QueryError> {
        let start = reader.position();
        if !reader.consume(';') {
            return Ok(None);
        }
        budget.charge(1, zirql::Location::Query(start..reader.position()))?;
        Ok(Some(Vec::new()))
    }
    fn read_output_prefix(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
    ) -> Option<zirql::ProgramExpression> {
        self.suppressed_output = reader.keyword("do");
        self.suppressed_output
            .then_some(zirql::ProgramExpression::Evaluate)
    }
    fn read_program_expression(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        names: &[&str],
        _: zirql::ProgramExpression,
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<Expression, zirql::QueryError> {
        let value = zirql::read_value_expression_in_scope(reader, self, names, budget, nesting)?;
        let range = value.range.clone();
        match value.value {
            zirql::ValueExpressionKind::NativeQuery(expression)
            | zirql::ValueExpressionKind::NativeValue(expression) => Ok(*expression),
            kind => Ok(Located::new(
                ExpressionKind::Sequence {
                    input: Box::new(Located::new(
                        ExpressionKind::Input,
                        range.start..range.start,
                    )),
                    steps: vec![Located::new(
                        zirql::ExpressionStep::Value(Arc::new(Located::new(kind, range.clone()))),
                        range.clone(),
                    )],
                },
                range,
            )),
        }
    }
    fn implicit_output(&mut self, purpose: zirql::ProgramExpression) -> Option<()> {
        (purpose != zirql::ProgramExpression::Suppressed
            && !std::mem::take(&mut self.suppressed_output))
        .then_some(())
    }
}
impl DiffReader {
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
                zirql::ValueExpressionKind::NativeValue(Box::new(Located::new(
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
impl zirql::ValueReaderHost for DiffReader {
    fn retains_native_values(&self) -> bool {
        true
    }
    type Stage = DiffStage;
    type SetOperator = SetOperator;
    type Emitter = ();
    fn field_pipeline_source(&self) -> zirql::FieldPipelineSource {
        zirql::FieldPipelineSource::ValueArgument
    }
    fn is_binding(&self, name: &str) -> bool {
        !self.function_scope && name == "input"
    }
    fn is_constructed_function(&self, name: &str) -> bool {
        self.constructed_functions.contains(name)
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
    ) -> Result<zirql::ExpressionStep<DiffStage, (), zirql::NativeSetOperator>, zirql::QueryError>
    {
        self.parse_stage_in_scope(reader, &[], budget, self.nesting)
    }
    fn parse_stage_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<zirql::ExpressionStep<DiffStage, (), zirql::NativeSetOperator>, zirql::QueryError>
    {
        let start = reader.position();
        let mut shared_argument = false;
        let result = (|| {
            if reader.consume('(') {
                shared_argument = true;
                let value =
                    zirql::read_value_expression_in_scope(reader, self, locals, budget, nesting)?;
                reader.expect(')')?;
                let range = value.range.clone();
                let value = match value.value {
                    zirql::ValueExpressionKind::NativeQuery(expression)
                    | zirql::ValueExpressionKind::NativeValue(expression) => {
                        zirql::ValueExpressionKind::NativeValue(Box::new(Located::new(
                            ExpressionKind::Group(expression),
                            range.clone(),
                        )))
                    }
                    kind => kind,
                };
                return Ok(zirql::ExpressionStep::Value(std::sync::Arc::new(
                    Located::new(value, range),
                )));
            }
            let name = reader.word()?;
            shared_argument = matches!(name.value, "head" | "tail" | "check" | "root");
            if name.value == "root" {
                reader.expect('(')?;
                let predicate =
                    zirql::read_native_predicate_argument(reader, self, locals, budget, nesting)?;
                reader.expect(')')?;
                let predicate = super::diff_program::native_predicate(predicate);
                let pipeline = zirql::Pipeline::with_group_not_nesting(
                    vec![Located::new(
                        Stage::Filter(predicate),
                        start..reader.position(),
                    )],
                    start..reader.position(),
                    zirql::MAX_NESTING,
                )?;
                return Ok(zirql::ExpressionStep::Stage(DiffStage::Root(pipeline)));
            }
            let collection = match name.value {
                "head" | "tail" => {
                    reader.expect('(')?;
                    let count = zirql::read_value_expression_in_scope(
                        reader, self, locals, budget, nesting,
                    )?;
                    reader.expect(')')?;
                    return Ok(if name.value == "head" {
                        zirql::ExpressionStep::Head(Arc::new(count))
                    } else {
                        zirql::ExpressionStep::Tail(Arc::new(count))
                    });
                }
                "check" => {
                    let (expected, message) = if reader.consume('(') {
                        let expected = zirql::read_value_expression_in_scope(
                            reader, self, locals, budget, nesting,
                        )?;
                        let message = if reader.consume(',') {
                            Some(Arc::new(zirql::read_value_expression_in_scope(
                                reader, self, locals, budget, nesting,
                            )?))
                        } else {
                            None
                        };
                        reader.expect(')')?;
                        (Some(Arc::new(expected)), message)
                    } else {
                        (None, None)
                    };
                    return Ok(zirql::ExpressionStep::Check {
                        operand: expected,
                        message,
                    });
                }
                "unique" => Some(Stage::Unique),
                "count" => Some(Stage::Count),
                "tally" => Some(Stage::Tally),
                "reverse" => Some(Stage::Reverse),
                _ => None,
            };
            if let Some(collection) = collection {
                refuse_arguments(reader, start)?;
                return Ok(zirql::ExpressionStep::Collection(collection));
            }
            if let Some(order) = match name.value {
                "sort" => Some(zirql::NativeOrder::Sort),
                "min" => Some(zirql::NativeOrder::Min),
                "max" => Some(zirql::NativeOrder::Max),
                "min_all" => Some(zirql::NativeOrder::MinAll),
                "max_all" => Some(zirql::NativeOrder::MaxAll),
                _ => None,
            } {
                refuse_arguments(reader, start)?;
                return Ok(zirql::ExpressionStep::Ordering(Arc::new(
                    zirql::NativeOrdering {
                        order,
                        selector: None,
                    },
                )));
            }
            let stage = match name.value {
                "users" | "defs" => {
                    let index = if reader.consume('(') {
                        let index = reader.unsigned_integer()?;
                        reader.expect(')')?;
                        Some(index.value)
                    } else {
                        None
                    };
                    if name.value == "users" {
                        DiffStage::Users(index)
                    } else {
                        DiffStage::Defs(index)
                    }
                }
                "attr" | "print" => {
                    reader.expect('(')?;
                    let argument = reader.quoted_string()?;
                    reader.expect(')')?;
                    if name.value == "attr" {
                        DiffStage::Attr(argument.value)
                    } else {
                        DiffStage::Print(argument.value)
                    }
                }
                "paths" => {
                    return Err(zirql::QueryError::query(
                        start..reader.position(),
                        "native paths are unavailable on diff queries",
                    ));
                }
                name => {
                    refuse_arguments(reader, start)?;
                    match name {
                        "input" => DiffStage::Input,
                        "before" => DiffStage::Project(crate::diff::DiffSide::Before),
                        "after" => DiffStage::Project(crate::diff::DiffSide::After),
                        "parent" => DiffStage::Parent,
                        "children" => DiffStage::Children,
                        "subtree" => DiffStage::Subtree,
                        "reachable" => DiffStage::Graph(OperationTraversal::Reachable),
                        "closure" => DiffStage::Graph(OperationTraversal::Closure),
                        "slice" | "backward_slice" => DiffStage::Graph(OperationTraversal::Slice),
                        "forward_slice" => DiffStage::Graph(OperationTraversal::ForwardSlice),
                        "names" => DiffStage::Names,
                        "result_types" => DiffStage::ResultTypes,
                        "operand_types" => DiffStage::OperandTypes,
                        "emit" => DiffStage::Noop,
                        "json" => DiffStage::Json,
                        "markdown" => DiffStage::Markdown,
                        _ => {
                            return Err(zirql::QueryError::query(
                                start..reader.position(),
                                format!("unknown or unsupported diff query stage `{name}`"),
                            ));
                        }
                    }
                }
            };
            budget.charge(
                reader.position() - start,
                zirql::Location::Query(start..reader.position()),
            )?;
            Ok(zirql::ExpressionStep::Stage(stage))
        })();
        result.map_err(|mut error: zirql::QueryError| {
            if error.category == zirql::QueryErrorCategory::Other
                && (!shared_argument
                    || matches!(error.message.as_str(), "expected ')'" | "expected '('"))
            {
                error.location = zirql::Location::Query(stage_range(reader.source(), start));
            }
            error
        })
    }
    fn parse_predicate_leaf(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        budget: &mut zirql::Budget,
    ) -> Result<DiffStage, zirql::QueryError> {
        self.leaf(reader, None, &[], budget, self.nesting)
            .map(DiffStage::Guard)
    }
    fn parse_predicate_leaf_in_scope(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<DiffStage, zirql::QueryError> {
        self.leaf(reader, None, locals, budget, nesting)
            .map(DiffStage::Guard)
    }
    fn parse_predicate_leaf_in_argument(
        &mut self,
        reader: &mut zirql::ArgumentReader<'_>,
        argument: Option<&zirql::Pattern>,
        locals: &[&str],
        budget: &mut zirql::Budget,
        nesting: usize,
    ) -> Result<DiffStage, zirql::QueryError> {
        self.leaf(reader, argument, locals, budget, nesting)
            .map(DiffStage::Guard)
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
    ) -> Result<DiffStage, zirql::QueryError> {
        let subject = Located::new(
            zirql::ValueExpressionKind::NativeValue(Box::new(Located::new(
                ExpressionKind::Input,
                range.clone(),
            ))),
            range,
        );
        Ok(DiffStage::Guard(Leaf::Expression(
            super::diff_program::SharedValue(std::sync::Arc::new(predicate.bind(subject))),
        )))
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

fn refuse_arguments(
    reader: &mut zirql::ArgumentReader<'_>,
    start: usize,
) -> Result<(), zirql::QueryError> {
    if !reader.consume('(') {
        return Ok(());
    }
    let range = stage_range(reader.source(), start);
    reader.advance(range.end - reader.position())?;
    Err(zirql::QueryError::query(
        range,
        "diff stage does not accept arguments",
    ))
}

fn stage_range(source: &str, start: usize) -> Range<usize> {
    let lexed = super::lexer::lex(&source[start..]);
    let mut depth = 0usize;
    for token in lexed.tokens() {
        match token.kind() {
            super::lexer::TokenKind::LParen => depth += 1,
            super::lexer::TokenKind::RParen if depth == 1 => {
                return start..start + token.range().end() as usize;
            }
            super::lexer::TokenKind::RParen => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    start..source.len()
}
