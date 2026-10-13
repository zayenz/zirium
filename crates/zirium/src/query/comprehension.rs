//! Contextual report syntax and native projections. Zirql owns value evaluation.
use super::{EvaluationError, NativeValue, QueryError, QueryOutput, core_error, lexer, program};
use lexer::{Token, TokenKind as T};
use std::{
    collections::{BTreeMap, HashSet},
    ops::Range,
    sync::Arc,
};
use zirql::{
    ArrayType, Bindings, Budget, ConstructedValue as V, Located, Location,
    ValueBinaryOperator as B, ValueExpression as E, ValueExpressionKind as K, ValueGenerator,
    ValueType, ValueUnaryOperator as U,
};

pub(super) fn default_limits() -> zirql::ProgramLimits {
    let limits = super::EvaluationLimits::default();
    zirql::ProgramLimits {
        work: limits.max_work,
        results: 1_000_000,
        nesting: super::parser::DEFAULT_NESTING_LIMIT,
        stream_items: limits.max_items,
    }
}

/// A lowered report expression. Cloning shares its immutable tree.
#[derive(Debug, PartialEq)]
pub struct ParsedComprehension {
    expression: Arc<E>,
    globals: Arc<[String]>,
}
impl Clone for ParsedComprehension {
    fn clone(&self) -> Self {
        Self {
            expression: Arc::clone(&self.expression),
            globals: Arc::clone(&self.globals),
        }
    }
}
// Only this frontend constructs the tree, and it refuses nonfinite literals.
impl Eq for ParsedComprehension {}

/// A native constructed report with its query location. Cloning shares the
/// immutable value; evaluation snapshots use Zirql's budgeted copy operation.
#[derive(Debug, PartialEq)]
pub struct ConstructedReport {
    value: Arc<V>,
    range: Range<usize>,
}
impl Clone for ConstructedReport {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            range: self.range.clone(),
        }
    }
}
// Private construction checks every float before admitting this public Eq value.
impl Eq for ConstructedReport {}
impl ConstructedReport {
    fn new(value: V, range: Range<usize>, budget: &mut Budget) -> Result<Self, EvaluationError> {
        zirql::constructed_value_type(&value, budget, Location::Query(range.clone()))
            .map_err(core_error)?;
        budget
            .charge(1, Location::Query(range.clone()))
            .map_err(core_error)?;
        Ok(Self {
            value: Arc::new(value),
            range,
        })
    }
    pub fn value(&self) -> &V {
        &self.value
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub(super) fn copy(
        &self,
        budget: &mut Budget,
        range: &Range<usize>,
    ) -> Result<Self, EvaluationError> {
        let value =
            zirql::copy_constructed_value(self.value(), budget, Location::Query(range.clone()))
                .map_err(core_error)?;
        Self::new(value, self.range.clone(), budget)
    }
    pub(super) fn size(&self) -> usize {
        value_size(self.value())
    }
    /// Convert the typed report to ordinary JSON under an aggregate request budget.
    pub fn to_json_with_budget(
        &self,
        budget: &mut Budget,
    ) -> Result<serde_json::Value, EvaluationError> {
        if budget.remaining_results().is_none() {
            return Err(EvaluationError::new(
                "constructed reports require an aggregate result allowance",
            )
            .at(self.range.clone()));
        }
        zirql::constructed_value_type(self.value(), budget, Location::Query(self.range.clone()))
            .map_err(core_error)?;
        to_json(self.value(), budget, &self.range, 0)
    }
    /// Render ordinary pretty JSON, spending bytes before buffering them.
    pub fn render_json_with_budget(&self, budget: &mut Budget) -> Result<String, EvaluationError> {
        let value = self.to_json_with_budget(budget)?;
        program::serialize_json(&value, budget, &self.range, true)
    }
    /// Render one compact document/result envelope with this same allowance.
    pub fn render_ndjson_with_budget(
        &self,
        document_name: &str,
        budget: &mut Budget,
    ) -> Result<String, EvaluationError> {
        let value = self.to_json_with_budget(budget)?;
        admit(budget, 4, document_name.len() + 14, &self.range)?;
        let mut envelope = serde_json::Map::new();
        envelope.insert(
            "document".to_owned(),
            serde_json::Value::String(document_name.to_owned()),
        );
        envelope.insert("result".to_owned(), value);
        program::serialize_json(
            &serde_json::Value::Object(envelope),
            budget,
            &self.range,
            false,
        )
    }
}
fn value_size(value: &V) -> usize {
    match value {
        V::Array(values)
        | V::Tuple(values)
        | V::Set {
            elements: values, ..
        } => values.iter().fold(1usize, |n, v| {
            n.saturating_add(1).saturating_add(value_size(v))
        }),
        V::Record(fields) => fields.iter().fold(1usize, |n, (_, v)| {
            n.saturating_add(1).saturating_add(value_size(v))
        }),
        V::KeyedArray { axes, entries } => {
            let axes_size = axes.iter().fold(1usize, |size, axis| {
                size.saturating_add(1).saturating_add(axis.keys.len())
            });
            entries.iter().fold(axes_size, |size, entry| {
                size.saturating_add(1)
                    .saturating_add(entry.key.len())
                    .saturating_add(value_size(&entry.value))
            })
        }
        V::Boolean(_) | V::Integer(_) | V::Float(_) | V::String(_) | V::Member(_) => 1,
    }
}

// Recognize `for binder in` at this array's own level. `[for]` stays a literal.
// The allocation-free admission scan ignores quoted strings and comments, so
// legacy names and text containing `for` retain their old parsing allowance.
pub(super) fn has_syntax(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut cursor = 0;
    let mut arrays = 0usize;
    let mut previous_for = false;
    let mut before_previous_for = false;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'"' => {
                cursor += 1;
                while cursor < bytes.len() {
                    match bytes[cursor] {
                        b'\\' => cursor = cursor.saturating_add(2),
                        b'"' => {
                            cursor += 1;
                            break;
                        }
                        _ => cursor += 1,
                    }
                }
                previous_for = false;
                before_previous_for = false;
            }
            b'#' => {
                while cursor < bytes.len() && bytes[cursor] != b'\n' {
                    cursor += 1;
                }
            }
            byte if byte.is_ascii_whitespace() => cursor += 1,
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = cursor;
                cursor += 1;
                while bytes
                    .get(cursor)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
                {
                    cursor += 1;
                }
                let word = &source[start..cursor];
                if arrays > 0 && before_previous_for && word == "in" {
                    return true;
                }
                before_previous_for = previous_for;
                previous_for = word == "for";
            }
            byte => {
                if byte == b'[' {
                    arrays += 1;
                } else if byte == b']' {
                    arrays = arrays.saturating_sub(1);
                }
                previous_for = false;
                before_previous_for = false;
                cursor += 1;
            }
        }
    }
    false
}

pub(super) fn starts(source: &str, tokens: &[Token], cursor: usize) -> bool {
    let mut depth = 0usize;
    let mut previous = T::LBracket;
    for (i, token) in tokens.iter().enumerate().skip(cursor) {
        match token.kind() {
            T::LBracket | T::LBrace | T::LParen => depth += 1,
            T::RBracket | T::RBrace | T::RParen => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return false;
                }
            }
            T::Eof => return false,
            T::Trivia => continue,
            T::Identifier
                if depth == 1
                    && &source[token.range().as_range()] == "for"
                    && !matches!(previous, T::LBracket | T::Comma) =>
            {
                let mut next = tokens[i + 1..].iter().filter(|t| t.kind() != T::Trivia);
                if next.next().is_some_and(|t| t.kind() == T::Identifier)
                    && next.next().is_some_and(|t| {
                        t.kind() == T::Identifier && &source[t.range().as_range()] == "in"
                    })
                {
                    return true;
                }
            }
            _ => {}
        }
        previous = token.kind();
    }
    false
}

pub(super) fn lower(
    source: &str,
    tokens: &[Token],
    cursor: usize,
    globals: &HashSet<String>,
    budget: &mut Budget,
    nesting: usize,
) -> Result<(ParsedComprehension, usize), QueryError> {
    let mut lower = Lower {
        source,
        tokens,
        cursor,
        budget,
        nesting: nesting.min(64),
    };
    let expression = lower.expression(0, 0)?;
    lower.charge(globals.iter().map(|n| n.len() + 1).sum())?;
    lower.slots::<String>(globals.len())?;
    let mut globals: Vec<_> = globals.iter().cloned().collect();
    globals.sort();
    Ok((
        ParsedComprehension {
            expression: Arc::new(expression),
            globals: globals.into(),
        },
        lower.cursor,
    ))
}
struct Lower<'a, 'b> {
    source: &'a str,
    tokens: &'a [Token],
    cursor: usize,
    budget: &'b mut Budget,
    nesting: usize,
}
impl Lower<'_, '_> {
    fn token(&mut self) -> Token {
        while self.tokens[self.cursor].kind() == T::Trivia {
            self.cursor += 1;
        }
        self.tokens[self.cursor]
    }
    fn text(&mut self) -> &str {
        let token = self.token();
        &self.source[token.range().as_range()]
    }
    fn fail(&mut self, message: &'static str) -> QueryError {
        QueryError {
            position: self.token().range().start() as usize,
            message,
        }
    }
    fn charge(&mut self, units: usize) -> Result<(), QueryError> {
        let location = Location::Query(self.token().range().as_range());
        self.budget
            .charge(units, location)
            .map_err(|_| self.fail("query work limit exceeded during parsing"))
    }
    fn slots<Ty>(&mut self, count: usize) -> Result<(), QueryError> {
        if count
            .checked_mul(std::mem::size_of::<Ty>())
            .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(self.fail("value allocation size overflow"));
        }
        self.charge(count)?;
        let location = Location::Query(self.token().range().as_range());
        self.budget
            .charge_results(count, location)
            .map_err(|_| self.fail("result limit exceeded during parsing"))
    }
    fn expect(&mut self, kind: T, message: &'static str) -> Result<Token, QueryError> {
        let token = self.token();
        if token.kind() != kind {
            return Err(self.fail(message));
        }
        self.cursor += 1;
        Ok(token)
    }
    fn word(&mut self, word: &str) -> bool {
        self.token().kind() == T::Identifier && self.text() == word
    }
    fn name(&mut self) -> Result<Located<String>, QueryError> {
        let token = self.token();
        if token.kind() != T::Identifier {
            return Err(self.fail("expected a comprehension name"));
        }
        self.charge(token.range().len() as usize)?;
        self.cursor += 1;
        Ok(Located::new(
            self.source[token.range().as_range()].to_owned(),
            token.range().as_range(),
        ))
    }
    fn string(&mut self) -> Result<Located<String>, QueryError> {
        let token = self.token();
        self.charge((token.range().len() as usize).saturating_mul(6))?;
        self.cursor += 1;
        let text = &self.source[token.range().as_range()];
        let normalized = text
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        let value = serde_json::from_str(&normalized).map_err(|_| QueryError {
            position: token.range().start() as usize,
            message: "invalid JSON string escape or Unicode sequence",
        })?;
        Ok(Located::new(value, token.range().as_range()))
    }
    fn node(&mut self, kind: K, start: usize, end: usize) -> Result<E, QueryError> {
        self.slots::<E>(1)?;
        Ok(Located::new(kind, start..end))
    }
    fn expression(&mut self, minimum: u8, depth: usize) -> Result<E, QueryError> {
        if depth >= self.nesting {
            return Err(self.fail("value nesting limit exceeded"));
        }
        let mut left = self.atom(depth)?;
        let mut chain = 0;
        loop {
            let token = self.token();
            if token.kind() == T::LBracket && minimum <= 10 {
                self.cursor += 1;
                let index = self.expression(0, depth + 1)?;
                let end = self
                    .expect(T::RBracket, "expected `]` after array position")?
                    .range()
                    .end() as usize;
                let start = left.range.start;
                left = self.node(
                    K::ArrayAccess {
                        array: Box::new(left),
                        index: Box::new(index),
                    },
                    start,
                    end,
                )?;
            } else if self.text() == "." && minimum <= 10 {
                self.cursor += 1;
                let field = self.name()?;
                let (start, end) = (left.range.start, field.range.end);
                left = self.node(
                    K::Field {
                        value: Box::new(left),
                        field,
                    },
                    start,
                    end,
                )?;
            } else {
                let (precedence, operator) = match self.text() {
                    "or" => (1, Some(B::Or)),
                    "and" => (2, Some(B::And)),
                    "==" | "=" => (3, Some(B::Equal)),
                    "!=" => (3, Some(B::NotEqual)),
                    "<" => (4, Some(B::Less)),
                    "<=" => (4, Some(B::LessEqual)),
                    ">" => (4, Some(B::Greater)),
                    ">=" => (4, Some(B::GreaterEqual)),
                    "++" => (5, Some(B::Concatenate)),
                    ".." => (6, None),
                    "+" => (7, Some(B::Add)),
                    "-" => (7, Some(B::Subtract)),
                    "*" => (8, Some(B::Multiply)),
                    "/" => (8, Some(B::Divide)),
                    "div" => (8, Some(B::IntegerDivide)),
                    "mod" => (8, Some(B::Modulo)),
                    _ => break,
                };
                if precedence < minimum {
                    break;
                }
                self.cursor += 1;
                let right = self.expression(precedence + 1, depth + chain + 1)?;
                let (start, end) = (left.range.start, right.range.end);
                let kind = match operator {
                    Some(operator) => K::Binary {
                        operator,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    None => K::Range {
                        lower: Box::new(left),
                        upper: Box::new(right),
                    },
                };
                left = self.node(kind, start, end)?;
            }
            chain += 1;
            if depth + chain >= self.nesting {
                return Err(self.fail("value nesting limit exceeded"));
            }
        }
        Ok(left)
    }
    fn atom(&mut self, depth: usize) -> Result<E, QueryError> {
        let token = self.token();
        let start = token.range().start() as usize;
        let kind = match token.kind() {
            T::Integer | T::Number => {
                self.charge(token.range().len() as usize)?;
                self.cursor += 1;
                K::Literal(number(&self.source[token.range().as_range()], start)?)
            }
            T::String => K::Literal(V::String(self.string()?.value)),
            T::Identifier if matches!(self.text(), "true" | "false") => {
                let value = self.text() == "true";
                self.cursor += 1;
                K::Literal(V::Boolean(value))
            }
            T::Identifier if self.word("not") => {
                self.cursor += 1;
                K::Unary {
                    operator: U::Not,
                    operand: Box::new(self.expression(9, depth + 1)?),
                }
            }
            T::Operator if matches!(self.text(), "+" | "-") => {
                let negative = self.text() == "-";
                self.cursor += 1;
                let next = self.token();
                if matches!(next.kind(), T::Integer | T::Number)
                    && !self.source[next.range().as_range()].starts_with('-')
                {
                    self.charge(next.range().len() as usize + 1)?;
                    let spelling = format!(
                        "{}{}",
                        if negative { "-" } else { "" },
                        &self.source[next.range().as_range()]
                    );
                    self.cursor += 1;
                    K::Literal(number(&spelling, start)?)
                } else {
                    K::Unary {
                        operator: if negative { U::Negative } else { U::Positive },
                        operand: Box::new(self.expression(9, depth + 1)?),
                    }
                }
            }
            T::Identifier => {
                if self.word("null") {
                    return Err(self.fail("null is unsupported in positional comprehensions"));
                }
                let name = self.name()?.value;
                if self.token().kind() == T::LParen {
                    return Err(self.fail("calls are unsupported in positional comprehensions"));
                }
                K::Name(name)
            }
            T::LParen => {
                self.cursor += 1;
                let mut expression = self.expression(0, depth + 1)?;
                let end = self
                    .expect(T::RParen, "expected `)` after value expression")?
                    .range()
                    .end() as usize;
                expression.range = start..end;
                return Ok(expression);
            }
            T::LBracket => return self.array(depth),
            T::LBrace => {
                self.cursor += 1;
                let mut fields = Vec::new();
                while self.token().kind() != T::RBrace {
                    if self.token().kind() != T::String {
                        return Err(self.fail("expected a static quoted record label"));
                    }
                    let label = self.string()?;
                    self.expect(T::Colon, "expected `:` after record label")?;
                    let value = self.expression(0, depth + 1)?;
                    self.slots::<(Located<String>, E)>(1)?;
                    fields.push((label, value));
                    if self.token().kind() != T::Comma {
                        break;
                    }
                    self.cursor += 1;
                    if self.token().kind() == T::RBrace {
                        return Err(self.fail("trailing comma in record fields"));
                    }
                }
                self.expect(T::RBrace, "expected `}` after record fields")?;
                K::Record(fields)
            }
            _ => return Err(self.fail("unsupported expression in positional comprehensions")),
        };
        let end = self.tokens[self.cursor.saturating_sub(1)].range().end() as usize;
        self.node(kind, start, end)
    }
    fn array(&mut self, depth: usize) -> Result<E, QueryError> {
        let start = self.expect(T::LBracket, "expected `[`")?.range().start() as usize;
        let mut children = Vec::new();
        if self.token().kind() != T::RBracket {
            let head = self.expression(0, depth + 1)?;
            if self.word("for") {
                let mut clauses = Vec::new();
                while self.word("for") {
                    let clause_start = self.token().range().start() as usize;
                    self.cursor += 1;
                    let binder = self.name()?;
                    if !self.word("in") {
                        return Err(self.fail("expected `in` after comprehension binder"));
                    }
                    self.cursor += 1;
                    let source = self.expression(0, depth + 1)?;
                    let filter = if self.word("if") {
                        self.cursor += 1;
                        Some(self.expression(0, depth + 1)?)
                    } else {
                        None
                    };
                    let end = filter.as_ref().map_or(source.range.end, |f| f.range.end);
                    self.slots::<Located<ValueGenerator>>(1)?;
                    self.slots::<Located<Option<String>>>(1)?;
                    clauses.push(Located::new(
                        ValueGenerator::In {
                            binders: vec![Located::new(
                                (binder.value != "_").then_some(binder.value),
                                binder.range,
                            )],
                            source,
                            filter,
                        },
                        clause_start..end,
                    ));
                    if clauses.len() + depth >= self.nesting {
                        return Err(self.fail("value nesting limit exceeded"));
                    }
                }
                let end = self
                    .expect(T::RBracket, "expected `]` after comprehension clauses")?
                    .range()
                    .end() as usize;
                return self.node(
                    K::Comprehension {
                        constructor: zirql::ValueConstructor::Array,
                        clauses,
                        head: Box::new(head),
                    },
                    start,
                    end,
                );
            }
            self.slots::<E>(1)?;
            children.push(head);
            while self.token().kind() == T::Comma {
                self.cursor += 1;
                let child = self.expression(0, depth + 1)?;
                self.slots::<E>(1)?;
                children.push(child);
            }
        }
        let end = self
            .expect(T::RBracket, "expected `]` after array elements")?
            .range()
            .end() as usize;
        self.node(K::Array(children), start, end)
    }
}
fn number(text: &str, position: usize) -> Result<V, QueryError> {
    let error = |message| QueryError { position, message };
    if !text.contains(['.', 'e', 'E']) {
        return text
            .parse::<i64>()
            .map(V::Integer)
            .map_err(|_| error("integer literal exceeds signed 64-bit range"));
    }
    let value = text
        .parse::<f64>()
        .map_err(|_| error("invalid decimal float"))?;
    if !value.is_finite() {
        return Err(error("nonfinite float literal"));
    }
    let mantissa = text.split(['e', 'E']).next().unwrap();
    if value == 0.0 && mantissa.bytes().any(|b| matches!(b, b'1'..=b'9')) {
        return Err(error("nonzero decimal literal rounded to zero"));
    }
    Ok(V::Float(value))
}

fn declared(
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<BTreeMap<String, ValueType>, EvaluationError> {
    let mut types = BTreeMap::new();
    for (name, kind) in names {
        budget
            .charge(name.len() + 2, Location::Query(range.clone()))
            .map_err(core_error)?;
        let ty = match kind {
            program::Kind::Strings => {
                ValueType::Array(ArrayType::Elements(vec![ValueType::String]))
            }
            program::Kind::Count => ValueType::Integer,
            program::Kind::String => ValueType::String,
            _ => ValueType::Unestablished,
        };
        types.insert(name.clone(), ty);
    }
    Ok(types)
}
pub(super) fn validate(
    value: &ParsedComprehension,
    names: &BTreeMap<String, program::Kind>,
    budget: &mut Budget,
) -> Result<(), EvaluationError> {
    let types = declared(names, budget, &value.expression.range)?;
    let free = zirql::free_value_names(&value.expression, &types, budget).map_err(core_error)?;
    for name in free {
        if matches!(
            names[&name.value],
            program::Kind::Operations | program::Kind::Types | program::Kind::Attributes
        ) {
            return Err(EvaluationError::new(
                "native handles require an explicit scalar projection",
            )
            .at(name.range));
        }
    }
    zirql::validate_value_expression(&value.expression, &types, budget).map_err(core_error)
}
pub(super) fn evaluate(
    value: &ParsedComprehension,
    bindings: &Bindings<QueryOutput>,
    budget: &mut Budget,
) -> Result<ConstructedReport, EvaluationError> {
    let mut declared = BTreeMap::new();
    for name in value.globals.iter() {
        budget
            .charge(
                name.len() + 1,
                Location::Query(value.expression.range.clone()),
            )
            .map_err(core_error)?;
        declared.insert(name.clone(), ValueType::Unestablished);
    }
    let names =
        zirql::free_value_names(&value.expression, &declared, budget).map_err(core_error)?;
    let mut projected = Vec::new();
    let mut types = BTreeMap::new();
    for name in names {
        budget
            .charge(name.value.len() + 20, Location::Query(name.range.clone()))
            .map_err(core_error)?;
        let source = bindings
            .get(&name.value, name.range.clone())
            .map_err(core_error)?;
        let source = project(source, budget, &name.range)?;
        let ty =
            zirql::constructed_value_type(&source, budget, Location::Query(name.range.clone()))
                .map_err(core_error)?;
        budget
            .charge(name.value.len() + 2, Location::Query(name.range.clone()))
            .map_err(core_error)?;
        types.insert(name.value.clone(), ty);
        projected.push((name, source));
    }
    zirql::validate_value_expression(&value.expression, &types, budget).map_err(core_error)?;
    let bindings = Bindings::new(projected).map_err(core_error)?;
    let result = zirql::evaluate_value(&value.expression, &bindings, budget).map_err(core_error)?;
    ConstructedReport::new(result, value.expression.range.clone(), budget)
}
fn admit(
    budget: &mut Budget,
    count: usize,
    bytes: usize,
    range: &Range<usize>,
) -> Result<(), EvaluationError> {
    if count
        .checked_mul(std::mem::size_of::<V>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(EvaluationError::new("value allocation size overflow").at(range.clone()));
    }
    budget
        .charge(
            count.checked_add(bytes).ok_or_else(|| {
                EvaluationError::new("value allocation size overflow").at(range.clone())
            })?,
            Location::Query(range.clone()),
        )
        .map_err(core_error)?;
    budget
        .charge_results(count, Location::Query(range.clone()))
        .map_err(core_error)
}
fn project(
    value: &QueryOutput,
    budget: &mut Budget,
    range: &Range<usize>,
) -> Result<V, EvaluationError> {
    match value {
        QueryOutput::Native(NativeValue::Constructed(value)) => {
            zirql::copy_constructed_value(value.value(), budget, Location::Query(range.clone()))
                .map_err(core_error)
        }
        QueryOutput::Values(values) => {
            admit(budget, 1 + values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                admit(budget, 1, value.len(), range)?;
                result.push(V::String(value.clone()));
            }
            Ok(V::Array(result))
        }
        QueryOutput::Count(count) => {
            admit(budget, 1, 0, range)?;
            Ok(V::Integer(i64::try_from(*count).map_err(|_| {
                EvaluationError::new("count exceeds signed 64-bit range").at(range.clone())
            })?))
        }
        QueryOutput::Native(NativeValue::String(value)) => {
            admit(budget, 1, value.len(), range)?;
            Ok(V::String(value.clone()))
        }
        QueryOutput::Array(values) => {
            admit(budget, 1 + values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                result.push(project_json(value, budget, range, 0)?);
            }
            Ok(V::Array(result))
        }
        QueryOutput::Map(fields) => {
            admit(budget, 1 + fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project_json(value, budget, range, 0)?));
            }
            Ok(V::Record(result))
        }
        QueryOutput::RankedMap(fields) => {
            admit(budget, 1 + fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project_json(value, budget, range, 0)?));
            }
            Ok(V::Record(result))
        }
        QueryOutput::Native(NativeValue::Map(fields)) => {
            admit(budget, 1 + fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project(value, budget, range)?));
            }
            Ok(V::Record(result))
        }
        _ => Err(EvaluationError::new(
            "native handles and rendered output require an explicit scalar projection",
        )
        .at(range.clone())),
    }
}
fn project_json(
    value: &serde_json::Value,
    budget: &mut Budget,
    range: &Range<usize>,
    depth: usize,
) -> Result<V, EvaluationError> {
    if depth >= 64 {
        return Err(EvaluationError::new("value nesting limit exceeded").at(range.clone()));
    }
    admit(budget, 1, 0, range)?;
    Ok(match value {
        serde_json::Value::Null => {
            return Err(
                EvaluationError::new("null is unsupported in positional comprehensions")
                    .at(range.clone()),
            );
        }
        serde_json::Value::Bool(v) => V::Boolean(*v),
        serde_json::Value::Number(v) => {
            if v.is_i64() || v.is_u64() {
                V::Integer(v.as_i64().ok_or_else(|| {
                    EvaluationError::new("integer exceeds signed 64-bit range").at(range.clone())
                })?)
            } else {
                V::Float(
                    v.as_f64()
                        .ok_or_else(|| EvaluationError::new("nonfinite float").at(range.clone()))?,
                )
            }
        }
        serde_json::Value::String(v) => {
            admit(budget, 0, v.len(), range)?;
            V::String(v.clone())
        }
        serde_json::Value::Array(values) => {
            admit(budget, values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                result.push(project_json(value, budget, range, depth + 1)?);
            }
            V::Array(result)
        }
        serde_json::Value::Object(fields) => {
            admit(budget, fields.len(), 0, range)?;
            let mut result = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.push((key.clone(), project_json(value, budget, range, depth + 1)?));
            }
            V::Record(result)
        }
    })
}
fn to_json(
    value: &V,
    budget: &mut Budget,
    range: &Range<usize>,
    depth: usize,
) -> Result<serde_json::Value, EvaluationError> {
    if depth >= 64 {
        return Err(EvaluationError::new("value nesting limit exceeded").at(range.clone()));
    }
    admit(budget, 1, 0, range)?;
    Ok(match value {
        V::Member(_) | V::KeyedArray { .. } => {
            return Err(EvaluationError::new(
                "nominal and keyed values are unsupported in Zirium reports",
            )
            .at(range.clone()));
        }
        V::Boolean(v) => serde_json::Value::Bool(*v),
        V::Integer(v) => serde_json::Value::Number((*v).into()),
        V::Float(v) => serde_json::Value::Number(
            serde_json::Number::from_f64(*v).expect("private finite report invariant"),
        ),
        V::String(v) => {
            admit(budget, 0, v.len(), range)?;
            serde_json::Value::String(v.clone())
        }
        V::Array(values)
        | V::Tuple(values)
        | V::Set {
            elements: values, ..
        } => {
            admit(budget, values.len(), 0, range)?;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                result.push(to_json(value, budget, range, depth + 1)?);
            }
            serde_json::Value::Array(result)
        }
        V::Record(fields) => {
            admit(budget, fields.len(), 0, range)?;
            let mut result = serde_json::Map::new();
            for (key, value) in fields {
                admit(budget, 0, key.len(), range)?;
                result.insert(key.clone(), to_json(value, budget, range, depth + 1)?);
            }
            serde_json::Value::Object(result)
        }
    })
}
