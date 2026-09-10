//! Affine expression parsing and semantic interning.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
enum AffineToken {
    Identifier(String),
    Integer(i64),
    InvalidInteger(String),
    InvalidOperator(String),
    Plus,
    Minus,
    Star,
    FloorDiv,
    CeilDiv,
    Mod,
    LParen,
    RParen,
}

pub(super) fn lower_affine_attribute(
    spelling: &str,
    range: TextRange,
    doc: &mut Document,
) -> AttributeValue {
    let (kind, inner) = if let Some(inner) = angle_inner(spelling, "affine_map") {
        (SyntaxKind::AffineMap, inner)
    } else if let Some(inner) = angle_inner(spelling, "affine_set") {
        (SyntaxKind::IntegerSet, inner)
    } else {
        let kind = if spelling.trim().starts_with("affine_map") {
            SyntaxKind::AffineMap
        } else {
            SyntaxKind::IntegerSet
        };
        return invalid_affine_attribute(kind, range, doc, "malformed affine value");
    };
    let Some(after_open) = inner.strip_prefix('(') else {
        return invalid_affine_attribute(kind, range, doc, "malformed affine dimension arity");
    };
    let Some(dim_tail_end) = matching_delimiter(after_open, '(', ')') else {
        return invalid_affine_attribute(kind, range, doc, "malformed affine dimension arity");
    };
    let dim_end = dim_tail_end + 1;
    let dimensions = parse_affine_names(&inner[1..dim_end], "dimension", range, doc);
    let mut rest = inner.get(dim_end + 1..).unwrap_or("").trim();
    let symbols = if rest.starts_with('[') {
        let Some(after_open) = rest.strip_prefix('[') else {
            return invalid_affine_attribute(kind, range, doc, "malformed affine symbol arity");
        };
        let Some(tail_end) = matching_delimiter(after_open, '[', ']') else {
            return invalid_affine_attribute(kind, range, doc, "malformed affine symbol arity");
        };
        let end = tail_end + 1;
        let names = parse_affine_names(&rest[1..end], "symbol", range, doc);
        rest = rest.get(end + 1..).unwrap_or("").trim();
        names
    } else {
        Vec::new()
    };
    let separator = if kind == SyntaxKind::AffineMap {
        "->"
    } else {
        ":"
    };
    let Some(body) = rest.strip_prefix(separator).map(str::trim) else {
        return invalid_affine_attribute(
            kind,
            range,
            doc,
            &format!("malformed affine {separator} separator"),
        );
    };
    let Some(body) = bracket_inner(body, '(', ')') else {
        return invalid_affine_attribute(
            kind,
            range,
            doc,
            "malformed affine result or constraint list",
        );
    };
    if kind == SyntaxKind::AffineMap {
        let results = split_affine_items(body)
            .into_iter()
            .map(|expression| {
                lower_affine_expression(expression, &dimensions, &symbols, range, doc)
            })
            .collect::<Vec<_>>();
        let value = AffineMapValue {
            dimensions: dimensions.len() as u32,
            symbols: symbols.len() as u32,
            results,
        };
        let index = intern_affine_map(doc, value);
        AttributeValue::AffineMap(AffineMapId::new(index, doc.generation))
    } else {
        let constraints = split_affine_items(body)
            .into_iter()
            .map(|constraint| {
                lower_integer_constraint(constraint, &dimensions, &symbols, range, doc)
            })
            .collect::<Vec<_>>();
        let value = IntegerSetValue {
            dimensions: dimensions.len() as u32,
            symbols: symbols.len() as u32,
            constraints,
        };
        let index = intern_integer_set(doc, value);
        AttributeValue::IntegerSet(IntegerSetId::new(index, doc.generation))
    }
}

fn invalid_affine_attribute(
    kind: SyntaxKind,
    range: TextRange,
    doc: &mut Document,
    message: &str,
) -> AttributeValue {
    let expression = invalid_affine_expression(doc, range, message);
    if kind == SyntaxKind::AffineMap {
        let index = intern_affine_map(
            doc,
            AffineMapValue {
                dimensions: 0,
                symbols: 0,
                results: vec![expression],
            },
        );
        AttributeValue::AffineMap(AffineMapId::new(index, doc.generation))
    } else {
        let diagnostic = push_diagnostic(
            doc,
            SemanticDiagnosticCode::Affine,
            range,
            message.to_owned(),
        );
        let right = invalid_affine_expression(doc, range, message);
        let index = intern_integer_set(
            doc,
            IntegerSetValue {
                dimensions: 0,
                symbols: 0,
                constraints: vec![IntegerSetConstraint {
                    left: expression,
                    relation: IntegerSetRelation::Invalid(diagnostic),
                    right,
                }],
            },
        );
        AttributeValue::IntegerSet(IntegerSetId::new(index, doc.generation))
    }
}

fn invalid_affine_expression(doc: &mut Document, range: TextRange, message: &str) -> AffineExprId {
    let diagnostic = push_diagnostic(
        doc,
        SemanticDiagnosticCode::Affine,
        range,
        message.to_owned(),
    );
    let index = intern_affine_expression(doc, AffineExprValue::Invalid(diagnostic));
    AffineExprId::new(index, doc.generation)
}

fn split_affine_items(value: &str) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut result = Vec::new();
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                result.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(value[start..].trim());
    result
}

fn parse_affine_names(
    value: &str,
    kind: &str,
    range: TextRange,
    doc: &mut Document,
) -> Vec<String> {
    let mut names = Vec::new();
    for name in value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        if !name.bytes().enumerate().all(|(i, byte)| {
            byte == b'_' || byte.is_ascii_alphanumeric() && (i > 0 || !byte.is_ascii_digit())
        }) {
            push_diagnostic(
                doc,
                SemanticDiagnosticCode::Affine,
                range,
                format!("malformed affine {kind} identifier `{name}`"),
            );
        }
        if names.iter().any(|existing| existing == name) {
            push_diagnostic(
                doc,
                SemanticDiagnosticCode::DuplicateDefinition,
                range,
                format!("duplicate affine {kind} identifier `{name}`"),
            );
        }
        names.push(name.to_owned());
    }
    names
}

fn lower_integer_constraint(
    value: &str,
    dimensions: &[String],
    symbols: &[String],
    range: TextRange,
    doc: &mut Document,
) -> IntegerSetConstraint {
    let mut depth = 0usize;
    let mut found = None;
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b'=' | b'>' | b'<' if depth == 0 => {
                found = Some((index, byte));
                break;
            }
            _ => {}
        }
    }
    let (left, relation, right) = match found {
        Some((index, b'=')) if value[index..].starts_with("==") => (
            &value[..index],
            IntegerSetRelation::Equal,
            &value[index + 2..],
        ),
        Some((index, b'>')) if value[index..].starts_with(">=") => (
            &value[..index],
            IntegerSetRelation::GreaterEqual,
            &value[index + 2..],
        ),
        Some((index, b'<')) if value[index..].starts_with("<=") => (
            &value[..index],
            IntegerSetRelation::LessEqual,
            &value[index + 2..],
        ),
        Some((index, _)) => {
            let diagnostic = push_diagnostic(
                doc,
                SemanticDiagnosticCode::Affine,
                range,
                format!("invalid affine constraint operator in `{value}`"),
            );
            (
                &value[..index],
                IntegerSetRelation::Invalid(diagnostic),
                &value[index + 1..],
            )
        }
        None => {
            let diagnostic = push_diagnostic(
                doc,
                SemanticDiagnosticCode::Affine,
                range,
                format!("missing affine constraint operator in `{value}`"),
            );
            (value, IntegerSetRelation::Invalid(diagnostic), "")
        }
    };
    IntegerSetConstraint {
        left: lower_affine_expression(left, dimensions, symbols, range, doc),
        relation,
        right: lower_affine_expression(right, dimensions, symbols, range, doc),
    }
}

fn lower_affine_expression(
    value: &str,
    dimensions: &[String],
    symbols: &[String],
    range: TextRange,
    doc: &mut Document,
) -> AffineExprId {
    let tokens = tokenize_affine(value);
    let mut parser = AffineExpressionParser {
        tokens: &tokens,
        position: 0,
        dimensions,
        symbols,
        range,
        doc,
    };
    let expression = parser.expression(0);
    if parser.position != tokens.len() {
        return parser.invalid(format!("malformed affine expression `{}`", value.trim()));
    }
    expression
}

fn tokenize_affine(value: &str) -> Vec<AffineToken> {
    let bytes = value.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let token = match bytes[i] {
            b'+' => {
                i += 1;
                AffineToken::Plus
            }
            b'-' => {
                i += 1;
                AffineToken::Minus
            }
            b'*' => {
                i += 1;
                AffineToken::Star
            }
            b'(' => {
                i += 1;
                AffineToken::LParen
            }
            b')' => {
                i += 1;
                AffineToken::RParen
            }
            b'0'..=b'9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let literal = &value[start..i];
                match literal.parse() {
                    Ok(value) => AffineToken::Integer(value),
                    Err(_) => AffineToken::InvalidInteger(literal.to_owned()),
                }
            }
            b'/' | b'%' => {
                let operator = bytes[i] as char;
                i += 1;
                AffineToken::InvalidOperator(operator.to_string())
            }
            _ => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &value[start..i.max(start + 1)];
                if i == start {
                    i += 1;
                }
                match word {
                    "floordiv" => AffineToken::FloorDiv,
                    "ceildiv" => AffineToken::CeilDiv,
                    "mod" => AffineToken::Mod,
                    _ => AffineToken::Identifier(word.to_owned()),
                }
            }
        };
        tokens.push(token);
    }
    tokens
}

struct AffineExpressionParser<'a> {
    tokens: &'a [AffineToken],
    position: usize,
    dimensions: &'a [String],
    symbols: &'a [String],
    range: TextRange,
    doc: &'a mut Document,
}

impl AffineExpressionParser<'_> {
    fn expression(&mut self, minimum: u8) -> AffineExprId {
        let mut left = self.primary();
        loop {
            let (precedence, operator) = match self.tokens.get(self.position) {
                Some(AffineToken::Plus) => (1, AffineBinaryOperator::Add),
                Some(AffineToken::Minus) => (1, AffineBinaryOperator::Subtract),
                Some(AffineToken::Star) => (2, AffineBinaryOperator::Multiply),
                Some(AffineToken::FloorDiv) => (2, AffineBinaryOperator::FloorDiv),
                Some(AffineToken::CeilDiv) => (2, AffineBinaryOperator::CeilDiv),
                Some(AffineToken::Mod) => (2, AffineBinaryOperator::Mod),
                _ => break,
            };
            if precedence < minimum {
                break;
            }
            self.position += 1;
            let right = self.expression(precedence + 1);
            left = self.intern(AffineExprValue::Binary {
                operator,
                left,
                right,
            });
        }
        left
    }
    fn primary(&mut self) -> AffineExprId {
        match self.tokens.get(self.position).cloned() {
            Some(AffineToken::Integer(value)) => {
                self.position += 1;
                self.intern(AffineExprValue::Constant(value))
            }
            Some(AffineToken::InvalidInteger(literal)) => {
                self.position += 1;
                self.invalid(format!(
                    "affine integer literal `{literal}` is out of range for i64"
                ))
            }
            Some(AffineToken::InvalidOperator(operator)) => {
                self.position += 1;
                self.invalid(format!("unsupported affine operator `{operator}`"))
            }
            Some(AffineToken::Identifier(name)) => {
                self.position += 1;
                if let Some(index) = self.dimensions.iter().position(|value| value == &name) {
                    self.intern(AffineExprValue::Dimension(index as u32))
                } else if let Some(index) = self.symbols.iter().position(|value| value == &name) {
                    self.intern(AffineExprValue::Symbol(index as u32))
                } else {
                    self.invalid(format!(
                        "affine identifier `{name}` exceeds declared dimension/symbol arity"
                    ))
                }
            }
            Some(AffineToken::Minus) => {
                self.position += 1;
                let right = self.primary();
                let zero = self.intern(AffineExprValue::Constant(0));
                self.intern(AffineExprValue::Binary {
                    operator: AffineBinaryOperator::Subtract,
                    left: zero,
                    right,
                })
            }
            Some(AffineToken::LParen) => {
                self.position += 1;
                let value = self.expression(0);
                if self.tokens.get(self.position) == Some(&AffineToken::RParen) {
                    self.position += 1;
                    value
                } else {
                    self.invalid("unclosed affine expression".into())
                }
            }
            _ => self.invalid("missing affine expression operand".into()),
        }
    }
    fn intern(&mut self, value: AffineExprValue) -> AffineExprId {
        let index = intern_affine_expression(self.doc, value);
        AffineExprId::new(index, self.doc.generation)
    }
    fn invalid(&mut self, message: String) -> AffineExprId {
        let diagnostic = push_diagnostic(
            self.doc,
            SemanticDiagnosticCode::Affine,
            self.range,
            message,
        );
        self.intern(AffineExprValue::Invalid(diagnostic))
    }
}

fn intern_affine_expression(doc: &mut Document, value: AffineExprValue) -> usize {
    if let Some(index) = doc
        .affine_expressions
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.affine_expressions.len();
        doc.affine_expressions.push(value);
        index
    }
}
fn intern_affine_map(doc: &mut Document, value: AffineMapValue) -> usize {
    if let Some(index) = doc
        .affine_maps
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.affine_maps.len();
        doc.affine_maps.push(value);
        index
    }
}
fn intern_integer_set(doc: &mut Document, value: IntegerSetValue) -> usize {
    if let Some(index) = doc
        .integer_sets
        .iter()
        .position(|existing| existing == &value)
    {
        index
    } else {
        let index = doc.integer_sets.len();
        doc.integer_sets.push(value);
        index
    }
}
