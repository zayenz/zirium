//! Type syntax and affine expressions embedded in types.

use super::*;

impl Parser<'_> {
    pub(in crate::parser) fn function_type(&mut self) -> Result<(), CompactError> {
        self.function_type_with_input_count().map(|_| ())
    }

    pub(in crate::parser) fn function_type_with_input_count(
        &mut self,
    ) -> Result<usize, CompactError> {
        let ty = self.builder.start();
        let input_count = self.type_list_with_count(0)?;
        self.trivia()?;
        self.expect(TokenKind::Arrow)?;
        self.trivia()?;
        if self.at_type_start() {
            self.type_syntax(0)?;
        } else {
            self.type_list(0)?;
        }
        self.builder.complete(ty, SyntaxKind::FunctionType)?;
        Ok(input_count)
    }

    pub(in crate::parser) fn type_list(&mut self, depth: usize) -> Result<(), CompactError> {
        self.type_list_with_count(depth).map(|_| ())
    }

    fn type_list_with_count(&mut self, depth: usize) -> Result<usize, CompactError> {
        self.expect(TokenKind::LParen)?;
        self.trivia()?;
        let mut count = 0;
        while self.at_type_start() {
            self.type_syntax(depth)?;
            count += 1;
            self.trivia()?;
            if self.at(TokenKind::LBrace) {
                self.attribute_dict()?;
                self.trivia()?;
            }
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        self.expect(TokenKind::RParen)?;
        Ok(count)
    }

    pub(in crate::parser) fn at_type_start(&self) -> bool {
        matches!(
            self.current(),
            TokenKind::IntType
                | TokenKind::FloatType
                | TokenKind::IndexType
                | TokenKind::Complex
                | TokenKind::ExclamationIdentifier
                | TokenKind::Tuple
                | TokenKind::Tensor
                | TokenKind::Vector
                | TokenKind::MemRef
                | TokenKind::AffineMap
                | TokenKind::AffineSet
        )
    }

    pub(in crate::parser) fn type_syntax(&mut self, depth: usize) -> Result<bool, CompactError> {
        if depth >= MAX_TYPE_DEPTH {
            self.diagnostic();
            self.recover_type_boundary()?;
            return Ok(false);
        }
        let kind = match self.current() {
            TokenKind::IntType => SyntaxKind::IntegerType,
            TokenKind::FloatType => SyntaxKind::FloatType,
            TokenKind::IndexType => SyntaxKind::IndexType,
            TokenKind::ExclamationIdentifier if self.nth_nontrivia(1) == Some(TokenKind::Less) => {
                return self.opaque(SyntaxKind::OpaqueType);
            }
            TokenKind::ExclamationIdentifier => SyntaxKind::TypeAlias,
            TokenKind::Complex => return self.single_element_type(depth + 1),
            TokenKind::Tuple => return self.tuple_type(depth + 1),
            TokenKind::Tensor => return self.shaped_type(SyntaxKind::TensorType, depth + 1),
            TokenKind::Vector => return self.shaped_type(SyntaxKind::VectorType, depth + 1),
            TokenKind::MemRef => return self.shaped_type(SyntaxKind::MemRefType, depth + 1),
            TokenKind::AffineMap => return self.affine_value(SyntaxKind::AffineMap),
            TokenKind::AffineSet => return self.affine_value(SyntaxKind::IntegerSet),
            _ => return Ok(false),
        };
        let marker = self.builder.start();
        self.bump()?;
        self.builder.complete(marker, kind)?;
        Ok(true)
    }

    fn single_element_type(&mut self, depth: usize) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        if self.at_type_start() {
            good &= self.type_syntax(depth)?;
        } else {
            good = false;
            self.diagnostic();
            self.recover_type_boundary()?;
        }
        self.trivia()?;
        good &= self.expect(TokenKind::Greater)?;
        self.builder
            .complete_with_error(marker, SyntaxKind::ComplexType, !good)?;
        Ok(good)
    }

    fn tuple_type(&mut self, depth: usize) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        while self.at_type_start() {
            good &= self.type_syntax(depth)?;
            self.trivia()?;
            if self.at(TokenKind::LBrace) {
                self.attribute_dict()?;
                self.trivia()?;
            }
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        good &= self.expect(TokenKind::Greater)?;
        self.builder
            .complete_with_error(marker, SyntaxKind::TupleType, !good)?;
        Ok(good)
    }

    fn shaped_type(&mut self, kind: SyntaxKind, depth: usize) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        while matches!(
            self.current(),
            TokenKind::Integer | TokenKind::Question | TokenKind::Star
        ) || (kind == SyntaxKind::VectorType && self.at(TokenKind::LBracket))
        {
            let dimension = self.builder.start();
            let mut dimension_good = true;
            if self.at(TokenKind::LBracket) {
                self.bump()?;
                self.trivia()?;
                dimension_good &= self.expect(TokenKind::Integer)?;
                self.trivia()?;
                dimension_good &= self.expect(TokenKind::RBracket)?;
            } else {
                self.bump()?;
            }
            self.builder.complete_with_error(
                dimension,
                SyntaxKind::ShapedDimension,
                !dimension_good,
            )?;
            good &= dimension_good;
            self.trivia()?;
            good &= self.expect(TokenKind::X)?;
            self.trivia()?;
        }
        if self.at_type_start() {
            good &= self.type_syntax(depth)?;
        } else {
            good = false;
            self.diagnostic();
            self.recover_type_boundary()?;
        }
        self.trivia()?;
        if kind == SyntaxKind::TensorType && self.at(TokenKind::Comma) {
            let encoding = self.builder.start();
            self.bump()?;
            self.trivia()?;
            let encoding_good = self.attribute_value()?;
            self.builder.complete_with_error(
                encoding,
                SyntaxKind::TensorEncoding,
                !encoding_good,
            )?;
            good &= encoding_good;
            self.trivia()?;
        } else if kind == SyntaxKind::MemRefType && self.at(TokenKind::Comma) {
            good &= self.memref_suffix()?;
        }
        good &= self.expect(TokenKind::Greater)?;
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn memref_suffix(&mut self) -> Result<bool, CompactError> {
        self.bump()?;
        self.trivia()?;
        let mut good = true;
        if self.at(TokenKind::Integer) {
            good &= self.memref_memory_space()?;
            return Ok(good);
        }

        let layout = self.builder.start();
        let layout_good = match self.current() {
            TokenKind::AffineMap => self.affine_value(SyntaxKind::AffineMap)?,
            TokenKind::Strided => self.balanced_angle_node(SyntaxKind::StridedLayout)?,
            _ => self.attribute_value()?,
        };
        self.builder
            .complete_with_error(layout, SyntaxKind::MemRefLayout, !layout_good)?;
        good &= layout_good;
        self.trivia()?;
        if self.at(TokenKind::Comma) {
            self.bump()?;
            self.trivia()?;
            good &= self.memref_memory_space()?;
        }
        Ok(good)
    }

    fn memref_memory_space(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        let good = if self.at(TokenKind::Integer) {
            self.bump()?;
            true
        } else {
            self.attribute_value()?
        };
        self.builder
            .complete_with_error(marker, SyntaxKind::MemRefMemorySpace, !good)?;
        self.trivia()?;
        Ok(good)
    }

    fn balanced_angle_node(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        let mut stack = vec![TokenKind::Greater];
        while let Some(expected) = stack.last().copied() {
            if self.at(TokenKind::Eof) || self.at(TokenKind::RBrace) {
                good = false;
                self.diagnostic();
                break;
            }
            let current = self.current();
            if current == expected {
                stack.pop();
            } else if let Some(close) = close_for(current) {
                if stack.len() >= MAX_TYPE_DEPTH {
                    good = false;
                    self.diagnostic();
                    break;
                }
                stack.push(close);
            }
            self.bump()?;
        }
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn affine_value(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        good &= self.affine_identifier_list()?;
        self.trivia()?;
        if self.at(TokenKind::LBracket) {
            good &= self.affine_identifier_list()?;
            self.trivia()?;
        }
        good &= self.expect(if kind == SyntaxKind::AffineMap {
            TokenKind::Arrow
        } else {
            TokenKind::Colon
        })?;
        self.trivia()?;
        good &= self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while !self.at(TokenKind::RParen)
            && !self.at(TokenKind::Greater)
            && !self.at(TokenKind::Eof)
        {
            let before = self.position;
            let expression = self.affine_expression(0, 0)?;
            good &= expression.is_some();
            self.trivia()?;
            if kind == SyntaxKind::IntegerSet {
                let constraint = expression
                    .map(|expr| self.builder.precede(expr))
                    .transpose()?;
                if matches!(
                    self.current(),
                    TokenKind::Less | TokenKind::Greater | TokenKind::Equal
                ) {
                    self.bump()?;
                    if self.at(TokenKind::Equal) {
                        self.bump()?;
                    } else {
                        good = false;
                        self.diagnostic();
                    }
                    self.trivia()?;
                    good &= self.affine_expression(0, 0)?.is_some();
                } else {
                    good = false;
                    self.diagnostic();
                }
                if let Some(constraint) = constraint {
                    self.builder.complete_with_error(
                        constraint,
                        SyntaxKind::AffineConstraint,
                        !good,
                    )?;
                }
            }
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else {
                break;
            }
            self.ensure_progress(before)?;
        }
        good &= self.expect(TokenKind::RParen)?;
        self.trivia()?;
        good &= self.expect(TokenKind::Greater)?;
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn affine_identifier_list(&mut self) -> Result<bool, CompactError> {
        let close = if self.at(TokenKind::LParen) {
            TokenKind::RParen
        } else {
            TokenKind::RBracket
        };
        let mut good = self.expect(if close == TokenKind::RParen {
            TokenKind::LParen
        } else {
            TokenKind::LBracket
        })?;
        self.trivia()?;
        while self.at_identifier() {
            self.bump()?;
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else {
                break;
            }
        }
        good &= self.expect(close)?;
        Ok(good)
    }

    fn affine_expression(
        &mut self,
        _min_precedence: u8,
        _depth: usize,
    ) -> Result<Option<CompletedMarker>, CompactError> {
        let mut operands = Vec::<CompletedMarker>::new();
        let mut operators = Vec::<u8>::new();
        let mut delimiters = Vec::<(usize, usize, Marker, Option<Marker>)>::new();
        let mut expect_operand = true;
        let mut bad = false;

        loop {
            self.trivia()?;
            if expect_operand {
                if self.at(TokenKind::Minus) {
                    let unary = self.builder.start();
                    self.bump()?;
                    self.trivia()?;
                    if self.at(TokenKind::LParen) {
                        if delimiters.len() >= MAX_TYPE_DEPTH {
                            self.diagnostic();
                            bad = true;
                            operands.push(self.builder.complete_with_error(
                                unary,
                                SyntaxKind::AffineExpression,
                                true,
                            )?);
                            break;
                        }
                        let marker = self.builder.start();
                        self.bump()?;
                        delimiters.push((operators.len(), operands.len(), marker, Some(unary)));
                        continue;
                    }
                    let good = self.at(TokenKind::Integer) || self.at_identifier();
                    if good {
                        self.bump()?;
                    } else {
                        self.diagnostic();
                        bad = true;
                    }
                    operands.push(self.builder.complete_with_error(
                        unary,
                        SyntaxKind::AffineExpression,
                        !good,
                    )?);
                    expect_operand = false;
                    continue;
                } else if self.at(TokenKind::LParen) {
                    if delimiters.len() >= MAX_TYPE_DEPTH {
                        self.diagnostic();
                        bad = true;
                        break;
                    }
                    let marker = self.builder.start();
                    self.bump()?;
                    delimiters.push((operators.len(), operands.len(), marker, None));
                    continue;
                }
                if self.at(TokenKind::Integer) || self.at_identifier() {
                    let marker = self.builder.start();
                    let good = self.at(TokenKind::Integer) || self.at_identifier();
                    if good {
                        self.bump()?;
                    } else {
                        self.diagnostic();
                        bad = true;
                    }
                    operands.push(self.builder.complete_with_error(
                        marker,
                        SyntaxKind::AffineExpression,
                        !good,
                    )?);
                    expect_operand = false;
                    continue;
                }
                self.diagnostic();
                bad = true;
                break;
            }

            let precedence = match self.current() {
                TokenKind::Plus | TokenKind::Minus => Some(1),
                TokenKind::Star | TokenKind::Mod | TokenKind::FloorDiv | TokenKind::CeilDiv => {
                    Some(2)
                }
                _ => None,
            };
            if let Some(precedence) = precedence {
                let floor = delimiters.last().map_or(0, |frame| frame.0);
                while operators.len() > floor
                    && operators.last().is_some_and(|top| *top >= precedence)
                {
                    bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
                }
                self.bump()?;
                operators.push(precedence);
                expect_operand = true;
                continue;
            }

            if self.at(TokenKind::RParen) && !delimiters.is_empty() {
                let (operator_base, operand_base, marker, unary) = delimiters.pop().unwrap();
                while operators.len() > operator_base {
                    bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
                }
                let inner_good = operands.len() == operand_base + 1 && !expect_operand;
                self.bump()?;
                let inner = operands.pop();
                operands.truncate(operand_base);
                bad |= inner.is_none();
                let mut grouped = self.builder.complete_with_error(
                    marker,
                    SyntaxKind::AffineExpression,
                    !inner_good,
                )?;
                if let Some(unary) = unary {
                    grouped = self.builder.complete_with_error(
                        unary,
                        SyntaxKind::AffineExpression,
                        !inner_good,
                    )?;
                }
                operands.push(grouped);
                expect_operand = false;
                continue;
            }
            break;
        }

        if expect_operand && !operators.is_empty() {
            self.diagnostic();
            bad = true;
        }
        while let Some((operator_base, operand_base, marker, unary)) = delimiters.pop() {
            while operators.len() > operator_base {
                bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
            }
            let inner = operands.pop();
            operands.truncate(operand_base);
            bad |= inner.is_none();
            let mut grouped =
                self.builder
                    .complete_with_error(marker, SyntaxKind::AffineExpression, true)?;
            if let Some(unary) = unary {
                grouped =
                    self.builder
                        .complete_with_error(unary, SyntaxKind::AffineExpression, true)?;
            }
            operands.push(grouped);
        }
        while !operators.is_empty() {
            bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
        }
        let result = operands.pop();
        if !operands.is_empty() {
            self.diagnostic();
            bad = true;
        }
        if bad && let Some(result) = result {
            let marker = self.builder.precede(result)?;
            return Ok(Some(self.builder.complete_with_error(
                marker,
                SyntaxKind::AffineExpression,
                true,
            )?));
        }
        Ok(result)
    }

    fn reduce_affine_operator(
        &mut self,
        operands: &mut Vec<CompletedMarker>,
        operators: &mut Vec<u8>,
    ) -> Result<bool, CompactError> {
        operators.pop();
        let Some(right) = operands.pop() else {
            return Ok(false);
        };
        let Some(left) = operands.pop() else {
            operands.push(right);
            return Ok(false);
        };
        let parent = self.builder.precede(left)?;
        operands.push(
            self.builder
                .complete(parent, SyntaxKind::AffineExpression)?,
        );
        Ok(true)
    }

    fn recover_type_boundary(&mut self) -> Result<(), CompactError> {
        while !matches!(
            self.current(),
            TokenKind::Comma
                | TokenKind::RParen
                | TokenKind::RBrace
                | TokenKind::Greater
                | TokenKind::Eof
        ) {
            self.error_token()?;
        }
        Ok(())
    }
}
