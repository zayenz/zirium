use super::*;

/// Constrained token/CST access passed to registered syntax callbacks.
///
/// It deliberately exposes no semantic document or arena.
pub struct DialectParser<'a, 'registry> {
    pub(super) parser: &'a mut Parser<'registry>,
    pub(super) marker: Marker,
    pub(super) descriptor: &'registry OperationDescriptor,
}

impl DialectParser<'_, '_> {
    pub fn parse_assembly_program(&mut self) -> Result<(), CompactError> {
        use crate::dialect::AssemblyProgram;
        match self
            .descriptor
            .assembly
            .expect("validated assembly program")
        {
            AssemblyProgram::Module => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::AtIdentifier) {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::BareIdentifier)
                    && self.parser.current_text() == "attributes"
                {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    good &= self.parser.at(TokenKind::LBrace);
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.region()?;
                } else {
                    self.parser.diagnostic();
                    good = false;
                }
                self.complete_operation(good)
            }
            AssemblyProgram::Function => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::BareIdentifier)
                    && matches!(self.parser.current_text(), "public" | "private" | "nested")
                {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                }
                good &= self.parser.expect(TokenKind::AtIdentifier)?;
                self.parser.trivia()?;
                good &= self
                    .parser
                    .block_argument_list(SyntaxKind::BlockArgumentList)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::Arrow) {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    if self.parser.at(TokenKind::LParen) {
                        self.parser.type_list(0)?;
                    } else {
                        good &= self.parser.type_syntax(0)?;
                    }
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::BareIdentifier)
                    && self.parser.current_text() == "attributes"
                {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.region()?;
                }
                self.complete_operation(good)
            }
            AssemblyProgram::Call => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                good &= self.parser.symbol_reference()?;
                self.parser.trivia()?;
                self.parser.operand_list()?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                good &= self.parser.expect(TokenKind::Colon)?;
                self.parser.trivia()?;
                self.parser.function_type()?;
                self.complete_operation(good)
            }
            AssemblyProgram::ConditionalBranch => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                good &= self.parse_operand()?;
                self.parser.trivia()?;
                good &= self.parser.expect(TokenKind::Comma)?;
                self.parser.trivia()?;
                let list = self.parser.builder.start();
                for index in 0..2 {
                    good &= self.parse_successor()?;
                    self.parser.trivia()?;
                    if index == 0 {
                        good &= self.parser.expect(TokenKind::Comma)?;
                        self.parser.trivia()?;
                    }
                }
                self.parser
                    .builder
                    .complete_with_error(list, SyntaxKind::SuccessorList, !good)?;
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                }
                self.complete_operation(good)
            }
            AssemblyProgram::TypedAttribute => self.parse_zero_operand_constant(),
            AssemblyProgram::BinaryOperands => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                for index in 0..2 {
                    if self.parser.at(TokenKind::PercentIdentifier) {
                        let operand = self.parser.builder.start();
                        let use_marker = self.parser.builder.start();
                        self.parser.bump()?;
                        self.parser
                            .builder
                            .complete(use_marker, SyntaxKind::OperandUse)?;
                        self.parser.builder.complete(operand, SyntaxKind::Operand)?;
                    } else {
                        good = false;
                        self.parser.diagnostic();
                    }
                    self.parser.trivia()?;
                    if index == 0 {
                        good &= self.parser.expect(TokenKind::Comma)?;
                        self.parser.trivia()?;
                    }
                }
                if self.parser.at(TokenKind::BareIdentifier) {
                    if self.parser.current_text() == "overflow" {
                        good &= self.parse_overflow_flags()?;
                        self.parser.trivia()?;
                    } else {
                        self.parser.diagnostic();
                        good = false;
                        self.parser.bump()?;
                    }
                } else if !self.parser.at(TokenKind::LBrace) && !self.parser.at(TokenKind::Colon) {
                    self.parser.diagnostic();
                    good = false;
                    self.parser.bump()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                good &= self.parser.expect(TokenKind::Colon)?;
                self.parser.trivia()?;
                good &= self.parser.type_syntax(0)?;
                self.complete_operation(good)
            }
            AssemblyProgram::OptionalTypedOperands => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                let mut operand_count = 0;
                while self.parser.at(TokenKind::PercentIdentifier) {
                    let operand = self.parser.builder.start();
                    let use_marker = self.parser.builder.start();
                    self.parser.bump()?;
                    self.parser
                        .builder
                        .complete(use_marker, SyntaxKind::OperandUse)?;
                    self.parser.builder.complete(operand, SyntaxKind::Operand)?;
                    operand_count += 1;
                    self.parser.trivia()?;
                    if !self.parser.at(TokenKind::Comma) {
                        break;
                    }
                    self.parser.bump()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::Colon) {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    let type_count = self.parse_return_type_list()?;
                    if type_count != operand_count {
                        self.parser.diagnostic();
                        good = false;
                    }
                } else if operand_count != 0 {
                    self.parser.diagnostic();
                    good = false;
                }
                self.complete_operation(good)
            }
            AssemblyProgram::TypedSuccessor => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                let list = self.parser.builder.start();
                let successor = self.parser.builder.start();
                good &= self.parser.expect(TokenKind::CaretIdentifier)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::LParen) {
                    good &= self
                        .parser
                        .block_argument_list(SyntaxKind::SuccessorArguments)?;
                    self.parser.trivia()?;
                }
                self.parser
                    .builder
                    .complete_with_error(successor, SyntaxKind::Successor, !good)?;
                self.parser
                    .builder
                    .complete_with_error(list, SyntaxKind::SuccessorList, !good)?;
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                }
                self.complete_operation(good)
            }
        }
    }

    fn parse_operand(&mut self) -> Result<bool, CompactError> {
        let operand = self.parser.builder.start();
        let use_marker = self.parser.builder.start();
        let good = self.parser.expect(TokenKind::PercentIdentifier)?;
        self.parser
            .builder
            .complete_with_error(use_marker, SyntaxKind::OperandUse, !good)?;
        self.parser
            .builder
            .complete_with_error(operand, SyntaxKind::Operand, !good)?;
        Ok(good)
    }

    fn parse_successor(&mut self) -> Result<bool, CompactError> {
        let successor = self.parser.builder.start();
        let mut good = self.parser.expect(TokenKind::CaretIdentifier)?;
        self.parser.trivia()?;
        if self.parser.at(TokenKind::LParen) {
            good &= self
                .parser
                .block_argument_list(SyntaxKind::SuccessorArguments)?;
        }
        self.parser
            .builder
            .complete_with_error(successor, SyntaxKind::Successor, !good)?;
        Ok(good)
    }

    pub fn parse_zero_operand_constant(&mut self) -> Result<(), CompactError> {
        let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
        self.parser.trivia()?;
        let value = self.parser.builder.start();
        let attr_good = self.parser.constant_value()?;
        good &= attr_good;
        self.parser
            .builder
            .complete_with_error(value, SyntaxKind::ArithConstantValue, !good)?;
        self.parser.trivia()?;
        if self.parser.at(TokenKind::LBrace) {
            self.parser.attribute_dict()?;
            self.parser.trivia()?;
        }
        let colon_good = self.parser.expect(TokenKind::Colon)?;
        good &= colon_good;
        self.parser.trivia()?;
        let type_good = self.parser.type_syntax(0)?;
        good &= type_good;
        self.complete_operation(good)
    }

    fn complete_operation(&mut self, mut good: bool) -> Result<(), CompactError> {
        if self.parser.nth_nontrivia(0) == Some(TokenKind::Loc) {
            self.parser.trivia()?;
            let location = self.parser.builder.start();
            let location_good = self.parser.location_attribute()?;
            good &= location_good;
            self.parser.builder.complete_with_error(
                location,
                SyntaxKind::TrailingLocation,
                !location_good,
            )?;
        }
        self.parser
            .builder
            .complete_with_error(self.marker, self.descriptor.syntax_kind, !good)?;
        Ok(())
    }

    fn parse_overflow_flags(&mut self) -> Result<bool, CompactError> {
        let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
        self.parser.trivia()?;
        good &= self.parser.expect(TokenKind::Less)?;
        self.parser.trivia()?;
        let mut seen_nsw = false;
        let mut seen_nuw = false;
        let mut seen_none = false;
        let mut count = 0;
        loop {
            if !self.parser.at(TokenKind::BareIdentifier) {
                good = false;
                self.parser.diagnostic();
                break;
            }
            let flag = self.parser.current_text();
            let duplicate = match flag {
                "nsw" => std::mem::replace(&mut seen_nsw, true),
                "nuw" => std::mem::replace(&mut seen_nuw, true),
                "none" => std::mem::replace(&mut seen_none, true),
                _ => true,
            };
            if duplicate || (flag == "none" && count != 0) || (flag != "none" && seen_none) {
                good = false;
                self.parser.diagnostic();
            }
            self.parser.bump()?;
            self.parser.trivia()?;
            count += 1;
            if count > 2 {
                good = false;
                self.parser.diagnostic();
            }
            if !self.parser.at(TokenKind::Comma) {
                break;
            }
            self.parser.bump()?;
            self.parser.trivia()?;
        }
        if count == 0 {
            good = false;
        }
        good &= self.parser.expect(TokenKind::Greater)?;
        Ok(good)
    }

    fn parse_return_type_list(&mut self) -> Result<usize, CompactError> {
        if self.parser.at(TokenKind::LParen) {
            self.parser.bump()?;
            self.parser.trivia()?;
            let mut count = 0;
            while self.parser.at_type_start() {
                let good = self.parser.type_syntax(0)?;
                count += usize::from(good);
                self.parser.trivia()?;
                if !self.parser.at(TokenKind::Comma) {
                    break;
                }
                self.parser.bump()?;
                self.parser.trivia()?;
            }
            self.parser.expect(TokenKind::RParen)?;
            Ok(count)
        } else {
            let good = self.parser.type_syntax(0)?;
            Ok(usize::from(good))
        }
    }
}

pub(super) fn shaped_operation(
    parser: &mut Parser<'_>,
    marker: Marker,
    shape: OperationShape,
) -> Result<(), CompactError> {
    let mut good = parser.expect(TokenKind::BareIdentifier)?;
    parser.trivia()?;
    match shape {
        OperationShape::FuncLike => {
            good &= parser.expect(TokenKind::AtIdentifier)?;
            parser.trivia()?;
            good &= parser.block_argument_list(SyntaxKind::BlockArgumentList)?;
            parser.trivia()?;
            if parser.at(TokenKind::Arrow) {
                parser.bump()?;
                parser.trivia()?;
                if parser.at(TokenKind::LParen) {
                    parser.type_list(0)?;
                } else {
                    good &= parser.type_syntax(0)?;
                }
                parser.trivia()?;
            }
            if parser.at(TokenKind::BareIdentifier) && parser.current_text() == "attributes" {
                parser.bump()?;
                parser.trivia()?;
                parser.attribute_dict()?;
                parser.trivia()?;
            }
            if parser.at(TokenKind::LBrace) {
                parser.region()?;
            }
        }
        OperationShape::CallLike => {
            good &= parser.symbol_reference()?;
            parser.trivia()?;
            parser.operand_list()?;
            parser.trivia()?;
            if parser.at(TokenKind::LBrace) {
                parser.attribute_dict()?;
                parser.trivia()?;
            }
            good &= parser.expect(TokenKind::Colon)?;
            parser.trivia()?;
            parser.function_type()?;
        }
        OperationShape::BinaryOperands => {
            for index in 0..2 {
                let operand = parser.builder.start();
                let use_marker = parser.builder.start();
                good &= parser.expect(TokenKind::PercentIdentifier)?;
                parser
                    .builder
                    .complete(use_marker, SyntaxKind::OperandUse)?;
                parser.builder.complete(operand, SyntaxKind::Operand)?;
                parser.trivia()?;
                if index == 0 {
                    good &= parser.expect(TokenKind::Comma)?;
                    parser.trivia()?;
                }
            }
            if parser.at(TokenKind::LBrace) {
                parser.attribute_dict()?;
                parser.trivia()?;
            }
            good &= parser.expect(TokenKind::Colon)?;
            parser.trivia()?;
            if binary_trailer_has_arrow(parser) {
                let parenthesized = parser.at(TokenKind::LParen);
                let input_count = if parenthesized {
                    parser.function_type_with_input_count()?
                } else {
                    bare_function_type(parser)?
                };
                if (parenthesized && input_count != 2)
                    || (!parenthesized && !matches!(input_count, 1 | 2))
                {
                    parser.diagnostic();
                    good = false;
                }
            } else {
                good &= parser.type_syntax(0)?;
            }
        }
        OperationShape::UnaryOperand => {
            good &= shaped_operand(parser)?;
            parser.trivia()?;
            if parser.at(TokenKind::LBrace) {
                parser.attribute_dict()?;
                parser.trivia()?;
            }
            good &= parser.expect(TokenKind::Colon)?;
            parser.trivia()?;
            if binary_trailer_has_arrow(parser) {
                let input_count = if parser.at(TokenKind::LParen) {
                    parser.function_type_with_input_count()?
                } else {
                    bare_function_type(parser)?
                };
                if input_count != 1 {
                    parser.diagnostic();
                    good = false;
                }
            } else {
                good &= parser.type_syntax(0)?;
            }
        }
        OperationShape::VariadicOperands => {
            while parser.at(TokenKind::PercentIdentifier) {
                good &= shaped_operand(parser)?;
                parser.trivia()?;
                if !parser.at(TokenKind::Comma) {
                    break;
                }
                parser.bump()?;
                parser.trivia()?;
            }
            if parser.at(TokenKind::LBrace) {
                parser.attribute_dict()?;
                parser.trivia()?;
            }
            good &= parser.expect(TokenKind::Colon)?;
            parser.trivia()?;
            if binary_trailer_has_arrow(parser) {
                if parser.at(TokenKind::LParen) {
                    parser.function_type_with_input_count()?;
                } else {
                    bare_function_type(parser)?;
                }
            } else {
                good &= parser.type_syntax(0)?;
                parser.trivia()?;
                while parser.at(TokenKind::Comma) {
                    parser.bump()?;
                    parser.trivia()?;
                    good &= parser.type_syntax(0)?;
                    parser.trivia()?;
                }
            }
        }
        OperationShape::LiteralAttribute => {
            let value = parser.builder.start();
            let value_good = parser.constant_value()?;
            good &= value_good;
            parser.builder.complete_with_error(
                value,
                SyntaxKind::ArithConstantValue,
                !value_good,
            )?;
            parser.trivia()?;
            good &= parser.expect(TokenKind::Colon)?;
            parser.trivia()?;
            good &= parser.type_syntax(0)?;
            parser.trivia()?;
            if parser.at(TokenKind::Colon) {
                parser.bump()?;
                parser.trivia()?;
                good &= parser.type_syntax(0)?;
            }
        }
        OperationShape::OptionalTypedOperands => {
            let mut operand_count = 0;
            while parser.at(TokenKind::PercentIdentifier) {
                let operand = parser.builder.start();
                let use_marker = parser.builder.start();
                parser.bump()?;
                parser
                    .builder
                    .complete(use_marker, SyntaxKind::OperandUse)?;
                parser.builder.complete(operand, SyntaxKind::Operand)?;
                operand_count += 1;
                parser.trivia()?;
                if !parser.at(TokenKind::Comma) {
                    break;
                }
                parser.bump()?;
                parser.trivia()?;
            }
            if parser.at(TokenKind::LBrace) {
                parser.attribute_dict()?;
                parser.trivia()?;
            }
            if parser.at(TokenKind::Colon) {
                parser.bump()?;
                parser.trivia()?;
                let type_count = if parser.at(TokenKind::LParen) {
                    parser.bump()?;
                    parser.trivia()?;
                    let mut count = 0;
                    while parser.at_type_start() {
                        count += usize::from(parser.type_syntax(0)?);
                        parser.trivia()?;
                        if !parser.at(TokenKind::Comma) {
                            break;
                        }
                        parser.bump()?;
                        parser.trivia()?;
                    }
                    good &= parser.expect(TokenKind::RParen)?;
                    count
                } else {
                    usize::from(parser.type_syntax(0)?)
                };
                if type_count != operand_count {
                    parser.diagnostic();
                    good = false;
                }
            } else if operand_count != 0 {
                parser.diagnostic();
                good = false;
            }
        }
    }
    parser.trivia()?;
    if parser.at(TokenKind::Loc) {
        let location = parser.builder.start();
        let location_good = parser.location_attribute()?;
        good &= location_good;
        parser.builder.complete_with_error(
            location,
            SyntaxKind::TrailingLocation,
            !location_good,
        )?;
    }
    parser
        .builder
        .complete_with_error(marker, SyntaxKind::DialectOperation, !good)?;
    Ok(())
}

fn shaped_operand(parser: &mut Parser<'_>) -> Result<bool, CompactError> {
    let operand = parser.builder.start();
    let use_marker = parser.builder.start();
    let good = parser.expect(TokenKind::PercentIdentifier)?;
    parser
        .builder
        .complete_with_error(use_marker, SyntaxKind::OperandUse, !good)?;
    parser
        .builder
        .complete_with_error(operand, SyntaxKind::Operand, !good)?;
    Ok(good)
}

fn binary_trailer_has_arrow(parser: &Parser<'_>) -> bool {
    let mut depth = 0usize;
    for token in &parser.tokens[parser.position..] {
        match token.kind() {
            TokenKind::Whitespace
                if depth == 0
                    && parser.source
                        [token.range().start() as usize..token.range().end() as usize]
                        .contains(&b'\n') =>
            {
                return false;
            }
            TokenKind::Whitespace | TokenKind::LineComment => {}
            TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace | TokenKind::Less => {
                depth += 1;
            }
            TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace | TokenKind::Greater => {
                let Some(next_depth) = depth.checked_sub(1) else {
                    return false;
                };
                depth = next_depth;
            }
            TokenKind::Arrow if depth == 0 => return true,
            TokenKind::Loc | TokenKind::Eof if depth == 0 => return false,
            _ => {}
        }
    }
    false
}

fn bare_function_type(parser: &mut Parser<'_>) -> Result<usize, CompactError> {
    let ty = parser.builder.start();
    let mut input_count = 0;
    loop {
        parser.type_syntax(0)?;
        input_count += 1;
        parser.trivia()?;
        if !parser.at(TokenKind::Comma) {
            break;
        }
        parser.bump()?;
        parser.trivia()?;
    }
    parser.expect(TokenKind::Arrow)?;
    parser.trivia()?;
    if parser.at_type_start() {
        parser.type_syntax(0)?;
    } else {
        parser.type_list(0)?;
    }
    parser.builder.complete(ty, SyntaxKind::FunctionType)?;
    Ok(input_count)
}
