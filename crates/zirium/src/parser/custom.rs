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
            AssemblyProgram::BuiltinModule => {
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
            AssemblyProgram::FuncFunc => {
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
            AssemblyProgram::FuncCall => {
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
            AssemblyProgram::CfCondBr => {
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
            AssemblyProgram::ArithConstant => self.parse_zero_operand_constant(),
            AssemblyProgram::FuncReturn => {
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
            AssemblyProgram::CfBr => {
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
            let mut count = usize::from(self.parser.type_syntax(0)?);
            while self.parser.nth_nontrivia(0) == Some(TokenKind::Comma) {
                self.parser.trivia()?;
                self.parser.bump()?;
                self.parser.trivia()?;
                count += usize::from(self.parser.type_syntax(0)?);
            }
            Ok(count)
        }
    }
}

pub(super) fn shaped_operation(
    parser: &mut Parser<'_>,
    marker: Marker,
    shape: OperationShape,
    operation: &str,
) -> Result<(), CompactError> {
    let checkpoint = parser.shaped_operation_checkpoint();
    parser.reset_attempt_failure(checkpoint.0);
    if shaped_operation_attempt(parser, marker, shape, operation)? {
        Ok(())
    } else {
        parser.recover_shape_mismatch(marker, shape, checkpoint)
    }
}

pub(super) fn alternative_operation(
    parser: &mut Parser<'_>,
    marker: Marker,
    alternatives: &[OperationGrammar],
    operation: &str,
) -> Result<(), CompactError> {
    let checkpoint = parser.shaped_operation_checkpoint();
    parser.reset_attempt_failure(checkpoint.0);
    for (index, alternative) in alternatives.iter().enumerate() {
        parser.rewind_shaped_operation(checkpoint);
        for _ in 0..index {
            let selected = parser.builder.start();
            parser
                .builder
                .complete(selected, SyntaxKind::FormatAlternative)?;
        }
        let matched = match alternative {
            OperationGrammar::Shape(shape) => {
                shaped_operation_attempt(parser, marker, *shape, operation)?
            }
            OperationGrammar::Format(format) => {
                formatted_operation_attempt(parser, marker, format)?
            }
        };
        if matched {
            return Ok(());
        }
    }
    parser.rewind_shaped_operation(checkpoint);
    parser.recover_format_mismatch(marker, checkpoint)
}

pub(super) fn shaped_operation_attempt(
    parser: &mut Parser<'_>,
    marker: Marker,
    shape: OperationShape,
    operation: &str,
) -> Result<bool, CompactError> {
    let checkpoint = parser.shaped_operation_checkpoint();
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
            match shaped_type_trailer(parser) {
                ShapedTypeTrailer::Function => {
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
                }
                ShapedTypeTrailer::Conversion => good &= conversion_type_trailer(parser)?,
                ShapedTypeTrailer::Shared => good &= parser.type_syntax(0)?,
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
            match shaped_type_trailer(parser) {
                ShapedTypeTrailer::Function => {
                    let input_count = if parser.at(TokenKind::LParen) {
                        parser.function_type_with_input_count()?
                    } else {
                        bare_function_type(parser)?
                    };
                    if input_count != 1 {
                        parser.diagnostic();
                        good = false;
                    }
                }
                ShapedTypeTrailer::Conversion => good &= conversion_type_trailer(parser)?,
                ShapedTypeTrailer::Shared => good &= parser.type_syntax(0)?,
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
            match shaped_type_trailer(parser) {
                ShapedTypeTrailer::Function => {
                    if parser.at(TokenKind::LParen) {
                        parser.function_type_with_input_count()?;
                    } else {
                        bare_function_type(parser)?;
                    }
                }
                ShapedTypeTrailer::Conversion => good &= conversion_type_trailer(parser)?,
                ShapedTypeTrailer::Shared => {
                    good &= parser.type_syntax(0)?;
                    while parser.nth_nontrivia(0) == Some(TokenKind::Comma) {
                        parser.trivia()?;
                        parser.bump()?;
                        parser.trivia()?;
                        good &= parser.type_syntax(0)?;
                    }
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
            let dictionary_before_type = parser.at(TokenKind::LBrace);
            if dictionary_before_type {
                parser.attribute_dict()?;
                parser.trivia()?;
            }
            if parser.at(TokenKind::Colon) {
                parser.bump()?;
                parser.trivia()?;
                good &= parser.type_syntax(0)?;
                parser.trivia()?;
                if !dictionary_before_type {
                    let dictionary_after_type = parser.at(TokenKind::LBrace);
                    if dictionary_after_type {
                        parser.attribute_dict()?;
                        parser.trivia()?;
                    }
                    if dictionary_after_type || parser.at(TokenKind::Colon) {
                        good &= parser.expect(TokenKind::Colon)?;
                        parser.trivia()?;
                        good &= parser.type_syntax(0)?;
                    }
                }
            }
        }
        OperationShape::OperandClauses => good &= operand_clauses(parser, operation)?,
        OperationShape::RegionClauses => good &= region_clauses(parser, operation)?,
        OperationShape::OptionalTypedOperands => {
            good &= optional_typed_operands(parser, false)?;
        }
        OperationShape::AttrFirstOptionalTypedOperands => {
            good &= optional_typed_operands(parser, true)?;
        }
    }
    let mut boundary_trivia = parser.position;
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
        boundary_trivia = parser.position;
        parser.trivia()?;
    }
    let crossed_line = parser.trivia_crosses_line(boundary_trivia);
    if !good || !parser.shaped_operation_boundary(crossed_line) {
        parser.record_attempt_failure();
        parser.rewind_shaped_operation(checkpoint);
        return Ok(false);
    }
    parser
        .builder
        .complete_with_error(marker, SyntaxKind::DialectOperation, !good)?;
    Ok(true)
}

fn optional_typed_operands(
    parser: &mut Parser<'_>,
    dictionary_first: bool,
) -> Result<bool, CompactError> {
    let mut good = true;
    if dictionary_first && parser.at(TokenKind::LBrace) {
        parser.attribute_dict()?;
        parser.trivia()?;
    }

    let mut operand_count = 0;
    while parser.at(TokenKind::PercentIdentifier) {
        good &= shaped_operand(parser)?;
        operand_count += 1;
        parser.trivia()?;
        if !parser.at(TokenKind::Comma) {
            break;
        }
        parser.bump()?;
        parser.trivia()?;
    }

    if !dictionary_first && parser.at(TokenKind::LBrace) {
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
    Ok(good)
}

fn operand_clauses(parser: &mut Parser<'_>, operation: &str) -> Result<bool, CompactError> {
    let mut good = true;
    let mut delimiters = Vec::new();
    loop {
        if parser.at(TokenKind::Eof)
            || (delimiters.is_empty()
                && matches!(
                    parser.current(),
                    TokenKind::RBrace | TokenKind::CaretIdentifier
                ))
        {
            parser.diagnostic();
            return Ok(false);
        }

        if delimiters.is_empty() && parser.at(TokenKind::Colon) && signature_type_follows(parser) {
            parser.bump()?;
            parser.trivia()?;
            match shaped_type_trailer(parser) {
                ShapedTypeTrailer::Function => {
                    if parser.at(TokenKind::LParen) {
                        parser.function_type_with_input_count()?;
                    } else {
                        bare_function_type(parser)?;
                    }
                }
                ShapedTypeTrailer::Conversion => good &= conversion_type_trailer(parser)?,
                ShapedTypeTrailer::Shared => {
                    good &= parser.type_syntax(0)?;
                    while parser.nth_nontrivia(0) == Some(TokenKind::Comma) {
                        parser.trivia()?;
                        parser.bump()?;
                        parser.trivia()?;
                        good &= parser.type_syntax(0)?;
                    }
                }
            }
            return Ok(good);
        }
        if delimiters.is_empty() && parser.at(TokenKind::Arrow) {
            parser.bump()?;
            parser.trivia()?;
            if parser.at(TokenKind::LParen) {
                parser.type_list(0)?;
            } else {
                good &= parser.type_syntax(0)?;
            }
            return Ok(good);
        }

        if parser.at(TokenKind::PercentIdentifier) {
            good &= shaped_operand(parser)?;
            parser.trivia()?;
            continue;
        }
        if delimiters.is_empty() && parser.at(TokenKind::LBrace) {
            parser.attribute_dict()?;
            parser.trivia()?;
            continue;
        }
        if delimiters.is_empty()
            && parser.inherent_attribute_starts()
            && !named_delimited_rhs_contains_operand(parser)
        {
            let paired = operation == "stablehlo.dot_general"
                && matches!(parser.current_text(), "contracting_dims" | "batching_dims");
            good &= parser.inherent_attribute(paired)?;
            parser.trivia()?;
            continue;
        }

        let current = parser.current();
        if delimiters.last() == Some(&current) {
            delimiters.pop();
        } else if let Some(close) = close_for(current) {
            if delimiters.len() >= parser.limits.max_delimiter_depth {
                parser.diagnostic();
                good = false;
            } else {
                delimiters.push(close);
            }
        } else if matches!(
            current,
            TokenKind::Greater | TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket
        ) && delimiters.is_empty()
        {
            parser.diagnostic();
            return Ok(false);
        }
        parser.bump()?;
        parser.trivia()?;
    }
}

fn region_clauses(parser: &mut Parser<'_>, operation: &str) -> Result<bool, CompactError> {
    let mut good = true;
    let mut delimiters = Vec::new();
    let mut region_count = 0usize;
    let mut positional_bindings = matches!(operation, "scf.forall" | "scf.parallel");
    let mut regionless_reduce = false;
    loop {
        if parser.at(TokenKind::Eof)
            || (delimiters.is_empty()
                && matches!(
                    parser.current(),
                    TokenKind::RBrace | TokenKind::CaretIdentifier
                ))
        {
            if region_count == 0 {
                parser.diagnostic();
                return Ok(false);
            }
            return Ok(good);
        }

        if delimiters.is_empty() && parser.at(TokenKind::LBrace) {
            let empty = parser.nth_nontrivia(1) == Some(TokenKind::RBrace);
            if parser.region_shaped_body() || empty || region_count != 0 {
                parser.region()?;
                region_count += 1;
                if region_continuation_follows(parser) {
                    parser.trivia()?;
                    continue;
                }
                return Ok(good);
            }
            parser.attribute_dict()?;
            parser.trivia()?;
            continue;
        }

        if delimiters.is_empty() && parser.at(TokenKind::Colon) && signature_type_follows(parser) {
            parser.bump()?;
            parser.trivia()?;
            match shaped_type_trailer(parser) {
                ShapedTypeTrailer::Function => {
                    if parser.at(TokenKind::LParen) {
                        parser.function_type_with_input_count()?;
                    } else {
                        bare_function_type(parser)?;
                    }
                }
                ShapedTypeTrailer::Conversion => good &= conversion_type_trailer(parser)?,
                ShapedTypeTrailer::Shared => good &= parser.type_syntax(0)?,
            }
            parser.trivia()?;
            continue;
        }

        if parser.at(TokenKind::PercentIdentifier) {
            let typed_binding = operation.starts_with("stablehlo.")
                && parser.nth_nontrivia(1) == Some(TokenKind::Colon);
            let binding = typed_binding
                || parser.nth_nontrivia(1) == Some(TokenKind::Equal)
                || (positional_bindings && !delimiters.is_empty());
            if binding {
                header_block_argument(parser, typed_binding)?;
            } else {
                good &= shaped_operand(parser)?;
            }
            parser.trivia()?;
            continue;
        }
        if delimiters.is_empty()
            && parser.inherent_attribute_starts()
            && !named_delimited_rhs_contains_operand(parser)
        {
            good &= parser.inherent_attribute(false)?;
            if regionless_reduce {
                parser.trivia()?;
                if parser.at(TokenKind::Colon) && signature_type_follows(parser) {
                    parser.bump()?;
                    parser.trivia()?;
                    match shaped_type_trailer(parser) {
                        ShapedTypeTrailer::Function => {
                            if parser.at(TokenKind::LParen) {
                                parser.function_type_with_input_count()?;
                            } else {
                                bare_function_type(parser)?;
                            }
                        }
                        ShapedTypeTrailer::Conversion => good &= conversion_type_trailer(parser)?,
                        ShapedTypeTrailer::Shared => good &= parser.type_syntax(0)?,
                    }
                }
                return Ok(good);
            }
            parser.trivia()?;
            continue;
        }

        let current = parser.current();
        if operation == "stablehlo.reduce"
            && delimiters.is_empty()
            && current == TokenKind::BareIdentifier
            && parser.current_text() == "applies"
        {
            regionless_reduce = true;
        }
        if delimiters.is_empty()
            && ((operation == "scf.forall"
                && current == TokenKind::BareIdentifier
                && parser.current_text() == "in")
                || (operation == "scf.parallel" && current == TokenKind::Equal))
        {
            positional_bindings = false;
        }
        if delimiters.last() == Some(&current) {
            delimiters.pop();
        } else if let Some(close) = close_for(current) {
            if delimiters.len() >= parser.limits.max_delimiter_depth {
                parser.diagnostic();
                good = false;
            } else {
                delimiters.push(close);
            }
        } else if matches!(
            current,
            TokenKind::Greater | TokenKind::RParen | TokenKind::RBracket
        ) && delimiters.is_empty()
        {
            parser.diagnostic();
            return Ok(false);
        }
        parser.bump()?;
        parser.trivia()?;
    }
}

fn region_continuation_follows(parser: &Parser<'_>) -> bool {
    let mut tokens = (parser.position..parser.tokens.len())
        .filter(|index| !is_trivia(parser.tokens[*index].kind()));
    let Some(first) = tokens.next() else {
        return false;
    };

    let region = match parser.tokens[first].kind() {
        TokenKind::LBrace => first,
        TokenKind::Comma => match tokens.next() {
            Some(index) if parser.tokens[index].kind() == TokenKind::LBrace => index,
            _ => return false,
        },
        TokenKind::BareIdentifier => match parser.nth_nontrivia_text(0) {
            Some("else" | "do" | "default") => match tokens.next() {
                Some(index) if parser.tokens[index].kind() == TokenKind::LBrace => index,
                _ => return false,
            },
            Some("case") => {
                let Some(mut selector) = tokens.next() else {
                    return false;
                };
                if parser.tokens[selector].kind() == TokenKind::Minus {
                    let Some(value) = tokens.next() else {
                        return false;
                    };
                    selector = value;
                }
                if !matches!(
                    parser.tokens[selector].kind(),
                    TokenKind::Integer | TokenKind::WideInteger
                ) {
                    return false;
                }
                match tokens.next() {
                    Some(index) if parser.tokens[index].kind() == TokenKind::LBrace => index,
                    _ => return false,
                }
            }
            _ => return false,
        },
        _ => return false,
    };

    balanced_region_follows(parser, region)
}

fn balanced_region_follows(parser: &Parser<'_>, start: usize) -> bool {
    let mut delimiters = vec![TokenKind::RBrace];
    for token in &parser.tokens[start + 1..] {
        let current = token.kind();
        if delimiters.last() == Some(&current) {
            delimiters.pop();
            if delimiters.is_empty() {
                return true;
            }
        } else if let Some(close) = close_for(current) {
            if delimiters.len() >= parser.limits.max_delimiter_depth {
                return false;
            }
            delimiters.push(close);
        } else if matches!(
            current,
            TokenKind::Greater | TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket
        ) || current == TokenKind::Eof
        {
            return false;
        }
    }
    false
}

fn named_delimited_rhs_contains_operand(parser: &Parser<'_>) -> bool {
    let mut tokens = parser.tokens[parser.position..]
        .iter()
        .filter(|token| !matches!(token.kind(), TokenKind::Whitespace | TokenKind::LineComment));
    let Some(name) = tokens.next() else {
        return false;
    };
    if !matches!(name.kind(), TokenKind::BareIdentifier | TokenKind::String)
        || tokens.next().map(|token| token.kind()) != Some(TokenKind::Equal)
    {
        return false;
    }
    let Some(open) = tokens.next() else {
        return false;
    };
    let Some(close) = close_for(open.kind()) else {
        return false;
    };

    let mut delimiters = vec![close];
    let mut contains_operand = false;
    for token in tokens {
        let current = token.kind();
        if delimiters.last() == Some(&current) {
            delimiters.pop();
            if delimiters.is_empty() {
                return contains_operand;
            }
        } else if let Some(close) = close_for(current) {
            if delimiters.len() >= parser.limits.max_delimiter_depth {
                return false;
            }
            delimiters.push(close);
        } else if matches!(
            current,
            TokenKind::Greater | TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket
        ) {
            return false;
        } else if current == TokenKind::PercentIdentifier {
            contains_operand = true;
        } else if current == TokenKind::Eof {
            return false;
        }
    }
    false
}

fn header_block_argument(parser: &mut Parser<'_>, typed: bool) -> Result<(), CompactError> {
    let list = parser.builder.start();
    let argument = parser.builder.start();
    parser.bump()?;
    if typed {
        parser.trivia()?;
        parser.expect(TokenKind::Colon)?;
        parser.trivia()?;
        parser.type_syntax(0)?;
    }
    parser
        .builder
        .complete(argument, SyntaxKind::BlockArgument)?;
    parser
        .builder
        .complete(list, SyntaxKind::BlockArgumentList)?;
    Ok(())
}

fn signature_type_follows(parser: &Parser<'_>) -> bool {
    matches!(
        parser.nth_nontrivia(1),
        Some(
            TokenKind::LParen
                | TokenKind::IntType
                | TokenKind::FloatType
                | TokenKind::IndexType
                | TokenKind::ExclamationIdentifier
                | TokenKind::Tuple
                | TokenKind::Tensor
                | TokenKind::Vector
                | TokenKind::MemRef
                | TokenKind::AffineMap
                | TokenKind::AffineSet
        )
    )
}

pub(super) fn formatted_operation(
    parser: &mut Parser<'_>,
    marker: Marker,
    format: &OperationFormat,
) -> Result<(), CompactError> {
    let checkpoint = parser.shaped_operation_checkpoint();
    parser.reset_attempt_failure(checkpoint.0);
    if formatted_operation_attempt(parser, marker, format)? {
        Ok(())
    } else {
        parser.recover_format_mismatch(marker, checkpoint)
    }
}

pub(super) fn formatted_operation_attempt(
    parser: &mut Parser<'_>,
    marker: Marker,
    format: &OperationFormat,
) -> Result<bool, CompactError> {
    let checkpoint = parser.shaped_operation_checkpoint();
    let mut good = parser.expect(TokenKind::BareIdentifier)?;
    parser.trivia()?;
    let mut nodes = Vec::new();
    let mut operand_count = 0;
    let mut boundary_trivia = parser.position;
    for step in format.steps() {
        let consumes_source = !matches!(step, FormatStep::Begin(_) | FormatStep::End(_));
        match step {
            FormatStep::Begin(kind) => nodes.push((*kind, parser.builder.start())),
            FormatStep::End(kind) => {
                let Some((started_kind, started)) = nodes.pop() else {
                    good = false;
                    parser.diagnostic();
                    continue;
                };
                debug_assert_eq!(started_kind, *kind);
                parser.builder.complete(started, *kind)?;
            }
            FormatStep::Capture(FormatCapture::Operands) => {
                while parser.at(TokenKind::PercentIdentifier) {
                    good &= shaped_operand(parser)?;
                    operand_count += 1;
                    parser.trivia()?;
                    if !parser.at(TokenKind::Comma) {
                        break;
                    }
                    parser.bump()?;
                    parser.trivia()?;
                }
            }
            FormatStep::Capture(FormatCapture::Operand(_)) => {
                good &= shaped_operand(parser)?;
                operand_count += 1;
            }
            FormatStep::Capture(FormatCapture::Value) => good &= parser.constant_value()?,
            FormatStep::Capture(FormatCapture::Callee) => good &= parser.symbol_reference()?,
            FormatStep::AttributeDictionary => {
                if parser.at(TokenKind::LBrace) {
                    parser.attribute_dict()?;
                }
            }
            FormatStep::Literal(literal) => {
                if parser.current_text() == literal {
                    parser.bump()?;
                } else {
                    parser.diagnostic();
                    good = false;
                }
            }
            FormatStep::Type(capture) => {
                if capture.is_per_operand_list() {
                    for index in 0..operand_count {
                        if index > 0 {
                            good &= parser.expect(TokenKind::Comma)?;
                            parser.trivia()?;
                        }
                        good &= parser.type_syntax(0)?;
                        parser.trivia()?;
                    }
                } else if capture.accepts_parenthesized_list() && parser.at(TokenKind::LParen) {
                    parser.type_list(0)?;
                } else {
                    good &= parser.type_syntax(0)?;
                }
            }
        }
        if consumes_source {
            boundary_trivia = parser.position;
            parser.trivia()?;
        }
    }
    debug_assert!(nodes.is_empty());

    if parser.at(TokenKind::Loc) {
        let location = parser.builder.start();
        let location_good = parser.location_attribute()?;
        good &= location_good;
        parser.builder.complete_with_error(
            location,
            SyntaxKind::TrailingLocation,
            !location_good,
        )?;
        boundary_trivia = parser.position;
        parser.trivia()?;
    }
    let crossed_line = parser.trivia_crosses_line(boundary_trivia);
    if !good || !parser.shaped_operation_boundary(crossed_line) {
        parser.record_attempt_failure();
        parser.rewind_shaped_operation(checkpoint);
        return Ok(false);
    }
    parser
        .builder
        .complete(marker, SyntaxKind::DialectOperation)?;
    Ok(true)
}

fn shaped_operand(parser: &mut Parser<'_>) -> Result<bool, CompactError> {
    let operand = parser.builder.start();
    let use_marker = parser.builder.start();
    let good = parser.expect(TokenKind::PercentIdentifier)?;
    if parser.at(TokenKind::HashIdentifier) {
        parser.bump()?;
    }
    parser
        .builder
        .complete_with_error(use_marker, SyntaxKind::OperandUse, !good)?;
    parser
        .builder
        .complete_with_error(operand, SyntaxKind::Operand, !good)?;
    Ok(good)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapedTypeTrailer {
    Shared,
    Function,
    Conversion,
}

fn shaped_type_trailer(parser: &Parser<'_>) -> ShapedTypeTrailer {
    let mut depth = 0usize;
    for token in &parser.tokens[parser.position..] {
        match token.kind() {
            TokenKind::Whitespace
                if depth == 0
                    && parser.source
                        [token.range().start() as usize..token.range().end() as usize]
                        .contains(&b'\n') =>
            {
                return ShapedTypeTrailer::Shared;
            }
            TokenKind::Whitespace | TokenKind::LineComment => {}
            TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace | TokenKind::Less => {
                depth += 1;
            }
            TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace | TokenKind::Greater => {
                let Some(next_depth) = depth.checked_sub(1) else {
                    return ShapedTypeTrailer::Shared;
                };
                depth = next_depth;
            }
            TokenKind::Arrow if depth == 0 => return ShapedTypeTrailer::Function,
            TokenKind::BareIdentifier if depth == 0 => {
                let range = token.range();
                if parser.source[range.start() as usize..range.end() as usize] == *b"to" {
                    return ShapedTypeTrailer::Conversion;
                }
            }
            TokenKind::Loc | TokenKind::Eof if depth == 0 => {
                return ShapedTypeTrailer::Shared;
            }
            _ => {}
        }
    }
    ShapedTypeTrailer::Shared
}

fn conversion_type_trailer(parser: &mut Parser<'_>) -> Result<bool, CompactError> {
    let mut good = parser.type_syntax(0)?;
    parser.trivia()?;
    if parser.at(TokenKind::BareIdentifier) && parser.current_text() == "to" {
        parser.bump()?;
    } else {
        parser.diagnostic();
        good = false;
    }
    parser.trivia()?;
    good &= parser.type_syntax(0)?;
    Ok(good)
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
