use super::*;

/// Parses quoted generic operations into a lossless, syntactic CST.
pub fn parse_generic_operations(lexed: &Lexed) -> Result<ParsedSyntax, CompactError> {
    parse_generic_operations_with_limits(lexed, ParserLimits::default())
}

/// Builds a lossless generic-operation CST with explicit parser limits.
///
/// Recoverable grammar problems appear in [`ParsedSyntax::diagnostics`].
/// [`CompactError`] is reserved for event or compact-tree invariants.
pub fn parse_generic_operations_with_limits(
    lexed: &Lexed,
    limits: ParserLimits,
) -> Result<ParsedSyntax, CompactError> {
    parse_operations_with_registry(lexed, &[], &DialectRegistry::EMPTY, limits)
}

/// Builds a lossless CST with registered custom operation syntax.
///
/// `source` must be the byte buffer used to create `lexed`; registered parsers
/// inspect it through token ranges. Recoverable syntax problems appear in the
/// returned diagnostics.
///
/// # Errors
///
/// Returns [`CompactError`] when the parser cannot maintain its event or
/// compact-tree invariants.
pub fn parse_operations_with_registry(
    lexed: &Lexed,
    source: &[u8],
    registry: &DialectRegistry,
    limits: ParserLimits,
) -> Result<ParsedSyntax, CompactError> {
    let (builder, diagnostics) = produce_operation_events(lexed, source, registry, limits)?;
    Ok(ParsedSyntax {
        tree: std::sync::Arc::new(builder.finish(lexed.tokens().to_vec())?),
        diagnostics,
    })
}

fn produce_operation_events(
    lexed: &Lexed,
    source: &[u8],
    registry: &DialectRegistry,
    limits: ParserLimits,
) -> Result<(EventBuilder, Vec<ParseDiagnostic>), CompactError> {
    let mut parser = Parser {
        tokens: lexed.tokens(),
        position: 0,
        builder: EventBuilder::new(),
        diagnostics: Vec::new(),
        limits,
        nesting_depth: 0,
        source,
        registry,
    };
    let root = parser.builder.start();
    while !parser.at(TokenKind::Eof) {
        let before = parser.position;
        parser.trivia()?;
        if parser.at(TokenKind::FileMetadataBegin) {
            parser.file_metadata()?;
        } else if matches!(
            parser.current(),
            TokenKind::HashIdentifier | TokenKind::ExclamationIdentifier
        ) && parser.nth_nontrivia(1) == Some(TokenKind::Equal)
        {
            parser.alias_definition()?;
        } else if matches!(
            parser.current(),
            TokenKind::String | TokenKind::PercentIdentifier | TokenKind::BareIdentifier
        ) {
            parser.operation()?;
        } else if !parser.at(TokenKind::Eof) {
            parser.error_token()?;
        }
        parser.ensure_progress(before)?;
    }
    parser.bump()?;
    parser.builder.complete(root, SyntaxKind::File)?;
    Ok((parser.builder, parser.diagnostics))
}

#[cfg(test)]
#[path = "parser_construction_benchmark.rs"]
mod parser_construction_benchmark;

pub(super) struct Parser<'a> {
    pub(super) tokens: &'a [crate::lexer::Token],
    pub(super) position: usize,
    pub(super) builder: EventBuilder,
    pub(super) diagnostics: Vec<ParseDiagnostic>,
    pub(super) limits: ParserLimits,
    pub(super) nesting_depth: usize,
    pub(super) source: &'a [u8],
    pub(super) registry: &'a DialectRegistry,
}

const MAX_TYPE_DEPTH: usize = 64;

impl Parser<'_> {
    fn file_metadata(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        while !self.at(TokenKind::FileMetadataEnd) && !self.at(TokenKind::Eof) {
            self.bump()?;
        }
        let unterminated = self.at(TokenKind::Eof);
        if unterminated {
            self.diagnostic();
        } else {
            self.bump()?;
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::FileMetadata, unterminated)?;
        Ok(())
    }

    fn alias_definition(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        let is_type = self.at(TokenKind::ExclamationIdentifier);
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Equal)?;
        self.trivia()?;
        if is_type {
            // MLIR permits the documentary `type` keyword in type aliases.
            if self.at(TokenKind::BareIdentifier) {
                self.bump()?;
                self.trivia()?;
            }
            if self.at(TokenKind::LParen) {
                self.function_type()?;
            } else {
                good &= self.type_syntax(0)?;
            }
        } else {
            good &= self.attribute_value()?;
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::AliasDefinition, !good)?;
        Ok(())
    }

    fn operation(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        if self.at(TokenKind::PercentIdentifier) {
            let result_list = self.builder.start();
            loop {
                let result = self.builder.start();
                self.bump()?;
                self.trivia()?;
                if self.at(TokenKind::Colon) {
                    let number = self.builder.start();
                    self.bump()?;
                    self.trivia()?;
                    let good = self.expect(TokenKind::Integer)?;
                    self.builder
                        .complete_with_error(number, SyntaxKind::ResultNumber, !good)?;
                    self.trivia()?;
                }
                self.builder.complete(result, SyntaxKind::ResultGroup)?;
                if !self.at(TokenKind::Comma) {
                    break;
                }
                self.bump()?;
                self.trivia()?;
                if !self.at(TokenKind::PercentIdentifier) {
                    self.diagnostic();
                    break;
                }
            }
            self.expect(TokenKind::Equal)?;
            self.builder.complete(result_list, SyntaxKind::Result)?;
            self.trivia()?;
        }
        if self.at(TokenKind::BareIdentifier) {
            let range = self.tokens[self.position].range();
            let name = std::str::from_utf8(
                self.source
                    .get(range.start() as usize..range.end() as usize)
                    .unwrap_or_default(),
            )
            .unwrap_or("");
            if let Some(descriptor) = self.registry.custom_operation(name) {
                if descriptor.assembly.is_some() {
                    return DialectParser {
                        parser: self,
                        marker,
                        descriptor,
                    }
                    .parse_assembly_program();
                }
                if let Some(parse) = descriptor.parse {
                    return parse(&mut DialectParser {
                        parser: self,
                        marker,
                        descriptor,
                    });
                }
            }
            if let Some(shape) = self.registry.operation_shape(name) {
                return shaped_operation(self, marker, shape);
            }
            return self.unparsed_custom_operation(Some(marker));
        }
        self.expect(TokenKind::String)?;
        self.trivia()?;
        self.operand_list()?;
        self.trivia()?;
        if self.at(TokenKind::LBracket) {
            self.successor_list()?;
            self.trivia()?;
        }
        if self.at(TokenKind::Less) {
            self.property_dict()?;
            self.trivia()?;
        }
        if self.at(TokenKind::LParen) && self.nth_nontrivia(1) == Some(TokenKind::LBrace) {
            self.region_list()?;
            self.trivia()?;
        }
        if self.at(TokenKind::LBrace) {
            self.attribute_dict()?;
            self.trivia()?;
        }
        self.expect(TokenKind::Colon)?;
        self.trivia()?;
        self.function_type()?;
        self.trivia()?;
        if self.at(TokenKind::Loc) {
            let location = self.builder.start();
            let good = self.location_attribute()?;
            self.builder
                .complete_with_error(location, SyntaxKind::TrailingLocation, !good)?;
        }
        self.builder.complete(marker, SyntaxKind::Operation)?;
        Ok(())
    }

    pub(super) fn operand_list(&mut self) -> Result<(), CompactError> {
        self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while self.at(TokenKind::PercentIdentifier) {
            let operand = self.builder.start();
            let operand_use = self.builder.start();
            self.bump()?;
            if self.at(TokenKind::HashIdentifier) {
                self.bump()?;
            }
            self.builder.complete(operand_use, SyntaxKind::OperandUse)?;
            self.builder.complete(operand, SyntaxKind::Operand)?;
            self.trivia()?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        self.expect(TokenKind::RParen)?;
        Ok(())
    }

    fn successor_list(&mut self) -> Result<(), CompactError> {
        let list = self.builder.start();
        let mut good = self.expect(TokenKind::LBracket)?;
        self.trivia()?;
        while !self.at(TokenKind::RBracket)
            && !self.at(TokenKind::RBrace)
            && !self.at(TokenKind::Eof)
        {
            let successor = self.builder.start();
            let mut item_good = self.expect(TokenKind::CaretIdentifier)?;
            self.trivia()?;
            if self.at(TokenKind::Colon) {
                self.bump()?;
                self.trivia()?;
                item_good &= self.block_argument_list(SyntaxKind::SuccessorArguments)?;
            }
            self.builder
                .complete_with_error(successor, SyntaxKind::Successor, !item_good)?;
            good &= item_good;
            self.trivia()?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        good &= self.expect(TokenKind::RBracket)?;
        self.builder
            .complete_with_error(list, SyntaxKind::SuccessorList, !good)?;
        Ok(())
    }

    fn property_dict(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        good &= self.dictionary_entries()?;
        self.trivia()?;
        good &= self.expect(TokenKind::Greater)?;
        self.builder
            .complete_with_error(marker, SyntaxKind::PropertyDict, !good)?;
        Ok(())
    }

    fn region_list(&mut self) -> Result<(), CompactError> {
        self.expect(TokenKind::LParen)?;
        self.trivia()?;
        loop {
            self.region()?;
            self.trivia()?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        self.expect(TokenKind::RParen)?;
        Ok(())
    }

    pub(super) fn region(&mut self) -> Result<(), CompactError> {
        let region = self.builder.start();
        if self.nesting_depth >= self.limits.max_delimiter_depth {
            self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
            self.recover_balanced_region()?;
            self.builder
                .complete_with_error(region, SyntaxKind::Region, true)?;
            return Ok(());
        }
        self.nesting_depth += 1;
        self.expect(TokenKind::LBrace)?;
        self.trivia()?;
        let mut block = (!self.at(TokenKind::CaretIdentifier)).then(|| self.builder.start());
        loop {
            let before = self.position;
            self.trivia()?;
            if self.at(TokenKind::RBrace) || self.at(TokenKind::Eof) {
                break;
            }
            if self.at(TokenKind::CaretIdentifier) {
                if let Some(open) = block.take() {
                    self.builder.complete(open, SyntaxKind::Block)?;
                }
                let labeled = self.builder.start();
                let label = self.builder.start();
                self.bump()?;
                self.trivia()?;
                if self.at(TokenKind::LParen) {
                    self.block_argument_list(SyntaxKind::BlockArgumentList)?;
                    self.trivia()?;
                }
                self.expect(TokenKind::Colon)?;
                self.builder.complete(label, SyntaxKind::BlockLabel)?;
                block = Some(labeled);
                continue;
            }
            if self.at(TokenKind::String)
                || self.at(TokenKind::PercentIdentifier)
                || self.at(TokenKind::BareIdentifier)
            {
                self.operation()?;
            } else {
                self.error_token()?;
            }
            self.ensure_progress(before)?;
        }
        if let Some(open) = block {
            self.builder.complete(open, SyntaxKind::Block)?;
        }
        self.expect(TokenKind::RBrace)?;
        self.nesting_depth -= 1;
        self.builder.complete(region, SyntaxKind::Region)?;
        Ok(())
    }

    pub(super) fn block_argument_list(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let list = self.builder.start();
        let mut good = self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while self.at(TokenKind::PercentIdentifier) {
            let argument = self.builder.start();
            self.bump()?;
            self.trivia()?;
            good &= self.expect(TokenKind::Colon)?;
            self.trivia()?;
            good &= self.type_syntax(0)?;
            self.trivia()?;
            if self.at(TokenKind::LBrace) {
                self.attribute_dict()?;
                self.trivia()?;
            }
            if self.at(TokenKind::Loc) {
                good &= self.location_attribute()?;
                self.trivia()?;
            }
            self.builder
                .complete_with_error(argument, SyntaxKind::BlockArgument, !good)?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        good &= self.expect(TokenKind::RParen)?;
        self.builder.complete_with_error(list, kind, !good)?;
        Ok(good)
    }

    pub(super) fn attribute_dict(&mut self) -> Result<(), CompactError> {
        let dict = self.builder.start();
        if !self.enter_attribute_container(TokenKind::LBrace, TokenKind::RBrace)? {
            self.builder
                .complete_with_error(dict, SyntaxKind::AttributeDict, true)?;
            return Ok(());
        }
        let bad = !self.dictionary_entries()?;
        self.nesting_depth -= 1;
        self.builder
            .complete_with_error(dict, SyntaxKind::AttributeDict, bad)?;
        Ok(())
    }

    fn dictionary_entries(&mut self) -> Result<bool, CompactError> {
        let mut good = self.expect(TokenKind::LBrace)?;
        self.trivia()?;
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.position;
            let attr = self.builder.start();
            let mut bad = false;
            if self.at_identifier() || self.at(TokenKind::String) {
                let key_range = self.tokens[self.position].range();
                let key = std::str::from_utf8(
                    self.source
                        .get(key_range.start() as usize..key_range.end() as usize)
                        .unwrap_or_default(),
                )
                .unwrap_or("");
                self.bump()?;
                self.trivia()?;
                if key == "no_inline" && (self.at(TokenKind::RBrace) || self.at(TokenKind::Comma)) {
                    // The registered func.func schema admits MLIR's unit
                    // attribute spelling: {no_inline}.
                } else {
                    bad |= !self.expect(TokenKind::Equal)?;
                    self.trivia()?;
                    if self.at(TokenKind::RBrace) {
                        self.diagnostic();
                        bad = true;
                    } else {
                        bad |= !self.attribute_value()?;
                    }
                }
            } else {
                bad = true;
                self.error_token()?;
            }
            self.builder
                .complete_with_error(attr, SyntaxKind::Attribute, bad)?;
            good &= !bad;
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else if !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                good = false;
                self.diagnostic();
            }
            self.ensure_progress(before)?;
        }
        good &= self.expect(TokenKind::RBrace)?;
        Ok(good)
    }

    pub(super) fn function_type(&mut self) -> Result<(), CompactError> {
        let ty = self.builder.start();
        self.type_list(0)?;
        self.trivia()?;
        self.expect(TokenKind::Arrow)?;
        self.trivia()?;
        if self.at_type_start() {
            self.type_syntax(0)?;
        } else {
            self.type_list(0)?;
        }
        self.builder.complete(ty, SyntaxKind::FunctionType)?;
        Ok(())
    }

    pub(super) fn type_list(&mut self, depth: usize) -> Result<(), CompactError> {
        self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while self.at_type_start() {
            self.type_syntax(depth)?;
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
        Ok(())
    }

    pub(super) fn at_type_start(&self) -> bool {
        matches!(
            self.current(),
            TokenKind::IntType
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
    }

    pub(super) fn type_syntax(&mut self, depth: usize) -> Result<bool, CompactError> {
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
        if bad {
            if let Some(result) = result {
                let marker = self.builder.precede(result)?;
                return Ok(Some(self.builder.complete_with_error(
                    marker,
                    SyntaxKind::AffineExpression,
                    true,
                )?));
            }
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

    fn attribute_value(&mut self) -> Result<bool, CompactError> {
        match self.current() {
            TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Integer
            | TokenKind::WideInteger
            | TokenKind::Float => self.numeric_attribute(),
            TokenKind::String => self.string_attribute(),
            TokenKind::IntType
            | TokenKind::FloatType
            | TokenKind::IndexType
            | TokenKind::ExclamationIdentifier
            | TokenKind::Tuple
            | TokenKind::Tensor
            | TokenKind::Vector
            | TokenKind::MemRef
            | TokenKind::AffineMap
            | TokenKind::AffineSet => {
                let marker = self.builder.start();
                let good = self.type_syntax(0)?;
                self.builder
                    .complete_with_error(marker, SyntaxKind::TypeAttribute, !good)?;
                Ok(good)
            }
            TokenKind::Dense => self.payload_attribute(SyntaxKind::DenseElementsAttribute),
            TokenKind::Sparse => self.payload_attribute(SyntaxKind::SparseElementsAttribute),
            TokenKind::DenseResource => {
                self.payload_attribute(SyntaxKind::DenseResourceElementsAttribute)
            }
            TokenKind::HashIdentifier if self.nth_nontrivia(1) == Some(TokenKind::Less) => {
                self.opaque(SyntaxKind::OpaqueAttribute)
            }
            TokenKind::HashIdentifier => self.leaf(SyntaxKind::AttributeAlias),
            TokenKind::AtIdentifier => self.symbol_reference(),
            TokenKind::BareIdentifier if self.current_text() == "type" => {
                let marker = self.builder.start();
                self.bump()?;
                self.trivia()?;
                let mut good = self.expect(TokenKind::Less)?;
                self.trivia()?;
                if self.at(TokenKind::LParen) {
                    self.function_type()?;
                } else {
                    good &= self.type_syntax(0)?;
                }
                self.trivia()?;
                good &= self.expect(TokenKind::Greater)?;
                self.builder
                    .complete_with_error(marker, SyntaxKind::TypeAttribute, !good)?;
                Ok(good)
            }
            TokenKind::BareIdentifier if self.current_text() == "unit" => {
                let marker = self.builder.start();
                self.bump()?;
                self.builder.complete(marker, SyntaxKind::AttributeAlias)?;
                Ok(true)
            }
            TokenKind::BareIdentifier if matches!(self.current_text(), "true" | "false") => {
                self.leaf(SyntaxKind::BooleanAttribute)
            }
            TokenKind::LBracket => self.array_attribute(),
            TokenKind::LBrace => self.dictionary_attribute(),
            TokenKind::Loc => self.location_attribute(),
            _ => {
                self.error_token()?;
                Ok(false)
            }
        }
    }

    pub(super) fn constant_value(&mut self) -> Result<bool, CompactError> {
        if self.at(TokenKind::String) {
            let marker = self.builder.start();
            self.bump()?;
            self.builder.complete(marker, SyntaxKind::StringAttribute)?;
            return Ok(true);
        }
        if matches!(
            self.current(),
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Integer
                | TokenKind::WideInteger
                | TokenKind::Float
        ) {
            let marker = self.builder.start();
            if matches!(self.current(), TokenKind::Plus | TokenKind::Minus) {
                self.bump()?;
            }
            let kind = match self.current() {
                TokenKind::Integer => SyntaxKind::IntegerAttribute,
                TokenKind::WideInteger => SyntaxKind::WideNumber,
                TokenKind::Float => SyntaxKind::FloatAttribute,
                _ => {
                    self.diagnostic();
                    self.builder
                        .complete_with_error(marker, SyntaxKind::IntegerAttribute, true)?;
                    return Ok(false);
                }
            };
            self.bump()?;
            self.builder.complete(marker, kind)?;
            Ok(true)
        } else {
            self.attribute_value()
        }
    }

    fn leaf(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.builder.complete(marker, kind)?;
        Ok(true)
    }

    fn numeric_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        if matches!(self.current(), TokenKind::Plus | TokenKind::Minus) {
            self.bump()?;
        }
        let mut kind = match self.current() {
            TokenKind::Integer => SyntaxKind::IntegerAttribute,
            TokenKind::WideInteger => SyntaxKind::WideNumber,
            TokenKind::Float => SyntaxKind::FloatAttribute,
            _ => {
                self.diagnostic();
                self.builder
                    .complete_with_error(marker, SyntaxKind::IntegerAttribute, true)?;
                return Ok(false);
            }
        };
        let numeric_range = self.tokens[self.position].range();
        let oversized = self.current_token_bytes() > self.limits.max_numeric_literal_bytes;
        self.bump()?;
        self.trivia()?;
        let mut good = true;
        if self.at(TokenKind::Colon) {
            self.bump()?;
            self.trivia()?;
            good = self.type_syntax(0)?;
            if self
                .tokens
                .get(self.position - 1)
                .is_some_and(|token| token.kind() == TokenKind::FloatType)
            {
                kind = SyntaxKind::FloatAttribute;
            }
            if !good {
                self.diagnostic();
            }
        }
        if oversized {
            self.diagnostics.push(ParseDiagnostic {
                range: numeric_range,
                kind: ParseDiagnosticKind::Syntax,
            });
            self.builder
                .complete_with_error(marker, SyntaxKind::WideNumber, true)?;
            return Ok(false);
        }
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn payload_attribute(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        let mut good = self.expect(TokenKind::Less)?;
        let payload_start = self.tokens[self.position.saturating_sub(1)].range().end();
        good &= self.scan_balanced_payload(payload_start)?;
        self.trivia()?;
        good &= self.expect(TokenKind::Colon)?;
        self.trivia()?;
        good &= self.type_syntax(0)?;
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn scan_balanced_payload(&mut self, payload_start: u32) -> Result<bool, CompactError> {
        let mut stack = vec![TokenKind::Greater];
        let mut good = true;
        while let Some(expected) = stack.last().copied() {
            if self.at(TokenKind::Eof) || (stack.len() == 1 && self.at(TokenKind::RBrace)) {
                self.diagnostic();
                return Ok(false);
            }
            let current = self.current();
            if current == expected {
                stack.pop();
                self.bump()?;
                continue;
            }
            if is_close(current) {
                self.diagnostic();
                good = false;
                if current == TokenKind::RBrace {
                    return Ok(false);
                }
                self.bump()?;
                continue;
            }
            if let Some(close) = close_for(current) {
                if stack.len() >= self.limits.max_delimiter_depth {
                    self.diagnostic();
                    good = false;
                } else {
                    stack.push(close);
                }
            }
            if matches!(
                current,
                TokenKind::Integer | TokenKind::WideInteger | TokenKind::Float
            ) && self.current_token_bytes() > self.limits.max_numeric_literal_bytes
            {
                self.diagnostic();
                good = false;
            }
            if self.tokens[self.position]
                .range()
                .end()
                .saturating_sub(payload_start) as usize
                > self.limits.max_payload_bytes
            {
                self.diagnostic();
                good = false;
            }
            self.bump()?;
        }
        Ok(good)
    }

    fn string_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = true;
        if self.at(TokenKind::Colon) {
            self.bump()?;
            self.trivia()?;
            good = self.type_syntax(0)?;
            if !good {
                self.diagnostic();
            }
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::StringAttribute, !good)?;
        Ok(good)
    }

    fn symbol_reference(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        let mut good = true;
        loop {
            self.trivia()?;
            if !self.at(TokenKind::Colon) {
                break;
            }
            self.bump()?;
            good &= self.expect(TokenKind::Colon)?;
            good &= self.expect(TokenKind::AtIdentifier)?;
            if !good {
                break;
            }
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::SymbolReference, !good)?;
        Ok(good)
    }

    fn array_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        if !self.enter_attribute_container(TokenKind::LBracket, TokenKind::RBracket)? {
            self.builder
                .complete_with_error(marker, SyntaxKind::ArrayAttribute, true)?;
            return Ok(false);
        }
        let mut good = self.expect(TokenKind::LBracket)?;
        self.trivia()?;
        while !self.at(TokenKind::RBracket)
            && !self.at(TokenKind::RBrace)
            && !self.at(TokenKind::Eof)
        {
            let before = self.position;
            good &= self.attribute_value()?;
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else if !self.at(TokenKind::RBracket) {
                good = false;
                self.diagnostic();
            }
            debug_assert!(self.position > before);
        }
        good &= self.expect(TokenKind::RBracket)?;
        self.nesting_depth -= 1;
        self.builder
            .complete_with_error(marker, SyntaxKind::ArrayAttribute, !good)?;
        Ok(good)
    }

    fn dictionary_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        if !self.enter_attribute_container(TokenKind::LBrace, TokenKind::RBrace)? {
            self.builder
                .complete_with_error(marker, SyntaxKind::DictionaryAttribute, true)?;
            return Ok(false);
        }
        let good = self.dictionary_entries()?;
        self.nesting_depth -= 1;
        self.builder
            .complete_with_error(marker, SyntaxKind::DictionaryAttribute, !good)?;
        Ok(good)
    }

    fn location_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::LParen)?;
        self.trivia()?;
        let detail = self.builder.start();
        let detail_kind = match self.current() {
            TokenKind::Unknown => SyntaxKind::UnknownLocation,
            TokenKind::CallSite => SyntaxKind::CallSiteLocation,
            TokenKind::Fused => SyntaxKind::FusedLocation,
            TokenKind::String if self.nth_nontrivia(1) == Some(TokenKind::Colon) => {
                SyntaxKind::FileLineColLocation
            }
            _ => SyntaxKind::NameLocation,
        };
        let mut stack = vec![TokenKind::RParen];
        while !self.at(TokenKind::Eof) {
            let kind = self.current();
            let following_operation = stack.len() == 1
                && kind == TokenKind::String
                && self.nth_nontrivia(1) == Some(TokenKind::LParen)
                && !matches!(
                    self.nth_nontrivia(2),
                    Some(
                        TokenKind::HashIdentifier
                            | TokenKind::ExclamationIdentifier
                            | TokenKind::String
                            | TokenKind::Loc
                            | TokenKind::Unknown
                            | TokenKind::CallSite
                            | TokenKind::Fused
                    )
                );
            let at_enclosing_boundary = !stack.is_empty()
                && (kind == TokenKind::RBrace
                    || (kind == TokenKind::Comma && stack.last() == Some(&TokenKind::RParen)));
            if (stack.len() == 1 && kind == TokenKind::RParen)
                || at_enclosing_boundary
                || following_operation
            {
                break;
            }
            if let Some(close) = close_for(kind) {
                if stack.len() >= self.limits.max_delimiter_depth {
                    self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
                    good = false;
                } else {
                    stack.push(close);
                }
            } else if stack.last() == Some(&kind) {
                stack.pop();
            }
            self.bump()?;
        }
        if self.at(TokenKind::Eof)
            || self.at(TokenKind::RBrace)
            || (self.at(TokenKind::String)
                && self.nth_nontrivia(1) == Some(TokenKind::LParen)
                && !matches!(
                    self.nth_nontrivia(2),
                    Some(
                        TokenKind::HashIdentifier
                            | TokenKind::ExclamationIdentifier
                            | TokenKind::String
                            | TokenKind::Loc
                            | TokenKind::Unknown
                            | TokenKind::CallSite
                            | TokenKind::Fused
                    )
                ))
            || (self.at(TokenKind::Comma) && stack.last() == Some(&TokenKind::RParen))
        {
            good = false;
            self.diagnostic();
        }
        self.builder
            .complete_with_error(detail, detail_kind, !good)?;
        good &= self.expect(TokenKind::RParen)?;
        self.builder
            .complete_with_error(marker, SyntaxKind::LocationAttribute, !good)?;
        Ok(good)
    }

    fn opaque(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        let mut bad = false;
        if self.at(TokenKind::Less) {
            let body = self.builder.start();
            self.bump()?;
            let payload_start = self.tokens[self.position - 1].range().end();
            bad = !self.scan_balanced_payload(payload_start)?;
            self.builder.complete_with_error(
                body,
                if kind == SyntaxKind::OpaqueType {
                    SyntaxKind::OpaqueTypeBody
                } else {
                    SyntaxKind::OpaqueAttributeBody
                },
                bad,
            )?;
        }
        self.builder.complete_with_error(marker, kind, bad)?;
        Ok(!bad)
    }

    pub(super) fn trivia(&mut self) -> Result<(), CompactError> {
        while matches!(
            self.current(),
            TokenKind::Whitespace | TokenKind::LineComment
        ) {
            self.bump()?;
        }
        Ok(())
    }
    pub(super) fn expect(&mut self, kind: TokenKind) -> Result<bool, CompactError> {
        if self.at(kind) {
            self.bump()?;
            Ok(true)
        } else {
            self.diagnostic();
            Ok(false)
        }
    }
    fn error_token(&mut self) -> Result<(), CompactError> {
        let error = self.builder.start();
        self.diagnostic();
        if !self.at(TokenKind::Eof) {
            self.bump()?;
        }
        self.builder
            .complete_with_error(error, SyntaxKind::Error, true)?;
        Ok(())
    }
    pub(super) fn diagnostic(&mut self) {
        self.diagnostic_kind(ParseDiagnosticKind::Syntax);
    }
    fn diagnostic_kind(&mut self, kind: ParseDiagnosticKind) {
        self.diagnostics.push(ParseDiagnostic {
            range: self.tokens[self.position].range(),
            kind,
        });
    }
    fn ensure_progress(&mut self, before: usize) -> Result<(), CompactError> {
        if self.position == before && !self.at(TokenKind::Eof) {
            self.diagnostic_kind(ParseDiagnosticKind::ProgressLimit);
            self.error_token()?;
        }
        Ok(())
    }
    fn unparsed_custom_operation(&mut self, marker: Option<Marker>) -> Result<(), CompactError> {
        let marker = marker.unwrap_or_else(|| self.builder.start());
        self.diagnostic_kind(ParseDiagnosticKind::UnknownCustomOperation);
        let start = self.position;
        let mut stack = Vec::new();
        let mut line_boundary = false;
        let mut completed_payload = false;
        while !self.at(TokenKind::Eof) {
            let current = self.current();
            if stack.is_empty() && current == TokenKind::LBrace && self.region_shaped_body() {
                self.region()?;
                completed_payload = true;
                continue;
            }
            if stack.is_empty()
                && (current == TokenKind::RBrace
                    || current == TokenKind::CaretIdentifier
                    || self.is_generic_operation_start()
                    || (self.position > start
                        && ((current == TokenKind::BareIdentifier
                            && self.bare_custom_operation_start()
                            && (line_boundary || self.current_text().contains('.')))
                            || (current == TokenKind::PercentIdentifier
                                && self.result_custom_operation_start()))
                        && (line_boundary || completed_payload))
                    || (self.position > start
                        && current == TokenKind::BareIdentifier
                        && self.previous_nontrivia() == Some(TokenKind::PercentIdentifier)
                        && self.nth_nontrivia(1) == Some(TokenKind::PercentIdentifier)))
            {
                break;
            }
            if stack.is_empty() && completed_payload && !is_trivia(current) {
                completed_payload = false;
            }
            if stack.last() == Some(&current) {
                stack.pop();
                if stack.is_empty() {
                    completed_payload = true;
                }
            } else if let Some(close) = close_for(current) {
                if stack.len() >= self.limits.max_delimiter_depth {
                    self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
                } else {
                    stack.push(close);
                }
            } else if is_close(current) && stack.is_empty() {
                break;
            }
            let token_has_line_break = current == TokenKind::Whitespace
                && self
                    .source
                    .get(
                        self.tokens[self.position].range().start() as usize
                            ..self.tokens[self.position].range().end() as usize,
                    )
                    .is_some_and(|bytes| bytes.contains(&b'\n'));
            self.bump()?;
            if token_has_line_break {
                line_boundary = true;
            } else if !is_trivia(current) {
                line_boundary = false;
            }
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::UnparsedCustomOperation, true)?;
        Ok(())
    }
    fn result_custom_operation_start(&self) -> bool {
        self.result_assignment_starts_operation(true)
    }
    fn bare_custom_operation_start(&self) -> bool {
        let range = self.tokens[self.position].range();
        self.source
            .get(range.start() as usize..range.end() as usize)
            != Some(b"attributes")
    }
    fn region_shaped_body(&self) -> bool {
        let mut index = self.position + 1;
        while self
            .tokens
            .get(index)
            .is_some_and(|token| is_trivia(token.kind()))
        {
            index += 1;
        }
        match self.tokens.get(index).map(|token| token.kind()) {
            Some(TokenKind::CaretIdentifier | TokenKind::PercentIdentifier) => true,
            Some(TokenKind::String) => {
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::LParen)
            }
            Some(TokenKind::RBrace) => false,
            Some(TokenKind::BareIdentifier) => {
                let mnemonic = self.tokens[index].range();
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                self.tokens.get(index).map(|token| token.kind()) != Some(TokenKind::Equal)
                    && self
                        .source
                        .get(mnemonic.start() as usize..mnemonic.end() as usize)
                        .is_some_and(|text| text.contains(&b'.'))
            }
            _ => false,
        }
    }
    fn is_generic_operation_start(&self) -> bool {
        if self.at(TokenKind::String) {
            return self.nth_nontrivia(1) == Some(TokenKind::LParen);
        }
        self.result_assignment_starts_operation(false)
    }
    fn result_assignment_starts_operation(&self, allow_bare: bool) -> bool {
        if !self.at(TokenKind::PercentIdentifier) {
            return false;
        }
        let mut index = self.position;
        loop {
            if self.tokens.get(index).map(|token| token.kind())
                != Some(TokenKind::PercentIdentifier)
            {
                return false;
            }
            index += 1;
            while self
                .tokens
                .get(index)
                .is_some_and(|token| is_trivia(token.kind()))
            {
                index += 1;
            }
            if self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::Colon) {
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                if self.tokens.get(index).map(|token| token.kind()) != Some(TokenKind::Integer) {
                    return false;
                }
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
            }
            if self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::Comma) {
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                continue;
            }
            break;
        }
        while self
            .tokens
            .get(index)
            .is_some_and(|token| is_trivia(token.kind()))
        {
            index += 1;
        }
        if self.tokens.get(index).map(|token| token.kind()) != Some(TokenKind::Equal) {
            return false;
        }
        index += 1;
        while self
            .tokens
            .get(index)
            .is_some_and(|token| is_trivia(token.kind()))
        {
            index += 1;
        }
        matches!(
            self.tokens.get(index).map(|token| token.kind()),
            Some(TokenKind::String)
        ) || (allow_bare
            && self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::BareIdentifier))
    }
    fn previous_nontrivia(&self) -> Option<TokenKind> {
        self.tokens[..self.position]
            .iter()
            .rev()
            .find(|token| !is_trivia(token.kind()))
            .map(|token| token.kind())
    }
    fn recover_balanced_region(&mut self) -> Result<(), CompactError> {
        let mut depth = 0usize;
        while !self.at(TokenKind::Eof) {
            match self.current() {
                TokenKind::LBrace => depth = depth.saturating_add(1),
                TokenKind::RBrace if depth <= 1 => {
                    self.bump()?;
                    return Ok(());
                }
                TokenKind::RBrace => depth -= 1,
                _ => {}
            }
            self.bump()?;
        }
        Ok(())
    }
    fn enter_attribute_container(
        &mut self,
        open: TokenKind,
        close: TokenKind,
    ) -> Result<bool, CompactError> {
        if self.nesting_depth < self.limits.max_delimiter_depth {
            self.nesting_depth += 1;
            return Ok(true);
        }
        self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
        let mut depth = 0usize;
        while !self.at(TokenKind::Eof) {
            let current = self.current();
            if current == open {
                depth += 1;
            } else if current == close {
                self.bump()?;
                if depth <= 1 {
                    break;
                }
                depth -= 1;
                continue;
            }
            self.bump()?;
        }
        Ok(false)
    }
    fn current_token_bytes(&self) -> usize {
        self.tokens[self.position].range().len() as usize
    }
    pub(super) fn bump(&mut self) -> Result<(), CompactError> {
        self.builder.token(self.position)?;
        self.position += 1;
        Ok(())
    }
    fn current(&self) -> TokenKind {
        self.tokens[self.position].kind()
    }
    pub(super) fn current_text(&self) -> &str {
        let range = self.tokens[self.position].range();
        std::str::from_utf8(
            self.source
                .get(range.start() as usize..range.end() as usize)
                .unwrap_or_default(),
        )
        .unwrap_or("")
    }
    pub(super) fn at(&self, kind: TokenKind) -> bool {
        self.current() == kind
    }
    fn at_identifier(&self) -> bool {
        matches!(
            self.current(),
            TokenKind::BareIdentifier
                | TokenKind::IntType
                | TokenKind::FloatType
                | TokenKind::IndexType
                | TokenKind::Tuple
                | TokenKind::Tensor
                | TokenKind::Vector
                | TokenKind::MemRef
                | TokenKind::AffineMap
                | TokenKind::AffineSet
                | TokenKind::Mod
                | TokenKind::FloorDiv
                | TokenKind::CeilDiv
                | TokenKind::Strided
                | TokenKind::Loc
                | TokenKind::Unknown
                | TokenKind::CallSite
                | TokenKind::Fused
        )
    }
    fn nth_nontrivia(&self, n: usize) -> Option<TokenKind> {
        self.tokens[self.position..]
            .iter()
            .filter(|token| !matches!(token.kind(), TokenKind::Whitespace | TokenKind::LineComment))
            .nth(n)
            .map(|token| token.kind())
    }
}

pub(super) fn close_for(kind: TokenKind) -> Option<TokenKind> {
    match kind {
        TokenKind::Less => Some(TokenKind::Greater),
        TokenKind::LParen => Some(TokenKind::RParen),
        TokenKind::LBrace => Some(TokenKind::RBrace),
        TokenKind::LBracket => Some(TokenKind::RBracket),
        _ => None,
    }
}

fn is_close(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Greater | TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket
    )
}

/// Test support for constructing a balanced-brace fixture tree.
#[doc(hidden)]
pub fn parse_brace_fixture(lexed: &Lexed) -> Result<SyntaxTree, CompactError> {
    let mut builder = EventBuilder::new();
    let root = builder.start();
    let mut braces = Vec::<Marker>::new();
    for (i, token) in lexed.tokens().iter().enumerate() {
        match token.kind() {
            TokenKind::LBrace => {
                let marker = builder.start();
                builder.token(i)?;
                braces.push(marker)
            }
            TokenKind::RBrace => {
                if let Some(marker) = braces.pop() {
                    builder.token(i)?;
                    builder.complete(marker, SyntaxKind::Region)?;
                } else {
                    let error = builder.start();
                    builder.token(i)?;
                    builder.complete_with_error(error, SyntaxKind::Error, true)?;
                }
            }
            _ => builder.token(i)?,
        }
    }
    while let Some(marker) = braces.pop() {
        builder.complete_with_error(marker, SyntaxKind::Region, true)?;
    }
    builder.complete(root, SyntaxKind::File)?;
    builder.finish(lexed.tokens().to_vec())
}
