mod attributes;
mod recovery;
mod types;

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

pub(crate) fn parse_owned_operations_with_registry(
    lexed: Lexed,
    source: &[u8],
    registry: &DialectRegistry,
    limits: ParserLimits,
) -> Result<(ParsedSyntax, Vec<LexDiagnostic>), CompactError> {
    let (builder, diagnostics) = produce_operation_events(&lexed, source, registry, limits)?;
    let (tokens, lexer_diagnostics) = lexed.into_parts();
    Ok((
        ParsedSyntax {
            tree: std::sync::Arc::new(builder.finish_parser(tokens)?),
            diagnostics,
        },
        lexer_diagnostics,
    ))
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
        furthest_attempt: 0,
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
    pub(super) furthest_attempt: usize,
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
            if let Some(alternatives) = self.registry.operation_grammars(name) {
                let name = name.to_owned();
                return alternative_operation(self, marker, alternatives, &name);
            }
            if let Some(shape) = self.registry.operation_shape(name) {
                let name = name.to_owned();
                return shaped_operation(self, marker, shape, &name);
            }
            if let Some(format) = self.registry.operation_format(name) {
                return formatted_operation(self, marker, format);
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
    fn current_token_bytes(&self) -> usize {
        self.tokens[self.position].range().len() as usize
    }
    pub(super) fn bump(&mut self) -> Result<(), CompactError> {
        self.builder.token(self.position)?;
        self.position += 1;
        Ok(())
    }
    pub(super) fn current(&self) -> TokenKind {
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
    pub(super) fn nth_nontrivia(&self, n: usize) -> Option<TokenKind> {
        self.tokens[self.position..]
            .iter()
            .filter(|token| !matches!(token.kind(), TokenKind::Whitespace | TokenKind::LineComment))
            .nth(n)
            .map(|token| token.kind())
    }
    pub(super) fn nth_nontrivia_text(&self, n: usize) -> Option<&str> {
        let token = self.tokens[self.position..]
            .iter()
            .filter(|token| !matches!(token.kind(), TokenKind::Whitespace | TokenKind::LineComment))
            .nth(n)?;
        let range = token.range();
        std::str::from_utf8(
            self.source
                .get(range.start() as usize..range.end() as usize)?,
        )
        .ok()
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
