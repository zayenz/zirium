//! Attribute dictionaries, values, payloads, and locations.

use super::*;

impl Parser<'_> {
    pub(in crate::parser) fn attribute_dict(&mut self) -> Result<(), CompactError> {
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

    pub(in crate::parser) fn inherent_attribute_starts(&self) -> bool {
        (self.at_identifier() || self.at(TokenKind::String))
            && self.nth_nontrivia(1) == Some(TokenKind::Equal)
    }

    pub(in crate::parser) fn inherent_attribute(
        &mut self,
        paired: bool,
    ) -> Result<bool, CompactError> {
        let attribute = self.builder.start();
        let mut good = self.at_identifier() || self.at(TokenKind::String);
        if good {
            self.bump()?;
        } else {
            self.diagnostic();
        }
        self.trivia()?;
        good &= self.expect(TokenKind::Equal)?;
        self.trivia()?;
        if paired {
            // StableHLO dimension clauses are a pair, not just the first array.
            // Keep their complete custom spelling as one opaque value.
            let value = self.builder.start();
            good &= self.at(TokenKind::LBracket);
            good &= self.attribute_value()?;
            self.trivia()?;
            if matches!(self.current(), TokenKind::BareIdentifier | TokenKind::X)
                && self.current_text() == "x"
            {
                self.bump()?;
            } else {
                self.diagnostic();
                good = false;
            }
            self.trivia()?;
            good &= self.at(TokenKind::LBracket);
            good &= self.attribute_value()?;
            self.builder
                .complete_with_error(value, SyntaxKind::OpaqueAttribute, !good)?;
            self.builder
                .complete_with_error(attribute, SyntaxKind::Attribute, !good)?;
            return Ok(good);
        }
        good &= if self.at(TokenKind::BareIdentifier)
            && !matches!(
                self.current_text(),
                "array" | "false" | "true" | "type" | "unit"
            ) {
            self.leaf(SyntaxKind::OpaqueAttribute)?
        } else if matches!(
            self.current(),
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Integer
                | TokenKind::WideInteger
                | TokenKind::Float
        ) {
            self.constant_value()?
        } else {
            self.attribute_value()?
        };
        self.builder
            .complete_with_error(attribute, SyntaxKind::Attribute, !good)?;
        Ok(good)
    }

    pub(super) fn dictionary_entries(&mut self) -> Result<bool, CompactError> {
        let mut good = self.expect(TokenKind::LBrace)?;
        self.trivia()?;
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.position;
            let attr = self.builder.start();
            let mut bad = false;
            if self.at_identifier() || self.at(TokenKind::String) {
                self.bump()?;
                self.trivia()?;
                if self.at(TokenKind::RBrace) || self.at(TokenKind::Comma) {
                    // MLIR dictionary entries without `= value` are unit-valued.
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

    pub(super) fn attribute_value(&mut self) -> Result<bool, CompactError> {
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
            TokenKind::LParen => {
                let marker = self.builder.start();
                self.function_type()?;
                self.builder.complete(marker, SyntaxKind::TypeAttribute)?;
                Ok(true)
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
            TokenKind::BareIdentifier if self.current_text() == "array" => {
                self.dense_array_attribute()
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

    pub(in crate::parser) fn constant_value(&mut self) -> Result<bool, CompactError> {
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
                operation_range: None,
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

    pub(in crate::parser) fn symbol_reference(&mut self) -> Result<bool, CompactError> {
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

    fn dense_array_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        if !self.enter_attribute_container(TokenKind::Less, TokenKind::Greater)? {
            self.builder
                .complete_with_error(marker, SyntaxKind::DenseArrayAttribute, true)?;
            return Ok(false);
        }
        let mut good = self.expect(TokenKind::Less)?;
        let payload_start = self.tokens[self.position.saturating_sub(1)].range().end();
        self.trivia()?;
        good &= matches!(self.current(), TokenKind::IntType | TokenKind::FloatType)
            && matches!(
                self.current_text(),
                "i1" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
            );
        if matches!(self.current(), TokenKind::IntType | TokenKind::FloatType) {
            self.bump()?;
        } else {
            self.error_token()?;
        }
        self.trivia()?;
        if self.at(TokenKind::Colon) {
            self.bump()?;
            self.trivia()?;
            let mut need_value = true;
            while !self.at(TokenKind::Greater) && !self.at(TokenKind::Eof) {
                if self.tokens[self.position]
                    .range()
                    .end()
                    .saturating_sub(payload_start) as usize
                    > self.limits.max_payload_bytes
                {
                    self.diagnostic();
                    good = false;
                }
                let value_good = match self.current() {
                    TokenKind::Plus
                    | TokenKind::Minus
                    | TokenKind::Integer
                    | TokenKind::WideInteger
                    | TokenKind::Float => self.numeric_attribute()?,
                    TokenKind::BareIdentifier
                        if matches!(self.current_text(), "true" | "false") =>
                    {
                        self.leaf(SyntaxKind::BooleanAttribute)?
                    }
                    _ => {
                        self.error_token()?;
                        false
                    }
                };
                good &= value_good;
                need_value = false;
                self.trivia()?;
                if self.at(TokenKind::Comma) {
                    self.bump()?;
                    self.trivia()?;
                    need_value = true;
                } else if !self.at(TokenKind::Greater) {
                    good = false;
                    self.diagnostic();
                }
            }
            if need_value {
                good = false;
                self.diagnostic();
            }
        }
        if self.tokens[self.position]
            .range()
            .start()
            .saturating_sub(payload_start) as usize
            > self.limits.max_payload_bytes
        {
            self.diagnostic();
            good = false;
        }
        good &= self.expect(TokenKind::Greater)?;
        self.nesting_depth -= 1;
        self.builder
            .complete_with_error(marker, SyntaxKind::DenseArrayAttribute, !good)?;
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

    pub(in crate::parser) fn location_attribute(&mut self) -> Result<bool, CompactError> {
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
            let at_enclosing_boundary = stack.last() == Some(&TokenKind::RParen)
                && matches!(kind, TokenKind::RBrace | TokenKind::Comma);
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

    pub(super) fn opaque(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
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
}
