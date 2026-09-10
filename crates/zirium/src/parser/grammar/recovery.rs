//! Syntax diagnostics, speculative parsing checkpoints, and recovery boundaries.

use super::*;

impl Parser<'_> {
    pub(super) fn error_token(&mut self) -> Result<(), CompactError> {
        let error = self.builder.start();
        self.diagnostic();
        if !self.at(TokenKind::Eof) {
            self.bump()?;
        }
        self.builder
            .complete_with_error(error, SyntaxKind::Error, true)?;
        Ok(())
    }
    pub(in crate::parser) fn diagnostic(&mut self) {
        self.diagnostic_kind(ParseDiagnosticKind::Syntax);
    }
    pub(super) fn diagnostic_kind(&mut self, kind: ParseDiagnosticKind) {
        self.diagnostics.push(ParseDiagnostic {
            range: self.tokens[self.position].range(),
            operation_range: None,
            kind,
        });
    }
    pub(in crate::parser) fn reset_attempt_failure(&mut self, position: usize) {
        self.furthest_attempt = position;
    }
    pub(in crate::parser) fn record_attempt_failure(&mut self) {
        self.furthest_attempt = self.furthest_attempt.max(self.position);
    }
    fn diagnostic_mismatch(
        &mut self,
        kind: ParseDiagnosticKind,
        position: usize,
        operation_position: usize,
    ) {
        self.diagnostics.push(ParseDiagnostic {
            range: self.tokens[position.min(self.tokens.len() - 1)].range(),
            operation_range: Some(
                self.tokens[operation_position.min(self.tokens.len() - 1)].range(),
            ),
            kind,
        });
    }
    pub(in crate::parser) fn shaped_operation_checkpoint(&self) -> (usize, usize, usize, usize) {
        (
            self.position,
            self.builder.checkpoint(),
            self.diagnostics.len(),
            self.nesting_depth,
        )
    }
    pub(in crate::parser) fn shaped_operation_boundary(&self, crossed_line: bool) -> bool {
        crossed_line
            || matches!(
                self.current(),
                TokenKind::Eof | TokenKind::RBrace | TokenKind::CaretIdentifier
            )
            || self.is_generic_operation_start()
            || self.result_custom_operation_start()
    }
    pub(in crate::parser) fn trivia_crosses_line(&self, start: usize) -> bool {
        let mut start = start;
        if start == self.position {
            while start > 0 && is_trivia(self.tokens[start - 1].kind()) {
                start -= 1;
            }
        }
        self.tokens[start..self.position].iter().any(|token| {
            token.kind() == TokenKind::Whitespace
                && self
                    .source
                    .get(token.range().start() as usize..token.range().end() as usize)
                    .is_some_and(|text| text.contains(&b'\n'))
        })
    }
    pub(in crate::parser) fn recover_shape_mismatch(
        &mut self,
        marker: Marker,
        shape: OperationShape,
        checkpoint: (usize, usize, usize, usize),
    ) -> Result<(), CompactError> {
        let (position, events, diagnostics, nesting_depth) = checkpoint;
        self.position = position;
        self.builder.rewind(events);
        self.diagnostics.truncate(diagnostics);
        self.nesting_depth = nesting_depth;
        self.diagnostic_mismatch(
            ParseDiagnosticKind::ShapeMismatch(shape),
            self.furthest_attempt,
            position,
        );
        self.recover_custom_operation(marker)
    }
    pub(in crate::parser) fn rewind_shaped_operation(
        &mut self,
        checkpoint: (usize, usize, usize, usize),
    ) {
        let (position, events, diagnostics, nesting_depth) = checkpoint;
        self.position = position;
        self.builder.rewind(events);
        self.diagnostics.truncate(diagnostics);
        self.nesting_depth = nesting_depth;
    }
    pub(in crate::parser) fn recover_format_mismatch(
        &mut self,
        marker: Marker,
        checkpoint: (usize, usize, usize, usize),
    ) -> Result<(), CompactError> {
        let (position, events, diagnostics, nesting_depth) = checkpoint;
        self.position = position;
        self.builder.rewind(events);
        self.diagnostics.truncate(diagnostics);
        self.nesting_depth = nesting_depth;
        self.diagnostic_mismatch(
            ParseDiagnosticKind::FormatMismatch,
            self.furthest_attempt,
            position,
        );
        self.recover_custom_operation(marker)
    }
    pub(super) fn ensure_progress(&mut self, before: usize) -> Result<(), CompactError> {
        if self.position == before && !self.at(TokenKind::Eof) {
            self.diagnostic_kind(ParseDiagnosticKind::ProgressLimit);
            self.error_token()?;
        }
        Ok(())
    }
    pub(super) fn unparsed_custom_operation(
        &mut self,
        marker: Option<Marker>,
    ) -> Result<(), CompactError> {
        let marker = marker.unwrap_or_else(|| self.builder.start());
        self.diagnostic_kind(ParseDiagnosticKind::UnknownCustomOperation);
        self.recover_custom_operation(marker)
    }
    fn recover_custom_operation(&mut self, marker: Marker) -> Result<(), CompactError> {
        let start = self.position;
        let mut stack = Vec::new();
        let mut line_boundary = false;
        let mut completed_payload = false;
        let mut in_type_tail = false;
        while !self.at(TokenKind::Eof) {
            let current = self.current();
            if stack.is_empty() && current == TokenKind::LBrace && self.region_shaped_body() {
                self.region()?;
                completed_payload = true;
                continue;
            }
            if stack.is_empty()
                && current == TokenKind::LBrace
                && self.attribute_dictionary_precedes_type_tail()
            {
                self.attribute_dict()?;
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
            if stack.is_empty()
                && self.position > start
                && !in_type_tail
                && current == TokenKind::PercentIdentifier
            {
                let operand = self.builder.start();
                let operand_use = self.builder.start();
                self.bump()?;
                if self.at(TokenKind::HashIdentifier) {
                    self.bump()?;
                }
                self.builder.complete(operand_use, SyntaxKind::OperandUse)?;
                self.builder.complete(operand, SyntaxKind::Operand)?;
                continue;
            }
            if stack.is_empty() && current == TokenKind::Colon {
                in_type_tail = true;
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
    fn attribute_dictionary_precedes_type_tail(&self) -> bool {
        let mut index = self.position;
        let mut stack = vec![TokenKind::RBrace];
        index += 1;
        while let Some(expected) = stack.last().copied() {
            let Some(token) = self.tokens.get(index) else {
                return false;
            };
            let current = token.kind();
            if current == expected {
                stack.pop();
            } else if let Some(close) = close_for(current) {
                stack.push(close);
            } else if current == TokenKind::Eof {
                return false;
            }
            index += 1;
        }
        while self
            .tokens
            .get(index)
            .is_some_and(|token| is_trivia(token.kind()))
        {
            index += 1;
        }
        self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::Colon)
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
    pub(in crate::parser) fn region_shaped_body(&self) -> bool {
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
    pub(super) fn recover_balanced_region(&mut self) -> Result<(), CompactError> {
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
}
