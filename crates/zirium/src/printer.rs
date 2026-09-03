//! Deterministic generic MLIR output from semantic documents.

mod compare;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    io::{self, Write},
    path::Path,
};

use crate::dialect::DialectRegistry;
use crate::lexer::TokenKind;
use crate::semantic::{
    AffineExprValue, AttributeValue, BlockId, Document, LargeAttributeValue, LocationValue,
    MemRefLayout, OperationId, RegionId, ShapedDimension, TypeValue, ValidationError, ValueId,
    ValueReference,
};
use crate::{SyntaxKind, source::TextRange};

/// Whitespace policy for generated semantic output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrintLayout {
    Compact,
    #[default]
    Pretty,
}

/// Chooses generic MLIR or registered custom syntax where available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DialectPrintMode {
    #[default]
    GenericOnly,
    PreferCustom,
}

/// Failure while printing a semantic document.
#[derive(Debug)]
pub enum PrintError {
    IncompleteDocument,
    InvalidDocument(ValidationError),
    Format(fmt::Error),
    Io(io::Error),
}

/// Failure while combining retained source with generated replacements.
#[derive(Debug)]
pub enum PreserveError {
    NotHybrid,
    IncompleteDocument,
    InvalidDocument(ValidationError),
    MissingSyntaxMapping,
    UnknownCustomSyntax(TextRange),
    Format(fmt::Error),
    Io(io::Error),
}

impl fmt::Display for PreserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotHybrid => f.write_str("source-preserving output requires a hybrid document"),
            Self::IncompleteDocument => {
                f.write_str("cannot regenerate an incomplete semantic document")
            }
            Self::InvalidDocument(error) => write!(f, "cannot preserve invalid document: {error}"),
            Self::MissingSyntaxMapping => {
                f.write_str("dirty semantic syntax has no retained source range")
            }
            Self::UnknownCustomSyntax(range) => write!(
                f,
                "dirty replacement at {range} contains unknown custom syntax"
            ),
            Self::Format(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for PreserveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDocument(error) => Some(error),
            Self::Format(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}
impl fmt::Display for PrintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompleteDocument => f.write_str("cannot print an incomplete semantic document"),
            Self::InvalidDocument(error) => write!(f, "cannot print invalid document: {error}"),
            Self::Format(e) => e.fmt(f),
            Self::Io(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for PrintError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDocument(error) => Some(error),
            Self::Format(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::IncompleteDocument => None,
        }
    }
}

impl Document {
    /// Prints selected operations with the operation, region, and block shells
    /// that contain them. Unselected siblings are omitted.
    pub fn write_selection<W: io::Write>(
        &self,
        sink: &mut W,
        selected: &[OperationId],
        layout: PrintLayout,
        registry: &DialectRegistry,
    ) -> Result<(), PrintError> {
        if !self.is_semantically_complete() {
            return Err(PrintError::IncompleteDocument);
        }
        let mut selected = selected.iter().copied().collect::<HashSet<_>>();
        let mut worklist = selected.iter().copied().collect::<Vec<_>>();
        while let Some(operation) = worklist.pop() {
            for &region in self.operation_regions(operation).unwrap_or(&[]) {
                for &block in self
                    .region(region)
                    .and_then(|region| region.blocks(self))
                    .unwrap_or(&[])
                {
                    for &child in self.block_operations(block).unwrap_or(&[]) {
                        if selected.insert(child) {
                            worklist.push(child);
                        }
                    }
                }
            }
        }
        let mut adapter = IoAdapter { sink, error: None };
        let result =
            Printer::new_selection(self, &mut adapter, layout, registry, &selected).document();
        if let Some(error) = adapter.error {
            return Err(PrintError::Io(error));
        }
        result.map_err(PrintError::Format)
    }

    /// Writes deterministic generic MLIR to a [`fmt::Write`] sink.
    ///
    /// # Errors
    ///
    /// Rejects incomplete or structurally invalid documents and forwards sink
    /// formatting errors.
    pub fn print<W: fmt::Write>(
        &self,
        sink: &mut W,
        layout: PrintLayout,
    ) -> Result<(), PrintError> {
        self.preflight_print()?;
        Printer::new(
            self,
            sink,
            layout,
            DialectPrintMode::GenericOnly,
            &DialectRegistry::EMPTY,
        )
        .document()
        .map_err(PrintError::Format)
    }
    /// Alias for [`Self::print`] that names the canonical-output contract.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::print`].
    pub fn print_canonical<W: fmt::Write>(
        &self,
        sink: &mut W,
        layout: PrintLayout,
    ) -> Result<(), PrintError> {
        self.print(sink, layout)
    }
    /// Returns deterministic generic MLIR as bytes.
    ///
    /// # Errors
    ///
    /// Rejects incomplete or structurally invalid documents.
    pub fn canonical_bytes(&self, layout: PrintLayout) -> Result<Vec<u8>, PrintError> {
        let mut bytes = Vec::new();
        self.write_canonical(&mut bytes, layout)?;
        Ok(bytes)
    }
    /// Writes semantic output with optional registered custom syntax.
    ///
    /// In [`DialectPrintMode::PreferCustom`], a registered operation uses its
    /// built-in assembly printer or callback when that printer succeeds, and
    /// falls back to generic syntax otherwise.
    ///
    /// # Errors
    ///
    /// Rejects incomplete or structurally invalid documents and forwards sink
    /// formatting errors.
    pub fn print_with_registry<W: fmt::Write>(
        &self,
        sink: &mut W,
        layout: PrintLayout,
        mode: DialectPrintMode,
        registry: &DialectRegistry,
    ) -> Result<(), PrintError> {
        self.preflight_print()?;
        Printer::new(self, sink, layout, mode, registry)
            .document()
            .map_err(PrintError::Format)
    }
    pub fn print_io<W: io::Write>(
        &self,
        sink: &mut W,
        layout: PrintLayout,
    ) -> Result<(), PrintError> {
        self.preflight_print()?;
        let mut adapter = IoAdapter { sink, error: None };
        let result = Printer::new(
            self,
            &mut adapter,
            layout,
            DialectPrintMode::GenericOnly,
            &DialectRegistry::EMPTY,
        )
        .document();
        if let Some(error) = adapter.error {
            return Err(PrintError::Io(error));
        }
        result.map_err(PrintError::Format)
    }
    /// Writes deterministic generic MLIR derived entirely from semantic storage.
    ///
    /// # Errors
    ///
    /// Rejects incomplete or structurally invalid documents and forwards I/O
    /// errors from the sink.
    pub fn write_canonical<W: io::Write>(
        &self,
        sink: &mut W,
        layout: PrintLayout,
    ) -> Result<(), PrintError> {
        self.print_io(sink, layout)
    }
    /// Writes deterministic generic MLIR to a newly created file.
    ///
    /// # Errors
    ///
    /// Returns validation, formatting, file creation, write, or flush errors.
    pub fn print_to_file(
        &self,
        path: impl AsRef<Path>,
        layout: PrintLayout,
    ) -> Result<(), PrintError> {
        self.preflight_print()?;
        self.print_with_registry_to_preflighted_file(
            path,
            layout,
            DialectPrintMode::GenericOnly,
            &DialectRegistry::EMPTY,
        )
    }

    /// Writes semantic output with optional custom syntax to a new file.
    ///
    /// # Errors
    ///
    /// Returns validation, formatting, file creation, write, or flush errors.
    pub fn print_with_registry_to_file(
        &self,
        path: impl AsRef<Path>,
        layout: PrintLayout,
        mode: DialectPrintMode,
        registry: &DialectRegistry,
    ) -> Result<(), PrintError> {
        self.preflight_print()?;
        self.print_with_registry_to_preflighted_file(path, layout, mode, registry)
    }

    fn print_with_registry_to_preflighted_file(
        &self,
        path: impl AsRef<Path>,
        layout: PrintLayout,
        mode: DialectPrintMode,
        registry: &DialectRegistry,
    ) -> Result<(), PrintError> {
        let file = std::fs::File::create(path).map_err(PrintError::Io)?;
        let mut writer = io::BufWriter::new(file);
        let mut adapter = IoAdapter {
            sink: &mut writer,
            error: None,
        };
        let result = Printer::new(self, &mut adapter, layout, mode, registry).document();
        if let Some(error) = adapter.error {
            return Err(PrintError::Io(error));
        }
        result.map_err(PrintError::Format)?;
        writer.flush().map_err(PrintError::Io)
    }
    fn preflight_print(&self) -> Result<(), PrintError> {
        if !self.is_semantically_complete() {
            return Err(PrintError::IncompleteDocument);
        }
        self.validate().map_err(PrintError::InvalidDocument)
    }

    /// Returns source-preserving output as bytes.
    ///
    /// Unchanged ranges are copied from retained source. Dirty operations and
    /// blocks are regenerated using generic syntax.
    ///
    /// # Errors
    ///
    /// Requires a complete, valid [`RetentionProfile::Hybrid`](crate::semantic::RetentionProfile::Hybrid)
    /// document with syntax mappings for every generated replacement. Replacing
    /// a range containing unknown custom syntax also fails.
    pub fn preserving_bytes(&self, layout: PrintLayout) -> Result<Vec<u8>, PreserveError> {
        let plan = self.preserving_plan(layout)?;
        let source = self.source_bytes().ok_or(PreserveError::NotHybrid)?;
        let mut output = Vec::with_capacity(source.len());
        self.write_preserving_plan(&mut output, layout, &plan)?;
        Ok(output)
    }

    fn preserving_plan(
        &self,
        layout: PrintLayout,
    ) -> Result<Vec<PreservingReplacement>, PreserveError> {
        self.source_bytes().ok_or(PreserveError::NotHybrid)?;
        if self.retention_profile() != crate::semantic::RetentionProfile::Hybrid {
            return Err(PreserveError::NotHybrid);
        }
        if self.dirty_operations().is_empty() && self.dirty_blocks().is_empty() {
            return Ok(Vec::new());
        }
        if !self.is_semantically_complete() {
            return Err(PreserveError::IncompleteDocument);
        }
        self.validate().map_err(PreserveError::InvalidDocument)?;

        let mut replacements = Vec::new();
        for &operation in self.dirty_operations() {
            let range = self
                .operation_syntax_range(operation)
                .ok_or(PreserveError::MissingSyntaxMapping)?;
            replacements.push(PreservingReplacement::Operation(
                self.trim_trailing_trivia(range),
                operation,
            ));
        }
        for &block in self.dirty_blocks() {
            let range = self
                .block_syntax_range(block)
                .ok_or(PreserveError::MissingSyntaxMapping)?;
            replacements.push(PreservingReplacement::Block(
                self.trim_trailing_trivia(range),
                block,
            ));
        }
        replacements.sort_by_key(|replacement| {
            (
                replacement.range().start(),
                std::cmp::Reverse(replacement.range().end()),
            )
        });
        let mut planned = Vec::new();
        for replacement in replacements {
            if planned.last().is_some_and(|outer: &PreservingReplacement| {
                outer.range().end() >= replacement.range().end()
            }) {
                continue;
            }
            planned.push(replacement);
        }
        let tree = self.syntax_tree().ok_or(PreserveError::NotHybrid)?;
        let mut replacement = planned.iter().peekable();
        for custom in tree
            .subtree(tree.root())
            .into_iter()
            .flatten()
            .filter(|node| tree.kind(*node) == Some(SyntaxKind::UnparsedCustomOperation))
            .filter_map(|node| tree.text_range(node))
        {
            while replacement
                .peek()
                .is_some_and(|replacement| replacement.range().end() <= custom.start())
            {
                replacement.next();
            }
            if replacement.peek().is_some_and(|replacement| {
                let range = replacement.range();
                range.start() <= custom.start() && custom.start() < range.end()
            }) {
                return Err(PreserveError::UnknownCustomSyntax(custom));
            }
        }

        // Formatting is infallible for a validated document in practice, but run it
        // during preflight so no formatting error can occur after the first sink write.
        for replacement in &planned {
            let _ = self.render_replacement(*replacement, layout)?;
        }
        Ok(planned)
    }

    fn write_preserving_plan<W: io::Write>(
        &self,
        sink: &mut W,
        layout: PrintLayout,
        plan: &[PreservingReplacement],
    ) -> Result<(), PreserveError> {
        let source = self.source_bytes().ok_or(PreserveError::NotHybrid)?;
        let mut cursor = 0usize;
        for &replacement in plan {
            let range = replacement.range();
            let start = range.start() as usize;
            let end = range.end() as usize;
            sink.write_all(&source[cursor..start])
                .map_err(PreserveError::Io)?;
            let bytes = self.render_replacement(replacement, layout)?;
            sink.write_all(&bytes).map_err(PreserveError::Io)?;
            cursor = end;
        }
        sink.write_all(&source[cursor..]).map_err(PreserveError::Io)
    }

    /// Retains unchanged source bytes and regenerates edited regions.
    ///
    /// This mode requires [`crate::semantic::RetentionProfile::Hybrid`].
    ///
    /// # Errors
    ///
    /// Returns the planning errors described by [`Self::preserving_bytes`] or an
    /// I/O error from the sink. Planning completes before the first write.
    pub fn write_preserving<W: io::Write>(
        &self,
        sink: &mut W,
        layout: PrintLayout,
    ) -> Result<(), PreserveError> {
        let plan = self.preserving_plan(layout)?;
        self.write_preserving_plan(sink, layout, &plan)
    }

    /// Writes source-preserving output to a newly created file.
    ///
    /// # Errors
    ///
    /// Returns the planning errors described by [`Self::preserving_bytes`] or a
    /// file creation or write error.
    pub fn write_preserving_to_file(
        &self,
        path: impl AsRef<Path>,
        layout: PrintLayout,
    ) -> Result<(), PreserveError> {
        let plan = self.preserving_plan(layout)?;
        let mut file = std::fs::File::create(path).map_err(PreserveError::Io)?;
        self.write_preserving_plan(&mut file, layout, &plan)
    }

    fn render_replacement(
        &self,
        replacement: PreservingReplacement,
        layout: PrintLayout,
    ) -> Result<Vec<u8>, PreserveError> {
        match replacement {
            PreservingReplacement::Operation(_, operation) => {
                self.render_operation(operation, layout)
            }
            PreservingReplacement::Block(_, block) => self.render_block(block, layout),
        }
    }

    fn render_operation(
        &self,
        operation: OperationId,
        layout: PrintLayout,
    ) -> Result<Vec<u8>, PreserveError> {
        let mut output = String::new();
        Printer::new(
            self,
            &mut output,
            layout,
            DialectPrintMode::GenericOnly,
            &DialectRegistry::EMPTY,
        )
        .operation(operation, 0)
        .map_err(PreserveError::Format)?;
        Ok(output.into_bytes())
    }

    fn render_block(&self, block: BlockId, layout: PrintLayout) -> Result<Vec<u8>, PreserveError> {
        let mut output = String::new();
        Printer::new(
            self,
            &mut output,
            layout,
            DialectPrintMode::GenericOnly,
            &DialectRegistry::EMPTY,
        )
        .block_replacement(block)
        .map_err(PreserveError::Format)?;
        Ok(output.into_bytes())
    }

    fn block_syntax_range(&self, block: BlockId) -> Option<TextRange> {
        let operations = self.block_operations(block)?;
        let first = self.operation_syntax_range(*operations.first()?)?;
        let last = self.operation_syntax_range(*operations.last()?)?;
        let tree = self.syntax_tree()?;
        tree.subtree(tree.root())?
            .filter(|node| tree.kind(*node) == Some(SyntaxKind::Block))
            .filter_map(|node| tree.text_range(node))
            .filter(|range| range.start() <= first.start() && last.end() <= range.end())
            .min_by_key(|range| range.len())
    }

    fn trim_trailing_trivia(&self, range: TextRange) -> TextRange {
        let Some(tree) = self.syntax_tree() else {
            return range;
        };
        let end = (0..tree.token_count())
            .filter_map(|index| tree.token(index).copied())
            .filter(|token| {
                range.start() <= token.range().start() && token.range().end() <= range.end()
            })
            .filter(|token| {
                !matches!(
                    token.kind(),
                    TokenKind::Whitespace | TokenKind::LineComment | TokenKind::Eof
                )
            })
            .map(|token| token.range().end())
            .max()
            .unwrap_or(range.end());
        TextRange::new(range.start(), end).expect("trimmed source range remains ordered")
    }
}

#[derive(Clone, Copy)]
enum PreservingReplacement {
    Operation(TextRange, OperationId),
    Block(TextRange, BlockId),
}

impl PreservingReplacement {
    fn range(self) -> TextRange {
        match self {
            Self::Operation(range, _) | Self::Block(range, _) => range,
        }
    }
}

struct IoAdapter<'a, W> {
    sink: &'a mut W,
    error: Option<io::Error>,
}

impl<W: io::Write> fmt::Write for IoAdapter<'_, W> {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.sink.write_all(value.as_bytes()).map_err(|error| {
            self.error = Some(error);
            fmt::Error
        })
    }
}

struct Printer<'a, W> {
    doc: &'a Document,
    sink: &'a mut W,
    layout: PrintLayout,
    values: HashMap<ValueId, usize>,
    blocks: HashMap<BlockId, usize>,
    mode: DialectPrintMode,
    registry: &'a DialectRegistry,
    selected: Option<&'a HashSet<OperationId>>,
}
impl<'a, W: fmt::Write> Printer<'a, W> {
    fn new(
        doc: &'a Document,
        sink: &'a mut W,
        layout: PrintLayout,
        mode: DialectPrintMode,
        registry: &'a DialectRegistry,
    ) -> Self {
        let (mut values, mut next_value) = (HashMap::new(), 0);
        for operation in doc.operations() {
            for result in 0..doc.result_types(operation).map_or(0, <[_]>::len) {
                values.insert(
                    ValueId::OperationResult {
                        operation,
                        result: result as u32,
                    },
                    next_value,
                );
                next_value += 1;
            }
        }
        let (mut blocks, mut next_block) = (HashMap::new(), 0);
        for operation in doc.operations() {
            for &region in doc.operation_regions(operation).unwrap_or(&[]) {
                for &block in doc
                    .region(region)
                    .and_then(|r| r.blocks(doc))
                    .unwrap_or(&[])
                {
                    blocks.insert(block, next_block);
                    next_block += 1;
                    for argument in 0..doc.block_argument_types(block).map_or(0, <[_]>::len) {
                        values.insert(
                            ValueId::BlockArgument {
                                block,
                                argument: argument as u32,
                            },
                            next_value,
                        );
                        next_value += 1;
                    }
                }
            }
        }
        Self {
            doc,
            sink,
            layout,
            values,
            blocks,
            mode,
            registry,
            selected: None,
        }
    }
    fn new_selection(
        doc: &'a Document,
        sink: &'a mut W,
        layout: PrintLayout,
        registry: &'a DialectRegistry,
        selected: &'a HashSet<OperationId>,
    ) -> Self {
        let mut printer = Self::new(doc, sink, layout, DialectPrintMode::PreferCustom, registry);
        printer.selected = Some(selected);
        printer
    }
    fn retained(&self, operation: OperationId) -> bool {
        self.selected.is_none_or(|selected| {
            selected.contains(&operation)
                || self
                    .doc
                    .operation_regions(operation)
                    .unwrap_or(&[])
                    .iter()
                    .any(|&region| {
                        self.doc
                            .region(region)
                            .and_then(|r| r.blocks(self.doc))
                            .unwrap_or(&[])
                            .iter()
                            .any(|&block| {
                                self.doc
                                    .block_operations(block)
                                    .unwrap_or(&[])
                                    .iter()
                                    .any(|&child| self.retained(child))
                            })
                    })
        })
    }
    fn selected_comment(&mut self, operation: OperationId, indent: usize) -> fmt::Result {
        let Some(selected) = self.selected else {
            return Ok(());
        };
        if !selected.contains(&operation) {
            return Ok(());
        }
        let (Some(source), Some(range)) = (
            self.doc.source_bytes(),
            self.doc.operation_syntax_range(operation),
        ) else {
            return Ok(());
        };
        let start = range.start() as usize;
        let before = &source[..start];
        let line_start = before
            .iter()
            .rposition(|&b| b == b'\n')
            .map_or(0, |i| i + 1);
        if !source[line_start..start]
            .iter()
            .all(|byte| byte.is_ascii_whitespace())
        {
            return Ok(());
        }
        let mut comments = Vec::new();
        let mut cursor = line_start;
        while cursor > 0 {
            let previous_end = cursor - 1;
            let previous_start = source[..previous_end]
                .iter()
                .rposition(|&b| b == b'\n')
                .map_or(0, |i| i + 1);
            let line = &source[previous_start..previous_end];
            if String::from_utf8_lossy(line).trim_start().starts_with("//") {
                comments.push(String::from_utf8_lossy(line).trim().to_owned());
                cursor = previous_start;
            } else {
                break;
            }
        }
        comments.reverse();
        for comment in comments {
            self.sink.write_str(&comment)?;
            self.newline(indent)?;
        }
        Ok(())
    }
    fn selected_trailing_comment(&mut self, operation: OperationId) -> fmt::Result {
        let Some(selected) = self.selected else {
            return Ok(());
        };
        if !selected.contains(&operation) {
            return Ok(());
        }
        let (Some(source), Some(range)) = (
            self.doc.source_bytes(),
            self.doc
                .operation_syntax_range(operation)
                .map(|range| self.doc.trim_trailing_trivia(range)),
        ) else {
            return Ok(());
        };
        let end = range.end() as usize;
        let line_end = source[end..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(source.len(), |i| end + i);
        let tail = String::from_utf8_lossy(&source[end..line_end]);
        if let Some(comment) = tail.find("//") {
            if !tail.as_bytes()[..comment]
                .iter()
                .all(|byte| matches!(byte, b' ' | b'\t'))
            {
                return Ok(());
            }
            self.sink.write_char(' ')?;
            self.sink.write_str(tail[comment..].trim_end())?;
        }
        Ok(())
    }
    fn document(&mut self) -> fmt::Result {
        let roots = self
            .doc
            .root_operations()
            .iter()
            .copied()
            .filter(|&op| self.retained(op))
            .collect::<Vec<_>>();
        let has_roots = !roots.is_empty();
        let mut index = 0;
        for operation in roots {
            if index != 0 {
                self.newline(0)?;
            }
            self.selected_comment(operation, 0)?;
            self.operation(operation, 0)?;
            self.selected_trailing_comment(operation)?;
            index += 1;
        }
        if self.layout == PrintLayout::Pretty && has_roots {
            self.sink.write_char('\n')?;
        }
        Ok(())
    }
    fn operation(&mut self, id: OperationId, indent: usize) -> fmt::Result {
        let results = self.doc.result_types(id).ok_or(fmt::Error)?;
        for result in 0..results.len() {
            if result != 0 {
                self.sink.write_str(", ")?;
            }
            self.value(ValueId::OperationResult {
                operation: id,
                result: result as u32,
            })?;
        }
        if !results.is_empty() {
            self.sink.write_str(" = ")?;
        }
        if self.mode == DialectPrintMode::PreferCustom {
            if let Some(custom) = self
                .doc
                .operation_name(id)
                .and_then(|name| self.registry.operation(name))
                .and_then(|descriptor| {
                    descriptor
                        .assembly
                        .and_then(|program| program.print(self.doc, id))
                        .or_else(|| descriptor.print.and_then(|print| print(self.doc, id)))
                })
            {
                self.sink.write_str(&custom)?;
                let assembly = self
                    .doc
                    .operation_name(id)
                    .and_then(|name| self.registry.operation(name))
                    .and_then(|descriptor| descriptor.assembly);
                if matches!(
                    assembly,
                    Some(
                        crate::dialect::AssemblyProgram::Module
                            | crate::dialect::AssemblyProgram::Function
                    )
                ) {
                    let regions = self.doc.operation_regions(id).ok_or(fmt::Error)?;
                    if let Some(region) = regions.first() {
                        return self.region(*region, indent);
                    }
                }
                return Ok(());
            }
        }
        write!(
            self.sink,
            "\"{}\"(",
            self.doc.operation_name(id).ok_or(fmt::Error)?
        )?;
        self.value_references(self.doc.operands(id).ok_or(fmt::Error)?)?;
        self.sink.write_char(')')?;
        let successors = self.doc.successors(id).ok_or(fmt::Error)?;
        if !successors.is_empty() {
            self.sink.write_str(" [")?;
            for (index, successor) in successors.iter().copied().enumerate() {
                if index != 0 {
                    self.sink.write_str(", ")?;
                }
                self.block_name(successor.block)?;
                let arguments = self.doc.successor_arguments(successor).ok_or(fmt::Error)?;
                if !arguments.is_empty() {
                    self.sink.write_str(" : (")?;
                    for (index, &argument) in arguments.iter().enumerate() {
                        if index != 0 {
                            self.sink.write_str(", ")?;
                        }
                        self.value_reference(argument)?;
                        self.sink.write_str(" : ")?;
                        let value_type = self.value_type(argument).ok_or(fmt::Error)?.clone();
                        self.type_value(&value_type)?;
                    }
                    self.sink.write_char(')')?;
                }
            }
            self.sink.write_char(']')?;
        }
        let regions = self.doc.operation_regions(id).ok_or(fmt::Error)?;
        if !regions.is_empty() {
            self.sink.write_str(" (")?;
            for (index, &region) in regions.iter().enumerate() {
                if index != 0 {
                    self.sink.write_str(", ")?;
                }
                self.region(region, indent)?;
            }
            self.sink.write_char(')')?;
        }
        self.dictionary(
            self.doc.operation_properties(id).ok_or(fmt::Error)?,
            " <{",
            "}>",
        )?;
        self.dictionary(
            self.doc.operation_attributes(id).ok_or(fmt::Error)?,
            " {",
            "}",
        )?;
        self.sink.write_str(" : ")?;
        self.type_id(self.doc.function_type(id).ok_or(fmt::Error)?)?;
        if let Some(location) = self.doc.operation_location_id(id).ok_or(fmt::Error)? {
            self.sink.write_str(" loc(")?;
            self.location(self.doc.location_value(location).ok_or(fmt::Error)?)?;
            self.sink.write_char(')')?;
        }
        Ok(())
    }
    fn block_replacement(&mut self, block: BlockId) -> fmt::Result {
        let explicit = self.doc.block_label(block).ok_or(fmt::Error)?.is_some()
            || !self
                .doc
                .block_argument_types(block)
                .ok_or(fmt::Error)?
                .is_empty();
        if explicit {
            self.block_name(block)?;
            let arguments = self.doc.block_argument_types(block).ok_or(fmt::Error)?;
            if !arguments.is_empty() {
                self.sink.write_char('(')?;
                for (index, &ty) in arguments.iter().enumerate() {
                    if index != 0 {
                        self.sink.write_str(", ")?;
                    }
                    self.value(ValueId::BlockArgument {
                        block,
                        argument: index as u32,
                    })?;
                    self.sink.write_str(" : ")?;
                    self.type_id(ty)?;
                }
                self.sink.write_char(')')?;
            }
            self.sink.write_char(':')?;
            if !self
                .doc
                .block_operations(block)
                .ok_or(fmt::Error)?
                .is_empty()
            {
                self.newline(1)?;
            }
        }
        for (index, &operation) in self
            .doc
            .block_operations(block)
            .ok_or(fmt::Error)?
            .iter()
            .enumerate()
        {
            if index != 0 {
                self.newline(usize::from(explicit))?;
            }
            self.operation(operation, usize::from(explicit))?;
        }
        Ok(())
    }
    fn region(&mut self, id: crate::semantic::RegionId, indent: usize) -> fmt::Result {
        self.sink.write_char('{')?;
        let blocks = self
            .doc
            .region(id)
            .and_then(|r| r.blocks(self.doc))
            .ok_or(fmt::Error)?
            .iter()
            .copied()
            .filter(|&block| {
                self.doc
                    .block_operations(block)
                    .unwrap_or(&[])
                    .iter()
                    .any(|&op| self.retained(op))
            })
            .collect::<Vec<_>>();
        for (block_index, &block) in blocks.iter().enumerate() {
            let explicit = block_index != 0
                || self.doc.block_label(block).ok_or(fmt::Error)?.is_some()
                || !self
                    .doc
                    .block_argument_types(block)
                    .ok_or(fmt::Error)?
                    .is_empty();
            if explicit {
                self.newline(indent + 1)?;
                self.block_name(block)?;
                let arguments = self.doc.block_argument_types(block).ok_or(fmt::Error)?;
                if !arguments.is_empty() {
                    self.sink.write_char('(')?;
                    for (index, &ty) in arguments.iter().enumerate() {
                        if index != 0 {
                            self.sink.write_str(", ")?;
                        }
                        self.value(ValueId::BlockArgument {
                            block,
                            argument: index as u32,
                        })?;
                        self.sink.write_str(" : ")?;
                        self.type_id(ty)?;
                    }
                    self.sink.write_char(')')?;
                }
                self.sink.write_char(':')?;
            }
            let operations = self
                .doc
                .block_operations(block)
                .ok_or(fmt::Error)?
                .iter()
                .copied()
                .filter(|&op| self.retained(op))
                .collect::<Vec<_>>();
            for operation in operations {
                self.newline(indent + 1 + usize::from(explicit))?;
                self.selected_comment(operation, indent + 1 + usize::from(explicit))?;
                self.operation(operation, indent + 1 + usize::from(explicit))?;
                self.selected_trailing_comment(operation)?;
            }
        }
        if !blocks.is_empty() {
            self.newline(indent)?;
        }
        self.sink.write_char('}')
    }
    fn dictionary(
        &mut self,
        entries: &[(u32, crate::semantic::AttributeId)],
        open: &str,
        close: &str,
    ) -> fmt::Result {
        if entries.is_empty() {
            return Ok(());
        }
        self.sink.write_str(open)?;
        for (index, &(name, value)) in entries.iter().enumerate() {
            if index != 0 {
                self.sink.write_str(", ")?;
            }
            write!(self.sink, "{} = ", self.doc.string(name).ok_or(fmt::Error)?)?;
            self.attribute(self.doc.attribute_value(value).ok_or(fmt::Error)?)?;
        }
        self.sink.write_str(close)
    }
    fn value_references(&mut self, values: &[ValueReference]) -> fmt::Result {
        for (index, &value) in values.iter().enumerate() {
            if index != 0 {
                self.sink.write_str(", ")?;
            }
            self.value_reference(value)?;
        }
        Ok(())
    }
    fn value_reference(&mut self, value: ValueReference) -> fmt::Result {
        match value {
            ValueReference::Resolved(value) => self.value(value),
            ValueReference::Invalid(_) => Err(fmt::Error),
        }
    }
    fn value(&mut self, value: ValueId) -> fmt::Result {
        write!(
            self.sink,
            "%v{}",
            self.values.get(&value).ok_or(fmt::Error)?
        )
    }
    fn block_name(&mut self, block: BlockId) -> fmt::Result {
        write!(
            self.sink,
            "^bb{}",
            self.blocks.get(&block).ok_or(fmt::Error)?
        )
    }
    fn value_type(&self, value: ValueReference) -> Option<&TypeValue> {
        let ty = match value {
            ValueReference::Resolved(ValueId::OperationResult { operation, result }) => {
                *self.doc.result_types(operation)?.get(result as usize)?
            }
            ValueReference::Resolved(ValueId::BlockArgument { block, argument }) => *self
                .doc
                .block_argument_types(block)?
                .get(argument as usize)?,
            ValueReference::Invalid(_) => return None,
        };
        self.doc.type_value(ty)
    }
    fn type_id(&mut self, id: crate::semantic::TypeId) -> fmt::Result {
        self.type_value(self.doc.type_value(id).ok_or(fmt::Error)?)
    }
    fn type_value(&mut self, value: &TypeValue) -> fmt::Result {
        match value {
            TypeValue::Integer { width, signedness } => write!(
                self.sink,
                "{}i{width}",
                match signedness {
                    Some(true) => "s",
                    Some(false) => "u",
                    None => "",
                }
            ),
            TypeValue::Float(name) => self.sink.write_str(name),
            TypeValue::Index => self.sink.write_str("index"),
            TypeValue::Tuple(values) => {
                self.sink.write_str("tuple<")?;
                self.types(values)?;
                self.sink.write_str(" >")
            }
            TypeValue::Function { inputs, results } => {
                self.type_list(inputs)?;
                self.sink.write_str(" -> ")?;
                self.type_list(results)
            }
            TypeValue::Tensor {
                dimensions,
                element,
                encoding,
                unranked,
            } => {
                self.sink.write_str("tensor<")?;
                if *unranked {
                    self.sink.write_str("*x")?;
                } else {
                    self.dimensions(dimensions)?;
                }
                self.type_value(element)?;
                if let Some(encoding) = encoding {
                    self.sink.write_str(", ")?;
                    self.attribute(encoding)?;
                }
                self.sink.write_str(" >")
            }
            TypeValue::Vector {
                dimensions,
                element,
                scalable,
            } => {
                self.sink.write_str("vector<")?;
                self.dimensions_with_scalable(dimensions, scalable)?;
                self.type_value(element)?;
                self.sink.write_str(" >")
            }
            TypeValue::MemRef {
                dimensions,
                element,
                layout,
                memory_space,
            } => {
                self.sink.write_str("memref<")?;
                self.dimensions(dimensions)?;
                self.type_value(element)?;
                if let Some(layout) = layout {
                    self.sink.write_str(", ")?;
                    self.memref_layout(layout)?;
                }
                if let Some(space) = memory_space {
                    self.sink.write_str(", ")?;
                    self.attribute(space)?;
                }
                self.sink.write_str(" >")
            }
            TypeValue::Opaque(bytes) => self.opaque(bytes),
            TypeValue::Invalid(_) => Err(fmt::Error),
        }
    }
    fn type_list(&mut self, values: &[TypeValue]) -> fmt::Result {
        if values.len() == 1 {
            return self.type_value(&values[0]);
        }
        self.sink.write_char('(')?;
        self.types(values)?;
        self.sink.write_char(')')
    }
    fn types(&mut self, values: &[TypeValue]) -> fmt::Result {
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                self.sink.write_str(", ")?;
            }
            self.type_value(value)?;
        }
        Ok(())
    }
    fn dimensions(&mut self, values: &[ShapedDimension]) -> fmt::Result {
        self.dimensions_with_scalable(values, &[])
    }
    fn dimensions_with_scalable(
        &mut self,
        values: &[ShapedDimension],
        scalable: &[bool],
    ) -> fmt::Result {
        for (index, value) in values.iter().enumerate() {
            if value.scalable || scalable.get(index).copied().unwrap_or(false) {
                self.sink.write_char('[')?;
            }
            match value.size {
                Some(size) => write!(self.sink, "{size}")?,
                None => self.sink.write_char('?')?,
            }
            if value.scalable || scalable.get(index).copied().unwrap_or(false) {
                self.sink.write_char(']')?;
            }
            self.sink.write_char('x')?;
        }
        Ok(())
    }
    fn memref_layout(&mut self, value: &MemRefLayout) -> fmt::Result {
        match value {
            MemRefLayout::AffineMap(map) => self.affine_map(*map),
            MemRefLayout::Opaque {
                spelling,
                parameters,
            } => self.opaque_layout(spelling, parameters),
            MemRefLayout::Attribute(value) => self.attribute(value),
            MemRefLayout::Invalid(_) => Err(fmt::Error),
        }
    }
    fn attribute(&mut self, value: &AttributeValue) -> fmt::Result {
        match value {
            AttributeValue::Boolean(value) => {
                self.sink.write_str(if *value { "true" } else { "false" })
            }
            AttributeValue::Integer(v) | AttributeValue::Float(v) | AttributeValue::String(v) => {
                self.sink.write_str(v)
            }
            AttributeValue::Type(v) => {
                self.sink.write_str("type<")?;
                self.type_value(v)?;
                self.sink.write_char('>')
            }
            AttributeValue::Symbol(parts) => {
                for (index, part) in parts.iter().enumerate() {
                    if index != 0 {
                        self.sink.write_str("::")?;
                    }
                    self.sink.write_char('@')?;
                    self.sink.write_str(part)?;
                }
                Ok(())
            }
            AttributeValue::Array(values) => {
                self.sink.write_char('[')?;
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        self.sink.write_str(", ")?;
                    }
                    self.attribute(value)?;
                }
                self.sink.write_char(']')
            }
            AttributeValue::Dictionary(values) => {
                self.sink.write_char('{')?;
                for (index, (name, value)) in values.iter().enumerate() {
                    if index != 0 {
                        self.sink.write_str(", ")?;
                    }
                    write!(self.sink, "{name} = ")?;
                    self.attribute(value)?;
                }
                self.sink.write_char('}')
            }
            AttributeValue::Location(v) => {
                self.sink.write_str("loc(")?;
                self.location(v)?;
                self.sink.write_char(')')
            }
            AttributeValue::AffineMap(map) => self.affine_map(*map),
            AttributeValue::IntegerSet(set) => self.integer_set(*set),
            AttributeValue::Large(
                LargeAttributeValue::Dense(v)
                | LargeAttributeValue::Sparse(v)
                | LargeAttributeValue::Resource(v),
            )
            | AttributeValue::WideNumber(v)
            | AttributeValue::Opaque(v) => self.opaque(v),
            AttributeValue::Invalid(_) => Err(fmt::Error),
        }
    }
    fn opaque_layout(&mut self, spelling: &str, parameters: &[AttributeValue]) -> fmt::Result {
        if parameters.is_empty() {
            return self.sink.write_str(spelling);
        }
        let bytes = spelling.as_bytes();
        let mut start = 0;
        let mut parameter = 0;
        let mut index = 0;
        while index < bytes.len() {
            if matches!(bytes[index], b'#' | b'!') {
                let token_start = index;
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric()
                        || matches!(bytes[index], b'_' | b'$' | b'.' | b'-'))
                {
                    index += 1;
                }
                self.sink.write_str(&spelling[start..token_start])?;
                if let Some(value) = parameters.get(parameter) {
                    self.attribute(value)?;
                    parameter += 1;
                } else {
                    self.sink.write_str(&spelling[token_start..index])?;
                }
                start = index;
            } else {
                index += 1;
            }
        }
        self.sink.write_str(&spelling[start..])
    }
    fn affine_map(&mut self, id: crate::semantic::AffineMapId) -> fmt::Result {
        let map = self.doc.affine_map(id).ok_or(fmt::Error)?;
        self.sink.write_str("affine_map<(")?;
        self.affine_vars('d', map.dimensions)?;
        self.sink.write_char(')')?;
        if map.symbols != 0 {
            self.sink.write_char('[')?;
            self.affine_vars('s', map.symbols)?;
            self.sink.write_char(']')?;
        }
        self.sink.write_str(" -> (")?;
        for (index, &expr) in map.results.iter().enumerate() {
            if index != 0 {
                self.sink.write_str(", ")?;
            }
            self.affine_expr(expr)?;
        }
        self.sink.write_str(")>")
    }
    fn integer_set(&mut self, id: crate::semantic::IntegerSetId) -> fmt::Result {
        use crate::semantic::IntegerSetRelation::*;
        let set = self.doc.integer_set(id).ok_or(fmt::Error)?;
        self.sink.write_str("affine_set<(")?;
        self.affine_vars('d', set.dimensions)?;
        self.sink.write_char(')')?;
        if set.symbols != 0 {
            self.sink.write_char('[')?;
            self.affine_vars('s', set.symbols)?;
            self.sink.write_char(']')?;
        }
        self.sink.write_str(" : (")?;
        for (index, constraint) in set.constraints.iter().enumerate() {
            if index != 0 {
                self.sink.write_str(", ")?;
            }
            self.affine_expr(constraint.left)?;
            self.sink.write_str(match constraint.relation {
                Equal => " == ",
                GreaterEqual => " >= ",
                LessEqual => " <= ",
                Invalid(_) => return Err(fmt::Error),
            })?;
            self.affine_expr(constraint.right)?;
        }
        self.sink.write_str(")>")
    }
    fn affine_vars(&mut self, prefix: char, count: u32) -> fmt::Result {
        for index in 0..count {
            if index != 0 {
                self.sink.write_str(", ")?;
            }
            write!(self.sink, "{prefix}{index}")?;
        }
        Ok(())
    }
    fn affine_expr(&mut self, id: crate::semantic::AffineExprId) -> fmt::Result {
        self.affine_expr_prec(id, 0, false)
    }
    fn affine_expr_prec(
        &mut self,
        id: crate::semantic::AffineExprId,
        parent_precedence: u8,
        right_child: bool,
    ) -> fmt::Result {
        use crate::semantic::{AffineBinaryOperator::*, AffineExprValue::*};
        match self.doc.affine_expression(id).ok_or(fmt::Error)? {
            Dimension(i) => write!(self.sink, "d{i}"),
            Symbol(i) => write!(self.sink, "s{i}"),
            Constant(v) => write!(self.sink, "{v}"),
            Binary {
                operator,
                left,
                right,
            } => {
                if matches!(operator, Subtract)
                    && matches!(self.doc.affine_expression(*left), Some(Constant(0)))
                {
                    self.sink.write_str("- ")?;
                    return self.affine_expr_prec(*right, 3, true);
                }
                let precedence = match operator {
                    Add | Subtract => 1,
                    Multiply | FloorDiv | CeilDiv | Mod => 2,
                };
                let parentheses = precedence < parent_precedence
                    || (right_child && precedence == parent_precedence);
                if parentheses {
                    self.sink.write_char('(')?;
                }
                self.affine_expr_prec(*left, precedence, false)?;
                self.sink.write_str(match operator {
                    Add => " + ",
                    Subtract => " - ",
                    Multiply => " * ",
                    FloorDiv => " floordiv ",
                    CeilDiv => " ceildiv ",
                    Mod => " mod ",
                })?;
                self.affine_expr_prec(*right, precedence, true)?;
                if parentheses {
                    self.sink.write_char(')')?;
                }
                Ok(())
            }
            Invalid(_) => Err(fmt::Error),
        }
    }
    fn location(&mut self, value: &LocationValue) -> fmt::Result {
        match value {
            LocationValue::Unknown => self.sink.write_str("unknown"),
            LocationValue::FileLineColumn { file, line, column } => {
                write!(self.sink, "{file}:{line}:{column}")
            }
            LocationValue::Name {
                name,
                child,
                metadata,
            } => {
                self.sink.write_str(name)?;
                if let Some(child) = child {
                    self.sink.write_char('(')?;
                    self.location(child)?;
                    self.sink.write_char(')')?;
                }
                if let Some(metadata) = metadata {
                    self.sink.write_str(metadata)?;
                }
                Ok(())
            }
            LocationValue::CallSite { callee, caller } => {
                self.sink.write_str("callsite(")?;
                self.location(callee)?;
                self.sink.write_str(" at ")?;
                self.location(caller)?;
                self.sink.write_char(')')
            }
            LocationValue::Fused {
                metadata,
                locations,
            } => {
                self.sink.write_str("fused")?;
                if let Some(metadata) = metadata {
                    write!(self.sink, "<{metadata}>")?;
                }
                self.sink.write_char('[')?;
                for (index, location) in locations.iter().enumerate() {
                    if index != 0 {
                        self.sink.write_str(", ")?;
                    }
                    self.location(location)?;
                }
                self.sink.write_char(']')
            }
            LocationValue::Invalid(_) => Err(fmt::Error),
        }
    }
    fn opaque(&mut self, bytes: &[u8]) -> fmt::Result {
        self.sink
            .write_str(std::str::from_utf8(bytes).map_err(|_| fmt::Error)?)
    }
    fn newline(&mut self, indent: usize) -> fmt::Result {
        match self.layout {
            PrintLayout::Compact => self.sink.write_char(' '),
            PrintLayout::Pretty => {
                self.sink.write_char('\n')?;
                for _ in 0..indent {
                    self.sink.write_str("  ")?;
                }
                Ok(())
            }
        }
    }
}
