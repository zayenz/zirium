//! Lossless MLIR syntax, semantic processing, editing, and output.
//!
//! The usual flow is to parse an owning [`parser::ParsedFile`], lower it to a
//! [`semantic::Document`], validate or verify it, optionally edit it, and then
//! choose an output mode:
//!
//! - [`parser::ParsedFile::write_original`] reproduces the input bytes exactly.
//! - [`semantic::Document::write_canonical`](crate::semantic::Document::write_canonical)
//!   emits deterministic generic MLIR from semantic storage.
//! - [`semantic::Document::write_preserving`](crate::semantic::Document::write_preserving)
//!   reuses unchanged source and regenerates edited regions; it requires
//!   [`semantic::RetentionProfile::Hybrid`].
//!
//! [`semantic::RetentionProfile`] controls whether lowering retains syntax,
//! semantic storage, or both. [`semantic::LoweringMode::Strict`] returns no
//! document when lowering is incomplete, while `BestEffort` returns a document
//! containing invalid sentinels together with diagnostics. Structural
//! [`semantic::Document::validate_structure`] checks storage invariants;
//! [`semantic::Document::verify_semantics`] additionally runs registered
//! dialect schemas and verifiers.
//!
//! ```
//! use zirium::{
//!     dialect::DialectRegistry,
//!     parser::ParsedFile,
//!     printer::PrintLayout,
//!     semantic::{LoweringMode, RetentionProfile, lower_with_dialect_registry_and_retention},
//! };
//!
//! let parsed = ParsedFile::parse(b"\"example\"() : () -> ()".as_slice())?;
//! let lowered = lower_with_dialect_registry_and_retention(
//!     &parsed,
//!     LoweringMode::Strict,
//!     RetentionProfile::SemanticOnly,
//!     &DialectRegistry::EMPTY,
//! );
//! let document = lowered.document.expect("strict lowering succeeded");
//! document.validate_structure()?;
//! let bytes = document.canonical_bytes(PrintLayout::Pretty)?;
//! assert!(!bytes.is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod dialect;
pub mod lexer;
pub mod parser;
pub mod printer;
mod representation;
pub mod semantic;
pub mod source;

pub use representation::{
    CompactError, CompletedMarker, Event, EventBuilder, Marker, NodeId, SyntaxElement, SyntaxKind,
    SyntaxTree,
};
