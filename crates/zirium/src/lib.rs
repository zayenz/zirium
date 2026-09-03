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
//! This example edits a hybrid document, commits the transaction, writes
//! source-preserving output, and then shows what happens to an erased handle:
//!
//! ```
//! use zirium::{
//!     dialect::DialectRegistry,
//!     parser::ParsedFile,
//!     printer::PrintLayout,
//!     semantic::{
//!         AttributeSpec, AttributeValue, EditError, LoweringMode, RetentionProfile,
//!         lower_with_dialect_registry_and_retention,
//!     },
//! };
//!
//! let source = b"\"keep\"() : () -> ()\n\"edit\"() : () -> ()\n";
//! let parsed = ParsedFile::parse(source.as_slice())?;
//! let registry = DialectRegistry::EMPTY;
//! let lowered = lower_with_dialect_registry_and_retention(
//!     &parsed,
//!     LoweringMode::Strict,
//!     RetentionProfile::Hybrid,
//!     &registry,
//! );
//! let mut document = lowered.document.expect("strict lowering succeeds");
//! document.verify_semantics(&registry)?;
//! let edited = document.root_operations()[1];
//!
//! let mut transaction = document.edit(&registry)?;
//! transaction.set_attribute(
//!     edited,
//!     AttributeSpec {
//!         name: "tag".into(),
//!         spelling: "\"checked\"".into(),
//!         value: AttributeValue::String("\"checked\"".into()),
//!     },
//! )?;
//! transaction.commit()?;
//! let output = document.preserving_bytes(PrintLayout::Pretty)?;
//! assert!(String::from_utf8(output)?.contains("tag = \"checked\""));
//!
//! let mut transaction = document.edit(&registry)?;
//! transaction.erase(edited)?;
//! transaction.commit()?;
//! assert!(matches!(
//!     document.check_operation(edited),
//!     Err(EditError::StaleOperation(_))
//! ));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod dialect;
pub mod lexer;
pub mod parser;
pub mod printer;
pub mod query;
mod representation;
pub mod semantic;
pub mod source;

pub use representation::{
    CompactError, CompletedMarker, Event, EventBuilder, Marker, NodeId, SyntaxElement, SyntaxKind,
    SyntaxTree,
};
