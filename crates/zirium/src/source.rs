//! Owned source bytes and checked byte ranges.
//!
//! Offsets use `u32`, which keeps syntax storage compact and limits one source
//! buffer to `u32::MAX` bytes.

use std::ops::Range;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextRange {
    start: u32,
    end: u32,
}

impl TextRange {
    pub fn new(start: u32, end: u32) -> Option<Self> {
        (start <= end).then_some(Self { start, end })
    }
    pub fn at(offset: u32) -> Self {
        Self {
            start: offset,
            end: offset,
        }
    }
    pub fn start(self) -> u32 {
        self.start
    }
    pub fn end(self) -> u32 {
        self.end
    }
    pub fn len(self) -> u32 {
        self.end - self.start
    }
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
    pub fn as_range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

impl std::fmt::Display for TextRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceError {
    TooLarge { len: usize },
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { len } => write!(f, "source length {len} exceeds the 4 GiB limit"),
        }
    }
}

impl std::error::Error for SourceError {}

#[derive(Debug)]
pub struct Source {
    bytes: Arc<[u8]>,
    line_starts: OnceLock<Vec<u32>>,
}

impl Source {
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Result<Self, SourceError> {
        let bytes = bytes.into();
        if u32::try_from(bytes.len()).is_err() {
            return Err(SourceError::TooLarge { len: bytes.len() });
        }
        Ok(Self {
            bytes,
            line_starts: OnceLock::new(),
        })
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(crate) fn shared_bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }
    pub fn len(&self) -> u32 {
        self.bytes.len() as u32
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
    pub fn slice(&self, range: TextRange) -> Option<&[u8]> {
        self.bytes.get(range.as_range())
    }
    pub fn line_index_is_built(&self) -> bool {
        self.line_starts.get().is_some()
    }
    pub fn line(&self, offset: u32) -> Option<usize> {
        if offset > self.len() {
            return None;
        }
        let starts = self.line_starts.get_or_init(|| {
            let mut starts = vec![0];
            let mut index = 0;
            while index < self.bytes.len() {
                match self.bytes[index] {
                    b'\r' => {
                        index += 1;
                        if self.bytes.get(index) == Some(&b'\n') {
                            index += 1;
                        }
                        starts.push(index as u32);
                    }
                    b'\n' => {
                        index += 1;
                        starts.push(index as u32);
                    }
                    _ => index += 1,
                }
            }
            starts
        });
        Some(
            starts
                .partition_point(|start| *start <= offset)
                .saturating_sub(1),
        )
    }
}
