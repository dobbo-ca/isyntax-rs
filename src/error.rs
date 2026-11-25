use std::{fmt, io};

/// One variant per Go error (the mapping is in `docs/parity.md`). The string a variant carries
/// is for people; callers match on the variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `context.Canceled`: the cancel flag was set.
    Cancelled,
    /// An error from the caller's reader, as its kind (`isyntax: reading ...: %w`).
    Io(io::ErrorKind),
    /// `isyntax: XML header: %w`: the XML header does not parse.
    Xml(String),
    /// The header or a table in it is malformed (every other `fmt.Errorf` of `internal/container`).
    Header(String),
    /// `ErrUnsupported`: a compressor, bit depth, colour plane name or codeblock size this crate
    /// does not decode, or a file without a measured SDK truncation table.
    Unsupported(String),
    /// A codeblock does not decode (`isyntax: codeblock at N: hulsken: ...`).
    Codeblock(String),
    /// `ErrMemoryLimit`: other decodes hold the memory this one needs. Transient: retry.
    MemoryLimit,
    /// `errTooBig`: the call needs more than the whole `max_alloc` (the parse, or one tile's
    /// working memory). Not transient.
    AllocLimit,
    /// The image has more pixels than `Limits::max_pixels`.
    PixelLimit,
    /// The XML header is longer than `Limits::max_header` (Go: 256 MiB).
    HeaderLimit,
    /// The allocator refused a size that is within the limits.
    OutOfMemory,
    /// A bad level, destination or stride (Go returns an error; the Go decoder panics nowhere).
    InvalidArgument(String),
    /// A worker thread panicked (Go: `isyntax: internal error: panic`).
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("isyntax: cancelled"),
            Self::Io(kind) => write!(f, "isyntax: read error: {kind}"),
            Self::Xml(s) => write!(f, "isyntax: XML header: {s}"),
            Self::Header(s) => write!(f, "isyntax: {s}"),
            Self::Unsupported(s) => write!(f, "isyntax: unsupported file: {s}"),
            Self::Codeblock(s) => write!(f, "isyntax: codeblock: {s}"),
            Self::MemoryLimit => {
                f.write_str("isyntax: memory limit held by other decodes (transient, retry later)")
            }
            Self::AllocLimit => f.write_str("isyntax: decode needs more than the memory limit"),
            Self::PixelLimit => f.write_str("isyntax: image has more pixels than the limit"),
            Self::HeaderLimit => f.write_str("isyntax: XML header over the limit"),
            Self::OutOfMemory => f.write_str("isyntax: out of memory"),
            Self::InvalidArgument(s) => write!(f, "isyntax: invalid argument: {s}"),
            Self::Internal(s) => write!(f, "isyntax: internal error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e.kind())
    }
}

pub(crate) fn header(s: impl Into<String>) -> Error {
    Error::Header(s.into())
}

pub(crate) fn codeblock(s: impl Into<String>) -> Error {
    Error::Codeblock(s.into())
}

pub(crate) fn check_cancel(cancel: &std::sync::atomic::AtomicBool) -> Result<(), Error> {
    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
