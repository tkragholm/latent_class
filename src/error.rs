//! Error types for Latent Class Analysis.

/// Error types for LCA operations.
#[derive(Debug, thiserror::Error)]
pub enum LcaError {
    /// Data validation error (e.g., mismatched column lengths, all-missing columns).
    #[error("invalid data: {0}")]
    InvalidData(String),

    /// Number of classes must be at least 1.
    #[error("n_classes must be >= 1, got {0}")]
    InvalidNClasses(usize),

    /// Numerical instability (e.g., NaN log-likelihood, empty classes).
    #[error("numerical instability: {0}")]
    NumericalError(String),

    /// No EM restart converged within the maximum allowed iterations.
    #[error("no restart converged within max_iter ({max_iter}) iterations")]
    NoConvergence {
        /// Maximum iterations allowed.
        max_iter: usize,
    },

    /// I/O error during data export.
    #[error("I/O error writing {path}: {source}")]
    Io {
        /// File path.
        path: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
}
