//! Data structures for Latent Class Analysis.
//!
//! This module provides the [`IndicatorMatrix`] which is the primary container
//! for categorical (binary) indicator data used in LCA. It handles missing values
//! using a memory-efficient bitmask and provides utilities for data validation.

use crate::error::LcaError;
use faer::Mat;
use serde::{Deserialize, Serialize};

/// Encoding for a single cell in the indicator matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndicatorValue {
    /// Observed value 0.
    Zero,
    /// Observed value 1.
    One,
    /// Observation exists but value is unknown.
    /// Excluded from sufficient statistics for that indicator only (FIML approach).
    Missing,
}

/// A validated N × J indicator matrix (row = individual, column = variable).
///
/// The source of truth is a raw `Vec<u8>` (N × J, values 0/1/255) paired with a
/// packed bitmask indicating missingness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorMatrix {
    /// Raw data: N x J, values 0, 1, or 255 for Missing.
    data: Vec<u8>,
    /// Packed bitmask indicating missingness (1 bit per cell).
    /// bit == 1 means missing.
    missing_mask: Vec<u64>,
    n_individuals: usize,
    n_indicators: usize,
    variable_names: Vec<String>,
    /// Optional human-readable labels for indicators.
    variable_labels: Option<Vec<String>>,
    ids: Option<Vec<String>>,
}

impl IndicatorMatrix {
    /// Construct from named columns with optional missing values.
    /// `None` encodes a missing observation.
    ///
    /// # Errors
    /// Returns [`LcaError::InvalidData`] if columns have mismatched lengths or if
    /// any column is 100% missing.
    pub fn from_columns(columns: Vec<(String, Vec<Option<u8>>)>) -> Result<Self, LcaError> {
        Self::from_columns_with_ids(columns, Vec::new())
    }

    /// Construct from named columns with external IDs for downstream joins.
    ///
    /// # Errors
    /// Returns [`LcaError::InvalidData`] if ID count does not match row count.
    pub fn from_columns_with_ids(
        columns: Vec<(String, Vec<Option<u8>>)>,
        ids: Vec<String>,
    ) -> Result<Self, LcaError> {
        if columns.is_empty() {
            return Err(LcaError::InvalidData("no columns provided".to_string()));
        }

        let n_indicators = columns.len();
        let n_individuals = columns[0].1.len();

        if !ids.is_empty() && ids.len() != n_individuals {
            return Err(LcaError::InvalidData(format!(
                "ID count ({}) does not match individual count ({})",
                ids.len(),
                n_individuals
            )));
        }

        for (name, col) in &columns {
            if col.len() != n_individuals {
                return Err(LcaError::InvalidData(format!(
                    "Column '{}' has length {}, expected {}",
                    name,
                    col.len(),
                    n_individuals
                )));
            }
            if col.iter().all(std::option::Option::is_none) {
                return Err(LcaError::InvalidData(format!(
                    "Column '{name}' is 100% missing"
                )));
            }
        }

        let mut data = vec![0u8; n_individuals * n_indicators];
        let n_cells = n_individuals * n_indicators;
        let mut missing_mask = vec![0u64; n_cells.div_ceil(64)];

        for (j, (name, col)) in columns.iter().enumerate() {
            for (i, val) in col.iter().enumerate() {
                let idx = i * n_indicators + j;
                match val {
                    Some(0) => data[idx] = 0,
                    Some(1) => data[idx] = 1,
                    Some(v) => {
                        return Err(LcaError::InvalidData(format!(
                            "Invalid indicator value {v} in column '{name}' at row {i}"
                        )));
                    }
                    None => {
                        data[idx] = 255;
                        let mask_idx = idx / 64;
                        let bit_idx = idx % 64;
                        missing_mask[mask_idx] |= 1 << bit_idx;
                    }
                }
            }
        }

        Ok(Self {
            data,
            missing_mask,
            n_individuals,
            n_indicators,
            variable_names: columns.into_iter().map(|(n, _)| n).collect(),
            variable_labels: None,
            ids: if ids.is_empty() { None } else { Some(ids) },
        })
    }

    /// Set human-readable labels for the indicators.
    ///
    /// # Errors
    /// Returns [`LcaError::InvalidData`] if label count does not match indicator count.
    pub fn with_variable_labels(mut self, labels: Vec<String>) -> Result<Self, LcaError> {
        if labels.len() != self.n_indicators {
            return Err(LcaError::InvalidData(format!(
                "Label count ({}) does not match indicator count ({})",
                labels.len(),
                self.n_indicators
            )));
        }
        self.variable_labels = Some(labels);
        Ok(self)
    }

    /// Number of individuals (rows).
    #[must_use]
    pub const fn n_individuals(&self) -> usize {
        self.n_individuals
    }

    /// Number of indicators (columns).
    #[must_use]
    pub const fn n_indicators(&self) -> usize {
        self.n_indicators
    }

    /// Names of the indicator variables.
    #[must_use]
    pub fn variable_names(&self) -> &[String] {
        &self.variable_names
    }

    /// Optional human-readable labels for indicators.
    #[must_use]
    pub fn variable_labels(&self) -> Option<&[String]> {
        self.variable_labels.as_deref()
    }

    /// Get label for indicator at index `j`, falling back to variable name.
    #[must_use]
    pub fn indicator_label(&self, j: usize) -> &str {
        self.variable_labels
            .as_ref()
            .map_or_else(|| &self.variable_names[j], |labels| &labels[j])
    }

    /// Optional external IDs for individuals.
    #[must_use]
    pub fn ids(&self) -> Option<&[String]> {
        self.ids.as_deref()
    }

    /// Per-variable missing rate (0.0 – 1.0).
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "dataset cardinalities are converted to rates for diagnostics/reporting"
    )]
    pub fn missing_rates(&self) -> Vec<f64> {
        let mut rates = vec![0.0; self.n_indicators];
        for (j, rate) in rates.iter_mut().enumerate().take(self.n_indicators) {
            let mut missing_count = 0;
            for i in 0..self.n_individuals {
                if self.is_missing(i, j) {
                    missing_count += 1;
                }
            }
            *rate = f64::from(missing_count) / self.n_individuals as f64;
        }
        rates
    }

    /// Per-variable prevalence of value=1 among non-missing rows.
    #[must_use]
    pub fn prevalences(&self) -> Vec<f64> {
        let mut prevs = vec![0.0; self.n_indicators];
        for (j, prev) in prevs.iter_mut().enumerate().take(self.n_indicators) {
            let mut count_one = 0;
            let mut count_total = 0;
            for i in 0..self.n_individuals {
                if !self.is_missing(i, j) {
                    count_total += 1;
                    if self.data[i * self.n_indicators + j] == 1 {
                        count_one += 1;
                    }
                }
            }
            *prev = if count_total > 0 {
                f64::from(count_one) / f64::from(count_total)
            } else {
                0.0
            };
        }
        prevs
    }

    /// Returns true if the cell (i, j) is missing.
    #[inline]
    #[must_use]
    pub fn is_missing(&self, i: usize, j: usize) -> bool {
        let idx = i * self.n_indicators + j;
        let mask_idx = idx / 64;
        let bit_idx = idx % 64;
        (self.missing_mask[mask_idx] & (1 << bit_idx)) != 0
    }

    /// Returns a `faer::Mat<f64>` work buffer.
    /// Missing cells are stored as 0.0 and must be consulted via `is_missing`.
    #[must_use]
    pub fn to_faer_work_buffer(&self) -> Mat<f64> {
        let mut mat = Mat::zeros(self.n_individuals, self.n_indicators);
        for i in 0..self.n_individuals {
            for j in 0..self.n_indicators {
                let val = self.data[i * self.n_indicators + j];
                if val != 255 {
                    mat[(i, j)] = f64::from(val);
                }
            }
        }
        mat
    }

    /// Raw data getter (internal use)
    pub(crate) fn data(&self) -> &[u8] {
        &self.data
    }
}
