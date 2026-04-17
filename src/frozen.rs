//! Out-of-sample scoring with frozen parameters.
//!
//! This module provides the [`FrozenLcaModel`] which carries only the model
//! parameters ($\pi$, $\rho$) and variable names, allowing it to score new
//! data without carrying the original training dataset.

use crate::data::IndicatorMatrix;
use crate::error::LcaError;
use crate::estep::{EStepWorkspace, compute_posteriors_with_workspace};
use faer::Mat;
use serde::{Deserialize, Serialize};

/// A parameter-only LCA model for out-of-sample scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenLcaModel {
    pub(crate) pi: Vec<f64>,
    #[serde(with = "crate::mat_serde")]
    pub(crate) rho: Mat<f64>,
    pub(crate) variable_names: Vec<String>,
    pub(crate) variable_labels: Option<Vec<String>>,
}

impl FrozenLcaModel {
    /// Create a new frozen model from parameters.
    #[must_use]
    pub const fn new(
        pi: Vec<f64>,
        rho: Mat<f64>,
        variable_names: Vec<String>,
        variable_labels: Option<Vec<String>>,
    ) -> Self {
        Self {
            pi,
            rho,
            variable_names,
            variable_labels,
        }
    }

    /// Compute posterior class probabilities for new data.
    ///
    /// # Errors
    /// Returns [`LcaError::InvalidData`] if the new data schema does not match.
    pub fn score_posteriors(&self, new_data: &IndicatorMatrix) -> Result<Mat<f64>, LcaError> {
        if new_data.n_indicators() != self.variable_names.len() {
            return Err(LcaError::InvalidData(format!(
                "New data has {} indicators, expected {}",
                new_data.n_indicators(),
                self.variable_names.len()
            )));
        }

        for (i, name) in self.variable_names.iter().enumerate() {
            if name != &new_data.variable_names()[i] {
                return Err(LcaError::InvalidData(format!(
                    "Indicator name mismatch at index {}: expected '{}', got '{}'",
                    i,
                    name,
                    new_data.variable_names()[i]
                )));
            }
        }

        let n = new_data.n_individuals();
        let k = self.n_classes();
        let j_count = self.variable_names.len();
        let mut posteriors = Mat::<f64>::zeros(n, k);
        let mut workspace = EStepWorkspace::new(k, j_count);
        let _ = compute_posteriors_with_workspace(
            new_data,
            &self.pi,
            &self.rho,
            &mut posteriors,
            &mut workspace,
        );

        Ok(posteriors)
    }

    /// Hard MAP assignment on new data.
    ///
    /// # Errors
    /// Returns [`LcaError::InvalidData`] if the new data schema does not match.
    pub fn score_assign(&self, new_data: &IndicatorMatrix) -> Result<Vec<usize>, LcaError> {
        let posteriors = self.score_posteriors(new_data)?;
        let n = posteriors.nrows();
        let k = posteriors.ncols();
        let mut assignment = Vec::with_capacity(n);

        for i in 0..n {
            let mut max_p = -1.0;
            let mut max_k = 0;
            for kk in 0..k {
                let p = posteriors[(i, kk)];
                if p > max_p {
                    max_p = p;
                    max_k = kk;
                }
            }
            assignment.push(max_k);
        }
        Ok(assignment)
    }

    /// Number of latent classes.
    #[must_use]
    pub const fn n_classes(&self) -> usize {
        self.pi.len()
    }

    /// Names of indicator variables.
    #[must_use]
    pub fn variable_names(&self) -> &[String] {
        &self.variable_names
    }

    /// Labels of indicator variables.
    #[must_use]
    pub fn variable_labels(&self) -> Option<&[String]> {
        self.variable_labels.as_deref()
    }

    /// Class prevalences.
    #[must_use]
    pub fn class_prevalences(&self) -> &[f64] {
        &self.pi
    }

    /// Item-response probabilities.
    #[must_use]
    pub const fn item_probs(&self) -> &Mat<f64> {
        &self.rho
    }
}
