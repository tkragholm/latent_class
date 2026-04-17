//! Results of a successful LCA fit.

use crate::data::IndicatorMatrix;
use crate::diagnostics::{DiagnosticsConfig, FitDiagnostics};
use crate::estep::{EStepWorkspace, compute_posteriors_with_workspace};
use crate::frozen::FrozenLcaModel;
use crate::posterior::ComputedPosteriors;
use crate::selection::ModelSelectionStats;
use faer::Mat;
use serde::{Deserialize, Serialize};

/// Result of an LCA fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LcaFit {
    pub(crate) pi: Vec<f64>,
    #[serde(with = "crate::mat_serde")]
    pub(crate) rho: Mat<f64>,
    pub(crate) log_likelihood: f64,
    pub(crate) n_iter: usize,
    pub(crate) converged: bool,
    pub(crate) ll_replications: usize,
    pub(crate) variable_names: Vec<String>,
    pub(crate) variable_labels: Option<Vec<String>>,
}

/// Human-readable summary of a single latent class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassSummary {
    /// Canonical index of the class.
    pub class_index: usize,
    /// Estimated prevalence in the population.
    pub prevalence: f64,
    /// Item-response probabilities, sorted by probability (descending).
    /// Tuple: (Label/Name, Probability)
    pub item_response_probs: Vec<(String, f64)>,
}

impl LcaFit {
    /// Estimated class prevalences π (length K), ordered by canonical ordering.
    #[must_use]
    pub fn class_prevalences(&self) -> &[f64] {
        &self.pi
    }

    /// Item-response probabilities ρ (K × J matrix, row = class, col = indicator).
    #[must_use]
    pub const fn item_probs(&self) -> &Mat<f64> {
        &self.rho
    }

    #[must_use]
    pub const fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    #[must_use]
    pub const fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Whether the EM reached the convergence tolerance.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.converged
    }

    /// Number of restarts that reached the same maximum log-likelihood.
    #[must_use]
    pub const fn ll_replications(&self) -> usize {
        self.ll_replications
    }

    /// Number of latent classes.
    #[must_use]
    pub const fn n_classes(&self) -> usize {
        self.pi.len()
    }

    /// Number of indicator variables.
    #[must_use]
    pub fn n_indicators(&self) -> usize {
        self.rho.ncols()
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

    /// Internal helper to get N.
    pub(crate) const fn n_individuals_internal(data: &IndicatorMatrix) -> usize {
        data.n_individuals()
    }

    /// Model selection and fit-quality statistics.
    #[must_use]
    pub fn selection_stats(&self, data: &IndicatorMatrix) -> ModelSelectionStats {
        ModelSelectionStats::new(self, data)
    }

    /// Fit diagnostics: small-class warnings, salsa-effect detection, LL replication check.
    #[must_use]
    pub fn diagnostics(&self, data: &IndicatorMatrix) -> FitDiagnostics {
        let posteriors = self.compute_posteriors(data);
        posteriors.diagnostics(self, data)
    }

    /// Fit diagnostics with custom thresholds.
    #[must_use]
    pub fn diagnostics_with_config(
        &self,
        data: &IndicatorMatrix,
        config: DiagnosticsConfig,
    ) -> FitDiagnostics {
        let posteriors = self.compute_posteriors(data);
        posteriors.diagnostics_with_config(self, data, config)
    }

    /// Extract a frozen parameter-only model for out-of-sample scoring.
    #[must_use]
    pub fn freeze(&self) -> FrozenLcaModel {
        FrozenLcaModel::new(
            self.pi.clone(),
            self.rho.clone(),
            self.variable_names.clone(),
            self.variable_labels.clone(),
        )
    }

    /// Compute posterior class membership probabilities: N × K matrix.
    #[must_use]
    pub fn posteriors(&self, data: &IndicatorMatrix) -> Mat<f64> {
        let n = data.n_individuals();
        let k = self.n_classes();
        let j_count = self.n_indicators();
        let mut posteriors = Mat::<f64>::zeros(n, k);
        let mut workspace = EStepWorkspace::new(k, j_count);
        let _ = compute_posteriors_with_workspace(
            data,
            &self.pi,
            &self.rho,
            &mut posteriors,
            &mut workspace,
        );

        posteriors
    }

    /// Compute posteriors once and return a reusable handle.
    #[must_use]
    pub fn compute_posteriors(&self, data: &IndicatorMatrix) -> ComputedPosteriors {
        ComputedPosteriors::new(self.posteriors(data))
    }

    /// Hard MAP assignment (argmax of posterior row).
    #[must_use]
    pub fn assign(&self, data: &IndicatorMatrix) -> Vec<usize> {
        self.compute_posteriors(data).assign()
    }

    /// Hard assignment with probability cutoff.
    #[must_use]
    pub fn assign_with_cutoff(&self, data: &IndicatorMatrix, cutoff: f64) -> Vec<Option<usize>> {
        self.compute_posteriors(data).assign_with_cutoff(cutoff)
    }

    /// Compute posterior-weighted means for a target outcome vector.
    #[must_use]
    pub fn weighted_means(&self, data: &IndicatorMatrix, outcomes: &[f64]) -> Vec<f64> {
        self.compute_posteriors(data).weighted_means(outcomes)
    }

    /// Generate human-readable summaries for all classes.
    #[must_use]
    pub fn class_summaries(&self) -> Vec<ClassSummary> {
        let k = self.n_classes();
        let j_count = self.n_indicators();
        let mut summaries = Vec::with_capacity(k);

        for kk in 0..k {
            let mut item_probs = Vec::with_capacity(j_count);
            for j in 0..j_count {
                let label = self.variable_labels.as_ref().map_or_else(
                    || self.variable_names[j].clone(),
                    |labels| labels[j].clone(),
                );
                item_probs.push((label, self.rho[(kk, j)]));
            }
            // Sort items by probability descending
            item_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            summaries.push(ClassSummary {
                class_index: kk,
                prevalence: self.pi[kk],
                item_response_probs: item_probs,
            });
        }
        summaries
    }

    /// Number of free parameters (for BIC/AIC/SABIC).
    #[must_use]
    pub fn n_params(&self) -> usize {
        let k = self.n_classes();
        let j = self.n_indicators();
        if k == 1 { j } else { (k - 1) + k * j }
    }

    /// Canonical class ordering (descending prevalence).
    pub(crate) fn canonicalize(&mut self) {
        let k = self.n_classes();
        if k <= 1 {
            return;
        }

        let mut indices: Vec<usize> = (0..k).collect();
        // Sort by descending prevalence, then by first item prob if ties
        indices.sort_by(|&a, &b| {
            self.pi[b]
                .partial_cmp(&self.pi[a])
                .unwrap()
                .then_with(|| self.rho[(b, 0)].partial_cmp(&self.rho[(a, 0)]).unwrap())
        });

        let mut new_pi = Vec::with_capacity(k);
        let mut new_rho = Mat::zeros(k, self.n_indicators());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            new_pi.push(self.pi[old_idx]);
            for j in 0..self.n_indicators() {
                new_rho[(new_idx, j)] = self.rho[(old_idx, j)];
            }
        }

        self.pi = new_pi;
        self.rho = new_rho;
    }

    /// Compare class profiles between this fit and another model.
    ///
    /// Returns a $`K_1` \times `K_2`$ matrix of Cosine Similarities between
    /// the item-response probability vectors of each class.
    ///
    /// # Errors
    /// Returns [`LcaError::InvalidData`] if the indicator variables do not match.
    pub fn compare_profiles(&self, other: &Self) -> Result<Mat<f64>, crate::error::LcaError> {
        if self.variable_names != other.variable_names {
            return Err(crate::error::LcaError::InvalidData(
                "Cannot compare models with different indicators".to_string(),
            ));
        }

        let k1 = self.n_classes();
        let k2 = other.n_classes();
        let mut sim_mat = Mat::zeros(k1, k2);

        for i in 0..k1 {
            let row1 = self.rho.row(i);
            let norm1 = row1.iter().map(|&x| x * x).sum::<f64>().sqrt();

            for j in 0..k2 {
                let row2 = other.rho.row(j);
                let norm2 = row2.iter().map(|&x| x * x).sum::<f64>().sqrt();

                let dot = row1
                    .iter()
                    .zip(row2.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();

                sim_mat[(i, j)] = if norm1 > 0.0 && norm2 > 0.0 {
                    dot / (norm1 * norm2)
                } else {
                    0.0
                };
            }
        }

        Ok(sim_mat)
    }
}
