//! Model selection statistics.
//!
//! This module provides the [`ModelSelectionStats`] structure for comparing
//! models with different numbers of classes ($K$) using information criteria.

use crate::data::IndicatorMatrix;
use crate::diagnostics::DiagnosticsConfig;
use crate::error::LcaError;
use crate::fit::LcaFit;
use crate::model::{EmConfig, LcaModel};
use serde::{Deserialize, Serialize};

/// Comparative statistics for model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionStats {
    /// Number of latent classes.
    pub n_classes: usize,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of free parameters in the model.
    pub n_params: usize,
    /// Number of individuals in the sample.
    pub n_individuals: usize,

    /// Akaike Information Criterion: $-2 \cdot LL + 2p$.
    pub aic: f64,

    /// Bayesian Information Criterion: $-2 \cdot LL + p \cdot \ln(N)$.
    pub bic: f64,

    /// Sample-size Adjusted BIC (SABIC): $-2 \cdot LL + p \cdot \ln((N+2)/24)$.
    pub sabic: f64,

    /// Average posterior entropy per individual.
    pub mean_posterior_entropy: f64,

    /// Proportion of individuals whose maximum posterior probability exceeds 0.7.
    pub modal_assignment_purity: f64,

    /// Number of restarts that replicated the maximum log-likelihood.
    pub ll_replications: usize,

    /// Whether the EM converged for this model.
    pub converged: bool,
}

impl ModelSelectionStats {
    /// Calculate stats for a fitted model.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "information criteria require floating-point sample/parameter counts"
    )]
    pub fn new(fit: &LcaFit, data: &IndicatorMatrix) -> Self {
        let n = LcaFit::n_individuals_internal(data);
        let p = fit.n_params();
        let ll = fit.log_likelihood();
        let n_f64 = n as f64;
        let p_f64 = p as f64;

        let posteriors = fit.compute_posteriors(data);

        Self {
            n_classes: fit.n_classes(),
            log_likelihood: ll,
            n_params: p,
            n_individuals: n,
            aic: (-2.0f64).mul_add(ll, 2.0 * p_f64),
            bic: (-2.0f64).mul_add(ll, p_f64 * n_f64.ln()),
            sabic: (-2.0f64).mul_add(ll, p_f64 * ((n_f64 + 2.0) / 24.0).ln()),
            mean_posterior_entropy: posteriors.mean_entropy(),
            modal_assignment_purity: posteriors.modal_assignment_purity(),
            ll_replications: fit.ll_replications(),
            converged: fit.converged(),
        }
    }
}

/// A collection of fitted models across a range of $K$ values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LcaSelectionTable {
    /// Models and their statistics.
    pub entries: Vec<(usize, ModelSelectionStats, LcaFit)>,
}

impl LcaSelectionTable {
    /// Create a new selection table.
    #[must_use]
    pub const fn new(entries: Vec<(usize, ModelSelectionStats, LcaFit)>) -> Self {
        Self { entries }
    }

    /// Returns the model with the absolute minimum BIC.
    #[must_use]
    pub fn best_by_bic(&self) -> Option<&LcaFit> {
        self.entries
            .iter()
            .filter(|(_, s, _)| s.converged && s.bic.is_finite())
            .min_by(|a, b| a.1.bic.total_cmp(&b.1.bic))
            .map(|(_, _, f)| f)
    }

    /// Identify the "BIC Elbow" candidate.
    ///
    /// Defined as the smallest $K$ that is within 10 BIC units of the minimum BIC.
    /// This is a common heuristic for avoiding over-extraction in large samples.
    pub fn bic_elbow(&self) -> Option<&LcaFit> {
        let min_bic = self
            .entries
            .iter()
            .filter(|(_, s, _)| s.converged && s.bic.is_finite())
            .map(|(_, s, _)| s.bic)
            .fold(f64::INFINITY, f64::min);

        if !min_bic.is_finite() {
            return None;
        }

        self.entries
            .iter()
            .filter(|(_, s, _)| s.converged && s.bic <= min_bic + 10.0)
            .min_by_key(|(k, _, _)| *k)
            .map(|(_, _, f)| f)
    }

    /// Filter for models that pass all diagnostic gates.
    #[must_use]
    pub fn viable_models(
        &self,
        data: &IndicatorMatrix,
        config: &DiagnosticsConfig,
    ) -> Vec<&LcaFit> {
        self.entries
            .iter()
            .filter(|(_, _, f)| {
                let diag = f.diagnostics_with_config(data, config.clone());
                f.converged()
                    && !diag.salsa_effect_detected
                    && !diag.insufficient_ll_replications
                    && diag.small_classes.is_empty()
            })
            .map(|(_, _, f)| f)
            .collect()
    }
}

/// Fit models for K = `k_min..=k_max` and return a selection table.
///
/// # Errors
/// Returns [`LcaError`] if any fit fails or if data is invalid.
pub fn select_n_classes(
    data: &IndicatorMatrix,
    k_min: usize,
    k_max: usize,
    config: &EmConfig,
) -> Result<LcaSelectionTable, LcaError> {
    let mut entries = Vec::new();

    for k in k_min..=k_max {
        let model = LcaModel::new(k)?.with_config(config.clone());
        let fit = model.fit(data)?;
        let stats = fit.selection_stats(data);
        entries.push((k, stats, fit));
    }

    Ok(LcaSelectionTable::new(entries))
}
