//! Posterior class probabilities and assignments.

use crate::data::IndicatorMatrix;
use crate::diagnostics::{DiagnosticsConfig, FitDiagnostics};
use crate::fit::LcaFit;
use faer::Mat;

/// Precomputed posterior probabilities for a fitted model on a given dataset.
///
/// Use this handle to perform assignments and compute diagnostics without
/// redundant recomputation of the posterior matrix.
pub struct ComputedPosteriors {
    pub(crate) posteriors: Mat<f64>, // N × K
}

impl ComputedPosteriors {
    /// Create a new handle from a raw posterior matrix.
    #[must_use]
    pub const fn new(posteriors: Mat<f64>) -> Self {
        Self { posteriors }
    }

    /// The raw N × K posterior matrix.
    #[must_use]
    pub const fn matrix(&self) -> &Mat<f64> {
        &self.posteriors
    }

    /// Hard MAP assignment (argmax of each row).
    #[must_use]
    pub fn assign(&self) -> Vec<usize> {
        let n = self.posteriors.nrows();
        let k = self.posteriors.ncols();
        let mut assignment = Vec::with_capacity(n);

        for i in 0..n {
            let mut max_p = -1.0;
            let mut max_k = 0;
            for kk in 0..k {
                let p = self.posteriors[(i, kk)];
                if p > max_p {
                    max_p = p;
                    max_k = kk;
                }
            }
            assignment.push(max_k);
        }
        assignment
    }

    /// Hard assignment with cutoff; returns `None` for individuals whose
    /// maximum posterior probability is below `cutoff`.
    #[must_use]
    pub fn assign_with_cutoff(&self, cutoff: f64) -> Vec<Option<usize>> {
        let n = self.posteriors.nrows();
        let k = self.posteriors.ncols();
        let mut assignment = Vec::with_capacity(n);

        for i in 0..n {
            let mut max_p = -1.0;
            let mut max_k = 0;
            for kk in 0..k {
                let p = self.posteriors[(i, kk)];
                if p > max_p {
                    max_p = p;
                    max_k = kk;
                }
            }
            if max_p >= cutoff {
                assignment.push(Some(max_k));
            } else {
                assignment.push(None);
            }
        }
        assignment
    }

    /// Average posterior entropy per individual.
    /// Normalized entropy: -1/N * `Σ_i` `Σ_k` `r_ik` `ln(r_ik)`.
    /// Lower values indicate more certain class assignments.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "mean entropy divides by sample size as floating-point"
    )]
    pub fn mean_entropy(&self) -> f64 {
        let n = self.posteriors.nrows();
        let k = self.posteriors.ncols();
        let mut total_entropy = 0.0;

        for i in 0..n {
            let mut ind_entropy = 0.0;
            for kk in 0..k {
                let p = self.posteriors[(i, kk)];
                if p > 1e-10 {
                    ind_entropy = p.mul_add(-p.ln(), ind_entropy);
                }
            }
            total_entropy += ind_entropy;
        }
        total_entropy / n as f64
    }

    /// Proportion of individuals whose max posterior exceeds 0.7.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "purity is reported as a floating-point fraction over sample size"
    )]
    pub fn modal_assignment_purity(&self) -> f64 {
        let n = self.posteriors.nrows();
        let k = self.posteriors.ncols();
        let mut pure_count = 0;

        for i in 0..n {
            let mut max_p = 0.0;
            for kk in 0..k {
                let p = self.posteriors[(i, kk)];
                if p > max_p {
                    max_p = p;
                }
            }
            if max_p > 0.7 {
                pure_count += 1;
            }
        }
        f64::from(pure_count) / n as f64
    }

    /// Compute diagnostics using these cached posteriors.
    #[must_use]
    pub fn diagnostics(&self, fit: &LcaFit, data: &IndicatorMatrix) -> FitDiagnostics {
        FitDiagnostics::new(fit, data, self, DiagnosticsConfig::default())
    }

    /// Compute diagnostics using custom thresholds.
    #[must_use]
    pub fn diagnostics_with_config(
        &self,
        fit: &LcaFit,
        data: &IndicatorMatrix,
        config: DiagnosticsConfig,
    ) -> FitDiagnostics {
        FitDiagnostics::new(fit, data, self, config)
    }

    /// Compute posterior-weighted means for a target outcome vector.
    ///
    /// For each class `k`, the weighted mean of outcome `Y` is:
    /// `\bar{Y}_k = \frac{\sum_i r_{ik} Y_i}{\sum_i r_{ik}}`.
    ///
    /// # Panics
    /// Panics if `outcomes.len()` does not match the number of individuals.
    #[must_use]
    pub fn weighted_means(&self, outcomes: &[f64]) -> Vec<f64> {
        let n = self.posteriors.nrows();
        let k = self.posteriors.ncols();
        assert_eq!(outcomes.len(), n, "Outcome vector length mismatch");

        let mut class_sums = vec![0.0; k];
        let mut class_weights = vec![0.0; k];

        for (i, y) in outcomes.iter().enumerate().take(n) {
            for kk in 0..k {
                let w = self.posteriors[(i, kk)];
                class_sums[kk] = w.mul_add(*y, class_sums[kk]);
                class_weights[kk] += w;
            }
        }

        class_sums
            .into_iter()
            .zip(class_weights)
            .map(|(sum, weight)| if weight > 1e-10 { sum / weight } else { 0.0 })
            .collect()
    }
}
