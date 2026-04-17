//! Shared E-step posterior computation with reusable workspace.

use crate::data::IndicatorMatrix;
use faer::Mat;

/// Reusable buffers for E-step-style posterior computation.
pub struct EStepWorkspace {
    log_post: Vec<f64>,
    log_pi: Vec<f64>,
    observed_indices: Vec<usize>,
    log_rho: Mat<f64>,
    log_1_minus_rho: Mat<f64>,
}

impl EStepWorkspace {
    #[must_use]
    pub(crate) fn new(k: usize, j_count: usize) -> Self {
        Self {
            log_post: vec![0.0; k],
            log_pi: vec![0.0; k],
            observed_indices: Vec::with_capacity(j_count),
            log_rho: Mat::zeros(k, j_count),
            log_1_minus_rho: Mat::zeros(k, j_count),
        }
    }

    fn ensure_dims(&mut self, k: usize, j_count: usize) {
        if self.log_post.len() != k {
            self.log_post.resize(k, 0.0);
        }
        if self.log_pi.len() != k {
            self.log_pi.resize(k, 0.0);
        }
        if self.log_rho.nrows() != k || self.log_rho.ncols() != j_count {
            self.log_rho = Mat::zeros(k, j_count);
        }
        if self.log_1_minus_rho.nrows() != k || self.log_1_minus_rho.ncols() != j_count {
            self.log_1_minus_rho = Mat::zeros(k, j_count);
        }
    }
}

/// Compute posterior class probabilities in-place and return total log-likelihood.
pub fn compute_posteriors_with_workspace(
    data: &IndicatorMatrix,
    pi: &[f64],
    rho: &Mat<f64>,
    posteriors: &mut Mat<f64>,
    workspace: &mut EStepWorkspace,
) -> f64 {
    let n = data.n_individuals();
    let k = pi.len();
    let j_count = data.n_indicators();
    let raw_data = data.data();
    let mut total_ll = 0.0;

    workspace.ensure_dims(k, j_count);
    for (idx, value) in pi.iter().copied().enumerate() {
        workspace.log_pi[idx] = value.ln();
    }
    for kk in 0..k {
        for jj in 0..j_count {
            let r = rho[(kk, jj)];
            workspace.log_rho[(kk, jj)] = r.ln();
            workspace.log_1_minus_rho[(kk, jj)] = (1.0 - r).ln();
        }
    }

    for i in 0..n {
        workspace.observed_indices.clear();
        for j in 0..j_count {
            if !data.is_missing(i, j) {
                workspace.observed_indices.push(j);
            }
        }

        workspace.log_post.copy_from_slice(&workspace.log_pi);

        for &j in &workspace.observed_indices {
            let val = raw_data[i * j_count + j];
            for kk in 0..k {
                workspace.log_post[kk] += if val == 1 {
                    workspace.log_rho[(kk, j)]
                } else {
                    workspace.log_1_minus_rho[(kk, j)]
                };
            }
        }

        let max_log = workspace
            .log_post
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f64 = workspace
            .log_post
            .iter()
            .map(|&lp| (lp - max_log).exp())
            .sum();
        let log_sum = max_log + sum_exp.ln();
        total_ll += log_sum;

        for kk in 0..k {
            posteriors[(i, kk)] = (workspace.log_post[kk] - log_sum).exp();
        }
    }

    total_ll
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn estep_invariants(
            pi_val in 0.01..0.99f64,
            rho_val in 0.01..0.99f64,
            indicator_val in 0..2u8
        ) {
            // Minimal setup to check core math logic
            let cols = vec![("v1".to_string(), vec![Some(indicator_val)])];
            let matrix = IndicatorMatrix::from_columns(cols).unwrap();
            let k = 2;
            let pi = vec![pi_val, 1.0 - pi_val];
            let rho = Mat::from_fn(k, 1, |_, _| rho_val);
            let mut posteriors = Mat::zeros(1, k);
            let mut workspace = EStepWorkspace::new(k, 1);

            let ll = compute_posteriors_with_workspace(&matrix, &pi, &rho, &mut posteriors, &mut workspace);

            prop_assert!(ll.is_finite());
            let post_sum: f64 = (0..k).map(|kk| posteriors[(0, kk)]).sum();
            prop_assert!((post_sum - 1.0).abs() < 1e-12);
        }
    }
}
