//! LCA Model specification and builder.
//!
//! This module provides the [`LcaModel`] builder for configuring and
//! executing the EM estimation process.

use crate::data::IndicatorMatrix;
use crate::em::run_em;
use crate::error::LcaError;
use crate::fit::LcaFit;
use crate::init::{k_modes_init, random_init};
use epistat_rng::replicate_seed;
use rayon::prelude::*;
use std::hash::{BuildHasher, RandomState};

/// A seed from the operating system's entropy, for fits without a fixed seed.
fn entropy_seed() -> u64 {
    RandomState::new().hash_one(0_u64)
}

/// Shared EM hyperparameters.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmConfig {
    /// Maximum number of EM iterations (default: 500).
    pub max_iter: usize,
    /// Convergence tolerance for relative log-likelihood change (default: 1e-6).
    pub tol: f64,
    /// Number of random restarts to guard against local optima (default: 50).
    pub n_restarts: usize,
    /// Minimum required replications of the maximum LL (default: 20).
    pub ll_replication_target: usize,
    /// Seed for reproducible restarts.
    pub seed: Option<u64>,
}

impl Default for EmConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            n_restarts: 50,
            ll_replication_target: 20,
            seed: None,
        }
    }
}

/// Builder for an LCA model with K classes.
pub struct LcaModel {
    /// Number of latent classes.
    pub n_classes: usize,
    /// EM configuration.
    pub config: EmConfig,
}

impl LcaModel {
    /// Create a model specification for `n_classes`.
    ///
    /// # Errors
    /// Returns [`LcaError::InvalidNClasses`] if `n_classes` is 0.
    pub fn new(n_classes: usize) -> Result<Self, LcaError> {
        if n_classes == 0 {
            return Err(LcaError::InvalidNClasses(0));
        }
        Ok(Self {
            n_classes,
            config: EmConfig::default(),
        })
    }

    /// Set the EM configuration.
    #[must_use]
    pub const fn with_config(mut self, config: EmConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub const fn with_tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the number of restarts.
    #[must_use]
    pub const fn with_n_restarts(mut self, n: usize) -> Self {
        self.config.n_restarts = n;
        self
    }

    /// Set the seed for reproducible restarts.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Run EM estimation. Parallel across restarts via rayon.
    ///
    /// Returns the best fit across all restarts. Callers should check
    /// [`LcaFit::converged`] to ensure the result reached the tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`LcaError`] when a non-recoverable restart fails or when all
    /// restarts are numerically degenerate.
    pub fn fit(&self, data: &IndicatorMatrix) -> Result<LcaFit, LcaError> {
        let n_restarts = if self.n_classes == 1 {
            1
        } else {
            self.config.n_restarts
        };
        let base_seed = self.config.seed.unwrap_or_else(entropy_seed);

        // Run restarts in parallel
        let results: Vec<_> = (0..n_restarts)
            .into_par_iter()
            .map(|i| {
                let seed = replicate_seed(base_seed, i as u64);
                // First restart (index 0) uses structured K-modes init if seed is present
                // OR if it's the only restart.
                let (pi_init, rho_init) = if i == 0 {
                    k_modes_init(data, self.n_classes, seed)
                } else {
                    random_init(self.n_classes, data.n_indicators(), seed)
                };
                run_em(data, self.n_classes, pi_init, rho_init, &self.config)
            })
            .collect();

        let mut best_fit: Option<LcaFit> = None;
        let mut ll_replications = 0;

        for res in results {
            match res {
                Ok(em_res) => {
                    let mut current_fit = LcaFit {
                        pi: em_res.pi,
                        rho: em_res.rho,
                        log_likelihood: em_res.log_likelihood,
                        n_iter: em_res.n_iter,
                        converged: em_res.converged,
                        ll_replications: 1,
                        variable_names: data.variable_names().to_vec(),
                        variable_labels: data
                            .variable_labels()
                            .map(<[std::string::String]>::to_vec),
                    };
                    current_fit.canonicalize();

                    if let Some(ref best) = best_fit {
                        if current_fit.log_likelihood > best.log_likelihood + self.config.tol {
                            best_fit = Some(current_fit);
                            ll_replications = 1;
                        } else if (current_fit.log_likelihood - best.log_likelihood).abs()
                            < self.config.tol
                        {
                            ll_replications += 1;
                        }
                    } else {
                        best_fit = Some(current_fit);
                        ll_replications = 1;
                    }
                }
                Err(LcaError::NumericalError(_)) => {}
                Err(e) => return Err(e),
            }
        }

        let mut fit = best_fit.ok_or_else(|| {
            LcaError::NumericalError("All restarts produced degenerate results".to_string())
        })?;

        fit.ll_replications = ll_replications;
        Ok(fit)
    }

    /// Run EM estimation (strict mode).
    ///
    /// # Errors
    /// Returns [`LcaError::NoConvergence`] if no restart converged within `max_iter`.
    pub fn fit_strict(&self, data: &IndicatorMatrix) -> Result<LcaFit, LcaError> {
        let fit = self.fit(data)?;
        if !fit.converged {
            return Err(LcaError::NoConvergence {
                max_iter: self.config.max_iter,
            });
        }
        Ok(fit)
    }
}
