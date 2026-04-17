//! Core EM algorithm implementation.

use crate::data::IndicatorMatrix;
use crate::error::LcaError;
use crate::estep::{EStepWorkspace, compute_posteriors_with_workspace};
use crate::model::EmConfig;
use cgp::prelude::*;
use faer::Mat;

#[cgp_component(EStep)]
pub trait CanPerformEStep {
    fn e_step(
        &self,
        data: &IndicatorMatrix,
        pi: &[f64],
        rho: &Mat<f64>,
        posteriors: &mut Mat<f64>,
    ) -> f64;
}

#[cgp_component(MStep)]
pub trait CanPerformMStep {
    fn m_step(
        &self,
        data: &IndicatorMatrix,
        posteriors: &Mat<f64>,
        pi: &mut [f64],
        rho: &mut Mat<f64>,
    );
}

#[cgp_component(ConvergencePolicy)]
pub trait HasConvergencePolicy {
    fn has_converged(&self, iter: usize, prev_ll: f64, current_ll: f64) -> bool;
}

/// Result of a single EM restart.
pub struct EmResult {
    /// Class prevalences.
    pub pi: Vec<f64>,
    /// Item-response probabilities.
    pub rho: Mat<f64>,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether convergence was reached.
    pub converged: bool,
}

#[cgp_component(EmConfigProvider)]
pub trait HasEmConfig {
    fn config(&self) -> &EmConfig;
}

pub struct EmAlgorithmContext<'a> {
    pub config: &'a EmConfig,
}

impl HasField<Symbol!("config")> for EmAlgorithmContext<'_> {
    type Value = EmConfig;
    fn get_field(&self, _tag: PhantomData<Symbol!("config")>) -> &EmConfig {
        self.config
    }
}

impl HasEmConfig for EmAlgorithmContext<'_> {
    fn config(&self) -> &EmConfig {
        self.config
    }
}

pub struct LcaEStep;

#[cgp_impl(LcaEStep)]
impl<Context> EStep for Context {
    fn e_step(
        &self,
        data: &IndicatorMatrix,
        pi: &[f64],
        rho: &Mat<f64>,
        posteriors: &mut Mat<f64>,
    ) -> f64 {
        e_step(data, pi, rho, posteriors)
    }
}

pub struct LcaMStep;

#[cgp_impl(LcaMStep)]
impl<Context> MStep for Context {
    fn m_step(
        &self,
        data: &IndicatorMatrix,
        posteriors: &Mat<f64>,
        pi: &mut [f64],
        rho: &mut Mat<f64>,
    ) {
        m_step(data, pi.len(), posteriors, pi, rho);
    }
}

pub struct LcaConvergencePolicy;

#[cgp_impl(LcaConvergencePolicy)]
impl<Context> ConvergencePolicy for Context
where
    Context: HasEmConfig,
{
    fn has_converged(&self, iter: usize, prev_ll: f64, current_ll: f64) -> bool {
        if iter == 0 {
            return false;
        }
        let config = self.config();
        let rel_change = (current_ll - prev_ll).abs() / prev_ll.abs().max(1.0);
        rel_change < config.tol
    }
}

delegate_components! {
    <'a> EmAlgorithmContext<'a> {
        EStepComponent: LcaEStep,
        MStepComponent: LcaMStep,
        ConvergencePolicyComponent: LcaConvergencePolicy,
    }
}

/// Run a single EM restart.
///
/// # Errors
/// Returns [`LcaError::NumericalError`] if LL becomes NaN or if a class becomes empty.
pub fn run_em(
    data: &IndicatorMatrix,
    k: usize,
    mut pi: Vec<f64>,
    mut rho: Mat<f64>,
    config: &EmConfig,
) -> Result<EmResult, LcaError> {
    if k == 1 {
        return Ok(run_null_model(data));
    }

    let context = EmAlgorithmContext { config };
    let n = data.n_individuals();
    let j_count = data.n_indicators();

    let mut posteriors = Mat::<f64>::zeros(n, k);
    let mut e_step_workspace = EStepWorkspace::new(k, j_count);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // E-step
        let current_ll = compute_posteriors_with_workspace(
            data,
            &pi,
            &rho,
            &mut posteriors,
            &mut e_step_workspace,
        );

        if current_ll.is_nan() {
            return Err(LcaError::NumericalError(
                "NaN log-likelihood in E-step".to_string(),
            ));
        }

        // M-step
        context.m_step(data, &posteriors, &mut pi, &mut rho);

        // Check for empty classes
        for &p in &pi {
            if p < 1e-10 {
                return Err(LcaError::NumericalError("Empty class detected".to_string()));
            }
        }

        // Convergence check
        if context.has_converged(iter, prev_ll, current_ll) {
            converged = true;
            prev_ll = current_ll;
            break;
        }
        prev_ll = current_ll;
    }

    Ok(EmResult {
        pi,
        rho,
        log_likelihood: prev_ll,
        n_iter,
        converged,
    })
}

fn e_step(data: &IndicatorMatrix, pi: &[f64], rho: &Mat<f64>, posteriors: &mut Mat<f64>) -> f64 {
    let mut workspace = EStepWorkspace::new(pi.len(), data.n_indicators());
    compute_posteriors_with_workspace(data, pi, rho, posteriors, &mut workspace)
}

fn m_step(
    data: &IndicatorMatrix,
    k: usize,
    posteriors: &Mat<f64>,
    pi: &mut [f64],
    rho: &mut Mat<f64>,
) {
    let n = data.n_individuals();
    let j_count = data.n_indicators();
    let raw_data = data.data();
    let eps = 1e-8; // Smoothing to avoid 0/1

    // Update π
    #[expect(
        clippy::cast_precision_loss,
        reason = "class prevalence update converts sample size to floating-point probability"
    )]
    for kk in 0..k {
        let mut sum_post = 0.0;
        for i in 0..n {
            sum_post += posteriors[(i, kk)];
        }
        pi[kk] = sum_post / n as f64;
    }

    // Update ρ
    for kk in 0..k {
        for j in 0..j_count {
            let mut weighted_sum_x = 0.0;
            let mut weighted_sum_total = 0.0;
            for i in 0..n {
                if !data.is_missing(i, j) {
                    let w = posteriors[(i, kk)];
                    weighted_sum_total += w;
                    if raw_data[i * j_count + j] == 1 {
                        weighted_sum_x += w;
                    }
                }
            }
            rho[(kk, j)] = (weighted_sum_x + eps) / 2.0f64.mul_add(eps, weighted_sum_total);
        }
    }
}

fn run_null_model(data: &IndicatorMatrix) -> EmResult {
    let n = data.n_individuals();
    let j_count = data.n_indicators();
    let raw_data = data.data();

    let pi = vec![1.0];
    let mut rho = Mat::zeros(1, j_count);
    let prevs = data.prevalences();
    for j in 0..j_count {
        rho[(0, j)] = prevs[j].clamp(1e-8, 1.0 - 1e-8);
    }

    let log_rho: Vec<f64> = (0..j_count).map(|j| rho[(0, j)].ln()).collect();
    let log_1_minus_rho: Vec<f64> = (0..j_count).map(|j| (1.0 - rho[(0, j)]).ln()).collect();

    let mut total_ll = 0.0;
    for i in 0..n {
        let mut ind_ll = 0.0;
        for j in 0..j_count {
            if !data.is_missing(i, j) {
                let val = raw_data[i * j_count + j];
                if val == 1 {
                    ind_ll += log_rho[j];
                } else {
                    ind_ll += log_1_minus_rho[j];
                }
            }
        }
        total_ll += ind_ll;
    }

    EmResult {
        pi,
        rho,
        log_likelihood: total_ll,
        n_iter: 1,
        converged: true,
    }
}

#[cfg(test)]
mod wiring_checks {
    use super::{ConvergencePolicyComponent, EStepComponent, EmAlgorithmContext, MStepComponent};
    use cgp::prelude::{CanUseComponent, check_components};

    check_components! {
        #[check_trait(CheckEmAlgorithmContextProviders)]
        EmAlgorithmContext<'static> {
            EStepComponent,
            MStepComponent,
            ConvergencePolicyComponent,
        }
    }
}
