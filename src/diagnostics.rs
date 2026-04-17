//! Post-fit diagnostic checks.
//!
//! This module implements automated quality checks for fitted LCA models,
//! such as small-class detection, Salsa-effect (parallel profiles),
//! and local independence violations.

use crate::data::IndicatorMatrix;
use crate::fit::LcaFit;
use crate::posterior::ComputedPosteriors;

/// Configurable thresholds for post-fit diagnostics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiagnosticsConfig {
    /// Minimum class prevalence for 2-class models (default: 0.15).
    pub small_class_threshold_2class: f64,
    /// Minimum class prevalence for 3+ class models (default: 0.10).
    pub small_class_threshold_3plus: f64,
    /// Kendall's W threshold for salsa-effect detection (default: 0.85).
    pub salsa_w_threshold: f64,
    /// Minimum required LL replications (default: 20).
    pub ll_replication_target: usize,
    /// Within-class correlation threshold for local independence check (default: 0.5).
    pub within_class_correlation_threshold: f64,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            small_class_threshold_2class: 0.15,
            small_class_threshold_3plus: 0.10,
            salsa_w_threshold: 0.85,
            ll_replication_target: 20,
            within_class_correlation_threshold: 0.5,
        }
    }
}

/// Report of automated diagnostic checks for a fitted model.
pub struct FitDiagnostics {
    /// Classes smaller than the guide's minimum thresholds.
    pub small_classes: Vec<SmallClassWarning>,

    /// Whether the item-response profiles suggest a "Salsa effect": all K class
    /// profiles are approximately parallel.
    pub salsa_effect_detected: bool,

    /// Whether the maximum log-likelihood was replicated fewer than the target number.
    pub insufficient_ll_replications: bool,
    /// Number of replications found.
    pub ll_replications: usize,
    /// Replication target.
    pub ll_replication_target: usize,

    /// Within-class correlation warnings (local independence violations).
    pub within_class_dependence: Vec<WithinClassDependenceWarning>,

    /// The thresholds used to produce this report.
    pub config: DiagnosticsConfig,
}

/// Warning for a class that is too small to be reliable.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SmallClassWarning {
    /// Index of the class.
    pub class_index: usize,
    /// Estimated prevalence.
    pub prevalence: f64,
    /// Minimum threshold used.
    pub threshold: f64,
    /// Description message.
    pub message: String,
}

/// Warning for local independence violation within a class.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WithinClassDependenceWarning {
    /// Index of the class.
    pub class_index: usize,
    /// Name of first variable.
    pub var_a: String,
    /// Name of second variable.
    pub var_b: String,
    /// Posterior-weighted correlation.
    pub correlation: f64,
}

impl FitDiagnostics {
    /// Run all diagnostic checks for a fit.
    #[must_use]
    pub fn new(
        fit: &LcaFit,
        data: &IndicatorMatrix,
        posteriors: &ComputedPosteriors,
        config: DiagnosticsConfig,
    ) -> Self {
        // 1. Small classes
        let mut small_classes = Vec::new();
        let k = fit.n_classes();
        let pi = fit.class_prevalences();
        let threshold = if k == 2 {
            config.small_class_threshold_2class
        } else {
            config.small_class_threshold_3plus
        };

        for (i, &p) in pi.iter().enumerate() {
            if p < threshold {
                small_classes.push(SmallClassWarning {
                    class_index: i,
                    prevalence: p,
                    threshold,
                    message: format!("Class {i} has prevalence {p:.3} < threshold {threshold:.3}"),
                });
            }
        }

        // 2. LL Replications
        let insufficient_ll_replications = fit.ll_replications() < config.ll_replication_target;

        // 3. Salsa effect
        let salsa_effect_detected = detect_salsa_effect(fit, config.salsa_w_threshold);

        // 4. Within-class dependence
        let within_class_dependence = detect_within_class_dependence(
            fit,
            data,
            posteriors,
            config.within_class_correlation_threshold,
        );

        Self {
            small_classes,
            salsa_effect_detected,
            insufficient_ll_replications,
            ll_replications: fit.ll_replications(),
            ll_replication_target: config.ll_replication_target,
            within_class_dependence,
            config,
        }
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "rank-based diagnostic converts counts to floating-point test statistic"
)]
fn detect_salsa_effect(fit: &LcaFit, threshold: f64) -> bool {
    let n_classes = fit.n_classes();
    if n_classes <= 1 {
        return false;
    }
    let n_indicators = fit.n_indicators();
    let rho = fit.item_probs();

    let mut indicator_ranks = Vec::new();
    for j in 0..n_indicators {
        let mut class_probs: Vec<(usize, f64)> =
            (0..n_classes).map(|kk| (kk, rho[(kk, j)])).collect();
        class_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; n_classes];
        for (rank, &(kk, _)) in class_probs.iter().enumerate() {
            ranks[kk] = rank as f64 + 1.0;
        }
        indicator_ranks.push(ranks);
    }

    let mut rank_sums = vec![0.0; n_classes];
    for ranks in indicator_ranks.iter().take(n_indicators) {
        for kk in 0..n_classes {
            rank_sums[kk] += ranks[kk];
        }
    }

    let mean_rank_sum = rank_sums.iter().sum::<f64>() / n_classes as f64;
    let rank_sum_sq_dev = rank_sums
        .iter()
        .map(|&rank_sum| (rank_sum - mean_rank_sum).powi(2))
        .sum::<f64>();

    let m = n_indicators as f64;
    let n = n_classes as f64;

    let denominator = m.powi(2) * (n.powi(3) - n);
    if denominator == 0.0 {
        return false;
    }

    let kendall_w = (12.0 * rank_sum_sq_dev) / denominator;
    kendall_w > threshold
}

fn detect_within_class_dependence(
    fit: &LcaFit,
    data: &IndicatorMatrix,
    posteriors: &ComputedPosteriors,
    threshold: f64,
) -> Vec<WithinClassDependenceWarning> {
    let k = fit.n_classes();
    let j_count = fit.n_indicators();
    let names = fit.variable_names();
    let mut warnings = Vec::new();

    for kk in 0..k {
        for j1 in 0..j_count {
            for j2 in (j1 + 1)..j_count {
                if let Some(corr) =
                    compute_posterior_weighted_correlation(data, posteriors, kk, j1, j2)
                    && corr.abs() > threshold
                {
                    warnings.push(WithinClassDependenceWarning {
                        class_index: kk,
                        var_a: names[j1].clone(),
                        var_b: names[j2].clone(),
                        correlation: corr,
                    });
                }
            }
        }
    }
    warnings
}

fn compute_posterior_weighted_correlation(
    data: &IndicatorMatrix,
    posteriors: &ComputedPosteriors,
    k_idx: usize,
    j1: usize,
    j2: usize,
) -> Option<f64> {
    let n = data.n_individuals();
    let raw_data = data.data();
    let n_vars = data.n_indicators();
    let post_matrix = posteriors.matrix();

    let mut weight_sum = 0.0;
    let mut sum_feature_a = 0.0;
    let mut sum_feature_b = 0.0;
    let mut sum_sq_feature_a = 0.0;
    let mut sum_sq_feature_b = 0.0;
    let mut sum_cross_features = 0.0;

    for i in 0..n {
        if !data.is_missing(i, j1) && !data.is_missing(i, j2) {
            let weight = post_matrix[(i, k_idx)];
            let x = f64::from(raw_data[i * n_vars + j1]);
            let y = f64::from(raw_data[i * n_vars + j2]);

            weight_sum += weight;
            sum_feature_a = weight.mul_add(x, sum_feature_a);
            sum_feature_b = weight.mul_add(y, sum_feature_b);
            sum_sq_feature_a = (weight * x).mul_add(x, sum_sq_feature_a);
            sum_sq_feature_b = (weight * y).mul_add(y, sum_sq_feature_b);
            sum_cross_features = (weight * x).mul_add(y, sum_cross_features);
        }
    }

    if weight_sum < 10.0 {
        return None;
    }

    let mean_x = sum_feature_a / weight_sum;
    let mean_y = sum_feature_b / weight_sum;

    let cov_xy = (sum_cross_features / weight_sum) - (mean_x * mean_y);
    let var_x = mean_x.mul_add(-mean_x, sum_sq_feature_a / weight_sum);
    let var_y = mean_y.mul_add(-mean_y, sum_sq_feature_b / weight_sum);

    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }

    let corr = cov_xy / (var_x.sqrt() * var_y.sqrt());
    Some(corr)
}
