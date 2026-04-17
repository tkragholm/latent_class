//! Pre-fit indicator quality checks.
//!
//! This module provides the [`IndicatorPrecheck`] which implements the
//! data-quality recommendations from the clinical LCA guide, such as
//! collinearity detection and low-prevalence warnings.

use crate::data::IndicatorMatrix;

/// Summary of pre-fit indicator checks.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PrecheckReport {
    /// Variable pairs with Pearson |r| > threshold.
    /// Indicates potential violation of the local independence assumption.
    pub high_correlation_pairs: Vec<(String, String, f64)>,

    /// Variables with prevalence < threshold.
    /// Low-prevalence indicators can cause unstable class extraction.
    pub low_prevalence_vars: Vec<(String, f64)>,

    /// Variables with missing rate > threshold.
    /// Fit statistics lose accuracy above 20% missingness.
    pub high_missing_vars: Vec<(String, f64)>,

    /// N < 300 warning.
    pub small_sample_warning: Option<SmallSampleWarning>,
}

/// Warning for small sample sizes where fit statistics may be unreliable.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SmallSampleWarning {
    /// Sample size.
    pub n: usize,
    /// Warning message.
    pub message: &'static str,
}

/// Configurable thresholds for pre-fit indicator checks.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndicatorPrecheck {
    /// Threshold for Pearson |r| (default: 0.5).
    pub correlation_threshold: f64,
    /// Threshold for indicator prevalence (default: 0.10).
    pub prevalence_threshold: f64,
    /// Threshold for missing-value rate (default: 0.20).
    pub missing_threshold: f64,
}

impl Default for IndicatorPrecheck {
    fn default() -> Self {
        Self {
            correlation_threshold: 0.5,
            prevalence_threshold: 0.10,
            missing_threshold: 0.20,
        }
    }
}

impl IndicatorPrecheck {
    /// Create a new precheck with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Run all checks on the given data and return a report.
    #[must_use]
    pub fn run(&self, data: &IndicatorMatrix) -> PrecheckReport {
        let n = data.n_individuals();
        let j_count = data.n_indicators();
        let names = data.variable_names();

        // 1. Small sample warning
        let small_sample_warning = if n < 300 {
            Some(SmallSampleWarning {
                n,
                message: "N < 300: BIC and LMR tests have reduced accuracy. Consider Monte Carlo power simulation before interpreting class number selection.",
            })
        } else {
            None
        };

        // 2. Missing rates
        let missing_rates = data.missing_rates();
        let mut high_missing_vars = Vec::new();
        for (j, &rate) in missing_rates.iter().enumerate() {
            if rate > self.missing_threshold {
                high_missing_vars.push((names[j].clone(), rate));
            }
        }

        // 3. Prevalences
        let prevalences = data.prevalences();
        let mut low_prevalence_vars = Vec::new();
        for (j, &prev) in prevalences.iter().enumerate() {
            if prev < self.prevalence_threshold {
                low_prevalence_vars.push((names[j].clone(), prev));
            }
        }

        // 4. Pairwise Pearson correlation
        let mut high_correlation_pairs = Vec::new();
        for j1 in 0..j_count {
            for j2 in (j1 + 1)..j_count {
                if let Some(corr) = compute_pairwise_pearson(data, j1, j2)
                    && corr.abs() > self.correlation_threshold
                {
                    high_correlation_pairs.push((names[j1].clone(), names[j2].clone(), corr));
                }
            }
        }

        PrecheckReport {
            high_correlation_pairs,
            low_prevalence_vars,
            high_missing_vars,
            small_sample_warning,
        }
    }
}

/// Compute Pearson correlation using pairwise complete observations.
fn compute_pairwise_pearson(data: &IndicatorMatrix, j1: usize, j2: usize) -> Option<f64> {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_cross = 0.0;
    let mut count = 0;

    let n_inds = data.n_individuals();
    let n_vars = data.n_indicators();
    let raw_data = data.data();

    for i in 0..n_inds {
        if !data.is_missing(i, j1) && !data.is_missing(i, j2) {
            let x = f64::from(raw_data[i * n_vars + j1]);
            let y = f64::from(raw_data[i * n_vars + j2]);

            sum_x += x;
            sum_y += y;
            sum_xx = x.mul_add(x, sum_xx);
            sum_yy = y.mul_add(y, sum_yy);
            sum_cross = x.mul_add(y, sum_cross);
            count += 1;
        }
    }

    if count < 2 {
        return None;
    }

    let n = f64::from(count);
    let numerator = n * sum_cross - sum_x * sum_y;
    let denominator =
        (sum_x.mul_add(-sum_x, n * sum_xx) * sum_y.mul_add(-sum_y, n * sum_yy)).sqrt();

    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}
