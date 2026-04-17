//! EM Initialization strategies.

use crate::data::IndicatorMatrix;
use faer::Mat;
use rand::prelude::*;
use rand_distr::{Beta, Distribution};

/// Create random initial parameters for EM.
///
/// Draws π from a symmetric Uniform(0.1, 1.0) and normalizes to 1.
/// Draws each ρ_{jk} from Beta(2,2) to bias away from 0/1.
///
/// # Panics
///
/// Panics if `Beta::new(2.0, 2.0)` fails to construct, which should not
/// happen for these valid positive shape parameters.
#[must_use]
pub fn random_init(n_classes: usize, n_indicators: usize, seed: u64) -> (Vec<f64>, Mat<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. π: class prevalences (sum to 1)
    let mut pi: Vec<f64> = (0..n_classes).map(|_| rng.random_range(0.1..1.0)).collect();
    let sum_pi: f64 = pi.iter().sum();
    for p in &mut pi {
        *p /= sum_pi;
    }

    // 2. ρ: item-response probabilities (K x J matrix)
    let beta = Beta::new(2.0, 2.0).unwrap();
    let mut rho = Mat::zeros(n_classes, n_indicators);
    for k in 0..n_classes {
        for j in 0..n_indicators {
            rho[(k, j)] = beta.sample(&mut rng);
        }
    }

    (pi, rho)
}

/// Create structured initial parameters using a K-modes pass.
///
/// This provides a deterministic (given seed) and often superior starting point
/// compared to pure random initialization by first finding coarse clusters
/// in the binary space.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "initial class prevalences are normalized as floating-point fractions"
)]
pub fn k_modes_init(data: &IndicatorMatrix, n_classes: usize, seed: u64) -> (Vec<f64>, Mat<f64>) {
    if n_classes == 1 {
        return random_init(1, data.n_indicators(), seed);
    }

    let n = data.n_individuals();
    let j_count = data.n_indicators();
    let raw_data = data.data();
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Initial modes: pick K random individuals as starting modes
    let mut modes = vec![vec![0u8; j_count]; n_classes];
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    for k in 0..n_classes {
        let idx = indices[k];
        for j in 0..j_count {
            modes[k][j] = if data.is_missing(idx, j) {
                rng.random_range(0..=1)
            } else {
                raw_data[idx * j_count + j]
            };
        }
    }

    // 2. Run a few iterations of K-modes
    let mut assignments = vec![0usize; n];
    for _iter in 0..5 {
        // Assignment step
        for i in 0..n {
            let mut min_dist = j_count + 1;
            let mut best_k = 0;
            for (k, mode) in modes.iter().enumerate().take(n_classes) {
                let mut dist = 0;
                for j in 0..j_count {
                    if !data.is_missing(i, j) && raw_data[i * j_count + j] != mode[j] {
                        dist += 1;
                    }
                }
                if dist < min_dist {
                    min_dist = dist;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
        }

        // Update step (calculate mode for each class)
        for (k, mode) in modes.iter_mut().enumerate().take(n_classes) {
            for j in 0..j_count {
                let mut count_one = 0;
                let mut count_total = 0;
                for (i, assigned) in assignments.iter().enumerate().take(n) {
                    if *assigned == k && !data.is_missing(i, j) {
                        count_total += 1;
                        if raw_data[i * j_count + j] == 1 {
                            count_one += 1;
                        }
                    }
                }
                if count_total > 0 {
                    mode[j] = u8::from(count_one > count_total / 2);
                }
            }
        }
    }

    // 3. Convert final clusters to π and ρ
    let mut pi = vec![0.0; n_classes];
    let mut rho = Mat::zeros(n_classes, j_count);
    let eps = 1e-2; // Slightly more aggressive smoothing for init

    for k in 0..n_classes {
        let mut count_k = 0;
        for assigned in assignments.iter().take(n) {
            if *assigned == k {
                count_k += 1;
            }
        }
        pi[k] = (f64::from(count_k) + 1.0) / (n as f64 + n_classes as f64);

        for j in 0..j_count {
            let mut count_one = 0;
            let mut count_total = 0;
            for i in 0..n {
                if assignments[i] == k && !data.is_missing(i, j) {
                    count_total += 1;
                    if raw_data[i * j_count + j] == 1 {
                        count_one += 1;
                    }
                }
            }
            rho[(k, j)] =
                (f64::from(count_one) + eps) / 2.0f64.mul_add(eps, f64::from(count_total));
        }
    }

    (pi, rho)
}
