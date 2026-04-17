//! Latent Class Analysis (LCA) for categorical registry data.
//!
//! This crate provides a high-performance, parallel implementation of
//! Latent Class Analysis with EM estimation, specifically tailored for
//! binary indicator data from administrative registers.
//!
//! # Features
//!
//! - **FIML-style missing data:** Handles missing values per-observation,
//!   retaining individuals with partial data.
//! - **Parallel Restarts:** Uses `rayon` to execute random restarts in
//!   parallel, ensuring global maximum recovery without performance penalty.
//! - **Comprehensive Diagnostics:** Implements clinical guide recommendations
//!   for small-class detection, Salsa-effect (parallel profiles), and
//!   local independence checks.
//! - **Model Selection:** Integrated calculation of BIC, SABIC, AIC,
//!   entropy, and classification purity.
//! - **Out-of-sample Scoring:** Supports "Frozen" models for applying fitted
//!   parameters to new datasets.
//!
//! # Example
//!
//! ```rust
//! use latent_class::{IndicatorMatrix, LcaModel, EmConfig};
//!
//! // 1. Prepare data
//! let cols = vec![
//!     ("v1".to_string(), vec![Some(1), Some(0), Some(1)]),
//!     ("v2".to_string(), vec![Some(1), Some(0), None]),
//! ];
//! let matrix = IndicatorMatrix::from_columns(cols).unwrap();
//!
//! // 2. Configure and fit model
//! let model = LcaModel::new(2).unwrap()
//!     .with_n_restarts(10)
//!     .with_seed(42);
//!
//! let fit = model.fit(&matrix).unwrap();
//!
//! // 3. Inspect results
//! println!("Best LL: {}", fit.log_likelihood());
//! let stats = fit.selection_stats(&matrix);
//! println!("BIC: {}", stats.bic);
//!
//! // 4. Assign classes
//! let posteriors = fit.compute_posteriors(&matrix);
//! let assignments = posteriors.assign();
//! ```
pub mod data;
pub mod diagnostics;
pub mod em;
pub mod error;
pub(crate) mod estep;
pub mod export;
pub mod fit;
pub mod frozen;
pub mod init;
pub(crate) mod mat_serde;
pub mod model;
pub mod posterior;
pub mod precheck;
pub mod selection;

pub use data::{IndicatorMatrix, IndicatorValue};
pub use diagnostics::{DiagnosticsConfig, FitDiagnostics};
pub use error::LcaError;
pub use export::{
    write_diagnostics_csv, write_params_csv, write_posteriors_csv, write_selection_stats_csv,
};
pub use fit::{ClassSummary, LcaFit};
pub use frozen::FrozenLcaModel;
pub use model::{EmConfig, LcaModel};
pub use posterior::ComputedPosteriors;
pub use precheck::{IndicatorPrecheck, PrecheckReport};
pub use selection::{LcaSelectionTable, ModelSelectionStats, select_n_classes};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::*;

    #[fixture]
    fn simple_matrix() -> IndicatorMatrix {
        let cols = vec![
            ("v1".to_string(), vec![Some(0), Some(1), None]),
            ("v2".to_string(), vec![Some(1), Some(1), Some(0)]),
        ];
        IndicatorMatrix::from_columns(cols).unwrap()
    }

    #[rstest]
    fn test_indicator_matrix_construction(simple_matrix: IndicatorMatrix) {
        assert_eq!(simple_matrix.n_individuals(), 3);
        assert_eq!(simple_matrix.n_indicators(), 2);
        assert_eq!(simple_matrix.variable_names(), &["v1", "v2"]);

        // Check values
        assert!(!simple_matrix.is_missing(0, 0));
        assert!(!simple_matrix.is_missing(1, 0));
        assert!(simple_matrix.is_missing(2, 0));

        // Check missing rates
        let rates = simple_matrix.missing_rates();
        assert_relative_eq!(rates[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(rates[1], 0.0, epsilon = 1e-12);

        // Check prevalences
        let prevs = simple_matrix.prevalences();
        assert_relative_eq!(prevs[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(prevs[1], 2.0 / 3.0, epsilon = 1e-12);
    }

    #[rstest]
    #[case::mismatched_lengths(
        vec![
            ("v1".to_string(), vec![Some(0u8), Some(1)]),
            ("v2".to_string(), vec![Some(1)]),
        ]
    )]
    #[case::fully_missing_column(
        vec![
            ("v1".to_string(), vec![None, None]),
            ("v2".to_string(), vec![Some(1u8), Some(0)]),
        ]
    )]
    fn test_indicator_matrix_invalid_data(#[case] cols: Vec<(String, Vec<Option<u8>>)>) {
        let res = IndicatorMatrix::from_columns(cols);
        assert!(matches!(res, Err(LcaError::InvalidData(_))));
    }

    #[test]
    fn test_indicator_precheck() {
        // v1 and v2 will be perfectly correlated (0,0,0,1,1,1)
        let cols = vec![
            (
                "v1".to_string(),
                vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)],
            ),
            (
                "v2".to_string(),
                vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)],
            ),
            (
                "v3".to_string(),
                vec![Some(0), Some(0), Some(0), Some(0), Some(0), Some(1)],
            ),
            (
                "v4".to_string(),
                vec![None, None, Some(0), Some(0), Some(0), Some(0)],
            ),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();
        let precheck = IndicatorPrecheck::default();
        let report = precheck.run(&matrix);

        // v1 and v2 are perfectly correlated
        assert!(!report.high_correlation_pairs.is_empty());
        assert_eq!(report.high_correlation_pairs[0].0, "v1");
        assert_eq!(report.high_correlation_pairs[0].1, "v2");
        assert!((report.high_correlation_pairs[0].2 - 1.0).abs() < 1e-6);

        // v3 prevalence is 1/6 = 0.166... (above 0.10)
        // let's adjust v3 to have 0 prevalence
        let cols2 = vec![
            (
                "v1".to_string(),
                vec![Some(0), Some(1), Some(0), Some(1), Some(0), Some(1)],
            ),
            (
                "v3".to_string(),
                vec![Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)],
            ),
            (
                "v4".to_string(),
                vec![None, None, Some(0), Some(0), Some(0), Some(0)],
            ),
        ];
        let matrix2 = IndicatorMatrix::from_columns(cols2).unwrap();
        let report2 = precheck.run(&matrix2);

        // v3 prevalence is 0
        assert!(
            report2
                .low_prevalence_vars
                .iter()
                .any(|(name, _)| name == "v3")
        );

        // v4 missing rate is 2/6 = 0.33... (above 0.20)
        assert!(
            report2
                .high_missing_vars
                .iter()
                .any(|(name, _)| name == "v4")
        );

        // Small sample warning should be present (N=6 < 300)
        assert!(report2.small_sample_warning.is_some());
    }

    #[test]
    fn test_em_basic_convergence() {
        // Two clear classes:
        // Class A: [1, 1, 0, 0]
        // Class B: [0, 0, 1, 1]
        let cols = vec![
            (
                "v1".to_string(),
                vec![Some(1), Some(1), Some(1), Some(0), Some(0), Some(0)],
            ),
            (
                "v2".to_string(),
                vec![Some(1), Some(1), Some(1), Some(0), Some(0), Some(0)],
            ),
            (
                "v3".to_string(),
                vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)],
            ),
            (
                "v4".to_string(),
                vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)],
            ),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();

        let (pi_init, rho_init) = init::random_init(2, 4, 42);
        let config = EmConfig::default();

        let res = em::run_em(&matrix, 2, pi_init, rho_init, &config).unwrap();

        assert!(res.converged);
        // Expect pi to be roughly [0.5, 0.5]
        assert_relative_eq!(res.pi[0], 0.5, epsilon = 0.01);
        assert_relative_eq!(res.pi[1], 0.5, epsilon = 0.01);

        // Expect rho to show clear separation
        let r = &res.rho;
        let c1_v1 = r[(0, 0)];
        let c1_v3 = r[(0, 2)];
        let c2_v1 = r[(1, 0)];
        let c2_v3 = r[(1, 2)];

        if c1_v1 > 0.9 {
            assert!(c1_v3 < 0.1);
            assert!(c2_v1 < 0.1);
            assert!(c2_v3 > 0.9);
        } else {
            assert!(c1_v1 < 0.1);
            assert!(c1_v3 > 0.9);
            assert!(c2_v1 > 0.9);
            assert!(c2_v3 < 0.1);
        }
    }

    #[test]
    fn test_em_k1_null_model() {
        let cols = vec![
            ("v1".to_string(), vec![Some(1), Some(0), Some(1), Some(0)]),
            ("v2".to_string(), vec![Some(1), Some(1), Some(0), None]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();
        let config = EmConfig::default();

        let (pi, rho) = init::random_init(1, 2, 42);
        let res = em::run_em(&matrix, 1, pi, rho, &config).unwrap();

        assert!(res.converged);
        assert_eq!(res.pi, vec![1.0]);
        assert_relative_eq!(res.rho[(0, 0)], 0.5); // (1+0+1+0)/4
        assert_relative_eq!(res.rho[(0, 1)], 2.0 / 3.0); // (1+1+0)/3
    }

    #[test]
    fn test_lca_model_fit_multi_restart() {
        let cols = vec![
            (
                "v1".to_string(),
                vec![Some(1), Some(1), Some(0), Some(0), Some(1), Some(0)],
            ),
            (
                "v2".to_string(),
                vec![Some(1), Some(1), Some(0), Some(0), Some(1), Some(0)],
            ),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();

        let model = LcaModel::new(2).unwrap().with_n_restarts(5).with_seed(42);
        let fit = model.fit(&matrix).unwrap();

        assert!(fit.converged());
        assert_eq!(fit.n_classes(), 2);
        assert!(fit.ll_replications() >= 1);

        // Test canonical ordering (descending prevalence)
        let pi = fit.class_prevalences();
        assert!(pi[0] >= pi[1]);

        // Test posteriors handle
        let posteriors = fit.compute_posteriors(&matrix);
        assert_eq!(posteriors.matrix().nrows(), 6);
        assert_eq!(posteriors.matrix().ncols(), 2);

        let assignments = posteriors.assign();
        assert_eq!(assignments.len(), 6);

        // Row 0 and 1 are identical ([1,1]), should be assigned to the same class
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[0], assignments[4]); // row 4 is also [1,1]
    }

    #[test]
    fn test_model_selection_bic() {
        // Generate data from 2 classes
        // Class 0: high p for v1,v2; low p for v3,v4
        // Class 1: low p for v1,v2; high p for v3,v4
        // N=100
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        let mut v3 = Vec::new();
        let mut v4 = Vec::new();

        for i in 0..100 {
            if i < 50 {
                // Class 0
                v1.push(Some(u8::from(rng.random_bool(0.9))));
                v2.push(Some(u8::from(rng.random_bool(0.9))));
                v3.push(Some(u8::from(rng.random_bool(0.1))));
                v4.push(Some(u8::from(rng.random_bool(0.1))));
            } else {
                // Class 1
                v1.push(Some(u8::from(rng.random_bool(0.1))));
                v2.push(Some(u8::from(rng.random_bool(0.1))));
                v3.push(Some(u8::from(rng.random_bool(0.9))));
                v4.push(Some(u8::from(rng.random_bool(0.9))));
            }
        }

        let cols = vec![
            ("v1".to_string(), v1),
            ("v2".to_string(), v2),
            ("v3".to_string(), v3),
            ("v4".to_string(), v4),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();

        let config = EmConfig {
            n_restarts: 10,
            seed: Some(123),
            ..Default::default()
        };

        let selection = select_n_classes(&matrix, 1, 3, &config).unwrap();

        // BIC should favor K=2 over K=1 and K=3
        let best = selection.best_by_bic().unwrap();
        assert_eq!(best.n_classes(), 2);

        let elbow = selection.bic_elbow().unwrap();
        assert_eq!(elbow.n_classes(), 2);
    }

    #[test]
    fn test_diagnostics_triggers() {
        // 1. Test small-class warning
        // Create 2-class model where one class is very small (5%)
        let pi = vec![0.95, 0.05];
        let mut rho = faer::Mat::zeros(2, 2);
        rho[(0, 0)] = 0.9;
        rho[(0, 1)] = 0.1;
        rho[(1, 0)] = 0.1;
        rho[(1, 1)] = 0.9;

        let fit = LcaFit {
            pi,
            rho,
            log_likelihood: -100.0,
            n_iter: 10,
            converged: true,
            ll_replications: 25,
            variable_names: vec!["v1".to_string(), "v2".to_string()],
            variable_labels: None,
        };

        // Synthetic data for N=100
        let cols = vec![
            ("v1".to_string(), vec![Some(1); 100]),
            ("v2".to_string(), vec![Some(0); 100]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();
        let diag = fit.diagnostics(&matrix);

        assert!(!diag.small_classes.is_empty());
        assert_eq!(diag.small_classes[0].class_index, 1);
        assert_relative_eq!(diag.small_classes[0].threshold, 0.15, epsilon = 1e-12); // K=2 threshold

        // 2. Test Salsa effect (parallel profiles)
        let pi_s = vec![0.5, 0.5];
        let mut rho_s = faer::Mat::zeros(2, 2);
        rho_s[(0, 0)] = 0.8;
        rho_s[(0, 1)] = 0.8;
        rho_s[(1, 0)] = 0.2;
        rho_s[(1, 1)] = 0.2;

        let fit_s = LcaFit {
            pi: pi_s,
            rho: rho_s,
            log_likelihood: -100.0,
            n_iter: 10,
            converged: true,
            ll_replications: 25,
            variable_names: vec!["v1".to_string(), "v2".to_string()],
            variable_labels: None,
        };
        let diag_s = fit_s.diagnostics(&matrix);
        assert!(diag_s.salsa_effect_detected);
    }

    #[test]
    fn test_frozen_model_scoring() {
        let cols = vec![
            ("v1".to_string(), vec![Some(1), Some(0), Some(1)]),
            ("v2".to_string(), vec![Some(1), Some(0), Some(0)]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();

        let pi = vec![0.6, 0.4];
        let mut rho = faer::Mat::zeros(2, 2);
        rho[(0, 0)] = 0.9;
        rho[(0, 1)] = 0.9;
        rho[(1, 0)] = 0.1;
        rho[(1, 1)] = 0.1;

        let names = vec!["v1".to_string(), "v2".to_string()];
        let frozen = FrozenLcaModel::new(pi, rho, names, None);

        let posteriors = frozen.score_posteriors(&matrix).unwrap();
        assert_eq!(posteriors.nrows(), 3);
        assert_eq!(posteriors.ncols(), 2);

        // Individual 0: [1, 1] -> should have high prob for class 0
        assert!(posteriors[(0, 0)] > 0.9);
        // Individual 1: [0, 0] -> should have high prob for class 1
        assert!(posteriors[(1, 1)] > 0.9);

        // Test score_assign wrapper
        let assignments = frozen.score_assign(&matrix).unwrap();
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 1);
    }

    #[test]
    fn test_export_csv() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = camino::Utf8Path::from_path(temp_dir.path()).unwrap();

        let cols = vec![
            ("v1".to_string(), vec![Some(1), Some(0)]),
            ("v2".to_string(), vec![Some(1), Some(0)]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();

        let model = LcaModel::new(2).unwrap().with_n_restarts(1).with_seed(42);
        let fit = model.fit(&matrix).unwrap();
        let posteriors = fit.compute_posteriors(&matrix);

        let post_path = path.join("posteriors.csv");
        write_posteriors_csv(&posteriors, &matrix, &post_path).unwrap();
        assert!(post_path.exists());

        let param_path = path.join("params.csv");
        write_params_csv(&fit, &param_path).unwrap();
        assert!(param_path.exists());

        let selection = select_n_classes(&matrix, 1, 2, &EmConfig::default()).unwrap();
        let stats_path = path.join("selection.csv");
        write_selection_stats_csv(&selection, &stats_path).unwrap();
        assert!(stats_path.exists());

        let diag = fit.diagnostics(&matrix);
        let diag_path = path.join("diagnostics.csv");
        write_diagnostics_csv(&diag, &diag_path).unwrap();
        assert!(diag_path.exists());
    }

    #[test]
    fn test_robustness_scenarios() {
        // 1. All-zero column
        let cols = vec![
            ("v1".to_string(), vec![Some(1), Some(0), Some(1), Some(0)]),
            ("v2".to_string(), vec![Some(0), Some(0), Some(0), Some(0)]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();
        let model = LcaModel::new(2).unwrap().with_n_restarts(1).with_seed(42);
        let fit = model.fit(&matrix).unwrap();
        assert!(fit.converged());
        assert!(fit.item_probs()[(0, 1)] < 1e-4);
        assert!(fit.item_probs()[(1, 1)] < 1e-4);

        // 2. Extreme imbalance
        let mut v1 = vec![Some(1); 95];
        v1.extend(vec![Some(0); 5]);
        let mut v2 = vec![Some(1); 95];
        v2.extend(vec![Some(0); 5]);
        let matrix_imbalance =
            IndicatorMatrix::from_columns(vec![("v1".to_string(), v1), ("v2".to_string(), v2)])
                .unwrap();

        let fit_imb = model.fit(&matrix_imbalance).unwrap();
        assert!(fit_imb.converged());
        let pi = fit_imb.class_prevalences();
        assert!(pi[0] > 0.90);
        assert!(pi[1] < 0.10);

        let diag = fit_imb.diagnostics(&matrix_imbalance);
        assert!(!diag.small_classes.is_empty());

        // 3. Collinear indicators
        let matrix_coll = IndicatorMatrix::from_columns(vec![
            ("v1".to_string(), vec![Some(1), Some(0), Some(1), Some(0)]),
            (
                "v1_dup".to_string(),
                vec![Some(1), Some(0), Some(1), Some(0)],
            ),
        ])
        .unwrap();
        let report = IndicatorPrecheck::default().run(&matrix_coll);
        assert!(!report.high_correlation_pairs.is_empty());
        assert_eq!(report.high_correlation_pairs[0].0, "v1");
        assert_eq!(report.high_correlation_pairs[0].1, "v1_dup");
    }

    #[test]
    fn test_missing_data_fiml_isolation() {
        let cols = vec![
            ("v1".to_string(), vec![Some(1), Some(0)]),
            ("v2".to_string(), vec![None, Some(1)]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();

        let pi = vec![0.5, 0.5];
        let mut rho = faer::Mat::zeros(2, 2);
        rho[(0, 0)] = 0.9;
        rho[(0, 1)] = 0.9;
        rho[(1, 0)] = 0.1;
        rho[(1, 1)] = 0.1;

        let fit = LcaFit {
            pi,
            rho,
            log_likelihood: 0.0,
            n_iter: 0,
            converged: true,
            ll_replications: 1,
            variable_names: vec!["v1".to_string(), "v2".to_string()],
            variable_labels: None,
        };

        let posteriors = fit.posteriors(&matrix);
        assert!(posteriors[(0, 0)] > 0.89);
        assert!(posteriors[(0, 0)] < 0.91);
    }

    #[test]
    fn test_serialization_and_labels() {
        let cols = vec![
            ("atc_n03".to_string(), vec![Some(1), Some(0)]),
            ("lpr_neu".to_string(), vec![Some(1), Some(0)]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols)
            .unwrap()
            .with_variable_labels(vec![
                "Anticonvulsants".to_string(),
                "Neurological DX".to_string(),
            ])
            .unwrap();

        let model = LcaModel::new(2).unwrap().with_n_restarts(1).with_seed(42);
        let fit = model.fit(&matrix).unwrap();

        // 1. Test label propagation to summary
        let summaries = fit.class_summaries();
        assert_eq!(summaries[0].item_response_probs[0].0, "Anticonvulsants");

        // 2. Test Serde serialization
        let json = serde_json::to_string(&fit).unwrap();
        let fit_de: LcaFit = serde_json::from_str(&json).unwrap();

        assert_eq!(fit_de.n_classes(), 2);
        assert_eq!(fit_de.variable_labels().unwrap()[0], "Anticonvulsants");

        // 3. Test Frozen Serde
        let frozen = fit.freeze();
        let frozen_json = serde_json::to_string(&frozen).unwrap();
        let frozen_de: FrozenLcaModel = serde_json::from_str(&frozen_json).unwrap();
        assert_eq!(frozen_de.variable_labels().unwrap()[0], "Anticonvulsants");
    }

    #[test]
    fn test_weighted_means_and_similarity() {
        let cols = vec![
            ("v1".to_string(), vec![Some(1), Some(1), Some(0), Some(0)]),
            ("v2".to_string(), vec![Some(0), Some(0), Some(1), Some(1)]),
        ];
        let matrix = IndicatorMatrix::from_columns(cols).unwrap();

        // Two classes: Class 0 is [1,0], Class 1 is [0,1]
        let model = LcaModel::new(2).unwrap().with_n_restarts(1).with_seed(42);
        let fit = model.fit(&matrix).unwrap();
        let posteriors = fit.compute_posteriors(&matrix);

        // 1. Test weighted means
        let outcome = vec![100.0, 100.0, 20.0, 20.0];
        let means = posteriors.weighted_means(&outcome);

        // Profiles are [1,0] and [0,1].
        // Row 0,1 are [1,0] -> Class 0
        // Row 2,3 are [0,1] -> Class 1
        assert_relative_eq!(means[0], 100.0, epsilon = 1e-1);
        assert_relative_eq!(means[1], 20.0, epsilon = 1e-1);

        // 2. Test profile similarity
        let sim = fit.compare_profiles(&fit).unwrap();
        assert_relative_eq!(sim[(0, 0)], 1.0);
        assert_relative_eq!(sim[(1, 1)], 1.0);
        // Orthogonal profiles [1,0] and [0,1] -> Cosine Similarity ≈ 0
        assert!(sim[(0, 1)] < 0.1);
    }
}
