# latent_class

High-performance Latent Class Analysis (LCA) for categorical registry data.

This crate provides a parallel implementation of the Expectation-Maximization (EM) algorithm for LCA, specifically designed for binary indicator data (e.g., diagnosis flags, medication use) common in administrative registers.

## Features

- **FIML Missing Data:** Implements Full Information Maximum Likelihood (FIML) style EM, where individuals with missing indicators are retained rather than dropped.
- **Parallel Restarts:** Executes multiple random restarts in parallel via `rayon`, providing a significant speedup for ensuring global maximum recovery.
- **Comprehensive Fit Statistics:** Calculates BIC, SABIC, AIC, and posterior entropy for model selection.
- **Automated Diagnostics:** Detects small classes, parallel item-profiles ("Salsa effect"), and local independence violations.
- **Out-of-sample Scoring:** Supports "frozen" models to apply fitted parameters to new data without refitting.
- **CSV Export:** Built-in support for exporting posteriors, assignments, parameters, and diagnostics to research-ready CSV files.

## Usage

```rust
use latent_class::{IndicatorMatrix, LcaModel, EmConfig};

// 1. Load data into an IndicatorMatrix
let cols = vec![
    ("cardiac".to_string(), vec![Some(1), Some(0), Some(1)]),
    ("respiratory".to_string(), vec![Some(1), Some(0), None]),
];
let data = IndicatorMatrix::from_columns(cols).unwrap();

// 2. Configure the model
let model = LcaModel::new(2)? // 2 classes
    .with_n_restarts(50)      // recommended minimum
    .with_seed(42);

// 3. Fit and inspect results
let fit = model.fit(&data).unwrap();
println!("BIC: {}", fit.selection_stats(&data).bic);

// 4. Export assignments
let posteriors = fit.compute_posteriors(&data);
latent_class::write_posteriors_csv(&posteriors, &data, "posteriors.csv".into())?;
```

## Performance

The crate is optimized for large registry cohorts ($N > 100,000$) using the `faer` linear algebra library and `rayon` for multi-core parallelism across EM restarts.
