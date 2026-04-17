//! CSV Export utilities for LCA results.

use crate::data::IndicatorMatrix;
use crate::diagnostics::FitDiagnostics;
use crate::error::LcaError;
use crate::fit::LcaFit;
use crate::posterior::ComputedPosteriors;
use crate::selection::LcaSelectionTable;
use camino::Utf8Path;
use csv::Writer;
use std::fs::File;

/// Write per-individual posteriors + hard assignment to CSV.
///
/// Columns: `id` (or `individual_index`), `class_0` ... `class_K-1`, `assigned_class`.
///
/// # Errors
/// Returns [`LcaError::Io`] if file creation or writing fails.
pub fn write_posteriors_csv(
    posteriors: &ComputedPosteriors,
    data: &IndicatorMatrix,
    path: &Utf8Path,
) -> Result<(), LcaError> {
    let file = File::create(path).map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    let mut wtr = Writer::from_writer(file);

    let n = data.n_individuals();
    let k = posteriors.matrix().ncols();
    let ids = data.ids();

    let mut header = Vec::new();
    if ids.is_some() {
        header.push("id".to_string());
    } else {
        header.push("individual_index".to_string());
    }
    for kk in 0..k {
        header.push(format!("class_{kk}"));
    }
    header.push("assigned_class".to_string());
    wtr.write_record(&header).map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e.into(),
    })?;

    let assignments = posteriors.assign();
    let matrix = posteriors.matrix();

    for i in 0..n {
        let mut row = Vec::new();
        if let Some(id_list) = ids {
            row.push(id_list[i].clone());
        } else {
            row.push(i.to_string());
        }

        for kk in 0..k {
            row.push(format!("{:.6}", matrix[(i, kk)]));
        }
        row.push(assignments[i].to_string());

        wtr.write_record(&row).map_err(|e| LcaError::Io {
            path: path.to_string(),
            source: e.into(),
        })?;
    }

    wtr.flush().map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    Ok(())
}

/// Write item-response probability table to CSV.
///
/// Columns: `variable`, `class`, `prob`.
///
/// # Errors
/// Returns [`LcaError::Io`] if file creation or writing fails.
pub fn write_params_csv(fit: &LcaFit, path: &Utf8Path) -> Result<(), LcaError> {
    let file = File::create(path).map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    let mut wtr = Writer::from_writer(file);

    wtr.write_record(["variable", "class", "prob"])
        .map_err(|e| LcaError::Io {
            path: path.to_string(),
            source: e.into(),
        })?;

    let k = fit.n_classes();
    let j_count = fit.n_indicators();
    let rho = fit.item_probs();
    let names = fit.variable_names();

    for j in 0..j_count {
        for kk in 0..k {
            wtr.write_record([&names[j], &kk.to_string(), &format!("{:.6}", rho[(kk, j)])])
                .map_err(|e| LcaError::Io {
                    path: path.to_string(),
                    source: e.into(),
                })?;
        }
    }

    wtr.flush().map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    Ok(())
}

/// Write model selection stats for all K values to CSV.
///
/// Columns: `k`, `log_likelihood`, `bic`, `sabic`, `aic`, `entropy`, `purity`, `replications`, `converged`.
///
/// # Errors
/// Returns [`LcaError::Io`] if file creation or writing fails.
pub fn write_selection_stats_csv(
    selection: &LcaSelectionTable,
    path: &Utf8Path,
) -> Result<(), LcaError> {
    let file = File::create(path).map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    let mut wtr = Writer::from_writer(file);

    wtr.write_record([
        "k",
        "log_likelihood",
        "bic",
        "sabic",
        "aic",
        "entropy",
        "purity",
        "replications",
        "converged",
    ])
    .map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e.into(),
    })?;

    for (k, s, _) in &selection.entries {
        wtr.write_record(&[
            k.to_string(),
            format!("{:.4}", s.log_likelihood),
            format!("{:.4}", s.bic),
            format!("{:.4}", s.sabic),
            format!("{:.4}", s.aic),
            format!("{:.4}", s.mean_posterior_entropy),
            format!("{:.4}", s.modal_assignment_purity),
            s.ll_replications.to_string(),
            s.converged.to_string(),
        ])
        .map_err(|e| LcaError::Io {
            path: path.to_string(),
            source: e.into(),
        })?;
    }

    wtr.flush().map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    Ok(())
}

/// Write fit diagnostics report to CSV.
///
/// # Errors
/// Returns [`LcaError::Io`] if file creation or writing fails.
pub fn write_diagnostics_csv(
    diagnostics: &FitDiagnostics,
    path: &Utf8Path,
) -> Result<(), LcaError> {
    let file = File::create(path).map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    let mut wtr = Writer::from_writer(file);

    wtr.write_record([
        "type",
        "class_index",
        "variable_a",
        "variable_b",
        "value",
        "threshold",
        "message",
    ])
    .map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e.into(),
    })?;

    for sc in &diagnostics.small_classes {
        wtr.write_record([
            "small_class",
            &sc.class_index.to_string(),
            "",
            "",
            &format!("{:.4}", sc.prevalence),
            &format!("{:.4}", sc.threshold),
            &sc.message,
        ])
        .map_err(|e| LcaError::Io {
            path: path.to_string(),
            source: e.into(),
        })?;
    }

    if diagnostics.salsa_effect_detected {
        wtr.write_record([
            "salsa_effect",
            "",
            "",
            "",
            "",
            &format!("{:.4}", diagnostics.config.salsa_w_threshold),
            "Salsa effect detected (profiles are approximately parallel)",
        ])
        .map_err(|e| LcaError::Io {
            path: path.to_string(),
            source: e.into(),
        })?;
    }

    if diagnostics.insufficient_ll_replications {
        wtr.write_record([
            "insufficient_ll_replications",
            "",
            "",
            "",
            &diagnostics.ll_replications.to_string(),
            &diagnostics.ll_replication_target.to_string(),
            "Maximum log-likelihood replicated fewer times than target",
        ])
        .map_err(|e| LcaError::Io {
            path: path.to_string(),
            source: e.into(),
        })?;
    }

    for wcd in &diagnostics.within_class_dependence {
        wtr.write_record([
            "within_class_dependence",
            &wcd.class_index.to_string(),
            &wcd.var_a,
            &wcd.var_b,
            &format!("{:.4}", wcd.correlation),
            &format!(
                "{:.4}",
                diagnostics.config.within_class_correlation_threshold
            ),
            "High within-class correlation detected",
        ])
        .map_err(|e| LcaError::Io {
            path: path.to_string(),
            source: e.into(),
        })?;
    }

    wtr.flush().map_err(|e| LcaError::Io {
        path: path.to_string(),
        source: e,
    })?;
    Ok(())
}
