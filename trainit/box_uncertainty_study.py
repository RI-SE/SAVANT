#!/usr/bin/env python3
"""
box_uncertainty_study - Does a learned factor on the baseline box sigma give better-calibrated uncertainty?

Reads residual tables from build-residuals and compares, per component (cx, cy, w, h, r), on held-out folds:
- baseline: the box's sigma (sigma_px / box_sigma), as markit writes it;
- baseline_scaled: the baseline times one scale fitted on the training folds (a fair reference: the RF gets one too);
- error_model: what markit writes with an error_model.json fitted (fit-error-model --residuals) on the training folds;
- rf: baseline x f(features), f a random forest fitted to |residual| / baseline on the training folds, then scaled.
Each scale is set so that 68.3 % of the training residuals fall within +-1 sigma (out-of-bag RF predictions).
The bias (mean = a + b*size per class, as in error_model.json) is fitted on the detector boxes of the training folds
and removed from the held-out residuals first, so sigma models the spread only (--keep-bias: sigma also covers the
bias, and error_model is fitted without a mean).

Folds hold out one recording at a time; with a single recording, objects (GT ids) are held out instead, which says
nothing about other recordings. Rows of one object are strongly correlated, hence large leaves in the forest.
Metrics: coverage at 1/2 sigma (ideal 0.683/0.954), mean Gaussian NLL (lower is better), sharpness (mean sigma),
Spearman rho between sigma and |residual|, and permutation importance of the features (rf, on held-out data).
Requires scikit-learn (dependency group `study`).
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from savant_common.quality import ALL_CLASSES, SIGMA_COMPONENTS, box_sigma
from trainit.build_residuals import FEATURE_COLUMNS
from trainit.fit_error_model import fit_localization, line, model_box, read_tables, table_arrays

logger = logging.getLogger(__name__)

ONE_SIGMA = math.erf(1 / math.sqrt(2))  # 0.683
TWO_SIGMA = math.erf(2 / math.sqrt(2))  # 0.954
MISSING = -1.0  # trees split it off; all features are >= 0
NUMERIC_EXTRA = ("score", "gap_distance")
CATEGORICAL = ("auto_class", "source", "origin")
MIN_FOLDS = 2
OBJECT_FOLDS = 5
FOREST = {"n_estimators": 200, "min_samples_leaf": 50, "max_features": "sqrt", "oob_score": True, "n_jobs": -1,
          "random_state": 0}
IMPORTANCE_ROWS = 5000
IMPORTANCE_REPEATS = 3
METHODS = ("baseline", "baseline_scaled", "error_model", "rf")


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    return MISSING if math.isnan(number) else number


def design_matrix(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
    """Numeric features, plus one-hot class/source/origin; columns that are missing everywhere are dropped."""
    names = [c for c in FEATURE_COLUMNS + list(NUMERIC_EXTRA) if any(_number(r.get(c)) != MISSING for r in rows)]
    columns = [[_number(r.get(c)) for r in rows] for c in names]
    for field in CATEGORICAL:
        for value in sorted({r.get(field) or "" for r in rows}):
            names.append(f"{field}={value}")
            columns.append([float((r.get(field) or "") == value) for r in rows])
    return np.array(columns, dtype=float).T, names


def folds(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, str]:
    """Fold number per row: by recording, or by GT object if there is only one recording."""
    recordings = sorted({r["recording"] for r in rows})
    if len(recordings) >= MIN_FOLDS:
        index = {name: i for i, name in enumerate(recordings)}
        return np.array([index[r["recording"]] for r in rows]), "recording"
    logger.warning("Only one recording: folds hold out objects, not recordings (no evidence for other recordings)")
    ids = sorted({r["gt_id"] for r in rows})
    index = {gt_id: i % OBJECT_FOLDS for i, gt_id in enumerate(np.random.default_rng(0).permutation(ids))}
    return np.array([index[r["gt_id"]] for r in rows]), "object"


def _gap(value) -> Optional[int]:
    return int(float(value)) if value not in (None, "", "nan") else None


def held_out_models(rows: List[Dict[str, str]], fold: np.ndarray, keep_bias: bool) -> Tuple[np.ndarray, np.ndarray]:
    """(bias, error-model sigma), each N x 5, from an error model fitted on the detector boxes of the other folds.

    The sigma is box_sigma() of that model, as markit would write it (origin rule included); the bias is zero with
    keep_bias, and the model then has no mean, so its sigma covers the bias.
    """
    residuals, sizes, classes = table_arrays(rows)
    fit_rows = np.array([model_box(r) for r in rows])
    bias, sigma = np.zeros_like(residuals), np.zeros_like(residuals)
    for k in np.unique(fold):
        train = (fold != k) & fit_rows
        localization = fit_localization(residuals[train], sizes[train], classes[train], with_mean=not keep_bias)
        by_class = localization["by_class"]
        for i in np.flatnonzero(fold == k):
            row, coefficients = rows[i], by_class.get(classes[i]) or by_class[ALL_CLASSES]
            sigma[i] = box_sigma(localization, classes[i], float(row["w"]), float(row["h"]), row.get("origin"),
                                 _gap(row.get("gap_distance")))
            if not keep_bias:
                bias[i] = [line(coefficients["mean"][c], sizes[i]) for c in SIGMA_COMPONENTS]
    return bias, sigma


def scale_for_coverage(ratio: np.ndarray) -> float:
    """Factor on sigma so that ONE_SIGMA of |residual|/sigma fall within 1."""
    return float(np.quantile(ratio, ONE_SIGMA))


def metrics(residual: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    from scipy.stats import spearmanr

    z = np.abs(residual) / sigma
    nll = 0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * z ** 2
    return {"cov1": float(np.mean(z <= 1)), "cov2": float(np.mean(z <= 2)), "nll": float(np.mean(nll)),
            "sharpness": float(np.mean(sigma)),
            "spearman": math.nan if np.ptp(sigma) == 0 else float(spearmanr(sigma, np.abs(residual))[0])}


def fit_fold(x_train, ratio_train, x_test):
    """Forest predicting |residual|/baseline; (factor on the test rows, fitted forest)."""
    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor(**FOREST).fit(x_train, ratio_train)
    oob = np.maximum(forest.oob_prediction_, 1e-6)
    scale = scale_for_coverage(ratio_train / oob)
    return np.maximum(forest.predict(x_test), 1e-6) * scale, forest


def importance(forest, x_test, ratio_test, names: List[str]) -> Dict[str, float]:
    from sklearn.inspection import permutation_importance

    rows = np.random.default_rng(0).permutation(len(x_test))[:IMPORTANCE_ROWS]
    result = permutation_importance(forest, x_test[rows], ratio_test[rows], n_repeats=IMPORTANCE_REPEATS,
                                    scoring="neg_mean_absolute_error", random_state=0, n_jobs=-1)
    return dict(zip(names, result.importances_mean.tolist()))


def study_component(component: str, rows, x, names, fold, bias: np.ndarray, model_sigma: np.ndarray) -> Dict:
    """Held-out metrics per method and mean feature importance for one sigma component, after removing the bias."""
    raw = np.array([float(r[f"res_{component}"]) for r in rows])
    residual = raw - bias
    base = np.array([float(r[f"sigma_{component}"]) for r in rows])
    valid = base > 0
    raw, residual, base, x_c, fold_c = raw[valid], residual[valid], base[valid], x[valid], fold[valid]
    model_sigma = model_sigma[valid]
    ratio = np.abs(residual) / base
    sigma = {m: np.empty_like(base) for m in METHODS}
    importances = []
    for k in np.unique(fold_c):
        test, train = fold_c == k, fold_c != k
        sigma["baseline"][test] = base[test]
        sigma["baseline_scaled"][test] = base[test] * scale_for_coverage(ratio[train])
        sigma["error_model"][test] = model_sigma[test]
        factor, forest = fit_fold(x_c[train], ratio[train], x_c[test])
        sigma["rf"][test] = base[test] * factor
        importances.append(importance(forest, x_c[test], ratio[test], names))
    per_fold = {m: [metrics(residual[fold_c == k], sigma[m][fold_c == k]) for k in np.unique(fold_c)]
                for m in METHODS}
    return {
        "n": int(valid.sum()),
        "bias": {"mean": float(np.mean(bias[valid])),
                 "ms_share": float(1 - np.mean(residual ** 2) / max(np.mean(raw ** 2), 1e-12))},
        "pooled": {m: metrics(residual, sigma[m]) for m in METHODS},
        "per_fold": per_fold,
        "importance": {n: float(np.mean([imp[n] for imp in importances])) for n in names},
    }


def text_report(report: Dict) -> str:
    lines = [f"Folds by {report['folds']} ({report['n_folds']}), {report['n_rows']} boxes, "
             f"{report['n_recordings']} recording(s). Ideal coverage: {ONE_SIGMA:.3f} / {TWO_SIGMA:.3f}.",
             "Bias kept in the residuals." if report["keep_bias"] else
             "Bias removed (fitted on the training folds); ms_share = share of the mean squared residual it explains.",
             ""]
    if not report["keep_bias"]:
        lines += [f"  bias {c}: mean {r['bias']['mean']:.3f}, ms_share {r['bias']['ms_share']:.2f}"
                  for c, r in report["components"].items()] + [""]
    header = f"{'comp':<5}{'method':<17}{'cov1':>7}{'cov2':>7}{'nll':>9}{'sharp':>9}{'rho':>7}"
    lines.append(header)
    for component, result in report["components"].items():
        for method in METHODS:
            m = result["pooled"][method]
            lines.append(f"{component:<5}{method:<17}{m['cov1']:>7.3f}{m['cov2']:>7.3f}{m['nll']:>9.3f}"
                         f"{m['sharpness']:>9.3f}{m['spearman']:>7.3f}")
    lines += ["", "Top features (permutation importance, mean over folds):"]
    for component, result in report["components"].items():
        top = sorted(result["importance"].items(), key=lambda item: -item[1])[:6]
        lines.append(f"  {component}: " + ", ".join(f"{name} {value:.3f}" for name, value in top))
    return "\n".join(lines)


def plot_importance(report: Dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = list(report["components"])
    fig, axes = plt.subplots(1, len(components), figsize=(4 * len(components), 5), sharex=False)
    for ax, component in zip(np.atleast_1d(axes), components):
        top = sorted(report["components"][component]["importance"].items(), key=lambda item: item[1])[-10:]
        ax.barh([name for name, _ in top], [value for _, value in top])
        ax.set_title(component)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(paths: List[Path], output_dir: Path, keep_bias: bool = False) -> Dict:
    rows = read_tables(paths)
    x, names = design_matrix(rows)
    fold, fold_kind = folds(rows)
    bias, model_sigma = held_out_models(rows, fold, keep_bias)
    report = {
        "inputs": [str(p) for p in paths], "n_rows": len(rows), "folds": fold_kind,
        "n_folds": int(len(np.unique(fold))), "n_recordings": len({r["recording"] for r in rows}),
        "keep_bias": keep_bias, "features": names, "forest": {k: v for k, v in FOREST.items() if k != "n_jobs"},
        "components": {c: study_component(c, rows, x, names, fold, bias[:, i], model_sigma[:, i])
                       for i, c in enumerate(SIGMA_COMPONENTS)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "report.txt").write_text(text_report(report) + "\n")
    plot_importance(report, output_dir / "importance.png")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="+", type=Path, help="CSV tables from build-residuals")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where report.json/.txt and importance.png go")
    parser.add_argument("--keep-bias", action="store_true", help="Do not remove the fitted bias (sigma covers it)")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        import sklearn  # noqa: F401
    except ImportError:
        logger.error("scikit-learn is missing: uv sync --group trainit --group study")
        return 1
    report = run(args.tables, args.output_dir, args.keep_bias)
    print(text_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
