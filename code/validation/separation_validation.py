"""
Validation script for Regional-Residual Separation against analytical prism fields.

Compares the pipeline's estimated Total (RTP), Regional, and Residual fields
against the exact analytical magnetic anomalies (evaluated at the pole,
I = 90 deg, D = 0 deg) of the synthetic prisms defined in true_source_config.json.

Separating the validation into three rows (Total RTP, Regional, Residual) isolates
how much error originates in the low-latitude RTP step versus the upward/downward
continuation separation step.
"""

import argparse
import json
import sys
from pathlib import Path

import harmonica as hm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"


def find_true_source_config(dataset_dir: Path, config_name: str = "true_source_config.json") -> Path:
    """Locates true_source_config.json in common dataset locations."""
    candidates = [
        dataset_dir / config_name,
        dataset_dir / "input" / config_name,
        dataset_dir / "config" / config_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find '{config_name}'. Searched: {searched}")


def compute_prism_tfa(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    prism_bounds: list[float],
    intensity: float,
    inc: float,
    dec: float,
) -> np.ndarray:
    """
    Computes the analytical Total Field Anomaly (nT) of a single rectangular prism
    using Harmonica (ENU coordinate system, inclination positive downward).
    """
    mag_e, mag_n, mag_u = hm.magnetic_angles_to_vec(intensity, inc, dec)
    magnetization = (
        np.array([mag_e], dtype=np.float64),
        np.array([mag_n], dtype=np.float64),
        np.array([mag_u], dtype=np.float64),
    )
    b_e, b_n, b_u = hm.prism_magnetic(
        coordinates,
        [prism_bounds],
        magnetization=magnetization,
        field="b",
    )
    tfa = hm.total_field_anomaly([b_e, b_n, b_u], inc, dec)
    return np.asarray(tfa, dtype=np.float64)


def compute_metrics(estimated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """
    Computes quantitative comparison metrics.
    Error convention: error = estimated - reference (positive = overestimation).
    """
    error = estimated - reference
    amp = float(np.max(reference) - np.min(reference))
    rmse = float(np.sqrt(np.mean(error**2)))
    rel_error_pct = (100.0 * rmse / amp) if amp > 0.0 else float("nan")
    corr = float(np.corrcoef(estimated, reference)[0, 1])

    return {
        "rmse_nT": rmse,
        "rel_error_pct": rel_error_pct,
        "max_abs_err_nT": float(np.max(np.abs(error))),
        "bias_nT": float(np.mean(error)),
        "corr": corr,
        "true_min_nT": float(np.min(reference)),
        "true_max_nT": float(np.max(reference)),
        "est_min_nT": float(np.min(estimated)),
        "est_max_nT": float(np.max(estimated)),
    }


def _draw_prism_outline(ax: plt.Axes, prism_bounds: list[float], style: str = "k--", label: str | None = None) -> None:
    """Overlays the horizontal footprint [West, East, South, North] of a prism."""
    w, e, s, n = prism_bounds[:4]
    ax.plot([w, e, e, w, w], [s, s, n, n, s], style, lw=1.2, label=label)


def plot_validation_grid(
    X: np.ndarray,
    Y: np.ndarray,
    rows_data: list[dict],
    figname: Path,
    unit: str = "nT",
) -> None:
    """
    Plots a 3x3 validation grid:
      Columns: Analytical True | Pipeline Estimated | Error (Estimated - True)
      Rows:    Total Field (RTP) | Regional (Deep) | Residual (Shallow)
    """
    fig, axes = plt.subplots(3, 3, figsize=(17, 15), constrained_layout=True)

    for row_idx, item in enumerate(rows_data):
        true_grid = item["true"]
        est_grid = item["est"]
        err_grid = est_grid - true_grid
        m = item["metrics"]

        vmin = min(float(np.min(true_grid)), float(np.min(est_grid)))
        vmax = max(float(np.max(true_grid)), float(np.max(est_grid)))
        levels_val = np.linspace(vmin, vmax, 21)

        err_max = max(float(np.max(np.abs(err_grid))), 1e-9)
        levels_err = np.linspace(-err_max, err_max, 21)

        ax_true, ax_est, ax_err = axes[row_idx]

        # Column 1: Analytical True
        c1 = ax_true.contourf(X, Y, true_grid, levels=levels_val, cmap="jet", vmin=vmin, vmax=vmax)
        fig.colorbar(c1, ax=ax_true, label=unit)
        ax_true.set_title(f"{item['label']} — Analytical True")

        # Column 2: Pipeline Estimated
        c2 = ax_est.contourf(X, Y, est_grid, levels=levels_val, cmap="jet", vmin=vmin, vmax=vmax)
        fig.colorbar(c2, ax=ax_est, label=unit)
        ax_est.set_title(f"{item['label']} — Estimated")

        # Column 3: Error (Est - True)
        c3 = ax_err.contourf(X, Y, err_grid, levels=levels_err, cmap="seismic", vmin=-err_max, vmax=err_max)
        fig.colorbar(c3, ax=ax_err, label=f"Error ({unit})")
        ax_err.set_title(
            f"{item['label']} — Error (Est - True)\n"
            f"RMSE: {m['rmse_nT']:.2f} {unit} | Rel: {m['rel_error_pct']:.1f}% | r: {m['corr']:.3f}"
        )

        # Formatting & prism footprints
        for ax in (ax_true, ax_est, ax_err):
            ax.set_aspect("equal")
            for prism_bounds, style, plabel in item["prisms"]:
                _draw_prism_outline(ax, prism_bounds, style=style, label=plabel if ax is ax_true else None)
            if row_idx == 2:
                ax.set_xlabel("Easting (m)")
        ax_true.set_ylabel("Northing (m)")
        ax_true.legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.suptitle(
        "Regional-Residual Separation Validation vs. Analytical Pole Fields",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Validation figure saved to: {figname}")


def run_validation(
    dataset_name: str,
    sep_name: str,
    orig_field: str = "obs_rtp",
    inc_pole: float = 90.0,
    dec_pole: float = 0.0,
) -> None:
    print(f"\n==== Validating Separation '{sep_name}' for Dataset '{dataset_name}' ====")

    dataset_dir = DATA_FOLDER / dataset_name
    sep_dir = dataset_dir / "separation" / sep_name
    results_csv = sep_dir / f"{sep_name}_results.csv"

    if not results_csv.exists():
        raise FileNotFoundError(f"Separation results CSV not found: {results_csv}")

    # 1. Load true source configuration
    src_cfg_path = find_true_source_config(dataset_dir)
    with src_cfg_path.open("r", encoding="utf-8") as f:
        src_cfg = json.load(f)

    prisms: list[list[float]] = src_cfg["prisms"]
    intensities: list[float] = src_cfg["intensities"]

    if len(prisms) != 2:
        raise ValueError(f"Expected exactly 2 prisms (shallow + deep), found {len(prisms)}.")

    # Identify shallow vs. deep automatically using z_top (index 5, larger = shallower in ENU)
    order = sorted(range(2), key=lambda idx: prisms[idx][5], reverse=True)
    shallow_idx, deep_idx = order[0], order[1]
    shallow_prism, shallow_int = prisms[shallow_idx], float(intensities[shallow_idx])
    deep_prism, deep_int = prisms[deep_idx], float(intensities[deep_idx])

    print(f"Shallow prism (z_top={shallow_prism[5]} m, M={shallow_int} A/m): {shallow_prism}")
    print(f"Deep prism    (z_top={deep_prism[5]} m, M={deep_int} A/m): {deep_prism}")

    # 2. Load separation results
    df = pd.read_csv(results_csv)
    df = df.sort_values(by=["Y", "X"]).reset_index(drop=True)

    for col in ("X", "Y", "Z", orig_field, "regional", "residual"):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in {results_csv.name}. Found: {df.columns.tolist()}")

    easting = df["X"].to_numpy(dtype=np.float64)
    northing = df["Y"].to_numpy(dtype=np.float64)
    upwards = df["Z"].to_numpy(dtype=np.float64)

    est_total = df[orig_field].to_numpy(dtype=np.float64)
    est_regional = df["regional"].to_numpy(dtype=np.float64)
    est_residual = df["residual"].to_numpy(dtype=np.float64)

    # 3. Compute analytical fields at the pole (I=90, D=0)
    coords = (easting, northing, upwards)
    true_shallow = compute_prism_tfa(coords, shallow_prism, shallow_int, inc_pole, dec_pole)
    true_deep = compute_prism_tfa(coords, deep_prism, deep_int, inc_pole, dec_pole)
    true_total = true_shallow + true_deep

    # 4. Evaluate metrics
    m_total = compute_metrics(est_total, true_total)
    m_regional = compute_metrics(est_regional, true_deep)
    m_residual = compute_metrics(est_residual, true_shallow)

    summary_rows = [
        {"component": "Total (RTP)", **m_total},
        {"component": "Regional (Deep)", **m_regional},
        {"component": "Residual (Shallow)", **m_residual},
    ]
    df_metrics = pd.DataFrame(summary_rows)

    print("\n--- Validation Metrics (Error = Estimated - Analytical) ---")
    print(
        f"{'Component':20s} {'RMSE (nT)':>10s} {'Rel Err':>9s} {'Max |Err|':>10s} "
        f"{'Bias':>9s} {'Corr (r)':>9s} {'True Range (nT)':>19s} {'Est Range (nT)':>19s}"
    )
    for row in summary_rows:
        print(
            f"{row['component']:20s} {row['rmse_nT']:10.2f} {row['rel_error_pct']:8.2f}% "
            f"{row['max_abs_err_nT']:10.2f} {row['bias_nT']:9.2f} {row['corr']:9.4f} "
            f"{row['true_min_nT']:8.1f} to {row['true_max_nT']:7.1f} "
            f"{row['est_min_nT']:8.1f} to {row['est_max_nT']:7.1f}"
        )

    # 5. Export tables and figure
    #out_dir = sep_dir / "validation"
    out_dir = PROJECT_HOME / "output" / "separation_validation"
    out_dir.mkdir(exist_ok=True, parents=True)

    metrics_csv = out_dir / "validation_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)

    df_fields = pd.DataFrame({
        "X": easting,
        "Y": northing,
        "Z": upwards,
        "true_total_pole": true_total,
        "est_total_rtp": est_total,
        "err_total": est_total - true_total,
        "true_regional_deep": true_deep,
        "est_regional": est_regional,
        "err_regional": est_regional - true_deep,
        "true_residual_shallow": true_shallow,
        "est_residual": est_residual,
        "err_residual": est_residual - true_shallow,
    })
    fields_csv = out_dir / "validation_fields.csv"
    df_fields.to_csv(fields_csv, index=False)

    nx = len(np.unique(easting))
    ny = len(np.unique(northing))
    grid_shape = (ny, nx)
    X = easting.reshape(grid_shape)
    Y = northing.reshape(grid_shape)

    rows_data = [
        {
            "label": "Total Field (RTP)",
            "true": true_total.reshape(grid_shape),
            "est": est_total.reshape(grid_shape),
            "metrics": m_total,
            "prisms": [
                (shallow_prism, "k--", "Shallow prism"),
                (deep_prism, "w:", "Deep prism"),
            ],
        },
        {
            "label": "Regional (Deep)",
            "true": true_deep.reshape(grid_shape),
            "est": est_regional.reshape(grid_shape),
            "metrics": m_regional,
            "prisms": [(deep_prism, "k--", "Deep prism")],
        },
        {
            "label": "Residual (Shallow)",
            "true": true_shallow.reshape(grid_shape),
            "est": est_residual.reshape(grid_shape),
            "metrics": m_residual,
            "prisms": [(shallow_prism, "k--", "Shallow prism")],
        },
    ]

    fig_path = out_dir / "separation_validation_comparison.png"
    plot_validation_grid(X, Y, rows_data, fig_path, unit="nT")
    print(f"Validation tables saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Regional-Residual separation against analytical prism pole fields."
    )
    parser.add_argument("--dataset", type=str, default="model4", help="Dataset folder name (default: model4)")
    parser.add_argument("--sep-name", type=str, default="sep_1", help="Separation run name (default: sep_1)")
    parser.add_argument(
        "--orig-field",
        type=str,
        default="obs_rtp",
        help="Column name of the observed/RTP field in separation results CSV (default: obs_rtp)",
    )
    parser.add_argument("--inc-pole", type=float, default=90.0, help="Target pole inclination (default: 90.0)")
    parser.add_argument("--dec-pole", type=float, default=0.0, help="Target pole declination (default: 0.0)")

    args = parser.parse_args()
    run_validation(
        dataset_name=args.dataset,
        sep_name=args.sep_name,
        orig_field=args.orig_field,
        inc_pole=args.inc_pole,
        dec_pole=args.dec_pole,
    )
