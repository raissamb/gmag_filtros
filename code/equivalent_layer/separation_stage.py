"""
Regional-Residual separation stage.

Computes Residual = Original - Regional using the observed data CSV
and the downward-continued regional field CSV produced by a deep equivalent layer.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"

import modules.utils as ut


def run_separation(dataset_name: str, sep_config_name: str) -> None:
    print(f"\n--- Starting Regional-Residual Separation for {dataset_name} ---")

    dataset_dir = DATA_FOLDER / dataset_name
    sep_config = ut.find_read_json_file(dataset_dir, sep_config_name)
    if not sep_config.get("execute", True):
        print(f"Skipping separation {sep_config_name} (execute=False)")
        return

    easting_col = sep_config.get("easting_col", "X")
    northing_col = sep_config.get("northing_col", "Y")
    orig_file = sep_config["original_file"]
    orig_field = sep_config["original_field"]
    reg_file = sep_config["regional_file"]
    reg_field = sep_config["regional_field"]
    unit = sep_config.get("unit", "nT")

    df_orig = pd.read_csv(ut.resolve_dataset_file(dataset_dir, orig_file))
    df_reg = pd.read_csv(ut.resolve_dataset_file(dataset_dir, reg_file))

    df_orig = df_orig.sort_values(by=[northing_col, easting_col]).reset_index(drop=True)
    df_reg = df_reg.sort_values(by=[northing_col, easting_col]).reset_index(drop=True)

    if len(df_orig) != len(df_reg):
        raise ValueError("Original and regional CSVs must have the same number of points.")

    easting = df_orig[easting_col].to_numpy()
    northing = df_orig[northing_col].to_numpy()
    d_orig = df_orig[orig_field].to_numpy()
    d_reg = df_reg[reg_field].to_numpy()
    d_res = d_orig - d_reg

    sep_name = Path(sep_config_name).stem
    out_dir = dataset_dir / "separation" / sep_name
    out_dir.mkdir(exist_ok=True, parents=True)

    df_out = df_orig.copy()
    df_out["regional"] = d_reg
    df_out["residual"] = d_res
    csv_path = out_dir / f"{sep_name}_results.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Separation results saved to: {csv_path}")

    # 3-panel plot: Observed | Regional | Residual
    N = len(d_orig)
    grid_shape = (int(np.sqrt(N)), int(np.sqrt(N)))
    X = easting.reshape(grid_shape)
    Y = northing.reshape(grid_shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, Z, title, cmap in [
        (ax1, d_orig.reshape(grid_shape), "Observed Field", "jet"),
        (ax2, d_reg.reshape(grid_shape), "Regional Field", "jet"),
        (ax3, d_res.reshape(grid_shape), "Residual Field (Obs - Reg)", "jet"),
    ]:
        c = ax.contourf(X, Y, Z, levels=21, cmap=cmap)
        fig.colorbar(c, ax=ax, label=unit)
        ax.set_title(title)
        ax.set_xlabel("Easting (m)")
        ax.set_aspect("equal")
    ax1.set_ylabel("Northing (m)")

    fig_path = out_dir / f"{sep_name}_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Separation figure saved to: {fig_path}")