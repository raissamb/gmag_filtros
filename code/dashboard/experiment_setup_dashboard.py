"""
Interactive Experimental Setup Dashboard for Gmag Filtros.
Run with:
    streamlit run code/experiment_setup_dashboard.py
"""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


# =========================================================
# 1. PATH & DATA LOADING HELPERS
# =========================================================

def find_data_folder(start_path: Path) -> Path:
    """Walks up parent directories to locate the project's 'data' folder."""
    for parent in [start_path, *start_path.parents]:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate a 'data' directory above '{start_path}'."
    )


def discover_models(data_folder: Path) -> list[str]:
    """Returns sorted model names that contain an input folder with source config."""
    models: list[str] = []
    for item in sorted(data_folder.iterdir()):
        if not item.is_dir():
            continue
        input_dir = item / "input"
        if (input_dir / "true_source_config.json").exists():
            models.append(item.name)
    return models


def load_json_if_exists(filepath: Path) -> Optional[dict[str, Any]]:
    """Loads a JSON file if present; returns None otherwise."""
    if not filepath.exists():
        return None
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_first_existing(input_dir: Path, candidates: list[str]) -> Optional[Path]:
    """Returns the first existing file path from a list of candidate filenames."""
    for name in candidates:
        path = input_dir / name
        if path.exists():
            return path
    return None


def build_sources_dataframe(
    source_cfg: dict[str, Any],
    survey_params: Optional[dict[str, Any]],
) -> pd.DataFrame:
    """Builds a summary table of all prismatic sources and their sampling counts."""
    prisms: list[list[float]] = source_cfg.get("prisms", [])
    densities: list[float] = source_cfg.get("densities", [])
    intensities: list[float] = source_cfg.get("intensities", [])
    pts_per_prism: list[int] = (
        survey_params.get("points_over_each_prism", [])
        if survey_params is not None
        else []
    )

    rows: list[dict[str, Any]] = []
    for idx, p in enumerate(prisms, start=1):
        west, east, south, north, bottom, top = p
        row: dict[str, Any] = {
            "Prism": f"P{idx}",
            "X [W, E] (m)": f"[{west:.0f}, {east:.0f}]",
            "Y [S, N] (m)": f"[{south:.0f}, {north:.0f}]",
            "Z [Bot, Top] (m)": f"[{bottom:.0f}, {top:.0f}]",
            "Width × Length (m)": f"{east - west:.0f} × {north - south:.0f}",
            "Thickness (m)": f"{top - bottom:.0f}",
            "Density (kg/m³)": densities[idx - 1] if idx - 1 < len(densities) else None,
            "Magnetization (A/m)": intensities[idx - 1] if idx - 1 < len(intensities) else None,
        }
        if idx - 1 < len(pts_per_prism):
            row["Points Over Prism"] = pts_per_prism[idx - 1]
        rows.append(row)

    return pd.DataFrame(rows)


def display_figure(path: Optional[Path], caption: str, missing_label: str) -> None:
    """Displays an image if found, or an informative placeholder."""
    if path is not None and path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: `{missing_label}`")


# =========================================================
# 2. STREAMLIT APP LAYOUT
# =========================================================

def main() -> None:
    st.set_page_config(
        page_title="Gmag Filtros — Experiment Setup",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    current_file = Path(__file__).resolve()
    try:
        data_folder = find_data_folder(current_file.parent)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    models = discover_models(data_folder)
    if not models:
        st.warning(f"No synthetic models found in `{data_folder}`.")
        return

    # --- Sidebar ---
    st.sidebar.title("Gmag Filtros")
    st.sidebar.caption("Synthetic Experiment Setup")

    selected_model = st.sidebar.selectbox("Select Model", models, index=0)
    input_dir = data_folder / selected_model / "input"

    source_cfg = load_json_if_exists(input_dir / "true_source_config.json")
    survey_params = load_json_if_exists(input_dir / "survey_parameters.json")

    if source_cfg is None:
        st.error(f"Missing `true_source_config.json` in `{input_dir}`.")
        return

    st.sidebar.divider()
    st.sidebar.markdown("**Magnetization Direction**")
    st.sidebar.write(
        f"Inclination: **{source_cfg.get('inc', 'N/A')}°**  \n"
        f"Declination: **{source_cfg.get('dec', 'N/A')}°**"
    )

    if survey_params is not None:
        region = survey_params.get("region", [])
        if len(region) == 4:
            st.sidebar.markdown("**Survey Region (m)**")
            st.sidebar.write(
                f"Easting: `{region[0]}` to `{region[1]}`  \n"
                f"Northing: `{region[2]}` to `{region[3]}`"
            )
        st.sidebar.markdown("**Reproducibility**")
        st.sidebar.write(f"Random Seed: `{survey_params.get('seed', 'N/A')}`")


    st.sidebar.divider()
    st.sidebar.markdown("**Presentation Export**")
    slide_path = input_dir / f"{selected_model}_slide_summary.png"
    if slide_path.exists():
        st.sidebar.download_button(
            label="Download Slide Summary (PNG)",
            data=slide_path.read_bytes(),
            file_name=slide_path.name,
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.sidebar.caption(
            f"`{slide_path.name}` not found. Re-run the synthetic data script to generate it."
        )
    
    
    # --- Main Header ---
    st.title(f"Experimental Setup: `{selected_model}`")

    # --- Top KPI Row ---
    if survey_params is not None:
        survey_type = str(survey_params.get("survey_type", "unknown")).capitalize()
        total_pts = survey_params.get("total_points", 0)
        unique_over = survey_params.get("total_unique_points_over_sources", 0)
        pct_over = survey_params.get("percentage_points_over_sources", 0.0)
        z_min = survey_params.get("z_min", 0.0)
        z_max = survey_params.get("z_max", 0.0)
        noise_pct = survey_params.get("noise_percentage", 0.0)

        if survey_type.lower() == "grid":
            nx = survey_params.get("nx")
            ny = survey_params.get("ny")
            spacing = survey_params.get("spacing")
            survey_detail = f"{nx}×{ny} ({spacing} m)"
        else:
            clearance = survey_params.get("base_clearance")
            survey_detail = f"Base {clearance} m"

        height_label = (
            f"{z_min:.1f} m (flat)"
            if abs(z_max - z_min) < 1e-6
            else f"{z_min:.1f} – {z_max:.1f} m"
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Survey Type", survey_type, survey_detail)
        k2.metric("Total Obs. Points", f"{total_pts:,}")
        k3.metric("Points Over Sources", f"{unique_over:,}", f"{pct_over:.1f}% of survey")
        k4.metric("Observation Height", height_label)
        k5.metric("Gaussian Noise", f"{noise_pct:.1f}%")
    else:
        st.warning(
            "`survey_parameters.json` not found for this model. "
            "Re-run the synthetic data script to populate survey metrics."
        )

    st.divider()

    # --- Tabs for Classroom Walkthrough ---
    tab_fields, tab_geometry, tab_metadata = st.tabs(
        ["Observed Fields", "Source Geometry & Sampling", "Raw Metadata"]
    )

    # Tab 1: Observed Fields
    with tab_fields:
        tfa_path = resolve_first_existing(
            input_dir,
            [f"{selected_model}_tfa_noisy.png", f"{selected_model}_tfa_scatter.png"],
        )
        gz_path = resolve_first_existing(
            input_dir,
            [f"{selected_model}_gz_noisy.png", f"{selected_model}_gz_scatter.png"],
        )
        topo_path = resolve_first_existing(
            input_dir, [f"{selected_model}_topography.png"]
        )

        col1, col2 = st.columns(2)
        with col1:
            display_figure(
                tfa_path,
                caption="Total Field Anomaly (TFA) with Noise",
                missing_label=f"{selected_model}_tfa_*.png",
            )
        with col2:
            display_figure(
                gz_path,
                caption="Vertical Gravity Component (g_z) with Noise",
                missing_label=f"{selected_model}_gz_*.png",
            )

        if topo_path is not None:
            st.subheader("Observation Surface Topography")
            c_left, c_mid, c_right = st.columns([1, 2, 1])
            with c_mid:
                display_figure(
                    topo_path,
                    caption="Draped Observation Height (m)",
                    missing_label=f"{selected_model}_topography.png",
                )

    # Tab 2: Source Geometry & Sampling
    with tab_geometry:
        footprint_path = resolve_first_existing(
            input_dir, [f"{selected_model}_source_footprint.png"]
        )
        combined_3d_path = resolve_first_existing(
            input_dir,
            [
                f"{selected_model}_3d_sources_and_surface.png",
                f"{selected_model}_3d_source_full.png",
                f"{selected_model}_3d_surface.png",
            ],
        )

        col1, col2 = st.columns(2)
        with col1:
            display_figure(
                footprint_path,
                caption="2D Source Footprints & Observation Point Coverage",
                missing_label=f"{selected_model}_source_footprint.png",
            )
        with col2:
            display_figure(
                combined_3d_path,
                caption="3D Prismatic Sources & Observation Surface",
                missing_label=f"{selected_model}_3d_sources_and_surface.png",
            )

        st.subheader("Prismatic Source Properties")
        sources_df = build_sources_dataframe(source_cfg, survey_params)
        st.dataframe(sources_df, use_container_width=True, hide_index=True)

    # Tab 3: Raw Metadata
    with tab_metadata:
        m1, m2 = st.columns(2)
        with m1:
            st.subheader("`survey_parameters.json`")
            if survey_params is not None:
                st.json(survey_params)
            else:
                st.info("File not available for this dataset.")
        with m2:
            st.subheader("`true_source_config.json`")
            st.json(source_cfg)


if __name__ == "__main__":
    main()