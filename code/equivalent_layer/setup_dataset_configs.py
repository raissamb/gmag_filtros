"""
Generates equivalent layer and transformation JSON config files
inside data/<DATASET_NAME>/config/ from the templates in modules/constants.py.
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"

import modules.constants as ct

# =========================================================
# 1. CONFIGURATION (Edit this section)
# =========================================================
DATASET_NAME = "model1"

# List one or more equivalent layer config names (e.g. ["eq_layer1", "eq_layer2"])
EQ_LAYER_NAMES = ["eq_layer1"]

# Select transformations to generate (must match keys in ct.TRANSFORMATION_TEMPLATES)
TRANSFORMATIONS = [
    "upward_continuation",
    "downward_continuation",
    "reduction_to_pole",
    "horizontal_gradient_magnitude",
    "poisson",
]

# Safety switch: False skips existing files so tuned configs aren't overwritten
OVERWRITE = False


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def save_json_safely(data: dict[str, Any], filepath: Path, overwrite: bool) -> str:
    """Writes a JSON file unless it already exists and overwrite is False."""
    if filepath.exists() and not overwrite:
        return f"SKIPPED (already exists): {filepath.name}"

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return f"CREATED: {filepath.name}"


def build_prefilled_eq_layer_config(dataset_name: str, input_dir: Path) -> dict[str, Any]:
    """
    Returns a deep copy of the equivalent layer template, pre-filled with
    the dataset's CSV filenames and true_source_config inc/dec when available.
    """
    cfg = copy.deepcopy(ct.eq_layer_config)

    # Pre-fill gravity and magnetic filenames
    grav_name = f"{dataset_name}_grav.csv"
    mag_name = f"{dataset_name}_mag.csv"
    cfg["project_settings"]["gravity_file"] = grav_name
    cfg["project_settings"]["magnetic_file"] = mag_name

    # Pre-fill inc/dec from true_source_config.json if present (synthetic datasets)
    source_cfg_path = input_dir / "true_source_config.json"
    if source_cfg_path.exists():
        with source_cfg_path.open("r", encoding="utf-8") as f:
            source_cfg = json.load(f)
        inc = source_cfg.get("inc")
        dec = source_cfg.get("dec")
        if inc is not None and dec is not None:
            cfg["inversion_parameters"]["incf"] = inc
            cfg["inversion_parameters"]["decf"] = dec
            cfg["inversion_parameters"]["incs"] = inc
            cfg["inversion_parameters"]["decs"] = dec

    return cfg


# =========================================================
# 3. EXECUTION WORKFLOW
# =========================================================

if __name__ == "__main__":
    dataset_dir = DATA_FOLDER / DATASET_NAME
    input_dir = dataset_dir / "input"
    config_dir = dataset_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up configs for dataset '{DATASET_NAME}' in: {config_dir}\n")

    # 1. Generate Equivalent Layer configs
    for eq_name in EQ_LAYER_NAMES:
        eq_cfg = build_prefilled_eq_layer_config(DATASET_NAME, input_dir)
        status = save_json_safely(eq_cfg, config_dir / f"{eq_name}.json", OVERWRITE)
        print(f"  [EQ Layer] {status}")

    # 2. Generate Transformation configs
    default_eq = EQ_LAYER_NAMES[0] if EQ_LAYER_NAMES else "eq_layer1"
    for t_name in TRANSFORMATIONS:
        if t_name not in ct.TRANSFORMATION_TEMPLATES:
            available = ", ".join(ct.TRANSFORMATION_TEMPLATES.keys())
            print(f"  [WARNING] Unknown transformation '{t_name}'. Available: {available}")
            continue

        t_cfg = copy.deepcopy(ct.TRANSFORMATION_TEMPLATES[t_name])
        t_cfg["eq_layer_config"] = default_eq
        status = save_json_safely(t_cfg, config_dir / f"{t_name}.json", OVERWRITE)
        print(f"  [Transform] {status}")

    print("\nConfig setup complete.")