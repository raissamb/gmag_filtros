"""
Module that contains the constants and configuration templates
used across the Gmag Filtros modules and scripts.
"""

from typing import Any

# Add noise to synthetic model
# Gaussian noise: zero mean value hard coded into the function
# Std value depends on a given percentage of the maximum absolute value of the data
noise_percentage = 3
seeds = [12, 21, 56, 78, 80]  # number of submodels


# =========================================================
# EQUIVALENT LAYER CONFIG TEMPLATE
# =========================================================

eq_layer_config: dict[str, Any] = {
    "project_settings": {
        "gravity_file": "NOME_DO_ARQUIVO_GRAV.csv",
        "magnetic_file": "NOME_DO_ARQUIVO_MAG.csv",
    },
    "data_columns": {
        "easting_col": "X",
        "northing_col": "Y",
        "upwards_col": "Z",
        "field": "obs",
    },
    "inversion_parameters": {
        "z_top": -200.0,
        "z_bottom": -1000.0,
        "incf": 45.0,
        "decf": 12.0,
        "incs": 45.0,
        "decs": 12.0,
        "solver": "cholesky",
        "reg_selector": "l_curve",
        "noise_sigma": None,
        "reg_min": 1e-6,
        "reg_max": 1e2,
        "steps": 20,
    },
}


# =========================================================
# TRANSFORMATION CONFIG TEMPLATES
# =========================================================
# Set "output_field" to a custom string (e.g. "tfa_up500") to override
# the default output filename suffix, or leave as None for default naming.

TRANSFORMATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "upward_continuation": {
        "eq_layer_config": "eq_layer1",
        "transformation_type": "upward_continuation",
        "data_type": "magnetic",
        "execute": True,
        "parameters": {
            "dz": 500.0,
            "output_field": None,
        },
    },
    "downward_continuation": {
        "eq_layer_config": "eq_layer1",
        "transformation_type": "downward_continuation",
        "data_type": "magnetic",
        "execute": False,
        "parameters": {
            "dz": -50.0,
            "output_field": None,
        },
    },
    "reduction_to_pole": {
        "eq_layer_config": "eq_layer1",
        "transformation_type": "reduction_to_pole",
        "data_type": "magnetic",
        "execute": True,
        "parameters": {
            "inc_new": 90.0,
            "dec_new": 0.0,
            "output_field": None,
        },
    },
    "horizontal_gradient_magnitude": {
        "eq_layer_config": "eq_layer1",
        "transformation_type": "horizontal_gradient_magnitude",
        "data_type": "magnetic",
        "execute": True,
        "parameters": {
            "delta": 1.0,
            "output_field": None,
        },
    },
    "poisson": {
        "eq_layer_config": "eq_layer1",
        "transformation_type": "poisson",
        "data_type": "both",
        "execute": True,
        "parameters": {
            "output_field": None,
        },
    },
    # --- Planned transformations (uncomment as colleagues implement them) ---
    # "vertical_derivative": {
    #     "eq_layer_config": "eq_layer1",
    #     "transformation_type": "vertical_derivative",
    #     "data_type": "magnetic",
    #     "execute": True,
    #     "parameters": {
    #         "order": 1,
    #         "output_field": None,
    #     },
    # },
    # "analytical_signal": {
    #     "eq_layer_config": "eq_layer1",
    #     "transformation_type": "analytical_signal",
    #     "data_type": "magnetic",
    #     "execute": True,
    #     "parameters": {
    #         "output_field": None,
    #     },
    # },
    # "tilt_angle": {
    #     "eq_layer_config": "eq_layer1",
    #     "transformation_type": "tilt_angle",
    #     "data_type": "magnetic",
    #     "execute": True,
    #     "parameters": {
    #         "output_field": None,
    #     },
    # },
    # "euler_deconvolution": {
    #     "eq_layer_config": "eq_layer1",
    #     "transformation_type": "euler_deconvolution",
    #     "data_type": "magnetic",
    #     "execute": True,
    #     "parameters": {
    #         "structural_index": 1.0,
    #         "window_size": 5,
    #         "output_field": None,
    #     },
    # },
}