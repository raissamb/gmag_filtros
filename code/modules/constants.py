"""
Module that contains the constants that need to be used in the another modules or scripts.
"""

# Add noise to synthetic model
# Gaussian noise: zero mean value hard coded into the function
# Std value depends on a given percentage of the maximum absolute value of the data
noise_percentage = 3
seeds = [12, 21, 56, 78, 80] # number of submodels


# Template for equivalent layer config file

eq_layer_config = {
    "project_settings": {
        "gravity_file": "NOME_DO_ARQUIVO_GRAV.csv",
        "magnetic_file": "NOME_DO_ARQUIVO_MAG.csv"
    },
    "data_columns": {
        "easting_col": "X",
        "northing_col": "Y",
        "upwards_col": "Z",
        "field": "obs"
    },
    "inversion_parameters": {
        "z_top": -200.0,
        "z_bottom": -1000.0,
        "incf": 45.0,
        "decf": 12.0,
        "incs": 45.0,
        "decs": 12.0,
        "reg_min": 1e-6,
        "reg_max": 1e2,
        "steps": 20
    }
}