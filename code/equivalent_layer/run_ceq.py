import sys
import argparse
from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import inversion_stage as inv_stage
import transformation_stage as trans_stage
import poisson_stage

def main(dataset_name: str, data_type: str):
    
    # Standardize data_type for all downstream file naming
    if data_type == "mag":
        data_type = "magnetic"
    elif data_type == "grav":
        data_type = "gravity"
    
    print(f"==== Processing Dataset: {dataset_name} | Type: {data_type.upper()} ====\n")
    
    inv_stage.run_inversion(dataset_name, data_type, None, None)
    trans_stage.run_transformations(dataset_name, data_type)
    poisson_stage.run_poisson_ratio(dataset_name)
    
    print("\n==== Pipeline Complete ====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Equivalent Layer Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Target dataset folder name (e.g., model2)")
    parser.add_argument("--type", type=str, required=True, choices=["mag", "magnetic", "grav", "gravity"], help="Data type to process")
    
    args = parser.parse_args()
    main(args.dataset, args.type)