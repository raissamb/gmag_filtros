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

def main(dataset_name: str, eq_config: str, trans_config: str, poisson_config: str, data_type: str):
    if data_type == "mag":
        data_type = "magnetic"
    elif data_type == "grav":
        data_type = "gravity"
        
    print(f"==== Processing Dataset: {dataset_name} ====\n")
    
    if eq_config:
        inv_stage.run_inversion(dataset_name, data_type, eq_config, None, None)
    if trans_config:
        trans_stage.run_transformations(dataset_name, trans_config)
    if poisson_config:
        poisson_stage.run_poisson_ratio(dataset_name, poisson_config)
        
    print("\n==== Pipeline Complete ====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Equivalent Layer Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Target dataset folder name (e.g., model2)")
    parser.add_argument("--type", type=str, choices=["mag", "magnetic", "grav", "gravity"], help="Data type to process")
    parser.add_argument("--eq-config", type=str, help="Equivalent layer config file (e.g., eq_layer1.json)")
    parser.add_argument("--trans-config", type=str, help="Transformation config file (e.g., rtp1.json)")
    parser.add_argument("--poisson-config", type=str, help="Poisson ratio config file")
    
    args = parser.parse_args()
    main(args.dataset, args.eq_config, args.trans_config, args.poisson_config, args.type)

