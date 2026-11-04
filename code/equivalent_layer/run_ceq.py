"""
This is the main script that performs the equivalent layer and transformations on a dataset.
"""

import sys
from pathlib import Path

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Import pipeline stages
import inversion_stage as inv_stage
import transformation_stage as trans_stage

def main(dataset_name: str):
    print(f"==== Processing Dataset: {dataset_name} ====\n")
    
    # Execute the stages
    inv_stage.run_inversion(dataset_name)
    trans_stage.run_transformations(dataset_name)
    
    print("\n==== Pipeline Complete ====")

if __name__ == "__main__":
    # WRITE HERE THE FOLDER NAME OF WHERE THE DATASET IS! 
    # THEN RUN THIS FILE TO EXECUTE THE PIPELINE
    
    target_dataset = "model1"  
    main(target_dataset)