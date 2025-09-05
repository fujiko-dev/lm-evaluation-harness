"""
Data loader for Long Code Arena library-based code generation task.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
import pandas as pd


def load_lca_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """
    Load the LCA library-based code generation dataset.
    
    Args:
        cache_dir: Optional cache directory for storing dataset
        
    Returns:
        Dataset object
    """
    # First try to load from local parquet file
    local_parquet_path = "/Users/xiaoyunting/Documents/ModelDev/working_path/datasets/long_code_arena/datasets/library_based_code_generation/data/test-00000-of-00001-518ed46ecbe35ff9.parquet"
    
    if os.path.exists(local_parquet_path):
        print(f"Loading dataset from local parquet file: {local_parquet_path}")
        try:
            df = pd.read_parquet(local_parquet_path)
            # Convert numpy arrays to lists for JSON serialization
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            return Dataset.from_pandas(df)
        except Exception as e:
            print(f"Warning: Could not load local parquet file: {e}")
    
    try:
        # Try to load from HuggingFace
        dataset = load_dataset(
            "JetBrains-Research/lca-library-based-code-generation",
            cache_dir=cache_dir
        )
        return dataset["test"]
    except Exception as e:
        print(f"Warning: Could not load dataset from HuggingFace: {e}")
        return _create_fallback_dataset()


def _create_fallback_dataset() -> Dataset:
    """
    Create a fallback dataset with sample data for testing purposes.
    This should be replaced with actual data when the dataset is accessible.
    """
    sample_data = [
        {
            "instruction": "Create a function that uses pandas to read a CSV file and return the number of rows",
            "reference": "import pandas as pd\n\ndef count_rows_in_csv(filename):\n    df = pd.read_csv(filename)\n    return len(df)",
            "repo_full_name": "sample/pandas-example",
            "libraries": ["pandas"],
            "difficulty": "easy"
        },
        {
            "instruction": "Write a function using NumPy to calculate the mean of a 2D array along axis 0",
            "reference": "import numpy as np\n\ndef calculate_mean_axis0(arr):\n    return np.mean(arr, axis=0)",
            "repo_full_name": "sample/numpy-example",
            "libraries": ["numpy"],
            "difficulty": "medium"
        },
        {
            "instruction": "Create a function using matplotlib to create a simple line plot",
            "reference": "import matplotlib.pyplot as plt\n\ndef create_line_plot(x, y):\n    plt.figure()\n    plt.plot(x, y)\n    plt.show()",
            "repo_full_name": "sample/matplotlib-example",
            "libraries": ["matplotlib"],
            "difficulty": "medium"
        }
    ]
    
    return Dataset.from_list(sample_data)


def save_dataset_locally(dataset: Dataset, output_path: str):
    """
    Save dataset to local JSON file.
    
    Args:
        dataset: Dataset to save
        output_path: Path to save the dataset
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    for item in dataset:
        data.append(dict(item))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_local_dataset(json_path: str) -> Dataset:
    """
    Load dataset from local JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dataset object
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Dataset.from_list(data)


def validate_dataset_format(dataset: Dataset) -> bool:
    """
    Validate that the dataset has the required fields.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        True if dataset format is valid
    """
    required_fields = ["instruction", "reference"]
    
    if len(dataset) == 0:
        return False
    
    sample = dataset[0]
    
    for field in required_fields:
        if field not in sample:
            print(f"Missing required field: {field}")
            return False
    
    return True
