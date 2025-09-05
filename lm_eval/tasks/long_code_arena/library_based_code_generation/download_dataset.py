#!/usr/bin/env python3
"""
Script to download the LCA library-based code generation dataset from HuggingFace.
Run this script when you have internet access to cache the dataset locally.
"""

import os
import sys
import json
from pathlib import Path

try:
    from datasets import load_dataset
    from huggingface_hub import HfApi
except ImportError:
    print("Please install required packages: pip install datasets huggingface_hub")
    sys.exit(1)


def download_lca_dataset(output_dir="./lca_data"):
    """
    Download the LCA library-based code generation dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print("Downloading LCA library-based code generation dataset...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        dataset = load_dataset("JetBrains-Research/lca-library-based-code-generation")
        
        print(f"Dataset downloaded successfully!")
        print(f"Dataset info: {dataset}")
        
        # Save to JSON for easy access
        test_data = []
        for item in dataset["test"]:
            test_data.append(dict(item))
        
        output_file = os.path.join(output_dir, "lca_library_based_code_generation.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to: {output_file}")
        print(f"Total examples: {len(test_data)}")
        
        # Print sample
        if test_data:
            print("\nSample example:")
            print(f"Keys: {list(test_data[0].keys())}")
            print(f"Instruction: {test_data[0]['instruction'][:100]}...")
            print(f"Reference preview: {test_data[0]['reference'][:100]}...")
        
        # Update the sample_data.json with real data (first 5 examples)
        sample_file = "lm_eval/tasks/long_code_arena/sample_data.json"
        if os.path.exists(sample_file):
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(test_data[:5], f, indent=2, ensure_ascii=False)
            print(f"Updated sample data: {sample_file}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("You can still use the fallback version with sample data.")
        return False


def check_dataset_availability():
    """Check if the dataset is available online."""
    try:
        api = HfApi()
        repo_info = api.repo_info("JetBrains-Research/lca-library-based-code-generation", repo_type="dataset")
        print(f"Dataset is available. Last modified: {repo_info.last_modified}")
        return True
    except Exception as e:
        print(f"Dataset not accessible: {e}")
        return False


if __name__ == "__main__":
    print("LCA Dataset Downloader")
    print("=====================")
    
    if check_dataset_availability():
        success = download_lca_dataset()
        if success:
            print("\n✅ Dataset download completed successfully!")
            print("You can now use the main LCA task: lca_library_based_code_generation")
        else:
            print("\n❌ Dataset download failed.")
            print("Use the fallback task: lca_library_based_code_generation_fallback")
    else:
        print("\n⚠️ Dataset not accessible. Using fallback mode.")
        print("Use the fallback task: lca_library_based_code_generation_fallback")
