import pandas as pd
import torch
import os

# --- Configuration ---
CSV_FILE_PATH = 'hubert_with_listeners.csv'
NUM_FILES_TO_CHECK = 3

def peek():
    """Loads the first few .pt files listed in the CSV and inspects their content."""
    print("=====================================================")
    print(f"  Peeking into .pt files listed in '{CSV_FILE_PATH}'")
    print("=====================================================")

    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ ERROR: '{CSV_FILE_PATH}' not found in the current directory.")
        print("Please copy this script to the directory containing the CSV and run it from there.")
        return

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"❌ ERROR: Failed to read CSV file. Details: {e}")
        return
    
    # Determine the correct column name for the file paths
    path_column = 'hubert' if 'hubert' in df.columns else 'feature'
    if path_column not in df.columns:
        print(f"❌ ERROR: Neither 'hubert' nor 'feature' column found in the CSV.")
        return

    file_paths = df[path_column].head(NUM_FILES_TO_CHECK).tolist()
    
    print(f"Found path column: '{path_column}'. Checking first {len(file_paths)} files...")
    
    tensors = []
    for i, path in enumerate(file_paths):
        print(f"\n--- Loading file {i+1}: {path} ---")
        if not os.path.exists(path):
            print(f"  ❌ ERROR: File does not exist at this path.")
            continue
            
        try:
            # Load the data from the .pt file
            data = torch.load(path, map_location='cpu')
            
            # Extract the tensor, handling both 'hubert' and 'hubert_feats' keys
            tensor = None
            if isinstance(data, dict):
                tensor = data.get('hubert', data.get('hubert_feats'))
            else:
                tensor = data # Assume the file itself is the tensor
            
            if tensor is None:
                print(f"  ❌ ERROR: Loaded data is a dict, but contains neither 'hubert' nor 'hubert_feats' key.")
                continue

            tensors.append(tensor)
            print(f"  - Tensor Shape: {tensor.shape}")
            print(f"  - Tensor Mean:  {tensor.mean().item():.6f}")
            print(f"  - Tensor Std:   {tensor.std().item():.6f}")
            print(f"  - Tensor Sum:   {tensor.sum().item():.6f}")

        except Exception as e:
            print(f"  ❌ ERROR: Failed to load or inspect file. Details: {e}")

    # Compare the loaded tensors to see if they are identical
    if len(tensors) > 1:
        print("\n--- Comparing Tensors ---")
        if torch.equal(tensors[0], tensors[1]):
            print("  - ‼️  CRITICAL: Tensor 1 and Tensor 2 are IDENTICAL.")
        else:
            print("  - ✅ Tensor 1 and Tensor 2 are different.")
    
    if len(tensors) > 2:
        if torch.equal(tensors[0], tensors[2]):
            print("  - ‼️  CRITICAL: Tensor 1 and Tensor 3 are IDENTICAL.")
        else:
            print("  - ✅ Tensor 1 and Tensor 3 are different.")
            
    print("\n--- Done ---")


if __name__ == '__main__':
    peek()
