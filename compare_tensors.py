import torch
import argparse
import sys
import os

def extract_tensor_from_file(data):
    """
    Extracts the tensor from the loaded .pt file data, checking for common keys.
    """
    if isinstance(data, dict):
        # Check for 'hubert' or 'hubert_feats' keys
        tensor = data.get('hubert', data.get('hubert_feats'))
        if tensor is not None:
            return tensor
        else:
            # If keys are not found, check if there's only one key in the dict
            if len(data.keys()) == 1:
                return data[list(data.keys())[0]]
            return None # Could not determine the tensor
    elif isinstance(data, torch.Tensor):
        return data # The file itself is a tensor
    return None

def compare_files(file1_path, file2_path):
    """Loads and compares two .pt files."""
    print("=======================================")
    print("        Comparing Tensor Files         ")
    print("=======================================")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print("---------------------------------------")

    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"❌ ERROR: File not found -> {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"❌ ERROR: File not found -> {file2_path}")
        return

    try:
        # --- Load and inspect File 1 ---
        print(f"\n--- Loading Tensor 1 ---")
        data1 = torch.load(file1_path, map_location='cpu')
        tensor1 = extract_tensor_from_file(data1)
        if tensor1 is None:
            print(f"❌ ERROR: Could not extract a valid tensor from {file1_path}")
            return
        print(f"  - Shape: {tensor1.shape}")
        print(f"  - Mean:  {tensor1.mean().item():.6f}")
        print(f"  - Std:   {tensor1.std().item():.6f}")

        # --- Load and inspect File 2 ---
        print(f"\n--- Loading Tensor 2 ---")
        data2 = torch.load(file2_path, map_location='cpu')
        tensor2 = extract_tensor_from_file(data2)
        if tensor2 is None:
            print(f"❌ ERROR: Could not extract a valid tensor from {file2_path}")
            return
        print(f"  - Shape: {tensor2.shape}")
        print(f"  - Mean:  {tensor2.mean().item():.6f}")
        print(f"  - Std:   {tensor2.std().item():.6f}")

        # --- Compare the tensors ---
        print("\n--- Comparison Result ---")
        if torch.equal(tensor1, tensor2):
            print("‼️  IDENTICAL: The tensors in both files are exactly the same.")
        else:
            print("✅ DIFFERENT: The tensors are different.")
            if tensor1.shape != tensor2.shape:
                print("  - Note: Tensor shapes are also different.")
            # Check for small differences
            elif torch.allclose(tensor1, tensor2):
                 print("  - Note: Tensors are not identical, but are very close in value (allclose).")


    except Exception as e:
        print(f"\n❌ ERROR: An error occurred during comparison. Details: {e}")


def main():
    """Parses command-line arguments and runs the comparison."""
    parser = argparse.ArgumentParser(
        description="Compares two .pt tensor files, checking for common keys like 'hubert' or 'hubert_feats'."
    )
    parser.add_argument("file1", type=str, help="Path to the first .pt file.")
    parser.add_argument("file2", type=str, help="Path to the second .pt file.")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    compare_files(args.file1, args.file2)

if __name__ == '__main__':
    main()
