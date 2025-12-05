import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from functools import partial
import shutil

# Temporarily add the current directory to the path to import local modules
import sys
sys.path.insert(0, '.')

try:
    from hubert_dataset import HubertDataset, data_processing
    from coral_loss import ordinal_labels
except ImportError as e:
    print(f"Error: Failed to import necessary modules. Make sure hubert_dataset.py and coral_loss.py are in the same directory.")
    print(f"Details: {e}")
    sys.exit(1)

def create_dummy_files():
    """Creates a dummy directory for hubert files and a test CSV."""
    dummy_dir = 'dummy_hubert_for_verification'
    csv_path = 'verify_data.csv'
    os.makedirs(dummy_dir, exist_ok=True)

    # Create a single dummy hubert .pt file that all rows will point to
    dummy_hubert_path = os.path.join(dummy_dir, 'dummy.pt')
    torch.save(torch.randn(10, 768), dummy_hubert_path)

    # Create the test CSV data
    csv_data = {
        'wav_path': [f'dummy_{i}.wav' for i in range(6)],
        'speaker': ['spk1'] * 6,
        'intelligibility': [1.0, 1.5, 2.3, 4.8, 5.0, np.nan],
        'hubert': [dummy_hubert_path] * 6
    }
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"Created dummy hubert file: {dummy_hubert_path}")
    print(f"Created test data file: {csv_path}")
    return dummy_dir, csv_path

def cleanup_dummy_files(dummy_dir, csv_path):
    """Removes the dummy files and directory."""
    if os.path.exists(dummy_dir):
        shutil.rmtree(dummy_dir)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    print(f"\nCleaned up dummy files and directory.")

def verify():
    """Runs the verification process."""
    print("--- Starting Dataset Verification ---")
    
    dummy_dir, csv_path = create_dummy_files()

    try:
        # 2. Define a minimal config for the test
        config = {
            'train_path': csv_path,
            'batch_size': 6, # Process all dummy data in one batch
            'model': {
                'tasks': {
                    'intelligibility': {
                        'loss': 'coral',
                        'params': {'num_classes': 9}
                    }
                }
            }
        }
        num_classes = config['model']['tasks']['intelligibility']['params']['num_classes']

        # 3. Prepare Dataset and DataLoader
        # The collate_fn needs the tasks config
        collate_fn_with_config = partial(data_processing, tasks_config=config['model']['tasks'])
        
        dataset = HubertDataset(path=config['train_path'], config=config)
        loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn_with_config)

        # 4. Get the processed batch and verify
        huberts, labels_dict, _, _ = next(iter(loader))
        
        original_df = pd.read_csv(csv_path)
        processed_ranks = labels_dict['intelligibility_rank']
        processed_levels = labels_dict['intelligibility']

        print("\n--- Verification Results ---")
        all_ok = True
        for i in range(len(original_df)):
            row = original_df.iloc[i]
            score = row['intelligibility']
            
            # Manually calculate the expected values
            if pd.isna(score):
                expected_rank = 0
            else:
                # This is the exact logic from the score_to_rank method
                expected_rank = int(round((score - 1.0) / 0.5)) + 1
            
            # Manually calculate the expected ordinal levels
            expected_levels_np = np.zeros(num_classes - 1)
            if expected_rank > 0:
                # The label should have ones up to rank-1
                for j in range(expected_rank - 1):
                    expected_levels_np[j] = 1.0

            # Get the actual values produced by the data pipeline
            actual_rank = processed_ranks[i].item()
            actual_levels_np = processed_levels[i].cpu().numpy()

            # Compare and print results
            rank_ok = (expected_rank == actual_rank)
            levels_ok = np.array_equal(expected_levels_np, actual_levels_np)
            
            print(f"\nSample {i}: Score = {score}")
            print(f"  - Rank Conversion:   Expected={expected_rank}, Got={actual_rank} -> {'OK' if rank_ok else 'FAIL'}")
            print(f"  - Levels Conversion: Expected={expected_levels_np}, Got={actual_levels_np} -> {'OK' if levels_ok else 'FAIL'}")

            if not (rank_ok and levels_ok):
                all_ok = False

        print("\n--- Summary ---")
        if all_ok:
            print("✅ All data processing steps (score -> rank -> levels) are correct.")
        else:
            print("❌ Found inconsistencies in data processing. This is likely the cause of the problem.")

    finally:
        # 5. Clean up dummy files
        cleanup_dummy_files(dummy_dir, csv_path)

if __name__ == '__main__':
    verify()
