import pandas as pd
import os
import sys

# --- Configuration ---
# Input CSV file
INPUT_CSV = 'hubert_with_listeners.csv'
# The new CSV file that will be created
OUTPUT_CSV = 'hubert_with_listeners_remapped.csv'
# The path prefix to be replaced
OLD_BASE_PATH = '/home/akio/deaf_si_2025/data/hubert/'
# The new path prefix
NEW_BASE_PATH = '/media/akio/hdd1/si_model/hubert/'

def remap_and_filter():
    """
    Remaps file paths in a CSV and filters it to include only rows
    where the remapped file path exists.
    """
    print("=============================================")
    print("     Remapping and Filtering CSV Paths     ")
    print("=============================================")
    print(f"Input file:      {INPUT_CSV}")
    print(f"Output file:     {OUTPUT_CSV}")
    print(f"Old path prefix: {OLD_BASE_PATH}")
    print(f"New path prefix: {NEW_BASE_PATH}")
    print("---------------------------------------------")

    # --- 1. Load Input CSV ---
    if not os.path.exists(INPUT_CSV):
        print(f"❌ ERROR: Input file not found: '{INPUT_CSV}'")
        print("Please place this script in the same directory as the CSV file.")
        return

    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} rows from input CSV.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load '{INPUT_CSV}'. Details: {e}")
        return

    # --- 2. Identify Path Column ---
    path_column = 'hubert' if 'hubert' in df.columns else 'feature'
    if path_column not in df.columns:
        print(f"❌ ERROR: Neither 'hubert' nor 'feature' column found in the CSV.")
        return
    print(f"Found path column to modify: '{path_column}'")

    # --- 3. Remap Paths ---
    # Create a new column with the remapped paths
    remapped_column = 'remapped_path'
    df[remapped_column] = df[path_column].str.replace(OLD_BASE_PATH, NEW_BASE_PATH, regex=False)
    print("Remapped paths to new base directory in a temporary column.")

    # --- 4. Filter by Existence ---
    print("Checking for file existence... (this may take a moment)")
    
    # Create a boolean mask where True means the file exists
    try:
        # Using a list comprehension with a simple progress indicator
        total = len(df)
        mask = []
        for i, path in enumerate(df[remapped_column]):
            mask.append(os.path.exists(path))
            # Print progress
            if (i + 1) % 100 == 0 or (i + 1) == total:
                sys.stdout.write(f"\rVerifying: {i + 1}/{total}")
                sys.stdout.flush()
        print() # Newline after progress indicator
    except Exception as e:
        print(f"\n❌ ERROR: An error occurred while checking file paths. Details: {e}")
        return

    filtered_df = df[mask].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    num_original = len(df)
    num_filtered = len(filtered_df)
    num_dropped = num_original - num_filtered

    print(f"\nVerification complete.")
    print(f"  - Original rows:            {num_original}")
    print(f"  - Rows with existing files: {num_filtered}")
    print(f"  - Rows dropped:             {num_dropped}")

    # --- 5. Save New CSV ---
    if num_filtered > 0:
        # Replace the original path column with the verified, remapped paths
        filtered_df[path_column] = filtered_df[remapped_column]
        # Drop the temporary column
        filtered_df.drop(columns=[remapped_column], inplace=True)
        
        try:
            filtered_df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n✅ Successfully saved {num_filtered} rows to '{OUTPUT_CSV}'.")
            print("\nYou can now update your config file to use this new CSV for training.")
        except Exception as e:
            print(f"\n❌ ERROR: Failed to save the new CSV file. Details: {e}")
    else:
        print("\n❌ CRITICAL: No existing files found after remapping paths. Output file was not created.")

if __name__ == '__main__':
    remap_and_filter()
