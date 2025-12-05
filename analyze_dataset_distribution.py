import pandas as pd
import numpy as np
import os

# --- Configuration ---
# The script will analyze this file.
CSV_FILE_PATH = 'hubert_with_listeners.csv'

# --- Function copied from hubert_dataset.py for self-containment ---
def score_to_rank(score: float) -> int:
    """Converts 1.0-5.0 score to 1-9 rank, as defined in the project."""
    if pd.isna(score):
        return 0  # Rank 0 is used for NaN/invalid scores.
    # This logic must be identical to the one in hubert_dataset.py
    return int(round((score - 1.0) / 0.5)) + 1

def analyze_column(df, column_name):
    """Analyzes a single column of the dataframe for potential issues."""
    print(f"\n--- Analyzing column: '{column_name}' ---")
    
    if column_name not in df.columns:
        print(f"SKIPPED: Column '{column_name}' not found in the CSV file.")
        return

    # 1. Basic statistics for the raw score
    print("\n[1. Basic Statistics for Raw Scores]")
    print(df[column_name].describe())

    # 2. Count of NaN (missing) values
    nan_count = df[column_name].isna().sum()
    total_count = len(df)
    nan_percentage = (nan_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n[2. Missing Values (NaN)]")
    print(f"{nan_count} out of {total_count} rows are NaN ({nan_percentage:.2f}%)")

    # 3. Calculate rank distribution
    rank_column_name = f'{column_name}_rank'
    try:
        df[rank_column_name] = df[column_name].apply(score_to_rank)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to apply 'score_to_rank' function. Details: {e}")
        return
    
    print(f"\n[3. Rank Distribution (0 = NaN, 1-9 = Valid Ranks)]")
    rank_distribution = df[rank_column_name].value_counts().sort_index()
    print(rank_distribution)
    
    # 4. Final check and summary for the column
    valid_ranks = df[df[rank_column_name] != 0][rank_column_name]
    if valid_ranks.empty:
        print("\n❌ CRITICAL: No valid ranks were generated from this column. The model has nothing to learn.")
    elif len(valid_ranks.unique()) == 1:
        print(f"\n❌ CRITICAL: All valid samples have the exact same rank ({valid_ranks.unique()[0]}). The model cannot learn from this.")
    else:
        print("\n✅ This column appears to provide a reasonable distribution of ranks for learning.")


def main():
    """Main function to run the analysis."""
    print("========================================")
    print("  Analyzing Dataset Distribution")
    print("========================================")
    print(f"Target file: '{os.path.abspath(CSV_FILE_PATH)}'")

    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n❌ ERROR: The file '{CSV_FILE_PATH}' was not found in the current directory.")
        print("Please copy this script to the directory containing the CSV file and run it from there.")
        return

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        analyze_column(df, 'intelligibility')
        analyze_column(df, 'naturalness')
    except Exception as e:
        print(f"\n❌ ERROR: An error occurred while processing the file: {e}")

if __name__ == '__main__':
    main()