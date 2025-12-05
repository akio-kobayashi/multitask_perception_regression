#!/bin/bash

# --- Configuration ---
# ベースとなる設定ファイル (例: configs/config_intel_nat_cbs.yml)
# 必要に応じて変更してください
#TAG="intel_nat_cbs"
TAG="intel_only"
BASE_CONFIG=configs/config_${TAG}.yml 
MAIN_DATA_CSV="hubert_with_listeners.csv"      # メインのデータセットCSV
OUTPUT_ROOT_DIR=one_speaker_out_runs/${TAG}    # 各実行結果を保存するルートディレクトリ

# --- Create output root directory ---
mkdir -p "$OUTPUT_ROOT_DIR"

# --- Get unique speakers from the main data CSV ---
echo "Extracting unique speakers from $MAIN_DATA_CSV..."
# Pythonを使ってユニークな話者リストを取得
UNIQUE_SPEAKERS=$(python3 -c "
import pandas as pd
import os

# 絶対パスで指定 (現在のディレクトリがanalysis/なので、親ディレクトリのCSV)
main_data_path = os.path.join(os.getcwd(), '$MAIN_DATA_CSV')
df = pd.read_csv(main_data_path)
speakers = df['speaker'].unique().tolist()
print(' '.join(speakers))
")

if [ -z "$UNIQUE_SPEAKERS" ]; then
    echo "Error: No unique speakers found in $MAIN_DATA_CSV. Exiting."
    exit 1
fi

echo "Found speakers: $UNIQUE_SPEAKERS"

# --- Create Python helper script for splitting data ---
SPLIT_SCRIPT="split_data.py"
cat <<EOF > "$SPLIT_SCRIPT"
import pandas as pd
import argparse
import os

def split_data(input_csv, output_train_csv, output_valid_csv, target_speaker):
    df = pd.read_csv(input_csv)
    df_train = df.query('speaker != @target_speaker')
    df_valid = df.query('speaker == @target_speaker')

    df_train.to_csv(output_train_csv, index=False)
    df_valid.to_csv(output_valid_csv, index=False)

    print(f"Split data for speaker '{target_speaker}':")
    print(f"  Train samples: {len(df_train)}")
    print(f"  Valid samples: {len(df_valid)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split data into train/validation for one-speaker-out.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the main dataset CSV.")
    parser.add_argument('--output_train_csv', type=str, required=True, help="Path for the output training CSV.")
    parser.add_argument('--output_valid_csv', type=str, required=True, help="Path for the output validation CSV.")
    parser.add_argument('--target_speaker', type=str, required=True, help="Speaker ID to use for validation.")
    args = parser.parse_args()

    split_data(args.input_csv, args.output_train_csv, args.output_valid_csv, args.target_speaker)
EOF

# --- Loop through each speaker ---
for TARGET_SPEAKER in $UNIQUE_SPEAKERS; do
    echo "--- Processing speaker: $TARGET_SPEAKER (held out for validation) ---" 
    
    RUN_OUTPUT_DIR="$OUTPUT_ROOT_DIR/$TARGET_SPEAKER"
    mkdir -p "$RUN_OUTPUT_DIR"

    TRAIN_CSV="$RUN_OUTPUT_DIR/train_${TARGET_SPEAKER}.csv"
    VALID_CSV="$RUN_OUTPUT_DIR/valid_${TARGET_SPEAKER}.csv"
    TEMP_CONFIG="$RUN_OUTPUT_DIR/config_${TARGET_SPEAKER}.yml"
    PREDICTIONS_CSV="${RUN_OUTPUT_DIR}/predictions_${TARGET_SPEAKER}.csv"
    
    # 1. Split data
    echo "Splitting data for $TARGET_SPEAKER..."
    python3 "$SPLIT_SCRIPT" \
        --input_csv "$MAIN_DATA_CSV" \
        --output_train_csv "$TRAIN_CSV" \
        --output_valid_csv "$VALID_CSV" \
        --target_speaker "$TARGET_SPEAKER"

    # 2. Prepare temporary config file
    echo "Preparing config for $TARGET_SPEAKER..."
    # Copy base config and update paths
    cp "$BASE_CONFIG" "$TEMP_CONFIG"
    # macOS と Linux の sed の違いに対応
    if [[ "$OSTYPE" == "darwin"* ]]; then # macOS
        sed -i '' "s|train_path: .*|train_path: ${TRAIN_CSV}|" "$TEMP_CONFIG"
        sed -i '' "s|valid_path: .*|valid_path: ${VALID_CSV}|" "$TEMP_CONFIG"
        sed -i '' "s|name: multi_task_hubert|name: multi_task_hubert_${TARGET_SPEAKER}|" "$TEMP_CONFIG"
        sed -i '' "s|dirpath: \"checkpoints\"|dirpath: \"${RUN_OUTPUT_DIR}/checkpoints\"|" "$TEMP_CONFIG"
        sed -i '' "s|save_dir: \"logs\"|save_dir: \"${RUN_OUTPUT_DIR}/logs\"|" "$TEMP_CONFIG"
	# MODIFICATION: Use Python script to update output_csv reliably
    python3 multitask_perception_regression/update_config.py \
        --config_path "$TEMP_CONFIG" \
        --output_csv_value "$PREDICTIONS_CSV"
    # --- DEBUGGING START ---
    echo "--- DEBUG: Content of TEMP_CONFIG after sed for speaker $TARGET_SPEAKER ---"
    cat "$TEMP_CONFIG"
    echo "-------------------------------------------------------------------------"
    # --- DEBUGGING END ---
    else # Linux など
        sed -i "s|train_path: .*|train_path: ${TRAIN_CSV}|" "$TEMP_CONFIG"
        sed -i "s|valid_path: .*|valid_path: ${VALID_CSV}|" "$TEMP_CONFIG"
        sed -i "s|name: multi_task_hubert|name: multi_task_hubert_${TARGET_SPEAKER}|" "$TEMP_CONFIG"
        sed -i "s|dirpath: \"checkpoints\"|dirpath: \"${RUN_OUTPUT_DIR}/checkpoints\"|" "$TEMP_CONFIG"
        sed -i "s|save_dir: \"logs\"|save_dir: \"${RUN_OUTPUT_DIR}/logs\"|" "$TEMP_CONFIG"
	sed -i "s|predictions_path: .*|predictions_path: ${PREDICTIONS_CSV}|" "$TEMP_CONFIG"	
    fi


    # 3. Run training
    echo "Starting training for $TARGET_SPEAKER (output to $RUN_OUTPUT_DIR/logs)..."
    python3 hubert_train.py --config "$TEMP_CONFIG"

    echo "--- Finished processing speaker: $TARGET_SPEAKER ---"
done

# --- Cleanup ---
rm "$SPLIT_SCRIPT"
echo "All one-speaker-out runs completed."
