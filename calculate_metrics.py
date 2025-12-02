import pandas as pd
import numpy as np
import os
import argparse
import fnmatch
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, kendalltau

def calculate_and_print_metrics(csv_file_path: str):
    """
    指定されたCSVファイルからintelligibilityとnaturalnessのMAE, Spearman, Kendallを計算し表示する。
    """
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"エラー: ファイル '{csv_file_path}' の読み込み中に問題が発生しました: {e}")
        return

    print(f"\n--- Metrics for: {csv_file_path} ---")

    # 評価対象のタスクを動的に検出
    tasks_to_evaluate = []
    if 'intelligibility' in df.columns and 'pred_intelligibility_score' in df.columns:
        tasks_to_evaluate.append('intelligibility')
    if 'naturalness' in df.columns and 'pred_naturalness_score' in df.columns:
        tasks_to_evaluate.append('naturalness')

    if not tasks_to_evaluate:
        print("  評価可能なタスク (intelligibility, naturalness) が見つかりませんでした。")
        return

    for task_name in tasks_to_evaluate:
        true_scores = df[task_name].dropna()
        pred_scores = df[f'pred_{task_name}_score'].dropna()

        # NaN値を除外した後のデータ数が一致することを確認
        common_indices = true_scores.index.intersection(pred_scores.index)
        if len(common_indices) == 0:
            print(f"  タスク '{task_name}': 比較可能なデータがありません。")
            continue
        
        true_scores = true_scores.loc[common_indices]
        pred_scores = pred_scores.loc[common_indices]

        if len(true_scores) < 2: # 相関係数計算には最低2点必要
            print(f"  タスク '{task_name}': データ点が少なすぎるため、メトリクスを計算できません。")
            continue

        try:
            mae = mean_absolute_error(true_scores, pred_scores)
            spearman_rho, _ = spearmanr(true_scores, pred_scores)
            kendall_tau, _ = kendalltau(true_scores, pred_scores)

            print(f"  タスク: {task_name}")
            print(f"    MAE: {mae:.4f}")
            print(f"    Spearman's rho: {spearman_rho:.4f}")
            print(f"    Kendall's tau: {kendall_tau:.4f}")
        except Exception as e:
            print(f"  タスク '{task_name}': メトリクス計算中にエラーが発生しました: {e}")


def main():
    parser = argparse.ArgumentParser(description="指定されたディレクトリから予測CSVファイルを再帰的に探し、評価指標を計算します。")
    parser.add_argument('directory', type=str, help="予測CSVファイルを探すルートディレクトリのパス")
    args = parser.parse_args()

    root_dir = args.directory
    if not os.path.isdir(root_dir):
        print(f"エラー: '{root_dir}' は有効なディレクトリではありません。")
        return

    found_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, 'predictions_*.csv'):
            found_files.append(os.path.join(dirpath, filename))

    if not found_files:
        print(f"'{root_dir}' 以下に 'predictions_*.csv' ファイルは見つかりませんでした。")
        return

    for file_path in sorted(found_files):
        calculate_and_print_metrics(file_path)

if __name__ == '__main__':
    main()
