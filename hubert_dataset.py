import torch
import torch.utils.data as data
import pandas as pd
from typing import List, Tuple, Dict, Any
from torch import Tensor
import torch.nn as nn
import os # Added import

import loss # ordinal_labels を含む loss.py を想定

class HubertDataset(torch.utils.data.Dataset):
    """
    設定ファイルに基づき、事前計算されたHuBERT埋め込みと複数の正解ラベルを読み込む。
    """
    def __init__(self, path: str, config: dict) -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        # configから、このデータセットが扱うべきタスクのリストを取得
        self.tasks = config['model']['tasks']
        self.cb_columns = ['cb1', 'cb2', 'cb3', 'cb4']

        # --- smileカラムのパスを絶対パス化 ---
        self.base_data_path = "/home/akio/deaf_si_2025/data/" # ベースパスを定義
        if 'smile' in self.df.columns:
            # 相対パスを絶対パスに変換
            self.df['smile'] = self.df['smile'].apply(lambda p: os.path.join(self.base_data_path, p) if pd.notna(p) else p)

    @staticmethod
    def score_to_rank(score: float) -> int:
        """1.0-5.0 (0.5刻み) のスコアを 1-9 のランクに変換する"""
        if pd.isna(score):
            return 0  # 不正な値はランク0とする
        return int(round((score - 1.0) / 0.5)) + 1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Any], int]: # Added int for idx
        row = self.df.iloc[idx]
        
        # HuBERT特徴量をファイルから読み込み
        loaded_data = torch.load(row['hubert'], map_location='cpu')
        
        # ★修正点: 'hubert' キーを優先してチェックする ★
        if isinstance(loaded_data, dict):
            if 'hubert' in loaded_data: # まず 'hubert' キーをチェック
                hubert = loaded_data['hubert']
            elif 'hubert_feats' in loaded_data: # 次に 'hubert_feats' キーをチェック
                hubert = loaded_data['hubert_feats']
            else:
                raise ValueError(f"HuBERT特徴量ファイル '{row['hubert']}' は辞書ですが、'hubert'または'hubert_feats'キーがありません。")
        elif isinstance(loaded_data, torch.Tensor):
            hubert = loaded_data # テンソルそのものが保存されている場合
        else:
            raise ValueError(f"HuBERT特徴量ファイル '{row['hubert']}' の形式が不明です。辞書でもテンソルでもありません。")


        # このサンプルに対する全ての正解ラベルを辞書として準備
        labels_dict = {}
        
        # 順序回帰タスクのラベルを処理
        if 'intelligibility' in self.tasks:
            score = row['intelligibility']
            rank = self.score_to_rank(score)
            labels_dict['intelligibility'] = rank
            labels_dict['intelligibility_rank'] = rank # 精度計算用に元のrankも保持

        if 'naturalness' in self.tasks:
            score = row['naturalness']
            rank = self.score_to_rank(score)
            labels_dict['naturalness'] = rank
            labels_dict['naturalness_rank'] = rank # 精度計算用に元のrankも保持

        # 多ラベル分類タスクのラベルを処理
        if 'cbs' in self.tasks:
            cbs_labels = row[self.cb_columns].values.astype(bool)
            labels_dict['cbs'] = torch.tensor(cbs_labels, dtype=torch.bool)

        return hubert, labels_dict, idx # Added idx to return

def data_processing(batch: List[Tuple[Tensor, Dict[str, Any], int]], tasks_config: Dict[str, Any]): # Added int to Tuple
    """
    マルチタスク学習用にデータを整形（collate）する関数。
    """
    huberts, lengths, indices = [], [], [] # Added indices
    # 各タスクのラベルを一時的に保持するリストの辞書を初期化
    labels_lists = {task_name: [] for task_name in tasks_config}
    if 'intelligibility' in tasks_config: labels_lists['intelligibility_rank'] = []
    if 'naturalness' in tasks_config: labels_lists['naturalness_rank'] = []

    # バッチ内の各サンプルをループ
    for hubert_feats, labels_dict, idx in batch: # Added idx
        lengths.append(hubert_feats.shape[0])
        huberts.append(hubert_feats)
        indices.append(idx) # Collect indices
        
        for task_name, label_value in labels_dict.items():
            if task_name in labels_lists:
                labels_lists[task_name].append(label_value)

    # 特徴量と長さをテンソルに変換
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    huberts_tensor = nn.utils.rnn.pad_sequence(huberts, batch_first=True)
    indices_tensor = torch.tensor(indices, dtype=torch.long) # Batch indices

    # ラベルをタスクタイプに応じて適切なテンソル形式に変換
    batched_labels = {}
    for task_name, labels in labels_lists.items():
        if not labels: continue

        if '_rank' in task_name:
            batched_labels[task_name] = torch.tensor(labels, dtype=torch.long)
        elif tasks_config.get(task_name):
            task_type = tasks_config[task_name]['type']
            if task_type == 'ordinal':
                num_classes = tasks_config[task_name]['params']['num_classes']
                ranks = torch.tensor(labels, dtype=torch.long)
                # coral_loss用のラベル形式に変換
                batched_labels[task_name] = loss.ordinal_labels(ranks, num_classes)
            elif task_type == 'multi_label':
                batched_labels[task_name] = torch.stack(labels)

    # 最終的にモデルに渡す形式
    return huberts_tensor, batched_labels, lengths_tensor, indices_tensor # Added indices_tensor