import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import List, Tuple, Dict, Any
from torch import Tensor
import loss as loss_utils

class HubertDataset(data.Dataset):
    """
    Reads precomputed HuBERT embeddings and various labels for multi-task learning.
    Based on the simple design of the original single-task dataset.
    """
    def __init__(self, path: str, config: dict):
        super().__init__()
        self.df = pd.read_csv(path)
        self.tasks = config['model']['tasks']
        self.cb_columns = [c for c in ['cb1', 'cb2', 'cb3', 'cb4'] if c in self.df.columns]

    @staticmethod
    def score_to_rank(score: float) -> int:
        """Converts 1.0-5.0 score to 1-9 rank."""
        if pd.isna(score):
            return 0  # Use 0 for invalid ranks.
        return int(round((score - 1.0) / 0.5)) + 1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Any], int]:
        row = self.df.iloc[idx]
        
        # Load Hubert features
        hubert_data = torch.load(row['hubert'], map_location='cpu')
        hubert_feats = hubert_data.get('hubert', hubert_data) if isinstance(hubert_data, dict) else hubert_data

        # Prepare a dictionary of labels for all configured tasks
        labels = {}
        for task_name in self.tasks.keys():
            if task_name == 'intelligibility' and 'intelligibility' in self.df.columns:
                labels['intelligibility'] = self.score_to_rank(row['intelligibility'])
            elif task_name == 'naturalness' and 'naturalness' in self.df.columns:
                labels['naturalness'] = self.score_to_rank(row['naturalness'])
            elif task_name == 'cbs' and self.cb_columns:
                labels['cbs'] = torch.tensor(row[self.cb_columns].values.astype(float), dtype=torch.float32)
        
        return hubert_feats, labels, idx

def data_processing(batch: List[Tuple[Tensor, Dict[str, Any], int]], tasks_config: Dict[str, Any]):
    """
    Collates a batch of multi-task data.
    This is a simplified, robust version.
    """
    huberts, all_labels, lengths, indices = [], [], [], []

    for hubert_feats, labels_dict, idx in batch:
        lengths.append(hubert_feats.shape[0])
        huberts.append(hubert_feats)
        all_labels.append(labels_dict)
        indices.append(idx)

    huberts_tensor = nn.utils.rnn.pad_sequence(huberts, batch_first=True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    batched_labels = {}
    for task_name, task_cfg in tasks_config.items():
        task_type = task_cfg['type']
        
        current_task_labels = [l.get(task_name) for l in all_labels if l.get(task_name) is not None]
        if not current_task_labels:
            continue

        if task_type == 'ordinal':
            ranks = torch.tensor(current_task_labels, dtype=torch.long)
            num_classes = task_cfg['params']['num_classes']
            # Create both the coral-style labels and keep the 1-indexed ranks
            batched_labels[task_name] = loss_utils.ordinal_labels(ranks, num_classes)
            batched_labels[f'{task_name}_rank'] = ranks
        elif task_type == 'multi_label':
            batched_labels[task_name] = torch.stack(current_task_labels)

    return huberts_tensor, batched_labels, lengths_tensor, indices_tensor
