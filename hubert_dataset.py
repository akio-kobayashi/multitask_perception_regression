import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import List, Tuple, Dict, Any
from torch import Tensor
import coral_loss

class HubertDataset(data.Dataset):
    """
    Reads precomputed HuBERT embeddings and various labels for multi-task learning.
    Minimally extended from the original single-task version.
    """
    def __init__(self, path: str, config: dict):
        super().__init__()
        self.df = pd.read_csv(path)
        # The tasks to be loaded are determined by the config file
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
        
        # Load Hubert features, handling different keys
        path_column = 'hubert' if 'hubert' in row.index else 'feature'
        hubert_data = torch.load(row[path_column], map_location='cpu')
        if isinstance(hubert_data, dict):
            hubert_feats = hubert_data.get('hubert', hubert_data.get('hubert_feats'))
            if hubert_feats is None:
                raise ValueError(f"HuBERT file {row[path_column]} is a dict but has no 'hubert' or 'hubert_feats' key.")
        else:
            hubert_feats = hubert_data

        # Prepare a dictionary of labels for all configured tasks
        labels = {}
        if 'intelligibility' in self.tasks and 'intelligibility' in self.df.columns:
            labels['intelligibility'] = self.score_to_rank(row['intelligibility'])
        if 'naturalness' in self.tasks and 'naturalness' in self.df.columns:
            labels['naturalness'] = self.score_to_rank(row['naturalness'])
        if 'cbs' in self.tasks and self.cb_columns:
            # Ensure CB labels are float tensors for BCE loss
            labels['cbs'] = torch.tensor(row[self.cb_columns].values.astype(float), dtype=torch.float32)
        
        return hubert_feats, labels, idx

def data_processing(batch: List[Tuple[Tensor, Dict[str, Any], int]], tasks_config: Dict[str, Any]):
    """
    Collates a batch of multi-task data into padded tensors.
    This version correctly returns 4 items.
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

    # Batch the labels for each task into a dictionary of tensors
    batched_labels = {}
    for task_name, task_cfg in tasks_config.items():
        # Collect all labels for the current task from the batch
        current_task_labels = [l.get(task_name) for l in all_labels if l.get(task_name) is not None]
        if not current_task_labels:
            continue

        task_type = task_cfg.get('loss', 'coral')

        if task_type == 'coral':
            ranks = torch.tensor(current_task_labels, dtype=torch.long)
            num_classes = task_cfg['params']['num_classes']
            # Create coral-style labels and also keep the original ranks for accuracy calculation
            batched_labels[task_name] = coral_loss.ordinal_labels(ranks, num_classes)
            batched_labels[f'{task_name}_rank'] = ranks
        elif task_type == 'bce': # For multi-label tasks like cbs
            batched_labels[task_name] = torch.stack(current_task_labels)

    return huberts_tensor, batched_labels, lengths_tensor, indices_tensor
