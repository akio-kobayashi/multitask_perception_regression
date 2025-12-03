import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import List, Tuple
from torch import Tensor
import coral_loss

class HubertDataset(torch.utils.data.Dataset):
    """
    データフレーム(path, feature, intelligibility)から
    precomputed HuBERT 埋め込みを読み込む Dataset
    """
    def __init__(self, path: str) -> None:
        super().__init__()
        self.df = pd.read_csv(path)

    @staticmethod
    def score_to_rank(score: float) -> int:
        # 例: スコア 1.0→ rank 1, 1.5→2, … 5.0→9
        return int(round((score - 1.0) / 0.5)) + 1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        row = self.df.iloc[idx]
        # Newer CSVs use 'hubert', original used 'feature'
        path_column = 'hubert' if 'hubert' in row else 'feature'
        data = torch.load(row[path_column], map_location='cpu')

        # MODIFICATION: Handle both 'hubert' and 'hubert_feats' keys
        if isinstance(data, dict):
            hubert = data.get('hubert', data.get('hubert_feats'))
            if hubert is None:
                raise ValueError(f"HuBERT file {row[path_column]} is a dict but contains neither 'hubert' nor 'hubert_feats' key.")
        else:
            hubert = data

        rank = self.score_to_rank(row['intelligibility'])
        return hubert, rank

def data_processing(batch: List[Tuple[Tensor, int]]):
    """
    batch: List of (hubert_feats, rank)
    Returns:
        huberts:   Tensor of shape (B, T_max, D)
        labels:    Tensor of shape (B, K-1)  (ordinal labels for coral_loss)
        ranks:     Tensor of shape (B,)
        lengths:   Tensor of shape (B,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huberts, ranks, lengths = [], [], []

    for hubert_feats, rank in batch:
        lengths.append(hubert_feats.shape[0])
        huberts.append(hubert_feats)
        ranks.append(rank)

    ranks   = torch.tensor(ranks, device=device)
    lengths = torch.tensor(lengths, device=device)
    # CORAL用のラベル行列 (B, num_classes-1)
    labels  = coral_loss.ordinal_labels(ranks, num_classes=9)

    huberts = nn.utils.rnn.pad_sequence(huberts, batch_first=True)

    return huberts, labels, ranks, lengths
