import torch
import torch.nn as nn
import torch.nn.functional as F


def logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    """
    CORAL の出力（logits）をラベルに変換．
    logits: Tensor of shape (batch_size, num_classes - 1) のシグモイド前の値
    Returns:
        Tensor of shape (batch_size,) with values in [1, num_classes]
    """
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1) + 1


class HubertOrdinalRegressionModel(nn.Module):
    def __init__(self,
                 hubert_dim: int = 768,
                 proj_dim: int = 256,
                 num_classes: int = 9,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.proj = nn.Linear(hubert_dim, proj_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 強化版 shared_fc
        self.shared_fc = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim // 2, 1)
        )

        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def extract_features(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(hubert_feats))       # (B, T, proj_dim)
        x = x.mean(dim=1)                         # (B, proj_dim)
        return self.dropout(x)

    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(hubert_feats)
        g = self.shared_fc(feat)                  # (B, 1)
        logits = g.repeat(1, self.num_classes - 1) - self.thresholds.view(1, -1)
        return logits

    def predict(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hubert_feats)
        return logits_to_label(logits)

class AttentionHubertOrdinalRegressionModel(nn.Module):
    def __init__(self,
                 hubert_dim: int = 768,
                 embed_dim: int = 256,
                 num_classes: int = 9,
                 n_heads: int = 4,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.hubert_proj = nn.Linear(hubert_dim, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)

        # 強化版 shared_fc
        self.shared_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, 1)
        )

        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def extract_features(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hubert_proj(hubert_feats))  # (B, T, embed_dim)
        x = x.transpose(0, 1)                       # (T, B, embed_dim)
        x = self.transformer(x)                     # (T, B, embed_dim)
        x = x.transpose(0, 1)                       # (B, T, embed_dim)
        feat = x.mean(dim=1)                        # (B, embed_dim)
        return self.dropout(feat)

    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(hubert_feats)
        g = self.shared_fc(feat)                    # (B, 1)
        logits = g.repeat(1, self.num_classes - 1) - self.thresholds.view(1, -1)
        return logits

    def predict(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hubert_feats)
        return logits_to_label(logits)

class HubertCornModel(nn.Module):
    """
    CORN: Conditional Ordinal Regression Network using Japanese HuBERT embeddings.
    - 入力: hubert_feats (batch, seq_len, hubert_dim)
    - 特徴抽出: 線形射影 + 平均プーリング
    """
    def __init__(self,
                 hubert_dim: int = 768,
                 proj_dim: int = 256,
                 num_classes: int = 9,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # HuBERT埋め込みを低次元に射影
        self.proj = nn.Linear(hubert_dim, proj_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 各閾値の二値分類器
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(proj_dim // 2, 1)
            )
            for _ in range(num_classes - 1)
        ])

    def extract_features(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        """
        hubert_feats: (B, T, hubert_dim)
        -> (B, proj_dim)
        """
        x = F.relu(self.proj(hubert_feats))  # (B, T, proj_dim)
        x = x.mean(dim=1)                    # (B, proj_dim)
        return self.dropout(x)

    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits: (B, num_classes - 1)
        """
        feat = self.extract_features(hubert_feats)  # (B, proj_dim)
        logits = [clf(feat).squeeze(1) for clf in self.classifiers]
        return torch.stack(logits, dim=1)

    def predict(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hubert_feats)           # (B, K-1)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1) + 1  # (B,)


class AttentionHubertCornModel(nn.Module):
    """
    CORN with self-attention on HuBERT embeddings.
    - TransformerEncoder を1層使用
    """
    def __init__(self,
                 hubert_dim: int = 768,
                 embed_dim: int = 256,
                 num_classes: int = 9,
                 n_heads: int = 4,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # HuBERT特徴を埋め込み次元に射影
        self.hubert_proj = nn.Linear(hubert_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)

        # 二値分類器群
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim // 2, 1)
            )
            for _ in range(num_classes - 1)
        ])

    def extract_features(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        """
        hubert_feats: (B, T, hubert_dim)
        -> (B, embed_dim)
        """
        x = F.relu(self.hubert_proj(hubert_feats))  # (B, T, embed_dim)
        x = x.transpose(0,1)
        x = self.transformer(x)
        x = x.transpose(0,1)
        feat = x.mean(dim=1)
        return self.dropout(feat)

    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(hubert_feats)  # (B, embed_dim)
        logits = [clf(feat).squeeze(1) for clf in self.classifiers]
        return torch.stack(logits, dim=1)

    def predict(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hubert_feats)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1) + 1
