import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# --- Utility Function ---
def logits_to_rank(logits: torch.Tensor) -> torch.Tensor:
    """Converts CORAL-style logits to a rank/label."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1) + 1

# --- 1. Backbone Modules (Feature Extractors) ---
class HubertBackbone(nn.Module):
    """Simple backbone: projection + mean pooling."""
    def __init__(self, hubert_dim: int = 768, proj_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(hubert_dim, proj_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = proj_dim

    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(hubert_feats))
        x = x.mean(dim=1)
        return self.dropout(x)

class AttentionHubertBackbone(nn.Module):
    """Attention backbone: projection + Transformer Encoder + mean pooling."""
    def __init__(self, hubert_dim: int = 768, embed_dim: int = 256, n_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.hubert_proj = nn.Linear(hubert_dim, embed_dim)
        # batch_first=True is important for compatibility with DataLoader
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = embed_dim

    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hubert_proj(hubert_feats))
        x = self.transformer(x)
        feat = x.mean(dim=1)
        return self.dropout(feat)

# --- 2. Head Modules (Prediction Layers) ---
class OrdinalRegressionHead(nn.Module):
    """CORAL-style ordinal regression head."""
    def __init__(self, input_dim: int, num_classes: int = 9, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, num_classes - 1) # CORAL outputs K-1 logits
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

class MultiLabelClassificationHead(nn.Module):
    """Multi-label classification head for CB factors."""
    def __init__(self, input_dim: int, num_labels: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, num_labels)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

# --- 3. Main Multi-Task Model ---
class MultiTaskHubertModel(nn.Module):
    """
    A flexible multi-task model that combines a backbone with multiple heads.
    The architecture is determined by the config file.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config.get('model', {})
        backbone_config = model_config.get('backbone', {})
        tasks_config = model_config.get('tasks', {})

        # Instantiate Backbone
        backbone_type = backbone_config.get('type', 'simple')
        backbone_params = backbone_config.get('params', {})
        if backbone_type == 'attention':
            self.backbone = AttentionHubertBackbone(**backbone_params)
        else:
            self.backbone = HubertBackbone(**backbone_params)

        # Instantiate Heads for each task
        self.heads = nn.ModuleDict()
        for task_name, task_cfg in tasks_config.items():
            head_type = task_cfg.get('type')
            head_params = task_cfg.get('params', {})
            
            if head_type == 'ordinal':
                self.heads[task_name] = OrdinalRegressionHead(input_dim=self.backbone.output_dim, **head_params)
            elif head_type == 'multi_label':
                self.heads[task_name] = MultiLabelClassificationHead(input_dim=self.backbone.output_dim, **head_params)
            else:
                raise ValueError(f"Unknown head type in config: {head_type}")

    def forward(self, hubert_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Get shared features from the backbone
        features = self.backbone(hubert_feats)

        # 2. Get outputs from each head
        outputs = {task_name: head(features) for task_name, head in self.heads.items()}
            
        return outputs