import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# --- Individual Head Modules ---

class OrdinalRegressionHead(nn.Module):
    """
    A head for ordinal regression tasks (CORAL/CORN) that structurally
    enforces ordinal constraints.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # A shared fully-connected layer to get a single latent ranking value 'g'
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, 1)
        )

        # Learnable thresholds for ordinal classification, initialized to be ascending.
        self.thresholds = nn.Parameter(torch.arange(num_classes - 1, dtype=torch.float32) - (num_classes - 1) / 2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Get the single latent value
        g = self.shared_fc(features)  # (B, 1)
        
        # Compare the latent value with thresholds to get logits
        # This enforces the ordinal constraint structurally
        logits = g.repeat(1, self.num_classes - 1) - self.thresholds.view(1, -1)
        return logits

class MultiLabelClassificationHead(nn.Module):
    """A head for multi-label classification tasks."""
    def __init__(self, input_dim: int, num_labels: int, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, num_labels)
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

# --- Individual Backbone Modules ---

class SimpleBackbone(nn.Module):
    """A simple backbone with a linear projection and mean pooling."""
    def __init__(self, hubert_dim: int = 768, proj_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(hubert_dim, proj_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = proj_dim
    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(hubert_feats))
        x = x.mean(dim=1)
        return self.dropout(x)

class AttentionBackbone(nn.Module):
    """An attention-based backbone with a Transformer encoder."""
    def __init__(self, hubert_dim: int = 768, embed_dim: int = 256, n_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.hubert_proj = nn.Linear(hubert_dim, embed_dim)
        # Use batch_first=True for compatibility with DataLoader
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = embed_dim
    def forward(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hubert_proj(hubert_feats))
        x = self.transformer(x)
        feat = x.mean(dim=1)
        return self.dropout(feat)

# --- Main Flexible Multi-Task Model ---

class MultiTaskHubertModel(nn.Module):
    """
    A flexible multi-task model that dynamically builds from a config.
    It combines a backbone with multiple heads based on the 'tasks' dictionary.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config.get('model', {})
        backbone_config = model_config.get('backbone', {})
        tasks_config = model_config.get('tasks', {})

        # 1. Instantiate Backbone
        backbone_type = backbone_config.get('type', 'simple')
        backbone_params = backbone_config.get('params', {})
        if backbone_type == 'attention':
            self.backbone = AttentionBackbone(**backbone_params)
        else:
            self.backbone = SimpleBackbone(**backbone_params)

        # 2. Instantiate Heads for each task
        self.heads = nn.ModuleDict()
        for task_name, task_cfg in tasks_config.items():
            loss_type = task_cfg.get('loss')
            head_params = task_cfg.get('params', {})
            
            if loss_type == 'coral':
                self.heads[task_name] = OrdinalRegressionHead(input_dim=self.backbone.output_dim, **head_params)
            elif loss_type == 'bce':
                self.heads[task_name] = MultiLabelClassificationHead(input_dim=self.backbone.output_dim, **head_params)
            else:
                raise ValueError(f"Unsupported loss/head type in config: '{loss_type}' for task '{task_name}'")

    def forward(self, hubert_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Shared features
        features = self.backbone(hubert_feats)
        # 2. Task-specific outputs
        outputs = {task_name: head(features) for task_name, head in self.heads.items()}
        return outputs
