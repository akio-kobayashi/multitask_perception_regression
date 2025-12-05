import torch
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any

# The flexible model and a helper function for accuracy calculation
from hubert_model import MultiTaskHubertModel
from coral_loss import logits_to_label 

# The loss functions
import coral_loss
import corn_loss

class LitHubert(pl.LightningModule):
    """
    The definitive, robust, multi-task solver designed to work with
    the flexible MultiTaskHubertModel.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # The model internally configures its backbone and heads from the config
        self.model = MultiTaskHubertModel(config)
        
        # Get task definitions for the solver's loops
        self.tasks = config['model']['tasks']
        self.loss_weights = {
            task_name: task_cfg.get('loss_weight', 1.0)
            for task_name, task_cfg in self.tasks.items()
        }
        self.validation_step_outputs = []

    def forward(self, hubert_feats: Tensor) -> Dict[str, Tensor]:
        return self.model(hubert_feats)

    def _calculate_and_log_losses(self, step_type: str, logits_dict: Dict, labels_dict: Dict) -> Tensor:
        total_loss = torch.tensor(0.0, device=self.device)

        for task_name, task_cfg in self.tasks.items():
            if task_name not in logits_dict or task_name not in labels_dict:
                continue

            logits = logits_dict[task_name]
            labels = labels_dict[task_name]
            loss_type = task_cfg.get('loss')
            task_loss = torch.tensor(0.0, device=self.device)

            if loss_type == 'coral':
                task_loss = coral_loss.coral_loss(logits, labels)
            elif loss_type == 'bce':
                task_loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                # For now, we only support coral and bce.
                # We can add corn back later if needed.
                continue
            
            self.log(f'{step_type}_{task_name}_loss', task_loss, sync_dist=True)
            total_loss += self.loss_weights.get(task_name, 1.0) * task_loss
        
        self.log(f'{step_type}_loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        huberts, labels_dict, lengths, indices = batch
        logits_dict = self.forward(huberts)
        loss = self._calculate_and_log_losses('train', logits_dict, labels_dict)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        huberts, labels_dict, lengths, indices = batch
        logits_dict = self.forward(huberts)
        self._calculate_and_log_losses('val', logits_dict, labels_dict)

        # Accuracy calculation (using 1-indexed labels)
        metrics = {}
        for task_name, task_cfg in self.tasks.items():
            if task_name not in logits_dict or f'{task_name}_rank' not in labels_dict:
                continue
            
            if task_cfg.get('loss') == 'coral':
                ranks = labels_dict[f'{task_name}_rank']
                preds = logits_to_label(logits_dict[task_name]) # 1-indexed
                correct = (preds == ranks).sum().item()
                total = ranks.size(0)
                metrics[f'val_acc_{task_name}'] = {'correct': correct, 'total': total}
        
        self.validation_step_outputs.append(metrics)

    def on_validation_epoch_end(self) -> None:
        if not self.validation_step_outputs: return
        
        agg_metrics = {}
        for batch_metrics in self.validation_step_outputs:
            for key, values in batch_metrics.items():
                if key not in agg_metrics:
                    agg_metrics[key] = {'correct': 0, 'total': 0}
                agg_metrics[key]['correct'] += values['correct']
                agg_metrics[key]['total'] += values['total']
        
        for key, values in agg_metrics.items():
            acc = values['correct'] / values['total'] if values['total'] > 0 else 0
            self.log(key, acc, prog_bar=True, sync_dist=True)
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        opt_cfg = self.config.get('optimizer', {}).copy()
        optimizer_type = opt_cfg.pop('type', 'Adam').lower()

        if 'lr' in opt_cfg: opt_cfg['lr'] = float(opt_cfg['lr'])
        if 'weight_decay' in opt_cfg: opt_cfg['weight_decay'] = float(opt_cfg['weight_decay'])
        if 'betas' in opt_cfg: opt_cfg['betas'] = tuple(float(b) for b in opt_cfg['betas'])

        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), **opt_cfg)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), **opt_cfg)

        sched_cfg = self.config.get('scheduler', {})
        if 'factor' in sched_cfg: sched_cfg['factor'] = float(sched_cfg['factor'])
        if 'patience' in sched_cfg: sched_cfg['patience'] = int(sched_cfg['patience'])
            
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **sched_cfg),
            'monitor': self.config.get('checkpoint', {}).get('monitor', 'val_loss')
        }
        return [optimizer], [scheduler]
