import torch
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any

import loss as loss_utils
from hubert_model import MultiTaskHubertModel, logits_to_rank

class LitHubert(pl.LightningModule):
    """
    A robust, simplified multi-task solver.
    It loops through configured tasks and applies the correct loss.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.model = MultiTaskHubertModel(config)
        self.tasks = config['model']['tasks']
        self.loss_weights = {
            task_name: task_cfg.get('loss_weight', 1.0)
            for task_name, task_cfg in self.tasks.items()
        }
        self.validation_step_outputs = []

    def forward(self, hubert_feats: Tensor) -> Dict[str, Tensor]:
        return self.model(hubert_feats)

    def _calculate_and_log_losses(self, step_type, logits_dict, labels_dict):
        total_loss = torch.tensor(0.0, device=self.device)

        for task_name, task_cfg in self.tasks.items():
            if task_name not in logits_dict or task_name not in labels_dict:
                continue

            logits = logits_dict[task_name]
            labels = labels_dict[task_name]
            task_type = task_cfg['type']
            task_loss = torch.tensor(0.0, device=self.device)

            if task_type == 'ordinal':
                task_loss = loss_utils.coral_loss(logits, labels)
            elif task_type == 'multi_label':
                task_loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            self.log(f'{step_type}_{task_name}_loss', task_loss, sync_dist=True)
            total_loss += self.loss_weights.get(task_name, 1.0) * task_loss
        
        self.log(f'{step_type}_loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx: int):
        huberts, labels_dict, lengths, indices = batch
        logits_dict = self.forward(huberts)
        loss = self._calculate_and_log_losses('train', logits_dict, labels_dict)
        return loss

    def validation_step(self, batch, batch_idx: int):
        huberts, labels_dict, lengths, indices = batch
        logits_dict = self.forward(huberts)
        self._calculate_and_log_losses('val', logits_dict, labels_dict)

        # Accuracy calculation (1-indexed)
        metrics = {}
        for task_name, task_cfg in self.tasks.items():
            if task_name not in logits_dict or f'{task_name}_rank' not in labels_dict:
                continue
            
            if task_cfg['type'] == 'ordinal':
                ranks = labels_dict[f'{task_name}_rank']
                preds = logits_to_rank(logits) # Returns 1-indexed ranks
                correct = (preds == ranks).sum().item()
                total = ranks.size(0)
                metrics[f'val_acc_{task_name}'] = {'correct': correct, 'total': total}
        
        self.validation_step_outputs.append(metrics)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
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
        optimizer_type = opt_cfg.pop('type', 'AdamW').lower()

        if 'lr' in opt_cfg: opt_cfg['lr'] = float(opt_cfg['lr'])
        if 'weight_decay' in opt_cfg: opt_cfg['weight_decay'] = float(opt_cfg['weight_decay'])

        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), **opt_cfg)
        else: # Default to Adam
            optimizer = torch.optim.Adam(self.model.parameters(), **opt_cfg)

        sched_cfg = self.config.get('scheduler', {})
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **sched_cfg),
            'monitor': self.config.get('checkpoint', {}).get('monitor', 'val_loss')
        }
        return [optimizer], [scheduler]