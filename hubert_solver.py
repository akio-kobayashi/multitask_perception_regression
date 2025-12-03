import torch
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any

import loss # coral_loss を含む loss.py を想定
from hubert_model import MultiTaskHubertModel, logits_to_rank # 新しいモデルとヘルパー関数

class LitHubert(pl.LightningModule):
    """
    HuBERT特徴量を用いたマルチタスク学習のためのPyTorch Lightningソルバー。
    設定ファイルに基づき、複数の予測ヘッド（順序回帰、多ラベル分類）を扱う。
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.tasks = config['model']['tasks']
        self.model = MultiTaskHubertModel(config)

        # configからタスクごとの損失の重みを取得（指定がなければ1.0）
        self.loss_weights = {
            task_name: task_cfg.get('loss_weight', 1.0)
            for task_name, task_cfg in self.tasks.items()
        }
        
        # 検証ステップの出力を保持するリスト
        self.validation_step_outputs = []

    def forward(self, hubert_feats: Tensor) -> Dict[str, Tensor]:
        return self.model(hubert_feats)

    def _calculate_loss(self, logits_dict: Dict[str, Tensor], labels_dict: Dict[str, Tensor]):
        """複数のタスクの損失を計算し、合計する"""
        total_loss = 0
        loss_dict = {}

        for task_name, task_cfg in self.tasks.items():
            if task_name not in logits_dict or task_name not in labels_dict:
                continue

            logits = logits_dict[task_name]
            labels = labels_dict[task_name]
            task_type = task_cfg['type']
            
            if task_type == 'ordinal':
                # CORAL損失を適用
                task_loss = loss.coral_loss(logits, labels)
            elif task_type == 'corn': # Added CORN loss
                # CORN損失を適用
                task_loss = loss.corn_loss(logits, labels)
            elif task_type == 'multi_label':
                # バイナリクロスエントロピー損失を適用
                task_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                raise ValueError(f"未定義のタスクタイプです: {task_type}")

            loss_dict[f'{task_name}_loss'] = task_loss
            total_loss += self.loss_weights[task_name] * task_loss
        
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx: int) -> Tensor:
        # バッチから特徴量とラベルの辞書を取得
        huberts, labels_dict, lengths, indices = batch
        
        logits_dict = self.forward(huberts)
        total_loss, loss_dict = self._calculate_loss(logits_dict, labels_dict)

        # ログ記録
        self.log('train_loss', total_loss, prog_bar=True)
        for name, value in loss_dict.items():
            self.log(f'train_{name}', value)
            
        return total_loss

    def validation_step(self, batch, batch_idx: int) -> None:
        huberts, labels_dict, lengths, indices = batch
        
        logits_dict = self.forward(huberts)
        total_loss, loss_dict = self._calculate_loss(logits_dict, labels_dict)

        self.log('val_loss', total_loss, prog_bar=True)
        for name, value in loss_dict.items():
            self.log(f'val_{name}', value)

        # --- 各タスクの精度を計算 ---
        metrics = {}
        for task_name, task_cfg in self.tasks.items():
            if task_name not in logits_dict:
                continue
            
            logits = logits_dict[task_name]
            
            if task_cfg['type'] == 'ordinal' or task_cfg['type'] == 'corn':
                # 精度計算のために、データセットから元のrankラベルを取得
                ranks = labels_dict[f'{task_name}_rank']
                preds = logits_to_rank(logits)
                correct = (preds == ranks).sum().item()
                total = ranks.size(0)
                metrics[f'val_acc_{task_name}'] = {'correct': correct, 'total': total}

            elif task_cfg['type'] == 'multi_label':
                labels = labels_dict[task_name]
                preds = torch.sigmoid(logits) > 0.5
                # 全てのラベルが一致した場合を正解とする
                correct = (preds == labels).all(dim=1).sum().item()
                total = labels.size(0)
                metrics[f'val_acc_{task_name}'] = {'correct': correct, 'total': total}
        
        self.validation_step_outputs.append(metrics)

    def on_validation_epoch_end(self) -> None:
        # validationステップ全体での精度を集計
        agg_metrics = {}
        for batch_metrics in self.validation_step_outputs:
            for key, values in batch_metrics.items():
                if key not in agg_metrics:
                    agg_metrics[key] = {'correct': 0, 'total': 0}
                agg_metrics[key]['correct'] += values['correct']
                agg_metrics[key]['total'] += values['total']
        
        # 集計した精度をログに記録
        for key, values in agg_metrics.items():
            acc = values['correct'] / values['total'] if values['total'] > 0 else 0
            self.log(key, acc, prog_bar=True)
            
        self.validation_step_outputs.clear() # メモリを解放

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
        huberts, labels_dict, lengths, indices = batch
        logits_dict = self.forward(huberts)
        
        predictions = {}
        for task_name, task_cfg in self.tasks.items():
            logits = logits_dict[task_name]
            if task_cfg['type'] == 'ordinal':
                predictions[task_name] = logits_to_rank(logits)
            elif task_cfg['type'] == 'corn': # Added CORN prediction
                predictions[task_name] = logits_to_rank(logits) # CORN prediction is same as CORAL
            elif task_cfg['type'] == 'multi_label':
                predictions[task_name] = (torch.sigmoid(logits) > 0.5).int() # Convert bool to int (0 or 1)
        return predictions

    def configure_optimizers(self):
        opt_cfg = self.config.get('optimizer', {}).copy()

        # --- デバッグコード開始 ---
        import sys
        print(f"--- DEBUG: opt_cfg before processing ---", file=sys.stderr)
        print(opt_cfg, file=sys.stderr)
        if 'lr' in opt_cfg:
            print(f"--- DEBUG: type of lr is {type(opt_cfg['lr'])} ---", file=sys.stderr)
        # --- デバッグコード終了 ---

        optimizer_type = opt_cfg.pop('type', 'Adam').lower()

        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), **opt_cfg)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), **opt_cfg)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        sched_cfg = self.config.get('scheduler', {})
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **sched_cfg),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
