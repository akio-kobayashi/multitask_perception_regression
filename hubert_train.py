import warnings

# pkg_resourcesの非推奨警告を抑制 (lightning importより前に設定)
warnings.filterwarnings('ignore', 'pkg_resources is deprecated as an API', UserWarning)
warnings.filterwarnings('ignore')  # 他の警告も引き続き抑制

import os
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from hubert_dataset import HubertDataset, data_processing
from functools import partial
from hubert_solver import LitHubert
from hubert_model import logits_to_rank
import pandas as pd
import yaml
from argparse import ArgumentParser
from string import Template
from typing import Dict, Any

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('config', cfg)


def rank_to_score(rank: int) -> float:
    """Convert predicted rank (1-9) back to MOS-like score (1.0-5.0)."""
    if rank <= 0:
        return float('nan')
    return 1.0 + 0.5 * (rank - 1)


def save_validation_predictions(model: LitHubert, dataloader: data.DataLoader, dataset: HubertDataset, output_path: str):
    """Run inference on the validation loader with the best checkpoint and save predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    tasks = dataset.tasks
    records = []

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for huberts, labels_dict, lengths, indices in dataloader: # Modified to get indices
            huberts = huberts.to(device, non_blocking=True)
            logits_dict = model(huberts)
            batch_size = huberts.size(0)

            ordinal_preds = {}
            multilabel_probs = {}
            multilabel_preds = {}

            for task_name, task_cfg in tasks.items():
                logits = logits_dict[task_name]
                if task_cfg['type'] == 'ordinal':
                    ordinal_preds[task_name] = logits_to_rank(logits).cpu()
                elif task_cfg['type'] == 'multi_label':
                    probs = torch.sigmoid(logits).cpu()
                    multilabel_probs[task_name] = probs
                    multilabel_preds[task_name] = (probs > 0.5).int()

            for i in range(batch_size):
                original_idx = indices[i].item() # Get original index
                row = dataset.df.iloc[original_idx].to_dict() # Use original index to get row

                for task_name, task_cfg in tasks.items():
                    if task_cfg['type'] == 'ordinal':
                        rank = int(ordinal_preds[task_name][i].item())
                        row[f'pred_{task_name}_rank'] = rank
                        row[f'pred_{task_name}_score'] = rank_to_score(rank)
                    elif task_cfg['type'] == 'multi_label':
                        probs = multilabel_probs[task_name][i].tolist()
                        preds = multilabel_preds[task_name][i].tolist()
                        for label_idx, (prob, pred) in enumerate(zip(probs, preds)):
                            row[f'pred_{task_name}_prob_{label_idx}'] = float(prob)
                            row[f'pred_{task_name}_label_{label_idx}'] = int(pred)

                records.append(row)

    pd.DataFrame(records).to_csv(output_path, index=False)


def main(args, config: dict):
    # 1) モデルとデータローダーを準備
    model = LitHubert(config)

    # Create a partial function for the collate_fn to pass the config
    collate_fn_with_config = partial(data_processing, tasks_config=config['model']['tasks'])

    train_dataset = HubertDataset(path=config['train_path'], config=config)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['process']['num_workers'],
        pin_memory=False,
        collate_fn=collate_fn_with_config
    )

    valid_dataset = HubertDataset(path=config['valid_path'], config=config)
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['process']['num_workers'],
        pin_memory=False,
        collate_fn=collate_fn_with_config
    )

    # 2) コールバックとロガー
    checkpoint_cb = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    early_stop_cb = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    logger = TensorBoardLogger(**config['logger'])

    # 3) Trainer の作成と学習
    trainer = pl.Trainer(
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=1,
        **config['trainer']
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.checkpoint
    )

    best_path = checkpoint_cb.best_model_path
    if best_path:
        print(f"Training complete. Best model saved at: {best_path}")
        print("Evaluating best checkpoint on the validation set...")
        val_results = trainer.validate(model=model, dataloaders=valid_loader, ckpt_path=best_path)
        if val_results:
            print(val_results)
    else:
        print("Training complete, but no best checkpoint was saved.")

    predictions_path = args.val_predictions_path or config.get('evaluation', {}).get('predictions_path')
    if best_path and predictions_path:
        print(f"Saving validation predictions to {predictions_path} ...")
        best_model = LitHubert.load_from_checkpoint(best_path, config=config, map_location='cpu')
        save_validation_predictions(best_model, valid_loader, valid_dataset, predictions_path)
        print("Validation predictions saved.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--val_predictions_path', type=str, default=None,
                        help='Path to save validation predictions (CSV).')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    config = load_config(args.config)
    main(args, config)
    
