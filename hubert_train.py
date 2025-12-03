import os
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from hubert_dataset import HubertDataset, data_processing
from hubert_solver import LitHubert
import pandas as pd
import yaml
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('config', cfg)

def main(args, config: dict):
    model = LitHubert(config)
    train_dataset = HubertDataset(path=config['train_path'])
    train_loader = data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['process']['num_workers'], pin_memory=False, collate_fn=data_processing
    )
    valid_dataset = HubertDataset(path=config['valid_path'])
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['process']['num_workers'], pin_memory=False, collate_fn=data_processing
    )
    checkpoint_cb = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer(
        callbacks=[checkpoint_cb], logger=logger, num_sanity_val_steps=0,
        accelerator='gpu', devices=1, **config['trainer']
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=args.checkpoint
    )
    best_path = checkpoint_cb.best_model_path
    if best_path:
        best = LitHubert.load_from_checkpoint(best_path, config=config)
        best.eval()
        df = pd.read_csv(config['valid_path'])
        preds = []
        # Note: The original evaluation loop is kept for simplicity.
        # It assumes a single output and specific batch structure.
        for huberts, labels, ranks, lengths in valid_loader:
            with torch.no_grad():
                logits = best.model(huberts.to(best.device))
                probs = torch.sigmoid(logits)
                batch_preds = (probs > 0.5).sum(dim=1) + 1
            preds.extend(batch_preds.cpu().tolist())
        
        output_csv_path = config.get('output_csv', 'predictions.csv')
        df['predict'] = [1.0 + (r-1)*0.5 for r in preds]
        df.to_csv(output_csv_path, index=False)
        correct = (df['intelligibility'] == df['predict']).sum()
        acc = correct / len(df) if len(df) > 0 else 0
        print(f"Validation accuracy saved to {output_csv_path}. Accuracy: {acc:.4f} ({correct}/{len(df)})")
    else:
        print("No best checkpoint found.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    config = load_config(args.config)
    main(args, config)
