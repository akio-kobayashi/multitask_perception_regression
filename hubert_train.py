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
import pandas as pd
import yaml
from argparse import ArgumentParser
from string import Template

def load_config(path: str) -> dict:
    raw = open(path, 'r', encoding='utf-8').read()
    rendered = Template(raw).substitute(**os.environ)
    cfg = yaml.safe_load(rendered)
    return cfg.get('config', cfg)


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
    logger = TensorBoardLogger(**config['logger'])

    # 3) Trainer の作成と学習
    trainer = pl.Trainer(
        callbacks=[checkpoint_cb],
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

    # 4) Log best model path
    best_path = checkpoint_cb.best_model_path
    if best_path:
        print(f"Training complete. Best model saved at: {best_path}")
    else:
        print("Training complete, but no best checkpoint was saved.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    config = load_config(args.config)
    main(args, config)
    
