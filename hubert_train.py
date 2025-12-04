import os
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from functools import partial
from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings

# Import all our modules
from hubert_dataset import HubertDataset, data_processing
from hubert_solver import LitHubert
from coral_loss import logits_to_label # Assuming 1-indexed labels

warnings.filterwarnings('ignore')

def load_config(path: str) -> dict:
    """Safely loads a YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('config', cfg)

def rank_to_score(rank: int) -> float:
    """Converts a 1-indexed rank (1-9) back to a MOS-like score (1.0-5.0)."""
    if not isinstance(rank, (int, float)) or rank < 1:
        return float('nan')
    return 1.0 + 0.5 * (rank - 1)

def save_predictions(model: LitHubert, dataloader: data.DataLoader, output_path: str):
    """
    Runs inference and saves predictions for all tasks to a single CSV.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Get the original dataframe from the dataset
    dataset_df = dataloader.dataset.df.copy()
    
    predictions = {task: [] for task in model.tasks.keys()}
    indices = []

    with torch.no_grad():
        for huberts, labels_dict, lengths, batch_indices in dataloader:
            huberts = huberts.to(device)
            logits_dict = model(huberts)
            indices.extend(batch_indices.cpu().tolist())

            for task_name, logits in logits_dict.items():
                task_cfg = model.tasks[task_name]
                if task_cfg.get('loss') == 'coral':
                    # Convert logits to 1-indexed ranks
                    preds = logits_to_label(logits).cpu().tolist()
                    predictions[task_name].extend(preds)
                elif task_cfg.get('loss') == 'bce':
                    # Convert logits to 0/1 labels
                    preds = (torch.sigmoid(logits) > 0.5).int().cpu().tolist()
                    predictions[task_name].extend(preds)
    
    # Create a DataFrame with predictions, aligned by original index
    pred_df = pd.DataFrame({'original_index': indices})
    for task_name, preds in predictions.items():
        if model.tasks[task_name].get('loss') == 'coral':
            pred_df[f'pred_{task_name}_rank'] = preds
            pred_df[f'pred_{task_name}_score'] = [rank_to_score(r) for r in preds]
        elif model.tasks[task_name].get('loss') == 'bce':
            for i in range(len(preds[0])):
                pred_df[f'pred_{task_name}_{i+1}'] = [p[i] for p in preds]

    # Merge predictions back into the original dataframe
    # Use index for robust merging
    dataset_df['original_index'] = dataset_df.index
    final_df = pd.merge(dataset_df, pred_df, on='original_index', how='left')
    final_df.drop(columns=['original_index'], inplace=True)

    # Save to CSV
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main(args, config: dict):
    # 1. Prepare model
    model = LitHubert(config)

    # 2. Prepare DataLoaders
    # Use functools.partial to pass the tasks config to the collate function
    collate_fn_with_config = partial(data_processing, tasks_config=config['model']['tasks'])

    train_dataset = HubertDataset(path=config['train_path'], config=config)
    train_loader = data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['process']['num_workers'], pin_memory=False, collate_fn=collate_fn_with_config
    )

    valid_dataset = HubertDataset(path=config['valid_path'], config=config)
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['process']['num_workers'], pin_memory=False, collate_fn=collate_fn_with_config
    )

    # 3. Prepare Callbacks and Logger
    checkpoint_cb = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor=config.get('checkpoint', {}).get('monitor', 'val_loss'),
        patience=config.get('scheduler', {}).get('patience', 3) + 2, # A bit more patient than scheduler
        mode=config.get('checkpoint', {}).get('mode', 'min')
    )
    logger = TensorBoardLogger(**config['logger'])

    # 4. Initialize and run Trainer
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

    # 5. Post-training evaluation and prediction saving
    best_path = checkpoint_cb.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"--- Training complete. Best model saved at: {best_path} ---")
        print("Running final validation on the best checkpoint...")
        
        # Load best model and run validation
        best_model = LitHubert.load_from_checkpoint(best_path, config=config)
        val_results = trainer.validate(model=best_model, dataloaders=valid_loader, verbose=False)
        print("Final validation metrics:", val_results)

        # Save predictions
        output_csv_path = config.get('output_csv', 'predictions.csv')
        print(f"Saving predictions for validation set to {output_csv_path}...")
        save_predictions(best_model, valid_loader, output_csv_path)

    else:
        print("Training complete, but no best checkpoint was saved or found.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint to resume training.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    config = load_config(args.config)
    main(args, config)