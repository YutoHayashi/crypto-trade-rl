import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import json

from dotenv import load_dotenv
load_dotenv()
model_path = os.getenv('MODEL_PATH', 'models')

import pandas as pd

import torch
from lob_transformer.module import LOBDataset, LOBDatasetConfig, LOBTransformer

from .dqn import DQNTrainer


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN model for crypto trading.")
    
    parser.add_argument('--preset', type=str, required=False, default='debug', help='Preset configuration to use.')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=False, default='train', help='Mode: train or evaluate.')
    
    parser.add_argument('--csv_path', type=str, required=False, default=None, help='Path to the CSV file containing market data.')
    parser.add_argument('--gamma', type=float, required=False, default=None, help='Discount factor for future rewards.')
    parser.add_argument('--lr', type=float, required=False, default=None, help='Learning rate for the optimizer.')
    parser.add_argument('--buffer_size', type=int, required=False, default=None, help='Size of the replay buffer.')
    parser.add_argument('--batch_size', type=int, required=False, default=None, help='Batch size for training.')
    parser.add_argument('--max_epochs', type=int, required=False, default=None, help='Maximum number of training epochs.')
    parser.add_argument('--init_epsilon', type=float, required=False, default=None, help='Initial epsilon for epsilon-greedy policy.')
    parser.add_argument('--min_epsilon', type=float, required=False, default=None, help='Minimum epsilon for epsilon-greedy policy.')
    parser.add_argument('--initial_cash', type=float, required=False, default=None, help='Initial cash for the trading environment.')
    parser.add_argument('--transaction_fee', type=float, required=False, default=None, help='Transaction fee percentage.')
    parser.add_argument('--max_positions', type=int, required=False, default=None, help='Maximum number of open positions allowed.')
    parser.add_argument('--profit_reward_weight', type=float, required=False, default=None, help='Weight for profit in the reward calculation.')
    parser.add_argument('--penalty_reward_weight', type=float, required=False, default=None, help='Weight for penalty in the reward calculation.')
    
    parser.add_argument('--ckpt_path', type=str, required=False, default=None, help='Path to a checkpoint file to resume training or for evaluation.')
    
    args = parser.parse_args()
    
    with open(os.path.join(os.path.dirname(__file__), 'presets.json'), 'r') as f:
        preset = json.load(f).get(args.preset, {})
    
    args = {**preset, **{k: v for k, v in vars(args).items() if v is not None}}
    
    print("Using configuration:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    
    return args


def load_all_lobtransformer_models():
    """
    To use the observation space for multi-horizon forecasting,
    load multiple LOBTransformer models.
    """
    models = []
    for filename in os.listdir(model_path):
        if filename.startswith('lobtransformer-') and filename.endswith('.ckpt'):
            model_ckpt_path = os.path.join(model_path, filename)
            lob_transformer = LOBTransformer.load_from_checkpoint(model_ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            models.append(lob_transformer)
            
            print(f"Loaded LOBTransformer model from {model_ckpt_path}")
    return models


def prepare_data(df: pd.DataFrame):
    from lightning.pytorch import Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=False
    )
    
    for lob_transformer in load_all_lobtransformer_models():
        dataset_config: LOBDatasetConfig = lob_transformer.hparams.dataset_config
        window_size = dataset_config.window_size
        horizon = dataset_config.horizon
        
        lob_dataset = LOBDataset(df.copy(), **{
            **lob_transformer.hparams.dataset_config.__dict__,
            'target_cols': [],
        })
        lob_dataloader = lob_dataset.to_dataloader(batch_size=1, shuffle=False)
        
        predictions = trainer.predict(lob_transformer, lob_dataloader)
        
        df_subset = df.iloc[(window_size) - 1:].copy()
        df_subset[f'downward_forecast_H{horizon}'] = [pred[0][0].item() for pred in predictions]
        df_subset[f'stable_forecast_H{horizon}'] = [pred[0][1].item() for pred in predictions]
        df_subset[f'upward_forecast_H{horizon}'] = [pred[0][2].item() for pred in predictions]
        
        df = pd.concat([df, df_subset.filter(regex=f'.*_H{horizon}$')], axis=1)
    
    df['best_ask'] = df['ask_price_1']
    df['best_bid'] = df['bid_price_1']
    
    df = (df
          .filter(items=['best_ask', 'best_bid'])
          .join(df.filter(regex=f'.*_H[0-9]+$')))
    
    df = df.dropna().reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    mode = args.get('mode')
    csv_path = args.get('csv_path')
    
    df = pd.read_csv(csv_path, index_col=0)
    df = prepare_data(df)
    
    trainer = DQNTrainer(df=df, model_path=model_path, **args)
    
    if mode in ['train']:
        dqn, checkpoint_callback = trainer.train()
        
        best_model_path = checkpoint_callback.best_model_path
        trainer.evaluate(best_model_path)
    
    elif mode in ['eval']:
        trainer.evaluate(args.get('ckpt_path'))


if __name__ == "__main__":
    main()