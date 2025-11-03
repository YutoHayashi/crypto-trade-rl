import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import json

from dotenv import load_dotenv
load_dotenv()

from .dqn_trainer import DQNTrainer, load_dqn_model

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN model for crypto trading.")
    
    parser.add_argument('--preset', type=str, choices=['minimum', 'dev', 'prod'], required=False, default='prod', help='Preset configuration to use.')
    parser.add_argument('--gamma', type=float, required=False, default=0.99, help='Discount factor for future rewards.')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--buffer_size', type=int, required=False, default=100000, help='Size of the replay buffer.')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size for training.')
    parser.add_argument('--max_epochs', type=int, required=False, default=1000, help='Maximum number of training epochs.')
    parser.add_argument('--initial_cash', type=float, required=False, default=1000000.0, help='Initial cash for the trading environment.')
    parser.add_argument('--transaction_fee', type=float, required=False, default=0.01/100, help='Transaction fee percentage.')
    parser.add_argument('--max_positions', type=int, required=False, default=5, help='Maximum number of open positions allowed.')
    parser.add_argument('--mode', type=str, choices=['train_and_eval', 'train', 'eval'], required=False, default='train_and_eval', help='Mode: train or evaluate.')
    
    args = parser.parse_args()
    
    with open(os.path.join(os.path.dirname(__file__), 'presets.json'), 'r') as f:
        preset = json.load(f).get(args.preset, {})
    
    args = vars(args) | preset
    
    print("Using configuration:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    
    return args

def main() -> None:
    args = parse_args()
    mode = args.get('mode', 'train_and_eval')
    model_path = os.getenv('MODEL_PATH', 'models')
    
    trainer = DQNTrainer(**args, model_path=model_path)
    
    trained_model = None
    
    if mode in ['train', 'train_and_eval']:
        trained_model = trainer.train()
    
    if mode in ['eval', 'train_and_eval']:
        if trained_model is None:
            trained_model = load_dqn_model(model_path)
        
        trainer.evaluate(trained_model)

if __name__ == "__main__":
    main()