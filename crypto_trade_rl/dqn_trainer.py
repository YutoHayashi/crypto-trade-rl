import os
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from lob_transformer.module import LOBDataset, calculate_target, load_lobtransformer_model

from .environments import Actions, CryptoExchangeEnv


class DummyDataset(Dataset):
    def __init__(self, size: int):
        self.data = torch.tensor([0 for _ in range(size)], dtype=torch.int8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class DeepQNetwork(LightningModule):
    def __init__(self,
                 env: CryptoExchangeEnv,
                 state_size: int,
                 action_size: int,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 init_epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 **kwargs):
        super(DeepQNetwork, self).__init__()
        self.save_hyperparameters()
        
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.init_epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        
        self.q_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.target_q_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.epsilon = self.init_epsilon
        
        self.total_rewards = 0.0
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
    
    def forward(self, x):
        return self.q_net(x)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(Actions) - 1)
        with torch.no_grad():
            q_values = self(state)
            return q_values.argmax().item()
    
    def training_step(self, batch, batch_idx):
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.int8).unsqueeze(1)
        
        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target)
        self.log('train_loss', loss)
        self.log('total_rewards', self.total_rewards)
        return loss
    
    def test_step(self, batch, batch_idx):
        pass
    
    def on_train_epoch_start(self):
        self.total_rewards = 0.0
        
        state, info = self.env.reset()
        done = False
        while not done:
            action = self.choose_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.append((state, action, reward, next_state, done))
            self.total_rewards += reward
            state = next_state


def load_dqn_model(model_path: str) -> DeepQNetwork:
    print("Loading the best model from checkpoint...")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), model_path)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the specified output path.")
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    model = DeepQNetwork.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loaded model from {checkpoint_path}")
    
    return model


class DQNTrainer:
    def __init__(self,
                 gamma: float = 0.99,
                 learning_rate: float = 1e-3,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 max_episodes: int = 1000,
                 init_epsilon: float = 1.0,
                 min_epsilon: float = 0.01,
                 initial_cash: float = 1000000.0,
                 transaction_fee: float = 0.01/100,
                 max_positions: int = 5,
                 window_size: int = 60,
                 rolling_window: int = 60,
                 model_path: str = 'models',
                 **kwargs):
        self.gamma = gamma
        self.lr = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.max_positions = max_positions
        self.window_size = window_size
        self.rolling_window = rolling_window
        self.model_path = model_path
        
        self.df = self.create_df()
        self.df = self.prepare_data(self.df)
        
        train_cutoff = 0.8
        self.train_df, self.test_df = (
            self.df.iloc[:int(len(self.df) * train_cutoff)].reset_index(drop=True),
            self.df.iloc[int(len(self.df) * train_cutoff):].reset_index(drop=True)
        )
        
        self.env = CryptoExchangeEnv(
            data=self.train_df,
            max_steps=int(len(self.train_df)) - 1,
            initial_cash=self.initial_cash,
            transaction_fee=self.transaction_fee,
            max_positions=self.max_positions,
            feature_columns=[
                'probabilities_up',
                'probabilities_stable',
                'probabilities_down',
            ]
        )
    
    def create_df(self):
        from supabase import create_client, ClientOptions
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        supabase_table = os.getenv('SUPABASE_TABLE_FOR_RL')
        
        supabase = create_client(
            supabase_url,
            supabase_key,
            options=ClientOptions(
                postgrest_client_timeout=604800,
                storage_client_timeout=604800
            )
        )
        
        limit = 1000
        df = pd.concat([pd.DataFrame(
            supabase.table(supabase_table)
            .select('*')
            .limit(limit)
            .offset(limit * o)
            .execute()
            .data
        ) for o in range(1)])
        
        df['target'] = calculate_target(df, steps_ahead=12, threshold=0.01/100)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame):
        predictions = self.predict_price_movement(df)
        
        df = df.iloc[(self.window_size + self.rolling_window) - 1:].copy().reset_index(drop=True)
        df['probabilities_up'] = [pred[0][0].item() for pred in predictions]
        df['probabilities_stable'] = [pred[0][1].item() for pred in predictions]
        df['probabilities_down'] = [pred[0][2].item() for pred in predictions]
        df['best_ask'] = df['ask_price_1']
        df['best_bid'] = df['bid_price_1']
        
        df = df.filter(items=[
            'best_ask',
            'best_bid',
            'probabilities_up',
            'probabilities_stable',
            'probabilities_down',
        ])
        
        return df
    
    def predict_price_movement(self, df: pd.DataFrame):
        from lightning.pytorch import Trainer
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto"
        )
        
        lob_transformer = load_lobtransformer_model(self.model_path)
        lob_transformer.eval()
        
        lob_dataset = LOBDataset(df, window_size=self.window_size)
        lob_dataloader = lob_dataset.to_dataloader(batch_size=1, shuffle=False)
        
        predictions = trainer.predict(lob_transformer, lob_dataloader)
        
        return predictions
    
    def train(self):
        print("Training DQN model...")
        
        dqn = DeepQNetwork(
            env=self.env,
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            gamma=self.gamma,
            lr=self.lr,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            init_epsilon=self.init_epsilon,
            epsilon_decay=(self.min_epsilon / self.init_epsilon) ** (1 / self.max_episodes * 0.9),
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path,
            filename="dqn-{epoch:02d}-{train_loss:.4f}-{total_rewards:.2f}",
            monitor="train_loss",
            save_top_k=1,
            mode="min",
        )
        
        trainer = Trainer(
            max_epochs=self.max_episodes,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[checkpoint_callback],
            logger=False
        )
        
        trainer.fit(dqn, train_dataloaders=DataLoader(
            DummyDataset(size=self.env.max_steps),
            batch_size=1,
            shuffle=False
        ))
        
        return dqn
    
    def evaluate(self, model: DeepQNetwork):
        pass