import os
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lob_transformer import LOBDataset, calculate_target

from environments import Actions, CryptoExchangeEnv

class DeepQNetwork(LightningModule):
    def __init__(self,
                 env: CryptoExchangeEnv,
                 state_size: int,
                 action_size: int,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 epsilon_decay: float = 0.995,
                 **kwargs):
        super(DeepQNetwork, self).__init__()
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
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
        
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.epsilon = self.init_epsilon
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        return [None]
    
    def forward(self, x):
        return self.q_net(x)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(Actions) - 1)
        with torch.no_grad():
            q_values = self(state)
            return q_values.argmax().item()
    
    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        
        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        target = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.functional.mse_loss(q_values, target)
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_start(self):
        state, info = self.env.reset()
        done = False
        while not done:
            action = self.choose_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                states, actions, rewards, next_states, dones = map(
                    lambda x: torch.tensor(np.array(x), dtype=torch.float32),
                    zip(*batch)
                )
                batch = (states, actions.long().unsqueeze(1), rewards.unsqueeze(1), next_states, dones.unsqueeze(1))
                self.training_step(batch, 0)

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
        self.model_path = model_path
        
        self.df = self.create_df()
        self.df = self.prepare_data(self.df)
        
        train_cutoff = 0.8
        
        self.env = CryptoExchangeEnv(
            data=self.df,
            max_steps=int(len(self.df) * train_cutoff),
            initial_cash=self.initial_cash,
            transaction_fee=self.transaction_fee,
            max_positions=self.max_positions,
            feature_columns=self.feature_columns
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
        
        limit = 10000
        df = pd.concat([pd.DataFrame(
            supabase.table(supabase_table)
            .select('*')
            .limit(limit)
            .offset(limit * o)
            .execute()
            .data
        ) for o in range(6)])
        
        df['target'] = calculate_target(df, steps_ahead=12, threshold=0.01/100)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame):
        df['best_ask'] = df['ask_price_1']
        df['best_bid'] = df['bid_price_1']
        df = df.filter(items=['mid_price', 'best_ask', 'best_bid'])
        return df
    
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
            epsilon_decay=(self.min_epsilon / self.init_epsilon) ** (1 / self.max_episodes * 0.9),
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-6,
            patience=3,
            verbose=False,
            mode='min'
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.path.dirname(__file__), self.model_path),
            filename="dqn-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
        
        trainer = Trainer(
            max_epochs=self.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stopping_callback, checkpoint_callback],
            logger=False
        )
        
        trainer.fit(dqn)
        
        return dqn
    
    def evaluate(self, model: DeepQNetwork):
        pass