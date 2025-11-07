import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from lob_transformer.module import LOBDataset, calculate_target, load_lobtransformer_model

from .environments import Actions, PositionType, CryptoExchangeEnv


class DummyDataset(Dataset):
    def __init__(self, size: int):
        self.data = torch.tensor([0 for _ in range(size)], dtype=torch.int8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class DeepQNetwork(LightningModule):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 init_epsilon: float = 1.0,
                 min_epsilon: float = 0.01,
                 epsilon_decay_episodes: int = 800,
                 **kwargs):
        super(DeepQNetwork, self).__init__()
        self.save_hyperparameters()
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.tau = 0.01
        
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
        
        self.replay_buffer = TensorDictReplayBuffer(
            storage=ListStorage(max_size=buffer_size),
            sampler=PrioritizedSampler(max_capacity=buffer_size, alpha=0.6, beta=0.4),
            batch_size=self.buffer_size
        )
        self.epsilon = self.init_epsilon
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
    
    def forward(self, x):
        return self.q_net(x)
    
    def get_epsilon(self):
        return self.min_epsilon + (self.init_epsilon - self.min_epsilon) * max(0, (self.epsilon_decay_episodes - self.episodes) / self.epsilon_decay_episodes)
    
    def memorize_experience(self, state, action, reward, next_state, done):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.long)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        done_tensor = torch.as_tensor(done, dtype=torch.int8)
        
        tensor_dict = TensorDict({
            'state': state_tensor,
            'action': action_tensor,
            'reward': reward_tensor,
            'next_state': next_state_tensor,
            'done': done_tensor
        }).to(self.device)
        
        self.replay_buffer.add(tensor_dict)
    
    def choose_action(self, state):
        if random.random() < self.get_epsilon():
            return random.randint(0, len(Actions) - 1)
        with torch.no_grad():
            q_values = self(state)
            return q_values.argmax().item()
    
    def on_train_start(self):
        self.state, info = self.env.reset()
        self.episode_reward = 0.0
        self.episodes: int = 0
    
    def training_step(self, batch, batch_idx):
        if (len(self.replay_buffer) < self.batch_size):
            return None
        
        sample, info = self.replay_buffer.sample(self.batch_size, return_info=True)
        indices = info['index']
        
        states = sample['state']
        actions = sample['action'].unsqueeze(1)
        rewards = sample['reward'].unsqueeze(1)
        next_states = sample['next_state']
        dones = sample['done'].unsqueeze(1)
        
        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        target = rewards + (1 - dones) * self.gamma * next_q_values
        
        td_error = torch.abs(q_values - target).detach()
        priorities = torch.clamp(td_error, min=1e-8, max=10.0)
        
        loss = self.loss_fn(q_values, target)
        
        self.replay_buffer.update_priority(indices, priorities)  # これここでいいのか微妙なところ...
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('epsilon', self.get_epsilon(), prog_bar=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        action = self.choose_action(torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0))
        next_state, reward, done, info = self.env.step(action)
        
        self.memorize_experience(self.state, action, reward, next_state, done)
        
        self.state = next_state
        self.episode_reward += reward
        
        if done:
            self.log('episode_reward', self.episode_reward, prog_bar=True)
            self.episode_reward = 0.0
            
            self.episodes += 1
            self.log('episode', self.episodes, prog_bar=True)
            
            self.state, info = self.env.reset()
    
    def on_train_epoch_end(self):
        losses = [o["loss"] for o in self.training_step_outputs if o is not None]
        total_loss = torch.stack(losses).sum()
        self.log("epoch_loss", total_loss, prog_bar=True)
    
    def on_test_start(self):
        self.state, info = self.env.reset()
        self.episode_reward = 0.0
        done = False
        
        while not done:
            state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self(state).argmax().item()
            next_state, reward, done, info = self.env.step(action)
            
            self.state = next_state
            self.episode_reward += reward
        
        self.log('total_steps', self.env.current_step)
        self.log('episode_reward', self.episode_reward)
        self.log('count_action_buy', len([x for x in self.env.history if x['action'] == Actions.BUY_AT_BEST_BID.value or x['action'] == Actions.SELL_AT_BEST_ASK.value]))
        self.log('count_action_sell', len([x for x in self.env.history if x['action'] == Actions.SELL_AT_BEST_ASK.value or x['action'] == Actions.BUY_AT_BEST_BID.value]))
        self.log('count_action_hold', len([x for x in self.env.history if x['action'] == Actions.DO_NOTHING.value]))
        self.log('last_cash', self.env.history[-1]['cash'])
        self.log('last_long_positions', len([p for p in self.env.portfolio.positions if p.position_type == PositionType.LONG]))
        self.log('last_short_positions', len([p for p in self.env.portfolio.positions if p.position_type == PositionType.SHORT]))
        self.log('last_unrealized_pnl', self.env.history[-1]['unrealized_pnl'])
    
    def test_step(self, batch, batch_idx):
        pass
    
    def set_env(self, env: CryptoExchangeEnv):
        self.env = env


def load_dqn_model(model_path: str) -> DeepQNetwork:
    print("Loading the best model from checkpoint...")
    checkpoint_dir = model_path
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('dqn-') and f.endswith('.ckpt')]
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
                 max_epochs: int = 1000,
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
        self.max_epochs = max_epochs
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_episodes = int(max_epochs * 0.8)
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
        
        env = CryptoExchangeEnv(
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
        
        dqn = DeepQNetwork(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            gamma=self.gamma,
            lr=self.lr,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            init_epsilon=self.init_epsilon,
            min_epsilon=self.min_epsilon,
            epsilon_decay_episodes=self.epsilon_decay_episodes
        )
        
        dqn.set_env(env)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path,
            filename="dqn-{epoch:02d}-{epoch_loss:.4f}-{episode_reward:.2f}",
            monitor="epoch_loss",
            save_top_k=1,
            mode="min",
        )
        
        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[checkpoint_callback],
            logger=False
        )
        
        trainer.fit(dqn, train_dataloaders=DataLoader(
            DummyDataset(size=env.max_steps + 1),
            batch_size=1,
            shuffle=False
        ))
        
        return dqn
    
    def evaluate(self, model: DeepQNetwork):
        print("Evaluating DQN model...")
        
        env = CryptoExchangeEnv(
            data=self.test_df,
            max_steps=int(len(self.test_df)) - 1,
            initial_cash=self.initial_cash,
            transaction_fee=self.transaction_fee,
            max_positions=self.max_positions,
            feature_columns=[
                'probabilities_up',
                'probabilities_stable',
                'probabilities_down',
            ]
        )
        model.set_env(env)

        trainer = Trainer(
            logger=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
        )
        
        result = trainer.test(
            model,
            dataloaders=DataLoader(
                DummyDataset(size=env.max_steps + 1),
                batch_size=1,
                shuffle=False
            )
        )