import random
import warnings
warnings.filterwarnings('ignore')
from collections import deque

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from .environment import CryptoTradeEnv


class DummyDataset(Dataset):
    def __init__(self, size: int):
        self.data = torch.tensor([0 for _ in range(size)], dtype=torch.int8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        feat = self.feature(x)
        adv = self.advantage(feat)
        val = self.value(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)


class ApeXActor:
    def __init__(self,
                 env: CryptoTradeEnv,
                 epsilon: float,
                 n_step: int,
                 gamma: float):
        self.env = env
        self.epsilon = epsilon
        self.n_step = n_step
        self.gamma = gamma
        
        self.buffer = deque()
        self.state = None
        self.episode_reward = 0.0
        self.reset()
    
    def reset(self):
        self.state, _ = self.env.reset()
        self.episode_reward = 0.0
        self.buffer.clear()
    
    def step(self, action: int):
        next_state, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        
        self.buffer.append((self.state, action, reward, next_state, done))
        self.state = next_state
        
        experiences = []
        
        if len(self.buffer) >= self.n_step:
            experiences.append(self._compute_n_step_return())
            self.buffer.popleft()
        
        if done:
            while len(self.buffer) > 0:
                experiences.append(self._compute_n_step_return())
                self.buffer.popleft()
            self.reset()
            
        return experiences, done, info

    def _compute_n_step_return(self):
        state_0, action_0, _, _, _ = self.buffer[0]
        reward_n = 0.0
        for i in range(len(self.buffer)):
            r = self.buffer[i][2]
            reward_n += r * (self.gamma ** i)
            if self.buffer[i][4]: # If done
                break
        
        next_state_n = self.buffer[-1][3]
        done_n = self.buffer[-1][4]
        
        # Check if any intermediate step was done
        for i in range(len(self.buffer)):
            if self.buffer[i][4]: # If done
                next_state_n = self.buffer[i][3]
                done_n = True
                break
        
        return (state_0, action_0, reward_n, next_state_n, done_n)
    
    def get_action(self, q_net: nn.Module, device: torch.device) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n - 1)
        with torch.no_grad():
            state_tensor = torch.tensor(self.state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_net(state_tensor)
            return q_values.argmax().item()


class ApeX(LightningModule):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 gamma: float,
                 lr: float,
                 buffer_size: int,
                 batch_size: int,
                 max_epsilon: float,
                 min_epsilon: float,
                 target_update_interval: int,
                 n_step: int,
                 num_actors: int,
                 **kwargs):
        super(ApeX, self).__init__()
        self.save_hyperparameters()
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update_interval = target_update_interval
        self.n_step = n_step
        self.num_actors = num_actors
        
        self.q_net = DuelingNetwork(state_size, action_size)
        self.target_q_net = DuelingNetwork(state_size, action_size)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = TensorDictReplayBuffer(
            storage=ListStorage(max_size=buffer_size),
            sampler=PrioritizedSampler(max_capacity=buffer_size, alpha=0.6, beta=0.4),
            batch_size=self.buffer_size
        )
        
        self.episodes = 0
        self.actors: list[ApeXActor] = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
    
    def forward(self, x):
        return self.q_net(x)
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['episodes'] = self.episodes
    
    def on_load_checkpoint(self, checkpoint):
        self.episodes = checkpoint['episodes']
    
    def _memorize_experience(self, state, action, reward, next_state, done):
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
    
    def on_train_start(self):
        for actor in self.actors:
            actor.reset()
        self.episodes: int = 0
    
    def on_train_epoch_start(self):
        self.losses = []
    
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
        
        # Double DQN with N-step
        next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_q_net(next_states).gather(1, next_actions).detach()
        
        gamma_n = self.gamma ** self.n_step
        target = rewards + (1 - dones) * gamma_n * next_q_values
        
        td_error = torch.abs(q_values - target).detach()
        priorities = torch.nan_to_num(td_error, nan=1e-8, posinf=10.0, neginf=1e-8)
        priorities = torch.clamp(priorities, min=1e-8, max=10.0)
        
        loss = self.loss_fn(q_values, target)
        self.losses.append(loss)
        
        self.replay_buffer.update_priority(indices, priorities)
        
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        for actor in self.actors:
            action = actor.get_action(self.q_net, self.device)
            experiences, done, info = actor.step(action)
            
            for exp in experiences:
                self._memorize_experience(*exp)
            
            if done:
                self.log('episode_reward', actor.episode_reward, prog_bar=True)
                
                self.episodes += 1
                self.log('episode', self.episodes, prog_bar=True)
    
    def on_train_epoch_end(self):
        total_loss = torch.stack(self.losses).sum()
        self.log("epoch_loss", total_loss, prog_bar=True)
    
    def on_test_start(self):
        env = self.actors[0].env
        state, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self(state).argmax().item()
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
        
        self.log('total_steps', env.current_step)
        self.log('episode_reward', episode_reward)
        self.log('portfolio_value', info.get('total_value', 0.))
    
    def test_step(self, batch, batch_idx):
        pass
    
    def set_envs(self, envs: list[CryptoTradeEnv]):
        actor_epsilons = np.logspace(np.log10(self.hparams.max_epsilon), np.log10(self.hparams.min_epsilon), num=self.num_actors)
        
        self.actors = []
        for i, env in enumerate(envs):
            epsilon = actor_epsilons[i]
            self.actors.append(ApeXActor(env, epsilon, self.n_step, self.gamma))


class ApeXTrainer:
    def __init__(self,
                 df: pd.DataFrame,
                 gamma: float,
                 lr: float,
                 buffer_size: int,
                 batch_size: int,
                 max_epochs: int,
                 max_epsilon: float,
                 min_epsilon: float,
                 target_update_interval: int,
                 initial_collateral: float,
                 transaction_fee: float,
                 trading_volume: float,
                 profit_target: float,
                 n_step: int,
                 num_actors: int,
                 model_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        self.df = df
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update_interval = target_update_interval
        self.initial_collateral = initial_collateral
        self.transaction_fee = transaction_fee
        self.trading_volume = trading_volume
        self.profit_target = profit_target
        self.n_step = n_step
        self.num_actors = num_actors
        self.model_path = model_path
        self.ckpt_path = ckpt_path
    
    def train(self):
        print(f"Training Ape-X (Dueling DDQN + N-step) model with {self.num_actors} actors...")
        
        # Create multiple environments for actors
        envs = []
        for _ in range(self.num_actors):
            env = CryptoTradeEnv(
                data=self.df,
                max_steps=int(len(self.df)) - 1,
                initial_collateral=self.initial_collateral,
                transaction_fee=self.transaction_fee,
                trading_volume=self.trading_volume,
                profit_target=self.profit_target,
                feature_columns=self.df.columns.difference(['best_ask', 'best_bid']).tolist()
            )
            envs.append(env)
        
        ref_env = envs[0]
        
        apex = ApeX(
            state_size=ref_env.observation_space.shape[0],
            action_size=ref_env.action_space.n,
            gamma=self.gamma,
            lr=self.lr,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            max_epsilon=self.max_epsilon,
            min_epsilon=self.min_epsilon,
            target_update_interval=self.target_update_interval,
            n_step=self.n_step,
            num_actors=self.num_actors
        )
        
        apex.set_envs(envs)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path,
            filename="apex-{epoch:02d}-{epoch_loss:.4f}-{episode_reward:.2f}",
            monitor="episode",
            save_top_k=1,
            mode="max",
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
        
        trainer.fit(apex, train_dataloaders=DataLoader(
            DummyDataset(size=ref_env.max_steps + 1),
            batch_size=1,
            shuffle=False,
        ), ckpt_path=self.ckpt_path)
        
        return (
            apex,
            checkpoint_callback
        )
    
    def evaluate(self, model_path: str):
        print("Evaluating Ape-X model...")
        
        env = CryptoTradeEnv(
            data=self.df,
            max_steps=int(len(self.df)) - 1,
            initial_collateral=self.initial_collateral,
            transaction_fee=self.transaction_fee,
            trading_volume=self.trading_volume,
            profit_target=self.profit_target,
            feature_columns=self.df.columns.difference(['best_ask', 'best_bid']).tolist()
        )
        
        model = ApeX.load_from_checkpoint(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.set_envs([env])
        
        trainer = Trainer(
            logger=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
        )
        
        return trainer.test(
            model,
            dataloaders=DataLoader(
                DummyDataset(size=env.max_steps + 1),
                batch_size=1,
                shuffle=False
            )
        )