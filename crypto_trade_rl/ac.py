import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

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


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        feat = self.feature(x)
        policy_logits = self.actor(feat)
        value = self.critic(feat)
        return policy_logits, value


class ACActor:
    def __init__(self, env: CryptoTradeEnv):
        self.env = env
        self.state, _ = self.env.reset()
        self.episode_reward = 0.0
    
    def reset(self):
        self.state, _ = self.env.reset()
        self.episode_reward = 0.0


class ActorCritic(LightningModule):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 gamma: float,
                 lr: float,
                 n_step: int,
                 num_actors: int,
                 entropy_beta: float = 0.01,
                 **kwargs):
        super(ActorCritic, self).__init__()
        self.save_hyperparameters()
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.n_step = n_step
        self.num_actors = num_actors
        self.entropy_beta = entropy_beta
        
        self.net = ActorCriticNetwork(state_size, action_size)
        
        self.episodes = 0
        self.actors: list[ACActor] = []
        self.losses = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
    
    def forward(self, x):
        return self.net(x)
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['episodes'] = self.episodes
    
    def on_load_checkpoint(self, checkpoint):
        self.episodes = checkpoint['episodes']
    
    def on_train_start(self):
        for actor in self.actors:
            actor.reset()
        self.episodes: int = 0
    
    def on_train_epoch_start(self):
        self.losses = []
    
    def training_step(self, batch, batch_idx):
        # Collect rollouts
        batch_log_probs = []
        batch_values = []
        batch_returns = []
        batch_entropies = []
        
        for actor in self.actors:
            actor_rewards = []
            actor_values = []
            actor_log_probs = []
            actor_masks = []
            actor_entropies = []
            
            state = actor.state
            
            # Rollout n_steps
            for _ in range(self.n_step):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, value = self.net(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                next_state, reward, done, info = actor.env.step(action.item())
                actor.episode_reward += reward
                
                actor_log_probs.append(dist.log_prob(action))
                actor_values.append(value)
                actor_rewards.append(torch.tensor([reward], device=self.device))
                actor_masks.append(torch.tensor([1 - int(done)], device=self.device))
                actor_entropies.append(dist.entropy())
                
                state = next_state
                
                if done:
                    self.log('episode_reward', actor.episode_reward, prog_bar=True)
                    self.episodes += 1
                    self.log('episode', self.episodes, prog_bar=True)
                    state, _ = actor.env.reset()
                    actor.episode_reward = 0.0
            
            actor.state = state
            
            # Compute returns (Bootstrapping)
            next_state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.net(next_state_tensor)
            R = next_value
            
            actor_returns = []
            for step in reversed(range(self.n_step)):
                R = actor_rewards[step] + self.gamma * R * actor_masks[step]
                actor_returns.insert(0, R)
            
            batch_log_probs.extend(actor_log_probs)
            batch_values.extend(actor_values)
            batch_returns.extend(actor_returns)
            batch_entropies.extend(actor_entropies)
            
        # Compute Loss
        log_probs = torch.cat(batch_log_probs)
        values = torch.cat(batch_values).squeeze()
        returns = torch.cat(batch_returns).squeeze()
        entropies = torch.cat(batch_entropies)
        
        advantage = returns - values
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -entropies.mean()
        
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_beta * entropy_loss
        
        self.losses.append(total_loss)
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('actor_loss', actor_loss)
        self.log('critic_loss', critic_loss)
        self.log('entropy_loss', entropy_loss)
        
        return total_loss
    
    def on_train_epoch_end(self):
        if self.losses:
            total_loss = torch.stack(self.losses).sum()
            self.log("epoch_loss", total_loss, prog_bar=True)
    
    def on_test_start(self):
        env = self.actors[0].env
        state, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.net(state)
            action = logits.argmax().item() # Greedy for test
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
        
        self.log('total_steps', env.current_step)
        self.log('episode_reward', episode_reward)
        self.log('portfolio_value', info.get('total_value', 0.))
    
    def test_step(self, batch, batch_idx):
        pass
    
    def set_envs(self, envs: list[CryptoTradeEnv]):
        self.actors = [ACActor(env) for env in envs]


class ActorCriticTrainer:
    def __init__(self,
                 df: pd.DataFrame,
                 gamma: float,
                 lr: float,
                 max_epochs: int,
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
        self.max_epochs = max_epochs
        self.initial_collateral = initial_collateral
        self.transaction_fee = transaction_fee
        self.trading_volume = trading_volume
        self.profit_target = profit_target
        self.n_step = n_step
        self.num_actors = num_actors
        self.model_path = model_path
        self.ckpt_path = ckpt_path
    
    def train(self):
        print(f"Training Actor-Critic model with {self.num_actors} actors...")
        
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
        
        model = ActorCritic(
            state_size=ref_env.observation_space.shape[0],
            action_size=ref_env.action_space.n,
            gamma=self.gamma,
            lr=self.lr,
            n_step=self.n_step,
            num_actors=self.num_actors
        )
        
        model.set_envs(envs)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path,
            filename="ac-{epoch:02d}-{epoch_loss:.4f}-{episode_reward:.2f}",
            monitor="episode",
            save_top_k=1,
            mode="max",
        )
        
        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.5, # Gradient clipping is important for A2C
            callbacks=[checkpoint_callback],
            logger=False
        )
        
        trainer.fit(model, train_dataloaders=DataLoader(
            DummyDataset(size=ref_env.max_steps + 1),
            batch_size=1,
            shuffle=False,
        ), ckpt_path=self.ckpt_path)
        
        return (
            model,
            checkpoint_callback
        )
    
    def evaluate(self, model_path: str):
        print("Evaluating Actor-Critic model...")
        
        env = CryptoTradeEnv(
            data=self.df,
            max_steps=int(len(self.df)) - 1,
            initial_collateral=self.initial_collateral,
            transaction_fee=self.transaction_fee,
            trading_volume=self.trading_volume,
            profit_target=self.profit_target,
            feature_columns=self.df.columns.difference(['best_ask', 'best_bid']).tolist()
        )
        
        model = ActorCritic.load_from_checkpoint(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
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
