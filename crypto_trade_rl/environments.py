from enum import Enum
from dataclasses import dataclass
import uuid
from collections import deque

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    DO_NOTHING = 0
    BUY_AT_BEST_BID = 1
    SELL_AT_BEST_ASK = 2

class PositionType(Enum):
    LONG = 1
    SHORT = -1

@dataclass
class Position:
    id: str
    position_type: PositionType
    quantity: float
    price: float
    
    def __repr__(self):
        return f"Position(type={self.position_type}, quantity={self.quantity}, price={self.price})"

class Portfolio:
    def __init__(self, initial_cash: float, transaction_fee: float = 0.01/100):
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.cash = self.initial_cash
        self.positions = []
    
    def reset(self) -> None:
        self.cash = self.initial_cash
        self.positions = []
    
    def open_position(self, position_type: PositionType, quantity: float, price: float) -> Position:
        id = uuid.uuid4()
        position = Position(id=id, position_type=position_type, quantity=quantity, price=price)
        self.positions.append(position)
        return position
    
    def close_position(self, position: Position, price: float) -> float:
        if position in self.positions:
            self.positions.remove(position)
            total_transaction_cost = (price + position.price) * position.quantity * self.transaction_fee
            if position.position_type == PositionType.LONG:
                pnl = (price - position.price) * position.quantity - total_transaction_cost
            elif position.position_type == PositionType.SHORT:
                pnl = (position.price - price) * position.quantity - total_transaction_cost
            self.cash += pnl
            return pnl
        else:
            raise ValueError("Position not found in portfolio")
    
    def unrealized_pnl(self, best_bid: float, best_ask: float) -> float:
        pnl = 0.0
        for pos in self.positions:
            if pos.position_type == PositionType.LONG:
                unrealized_cost = (best_bid + pos.price) * pos.quantity * self.transaction_fee
                pnl += (best_bid - pos.price) * pos.quantity - unrealized_cost
            elif pos.position_type == PositionType.SHORT:
                unrealized_cost = (best_ask + pos.price) * pos.quantity * self.transaction_fee
                pnl += (pos.price - best_ask) * pos.quantity - unrealized_cost
        return pnl
    
    def total_value(self, best_bid: float, best_ask: float) -> float:
        return self.cash + self.unrealized_pnl(best_bid, best_ask)
    
    @property
    def long_positions(self) -> list[Position]:
        return [pos for pos in self.positions if pos.position_type == PositionType.LONG]
    
    @property
    def short_positions(self) -> list[Position]:
        return [pos for pos in self.positions if pos.position_type == PositionType.SHORT]
    
    def __repr__(self):
        return f"Portfolio(cash={self.cash}, positions={self.positions})"

class CryptoExchangeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self,
                 data: pd.DataFrame,
                 max_steps: int,
                 initial_cash: float = 1_000_000,
                 transaction_fee: float = 0.01/100,
                 max_positions: int = 5,
                 profit_reward_weight: float = 1.0,
                 feature_columns: list = []):
        """
        Args:
            data (_type_): _description_
            initial_cash (float, optional): _description_. Defaults to 1_000_000.
            transaction_fee (float, optional): _description_. Defaults to 0.01/100.
            feature_columns (list, optional): _description_. Defaults to [].
        """
        super().__init__()
        self.data = data
        self.max_steps = max_steps
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.max_positions = max_positions
        self.profit_reward_weight = profit_reward_weight
        self.feature_columns = feature_columns
        
        self.current_step = 0
        self.portfolio = Portfolio(initial_cash=self.initial_cash, transaction_fee=self.transaction_fee)
        self.history = deque(maxlen=self.max_steps)
        self.done = False
        
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_columns)+2,), dtype=np.float32)
        
        self._validate_data()
    
    def _validate_data(self) -> ValueError|None:
        if type(self.data) is not pd.DataFrame:
            raise ValueError("Data must be a pandas DataFrame")
        
        required_columns = ['best_bid', 'best_ask'] + self.feature_columns
        for required_column in required_columns:
            if required_column not in self.data.columns:
                raise ValueError(f"Data must contain the column: {required_column}")
    
    def _get_observation(self) -> np.ndarray:
        if self.current_step <= self.max_steps:
            current_data = self.data.iloc[self.current_step]
            
            best_bid = current_data['best_bid']
            best_ask = current_data['best_ask']
            
            features = current_data[self.feature_columns].values.astype(np.float32)
            positions = sum([pos.quantity * (1 if pos.position_type == PositionType.LONG else -1) for pos in self.portfolio.positions])
            unrealized_pnl = self.portfolio.unrealized_pnl(best_bid=best_bid, best_ask=best_ask) / self.initial_cash
            
            return np.concatenate([
                features,
                [
                    positions,
                    unrealized_pnl
                ]
            ]).astype(np.float32)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _get_info(self) -> dict:
        return self.history[-1] if self.history else {}
    
    def _step_action(self, action: int) -> float:
        current_data = self.data.iloc[self.current_step]
        reward = 0.0
        
        best_bid = current_data['best_bid']
        best_ask = current_data['best_ask']
        pnl = 0.0
        hold_reward = 0.0
        
        if action == Actions.DO_NOTHING.value:
            hold_reward = self.portfolio.unrealized_pnl(best_bid=best_bid, best_ask=best_ask) * 0.01
        elif action == Actions.BUY_AT_BEST_BID.value:
            if self.portfolio.short_positions:
                position = self.portfolio.short_positions[0]
                pnl = self.portfolio.close_position(position, price=best_bid)
            elif len(self.portfolio.positions) < self.max_positions:
                self.portfolio.open_position(PositionType.LONG, quantity=0.01, price=best_bid)
        elif action == Actions.SELL_AT_BEST_ASK.value:
            if self.portfolio.long_positions:
                position = self.portfolio.long_positions[0]
                pnl = self.portfolio.close_position(position, price=best_ask)
            elif len(self.portfolio.positions) < self.max_positions:
                self.portfolio.open_position(PositionType.SHORT, quantity=0.01, price=best_ask)
        
        if pnl > 0:
            pnl *= self.profit_reward_weight
        
        reward += pnl + hold_reward
        
        if self.current_step >= self.max_steps or self.portfolio.cash <= 0:
            self.done = True
        
        self.history.append({
            "step": self.current_step,
            "action": action,
            "action_name": Actions(action).name,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "cash": self.portfolio.cash,
            "positions": self.portfolio.positions.copy(),
            "total_value": self.portfolio.total_value(best_bid=best_bid, best_ask=best_ask),
            "unrealized_pnl": self.portfolio.unrealized_pnl(best_bid=best_bid, best_ask=best_ask),
            "pnl": pnl,
            "reward": reward,
            "done": self.done
        })
        
        self.current_step += 1
        
        return reward
    
    def reset(self) -> tuple[np.ndarray, dict]:
        self.current_step = 0
        self.portfolio.reset()
        self.history.clear()
        self.done = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        reward = self._step_action(action)
        done = self.done
        next_observation = self._get_observation()
        info = self._get_info()
        
        return next_observation, reward, done, info