from enum import Enum
from collections import deque

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    DONT_HOLD = 0
    HOLD_A_LONG = 1
    HOLD_A_SHORT = 2

class Position(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

class CryptoTradeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self,
                 data: pd.DataFrame,
                 max_steps: int,
                 initial_collateral: float,
                 transaction_fee: float,
                 trading_volume: float,
                 profit_target: float,
                 feature_columns: list):
        """
        Args:
            data (pd.DataFrame): DataFrame containing market data with at least 'best_bid' and 'best_ask' columns.
            max_steps (int): Maximum number of steps per episode.
            initial_collateral (float): Initial collateral amount for the portfolio.
            transaction_fee (float): Transaction fee percentage (e.g., 0.01 for 0.01%).
            trading_volume (float): Fixed trading volume for each trade action.
            profit_target (float): Profit target relative to theoretical break-even distance.
            feature_columns (list): List of feature column names to include in the observation.
        """
        super().__init__()
        self.data = data
        self.max_steps = max_steps
        self.initial_collateral = initial_collateral
        self.transaction_fee = transaction_fee
        self.trading_volume = trading_volume
        self.profit_target = profit_target
        self.feature_columns = feature_columns
        
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_columns)+2,), dtype=np.float32)
        
        self.history = deque(maxlen=len(self.data))
        
        self.reset()
        self._validate_data()
    
    def _validate_data(self) -> ValueError|None:
        if type(self.data) is not pd.DataFrame:
            raise ValueError("Data must be a pandas DataFrame")
        
        required_columns = ['best_bid', 'best_ask'] + self.feature_columns
        for required_column in required_columns:
            if required_column not in self.data.columns:
                raise ValueError(f"Data must contain the column: {required_column}")
    
    @property
    def unrealized_pnl(self) -> float:
        """Future Realized P&L based on current position and market prices.
        
        To lower the barrier to entry,
        calculate P&L based solely on the spread difference,
        disregarding transaction fees.
        """
        current_data = self.data.iloc[self.current_step]
        current_best_bid = current_data['best_bid']
        current_best_ask = current_data['best_ask']
        
        if self.position == Position.LONG and self.entry_price > 0:
            difference = current_best_bid - self.entry_price
            return difference * self.trading_volume
        
        elif self.position == Position.SHORT and self.entry_price > 0:
            difference = self.entry_price - current_best_ask
            return difference * self.trading_volume
        
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        if self.current_step <= self.max_steps:
            current_data = self.data.iloc[self.current_step]
            current_best_bid = current_data['best_bid']
            current_best_ask = current_data['best_ask']
            current_spread = current_best_ask - current_best_bid
            
            features = current_data[self.feature_columns].values.astype(np.float32)
            position = self.position.value
            unrealized_pnl = 0.
            
            if self.position == Position.LONG and self.entry_price > 0:
                difference = current_best_bid - self.entry_price
                fee = (self.entry_price + current_best_bid) * self.transaction_fee * self.trading_volume
                theoretical_breakeven = fee + current_spread
                profit_target = theoretical_breakeven * (1 + self.profit_target)
                unrealized_pnl = (difference * self.trading_volume - fee) / profit_target
            
            elif self.position == Position.SHORT and self.entry_price > 0:
                difference = self.entry_price - current_best_ask
                fee = (self.entry_price + current_best_ask) * self.transaction_fee * self.trading_volume
                theoretical_breakeven = fee + current_spread
                profit_target = theoretical_breakeven * (1 + self.profit_target)
                unrealized_pnl = (difference * self.trading_volume - fee) / profit_target
            
            return np.concatenate([
                features,
                [
                    position,
                    unrealized_pnl
                ]
            ]).astype(np.float32)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _close_position(self) -> float:
        current_data = self.data.iloc[self.current_step]
        current_best_bid = current_data['best_bid']
        current_best_ask = current_data['best_ask']
        
        pnl = 0.
        
        if self.position == Position.LONG and self.entry_price > 0:
            difference = current_best_bid - self.entry_price
            fee = (self.entry_price + current_best_bid) * self.transaction_fee * self.trading_volume
            pnl = difference * self.trading_volume - fee
            self.collateral += pnl
            self.position = Position.FLAT
            self.entry_price = 0.0
        
        elif self.position == Position.SHORT and self.entry_price > 0:
            difference = self.entry_price - current_best_ask
            fee = (self.entry_price + current_best_ask) * self.transaction_fee * self.trading_volume
            pnl = difference * self.trading_volume - fee
            self.collateral += pnl
            self.position = Position.FLAT
            self.entry_price = 0.0
        
        return pnl
    
    def _get_info(self) -> dict:
        return self.history[-1] if self.history else {}
    
    def _step_action(self, action: Actions) -> float:
        current_data = self.data.iloc[self.current_step]
        current_best_bid = current_data['best_bid']
        current_best_ask = current_data['best_ask']
        
        reward = 0.
        
        if action == Actions.DONT_HOLD:
            self._close_position()
        
        elif action == Actions.HOLD_A_LONG and self.position != Position.LONG:
            if self.position == Position.SHORT:
                self._close_position()
            self.position = Position.LONG
            self.entry_price = current_best_ask
        
        elif action == Actions.HOLD_A_SHORT and self.position != Position.SHORT:
            if self.position == Position.LONG:
                self._close_position()
            self.position = Position.SHORT
            self.entry_price = current_best_bid
        
        prev_total_value = self.history[-1]['total_value'] if len(self.history) != 0 else self.initial_collateral
        current_total_value = self.collateral + self.unrealized_pnl
        
        reward += np.log(max(current_total_value, 1e-8) / max(prev_total_value, 1e-8))
        reward *= 100.0
        
        if self.current_step >= self.max_steps or self.collateral <= 0:
            self.done = True
        
        self.history.append({
            "step": self.current_step,
            "action": action,
            "best_bid": current_best_bid,
            "best_ask": current_best_ask,
            "collateral": self.collateral,
            "positions": self.position,
            "total_value": current_total_value,
            "unrealized_pnl": self.unrealized_pnl,
            "reward": reward,
            "done": self.done
        })
        
        return round(reward, 8)
    
    def reset(self) -> tuple[np.ndarray, dict]:
        self.current_step = 0
        self.entry_price = 0.
        self.position = Position.FLAT
        self.collateral = self.initial_collateral
        self.history.clear()
        self.done = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        reward = self._step_action(Actions(action))
        next_observation = self._get_observation()
        done = self.done
        info = self._get_info()
        
        self.current_step += 1
        
        return next_observation, reward, done, info