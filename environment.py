import gym
from gym import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, window_size=30, risk_aversion=0.4, transaction_cost=0.001, initial_capital=100_000):
        super().__init__()
        self.prices = prices.values
        self.n_days, self.n_assets = self.prices.shape
        self.window_size = window_size
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        self.returns = np.log(self.prices[1:] / self.prices[:-1])
        obs_size = window_size * self.n_assets + self.n_assets

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value = self.initial_capital
        self.current_weights = np.array([1.0/self.n_assets]*self.n_assets, dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        start = self.current_step - self.window_size
        ret = self.returns[start:self.current_step].flatten()
        return np.concatenate([ret, self.current_weights]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        new_weights = action / action.sum() if action.sum() > 0 else self.current_weights

        weight_change = np.abs(new_weights - self.current_weights).sum()
        cost = self.transaction_cost * weight_change

        asset_returns = self.returns[self.current_step - 1]
        portfolio_return = np.dot(new_weights, asset_returns)
        self.portfolio_value *= np.exp(portfolio_return)

        risk = np.std(asset_returns)
        reward = portfolio_return - self.risk_aversion * risk - cost

        self.current_weights = new_weights
        self.current_step += 1
        done = self.current_step >= (self.n_days - 1)

        return self._get_obs(), float(reward), done, {"portfolio_value": self.portfolio_value}
