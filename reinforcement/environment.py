#reinforcement/environment
import numpy as np

class TradingEnvironment:
    def __init__(self, prices, window_size=10):
        self.prices = prices
        self.window = window_size
        self.reset()

    def reset(self):
        self.current_step = self.window
        self.balance = 1.0
        self.position = 0
        return self._get_state()

    def _get_state(self):
        window = self.prices[self.current_step - self.window:self.current_step]
        normed = window / window[0] - 1
        return np.array(normed)

    def step(self, action):
        price = self.prices[self.current_step]
        reward = 0
        if action == 1:
            self.position = self.balance / price
            self.balance = 0
        elif action == 2 and self.position > 0:
            self.balance = self.position * price
            reward = self.balance - 1.0
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        return self._get_state(), reward, done