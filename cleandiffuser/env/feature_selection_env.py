import gym
import numpy as np
from gym import spaces
from sklearn.linear_model import LinearRegression

class FeatureSelectionEnv(gym.Env):
    def __init__(self, max_steps=59):
        super().__init__()
      
        self.state = np.ones(60, dtype=int)
        self.max_steps = max_steps
        self.steps_taken = 0
        self.action_space = spaces.Discrete(60)
        self.observation_space = spaces.Box(low=0, high=1, shape=(60,), dtype=int)
       
        np.random.seed(42)
        self.X = np.random.randn(1000, 60)  
        self.y = np.sum(self.X[:, :10] * np.random.randn(10), axis=1) + np.random.randn(1000) * 0.1 
        self.model = LinearRegression()
        self.prev_mse = float('inf')

    def reset(self):
        self.state = np.ones(60, dtype=int)
        self.steps_taken = 0
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)
        self.prev_mse = np.mean((y_pred - self.y) ** 2)
        return self.state.copy()

    def compute_mse_improvement(self, state):
        active_features = np.where(state == 1)[0]
        if len(active_features) == 0:
            return -1.0  # Penalty for no features
        X_subset = self.X[:, active_features]
        self.model.fit(X_subset, self.y)
        y_pred = self.model.predict(X_subset)
        mse = np.mean((y_pred - self.y) ** 2)
        reward = max(self.prev_mse - mse, -10.0)  
        self.prev_mse = mse
        return reward

    def step(self, action):
        if not (0 <= action < 60):
            raise ValueError("Action must be an integer between 0 and 59")
        if self.state[action] == 0:
            reward = -1.0 
        else:
            self.state[action] = 0
            reward = self.compute_mse_improvement(self.state)
        self.steps_taken += 1
        done = np.sum(self.state) == 1 or self.steps_taken >= self.max_steps
        return self.state.copy(), reward, done, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")