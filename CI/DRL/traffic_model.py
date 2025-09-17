import gym
from gym import spaces
import numpy as np

class TrafficEnv(gym.Env):
    """Custom Environment for Traffic Light Control."""
    def __init__(self, df):
        super(TrafficEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.max_steps = len(df) - 1

        # Define action and observation space
        # Actions: 0 for Red, 1 for Yellow, 2 for Green
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32)

    def reset(self):
        """Reset the state of the environment to an initial state."""
        self.current_step = 0
        return self.df.iloc[self.current_step].values

    def step(self, action):
        """Execute one time step within the environment."""
        self.current_step += 1
        if self.current_step > self.max_steps:
            self.current_step = 0 # Loop back to the start of the data

        # Define the reward function
        state = self.df.iloc[self.current_step]
        # Reward is higher for lower traffic speed (less congestion) and lower vehicle count
        reward = - (state['Traffic_Speed_kmh'] * 0.5 + state['Vehicle_Count'] * 0.5)

        done = self.current_step >= self.max_steps
        info = {}

        return state.values, reward, done, info

    def render(self, mode='human'):
        pass