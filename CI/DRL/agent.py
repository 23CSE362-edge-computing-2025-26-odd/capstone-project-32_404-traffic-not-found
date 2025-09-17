from stable_baselines3 import PPO

class TrafficAgent:
    def __init__(self, env):
        self.env = env
        self.model = PPO('MlpPolicy', self.env, verbose=1)

    def learn(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path, env=self.env)

    def predict(self, obs):
        action, _states = self.model.predict(obs)
        return action