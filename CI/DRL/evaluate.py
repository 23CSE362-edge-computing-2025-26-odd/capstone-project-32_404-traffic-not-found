from stable_baselines3.common.vec_env import DummyVecEnv
from data_preprocessing import preprocess_data
from traffic_model import TrafficEnv
from agent import TrafficAgent
import numpy as np

# 1. Load and preprocess the data
traffic_data = preprocess_data('traffic_data_new.csv')

# 2. Create the environment
env = DummyVecEnv([lambda: TrafficEnv(traffic_data)])

# 3. Create and load the agent
agent = TrafficAgent(env)
agent.load("ppo_traffic_controller")

# 4. Evaluate the agent
print("--- Starting Evaluation ---")
total_rewards = []
for i in range(10): # Evaluate for 10 episodes
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    total_rewards.append(episode_reward)

mean_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print("--- Evaluation Finished ---")