from stable_baselines3.common.vec_env import DummyVecEnv
from data_preprocessing import preprocess_data
from traffic_model import TrafficEnv
from agent import TrafficAgent

# 1. Load and preprocess the data
traffic_data = preprocess_data('traffic_data_new.csv')

# 2. Create the environment
env = DummyVecEnv([lambda: TrafficEnv(traffic_data)])

# 3. Create the agent
agent = TrafficAgent(env)

# 4. Train the agent
print("--- Starting Training ---")
agent.learn(total_timesteps=10000)
print("--- Training Finished ---")

# 5. Save the trained model
agent.save("ppo_traffic_controller")
print("--- Model Saved ---")