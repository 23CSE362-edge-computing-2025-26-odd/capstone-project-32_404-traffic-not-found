from stable_baselines3.common.vec_env import DummyVecEnv
from data_preprocessing import preprocess_data
from traffic_model import TrafficEnv
from agent import TrafficAgent

# 1. Load and preprocess the data
traffic_data = preprocess_data('traffic_data_new.csv')

# 2. Create the environment
env = DummyVecEnv([lambda: TrafficEnv(traffic_data)])

# 3. Create and load the agent
agent = TrafficAgent(env)
agent.load("ppo_traffic_controller")

# 4. Make a prediction
print("\n--- Starting Inference ---")
obs = env.reset()
action = agent.predict(obs)

# Map the action to a traffic light state
traffic_light_states = {0: 'Red', 1: 'Yellow', 2: 'Green'}
predicted_state = traffic_light_states[action[0]]

print(f"Sample Observation: {obs}")
print(f"Predicted Action (Traffic Light State): {predicted_state}")
print("--- Inference Finished ---")