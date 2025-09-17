import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_bas_lines3.common.vec_env import DummyVecEnv

# This special function creates a "preprocessor" that is aware of the entire dataset.
# This ensures that a single row is processed exactly the same way as the training data.
def create_preprocessor(full_data_path):
    # Load the full dataset to learn the data's structure and all possible categories
    full_df = pd.read_csv(full_data_path, thousands='.')
    
    # We select the same important features as in our final training script
    important_features = [
        'Vehicle_Count', 
        'Traffic_Speed_kmh', 
        'Road_Occupancy_%', 
        'Weather_Condition',
        'Traffic_Light_State'
    ]
    full_df = full_df[important_features]
    
    # Fit the one-hot encoder on the 'Weather_Condition' column of the full dataset
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(full_df[['Weather_Condition']])
    
    # This is the actual function that will be used on new, single rows of data
    def preprocess_single_row(data_row_df):
        # Apply the pre-fitted encoder
        weather_encoded = encoder.transform(data_row_df[['Weather_Condition']])
        weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['Weather_Condition']), index=data_row_df.index)
        
        # Combine and clean the data
        processed_df = pd.concat([data_row_df.drop(columns=['Weather_Condition']), weather_df], axis=1)
        
        # We need to drop the answer column for prediction
        if 'Traffic_Light_State' in processed_df.columns:
            processed_df = processed_df.drop(columns=['Traffic_Light_State'])
        
        for col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Fill any missing values with the average from the full training data
        processed_df.fillna(full_df.mean(numeric_only=True), inplace=True)
        return processed_df
        
    return preprocess_single_row

# We need the Environment and Agent classes for the model to load correctly
class TrafficEnv(gym.Env):
    def __init__(self, df):
        super(TrafficEnv, self).__init__()
        self.df = df
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
    def reset(self): return self.df.iloc[0].values.astype(np.float32)
    def step(self, action): return self.df.iloc[0].values.astype(np.float32), 0, True, {}

class TrafficAgent:
    def __init__(self, env):
        self.env = env
    def load(self, path):
        self.model = PPO.load(path, env=self.env)
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

# --- Main execution block ---
if __name__ == "__main__":
    print("--- 🔮 AI Prediction Script ---")

    # --- Configuration ---
    DATA_FILE_PATH = 'traffic_data_new.csv'
    MODEL_PATH = 'ppo_traffic_controller.zip'
    ROW_TO_TEST = 15 # <--- YOU CAN CHANGE THIS NUMBER to test any row

    # 1. Load data and select the row
    raw_data = pd.read_csv(DATA_FILE_PATH, thousands='.')
    single_row_df = raw_data.iloc[[ROW_TO_TEST]]
    actual_value = single_row_df['Traffic_Light_State'].values[0]

    # 2. Preprocess the single row
    preprocessor = create_preprocessor(DATA_FILE_PATH)
    processed_row = preprocessor(single_row_df)

    # 3. Load the trained agent
    dummy_env = DummyVecEnv([lambda: TrafficEnv(processed_row)])
    agent = TrafficAgent(dummy_env)
    agent.load(MODEL_PATH)

    # 4. Make a prediction
    observation = processed_row.values.astype(np.float32)
    prediction_id = agent.predict(observation)[0]
    traffic_light_states = {0: 'Red', 1: 'Yellow', 2: 'Green'}
    predicted_value = traffic_light_states[prediction_id]

    # --- Display Final Comparison ---
    results_df = pd.DataFrame({
        "Category": ["Actual Value", "Predicted Value"],
        "Traffic Light State": [actual_value, predicted_value]
    })
    print(f"\n--- 📝 Prediction Result for Row {ROW_TO_TEST} ---")
    print(results_df.to_string(index=False))
    
    if actual_value == predicted_value:
        print("\n✅ Match! The model's prediction was correct for this row.")
    else:
        print("\n❌ No Match. The model's prediction was different.")