# AI Traffic Light Controller using Deep Reinforcement Learning

## 🚀 Project Summary

This project documents the end-to-end creation of an intelligent, adaptive traffic light controller powered by **Deep Reinforcement Learning (DRL)**. The goal was to move beyond fixed-timer or simple sensor-based systems and build a sophisticated agent capable of making real-time, optimal decisions based on complex traffic data.

We began with the fundamental concepts of Reinforcement Learning and applied them to the complex problem of urban traffic management. The agent was trained on a real-world dataset (`traffic_data_new.csv`), ensuring its strategies are grounded in realistic traffic patterns and scenarios, not just simulations.

The final outcome is a fully functional DRL system that can:
- **Train** an agent from scratch on real-world traffic data to learn optimal control strategies.
- **Evaluate** the performance of the trained agent to understand its effectiveness.
- **Load** the trained agent and use it to **predict** the ideal traffic light state (Green, Yellow, or Red) for any given traffic condition.

This journey represents a complete machine learning project, moving from a foundational concept to a sophisticated, data-driven, and practical solution.

---

## 🧠 Conceptualization and Foundation

At its core, this project uses Reinforcement Learning to teach an agent how to optimize traffic flow. We established the core RL framework for this problem as follows:

- **State**: A snapshot of the traffic environment at a given moment. This is the information our agent "sees." We started with basic data like vehicle counts and speed, and later expanded it to include `Road_Occupancy_%` and `Weather_Condition` for more context.

- **Action**: The decision the agent makes based on the current state. In this project, the agent has three possible actions: turn the light **Green**, **Yellow**, or **Red**.

- **Reward**: A scoring system to provide feedback to the agent. We designed a reward function that encourages actions that improve traffic speed and reduce congestion. Essentially, the agent gets a "point" for making good decisions and loses a "point" for making bad ones. Over thousands of simulations, it learns to maximize its score, thereby learning to control traffic effectively.

---

## 🛠️ Project Structure

This project is organized into a modular structure, which is a best practice in software and machine learning engineering. Each file has a specific responsibility.

- **`data_preprocessing.py`**: This is the first step in our pipeline. It handles loading the `traffic_data_new.csv` file and preparing it for the model. This includes crucial data preprocessing steps, most notably using **one-hot encoding** to convert categorical weather data (like "Clear" or "Rainy") into a numerical format the neural network can understand.

- **`traffic_model.py`**: This file defines the custom simulation environment using the `gym` library. The `TrafficEnv` class is the "world" where our agent lives, learns, and interacts. It's responsible for managing the state, executing actions, and calculating rewards.

- **`agent.py`**: This script defines the `TrafficAgent`. It's a wrapper around the **Proximal Policy Optimization (PPO)** algorithm from the `stable-baselines3` library. PPO is a state-of-the-art DRL algorithm that is both powerful and stable. This file contains the logic for the agent to learn, save its "brain" (the trained model), and make predictions.

- **`run.py`**: This is the main script to kick off the training process. It orchestrates the entire workflow: it preprocesses the data, creates the environment, initializes the agent, and starts the learning loop.

- **`evaluate.py`**: After training, this script is used to objectively measure the performance of our agent. It runs the agent through a number of "episodes" (simulated traffic scenarios) and calculates the average reward, giving us a clear idea of how effective our agent's learned strategy is.

- **`infer.py`**: This script is for making a single, real-time prediction. It loads the pre-trained agent and feeds it a sample traffic state, and the agent outputs its decision for the traffic light. This demonstrates how the model would be used in a real-world application.

---

## ⚙️ How to Run

### 1. Install Dependencies

First, install all the required Python libraries.

It is recommended to create a `requirements.txt` file with the following content:

```
pandas
scikit-learn
joblib
```

Then install dependencies with:

```bash
pip install -r requirements.txt
```

---

### 2. Train the AI Model

Run the main training script. This will train a **Random Forest** model on 80% of the dataset and save it.

```bash
python train.py
```

After training completes, a file named **`traffic_model.joblib`** will be created in your project folder.
This is your trained model.

---

### 3. Make a Prediction

Once the model is trained, you can test it on a single row of your dataset.

Run the prediction script:

```bash
python predict.py
```

This will load the trained model and display a comparison between the **actual traffic light state** and the **predicted traffic light state**.
