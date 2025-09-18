import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traci
import numpy as np
import csv

# Import ANN
from ANN.model import TrafficANN, TrafficANNConfig

# =============== PPO Code ===============
class Memory:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.is_terminals = [], []

    def clear(self):
        self.states.clear(); self.actions.clear(); self.logprobs.clear()
        self.rewards.clear(); self.is_terminals.clear()

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, state, memory):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        memory.states.append(state_t)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return int(action.item())

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        return action_logprobs, torch.squeeze(state_values), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma; self.eps_clip = eps_clip; self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        rewards, discounted = [], 0
        for r, term in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if term: discounted = 0
            discounted = r + (self.gamma * discounted)
            rewards.insert(0, discounted)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if rewards.numel()>1 and rewards.std().item() != 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(memory.states).detach()
        old_actions = torch.tensor([int(a) for a in memory.actions], dtype=torch.long).detach()
        old_logprobs = torch.cat(memory.logprobs).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

# =============== ANN Loader ===============
def load_ann_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        cfg = TrafficANNConfig(**cfg_dict)
        model = TrafficANN(cfg)
        model.load_state_dict(checkpoint["model_state"])
    else:
        cfg = TrafficANNConfig(num_behaviors=5, backbone="mobilenet_v3_small")
        model = TrafficANN(cfg)
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# =============== Run Control ===============
def run_control(cfg_path, gui, ann_path, yellow_time, risk_threshold, steps):
    sumoBinary = "sumo-gui" if gui else "sumo"
    traci.start([sumoBinary, "-c", cfg_path])

    ann_model = load_ann_model(ann_path)
    print(f"[INIT] ANN model loaded from {ann_path}")

    ppo = PPO(state_dim=1, action_dim=4, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2)
    memory = Memory()

    history = []  # for CSV logging
    step = 0
    episode_wait_times = []

    while step < steps:
        avg_wait = np.mean([traci.edge.getWaitingTime(e) for e in traci.edge.getIDList()])
        episode_wait_times.append(avg_wait)

        # Dummy black image for ANN
        dummy_input = torch.zeros(1, 3, 224, 224)
        out = ann_model(dummy_input, apply_activation=True)
        risk_score = float(out["risk"].item())

        if risk_score > risk_threshold:
            print(f"[STEP {step}] High risk={risk_score:.2f} → YELLOW for {yellow_time}s")
            tls_ids = traci.trafficlight.getIDList()
            for tl in tls_ids:
                traci.trafficlight.setRedYellowGreenState(tl, "yyyy")
            for _ in range(yellow_time):
                traci.simulationStep()
                history.append([step, avg_wait, risk_score, "YELLOW"])
                step += 1
        else:
            state = np.array([avg_wait])
            action = ppo.policy_old.act(torch.FloatTensor(state), memory)
            traci.simulationStep()
            print(f"[STEP {step}] risk={risk_score:.2f}, PPO action={action}")
            history.append([step, avg_wait, risk_score, action])
            step += 1

    traci.close()
    print(f"[DONE] Avg wait time={np.mean(episode_wait_times):.2f}")

    # Save CSV
    with open("traffic_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_wait", "risk_score", "action"])
        writer.writerows(history)
    print("[SAVED] traffic_history.csv")

# =============== Main ===============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--yellow", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()

    run_control(args.cfg, args.gui, args.ann, args.yellow, args.threshold, args.steps)
