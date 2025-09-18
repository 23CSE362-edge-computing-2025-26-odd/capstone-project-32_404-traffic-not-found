# control_osm_train_ppo.py
# Put this file in the same folder as your osm.sumocfg
# Usage example (GUI):
#   python control_osm_train_ppo.py --cfg osm.sumocfg --gui --episodes 3 --steps 600
# Headless:
#   python control_osm_train_ppo.py --cfg osm.sumocfg --episodes 5 --steps 600

import argparse, csv, os, math, time
from pathlib import Path
import numpy as np
import traci
import torch
import torch.nn as nn
import torch.optim as optim
import json

# ----------------- PPO helpers -----------------
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
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
        rewards = []
        discounted = 0
        for r, term in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if term:
                discounted = 0
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
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

# ----------------- Env utils -----------------
STATE_DIM = 3    # [s_waiting, s_occ, s_time]
ACTION_DIM = 2   # 0=keep, 1=switch

ppo = PPO(STATE_DIM, ACTION_DIM)
memory = Memory()

def get_controlled_lanes(tl_id):
    links = traci.trafficlight.getControlledLinks(tl_id)
    lanes = []
    for phase in links:
        for link in phase:
            inlane = link[0]
            if inlane and inlane not in lanes:
                lanes.append(inlane)
    return lanes

def choose_busy_tl(tls_ids):
    best = None; best_count = -1
    for tl in tls_ids:
        try:
            lanes = get_controlled_lanes(tl)
            if len(lanes) > best_count:
                best_count = len(lanes); best = tl
        except Exception:
            continue
    return best if best is not None else tls_ids[0]

def read_state_for_tl(tl_id, last_switch_step, step):
    lanes = get_controlled_lanes(tl_id)
    waiting = 0
    occs = []
    for l in lanes:
        waiting += traci.lane.getLastStepHaltingNumber(l)
        occs.append(traci.lane.getLastStepOccupancy(l))
    mean_occ = float(np.mean(occs)) if occs else 0.0
    time_since = step - last_switch_step.get(tl_id, 0)
    s_wait = float(waiting) / 20.0
    s_occ  = float(mean_occ)
    s_time = float(time_since) / 30.0
    return [s_wait, s_occ, s_time], waiting, mean_occ

def get_phase_count(tl_id):
    defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
    if not defs: return 1
    return len(defs[0].phases)

# ----------------- Training loop -----------------
def run_episode(cfg_file, gui, episode_index, max_steps):
    sumo_bin = "sumo-gui" if gui else "sumo"
    traci.start([sumo_bin, "-c", cfg_file, "--start", "--quit-on-end"])
    step = 0
    tls = traci.trafficlight.getIDList()
    if not tls:
        print("No traffic lights in network. Exiting episode.")
        traci.close(); return None
    tl_to_control = choose_busy_tl(tls)
    print(f"[Episode {episode_index}] controlling TL: {tl_to_control} (lanes={len(get_controlled_lanes(tl_to_control))})")
    last_switch = {}
    total_wait_history = []
    while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
        traci.simulationStep()
        state, waiting, occ = read_state_for_tl(tl_to_control, last_switch, step)
        action = ppo.policy_old.act(state, memory)
        if action == 1:
            cur = traci.trafficlight.getPhase(tl_to_control)
            total_phases = get_phase_count(tl_to_control)
            traci.trafficlight.setPhase(tl_to_control, (cur + 1) % total_phases)
            last_switch[tl_to_control] = step
        reward = -float(waiting)
        memory.rewards.append(reward); memory.is_terminals.append(False)
        total_wait_history.append(waiting)
        step += 1
    traci.close()
    avg_wait = float(np.mean(total_wait_history)) if total_wait_history else 0.0
    print(f"[Episode {episode_index}] finished, avg waiting at TL = {avg_wait:.2f}")
    return avg_wait

def save_csv_history(history, file_path="ppo_history.csv"):
    with open(file_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "avg_waiting"])
        for i,val in enumerate(history):
            w.writerow([i, val])

def save_json_history(history, file_path="history.json"):
    data = {"episode_rewards": history}
    with open(file_path, "w") as f:
        json.dump(data, f)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to .sumocfg")
    ap.add_argument("--gui", action="store_true", help="use sumo-gui")
    ap.add_argument("--episodes", type=int, default=3, help="how many episodes to train")
    ap.add_argument("--steps", type=int, default=600, help="max steps per episode")
    ap.add_argument("--save", default="ppo_model.pth", help="where to save model")
    args = ap.parse_args()

    cfg = args.cfg
    episodes = args.episodes; steps = args.steps; gui = args.gui

    history = []
    for ep in range(episodes):
        avg_wait = run_episode(cfg, gui, ep, steps)
        if avg_wait is None:
            print("No TL found or episode aborted.")
            return
        history.append(avg_wait)
        print("[Training] Updating PPO policy...")
        ppo.update(memory)
        memory.clear()
        torch.save(ppo.policy.state_dict(), args.save)
        print(f"[Training] Model saved to {args.save}")

    save_csv_history(history, "ppo_history.csv")
    save_json_history(history, "history.json")
    print("Training complete. History saved to ppo_history.csv and history.json")

if __name__ == "__main__":
    main()
