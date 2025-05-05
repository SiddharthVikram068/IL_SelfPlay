# generate_dqn_expert_trajectories.py

import os
import torch
import gymnasium as gym
from dqn import DQN
import numpy as np
from itertools import count

# ==== Setup ====
env_id = "LunarLander-v3"  # Matches your environment
video_folder = "./dqn_expert_videos/"
os.makedirs(video_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Create Environment with Video Recording ====
env = gym.make(env_id, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder,
    episode_trigger=lambda ep: ep < 2,  # Save first 2 episodes
    name_prefix="dqn_expert"
)

# ==== Load Model ====
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("models/dqn_lunar_lander.pth", map_location=device))
policy_net.eval()

print("DQN model loaded successfully!")

# ==== Generate Expert Trajectories ====
expert_trajectories = []
n_episodes = 5

for episode in range(n_episodes):
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    trajectory = {"obs": [], "actions": [], "rewards": []}
    total_reward = 0

    for t in count():
        with torch.no_grad():
            action = policy_net(state_tensor).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)

        trajectory["obs"].append(state)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)

        total_reward += reward
        done = terminated or truncated
        if done:
            break

        state = next_state
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    expert_trajectories.append(trajectory)
    print(f"Episode {episode + 1} | Total reward: {total_reward}")

env.close()
print(f"Saved videos to: {video_folder}")
