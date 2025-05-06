import torch
import numpy as np
import gymnasium as gym
from dqn import DQN
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v3")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Parameters
MEMORY_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR = 0.0001
N_ITERATIONS = 50
EPISODES_PER_ITER = 20

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def evaluate_policy(model, env, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            state = next_state
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

# Start with the best model so far
best_model = DQN(n_observations, n_actions).to(device)
try:
    # Try to load the DAgger model first
    model_path = "models/dagger_model_iter5.pth"
    if not os.path.exists(model_path):
        model_path = "models/imitation_model.pth"
    if not os.path.exists(model_path):
        model_path = "models/dqn_lunar_lander.pth"
    
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded initial model from {model_path}")
except:
    print("Starting with a fresh model")

# Create target model
target_model = DQN(n_observations, n_actions).to(device)
target_model.load_state_dict(best_model.state_dict())
target_model.eval()

# Memory for self-play
memory = ReplayMemory(MEMORY_SIZE)

optimizer = optim.Adam(best_model.parameters(), lr=LR)

print("Starting self-play training...")
best_reward = evaluate_policy(best_model, env)
print(f"Initial performance: {best_reward:.2f}")

for iteration in range(N_ITERATIONS):
    # Collect experience with current best model
    for episode in range(EPISODES_PER_ITER):
        state, _ = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Epsilon-greedy action selection (with small epsilon)
            if random.random() < 0.05:  # 5% exploration
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = best_model(state_tensor).argmax(dim=1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            memory.push(
                state,
                action,
                next_state if not done else None,
                reward,
                done
            )
            
            state = next_state
            
            # Train if enough samples
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                
                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=device, dtype=torch.bool)
                
                non_final_next_states = torch.tensor(
                    np.array([s for s in batch.next_state if s is not None]),
                    dtype=torch.float32, device=device)
                
                state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
                action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, device=device)
                done_batch = torch.tensor(batch.done, device=device, dtype=torch.bool)
                
                # Compute Q(s_t, a)
                state_action_values = best_model(state_batch).gather(1, action_batch)
                
                # Compute expected Q values
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[0]
                
                expected_state_action_values = (next_state_values * GAMMA * (~done_batch)) + reward_batch
                
                # Compute loss and optimize
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Soft update target network
                for target_param, local_param in zip(target_model.parameters(), best_model.parameters()):
                    target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
    
    # Evaluate current policy
    current_reward = evaluate_policy(best_model, env)
    print(f"Iteration {iteration+1}/{N_ITERATIONS}, Reward: {current_reward:.2f}")
    
    # Save if improved
    if current_reward > best_reward:
        best_reward = current_reward
        torch.save(best_model.state_dict(), f"models/self_play_model_iter{iteration+1}.pth")
        print(f"  Saved improved model with reward {best_reward:.2f}")

print(f"Self-play training complete! Best reward: {best_reward:.2f}")
torch.save(best_model.state_dict(), "models/self_play_best.pth")