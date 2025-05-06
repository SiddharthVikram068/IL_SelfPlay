import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from dqn import DQN
import gymnasium as gym

# Load expert trajectories (modify to load your saved trajectories)
with open('expert_trajectories.pkl', 'rb') as f:
    expert_trajectories = pickle.load(f)

# Create Student Model (same architecture as DQN but fresh weights)
env = gym.make("LunarLander-v3")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_model = DQN(n_observations, n_actions).to(device)
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Prepare dataset from expert trajectories
states = []
actions = []
for trajectory in expert_trajectories:
    states.extend(trajectory['obs'])
    actions.extend(trajectory['actions'])

states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
actions = torch.tensor(actions, dtype=torch.long).to(device)

# Training loop
epochs = 100
batch_size = 64
dataset_size = len(states)

for epoch in range(epochs):
    # Shuffle data
    indices = torch.randperm(dataset_size)
    total_loss = 0.0
    batches = 0
    
    for start_idx in range(0, dataset_size, batch_size):
        # Get batch indices
        batch_indices = indices[start_idx:start_idx + batch_size]
        
        # Get batch data
        state_batch = states[batch_indices]
        action_batch = actions[batch_indices]
        
        # Forward pass
        logits = student_model(state_batch)
        loss = criterion(logits, action_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    avg_loss = total_loss / batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save the imitation learning model
torch.save(student_model.state_dict(), "models/imitation_model.pth")
print("Imitation Learning model saved successfully!")