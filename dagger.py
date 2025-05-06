import torch
import numpy as np
import gymnasium as gym
from dqn import DQN
import pickle
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Load the expert and student models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v3")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Expert
expert_model = DQN(n_observations, n_actions).to(device)
expert_model.load_state_dict(torch.load("models/dqn_lunar_lander.pth", map_location=device))
expert_model.eval()

# Student (initially from behavioral cloning)
student_model = DQN(n_observations, n_actions).to(device)
student_model.load_state_dict(torch.load("models/imitation_model.pth", map_location=device))

# Try to load existing dataset or create new one
try:
    with open('dagger_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    # Start with expert trajectories
    with open('expert_trajectories.pkl', 'rb') as f:
        expert_trajectories = pickle.load(f)
    
    # Convert to dataset format
    dataset = {'states': [], 'actions': []}
    for traj in expert_trajectories:
        dataset['states'].extend(traj['obs'])
        dataset['actions'].extend(traj['actions'])

# DAgger parameters
n_dagger_iterations = 5
n_episodes_per_iteration = 10
epochs_per_iteration = 20
batch_size = 64

optimizer = optim.Adam(student_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for dagger_iter in range(n_dagger_iterations):
    print(f"DAgger Iteration {dagger_iter+1}/{n_dagger_iterations}")
    
    # Collect data with student policy, but label with expert
    for episode in tqdm(range(n_episodes_per_iteration), desc="Collecting data"):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Student collects state
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Student takes action
            with torch.no_grad():
                student_action = student_model(state_tensor).argmax(dim=1).item()
            
            # Expert labels the state
            with torch.no_grad():
                expert_action = expert_model(state_tensor).argmax(dim=1).item()
            
            # Add to dataset with expert label
            dataset['states'].append(state)
            dataset['actions'].append(expert_action)
            
            # Execute student's action to get next state
            next_state, _, terminated, truncated, _ = env.step(student_action)
            done = terminated or truncated
            state = next_state
    
    # Convert to tensors
    states = torch.tensor(np.array(dataset['states']), dtype=torch.float32).to(device)
    actions = torch.tensor(dataset['actions'], dtype=torch.long).to(device)
    
    # Train student on aggregated dataset
    dataset_size = len(states)
    for epoch in range(epochs_per_iteration):
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
        
        if epoch % 5 == 0:
            avg_loss = total_loss / batches
            print(f"  Epoch {epoch+1}/{epochs_per_iteration}, Loss: {avg_loss:.4f}")
    
    # Save the updated model
    torch.save(student_model.state_dict(), f"models/dagger_model_iter{dagger_iter+1}.pth")
    
    # Save the dataset
    with open('dagger_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

print("DAgger training complete!")