import torch
import numpy as np
import gymnasium as gym
from dqn import DQN
import matplotlib.pyplot as plt

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v3", render_mode="rgb_array")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Load all models to compare
models = {
    "Expert DQN": "models/dqn_lunar_lander.pth",
    "Behavioral Cloning": "models/imitation_model.pth",
    "DAgger": "models/dagger_model_iter5.pth",  # Use the latest DAgger model
    "Self-Play": "models/self_play_best.pth"
}

# Function to evaluate a model
def evaluate_model(model_path, n_episodes=20):
    model = DQN(n_observations, n_actions).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except:
        print(f"Failed to load model: {model_path}")
        return []
    
    rewards = []
    for i in range(n_episodes):
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
        
        rewards.append(total_reward)
    
    return rewards

# Evaluate all models
results = {}
for name, path in models.items():
    try:
        print(f"Evaluating {name}...")
        rewards = evaluate_model(path)
        results[name] = rewards
        print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Plot results
plt.figure(figsize=(12, 6))
plt.boxplot([results[name] for name in results.keys()], labels=list(results.keys()))
plt.ylabel('Total Reward')
plt.title('Performance Comparison of Different Models')
plt.savefig('model_comparison.png')
plt.show()

# Print summary
print("\nSummary:")
for name in results:
    rewards = results[name]
    print(f"{name}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

env.close()