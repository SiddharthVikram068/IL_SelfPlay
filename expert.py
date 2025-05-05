'''
In this code, we setup the environment and train an expert on it using PPO method. 

The default MLpPolicy uses 64 hidden layers per unit. Even though training can with 
100,000+ timesteps can take a significant amount of time, the task of inference is already
at a good optimised level. But a student model might be able to optimise it a much smaller size, 
let's see
'''

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# We are using the LunarLander environment for our experiment
env = gym.make("LunarLander-v3")

# Using PPO to create the initial expert
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_lunar_tensorboard/", n_steps=2048)


model.learn(total_timesteps=500_000)
model.save("ppo_lunarlander_expert")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Expert Performance: Mean Reward = {mean_reward}, Std = {std_reward}")

env.close()