import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from ppo import PPO

# Initialize environment
env = gym.make("LunarLander-v2")
state_dim = env.observation_space.shape[0]  # 8 for LunarLander
action_dim = env.action_space.n  # 4 for discrete LunarLander
continuous = False  # Use True for LunarLanderContinuous

# Initialize PPO agent
agent = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=continuous,
    lr=3e-4,
    gamma=0.99,
    epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    gae_lambda=0.95,
    epochs=10,
    batch_size=64
)

# Training parameters
num_episodes = 1000
steps_per_update = 2048
max_steps_per_episode = 1000
target_reward = 250  # Consider the environment solved at this average reward

# For tracking progress
episode_rewards = []
avg_rewards = []
best_avg_reward = -float('inf')

# Training loop
for episode in range(num_episodes):
    # Collect trajectories
    experiences = agent.collect_trajectory(env, steps_per_update)

    # Update agent
    losses = agent.update(experiences)

    # Evaluate agent (test episode without exploration)
    state, _ = env.reset()
    episode_reward = 0
    done = False
    steps = 0

    while not done and steps < max_steps_per_episode:
        action, _ = agent.policy.get_action(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state
        steps += 1

    # Store reward
    episode_rewards.append(episode_reward)

    # Calculate average reward over last 100 episodes
    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    avg_rewards.append(avg_reward)

    # Save best model
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(agent.policy.state_dict(), 'best_policy.pth')
        torch.save(agent.value.state_dict(), 'best_value.pth')

    # Print progress
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Avg Reward = {avg_reward:.2f}")
    print(
        f"Losses: Policy = {losses['policy_loss']:.4f}, Value = {losses['value_loss']:.4f}, Entropy = {losses['entropy_loss']:.4f}")

    # Check if solved
    if avg_reward >= target_reward and len(episode_rewards) >= 100:
        print(f"Environment solved in {episode + 1} episodes!")
        break

# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Episode Reward')
plt.plot(avg_rewards, label='100-Episode Average')
plt.axhline(y=target_reward, color='r', linestyle='--', label='Target Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.title('PPO Training Progress')
plt.savefig('training_progress.png')
plt.show()

# Close environment
env.close()