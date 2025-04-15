import gym
import torch
from ppo import PPO
import time

# Setup environment
env = gym.make("LunarLander-v2", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
continuous = False

# Load agent with same architecture
agent = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=continuous,
    lr=3e-4,  # same hyperparameters
    gamma=0.99,
    epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    gae_lambda=0.95,
    epochs=10,
    batch_size=64
)

# Load trained weights
agent.policy.load_state_dict(torch.load("best_policy.pth"))
agent.policy.eval()

# Watch the trained agent
episodes_to_watch = 10

for ep in range(episodes_to_watch):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = agent.policy.get_action(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.02)  # Slow down to see properly

    print(f"Episode {ep+1} Reward: {total_reward:.2f}")

env.close()

