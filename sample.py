import gym

env = gym.make("LunarLander-v2")
obs = env.reset()
env.render()
