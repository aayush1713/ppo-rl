import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

from networks import PolicyNetwork, ValueNetwork


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 continuous=False,
                 lr=3e-4,
                 gamma=0.99,
                 epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 gae_lambda=0.95,
                 epochs=10,
                 batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.continuous = continuous

        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim, continuous)
        self.value = ValueNetwork(state_dim)

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    def compute_gae(self, rewards, values, next_values, done):
        # Initialize advantages and returns as zeros
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        # Initialize GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            # If terminal state, next value is 0
            next_val = next_values[t] * (1 - done[t])

            # Compute TD error
            delta = rewards[t] + self.gamma * next_val - values[t]

            # Compute GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - done[t]) * gae
            advantages[t] = gae

            # Compute returns
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def collect_trajectory(self, env, num_steps):
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        state, _ = env.reset()
        done = False

        for _ in range(num_steps):
            # The current state (state) is converted into a PyTorch tensor (state_tensor). PyTorch models typically
            # expect input in the form of tensors (multi-dimensional arrays). The current state (state) is converted
            # into a PyTorch tensor (state_tensor). PyTorch models typically expect input in the form of tensors (
            # multi-dimensional arrays).
            state_tensor = torch.FloatTensor(state)
            # This is passing the state tensor into the value function (typically the critic part of the PPO). The
            # value function estimates how "good" the current state is in terms of expected future rewards. It is
            # usually a neural network.
            value = self.value(state_tensor).detach().numpy()[0]

            # get action
            # This calls the policy network (or actor) to get the action the agent should take given the current state.
            action, log_prob = self.policy.get_action(state)

            # take action.
            # This applies the action to the environment, meaning the agent is interacting with the
            # environment. The environment then responds with: next_state,reward etc.

            new_state, reward, terminated , truncated, _ = env.step(action)
            done = terminated or truncated

            # store experience
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            # update next state
            state = new_state

            if done:
                state, _ = env.reset()

        # get values for final state

        state_tensor = torch.FloatTensor(state)
        # It's the value network (also called the critic).
        # It takes the current state and estimates how good that state is — i.e.,
        # how much future reward we can expect if we follow our policy from this state.
        # It returns a scalar value for that state.
        next_value = self.value(state_tensor).detach().numpy()[0] * (1 - done)
        # We add [next_value] because: You need the value of the next state for every time step. But the last one
        # (V(sₜ₊₁)) was’t collected because for the last one there might e no next reward hence no next state,
        # so we estimate it 0 and add it manually.
        next_values = values[1:] + [next_value]

        # compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # convert to tensors

        # Why FloatTensor? States are usually things like positions, angles, velocities — all real-valued numbers
        # (e.g., 0.3, 1.5, -2.0). Neural networks expect float inputs. So we use FloatTensor to convert them into
        # the right format.
        states = torch.FloatTensor(np.array(states))
        if self.continuous:
            # We use FloatTensor for anything that will go into a network as input or needs floating-point math.
            actions = torch.FloatTensor(np.array(actions))

        else:
            # We use LongTensor only for discrete actions,
            # because it acts like a class label in classification problems.
            actions = torch.LongTensor(np.array(actions))

        old_log_probs = torch.FloatTensor(np.array(log_probs))
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            'states': states,
            'actions': actions,
            'advantages': advantages,
            'old_log_probs': old_log_probs,
            'returns': returns

        }

    def update(self, experiences):

        states = experiences['states']
        actions = experiences['actions']
        old_log_probs = experiences['old_log_probs']
        advantages = experiences['advantages']
        returns = experiences['returns']

        policy_losses = []
        value_losses = []
        entropy_losses = []

        # this loop is doing:
        # Shuffle all the data
        # ✅ Split it into mini-batches of size batch_size (e.g., 64)
        # ✅ Use each mini-batch for training (calculating loss and updating the model)
        for _ in range(self.epochs):

            # Shuffle indices to randomize the order of experiences for better training stability
            indices = np.random.permutation(len(states))

            # This loop’s goal is to create mini-batches from the collected rollout data.
            # Let’s say:
            # You collected 2048 states (i.e., len(states) = 2048)
            # You set self.batch_size = 64 Then,
            # range(0, 2048, 64)
            # Will generate: [0, 64, 128, 192, ..., 1984]
            # That means you’ll get:
            # 👉 32 mini-batches, each of size 64
            # 👉 Each idx will contain 64 shuffled indices
            # 👉 Each mini-batch will be used to do one gradient update step
            for indx in range(0, len(states), self.batch_size):
                idx = indices[indx: indx + self.batch_size]

                mini_states = states[idx]
                mini_actions = actions[idx]
                mini_old_log_probs = old_log_probs[idx]
                mini_advantages = advantages[idx]  # Fixed typo
                mini_returns = returns[idx]

                if self.continuous:
                    mean, std = self.policy(mini_states)
                    dist = Normal(mean, std)
                    current_log_probs = dist.log_prob(mini_actions).sum(dim=1)
                    entropy = dist.entropy().sum(dim=1).mean()

                else:
                    probs = self.policy(mini_states)
                    dist = Categorical(probs)
                    current_log_probs = dist.log_prob(mini_actions)
                    entropy = dist.entropy().mean()

                # finding ratio for surrogate
                ratio = torch.exp(current_log_probs - mini_old_log_probs)

                # finding the surrogate
                surr1 = ratio * mini_advantages  # Fixed typo
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1 + self.epsilon) * mini_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # finding value loss
                value_pred = self.value(mini_states).squeeze()
                # Why MSE? Because it's a regression task — you're not predicting an action or class,
                # we are predicting a continuous value (the return).
                value_loss = nn.MSELoss()(value_pred, mini_returns)

                # Compute entropy bonus
                entropy_loss = -entropy

                # total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # update loss

                # 1. Zero the gradients for both the policy and value networks to avoid accumulation
                self.policy_optimizer.zero_grad()  # Clear previous gradients for policy network
                self.value_optimizer.zero_grad()  # Clear previous gradients for value network

                # 2. Compute gradients by performing backpropagation
                total_loss.backward()  # Calculate the gradients of the total loss with respect to the parameters

                # 3. Update the policy and value networks' parameters based on computed gradients.
                # Optimizer update the weights.
                # An optimizer is a tool (or algorithm) that updates the weights of your neural network,
                # so that it gets better at its task
                self.policy_optimizer.step()  # Update policy network's parameters
                self.value_optimizer.step()  # Update value network's parameters

                # store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())  # Fixed typo
                entropy_losses.append(entropy_loss.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),  # Fixed typo
            'entropy_loss': np.mean(entropy_losses)
        }
