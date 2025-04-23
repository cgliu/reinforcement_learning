import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyperparameters
gamma = 0.99  # Discount factor
learning_rate = 0.001  # Learning rate
num_episodes = 500  # Number of episodes
batch_size = 64  # Mini-batch size
env_name = "CartPole-v1"  # Environment name

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_fc = nn.Linear(state_dim, 128)
        self.actor_fc = nn.Linear(128, action_dim)
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.shared_fc(state))
        action_probs = torch.softmax(self.actor_fc(x), dim=-1)  # Actor (policy)
        state_value = self.critic_fc(x)  # Critic (value function)
        return action_probs, state_value

# Training Function for Advantage Actor-Critic (A2C)
def train_a2c(env, model, optimizer):
    for episode in range(num_episodes):
        state = env.reset()[0]  # Reset environment and get initial state
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        done = False
        total_reward = 0
        log_probs = []
        values = []
        rewards = []
        entropy = 0

        while not done:
            action_probs, state_value = model(state)
            
            # Sample an action based on probabilities (Actor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            # Log probabilities for policy gradient
            log_prob = dist.log_prob(action)
            
            # Store values, rewards, and log probabilities
            log_probs.append(log_prob)
            values.append(state_value)
            
            # Take the action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            total_reward += reward
            
            state = next_state

            # Compute advantage
            next_value = model(state)[1] if not done else torch.tensor(0.0)
            td_error = reward + gamma * next_value - state_value
            advantage = td_error.detach()

            # Compute the loss for actor and critic
            actor_loss = -log_prob * advantage  # Policy loss (actor)
            critic_loss = td_error.pow(2)  # Value loss (critic)

            # Total loss
            loss = actor_loss + critic_loss

            # Backpropagate and update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Episode {episode+1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # Initialize environment, model, optimizer
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_a2c(env, model, optimizer)
    torch.save(model.state_dict(), "a2c_cartpole.pth")

    # Close environment
    env.close()
