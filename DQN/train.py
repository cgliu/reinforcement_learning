import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Discretize the action space
ACTION_DISCRETIZATION = 5
ACTION_SPACE = np.linspace(-2, 2, ACTION_DISCRETIZATION)

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each discrete action

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# Train the DQN
def train_dqn():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = len(ACTION_SPACE)

    q_network = DQN(state_dim, action_dim)
    target_q_network = DQN(state_dim, action_dim)
    target_q_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    batch_size = 64
    episodes = 500

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(200):
            # Select action (ε-greedy)
            if np.random.rand() < epsilon:
                action_idx = np.random.choice(action_dim)
            else:
                with torch.no_grad():
                    q_values = q_network(torch.tensor(state, dtype=torch.float32))
                    action_idx = torch.argmax(q_values).item()

            action = ACTION_SPACE[action_idx]
            next_state, reward, done, _, _ = env.step([action])

            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the network
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_q_network(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = loss_fn(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    torch.save(q_network.state_dict(), "dqn_pendulum.pth")
    env.close()

if __name__ == "__main__":
    # Train DQN
    train_dqn()
