import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Discretize the action space
ACTION_DISCRETIZATION = 5
ACTION_SPACE = np.linspace(-2, 2, ACTION_DISCRETIZATION)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))  # Output probability distribution

def train_reinforce():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = len(ACTION_SPACE)

    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    gamma = 0.99
    episodes = 500

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        for _ in range(200):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            action_idx = torch.multinomial(action_probs, 1).item()

            action = ACTION_SPACE[action_idx]
            next_state, reward, done, _, _ = env.step([action])

            log_probs.append(torch.log(action_probs[action_idx]))
            rewards.append(reward)

            state = next_state
            total_reward += reward

        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        # Compute policy gradient loss
        loss = torch.stack([-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    torch.save(policy_net.state_dict(), "reinforce_pendulum.pth")
    env.close()

if __name__ == "__main__":    
    # Train REINFORCE
    train_reinforce()
