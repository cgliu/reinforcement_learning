import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

from train import (ACTION_DISCRETIZATION, ACTION_SPACE, DQN)

env = gym.make("Pendulum-v1", render_mode="rgb_array")
q_network = DQN(env.observation_space.shape[0], len(ACTION_SPACE))
q_network.load_state_dict(torch.load("dqn_pendulum.pth"))

obs, _ = env.reset()
# plt.ion()
img = plt.imshow(env.render())

for _ in range(200):
    with torch.no_grad():
        q_values = q_network(torch.tensor(obs, dtype=torch.float32))
        action_idx = torch.argmax(q_values).item()

    action = ACTION_SPACE[action_idx]
    obs, reward, done, _, _ = env.step([action])
    img.set_data(env.render())
    plt.pause(0.1)
    if done:
        break;

env.close()
