from __future__ import annotations
import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import logging

logger = logging.getLogger(__loader__.name)


class DeepQNetwork(nn.Module):
    def __init__(
        self,
        input_dims: tuple,
        learning_rate: float = 0.001,
        fc1_dims: int = 64,
        fc2_dims: int = 64,
        num_actions: int = 4,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.num_actions = num_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.HuberLoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        learning_rate,
        input_dims,
        batch_size,
        num_actions,
        max_memory_size=100000,
        epsilon_min=0.05,
        epsilon_dec=5e-4,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.learning_rate = learning_rate
        self.action_space = [i for i in range(num_actions)]
        self.memory_size = max_memory_size
        self.batch_size = batch_size
        self.memory_counter = 0
        self.iter_counter = 0
        self.replace_target = 100

        self.policy_network = DeepQNetwork(
            learning_rate, num_actions=num_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256
        )

        self.state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, terminal):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation])).to(self.policy_network.device)
            actions = self.policy_network.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory_counter < self.batch_size:
            return

        self.policy_network.optimizer.zero_grad()

        max_mem = min(self.memory_counter, self.memory_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.policy_network.device)
        next_state_batch = T.tensor(self.next_state_memory[batch]).to(self.policy_network.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.policy_network.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.policy_network.device)

        q = self.policy_network.forward(state_batch)[batch_index, action_batch]
        q_next = self.policy_network.forward(next_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.policy_network.loss(q_target, q).to(self.policy_network.device)
        loss.backward()
        self.policy_network.optimizer.step()

        self.iter_counter += 1
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
