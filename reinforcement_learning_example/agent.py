"""This module contains an implementation of a Deep Q Agent.

The current implementation uses a policy and a target network as well as a Experience Replay.
It is able to solved the LunarLander-v2 problem from the gym library.

Potential improvements (as can be seen at the end of this video) could be:
- Making a double Deep Q Network. This means using two seperate weights for selecting the
    best action and evaluating the best action.
- Prioritized Experience Replay. This means selecting previous experiences from the Experience Replay
    based on the absolute Bellman error.
- Dueling Deep Q Network: This is a modification in the DeepQNetwork architecture. In the final layers of
    the network, split the network in two and create a scalar output (representing the Value of state S)
    and a vector output (representing et Value-Advantage).

Other intersting additions can be found here:
https://arxiv.org/pdf/1710.02298.pdf

This module serves mainly as a example for learning purposes. It would most likely be much more
effective and scalable to integrate a RL library such as Stable Baselines (at time of writing, this is
the latest repo: https://github.com/DLR-RM/stable-baselines3)
"""
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


class ReplayMemory:
    """"""

    def __init__(self, input_dims: tuple, memory_size: int = 10_000) -> None:
        """"""
        self.memory_size = memory_size
        self.memory_counter = 0

        self.state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.memory_counter += 1

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray]:

        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        state_batch = self.state_memory[batch]
        next_state_batch = self.next_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        return state_batch, next_state_batch, action_batch, reward_batch, terminal_batch


def exponential_decay(step_num: int, start: float = 1, end: float = 0.01, decay: float = 200) -> float:
    """"""
    return end + (start - end) * math.exp(-1 * step_num / decay)


class Agent:
    def __init__(
        self,
        gamma: float,
        input_dims: np.ndarray,
        batch_size: int,
        num_actions: int,
        network: nn.Module,
        network_kwargs: Dict[str, Any] = None,
        memory_size: int = 100000,
        epsilon_update_function: Callable = exponential_decay,
        epsilon_update_kwargs: Dict[str, Any] = None,
        update_target_every_n_steps: int = 10_000,
    ):
        self.gamma = gamma
        self.epsilon_update_function = epsilon_update_function
        self.epsilon_update_kwargs = epsilon_update_kwargs or {}
        self.action_space = list(range(num_actions))
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_target_every_n_steps = update_target_every_n_steps

        self.step_num = 0
        self.epsilon = self.epsilon_update_function(step_num=self.step_num, **self.epsilon_update_kwargs)

        network_kwargs = network_kwargs or {}
        self.policy_network = network(**network_kwargs)
        self.target_network = network(**network_kwargs)
        self._update_target_network()  # Set the target network weights equal to the policy weights.

        self.replay_memory = ReplayMemory(input_dims=input_dims, memory_size=memory_size)

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        self.replay_memory.store_transition(
            state=state, action=action, reward=reward, next_state=next_state, terminal=terminal
        )

    def choose_action(self, observation, use_epsilon_exploration: bool = True):

        if np.random.random() > self.epsilon or not use_epsilon_exploration:
            state = T.tensor(np.array([observation])).to(self.policy_network.device)
            actions = self.policy_network.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        """"""
        self.step_num += 1

        if self.replay_memory.memory_counter < self.batch_size:
            return

        (state_batch, next_state_batch, action_batch, reward_batch, terminal_batch) = self.replay_memory.get_batch(
            batch_size=self.batch_size
        )

        state_batch = T.tensor(state_batch, device=self.policy_network.device)
        next_state_batch = T.tensor(next_state_batch, device=self.policy_network.device)
        reward_batch = T.tensor(reward_batch, device=self.policy_network.device)
        terminal_batch = T.tensor(terminal_batch, device=self.policy_network.device)

        q = self.policy_network.forward(state_batch)[np.arange(self.batch_size, dtype=np.int32), action_batch]
        q_next = self.target_network.forward(next_state_batch)
        # q_next = self.policy_network.forward(next_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.policy_network.loss(q_target, q).to(self.policy_network.device)

        self.policy_network.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_network.optimizer.step()

        self.epsilon = self.epsilon_update_function(
            step_num=self.step_num - self.batch_size, **self.epsilon_update_kwargs
        )

        if self.step_num % self.update_target_every_n_steps == 0:
            self._update_target_network()

    def _update_target_network(self) -> None:
        logger.debug("Updating target network.")
        self.target_network.load_state_dict(self.policy_network.state_dict())


# class Agent:
#     def __init__(
#         self,
#         gamma,
#         epsilon,
#         learning_rate,
#         input_dims,
#         batch_size,
#         num_actions,
#         max_memory_size=100000,
#         epsilon_min=0.05,
#         epsilon_dec=5e-4,
#     ):
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_dec = epsilon_dec
#         self.learning_rate = learning_rate
#         self.action_space = [i for i in range(num_actions)]
#         self.memory_size = max_memory_size
#         self.batch_size = batch_size
#         self.memory_counter = 0
#         self.iter_counter = 0
#         self.replace_target = 100

#         self.policy_network = DeepQNetwork(
#             learning_rate, num_actions=num_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256
#         )

#         self.state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
#         self.next_state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
#         self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
#         self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
#         self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

#     def store_transition(self, state, action, reward, next_state, terminal):
#         index = self.memory_counter % self.memory_size
#         self.state_memory[index] = state
#         self.next_state_memory[index] = next_state
#         self.reward_memory[index] = reward
#         self.action_memory[index] = action
#         self.terminal_memory[index] = terminal

#         self.memory_counter += 1

#     def choose_action(self, observation):
#         if np.random.random() > self.epsilon:
#             state = T.tensor(np.array([observation])).to(self.policy_network.device)
#             actions = self.policy_network.forward(state)
#             action = T.argmax(actions).item()
#         else:
#             action = np.random.choice(self.action_space)

#         return action

#     def learn(self):
#         if self.memory_counter < self.batch_size:
#             return

#         self.policy_network.optimizer.zero_grad()

#         max_mem = min(self.memory_counter, self.memory_size)

#         batch = np.random.choice(max_mem, self.batch_size, replace=False)
#         batch_index = np.arange(self.batch_size, dtype=np.int32)

#         state_batch = T.tensor(self.state_memory[batch]).to(self.policy_network.device)
#         next_state_batch = T.tensor(self.next_state_memory[batch]).to(self.policy_network.device)
#         action_batch = self.action_memory[batch]
#         reward_batch = T.tensor(self.reward_memory[batch]).to(self.policy_network.device)
#         terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.policy_network.device)

#         q = self.policy_network.forward(state_batch)[batch_index, action_batch]
#         q_next = self.policy_network.forward(next_state_batch)
#         q_next[terminal_batch] = 0.0

#         q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

#         loss = self.policy_network.loss(q_target, q).to(self.policy_network.device)
#         loss.backward()
#         self.policy_network.optimizer.step()

#         self.iter_counter += 1
#         self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
