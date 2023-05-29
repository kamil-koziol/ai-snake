from dataclasses import dataclass
import numpy as np
import sys
from nn import NeuralNetwork
from game import Snake, MoveDirection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchrl.data import ReplayBuffer, ListStorage
from collections import deque

import random 

def state_to_tensor(state):
    return torch.from_numpy(state).float()


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes):
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_sizes[0]))
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        print(layer_sizes)
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.layers.append(nn.Linear(hidden_sizes[-1], action_size))

    def forward(self, state):
        x = state
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)
    

@dataclass
class Sample:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer2():
    def __init__(self, capacity, batch_size):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, idx):
        return self.memory[idx]
    
    def push(self, sample: Sample):
        self.memory.append(sample)
        
@dataclass
class DeepQAgentConfig:
    board_size: int
    piece_size: int

    epochs: int
    epoch_size: int

    game_steps_per_epoch: int

    epsilon_decay: float
    epsilon_min: float
    gamma: float

    lr: float
    batch_size: int
    memory_size: int

    dataloader_shuffle: bool
    dataloader_num_workers: bool

    hidden_sizes: list[int]


class DeepQAgent:
    # memory:
    def __init__(self, config: DeepQAgentConfig) -> None:
        self.cfg = config

        self.epsilon = 1.0

        self.game = DeepQSnake(config.board_size, config.piece_size)
        self.current_state = self.game.get_state()

        state_shape = self.current_state.shape
        # assert state_shape is (1, x)
        assert len(state_shape) == 2
        assert state_shape[0] == 1

        state_size = state_shape[1]
        action_size = 4

        self.agentNetwork = QNetwork(state_size, action_size, config.hidden_sizes)
        self.optimizer = optim.Adam(self.agentNetwork.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(
            batch_size=config.batch_size,
            storage=ListStorage(max_size=config.memory_size),
            collate_fn=lambda x: x
        )


    def train(self, after_epoch_callback = lambda: None):
        for epoch in range(self.cfg.epochs):

            for step in range(self.cfg.game_steps_per_epoch):
                # generate random number between 0 and 1
                rand = np.random.uniform(0, 1)
                if rand < self.epsilon:
                    # random MoveDirection - explore
                    action = np.random.randint(0, 4)
                else:
                    # predict MoveDirection - exploit
                    action = self.agentNetwork(state_to_tensor(self.current_state))
                    action = torch.argmax(action).item()
                
                next_state, reward, done = self.game.step(action)
                # add to memory
                sample = Sample(self.current_state, action, reward, next_state, done)
                self.memory.add(sample)

                self.current_state = next_state
                # restart game if needed
                if done:
                    self.game.restart()
                    self.current_state = self.game.get_state()


            for batch_idx, batch in enumerate(self.memory):
                self.train_step(batch)

            # decrease epsilon
            self.epsilon = max(self.epsilon * self.cfg.epsilon_decay, self.cfg.epsilon_min)

            after_epoch_callback(epoch)

    def train_step(self, batch):
        # unpack batch
        # states = tensor of states
        states = torch.tensor([state_to_tensor(sample.state) for sample in batch])
        actions = np.array([sample.action for sample in batch])
        rewards = np.array([sample.reward for sample in batch])
        next_states = torch.tensor([state_to_tensor(sample.next_state) for sample in batch])
        dones = np.array([sample.done for sample in batch])


        # calculate q_target
        with torch.no_grad():
            future_rewards = self.agentNetwork(next_states)
            q_target = rewards + self.cfg.gamma * future_rewards



        # this was for single sample
        # if done:
        #     q_target_for_action = reward
        # else:
        #     with torch.no_grad():
        #         future_reward = torch.argmax(self.agentNetwork(state_to_tensor(next_state)))
        #         q_target_for_action = reward + self.cfg.gamma * future_reward
        
        self.optimizer.zero_grad()

        pred = self.agentNetwork(state_to_tensor(state))
        target = pred.clone().detach()
        target[0][action] = q_target_for_action

        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.agentNetwork(state_to_tensor(state))


class DeepQSnake(Snake):
    def __init__(self, board_size, piece_size, hunger_enabled=False):
        super().__init__(board_size, piece_size, hunger_enabled)

    def get_state(self):
        return self.rays

    def is_alive(self):
        return self.alive

    def get_score(self):
        return self.apples_eaten
    
    def calc_apple_dist(self):
        dist = np.linalg.norm(self.apple.pos - self.pos)
        # normalize to 0-1
        return dist / self.board_size * np.sqrt(2)

    def step(self, action):
        score_before = self.get_score()
        self.set_move_dir(MoveDirection(action))

        super().update()

        apple_eaten = self.get_score() > score_before

        # check if game is over
        done = not self.is_alive()

        # calculate reward
        reward = (apple_eaten * 100) + (self.calc_apple_dist() * -10)  + (done * -100)

        return self.get_state(), reward, done

