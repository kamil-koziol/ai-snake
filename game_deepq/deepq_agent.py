import numpy as np
import sys
from nn import NeuralNetwork
from game import Snake, MoveDirection

import torch
import torch.nn as nn
import torch.optim as optim


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
    


class DeepQAgent:
    # memory:
    def __init__(self, epochs, epoch_size, board_size, piece_size, hidden_sizes=[28, 64, 64, 4], lr=0.001, batch_size=32) -> None:
        self.epochs = epochs
        self.epoch_size = epoch_size

        # TODO parametrize
        self.gamma = 0.9
        self.epsilon = 1.0

        self.batch_size = batch_size

        self.board_size = board_size
        self.piece_size = piece_size

        self.game = DeepQSnake(board_size, piece_size)
        self.current_state = self.game.get_state()

        state_shape = self.current_state.shape
        # assert state_shape is (1, x)
        assert len(state_shape) == 2
        assert state_shape[0] == 1

        state_size = state_shape[1]
        action_size = 4

        self.agentNetwork = QNetwork(state_size, action_size, hidden_sizes)
        self.optimizer = optim.Adam(self.agentNetwork.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(self, after_epoch_callback = lambda: None):
        for epoch in range(self.epochs):
            for step in range(self.epoch_size):
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

                # train step
                self.train_step(self.current_state, action, reward, next_state, done)

                self.current_state = next_state
                # restart game if needed
                if done:
                    self.game.restart()
                    self.current_state = self.game.get_state()

            # decrease epsilon
            self.epsilon *= 0.9

            after_epoch_callback(epoch)

    def train_step(self, state, action, reward, next_state, done):
        if done:
            q_target_for_action = reward
        else:
            with torch.no_grad():
                future_reward = torch.argmax(self.agentNetwork(state_to_tensor(next_state)))
                q_target_for_action = reward + self.gamma * future_reward
        
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

