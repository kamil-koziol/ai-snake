import numpy as np
import sys
from nn import NeuralNetwork
from game import Snake, MoveDirection
from nn.activation_functions import linear, softmax, relu
from nn.layer import InputLayer, Layer

class DeepQAgent:
    agentNetwork: NeuralNetwork
    # memory:
    def __init__(self, epochs, epoch_size, board_size, piece_size) -> None:
        self.epochs = epochs
        self.epoch_size = epoch_size

        # TODO parametrize
        self.gamma = 0.9
        self.epsilon = 1.0

        self.agentNetwork = NeuralNetwork([
            InputLayer(28, linear),
            Layer(64, relu),
            Layer(64, relu),
            Layer(4, softmax)
        ])

        self.board_size = board_size
        self.piece_size = piece_size

        self.game = DeepQSnake(board_size, piece_size)
        self.current_state = self.game.get_state()

    def train(self):
        for epoch in range(self.epochs):
            for step in range(self.epoch_size):
                # generate random number between 0 and 1
                rand = np.random.uniform(0, 1)
                if rand < self.epsilon:
                    # random MoveDirection - explore
                    action = MoveDirection(np.random.randint(0, 3))
                else:
                    # predict MoveDirection - exploit
                    action = self.agentNetwork.predict(self.current_state)
                    action = MoveDirection(np.argmax(action))

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

    def train_step(self, state, action, reward, next_state, done):
        if done:
            q_target_for_action = reward
        else:
            q_target_for_action = reward + self.gamma * np.max(self.agentNetwork.predict(next_state))

        target = self.agentNetwork.predict(state)
        target[0][action.value] = q_target_for_action

        self.agentNetwork.backward(target, state)

    def loss(self, target, prediction):
        return np.sum(np.square(target - prediction))


class DeepQSnake(Snake):
    def __init__(self, board_size, piece_size, hunger_enabled=False):
        super().__init__(board_size, piece_size, hunger_enabled)

    def get_state(self):
        return self.rays

    def is_alive(self):
        return self.alive

    def get_score(self):
        return self.apples_eaten

    def step(self, action):
        score_before = self.get_score()
        self.set_move_dir(action)

        super().update()

        apple_eaten = self.get_score() > score_before

        # check if game is over
        done = not self.is_alive()

        # calculate reward
        reward = (apple_eaten * 10) + (done * -100)

        return self.get_state(), reward, done

