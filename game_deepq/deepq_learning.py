from game import Snake, MoveDirection
import numpy as np
from game_genetic import NeuralNetwork


class DeepQAgent:
    agentNetwork: NeuralNetwork
    #memory: 
    def __init__(self, epochs, epoch_size) -> None:
        self.epochs = epochs
        self.epoch_size = epoch_size

        self.epsilon = 1
        
        self.game = DeepQSnake()
        self.current_state = self.game.get_state()

        pass

    def train(self):
        for epoch in range(self.epochs):
            for step in range(self.epoch_size):
                # generate random number between 0 and 1

                rand = np.random.uniform(0, 1)
                if(rand < self.epsilon):
                    # random MoveDirection - explore
                    action = np.random.randint(0, 3)
                else:
                    # predict MoveDirection - exploit
                    action = self.agentNetwork.predict(self.current_state)
                    action = np.argmax(action)
 
                next_state, reward, done = self.game.step(action)

                # train step
                self.train_step(self.current_state, action, reward, next_state, done)



                self.current_state = next_state
                # restart game if needed
                if done:
                    self.game.restart()
                    self.current_state = self.game.get_state()

    def train_step(self, state, action, reward, next_state, done):
        # calculate target
        if done:
            q_target_for_action = reward
        else:
            q_target_for_action = reward + self.gamma * np.max(self.agentNetwork.predict(next_state))
        
        prediction = self.agentNetwork.predict(state)
        target = prediction
        target[action] = q_target_for_action

        # train network
        loss = loss(target, prediction)
        self.agentNetwork.backward(prediction, target, loss)


        return


    def loss (self, target, prediction):
        return np.sum(np.square(target - prediction))

class DeepQSnake(Snake):


    def get_state(self):
        return self.rays

    def step(self, action):
        # set move_dir
        self.set_move_dir(action)
        
        super().update()
        # After

        # check if game is over
        done = False

        # calculate reward
        reward = 0
        return self.get_state(), reward, done