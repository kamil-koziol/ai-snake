import numpy as np

from game import Snake, MoveDirection
from nn import NeuralNetwork


class GeneticSnake(Snake):
    brain: NeuralNetwork
    fitness: float

    def __init__(self, board_size, piece_size, model: NeuralNetwork):
        super().__init__(board_size, piece_size, True)
        self.brain = model

    def update(self):
        if not self.alive:
            return

        self.predict_move_dir()
        super().update()

    def predict_move_dir(self):
        predictions = self.brain.predict(self.rays)
        self.set_move_dir(MoveDirection(np.argmax(predictions)))

    def calculate_fitness(self) -> None:
        self.fitness = 0
        if self.apples_eaten < 7:
            self.fitness = (self.age * self.age) * (2 ** self.apples_eaten)
        else:
            self.fitness = (self.age * self.age) * (2 ** 7) * (self.apples_eaten - 6)


