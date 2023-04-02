from typing import List

import numpy as np

from game import Snake, MoveDirection
from game_genetic import GeneticNeuralNetwork
import pygame as pg


class GeneticSnake(Snake):
    brain: GeneticNeuralNetwork
    fitness: float
    history: List[pg.Vector2]

    def __init__(self, board_size, piece_size, model: GeneticNeuralNetwork):
        super().__init__(board_size, piece_size, True)
        self.brain = model
        self.DEFAULT_HUNGER = (board_size * board_size)
        self.history = []

    def update(self):
        if not self.alive:
            return

        self.predict_move_dir()
        super().update()

        self.history.append(self.pos.copy())

        counter = 0
        for pos in self.history:
            if pos.x == self.pos.x and pos.y == self.pos.y:
                counter += 1

        if counter >= 5:
            self.die()
            self.age //= 2

    def on_apple_eat(self):
        super().on_apple_eat()
        self.history.clear()

    def predict_move_dir(self):
        predictions = self.brain.predict(self.rays)
        self.set_move_dir(MoveDirection(np.argmax(predictions)))

    def calculate_fitness(self) -> None:
        self.fitness = 0
        if self.apples_eaten < 10:
            self.fitness = (self.age * self.age) * (2 ** self.apples_eaten)
        else:
            self.fitness = (self.age * self.age) * (2 ** 10) * (self.apples_eaten - 9)
