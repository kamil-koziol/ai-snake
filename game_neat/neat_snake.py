import random
from typing import List

import neat
import numpy as np

from game import Snake, MoveDirection
from game_genetic import GeneticNeuralNetwork
import pygame as pg


class NEATSnake(Snake):
    brain: neat.nn.FeedForwardNetwork
    fitness: float
    history: List[pg.Vector2]
    died_from_hunger = False
    id: int

    def __init__(self, board_size, piece_size, brain: neat.nn.FeedForwardNetwork, id: int):
        super().__init__(board_size, piece_size, True)
        self.brain = brain
        self.DEFAULT_HUNGER = (board_size * board_size)
        self.history = []
        self.died_from_hunger = False
        self.id = id

    def update(self, verbose=0, relative_dir = False):
        if not self.alive:
            return

        self.predict_move_dir()
        super().update(verbose, relative_dir)

        self.history.append(self.pos.copy())

        counter = 0
        for pos in self.history:
            if pos.x == self.pos.x and pos.y == self.pos.y:
                counter += 1

        if counter >= 5:
            self.die()
            self.died_from_hunger = True

    def on_apple_eat(self):
        super().on_apple_eat()
        self.history.clear()

    def predict_move_dir(self):
        self.update_rays(False)
        inputs = self.rays.flatten().tolist()
        output = np.array(self.brain.activate(inputs))
        new_move_dir = MoveDirection(np.argmax(output))
        self.set_move_dir(new_move_dir)

    def calculate_fitness(self) -> None:
        self.fitness = 0

        self.fitness += self.apples_eaten * 500

        if self.crashed_to_wall or self.crashed_to_self:
            self.fitness /= 2

        if self.died_from_hunger:
            self.fitness -= 250

        # if self.apples_eaten < 10:
        #     self.fitness = (self.age * self.age) * (2 ** self.apples_eaten)
        # else:
        #     self.fitness = (self.age * self.age) * (2 ** 10) * (self.apples_eaten - 9)
        #
        # if self.crashed_to_wall or self.crashed_to_self:
        #     self.fitness /= 5
        #
        # if self.died_from_hunger:
        #     self.fitness /= 2
