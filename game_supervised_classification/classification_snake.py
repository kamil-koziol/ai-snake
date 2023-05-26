from typing import List

import numpy as np

from game import Snake, MoveDirection
from game_genetic import GeneticNeuralNetwork
import pygame as pg

from game_supervised_classification.supervised_cassification_Network import SupervisedNeuralNetwork


class SupervisedClassificationSnake(Snake):
    brain: GeneticNeuralNetwork

    def __init__(self, board_size, piece_size, model: SupervisedNeuralNetwork):
        super().__init__(board_size, piece_size, True)
        self.brain = model
        self.DEFAULT_HUNGER = (board_size * board_size)
        self.history = []

    def update(self, verbose=0):
        if not self.alive:
            return

        self.predict_move_dir()

        super().update(verbose)

    def predict_move_dir(self):
        self.set_move_dir(MoveDirection(self.brain.predict(self.rays)))

