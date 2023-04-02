from __future__ import annotations

from typing import List
import random

from nn import NeuralNetwork, Layer


class GeneticNeuralNetwork(NeuralNetwork):
    def __init__(self, layers: List[Layer]):
        super().__init__(layers)

    def crossover(self, other: GeneticNeuralNetwork):
        for i in range(1, len(self.layers)):
            m, n = self.layers[i].weights.shape

            randr = random.randint(0, m - 1)
            randc = random.randint(0, n - 1)

            for row in range(m):
                for col in range(n):
                    if row < randr or (row == randr and col <= randc):
                        self.layers[i].weights[row, col] = other.layers[i].weights[row, col]

            m, n = self.layers[i].biases.shape

            randr = random.randint(0, m - 1)
            randc = random.randint(0, n - 1)

            for row in range(m):
                for col in range(n):
                    if row < randr or (row == randr and col <= randc):
                        self.layers[i].biases[row, col] = other.layers[i].biases[row, col]

    def mutate(self, mutation_rate: float, mutation_power: float):
        for i in range(1, len(self.layers)):
            m, n = self.layers[i].weights.shape

            for row in range(m):
                for col in range(n):
                    rnd = random.random()
                    if rnd < mutation_rate:
                        self.layers[i].weights[row, col] += random.gauss(0, 1)  * mutation_power

                        if self.layers[i].weights[row, col] < -30.0:
                            self.layers[i].weights[row, col] = -30.0

                        if self.layers[i].weights[row, col] > 30.0:
                            self.layers[i].weights[row, col] = 30.0

            m, n = self.layers[i].biases.shape

            for row in range(m):
                for col in range(n):
                    rnd = random.random()
                    if rnd < mutation_rate:
                        self.layers[i].biases[row, col] += random.gauss(0, 1) * mutation_power

                        if self.layers[i].biases[row, col] < -30.0:
                            self.layers[i].biases[row, col] = -30.0

                        if self.layers[i].biases[row, col] > 30.0:
                            self.layers[i].biases[row, col] = 30.0