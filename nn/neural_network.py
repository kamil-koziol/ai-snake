from __future__ import annotations

import random
from typing import List

import numpy as np

from nn import Layer


class NeuralNetwork:
    layers: List[Layer]

    def __init__(self, layers: List[Layer]):
        self.layers = layers.copy()
        self._setup_layers()

    def _setup_layers(self):
        self.layers[0].setup(None)

        previous_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.setup(previous_layer)
            previous_layer = layer

    def predict(self, _input: np.ndarray) -> np.ndarray:
        self._forward(_input)
        return self.layers[-1].neurons

    def _forward(self, _input: np.ndarray):
        # input layer is first
        self.layers[0].neurons = _input.copy()

        for i, layer in enumerate(self.layers[1:], 1):
            layer.calculate(self.layers[i - 1].neurons)

    def copy(self) -> NeuralNetwork:
        neural_network_copy = NeuralNetwork(self.layers)
        neural_network_copy._setup_layers()

        for i in range(len(self.layers)):
            neural_network_copy.layers[i].weights = self.layers[i].weights.copy()
            neural_network_copy.layers[i].biases = self.layers[i].biases.copy()

        return neural_network_copy

    def mutate(self, mutation_rate: float):
        for layer in self.layers[1:]:
            m, n = layer.weights.shape

            for row in range(m):
                for col in range(n):
                    rnd = random.random()
                    if rnd < mutation_rate:
                        layer.weights[row, col] = (random.random() * 2) - 1.0

            m, n = layer.biases.shape

            for row in range(m):
                for col in range(n):
                    rnd = random.random()
                    if rnd < mutation_rate:
                        layer.biases[row, col] = (random.random() * 2) - 1.0
