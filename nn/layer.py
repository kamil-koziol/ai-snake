from __future__ import annotations
from typing import List, Callable
import numpy as np

from nn.activation_functions import linear
class Layer:
    size: int
    activation: Callable
    weights: np.ndarray
    biases: np.ndarray
    neurons: np.ndarray
    delta: np.ndarray

    def __init__(self, size: int, activation: Callable):
        self.size = size
        self.activation = activation

    def setup(self, previous_layer: Layer):
        self.weights = (np.random.random((previous_layer.size, self.size)) * 2) - 1.0
        self.biases = (np.random.random((1, self.size)) * 2) - 1.0

    def calculate(self, _input: np.ndarray):
        self.neurons = (_input @ self.weights) + self.biases
        self.neurons = self.activation(self.neurons)


class InputLayer(Layer):
    def setup(self, previous_layer: Layer):
        self.weights = np.ndarray([])
        self.biases = np.ndarray([])
