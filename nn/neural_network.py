from __future__ import annotations

import random
from typing import List

import numpy as np

from nn import Layer
from nn.activation_functions import relu, softmax, linear
from nn.layer import InputLayer


class NeuralNetwork:
    layers: List[Layer]

    def __init__(self, layers: List[Layer]):
        self.layers = layers
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

    def backward(self, target: np.ndarray, input_data: np.ndarray):
        output_layer = self.layers[-1]
        output_delta = target - output_layer.neurons
        output_layer.delta = output_delta * output_layer.activation(output_layer.neurons)

        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            current_delta = (next_layer.delta @ next_layer.weights.T) * current_layer.activation(current_layer.neurons)
            current_layer.delta = current_delta

        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]
            previous_layer = self.layers[i - 1]

            current_layer.weights += np.dot(previous_layer.neurons.T, current_layer.delta)
            current_layer.biases += np.sum(current_layer.delta, axis=0)

    def copy(self):
        layers = []
        layers.append(InputLayer(self.layers[0].size, linear))
        for layer in self.layers[1:]:
            layers.append(Layer(layer.size, layer.activation))

        neural_network_copy = type(self)(layers)

        for i in range(len(self.layers)):
            np.copyto(neural_network_copy.layers[i].weights, self.layers[i].weights)
            np.copyto(neural_network_copy.layers[i].biases, self.layers[i].biases)

        return neural_network_copy

    def save(self, filename):
        with open(filename, 'wb') as f:
            for layer in self.layers:
                np.save(f, layer.weights)
                np.save(f, layer.biases)

    def load(self, filename):
        with open(filename, 'rb') as f:
            for layer in self.layers:
                layer.weights = np.load(f)
                layer.biases = np.load(f)


if __name__ == "__main__":
    model = NeuralNetwork([
        InputLayer(28, linear),
        Layer(16, relu),
        Layer(8, relu),
        Layer(4, softmax)
    ])

    print(model.layers[1].weights)
    x = model.copy()
    print(model.layers[1].weights[0, 0])
    print(x.layers[1].weights[0, 0])

    model.layers[1].weights[0, 0] = 2.0
    print(model.layers[1].weights[0, 0])
    print(x.layers[1].weights[0, 0])

    print(model.layers[1].weights[0])
    # model.save("test.npy")
    mcpy = NeuralNetwork([
        InputLayer(28, linear),
        Layer(16, relu),
        Layer(4, softmax)
    ])
    # mcpy.load("test.npy")
    print(model.layers[1].weights[0])

    for i in range(100):
        print(random.gauss(0, 1))
