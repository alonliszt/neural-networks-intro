"""
This example demonstrates forward propagation in a convolutional neural network. No backward propagation, pure python implementation.
"""

from functools import reduce
from random import random
import sys


class Perceptron(object):
    def __init__(self, weights, bias):
        self._weights = weights
        self._bias = bias

    def compute(self, inputs):
        assert len(self._weights) == len(inputs), f"Received an input layer of size {len(inputs)} but initialized to get {len(self._weights)} sized input"
        return sum(a * b for a, b in zip(self._weights, inputs)) + self._bias

    def __repr__(self):
        return f"(Perceptron w={self._weights} b={self._bias})"

    @classmethod
    def random(cls, previous_layer_size):
        return cls([random() for i in range(previous_layer_size)], random())


class Layer(object):
    def __init__(self, perceptrons):
        self._perceptrons = perceptrons

    def compute(self, inputs):
        return [perceptron.compute(inputs) for perceptron in self._perceptrons]

    def __repr__(self):
        return f"({str(self._perceptrons)}"

    @classmethod
    def random(cls, previous_layer_size, layer_size):
        return cls([Perceptron.random(previous_layer_size) for i in range(layer_size)])


class Network(object):
    def __init__(self, layers):
        """
        Input layer is given as the input
        """
        self._layers = layers

    def compute(self, input_values):
        return reduce(lambda input, layer: layer.compute(input), self._layers, input_values)

    @classmethod
    def random(cls, *layer_sizes):
        return cls([Layer.random(prev_size, size) for prev_size, size in zip(layer_sizes, layer_sizes[1:])])


def main():
    l1 = Layer([
        Perceptron([0, 2], 1),
        Perceptron([5, 2], 2)
    ])

    l2 = Layer([
        Perceptron([5, -2], 0)
    ])


    n = Network([l1, l2])
    # n = Network.random(2, 3, 1)

    print(n.compute([0, 1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
