from functools import reduce
import numpy as np


class Layer(object):
    """
    Inputs are represented as a 2d matrix of shape (N, 1) where N is number of inputs to this layer.
    Weights are represented as a 2d materix of shape (N, M) where M is the number of perceptrons in this layer
    Biases are represented as a 2d materix of shape (M, 1).
    """
    def __init__(self, weights, biases):
        self._weights = weights
        self._biases = biases

    def compute(self, inputs):
        return np.dot(self._weights, inputs) + self._biases


class Network(object):
    def __init__(self, layers):
        self._layers = layers

    def compute(self, initial_inputs):
        return reduce(lambda inputs, layer: layer.compute(inputs), self._layers, initial_inputs)


def main():
    l1 = Layer(
        weights=np.array([
            [0, 2],
            [5, 2]
        ]),
        biases=np.array([
            [1],
            [2]
        ])
    )

    l2 = Layer(
        weights=np.array([
            [5, -2],
        ]),
        biases=np.array([
            [0]
        ])
    )

    n = Network([l1, l2])
    print(n.compute(np.array([
        [0],
        [1]
    ])))


if __name__ == "__main__":
    main()
