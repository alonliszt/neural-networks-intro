import pytest
from .main import Network, Layer, Perceptron


def test_neural_network():
    l1 = Layer([
        Perceptron([0, 2], 1),
        Perceptron([5, 2], 2)
    ])

    l2 = Layer([
        Perceptron([5, -2], 0)
    ])


    n = Network([l1, l2])

    assert n.compute([0, 1]) == [7], "Wrong calculation"
