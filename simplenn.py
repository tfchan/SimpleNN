"""Module of simple neural net."""


import numpy as np


class Activation:
    """Activation function class."""

    @staticmethod
    def linear(x):
        """Perform linear activation."""
        return x

    @staticmethod
    def sigmoid(x):
        """Perform sigmoid activation."""
        return 1 / (1 + np.exp(-x))


class Layer:
    """A NN layer."""

    def __init__(self, units, input_shape, activation='linear'):
        """Initialize layer."""
        self._weight = np.random.rand(units, input_shape)
        self._bias = np.random.rand(units)
        self._activation = getattr(Activation, activation)

    def forward(self, x):
        """Compute wx+b, and activate."""
        z = np.dot(self._weight, x) + self._bias
        z = self._activation(z)
        return z


class SimpleNN:
    """Model of a simple neural network."""

    def __init__(self):
        """Initialize network model."""
        self._layers = []

    def add(self, layer):
        """Add a layer to the model."""
        self._layers.append(layer)
