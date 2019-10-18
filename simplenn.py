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


class Dense:
    """A Dense layer."""

    def __init__(self, units, input_shape=None, activation='linear'):
        """Initialize layer."""
        self._units = units
        self._input_shape = input_shape
        self._weight = None
        if self._input_shape is not None:
            self.init_weight()
        self._bias = np.random.rand(units)
        self._activation = getattr(Activation, activation)

    def init_weight(self):
        """Initialize weight of this layer."""
        self._weight = np.random.rand(self._units, self._input_shape)

    def forward(self, x):
        """Compute wx+b, and activate."""
        if self._input_shape is None:
            self._input_shape = x.shape[-1]
            self.init_weight()
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

    def _forward(self, x):
        """Feed forward through all layers."""
        for layer in self._layers:
            x = layer.forward(x)
        return x
