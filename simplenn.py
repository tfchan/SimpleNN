"""Module of simple neural net."""


import numpy as np


class Layer:
    """A NN layer."""

    def __init__(self, units, input_shape, activation=None):
        """Initialize layer."""
        self._weight = np.random.rand(input_shape, units)
        self._bias = np.random.rand(units)
        self.activation = activation
