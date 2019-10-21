"""Module of simple neural net."""

import numpy as np


class Activation:
    """Activation function class."""

    @staticmethod
    def linear(x, deactivate=False):
        """Perform linear activation or deactivation."""
        if not deactivate:
            return x
        else:
            return x

    @staticmethod
    def sigmoid(x, deactivate=False):
        """Perform sigmoid activation or deactivation."""
        if not deactivate:
            return 1 / (1 + np.exp(-x))
        else:
            return x * (1 - x)


class LossFunc:
    """Loss function class."""

    @staticmethod
    def l2_loss(y_true, y_pred, derivative=False):
        """Compute L2 loss or its derivative."""
        if not derivative:
            result = 0.5 * (y_pred - y_true) ** 2
        else:
            result = y_pred - y_true
        return result if result.ndim == 1 else result.mean(axis=0)


class Dense:
    """A Dense layer."""

    def __init__(self, units, input_shape=None, activation='linear',
                 use_bias=True):
        """Initialize layer."""
        self._units = units
        self._input_shape = input_shape
        self._weight = None
        if self._input_shape is not None:
            self.init_weight()
        self._bias = np.random.rand(units) if use_bias else np.zeros(units)
        self._activation = getattr(Activation, activation)
        self._last_input = None
        self._last_output = None
        self._gradient = None

    def init_weight(self):
        """Initialize weight of this layer."""
        self._weight = np.random.rand(self._input_shape, self._units)

    def forward(self, x):
        """Compute wx+b, and activate."""
        if self._input_shape is None:
            self._input_shape = x.shape[-1]
            self.init_weight()
        z = np.dot(x, self._weight) + self._bias
        z = self._activation(z)
        self._last_input, self._last_output = x, z
        return z

    def backprop(self, loss):
        """Compute the gradient of loss w.r.t. weight."""
        dl_da = loss
        da_dz = self._activation(self._last_output, deactivate=True)
        dz_dw = self._last_input
        dl_dz = dl_da * da_dz
        self._gradient = dl_dz * dz_dw[:, None]
        return (dl_dz * self._weight).sum(axis=-1)


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

    def _backprop(self, loss):
        """Back propagation through all layers."""
        for layer in reversed(self._layers):
            loss = layer.backprop(loss)
