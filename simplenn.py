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
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
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

    def update_weight(self, lr=0.01):
        """Update the weight base on computed gradient."""
        self._weight -= lr * self._gradient

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
        dl_dz = dl_da * da_dz  # Same shape as output
        self._gradient = dl_dz * dz_dw[:, None]  # Same shape as weight
        return (dl_dz * self._weight).sum(axis=-1)  # Same shape as input


class SimpleNN:
    """Model of a simple neural network."""

    def __init__(self, loss_func='l2_loss'):
        """Initialize network model."""
        self._layers = []
        self._loss_func = getattr(LossFunc, loss_func)

    def add(self, layer):
        """Add a layer to the model."""
        self._layers.append(layer)

    def _forward(self, x):
        """Feed forward through all layers."""
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def _backprop(self, d_loss):
        """Back propagation through all layers."""
        for layer in reversed(self._layers):
            d_loss = layer.backprop(d_loss)

    def _update(self, lr):
        """Update weight of all layers."""
        for layer in self._layers:
            layer.update_weight(lr)

    def fit(self, x, y, lr=0.01, batch_size=None, epochs=1,
            early_stopping_loss=0, verbose=None):
        """Train the model."""
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)
        loss = []
        for i in range(epochs):
            loss_epoch = 0
            for x_j, y_j in zip(x, y):
                pred = self._forward(x_j)
                loss_epoch += self._loss_func(y_j, pred)
                d_loss = self._loss_func(y_j, pred, derivative=True)
                self._backprop(d_loss)
                self._update(lr)
            loss.append(loss_epoch)
            if verbose is not None and i % verbose == 0:
                print(f'Epoch {i}, loss = {loss_epoch}')
            if loss_epoch <= early_stopping_loss:
                break
        return np.array(loss)

    def predict(self, x):
        """Predict input data with the model."""
        return self._forward(x)
