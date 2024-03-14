import numpy as np
from neural_network.cost import Cost
from neural_network.optimizer import Optimizer
from neural_network.layers import Dense
from typing import Dict
from numpy import ndarray

class Network:
    """
    A class used to represent a Neural Network

    ...

    Attributes
    ----------
    layers : list
        a list of Layer instances representing the layers of the network
    optimizer : Optimizer
        an instance of Optimizer to optimize the network
    cost : Cost
        an instance of Cost to calculate the cost of the network

    Methods
    -------
    forward_prop(X, Y):
        Performs forward propagation through the network
    backwards_prop(Y, cache):
        Performs backward propagation through the network
    train(X, Y, epochs, learning_rate=0.01, debug=False):
        Trains the network
    """

    def __init__(self, layers, optimizer, cost):
        """
        Constructs all the necessary attributes for the Network object.

        Parameters
        ----------
            layers : list
                a list of Layer instances representing the layers of the network
            optimizer : Optimizer
                an instance of Optimizer to optimize the network
            cost : Cost
                an instance of Cost to calculate the cost of the network
        """
        self._validate_input(layers, optimizer, cost)
        self.layers = layers
        self.optimizer = optimizer
        self.cost = cost

    def _validate_input(self, layers, optimizer, cost):
        """Validates the input parameters for the Network object."""

        if not isinstance(layers, list) or not all(isinstance(layer, Dense) for layer in layers):
            raise TypeError("layers must be a list of Layer instances")

        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be of type optimizer")

        if not isinstance(cost, Cost):
            raise TypeError("cost must be of type Cost")

    def forward_prop(self, X, Y):
        """
        Performs forward propagation through the network.

        Parameters
        ----------
            X : ndarray
                Input data
            Y : ndarray
                Target data

        Returns
        -------
        cache : dict
            Dictionary containing the intermediate values of the forward propagation
        """

        self._validate_input_shape(X)
        cache = self._initialize_cache(X)

        for i, layer in enumerate(self.layers):
            A, Z = layer.call(X)
            X = A
            self._update_cache(cache, A, Z, layer, i)

        loss = self.cost.calc_cost(A, Y, X.shape[1])
        cache['loss'] = loss

        return cache

    def _validate_input_shape(self, X):
        """Validates the shape of the input data X."""

        assert X.shape[0] == self.layers[0].input_shape(), f"The number of columns in A_prev is {A_prev.shape[0]} which is not equal to the input shape of layer 1 which is {self.layers[0].input_shape()}"

    def _initialize_cache(self, X):
        """Initializes the cache dictionary."""

        cache: Dict[str, ndarray] = {}
        cache['X'] = X
        return cache

    def _update_cache(self, cache, A, Z, layer, i):
        """Updates the cache dictionary with the intermediate values of the forward propagation."""

        cache[f'A_{i}'] = A
        cache[f'Z_{i}'] = Z
        cache[f'W_{i}'] = layer.get_weights()
        cache[f'b_{i}'] = layer.get_bias()

    def backwards_prop(self, Y, cache):
        """
        Performs backward propagation through the network.

        Parameters
        ----------
            Y : ndarray
                Target data
            cache : dict
                Dictionary containing the intermediate values of the forward propagation

        Returns
        -------
        loss_gradients : dict
            Dictionary containing the gradients of the loss with respect to the parameters
        """

        loss_gradients: Dict[str, ndarray] = {}
        L = len(self.layers) - 1
        dA = -2 * (Y - cache[f'A_{L}'])

        for l in reversed(range(L + 1)):
            dZ, dW, db, dA = self._calculate_gradients(dA, cache, l)
            loss_gradients[f'dW_{l}'] = dW
            loss_gradients[f'db_{l}'] = db

        return loss_gradients

    def _calculate_gradients(self, dA, cache, l):
        """Calculates the gradients of the loss with respect to the parameters."""

        dZ = dA * self.layers[l].activation_derivative(cache[f'Z_{l}'])
        dW = (1 / cache['X'].shape[1]) * np.dot(dZ, cache[f'A_{l - 1}'].T if l > 0 else cache['X'].T)
        db = (1 / cache['X'].shape[1]) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(cache[f'W_{l}'].T, dZ) if l > 0 else None
        return dZ, dW, db, dA

    def train(self, X, Y, epochs, learning_rate=0.01, debug=False):
        """
        Trains the network.

        Parameters
        ----------
            X : ndarray
                Input data
            Y : ndarray
                Target data
            epochs : int
                Number of epochs to train the network
            learning_rate : float, optional
                Learning rate for the optimizer (default is 0.01)
            debug : bool, optional
                If True, print debug information every 100 epochs (default is False)
        """

        for epoch in range(epochs):
            cache = self.forward_prop(X, Y)
            loss_gradients = self.backwards_prop(Y, cache)
            self._update_parameters(loss_gradients, learning_rate)

            if debug and epoch % 100 == 0:  # Print every 1 epochs
                self._print_debug_info(epoch, cache, loss_gradients)

    def _update_parameters(self, loss_gradients, learning_rate):
        """Updates the parameters of the network using the gradients and the learning rate."""

        for i, layer in enumerate(self.layers):
            layer.update_parameters(loss_gradients[f'dW_{i}'], loss_gradients[f'db_{i}'], learning_rate)

    def _print_debug_info(self, epoch, cache, loss_gradients):
        """Prints debug information."""

        print(f'Epoch {epoch}:')
        print('Cache:')
        for key, value in cache.items():
            print(f'{key}: {value}')

        print('Loss Gradients:')
        for key, value in loss_gradients.items():
            print(f'{key}: {value}')