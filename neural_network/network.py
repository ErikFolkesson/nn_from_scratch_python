import numpy as np
from neural_network.cost import Cost
from neural_network.optimizer import Optimizer
from neural_network.layers import Dense
from typing import Dict
from numpy import ndarray
from sklearn.model_selection import train_test_split

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

    def __init__(self, layers, cost, optimizer=None):
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

        if optimizer is not None and not isinstance(optimizer, Optimizer):
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

    def train(self, X, Y, epochs, learning_rate=0.01, validation_split=None):
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
            validation_split : float, optional
                Fraction of the data to be used as validation data (default is 0.2)

        Returns
        -------
        training_progress : dict
            Dictionary containing the training progress, including the cost at each epoch and validation cost
        """

        # Split the data into training and validation sets
        X = X.T
        Y = Y.T

        if validation_split == None: # If no validation split is provided, use the entire dataset for training
            X_train, X_val, Y_train, Y_val = X, X, Y, Y
        else:
            # Shuffle the indices of the data
            indices = np.random.permutation(X.shape[1])

            # Split the indices into training and validation sets
            split_idx = int((1 - validation_split) * X.shape[1])
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]

            # Create the training and validation sets
            X_train, X_val = X[:, train_idx], X[:, val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            print(f"X_train shape: {X_train.shape}")
            print(f"First 5 elements of Y_train: {Y_train[:5]}")
            print(f"X_val shape: {X_val.shape}")
            print(f"First 5 elements of Y_val: {Y_val[:5]}")


        training_progress = {
            'cost': [],
            'val_cost': [],  # Validation cost
            # Add more metrics here in the future
        }

        for epoch in range(epochs):
            # Train on the training data
            cache = self.forward_prop(X_train, Y_train)
            loss_gradients = self.backwards_prop(Y_train, cache)
            self._update_parameters(loss_gradients, learning_rate)

            # Save the cost at each epoch
            training_progress['cost'].append(cache['loss'])

            # Calculate and save the validation cost
            val_cache = self.forward_prop(X_val, Y_val)
            training_progress['val_cost'].append(val_cache['loss'])

            # Save other metrics here in the future

        return training_progress

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