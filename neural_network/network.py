import numpy as np
from neural_network.cost import Cost
from neural_network.optimizer import Optimizer
from neural_network.layers import Dense
from typing import Dict
from numpy import ndarray

class Network:
    def __init__(self, layers, optimizer, cost):
        if not isinstance(layers, list) or not all(isinstance(layer, Dense) for layer in layers):
            raise TypeError("layers must be a list of Layer instances")

        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be of type optimizer")

        if not isinstance(cost, Cost):
            raise TypeError("cost must be of type Cost")

        self.layers = layers
        self.optimizer = optimizer
        self.cost = cost

    def forward_prop(self, X, Y):
        assert X.shape[0] == self.layers[0].input_shape(), f"The number of columns in X is {X.shape[0]} which is not equal to the input shape of layer 1 which is {self.layers[0].input_shape()}"

        assert X.shape[1] == Y.shape[0], f"The shape of X and Y need to match."

        cache: Dict[str, ndarray] = {}
        cache['X'] = X

        for i, layer in enumerate(self.layers):
            Y_hat, N = layer.call(X)
            X = Y_hat

            cache[f'Y_hat_{i}'] = Y_hat
            cache[f'N_{i}'] = N

        loss = self.cost.calc_cost(Y_hat, Y)

        cache['loss'] = loss

        return cache

    def backwards_prop(self, Y, cache):
        dLdY_hat = -2 * (Y - cache['Y_hat'])

        dPdN = np.ones_like(cache['N'])

        dPdB = np.ones_like(self.layers[0].get_bias())

        dLdN = dLdY_hat * dPdN

        dNdW = cache['X']

        dLdW = (1 / cache['X'].shape[1]) * np.dot(dNdW, dLdN.T)

        dLdB = (1 / cache['X'].shape[1]) * (dLdY_hat * dPdB).sum(axis=1)

        loss_gradients: Dict[str, ndarray] = {}
        loss_gradients['dW'] = dLdW.T
        loss_gradients['dB'] = dLdB

        return loss_gradients

    def train(self, X, Y, epochs, learning_rate=0.1):
        for epoch in range(epochs):
            cache = self.forward_prop(X, Y)
            loss_gradients = self.backwards_prop(Y, cache)

            self.layers[0].update_parameters(loss_gradients, learning_rate)