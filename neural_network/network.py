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
        assert X.shape[0] == self.layers[0].input_shape(), f"The number of columns in A_prev is {A_prev.shape[0]} which is not equal to the input shape of layer 1 which is {self.layers[0].input_shape()}"

        cache: Dict[str, ndarray] = {}
        cache['X'] = X

        for i, layer in enumerate(self.layers):
            A, Z = layer.call(X)
            X = A

            cache[f'A_{i}'] = A
            cache[f'Z_{i}'] = Z
            cache[f'W_{i}'] = layer.get_weights()
            cache[f'b_{i}'] = layer.get_bias()

        loss = self.cost.calc_cost(A, Y, X.shape[1])

        cache['loss'] = loss

        return cache

    def backwards_prop(self, Y, cache):
        loss_gradients: Dict[str, ndarray] = {}

        L = len(self.layers) - 1

        dA = -2 * (Y - cache[f'A_{L}'])

        for l in reversed(range(L + 1)):
            dZ = dA * self.layers[l].activation_derivative(cache[f'Z_{l}'])
            dW = (1 / cache['X'].shape[1]) * np.dot(dZ, cache[f'A_{l - 1}'].T if l > 0 else cache['X'].T)
            db = (1 / cache['X'].shape[1]) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(cache[f'W_{l}'].T, dZ) if l > 0 else None

            loss_gradients[f'dW_{l}'] = dW
            loss_gradients[f'db_{l}'] = db

        return loss_gradients

    def train(self, X, Y, epochs, learning_rate=0.01, debug=False):
        for epoch in range(epochs):
            cache = self.forward_prop(X, Y)
            loss_gradients = self.backwards_prop(Y, cache)

            for i, layer in enumerate(self.layers):
                layer.update_parameters(loss_gradients[f'dW_{i}'], loss_gradients[f'db_{i}'], learning_rate)

            if debug and epoch % 100 == 0:  # Print every 1 epochs
                print(f'Epoch {epoch}:')
                print('Cache:')
                for key, value in cache.items():
                    print(f'{key}: {value}')

                print('Loss Gradients:')
                for key, value in loss_gradients.items():
                    print(f'{key}: {value}')