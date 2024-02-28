import numpy as np
from neural_network.activation import Activation

class Dense:
    """
    A dense layer class representing a fully connected neural network layer.

    Attributes:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        weights (np.ndarray): The weight matrix of the layer.
        bias (np.ndarray): The bias vector of the layer.
        activation (Activation): The activation function for the layer.

    """

    def __init__(self, input_size, output_size, activation):
        """
        Initializes the Dense layer with random weights and biases.

        Args:
            input_size (int): The size of the input features.
            output_size (int): The size of the output features.
            activation (str): The name of the activation function to use.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.activation = Activation(activation)

    def call(self, X):
        """
        Computes the output of the dense layer for a given input.

        Args:
            x (np.ndarray): The input matrix to the layer.

        Returns:
            np.ndarray: The output of the layer after applying the weights,
            biases, and activation function.
        """
        Z = np.dot(self.weights, X) + self.bias
        return self.activation.activate(Z)