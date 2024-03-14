import numpy as np
from neural_network.activation import Activation
from neural_network.optimizer import Optimizer

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
        assert X.shape[0] == self.weights.shape[1], f"The number of rows in X is {X.shape[0]} which is not equal to the number of weight columns which is {self.weights.shape[1]}."

        Z = np.dot(self.weights, X)
        A = self.activation.activate(Z + self.bias)
        return A, Z

    def input_shape(self):
        return self.input_size

    def get_bias(self):
        return self.bias

    def get_weights(self):
        return self.weights

    def update_parameters(self, dW, db, learning_rate):
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

    def activation_derivative(self, Z):
        return self.activation.activation_derivative(Z)