import numpy as np
from neural_network.activation import Activation
from neural_network.optimizer import Optimizer

class Dense:
    """
    A Dense layer class representing a fully connected layer in a neural network.

    Attributes:
        input_size (int): The number of input nodes to the layer.
        output_size (int): The number of output nodes from the layer.
        weights (np.ndarray): The weight matrix of the layer, initialized randomly.
        bias (np.ndarray): The bias vector of the layer, initialized randomly.
        activation (Activation): The activation function to be used in the layer.

    Methods:
        call(X): Computes the output of the dense layer for a given input.
        input_shape(): Returns the input size of the layer.
        get_bias(): Returns the bias vector of the layer.
        get_weights(): Returns the weight matrix of the layer.
        update_parameters(dW, db, learning_rate): Updates the weights and biases using the gradients and learning rate.
        activation_derivative(Z): Returns the derivative of the activation function at Z.
    """

    def __init__(self, input_size, output_size, activation):
        """
        Initializes the Dense layer with the specified input size, output size, and activation function.

        Args:
            input_size (int): The number of input nodes to the layer.
            output_size (int): The number of output nodes from the layer.
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
            X (np.ndarray): The input matrix to the layer.

        Returns:
            A (np.ndarray): The output of the layer after applying the weights, biases, and activation function.
            Z (np.ndarray): The linear part of the layer's output before applying the activation function.
        """
        assert X.shape[0] == self.weights.shape[1], f"The number of rows in X is {X.shape[0]} which is not equal to the number of weight columns which is {self.weights.shape[1]}."

        Z = np.dot(self.weights, X)
        A = self.activation.activate(Z + self.bias)
        return A, Z

    def input_shape(self):
        """
        Returns the input size of the layer.

        Returns:
            int: The input size of the layer.
        """
        return self.input_size

    def get_bias(self):
        """
        Returns the bias vector of the layer.

        Returns:
            np.ndarray: The bias vector of the layer.
        """
        return self.bias

    def get_weights(self):
        """
        Returns the weight matrix of the layer.

        Returns:
            np.ndarray: The weight matrix of the layer.
        """
        return self.weights

    def update_parameters(self, dW, db, learning_rate):
        """
        Updates the weights and biases using the gradients and learning rate.

        Args:
            dW (np.ndarray): The gradient of the weights.
            db (np.ndarray): The gradient of the biases.
            learning_rate (float): The learning rate.
        """
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

    def activation_derivative(self, Z):
        """
        Returns the derivative of the activation function at Z.

        Args:
            Z (np.ndarray): The linear part of the layer's output before applying the activation function.

        Returns:
            np.ndarray: The derivative of the activation function at Z.
        """
        return self.activation.activation_derivative(Z)