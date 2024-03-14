import numpy as np

class Activation:
    """
    A class to represent an activation function for use in neural networks.

    Currently supports the ReLU activation function.

    Attributes:
        activation_type (str): The type of the activation function. Currently,
                               only 'relu' and 'none' is supported.
    """

    activation_types = {"relu", "none"}

    def __init__(self, activation_type):
        """
        Initializes the Activation class with the specified activation type.

        Validates the input to ensure it is a supported activation type.

        Args:
            activation_type (str): The type of the activation function to use.

        Raises:
            TypeError: If the activation_type is not a string.
            ValueError: If the activation_type is not supported.
        """
        if not isinstance(activation_type, str):
            raise TypeError("activation_type must be a string")
        # if activation_type != "relu" or "none":
        #     raise ValueError("activation_type must be 'relu' or 'none'")

        self.activation_type = activation_type

    def activate(self, Z):
        """
        Applies the activation function to the input array.

        Args:
            Z (np.ndarray): The input array to the activation function.

        Returns:
            np.ndarray: The output of the activation function.
        """
        if self.activation_type == "relu":
            return np.maximum(0, Z)
        elif self.activation_type == "none":
            return Z

    def activation_derivative(self, Z):
        """
        Computes the gradient of the activation function.

        Args:
            Z (np.ndarray): The input array to the activation function.

        Returns:
            np.ndarray: The gradient of the activation function.
        """
        if self.activation_type == "relu":
            return Z > 0
        elif self.activation_type == "none":
            return np.ones_like(Z)
