import numpy as np

class Cost:
    """
    This class represents a cost function for a machine learning model.

    Attributes:
        cost_func (str): The cost function to be used. Currently, only 'MSE' (Mean Squared Error) is supported.

    Methods:
        calc_cost(Y_hat, Y, m): Calculate the cost of the prediction based on the given cost function and predicted and actual values.
    """

    def __init__(self, cost_func):
        """
        Initialize the Cost object with the specified cost function.

        Parameters:
            cost_func (str): The cost function to be used. Currently, only 'MSE' (Mean Squared Error) is supported.
        """
        if not isinstance(cost_func, str):
            raise TypeError("cost_func must be a string")
        if cost_func != "MSE":
            raise ValueError("cost_func can currently only be 'MSE'")

        self.cost_func = cost_func

    def calc_cost(self, Y_hat, Y, m):
        """
        Calculate the cost of the prediction based on the given cost function and predicted and actual values.

        Parameters:
            Y_hat (numpy array): The predicted values.
            Y (numpy array): The actual values.
            m (int): The number of training examples.

        Returns:
            float: The calculated cost.
        """
        if self.cost_func == "MSE":
            return 1/m * np.sum(np.square(Y_hat - Y))