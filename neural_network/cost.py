import numpy as np

class Cost:
    def __init__(self, cost_func):
        if not isinstance(cost_func, str):
            raise TypeError("loss_func must be a string")
        if cost_func != "MSE":
            raise ValueError("loss_func can currently only be 'MSE'")

        self.cost_func = cost_func

    def calc_cost(self, Y_hat, Y):
        """
        Calculate the cost of the prediction based on the given cost function and predicted and actual values.

        Parameters:
            Y_hat (numpy array): The predicted values.
            Y (numpy array): The actual values.

        Returns:
            float: The calculated cost.
        """
        if self.cost_func == "MSE":
            m, _ = Y_hat.shape
            return 1/(2*m) * np.sum(np.square(Y_hat - Y))
