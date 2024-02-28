import numpy as np

class Loss:
    def __init__(self, loss_func):
        if not isinstance(loss_func, str):
            raise TypeError("loss_func must be a string")
        if loss_func != "MSE":
            raise ValueError("loss_func can currently only be 'MSE'")

        self.loss_func = loss_func

    def calc_cost(self, Y_hat, Y):
        if self.loss_func == "MSE":
            m, _ = Y_hat.shape
            return 1/(2*m) * np.sum(np.square(Y_hat - Y))
