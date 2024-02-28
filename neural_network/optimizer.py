import numpy as np

class Optimizer:
    def __init__(self, optimizer_type, learning_rate):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate

    def calc_gradient(self, w, b, cost_func):
        if cost_func == "MSE":
            return np.maximum(0, z)
