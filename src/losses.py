from BaseClass.loss import Loss
import numpy as np


class MSE(Loss):

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:

        return np.mean(np.power(y_pred - y_true, 2)) / 2
    
    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:

        return (y_pred - y_true)/y_pred.shape[0]

class BCE(Loss):

    def __init__(self, epsilon: float= 1e-7):
        self.epsilon = epsilon

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        
        y_pred = np.clip(y_pred, self.epsilon, 1- self.epsilon)

        term_0 = (1 - y_true)*np.log(1 - y_pred + self.epsilon)
        term_1 = y_true * np.log(y_pred + self.epsilon)

        return -np.mean(term_0 + term_1)
    
    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:

        numerator = y_pred - y_true
        denominator = y_pred * (1 - y_pred) + self.epsilon
        return numerator/denominator    


