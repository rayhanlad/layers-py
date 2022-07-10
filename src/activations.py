from BaseClass.activation import Activation
import numpy as np


class Tanh(Activation):

    def activate(self, z: np.ndarray) -> np.ndarray:

        return np.tanh(z)
    
    def derivative(self, z: np.ndarray) -> np.ndarray:

        return 1 - np.tanh(z)**2

class ReLU(Activation):


    def activate(self, z: np.ndarray) -> np.ndarray:

        return np.maximum(0, z)

    def derivative(self, z: np.ndarray) -> np.ndarray:

        return np.where(z>=0, 1, 0)

class Sigmoid(Activation):

    def activate(self, z: np.ndarray) -> np.ndarray:

        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z: np.ndarray) -> np.ndarray:

        s = self.activate(z)

        return s * (1 - s)