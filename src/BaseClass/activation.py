from .layer import Layer
from abc import abstractmethod, ABC
import numpy as np

class Activation(Layer):

    def __init__(self) -> None:
        super().__init__(trainable=False)

    @abstractmethod
    def activate(self, z: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def derivative(self, z: np.ndarray) -> np.ndarray:
        pass
    
    
    def forward(self, input: np.ndarray) -> np.ndarray:

        self.input = input
        self.output = self.activate(input)

        return self.output
    
    def backward(self, output_error: np.ndarray) -> np.ndarray:

        return self.derivative(self.input) * output_error, None