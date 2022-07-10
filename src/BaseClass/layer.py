from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):

    def __init__(self, input_shape = None, trainable: bool = True) -> None:
        self.input = None
        self.output = None
        self.trainable = trainable

        self.input_shape = input_shape
    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_error: np.ndarray) -> np.ndarray:
        pass


            