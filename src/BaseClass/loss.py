from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):

    @abstractmethod
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass