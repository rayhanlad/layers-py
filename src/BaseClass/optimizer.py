import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(self, learning_rate:float = 0.02):

        self.learning_rate = learning_rate
    


    @abstractmethod
    def update(self, gradients):
        pass