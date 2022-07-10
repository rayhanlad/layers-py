import numpy as np
from BaseClass.optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.02):
        self.learning_rate = learning_rate
    
    def update(self, gradients):

        updates = [-self.learning_rate*grad for grad in gradients]

        return updates

    
class Momentum(Optimizer):



    def __init__(self, learning_rate: float = 0.02, rho: float= 0.9):
        self.learning_rate = learning_rate
        self.rho = rho
        self.vx = 0
    
    def update(self, gradients):

        updates= []

        for grad in gradients:
            self.vx = [-self.learning_rate*(self.rho * self.vx + grad) for grad in gradients]

        return self.vx
    
