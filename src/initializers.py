from BaseClass.initializer import Initializer
import numpy as np

class RandomNormal(Initializer):

    def __init__(self, mean: float=0.0, std: float = 1.0) -> None:

        self.mean = mean
        self.std = std
    
    def __call__(self, shape):

        return np.random.normal(loc=self.mean, scale=self.std, size=shape)
    
class RandomUniform(Initializer):

    def __init__(self, minval: float = -0.5, maxval: float = 0.5) -> None:
        self.minval = minval
        self.maxval = maxval

    
    def __call__(self, shape):
        
        return np.random.uniform(low=self.minval, high=self.maxval, size=shape)

class Xavier(Initializer):

    def __call__(self, shape, input_size):

        return np.random.randn(*shape) * np.sqrt(1. / input_size)

class Hue(Initializer):

    def __call__(self, shape, input_size):

        return np.random.randn(*shape) * np.sqrt(2. / input_size)


