from BaseClass.layer import Layer
from BaseClass.activation import Activation
import numpy as np
from initializers import RandomNormal




class Dense(Layer):

    def __init__(self, input_size: int, output_size: int, activation: Activation = None, initializer = RandomNormal()) -> None:

        super().__init__(True)
        self.input_size = input_size
        self.output_size = output_size

        self.activation = activation
        self.initializer = initializer
        self.input = None

        self._build()
    

    def _build(self) -> None:
        self.weight = self.initializer((self.output_size, self.input_size))
        self.bias = self.initializer((self.output_size, 1))
    

    def forward(self, input: np.ndarray) -> np.ndarray:

        self.input = input
        output = np.matmul(self.weight, self.input) + self.bias

        if self.activation:
            return self.activation.forward(output)

        return output
    
    def backward(self, output_error: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray]]:


        if self.activation:
            output_error = self.activation.backward(output_error)[0]        

        input_error = np.matmul(self.weight.T, output_error)
        

        weight_error = np.mean(np.matmul(output_error, self.input.transpose(0,2,1)), axis=0)
        bias_error = np.mean(output_error, axis=0)

        return input_error, [weight_error, bias_error]
    
    def update(self, updates):

        self.weight += updates[0]
        self.bias += updates[1]


class Flatten(Layer):

    def __init__(self, input_shape=None, trainable: bool = True) -> None:
        super().__init__(input_shape, trainable=False)
    
    def forward(self, input:np.ndarray) -> np.ndarray:

        self.input = input
        m = self.input.shape[0]
        flatten_dim = np.prod(self.input.shape[1:])

        return np.reshape(input, (m, flatten_dim, 1))
    
    def backward(self, output_error: np.ndarray) -> np.ndarray:

        return np.reshape(output_error, self.input.shape), None


