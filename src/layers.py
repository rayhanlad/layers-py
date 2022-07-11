from BaseClass.layer import Layer
from BaseClass.activation import Activation
import numpy as np
from initializers import RandomNormal

from scipy.signal import correlate2d





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


class Conv2D(Layer):

    def __init__(self, n_filters: int, filter_size: tuple[int], input_shape, activation, initializer, trainable: bool = True) -> None:

        super().__init__(trainable)

        self.n_filters = n_filters
        self.filter_size = filter_size

        self.input_shape = input_shape
        self.output_shape = None

        self.filters = None
        self.filters_shape = None
        self.channels = None

        self.activation = activation
        self.initializer = initializer

        self.input = None

        self._build()
    

    def _build(self) -> None:
        

        self.channels = self.input_shape[-1]
        self.filters_shape = (self.n_filters, *self.filter_size, self.channels)

        self.output_shape = (self.n_filters, 
                            self.input_shape[0]-self.filter_size[0]+1,
                            self.input_shape[1]-self.filter_size[1]+1)
        
        self.filters = self.initializer(self.filters_shape)



    def forward(self, input: np.ndarray) -> np.ndarray:

        self.input = input
        batch_size = len(input)

        forward = [
            [
                sum(
                    correlate2d(x[:,:,i], self.filters[k,:,:,i], mode='valid')
                    for i in range(self.channels)
                )
                for k in range(self.n_filters)     
            ]
            for x in input
            ]
        
        output = np.reshape(forward, (batch_size, *self.output_shape))

        if self.activation:
            output = self.activation.forward(output)

        return output
    
    def backward(self, output_error: np.ndarray) -> np.ndarray:

        batch = len(output_error)
        output_error = np.reshape(output_error, (batch, *self.output_shape))

        print(output_error.shape)
        print(self.input.shape)


        batch_filters_grad = [
            [
                [
                    correlate2d(x[:,:,i], error[k,:,:], mode='valid')
                    for i in range(self.channels)
                ]
                for k in range(self.n_filters)
            ]
            for x, error in zip(self.input, output_error)
        ]

        reshape_grad = np.reshape(batch_filters_grad, (len(self.input), self.n_filters, *self.filter_size, self.channels))

        filters_gradient = np.mean(reshape_grad, axis=0)

        #TO DO, retropogate error wrt the input.
        input_error = None

        return input_error, filters_gradient