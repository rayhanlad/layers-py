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

class CellSimpleRNN(Layer):

    def __init__(
        self, 
        input_size: int, 
        hidden_state: int,
        timesteps: int, 
        activation: Activation = None, 
        initializer = RandomNormal(),
        return_sequence: bool = False
        ) -> None:


        super().__init__(True)

        self.input_size = input_size
        self.hidden_state = hidden_state
        self.timesteps = timesteps

        self.activation = activation
        self.initializer = initializer
        self.input = None

        self.return_sequence = return_sequence

        self.states = None
        self.input_weight = None
        self.memory_weight = None
        self.bias = None

        self._build()
    
    def _build(self):

        self.input_weight = self.initializer((self.input_size, self.hidden_state))
        self.memory_weight = self.initializer((self.hidden_state, self.hidden_state))
        self.states = np.zeros((self.timesteps+1, self.hidden_state))
        self.bias = np.zeros((1, self.hidden_state))
        
    
    def forward(self, input: np.ndarray) -> np.ndarray:

        self.input = input
        self.states[1] = np.dot(self.states[0], self.memory_weight)


        for i in range(1, self.timesteps):

            self.states[i+1] = np.dot(self.states[i], self.memory_weight) 
            + np.dot(input[i], self.input_weight) 
            + self.bias

        if self.activation:
            self.states[1:] = np.asarray([self.activation.forward(act) for act in self.states[1:]])
            #self.activation.forward(self.states[1:])
        
        if self.return_sequence:
            return self.states[1:]

        return self.states[-1]
    
    def backward(self, output_error):

        if self.activation:
            output_error = self.activation.backward(output_error)[0]
                
        #Case with one input.
        ds = [output_error]
    
        for i in range(len(self.states)-1):

            ds += [np.dot(ds[-1], self.memory_weight)]
        
        ds = np.asarray(ds)

        print(self.input.T.shape)
        dwx = np.mean(np.dot(self.input.T, ds[1:]), axis=1)
        dwrec = np.mean(np.dot(np.array(self.states[:-1]).T, ds[1:]), axis=1)

        return output_error, [dwx, dwrec]
    
    def update(self, updates):

        self.input_weight += updates[0]*-0.000002
        self.memory_weight += updates[1]*-0.000002