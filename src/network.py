
from BaseClass.layer import Layer
from BaseClass.loss import Loss
import numpy as np
from utils import create_mini_batch

class Network:

    def __init__(self, layers: list[Layer]) -> None:

        self.layers = layers
        self.loss = None
        #sself.learning_rate = None
        self.optimizer = None
        

        

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int) -> list[float]:
    
        batches_X, batches_y = create_mini_batch(X_train, y_train, self.batch_size)
        losses = []

        for epoch in range(epochs):
            
            loss = 0
            for batch_x, batch_y in zip(batches_X, batches_y):

                y_hat = self.forward(batch_x)
                loss += self.loss.compute(y_hat, batch_y)
                error = self.loss.derivative(y_hat, batch_y)
                self.backward(error)

            losses.append(loss)

        return losses


    def forward(self, input: np.ndarray) -> np.ndarray:


        x = input
        for layer in self.layers:
            x = layer.forward(x)
    
        return x
    
    def backward(self, output: np.ndarray) -> np.ndarray:

        error = output
        for layer in reversed(self.layers):
            error, gradients = layer.backward(error)
            updates = self.optimizer.update(gradients)

            if layer.trainable:
                layer.update(updates)
            
        return error
    
    def compile(self, loss: Loss, learning_rate: float, batch_size:int = None, optimizer = None) -> None:

        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer.learning_rate = self.learning_rate        


    def predict(self, z):
        pass