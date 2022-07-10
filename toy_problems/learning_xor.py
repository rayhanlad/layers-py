import numpy as np
from src.layers import Dense
from src.activations import ReLU, Sigmoid
from src.losses import BCE
from src.network import Network
from src.optimizers import SGD
import matplotlib.pyplot as plt

def main() -> None:

    #Define XOR dataset.
    
    data = np.array([
                        [0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]
                        ]).reshape(-1, 2, 1)

    label = np.array([0, 1, 1, 0]).reshape(-1, 1, 1)

    #Fix a seed for reproductibility
    np.random.seed(42)

    #Network definition.
    layers = [
        Dense(2, 2, activation=ReLU()),
        Dense(2, 1, activation=Sigmoid())
                ]

    bce_loss = BCE()
    net = Network(layers)
    net.compile(loss = bce_loss, learning_rate = 0.2, batch_size=2, optimizer=SGD())

    #Training of our shallow neural network.
    losses = net.fit(data, label, 2000)

    plt.figure()
    plt.plot(losses)
    plt.show()

    print('-- Prediction --')
    print(net.forward(data).astype(np.float16))


if __name__ == '__main__':
    main()