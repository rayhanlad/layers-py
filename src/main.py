import numpy as np
import matplotlib.pyplot as plt
from activations import ReLU, Tanh
from initializers import RandomNormal

from losses import MSE
from optimizers import SGD
from layers import Dense, Flatten, CellSimpleRNN
from network import Network
from layers import Conv2D


def cnn_main():

    data = np.ones((100, 10, 10, 1))


    conv_layer = Conv2D(
                            n_filters=12, 
                            filter_size=(3,3), input_shape=(10, 10, 1),
                            activation=ReLU(), initializer=RandomNormal()
                        )

    flatten_layer = Flatten()
    


    
    conv_1 = conv_layer.forward(data)
    flattened = flatten_layer.forward(conv_1)
    
    dense_1_layer = Dense(flattened.shape[1], 60, activation=ReLU())

    dense_1 = dense_1_layer.forward(flattened)
    dense_2_layer = Dense(60, 1, activation=ReLU())
    dense_2 = dense_2_layer.forward(dense_1)

    print(dense_2.shape)

    output_grad = dense_2 - 0
    output_grad, weight2_grad = dense_2_layer.backward(output_grad)
    output_grad, weight1_grad = dense_1_layer.backward(output_grad)
    

    _, kernel_gradient = conv_layer.backward(output_grad)
   
    conv_layer.filters -= 0.1*kernel_gradient


    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 3
    for i in range(0, columns*rows):
        img = conv_layer.forward(data)[0][i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def get_XY():

    X = [np.array([i, i+10, i+20]) for i in range(0, 100, 10)]
    Y = [np.array([i+30]) for i in range(0, 100, 10)]

    return X, Y

def rnn_main():

    X, Y = get_XY()
    X = np.asarray(X)

    loss_f = MSE()


    
    rnn = CellSimpleRNN(input_size=1, hidden_state=2, timesteps=3,
                        activation=Tanh(), return_sequence=False)

    dense = Dense(2, 1)
    for epoch in range(10):
        for x_, y_ in zip(X, Y):
            
            x_ = x_.reshape(3, 1)
            rnn_forward = rnn.forward(x_).reshape(1, 2, 1)
            pred = dense.forward(rnn_forward)
            loss = loss_f.compute(pred, y_)

            print(f'Pred: {pred}')
            print(f'Loss: {loss}')

            output_error = loss_f.derivative(pred, y_)
            output_error, updates_dense = dense.backward(output_error)

            dense.update(updates_dense)
            print(rnn_forward.shape)
            print(output_error[0].shape)
            _, updates_rnn = rnn.backward(output_error[0])
            rnn.update(updates_rnn)





if __name__ == '__main__':

    rnn_main()