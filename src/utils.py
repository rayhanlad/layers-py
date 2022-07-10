import numpy as np


def create_mini_batch(X, y, batch_size):

    m = X.shape[0]
    batch_number = int(m/batch_size)

    print(batch_size*(0))

    batches_X = [X[batch_size*(i):batch_size*(i+1),:] for i in range(batch_number)]
    batches_y = [y[batch_size*(i):batch_size*(i+1),:] for i in range(batch_number)]

    if m%batch_size != 0:
        batches_X.append(X[batch_size*batch_number:,:])
        batches_y.append(y[batch_size*batch_number:,:])

    return batches_X, batches_y

