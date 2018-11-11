import numpy as np
import torch
import timeit

#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------
p = np.random.randn(Batch_size, R)
t = np.random.randn(Batch_size, a)
#----------------------------------------------------------------------------
# Randomly initialize weights
w1 = np.random.randn(R, S)
w2 = np.random.randn(S, a)

learning_rate = 1e-6
#----------------------------------------------------------------------------
# Change dtype to Torch Tensor
p = torch.from_numpy(p)
t = torch.from_numpy(t)
w1 = torch.from_numpy(w1)
w2 = torch.from_numpy(w2)

#----------------------------------------------------------------------------
start_time = timeit.default_timer()
for index in range(1000):

    # h = p.dot(w1)
    # h_relu = np.maximum(h, 0)
    # a_net = h_relu.dot(w2)
    #
    # loss = np.square(a_net - t).sum()
    # print(index, loss)
    #
    # grad_y_pred = 2.0 * (a_net - t)
    # grad_w2 = h_relu.T.dot(grad_y_pred)
    # grad_h_relu = grad_y_pred.dot(w2.T)
    # grad_h = grad_h_relu.copy()
    # grad_h[h < 0] = 0
    # grad_w1 = p.T.dot(grad_h)
    #
    # w1 -= learning_rate * grad_w1
    # w2 -= learning_rate * grad_w2

    h = p.mm(w1)
    h_relu = h.clamp(min=0)
    a_net = h_relu.mm(w2)

    loss = (a_net - t).pow(2).sum()
    print(index, loss.item())

    grad_y_pred = 2.0 * (a_net - t)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = p.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

elapsed = timeit.default_timer() - start_time
print("\nTime:", elapsed)
