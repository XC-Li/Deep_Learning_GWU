import numpy as np
import torch
import timeit
import matplotlib.pyplot as plt

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
p = torch.from_numpy(p).cuda()
t = torch.from_numpy(t).cuda()
w1 = torch.from_numpy(w1).cuda()
w2 = torch.from_numpy(w2).cuda()

#----------------------------------------------------------------------------
start_time = timeit.default_timer()
loss_list = []
grad_w1_list = []
grad_w2_list = []
max_epoch = 300
for index in range(max_epoch):

    h = p.mm(w1)
    h_relu = h.clamp(min=0)
    a_net = h_relu.mm(w2)

    loss = (a_net - t).pow(2).sum()
    print(index, loss.item())
    loss_list.append(loss.item())

    grad_y_pred = 2.0 * (a_net - t)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = p.t().mm(grad_h)

    grad_w1_list.append(grad_w1.mean().item())
    grad_w2_list.append(grad_w2.mean().item())

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

elapsed = timeit.default_timer() - start_time
print("\nTime:", elapsed)

plt.plot(range(max_epoch), loss_list)
# plt.yscale("log")
plt.xscale("log")
plt.title("MSE")
plt.show()

plt.plot(range(max_epoch), grad_w1_list)
# plt.yscale("log")
plt.xscale("log")
plt.title("Gradient of W1")
plt.show()

plt.plot(range(max_epoch), grad_w2_list)
# plt.yscale("log")
plt.xscale("log")
plt.title("Gradient of W2")
plt.show()