"""Machine Learning 2 Section 10 @ GWU
Quiz 4 - Solution for Q3
Author: Xiaochi (George) Li"""

import torch
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
#----------------------------------------------------------------------------
Q = 1            # Input size
S = 2000             # Number of neurons
a = 1              # Network output size
#----------------------------------------------------------------------------
torch.manual_seed(42)
size = 100
p = np.linspace(-3, 3, size)
t = np.exp(-np.abs(p)) * np.sin(np.pi * p)

p = torch.from_numpy(p).float().view(size, -1).cuda()
t = torch.from_numpy(t).float().view(size, -1).cuda()
#----------------------------------------------------------------------------
w1 = torch.randn(Q, S, device=device, dtype=dtype)
w2 = torch.randn(S, a, device=device, dtype=dtype)
learning_rate = 1e-6
#----------------------------------------------------------------------------
for index in range(5000):

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


h = p.mm(w1)
h_relu = h.clamp(min=0)
a_net = h_relu.mm(w2)

prediction = a_net.cpu().numpy()
real = t.cpu().numpy()
x = p.cpu().numpy()
plt.plot(x, real, label="Actual")
plt.scatter(x, prediction, label="NN Prediction")
plt.legend()
plt.title("title")
plt.show()