"""Machine Learning 2 Section 10 @ GWU
Quiz 4 - Solution for Q4
Author: Xiaochi (George) Li"""

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


torch.manual_seed(42)
size = 100
p = np.linspace(-3, 3, size)
t = np.exp(-np.abs(p)) * np.sin(np.pi * p)

p = Variable(torch.from_numpy(p)).float().view(size, -1).cuda()
t = Variable(torch.from_numpy(t)).float().view(size, -1).cuda()


R = 1           # Input size
S = 2000           # Number of neurons
a_size = 1            # Network output size

model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size)
)
model.cuda()

performance_index = torch.nn.MSELoss()

learning_rate = 0.1
max_epoch = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(max_epoch):

    a = model(p)
    loss = performance_index(a, t)
    print(epoch, loss.item())
    if loss.item() < 1e-4:
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# visualize
prediction = model(p).detach().cpu().numpy()
real = t.cpu().numpy()
x = p.cpu().numpy()
plt.plot(x, real, label="Actual")
plt.scatter(x, prediction, label="NN Prediction")
plt.legend()
plt.title("title")
plt.show()

