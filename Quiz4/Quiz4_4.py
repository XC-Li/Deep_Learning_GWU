import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

size = 100
p = np.linspace(-3, 3, size)
t = np.exp(-np.abs(p)) * np.sin(np.pi * p)

p = Variable(torch.from_numpy(p)).float().view(size, -1).cuda()
t = Variable(torch.from_numpy(t)).float().view(size, -1).cuda()


R = 1           # Input size
S = 20            # Number of neurons
a_size = 1            # Network output size

model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)
model.cuda()

performance_index = torch.nn.MSELoss()

learning_rate = 1e-1
max_epoch = 100
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(max_epoch):

    a = model(p)
    loss = performance_index(a, t)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# visualize
prediction = model(p).detach().cpu().numpy()
real = t.cpu().numpy()
x = p.cpu().numpy()
plt.plot(x, real, label="Actual")
plt.plot(x, prediction, label="NN Prediction")
plt.legend()
plt.title("title")
plt.show()

