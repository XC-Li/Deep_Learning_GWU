#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1            # Input size
S = 100             # Number of neurons
a_size = 1              # Network output size
#----------------------------------------------------------------------------
size = 100
p = np.linspace(-3, 3, size)
t = np.sin(p)

p = Variable(torch.from_numpy(p)).float().view(size, -1).cuda()
t = Variable(torch.from_numpy(t)).float().view(size, -1).cuda()

# p = Variable(torch.randn(Batch_size, R)).cuda()
# t = Variable(torch.randn(Batch_size, a_size), requires_grad=False).cuda()


model = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 25),
    torch.nn.ReLU(),
    torch.nn.Linear(25, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, a_size),
)
model.cuda()
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------
learning_rate = 1e-4
#----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#----------------------------------------------------------------------------
for index in range(10000):

    a = model(p)

    loss = performance_index(a, t)

    print(index, loss.item())

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
plt.title("t=sin(p)")
plt.show()