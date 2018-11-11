# ----------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
# ----------------------------------------------------------------------------
p = torch.randn(Batch_size, R)
t = torch.randn(Batch_size, a_size, requires_grad=False)
# ----------------------------------------------------------------------------
model = torch.nn.Sequential(  # 1-100-50-20-1
    torch.nn.Linear(R, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, a_size),
)
# ----------------------------------------------------------------------------
performance_index = torch.nn.MSELoss(reduction='sum')
# ----------------------------------------------------------------------------
learning_rate = 1e-4
loss_list = []
max_epoch = 500
for index in range(max_epoch):

    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.item())
    loss_list.append(loss.item())

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad

plt.plot(range(max_epoch), loss_list)
# plt.yscale("log")
plt.xscale("log")
plt.title("MSE")
plt.show()