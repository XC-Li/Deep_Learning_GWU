#----------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
p = torch.randn(Batch_size, R)
t = torch.randn(Batch_size, a_size, requires_grad=False)
#----------------------------------------------------------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)
#----------------------------------------------------------------------------
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------
learning_rate = 1e-4
loss_list = []
max_epoch = 10

for k, v in model.named_parameters():
    print(k, v.shape)

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