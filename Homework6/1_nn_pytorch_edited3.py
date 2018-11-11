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
max_epoch = 1000
param_dict = {}

for k, v in model.named_parameters():
    print(k, v.shape)
    param_dict[k] = []


for index in range(max_epoch):

    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.item())
    loss_list.append(loss.item())

    model.zero_grad()
    loss.backward()

    for k, v in model.named_parameters():
        param_dict[k].append(v)

    for param in model.parameters():
        param.data -= learning_rate * param.grad

import numpy as np
for k,v in param_dict.items():
    v = torch.cat(v,0).detach().numpy()
    with open(str(k) + ".csv", "ab") as grad_file:
        print("saving:", k, "Shape:", v.shape)
        np.savetxt(grad_file, v, delimiter=",")
