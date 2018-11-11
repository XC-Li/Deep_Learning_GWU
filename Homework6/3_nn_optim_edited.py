#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable

#----------------------------------------------------------------------------
Batch_size = 25600000     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
p = Variable(torch.randn(Batch_size, R)).cuda()
t = Variable(torch.randn(Batch_size, a_size), requires_grad=False).cuda()


model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size ),
)
model.cuda()
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------
learning_rate = 1e-4
#----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#----------------------------------------------------------------------------
for index in range(500):

    a = model(p)

    loss = performance_index(a, t)

    print(index, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()