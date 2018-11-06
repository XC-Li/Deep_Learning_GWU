#----------------------------------------------------------------------------
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
R = 1            
S = 20            
a_size = 1              
num_epochs = 50000
#----------------------------------------------------------------------------
inputs1 = np.linspace(-3,3,200, dtype=np.float32).reshape(-1,1)
targets1 =  0.1*np.power(inputs1, 2) * np.sin((inputs1)).reshape(-1,1)

p = Variable(torch.from_numpy(inputs1).cuda())
t = Variable(torch.from_numpy(targets1).cuda())
#----------------------------------------------------------------------------
check1 =p.is_cuda
print("Check inputs and trages are using cuda-------->  " + str(check1) )

model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.Tanh(),
    torch.nn.Linear(S, a_size),
)

model.cuda()
#----------------------------------------------------------------------------
check2 =next(model.parameters()).is_cuda
print("Check model is using cuda---------------------> " + str(check2) )
#----------------------------------------------------------------------------
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------
learning_rate = 1e-1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#----------------------------------------------------------------------------
for epoch in range(num_epochs):
    inputs = p
    targets = t
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
#----------------------------------------------------------------------------
zz = model(p)
zz1 = zz.data.cpu().numpy()
#----------------------------------------------------------------------------
plt.figure(1)
plt.scatter(inputs1, targets1,c='Red')
plt.hold(True)
plt.scatter(inputs1,zz1)
plt.show()

