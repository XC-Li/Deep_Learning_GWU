# ---------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
# --------------------------------------------------------------------------------------------
# Choose the right values for x.
input_size = 3 * 32 * 32
hidden_size = 30
num_classes = 10
num_epochs = 10
batch_size = 10000
learning_rate = 0.1
momentum = 0.9
# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------

train_set = torchvision.datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Find the right classes name. Save it as a tuple of size 10.
classes = ('', '', ...)
# --------------------------------------------------------------------------------------------
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------------------------------------------------------------
# Choose the right argument for xx
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.t2 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.t2(out)
        return out
# --------------------------------------------------------------------------------------------
# Choose the right argument for x
net = Net(input_size, hidden_size, num_classes)
# --------------------------------------------------------------------------------------------
# Choose the right argument for x
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data
        images= images.view(-1, 3 * 32 * 32)
        images, labels = Variable(images), Variable(labels)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, 5, loss.item()))
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 3 * 32 * 32))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels)


#print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print("correct:", correct.sum().item())
print("total:", total)
# --------------------------------------------------------------------------------------------

# _, predicted = torch.max(outputs.data, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1,3* 32 * 32))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels)
    for i in range(10):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')