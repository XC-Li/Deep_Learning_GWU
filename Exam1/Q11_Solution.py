"""Machine Learning 2 Section 10 @ GWU
Exam 1 - Solution for Q10
Author: Xiaochi (George) Li"""

# ---------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import timeit

torch.cuda.init()
torch.cuda.manual_seed(42)
# --------------------------------------------------------------------------------------------
# Choose the right values for x.
input_size = 3 * 32 * 32
hidden_size = 60
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
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# --------------------------------------------------------------------------------------------
# Define Neural Network
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
# Instantiation of the Neural Network
net = Net(input_size, hidden_size, num_classes)
net.cuda()
# --------------------------------------------------------------------------------------------
# Choose the loss function and optimization method
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
# --------------------------------------------------------------------------------------------
# Loop in epochs
total_time = 0
for epoch in range(num_epochs):
    # Loop in batches
    start_time = timeit.default_timer()
    for i, data in enumerate(train_loader):
        images, labels = data
        # images= images.view(-1, input_size)
        images, labels = Variable(images.view(-1, input_size).cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, 50000/batch_size, loss.item()))
    elapsed = timeit.default_timer() - start_time
    total_time += elapsed
    print('Epoch [%d/%d],  Loss: %.4f, Time:%4f' % (epoch + 1, num_epochs, loss.item(), elapsed))
print("Average time:", total_time/num_epochs)
# --------------------------------------------------------------------------------------------
# Overall accuracy rate
correct = 0
total = 0
for images, labels in test_loader:
    # images = Variable(images.view(-1, input_size).cuda())
    images, labels = Variable(images.view(-1, input_size).cuda()), Variable(labels.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels)

over_all_accuracy = correct.sum().item() / total
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * over_all_accuracy))
print('Misclassification of the network on the 10000 test images: %d %%' % (100 * (1 - over_all_accuracy)))

# --------------------------------------------------------------------------------------------
# Accuracy rate by category
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

images, labels = data
# images = Variable(images.view(-1,input_size).cuda())
images, labels = Variable(images.view(-1, input_size).cuda()), Variable(labels.cuda())
outputs = net(images)
_, predicted = torch.max(outputs.data, 1)
c = (predicted == labels)

"""predicted is the estimated target and labels is the ground truth,
we can construct the confusion matrix by sklearn easily"""
import warnings  # Mute warning
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix
predicted_np = predicted.data.cpu().numpy()
labels_np = labels.data.cpu().numpy()
confusion_matrix = confusion_matrix(labels_np, predicted_np)
print(confusion_matrix)

for i in range(c.size()[0]):
    label = labels[i].item()
    class_correct[label] += c[i].item()
    class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    # print(class_correct[i], class_total[i])
    accuracy = class_correct[i] / class_total[i]
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * accuracy))
    print('Misclassification of %5s : %2d %%' % (classes[i], 100 * (1 - accuracy)))
# --------------------------------------------------------------------------------------------
# Visualize network response for an arbitrary input
from torch.utils.data import RandomSampler
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


label_name = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
dataiter = iter(data_loader)
image, label = dataiter.next()
input = Variable(image.view(-1, input_size).cuda())
outputs = net(input)
_, predicted = torch.max(outputs.data, 1)

# show images
imshow(torchvision.utils.make_grid(image))
# print labels
print("Actual:", label.item(), label_name[label.item()])
print("Predicted:", predicted.item(), label_name[predicted.item()])
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')
