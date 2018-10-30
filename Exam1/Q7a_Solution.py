"""Machine Learning 2 Section 10 @ GWU
Exam 1 - Solution for Q7 Helper function
Author: Xiaochi (George) Li"""


def helper(file_name):
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
    hidden_size = 30
    num_classes = 10
    num_epochs = 1
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
    input_gradient_trace = []
    for epoch in range(num_epochs):
        # Loop in batches
        start_time = timeit.default_timer()
        for i, data in enumerate(train_loader):
            images, labels = data
            # images= images.view(-1, input_size)
            # modify the line here to make the input track gradients
            images, labels = Variable(images.view(-1, input_size).cuda(), requires_grad=True), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            input_gradient_trace.append(images.grad.data.cpu())

            # print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, 50000/batch_size, loss.item()))
        elapsed = timeit.default_timer() - start_time
        print('Epoch [%d/%d],  Loss: %.4f, Time:%4f' % (epoch + 1, num_epochs, loss.item(), elapsed))

    # print("input gradients:\n", images.grad)
    input_gradient_trace = torch.cat(input_gradient_trace, 0)  # concatenate to a tensor

    import numpy as np

    input_gradient_trace = input_gradient_trace.data.cpu().numpy()  # transform to numpy array
    print("Size of gradient trace:", input_gradient_trace.shape)
    avg_grad = np.average(input_gradient_trace, 1).reshape(1, -1)
    print("Size of average gradient trace:", avg_grad.shape)
    with open(file_name, "ab") as trace_file:
        print("Saving to", file_name)
        np.savetxt(trace_file, avg_grad, delimiter=',')
    return avg_grad