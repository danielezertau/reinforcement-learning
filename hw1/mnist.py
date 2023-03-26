import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


INPUT_SIZE = 784
NUM_CLASSES = 10


class Net1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


class Net2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net2, self).__init__()
        hidden_layer_size = 500
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


def data_setup(batch_size):
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def train(model, train_data_loader, optim, num_epochs, logdir):
    # Loss and Optimizer
    writer = SummaryWriter(log_dir=logdir)
    criterion = nn.CrossEntropyLoss()

    iter_num = 0
    # Train the Model
    for epoch in range(num_epochs):
        print(f"Started epoch {epoch}")
        for images, labels in train_data_loader:
            iter_num += 1
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optim.zero_grad()
            output = model(images)
            pred = torch.max(output, dim=1).indices
            writer.add_scalar('Accuracy/train', torch.sum((pred == labels)) / labels.size(0), iter_num)
            loss = criterion(output, labels)
            writer.add_scalar('Loss/train', loss, iter_num)
            loss.backward()
            optim.step()
        print(f"Finished epoch {epoch}")


def evaluate(model, test_data_loader):
    # Test the Model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = Variable(images.view(-1, 28*28))
            total += labels.size(0)
            output = model(images)
            pred = torch.max(output, dim=1).indices
            correct += torch.sum((pred == labels))
            accuracy = 100 * correct / total

    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)


def q1():
    batch_size = 100
    num_epochs = 100
    lr = 1e-3
    train_loader, test_loader = data_setup(batch_size)
    net = Net1(INPUT_SIZE, NUM_CLASSES)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, train_loader, optimizer, num_epochs, logdir="runs/orig-params")
    evaluate(net, test_loader)


def q2_sgd():
    batch_size = 128
    num_epochs = 100
    lr = 5e-1
    train_loader, test_loader = data_setup(batch_size)
    net = Net1(INPUT_SIZE, NUM_CLASSES)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, train_loader, optimizer, num_epochs, logdir="runs/5e-1-batch128-100-epochs")
    evaluate(net, test_loader)


def q2_adam():
    batch_size = 128
    num_epochs = 100
    lr = 5e-3
    train_loader, test_loader = data_setup(batch_size)
    net = Net1(INPUT_SIZE, NUM_CLASSES)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_loader, optimizer, num_epochs, logdir="runs/adam-5e-3-batch128-100-epochs")
    evaluate(net, test_loader)


def q3():
    batch_size = 128
    num_epochs = 100
    lr = 8e-3
    train_loader, test_loader = data_setup(batch_size)
    net = Net2(INPUT_SIZE, NUM_CLASSES)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_loader, optimizer, num_epochs, logdir="runs/relu-deep-8e-3-batch128-adam")
    evaluate(net, test_loader)


if __name__ == '__main__':
    q1()
    q2_sgd()
    q2_adam()
    q3()
