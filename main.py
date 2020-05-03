from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output

class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.4)
        self.dropout2 = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn32(x)
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn64(x)
        x = F.max_pool2d(x,2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num
    accuracy = 100. * correct / test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        accuracy))

    return test_loss, accuracy

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # get the current working directory
    path = os.getcwd()

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ColorJitter(),
                    transforms.RandomAffine(degrees = 20, scale = (0.8,1.2)),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    val_dataset = datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # assign indices for training/validation 
    # first convert the targets to a numpy array
    labels = train_dataset.targets.numpy()

    # get a list of the targets
    targets = np.unique(labels)

    # create lists to store the indicies for the training and validation sets
    subset_indices_train = []
    subset_indices_valid = []

    # choose the split for the validation set
    val_split = 0.85

    # choose the fraction of the training dataset that you want to train on
    train_frac = 1

    # ensure you always get the same train/validation splits
    np.random.seed(2020)

    # for each class split the dataset into training and validation sets
    for target in targets:
        # get all of the indices for the class
        class_indices = np.argwhere(labels == target).flatten()
        
        # find the splitting point
        split = int(np.floor(val_split * len(class_indices)))
        
        # shuffle the indicies
        np.random.shuffle(class_indices)
        
        # split the shuffled indicies into training and validation sets
        subset_indices_train.extend(class_indices[:int(split*train_frac)])
        subset_indices_valid.extend(class_indices[split:])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )
    
    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        # Get and load the training data
        train_dataset = datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ColorJitter(),
                    transforms.RandomAffine(degrees = 20, scale = (0.8,1.2)),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_train)
        )

        # Get and load the test data
        test_dataset = datasets.MNIST(path, train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        
        train_loss, train_accuracy = test(model, device, train_loader)
        test_loss, test_accuracy = test(model, device, test_loader)

        print('\nTrain Error = ', 100-train_accuracy)
        print('\nTest Error = ', 100-test_accuracy)
        
        return

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Set up lists to store loss and accuracy at the end of each epoch
    train_results = []
    valid_results = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss, train_accuracy = test(model, device, train_loader)
        valid_loss, valid_accuracy = test(model, device, val_loader)
        train_results.append([train_loss, train_accuracy])
        valid_results.append([valid_loss, valid_accuracy])
        scheduler.step()    # learning rate scheduler

    # Convert the results to a numpy array
    train_results = np.array(train_results)
    valid_results = np.array(valid_results)

    # Print out the results
    print('\nTraining Results')
    for i in range(len(train_results)):
        print('Epoch ', i+1, ': ', train_results[i])

    print('\nValidation Results')
    for i in range(len(valid_results)):
        print('Epoch ', i+1, ': ', valid_results[i])
    
    # find the epochs with the lowest loss and highest accuracy
    # for both the training and validation sets
    print("Min train loss = ", np.amin(train_results[:,0]),  " at epoch = ", np.argmin(train_results[:,0])+1)
    print("Max train accuracy = ", np.amax(train_results[:,1]),  " at epoch = ", np.argmax(train_results[:,1])+1)
    print("Min validation loss = ", np.amin(valid_results[:,0]),  " at epoch = ", np.argmin(valid_results[:,0])+1)
    print("Max validation accuracy = ", np.amax(valid_results[:,1]),  " at epoch = ", np.argmax(valid_results[:,1])+1)
    
    # Plot the training and validation loss as a function of
    # the epoch to monitor for overfitting
    plt.figure(1)
    plt.plot(np.arange(1,len(train_results)+1),train_results[:,0], marker = '.', label = 'Training Loss')
    plt.plot(np.arange(1,len(valid_results)+1),valid_results[:,0], marker = '.', label = 'Validation Loss')
    plt.ylim(bottom = 0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # You may optionally save your model at each epoch here
    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")

if __name__ == '__main__':
    main()
