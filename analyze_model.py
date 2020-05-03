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
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import os

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
        features = x
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output, features

def analyze_model(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    mistakes = np.empty((0,3))
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, features = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_truth = pred.eq(target.view_as(pred))

            # Convert tensors to numpy arrays
            data = data.numpy()
            target = target.numpy()
            features = features.numpy()
            pred = pred.numpy()
            pred = pred.flatten()

            # Define the mean and std corrections to be made
            mean_correction = 0.1307
            std_correction = 0.3081

            mistakes = []
            # loop throug the predictions
            for i in range(len(pred_truth)):
                    # find the incorrect predictions
                    if pred_truth[i] == False:
                        # Get the data for the image and denormalize
                        img = (data[i]*std_correction)+mean_correction
                        mistakes.append([img, target[i], pred[i]])
                    
    return data, target, pred, np.array(mistakes), features

# Set the device
device = torch.device("cpu")

# Get the current working directory
path = os.getcwd()

# Check to make sure the trained model is there
assert os.path.exists('mnist_model.pt')

# Set the test model
model = Net().to(device)
model.load_state_dict(torch.load('mnist_model.pt'))

# Get and load the test data
test_dataset = datasets.MNIST(path, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=False)

# Run the model on the test set and extract useful information
data, truth, pred, mistakes, features = analyze_model(model, device, test_loader)

'''
Visualize Mistakes
'''
# set up the grid for the images
fig, axarr = plt.subplots(3,3)
idx = 0
while idx < 9:
    for row in range(3):
        for col in range(3):
            axarr[row][col].imshow(mistakes[idx][0].squeeze(), cmap = 'gray')
            axarr[row][col].axis('off')
            axarr[row][col].set_title('{} -> {}'.format(mistakes[idx][1],mistakes[idx][2]))
            idx += 1

'''
Visualize Kernels
'''
# get the weights after the first convolution layer
kernels = model.conv1.weight.detach()

# set up the grid for plotting
fig, axarr = plt.subplots(3,3)
idx = 0
for row in range(3):
    for col in range(3):
        axarr[row][col].imshow(kernels[idx].squeeze(), cmap = 'gray')
        axarr[row][col].axis('off')
        axarr[row][col].set_title(idx+1)
        idx += 1

'''
Generate a Confusion Matrix
''' 
conf = confusion_matrix(truth, pred)
print('\nConfusion matrix:')
print(conf)

'''
Visualize the high-dimensional embedding
'''
# Use t-SNE to reduce the features to a 2D embedded space 
features_embedded = TSNE().fit_transform(features)

# Visualize the 2D embedded space
plt.figure()
scatter = plt.scatter(features_embedded[:,0],features_embedded[:,1], c = truth, cmap = 'tab10', alpha = 0.75)
plt.legend(*scatter.legend_elements(), title = 'Classes')
plt.title('High Dimensional Embedding Visualization')

'''
Visualize images with similar feature vectors
'''
# set up the grid for the images
fig, axarr = plt.subplots(5,9)
idx = 0
for row in range(5):
    # get the original feature vector
    x_0 = features[idx]

    # compute the euclidean distance to all other feature vectors
    dist = np.sqrt(np.sum((features-x_0)**2, axis = 1))

    # get the indices for the 9 images with the closest feature vectors
    # (note that this will include the original feature vector)
    img_idxs = np.argpartition(dist,9)
    img_idxs = img_idxs[:9]

    # get rid of the original feature vector
    img_idxs = img_idxs[img_idxs != idx]

    # display the original image
    axarr[row][0].imshow(data[idx].squeeze(), cmap = 'gray')
    axarr[row][0].axis('off')
    axarr[row][0].set_title('I_0')

    # display the 8 images with the closest feature vectors
    for col in range(1,9):
        axarr[row][col].imshow(data[img_idxs[col-1]].squeeze(), cmap = 'gray')
        axarr[row][col].axis('off')
        axarr[row][col].set_title('I_{}'.format(col))
        
    idx += 1

# show the all plots
plt.show()