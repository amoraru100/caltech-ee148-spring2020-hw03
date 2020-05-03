from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the number of training examples
train_examples = [50995, 25495, 12745, 6370, 3182]

# Define the training and test error for each
# number of training examples
train_errors = [0.8922443376801681, 1.157089625416745, 1.2946253432718748, 1.569858712715856, 2.2941546197360196]
test_errors = [0.7099999999999937, 0.7099999999999937, 0.9200000000000017, 1.4099999999999966, 1.9300000000000068]

# Plot the train and test errors as a function of training examples
plt.figure(1)
plt.plot(train_examples,train_errors, marker = '.', label = 'Training Error')
plt.plot(train_examples,test_errors, marker = '.', label = 'Test Error')
plt.loglog()
plt.xlabel('Training Eamples')
plt.ylabel('% Error')
plt.title('Training and Test Error vs Training Examples')
plt.legend()
plt.show()