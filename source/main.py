from __future__ import print_function
import torch
import net

from source import utils

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = './../models/mnist_cnn.pt'
use_cuda = True

# MNIST Test dataset and dataloader declaration
data_root = '~/Documents/data/mnist'
test_loader = utils.get_mnist_test_loader(data_root)

device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = net.Net().to(device)
# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = utils.test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# utils.plot_accuracy(epsilons, accuracies)
utils.plot_image(epsilons, examples)
