import torch
import torch.optim as optim
import util
from NN import ConvNet
import argparse
import numpy as np
import os


parser = argparse.ArgumentParser(description="Script to evaluate a convolutional network")
parser.add_argument("--path", default=os.path.join(util.cwd, "model.pt"))
parser.add_argument("--data", default=os.path.join(util.cwd, "mnist.npy"))


def load_model(conv_net, path):
    """
    Function to load the trained network into memory
    :param conv_net: The convolutional network that will be trained
    :param path: The path where the network is saved
    :return: The network once restored
    """
    conv_net.load_state_dict(torch.load(path))
    conv_net.eval()

    return conv_net


def eval(conv_net, test):
    """
    Function that evaluates the convolutional network on the test data
    :param conv_net:
    :param test:
    :return:
    """
    data, label = test
    data = np.reshape(data, [-1, 1, 28, 28])

    data, label = torch.from_numpy(data), torch.from_numpy(label)

    loss_op = torch.nn.CrossEntropyLoss()

    output = conv_net(data)
    loss = loss_op(output, label).item()
    acc = torch.eq(output.argmax(dim=1), label).float().mean().item()

    print("The test loss is: ", loss, "\t", "The test accuracy is: ", acc)


if "__main__" == __name__:
    namespace = parser.parse_args()

    # load the test data
    _, _, test = util.load_data(namespace.data)

    # Create convolutional network
    conv_net = ConvNet()

    # load the network for evaluation
    conv_net = load_model(conv_net, namespace.path)

    # evaluate the network
    eval(conv_net, test)
