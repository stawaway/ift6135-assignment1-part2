import torch
import torch.optim as optim
import util
from NN import ConvNet
import argparse
import numpy as np
import os


parser = argparse.ArgumentParser(description="Script to evaluate a convolutional network")
parser.add_argument("--path", default=util.cwd)


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


if "__main__" == __name__:
    namespace = parser.parse_args()

    # Create convolutional network
    conv_net = ConvNet()

    # load the network for evaluation
    conv_net = load_model(conv_net, namespace.path)
