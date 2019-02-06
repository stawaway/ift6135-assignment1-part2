import os
import numpy as np


cwd = os.path.dirname(os.path.realpath(__file__))


def load_data(path=cwd):
    """
    Function to load the data
    :param path: Path from where we load the data
    :return: The training set, validation set and test set
    """
    train, valid, test = np.load(os.path.join(path, "mnist.npy"))

    return train, valid, test
