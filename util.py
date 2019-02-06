import os
import numpy as np


cwd = "/Users/williamst-arnaud/U de M/IFT6135/assignment1"


def load_data(path=cwd):
    """
    Function to load the data
    :param path: Path from where we load the data
    :return: The training set, validation set and test set
    """
    train, valid, test = np.load(os.path.join(path, "mnist.npy"))

    return train, valid, test
