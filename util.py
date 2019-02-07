import os
import numpy as np


cwd = os.path.dirname(os.path.realpath(__file__))


def load_data(path=os.path.join(cwd, "mnist.py")):
    """
    Function to load the data
    :param path: Path from where we load the data
    :return: The training set, validation set and test set
    """
    train, valid, test = np.load(path)

    return train, valid, test
