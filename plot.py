from matplotlib import pyplot as plt
import numpy as np
from util import cwd
import os
import pickle


class Graph(object):

    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.test_loss = []
        self.epoch = 0

    def add_loss(self, loss, dataset):
        """
        Method that adds the loss to all the previous losses for a particular operation
        :param loss: The loss that we add after an epoch
        :param dataset: The data set from which the loss was computed
        :return:
        """
        list = getattr(self, dataset + "_loss")
        list.append(loss)
        self.epoch += 1

    def plot(self, dataset):
        """
        Method that plots the loss data
        :param dataset: The dataset from which the loss was computed
        :return:
        """
        list = getattr(self, dataset + "_loss")

        # Define graph
        plt.figure()
        plt.title("{set} loss after each epoch".format(set=dataset.capitalize()))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

        # Plot the data to the graph
        plt.plot(np.arange(len(list)), list)
        plt.tight_layout()

        # save the graph to file
        save_path = input("Where do you want to save the file?: ")
        plt.savefig(os.path.join(cwd, save_path))
