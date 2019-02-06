import torch
import torch.optim as optim
import util
from NN import ConvNet
import argparse
import numpy as np
import os


parser = argparse.ArgumentParser(description="Script to train a convolutional network")
parser.add_argument("--path", default=util.cwd)
parser.add_argument("--mb", default=64, type=int)
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--lr", default=None, type=float)
parser.add_argument("--restore", action="store_true")


def mini_batch(x, batch_size):
    """
    Function that takes a data set and returns a list of mini-batches
    :param x: The input data set
    :param batch_size: The size of the mini-batches
    :return: List of mini-batches
    """
    # data is shape [n, 28, 28] and label is shape [n, ]
    data, label = x
    n = data.shape[0]

    data_batches = []
    label_batches = []

    for start in range(n // batch_size):
        end = np.minimum(start + batch_size, n).astype(int)
        data_batches.append(data[start:end, :, :])
        label_batches.append(label[start:end])

    return data_batches, label_batches


def adam_optimizer(net, lr):
    """
    Function that creates an Adam optimizer for training the network
    :param net: The network that will be trained
    :param lr: The learning rate for the optimizer
    :return: A tuple consisting of the loss operation and the optimizer
    """
    # if learning rate is None, then use default one
    if lr is None:
        lr = 1e-3

    # The loss operation
    loss = torch.nn.CrossEntropyLoss()

    # Adam optimizer
    opt = optim.Adam(net.parameters(), lr=lr)

    return loss, opt


def train_net(conv_net, batch_size, n_epochs, lr=None, start_epoch=0, start_loss=0):
    """
    Function that trains a network
    :param conv_net: The network to train
    :param batch_size: The size of the mini-batches on which the network trains
    :param n_epochs: The number of training epochs
    :param lr: The learning rate of the network
    :param start_epoch: The epoch number from which to start. Used when restoring checkpoint
    :param start_loss: The starting training loss. Used when restoring checkpoint
    :return:
    """
    # get data
    train, valid, test = util.load_data()

    # reshape the data from MNIST to the right shapes
    reshape = lambda x: np.reshape(x, [-1, 1, 28, 28])
    train[0], valid[0], test[0] = [reshape(dataset[0]) for dataset in [train, valid, test]]

    # get mini-batches schedule for training
    train = mini_batch(train, batch_size)

    # Create the loss operation and the optimizer
    loss_op, opt = adam_optimizer(conv_net, lr)

    for epoch in range(start_epoch, n_epochs):
        train_loss = 0

        for idx, batch in enumerate(train[0]):
            label = train[1][idx]

            batch = torch.from_numpy(batch)
            label = torch.from_numpy(label)

            # set the parameters gradients to zero for this mini-batch
            opt.zero_grad()

            # Forward pass, backpropagation and update
            output = conv_net(batch)
            loss = loss_op(output, label)
            loss.backward()
            opt.step()

            # add current loss to the total training loss
            train_loss += loss.item()

        # print training loss
        print("Training loss after epoch {} is: ".format(epoch), train_loss / len(train[0]))

        # total validation loss for the epoch
        valid_loss = 0

        # after an epoch, compute the loss on the validation set
        data, label = valid
        data, label = torch.from_numpy(data), torch.from_numpy(label)
        output = conv_net(data)
        loss = loss_op(output, label)
        valid_loss += loss.item()


def checkpoint(conv_net, optimizer, loss, epoch, path=os.path.join(util.cwd, "checkpoints")):
    """
    Function that creates a checkpoint
    :param conv_net: The convolutional network
    :param optimizer: The optimizer that is used
    :param loss: The current loss
    :param epoch: The current epoch
    :param path: The path where to save the checkpoint
    :return:
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": conv_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, path)


def save_model(conv_net, path):
    """
    Function that saves a network's parameters to file
    :param conv_net: The network from which we get the parameters to save
    :param path: The path where we save the network
    """
    torch.save(conv_net.state_dict(), path)


def load_model(conv_net, optimizer, path):
    """
    Function that restores a checkpoint into memory
    :param conv_net: The convolutional network that will be trained
    :param optimizer: The optimizer that was used to train the network
    :param path: The path where the network is saved
    :return: The network once restored
    """
    checkpoint = torch.load(path)
    conv_net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    conv_net.train()

    return conv_net, optimizer, epoch, loss


if "__main__" == __name__:
    namespace = parser.parse_args()

    conv_net = ConvNet()
    _, optimizer = adam_optimizer(conv_net, namespace.lr)
    if namespace.restore:
        # restore the network
        conv_net, optimizer, epoch, loss = load_model(conv_net, optimizer, namespace.path)

        # train the network
        train_net(conv_net, namespace.mb, namespace.epochs, namespace.lr, start_epoch=epoch, start_loss=loss)
    else:
        # train the network
        train_net(conv_net, namespace.mb, namespace.epochs, namespace.lr)

    # save the parameters
    save_model(conv_net, namespace.path)
