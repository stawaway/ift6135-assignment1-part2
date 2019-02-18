import torch
import torch.optim as optim
import util
from NN import ConvNet
import argparse
import numpy as np
import os
from plot import Graph


parser = argparse.ArgumentParser(description="Script to train a convolutional network")
parser.add_argument("--path", default=os.path.join(util.cwd, "model.pt"))
parser.add_argument("--mb", default=64, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--plot", default=True, type=bool)
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
        lr = 1e-4

    # The loss operation
    loss = torch.nn.CrossEntropyLoss()

    # Adam optimizer
    opt = optim.Adam(net.parameters(), lr=lr)

    return loss, opt


def train_net(conv_net, batch_size, n_epochs, lr=None, graph=None):
    """
    Function that trains a network
    :param conv_net: The network to train
    :param batch_size: The size of the mini-batches on which the network trains
    :param n_epochs: The number of training epochs
    :param lr: The learning rate of the network
    :param graph: Graph object that will keep track of the loss function for the training set and the validation set
    :return:
    """
    # get data
    train, valid, test = util.load_data()

    # reshape the data from MNIST to the right shapes
    def reshape(x): return np.reshape(x, [-1, 1, 28, 28])
    train[0], valid[0], test[0] = [reshape(dataset[0]) for dataset in [train, valid, test]]

    # get mini-batches schedule for training
    train = mini_batch(train, batch_size)

    # Create the loss operation and the optimizer
    loss_op, opt = adam_optimizer(conv_net, lr)

    # get the starting epoch number
    start_epoch = 0
    if graph is not None:
        start_epoch = graph.epoch

    # Train the network
    for epoch in range(start_epoch, n_epochs+start_epoch):
        train_loss = 0
        train_acc = 0

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

            # add current loss and accuracy to the total training loss and accuracy
            train_loss += loss.item()
            train_acc += torch.eq(output.argmax(dim=1), label).float().mean().item()

        # print training loss and accuracy
        train_loss = train_loss / len(train[0])
        train_acc = train_acc / len(train[0])
        print("Training loss after epoch {} is: ".format(epoch), train_loss,
              "\t", "Training accuracy after epoch {} is: ".format(epoch), train_acc)

        # after an epoch, compute the loss and accuracy on the validation set
        data, label = valid
        data, label = torch.from_numpy(data), torch.from_numpy(label)
        output = conv_net(data)
        loss = loss_op(output, label)
        valid_loss = loss.item()
        valid_acc = torch.eq(output.argmax(dim=1), label).float().mean().item()

        # print validation loss and accuracy
        print("Validation loss after epoch {} is: ".format(epoch), valid_loss,
              "\t", "Validation accuracy after epoch {} is: ".format(epoch), valid_acc, "\n")

        # add the loss and accuracy to the training and validation set for the graph object
        if graph is not None:
            graph.add_epoch(train_loss, train_acc, "training")
            graph.add_epoch(valid_loss, valid_acc, "validation")

        # save checkpoint every 100 epochs
        if (epoch+1) % 100 == 0:
            checkpoint(conv_net, opt, valid_loss, epoch, graph,
                       os.path.join(util.cwd, "checkpoints/checkpoint_{}".format(epoch)))


def checkpoint(conv_net, optimizer, loss, epoch, graph, path):
    """
    Function that creates a checkpoint
    :param conv_net: The convolutional network
    :param optimizer: The optimizer that is used
    :param loss: The current loss
    :param epoch: The current epoch
    :param graph: The graph object to plot the data
    :param path: The path where to save the checkpoint
    :return:
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": conv_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "graph": graph
    }, os.path.join(path, "checkpoint_{}".format(epoch)))


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
    graph = checkpoint["graph"]
    conv_net.train()

    return conv_net, optimizer, epoch, loss, graph


if "__main__" == __name__:
    namespace = parser.parse_args()

    conv_net = ConvNet()
    _, optimizer = adam_optimizer(conv_net, namespace.lr)
    if namespace.restore:
        # restore the network
        conv_net, optimizer, epoch, loss, graph = load_model(conv_net, optimizer, namespace.path)

        # train the network
        epoch += 1
        train_net(conv_net, namespace.mb, namespace.epochs, namespace.lr, graph)
    else:
        # create new graph object if plot is  true
        graph = None
        if namespace.plot:
            graph = Graph()

        # train the network
        train_net(conv_net, namespace.mb, namespace.epochs, namespace.lr, graph)

        if namespace.plot:
            graph.plot("training")
            graph.plot("validation")

    # save the parameters
    save_model(conv_net, namespace.path)
