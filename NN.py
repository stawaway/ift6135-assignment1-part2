import torch
import torch.nn.functional as f


class ConvNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # convolution layer 1. Output shape is [batch_size, 16, 14, 14]
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1_bn = torch.nn.BatchNorm2d(16)

        # convolution layer 2. Output shape is [batch_size, 32, 7, 7]
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_bn = torch.nn.BatchNorm2d(32)

        # convolution layer 3. Output shape is [batch_size, 64, 3, 3]
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3_bn = torch.nn.BatchNorm2d(64)

        # convolution layer 4. Output shape is [batch_size, 128, 1, 1]
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # fully connected layer. Output shape is [batch_size, 10]
        self.fc1 = torch.nn.Linear(128, 10)

    def forward(self, x):
        """
        Method that performs the forward computation for the network
        :param x: The input to the network. Shape is [batch_size, 1, 28, 28]
        :return: The ouput of the network after a forward pass
        """
        # compute the first activation
        x = f.relu(self.conv1(x))

        # perform pooling operation
        x = self.pool1(x)

        # compute the second activation
        x = self.conv1_bn(x)
        x = f.relu(self.conv2(x))

        # perform pooling operation
        x = self.pool2(x)

        # compute the third activation
        x = self.conv2_bn(x)
        x = f.relu(self.conv3(x))

        # perform pooling operation
        x = self.pool3(x)

        # compute the fourth activation
        x = self.conv3_bn(x)
        x = f.relu(self.conv4(x))

        # perform the pooling operation
        x = self.pool4(x)

        # reshape the data for linear operation
        x = x.view(-1, 128)

        # fully-connected layer
        x = self.fc1(x)

        return x
