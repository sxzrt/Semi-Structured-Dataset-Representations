import sys

sys.path.append(".")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torch.nn.init as init


class RegNet_new(nn.Module):
    def __init__(self):
        super(RegNet_new, self).__init__()
        # Batch x Channel x Height x Width; 64- feature dimension
        self.conv0_1 = nn.Conv2d(1, 64, [30, 1], 1).apply(kaiming_init)
        self.conv0_2 = nn.Conv2d(64, 1 , 1, 1).apply(kaiming_init)
        self.fc0 = nn.Linear(64, 64).apply(kaiming_init)
        self.dropout0 = nn.Dropout2d(0.15)

        self.conv1_1 = nn.Conv2d(1, 64, [10, 1], 1).apply(kaiming_init)
        self.conv1_2 = nn.Conv2d(64, 1 , 1, 1).apply(kaiming_init)
        self.fc1 = nn.Linear(64, 64).apply(kaiming_init)
        self.dropout1 = nn.Dropout2d(0.15)


        self.conv2_1 = nn.Conv2d(1, 64, [100, 1], 1).apply(kaiming_init)
        self.conv2_2 = nn.Conv2d(64, 1 , 1, 1).apply(kaiming_init)
        self.fc2 = nn.Linear(64, 64).apply(kaiming_init)
        self.dropout2 = nn.Dropout2d(0.15)


        self.fc3 = nn.Linear(64+64+64, 64).apply(kaiming_init)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc4 = nn.Linear(64, 1).apply(kaiming_init)


    def forward(self, x, y, z): # shape, cluster, sample,
        x=self.conv0_1(x)
        x= F.relu(x)
        x=self.conv0_2(x)
        x= F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.dropout0(x)

        y = self.conv1_1(y)
        y = F.relu(y)
        y = self.conv1_2(y)
        y = F.relu(y)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.dropout1(y)

        z = self.conv2_1(z)
        z = F.relu(z)
        z = self.conv2_2(z)
        z = F.relu(z)
        z = torch.flatten(z, 1)
        z = self.fc1(z)
        z = F.relu(z)
        z = self.dropout2(z)

        f = torch.cat([x, y, z], dim=1)  # mean, variance, and fid
        f = self.fc3(f)
        f = self.dropout3(f)
        f = self.fc4(f)

        output = f.view(-1)
        return output

class Set_Rpresentation(data.Dataset):
    """
    """

    def __init__(self, path, data, label, transform=None, target_transform=None):
        super(Set_Rpresentation, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.path = path
        self.label_file = label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (shaper, cluster, samples) where target is index of the target class.
        """
        shape = np.load(self.path + '_' + str(self.data[index]) + '_shape.npy')
        cluster = np.load(self.path + '_' + str(self.data[index]) + '_cluster.npy')
        sample = np.load(self.path + '_' + str(self.data[index]) + '_fps.npy')
        mean = np.load(self.path + '_' + str(self.data[index]) + '_mean.npy')
        # var = np.load(self.path + '_' +  str(self.data[index]) + '_variance.npy')



        target = self.label_file[index]
        shape = torch.as_tensor(shape, dtype=torch.float).view(1, 30, 64)
        cluster = torch.as_tensor(cluster, dtype=torch.float).view(1, 10, 64)
        sample = torch.as_tensor(sample, dtype=torch.float).view(1, 100, 64)
        mean = torch.as_tensor(mean, dtype=torch.float)

        target = torch.as_tensor(target, dtype=torch.float)

        return shape, cluster, sample, mean, target

    def __len__(self):
        return len(self.data)


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        # Batch x Channel x Height x Width; 64- feature dimension
        self.conv0 = nn.Conv2d(1, 64, [1, 64], 1).apply(kaiming_init)
        self.conv0_1 = nn.Conv2d(64, 1 , 1, 1).apply(kaiming_init)
        self.fc0 = nn.Linear(9948, 64).apply(kaiming_init)

        self.conv1 = nn.Conv2d(1, 32, [64, 1], 1).apply(kaiming_init)
        self.conv2 = nn.Conv2d(32, 1, 1, 1).apply(kaiming_init)
        self.fc1 = nn.Linear(64, 32).apply(kaiming_init)
        self.fc2 = nn.Linear(64, 32).apply(kaiming_init)
        self.fc3 = nn.Linear(64 + 1, 32).apply(kaiming_init)
        self.fc4 = nn.Linear(32, 1).apply(kaiming_init)
        self.dropout1 = nn.Dropout2d(0.15)
        self.dropout2 = nn.Dropout2d(0.15)
        self.dropout3 = nn.Dropout2d(0.5)

    def forward(self, x, y, f):
        # x: cov; y: mean

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)

        y = self.fc1(y)
        y = self.dropout2(y)

        z = torch.cat([x, y, f], dim=1)  # mean, variance, and fid
        z = self.fc3(z)
        z = self.dropout3(z)
        z = self.fc4(z)

        output = z.view(-1)
        return output

class REG(data.Dataset):
    """
    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, path, data, label, fid, transform=None, target_transform=None):
        super(REG, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.path = path
        self.label_file = label
        self.fid = fid

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (mean, var, target) where target is index of the target class.
        """

        m1 = np.load(self.path  + 'train_mean.npy')
        s1 = np.load(self.path  + 'train_variance.npy')
        mean = abs(np.load(self.path + '_' + str(self.data[index]) + '_mean.npy')-m1)
        var = abs(np.load(self.path + '_' +  str(self.data[index]) + '_variance.npy')-s1)
        feature = np.load(self.path + '_' + str(self.data[index]) + '_feature.npy')
        # mean = np.load(self.path + '_' + str(self.data[index]) + '_mean.npy')
        # var = np.load(self.path + '_' +  str(self.data[index]) + '_variance.npy')



        target = self.label_file[index]
        fid = self.fid[index]
        fid = torch.as_tensor(fid, dtype=torch.float).view(1)
        mean = torch.as_tensor(mean, dtype=torch.float)
        var = torch.as_tensor(var, dtype=torch.float).view(1, 64, 64)

        target = torch.as_tensor(target, dtype=torch.float)
        return var, mean, target, fid, feature

    def __len__(self):
        return len(self.data)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

