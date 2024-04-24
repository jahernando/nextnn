import random            as random
import numpy             as np

import matplotlib.pyplot as plt

import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import xyimg.datapred as dp

#----------------------
# Data preparation
#----------------------

from collections import namedtuple
GoCNNBox = namedtuple('GoCNNBox' , ['model', 'dataset', 'testdata', 'epochs', 'y', 'ypred'])


#-------------------
# data
#-------------------


class GoDataset(Dataset):

    def __init__(self, filename, labels):
        print('Input data ', filename)
        self.filename = filename
        self.labels   = labels
        odata = dp.load(filename)
        xs = np.array([odata.xdic[label] for label in labels])
        xs = np.swapaxes(xs, 0, 1)
        self.xs  = xs
        print('x shape ', self.xs.shape)
        self.ys  = odata.y
        print('y shape ', self.ys.shape)
        self.ids = odata.id

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        xi = self.xs[idx]
        yi = self.ys[idx]
        xi = torch.tensor(xi, dtype = torch.float) # Add channel dimension
        # if (len(self.labels) == 1): xi.unsqueeze(0) #TODO
        yi = torch.tensor(yi, dtype = torch.float)
        return xi, yi


def test_godataset(ifilename):

    odata  = dp.load(ifilename)

    def _test(labels):
        dataset = GoDataset(ifilename, labels)
        assert dataset.xs.shape[1] == len(labels)
        nsize = len(dataset)
        assert dataset.xs.shape[0] == nsize
        assert dataset.xs.shape[0] == nsize
        i = random.choice(range(nsize))
        for j, label in enumerate(labels):
            assert np.all(odata.xdic[label][i] == dataset.xs[i, j])
        assert odata.y[i] == dataset.ys[i]
    
    _test(['esum', 'ecount'])
    _test(['esum',])
    return True


def subsets(dataset, fractions = (0.7, 0.2), batch_size = 200, shuffle = False):
    nsize = len(dataset)
    index = [int(i*nsize) for i in fractions]

    train_ = torch.utils.data.Subset(dataset, range(0, index[0]))
    train  = DataLoader(train_, batch_size = batch_size, shuffle = shuffle)

    test_  = torch.utils.data.Subset(dataset,range(index[0], index[0] + index[1]) )
    test   = DataLoader(test_, batch_size = index[1], shuffle = shuffle)
    
    val_   = torch.utils.data.Subset(dataset, range(index[0] + index[1], nsize))
    val    = DataLoader(val_, batch_size = batch_size, shuffle = shuffle)
    
    return train, test, val


#---------------------
# Model
#----------------------


class GoCNN(nn.Module):
    """ Starts from a (8, 8) image, process 3 convolutional layers and a linear layer to a unique output
    """

    def __init__(self, n_depth, n_width):
        super().__init__()
        m1, k1, p1 = 2 * n_depth, int(n_width/2)+1, 0
        m2, k2, p2 = 2 * m1, int(n_width/4) + 1, 0
        m3, k3, p3 = 2 * m2, int(n_width/8) + 1, 0
        self.debug  = True
        self.conv1  = nn.Conv2d(n_depth, m1, k1, padding = p1)
        self.bn1    = nn.BatchNorm2d(m1)
        self.conv2  = nn.Conv2d(m1, m2, k2, padding = p2)
        self.bn2    = nn.BatchNorm2d(m2)
        self.conv3  = nn.Conv2d(m2, m3, k3, padding = p3)
        self.bn3    = nn.BatchNorm2d(m3)
        self.pool   = nn.MaxPool2d(2, 2)
        n_out = n_width - (k1+k2+k3) + 2*(p1+p2+p2) + 3
        self.fc0    = nn.Linear(n_out * n_out * m3, m1)
        self.fc1    = nn.Linear(m1, 1)

    def forward(self, x):
        if (self.debug): print(x.size())
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        if (self.debug): print(x.size())
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        if (self.debug): print(x.size())
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        if (self.debug): print(x.size())
        x = x.flatten(start_dim=1)
        if (self.debug): print(x.size())
        #x = self.drop(x)
        x = self.fc0(x)
        if (self.debug): print(x.size())
        x = self.fc1(x)
        self.debug = False
        return x


#--------------------
# Fit
#---------------------


def chi2_loss(ys_pred, ys):
    squared_diffs = (ys_pred - ys) ** 2
    return squared_diffs.mean()

def in_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
        return x
    return x


def _training(model, optimizer, train):
    losses = []
    for xs, ys in train:
        xs = in_cuda(xs)
        ys = in_cuda(ys)
        model.train()
        optimizer.zero_grad()
        ys_pred = model(xs)
        loss    = chi2_loss(ys_pred, ys)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())
    return losses

def _validation(model, val):
    losses = []
    with torch.no_grad():
        model.eval()
        for xs, ys in val:
            xs, ys = in_cuda(xs), in_cuda(ys)
            ys_pred = model(xs)
            loss    = chi2_loss(ys_pred, ys)
            losses.append(loss.data.item())
    return losses


def _epoch(model, optimizer, train, val):

    losses_train = _training(model, optimizer, train)
    losses_val   = _validation(model, val)

    _sum = lambda x: (np.mean(x), np.std(x))

    sum =  (_sum(losses_train), _sum(losses_val))

    print('epoch ', sum)
    return sum 

def train_model(model, optimizer, train, val, nepochs = 20):

    sums = [_epoch(model, optimizer, train, val) for i in range(nepochs)]
    return sums
    

#---------------------
# Prediction
#-----------------------

def prediction(model, test):
    with torch.no_grad():
        model.eval()
        for xs, ys in test:
            xs, ys = in_cuda(xs), in_cuda(ys)
            ys_pred = model(xs)
    return ys.numpy(), ys_pred.numpy()

def roc_vals(ys, ys_pred):
    y0, _  = np.histogram(ys_pred[ys == 0], bins = 100, range = (-1.5, 1.5), density = True) 
    y1, _  = np.histogram(ys_pred[ys == 1], bins = 100, range = (-1.5, 1.5), density = True) 
    y0c    = np.cumsum(y0)/np.sum(y0)
    y1c    = 1. - np.cumsum(y1)/np.sum(y1)
    return y0c, y1c

#-------------
# Run
#-------------

def run(ifilename, labels, nepochs = 10, ofilename = ''):

    NNType = GoCNN

    dataset = GoDataset(ifilename, labels)
    train, test, val = subsets(dataset)
    n_depth, n_width, _ = dataset.xs[0].shape

    model = NNType(n_depth, n_width)
    model = in_cuda(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = train_model(model, optimizer, train, val, nepochs = nepochs)

    ys, ysp = prediction(model, test)

    if (ofilename != ''):
        np.savez(ofilename, epochs = epochs, ys = ys, ysp = ysp)

    return GoCNNBox(model, dataset, test, epochs, ys, ysp)

#-------------------
# Plot
#--------------------

def plot_epochs(epochs):
    us  = [sum[0][0] for sum in epochs]
    eus = [sum[0][1] for sum in epochs]
    vs  = [sum[1][0] for sum in epochs]
    evs = [sum[1][1] for sum in epochs]
    plt.errorbar(range(len(us)), us, yerr = eus, alpha = 0.5, label = "train")
    plt.errorbar(0.1+np.arange(len(vs)), vs, yerr = evs, alpha = 0.5, label = "val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend();

def plot_roc(ys, ysp, zoom = 0.5):
    plt.figure()
    plt.subplot(2, 2, 1)
    xrange = np.min(ysp), 1.1 * np.max(ysp)
    plt.hist(ysp[ys == 0], 100, range = xrange, density = True, alpha = 0.5, label = 'bkg')
    plt.hist(ysp[ys == 1], 100, range = xrange, density = True, alpha = 0.5, label = 'signal');
    plt.xlabel('output'); plt.ylabel('pdf'); plt.legend();
    plt.subplot(2, 2, 2)
    y0c, y1c = roc_vals(ys, ysp)
    plt.plot(y0c, y1c); plt.grid(); 
    plt.xlim((zoom, 1.))
    plt.xlabel('rejection'); plt.ylabel("efficiency"); 
    plt.tight_layout()

#--------
# Tests
#---------

# def test_load_godata():

#     pressure = '13bar'
#     width    = 6
#     labels   =['esum', 'ecount', 'emean']

#     odata = load_godata(pressure, width, labels)

#     def _test(y):
#         nsig = np.sum(y == 1)
#         nbkg = np.sum(y == 0)
#         assert (nsig >0) & (nbkg > 0)

#     _test(odata.y)
#     nsize = len(odata.y)
#     _test(odata.y[0:int(nsize/4)])
#     _test(odata.y[-int(nsize/4):])

#     for evt in range(min(10, nsize)):
#         esum   = odata.xdic['esum'][evt]
#         emean  = odata.xdic['ecount'][evt]
#         ecount = odata.xdic['emean'][evt]
#         assert np.all(np.isclose(esum, emean * ecount))

#     for evt in range(min(10, nsize)):
#         next  = np.sum(odata.zdic['ext'][evt] > 0)
#         assert (next >= 1) & (next <= 2)

#     return True

