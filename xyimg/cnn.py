import random            as random
import numpy             as np
import pandas            as pd

import matplotlib.pyplot as plt

import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import xyimg.dataprep as dp

#----------------------
#  General
#----------------------

from collections import namedtuple
GoCNNBox = namedtuple('GoCNNBox' , ['model', 'dataset', 'epochs', 'index', 'y', 'yp'])


#-------------------
# data
#-------------------

def _xs(dic, labels):
    """ select from the dictionary the list of arrays with labels with m-size
        and convert them into a numpy array with first dimension has m-size.
    """
    xs = np.array([dic[label] for label in labels])
    xs = np.swapaxes(xs, 0, 1)
    return xs

        

class GoDataset(Dataset):

    def __init__(self, filename, labels):
        self.filename = filename
        self.labels   = labels
        odata         = dp.load(filename)
        self.ys       = odata.y
        self.ids      = odata.id

        self.xs                = self._get_xs(odata, labels)
        self.zs, self.zlabels  = self._get_zs(odata)

    def _get_xs(self, odata, labels):
        return _xs(odata.xdic, labels)

    def _get_zs(self, odata):
        zlabels = list(odata.zdic.keys())
        return _xs(odata.zdic, zlabels), zlabels
        
    def filter(self, mask):
        self.xs   = self.xs [mask]
        self.zs   = self.zs [mask]
        self.ys   = self.ys [mask]
        self.ids  = self.ids[mask]
        self.mask = self.mask

    def __str__(self):
        s  = 'Dataset : \n'
        s += '   labels   : ' + str(self.labels)   + '\n'
        s += '   x shape  : ' + str(self.xs.shape) + '\n'
        s += '   y shape  : ' + str(self.ys.shape) + '\n'
        s += '   z labels : ' + str(self.zlabels)  + '\n'
        s += '   z shape  : ' + str(self.zs.shape) + '\n'
        return s

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        xi = self.xs[idx]
        yi = self.ys[idx]
        xi = torch.tensor(xi, dtype = torch.float) # Add channel dimension
        # if (len(self.labels) == 1): xi.unsqueeze(0) #TODO
        yi = torch.tensor(yi, dtype = torch.float)
        return xi, yi

class GoDatasetInv(GoDataset):
    """ Swap reco <-> mc images and now mc images are used a seed (x) for NN
    """

    #def __init__(self, filename, labels):
    #    super().__init__(filename, labels)

    def _get_xs(self, odata, labels):
        return _xs(odata.zdic, labels)
    
    def _get_zs(self, odata):
        zlabels = list(odata.xdic.keys())
        return _xs(odata.xdic, zlabels), zlabels

class GoDatasetTest(GoDataset):
    """ create test-seed (x) using the true MC information
    """

    def __init__(self, filename, labels):
        super().__init__(filename, labels)
        self.filter(self.mask)

    def _get_xs(self, odata, labels):
        assert len(labels) == 1
        label = labels[0]
        seg = odata.zdic['seg']
        ext = odata.zdic['ext']
        y   = odata.y
        tt  = [dp.ttimage(seg[i], ext[i], y[i]) for i in range(len(y))]
        dd = {label : tt}
        self.xs = _xs(dd, labels)
        mask = [bool(dp.good_ttimage(seg[i], ext[i], y[i], True)) for i in range(len(y))]
        self.mask = mask
        return _xs(dd, labels)
    
    


class GoDataset3D(GoDataset):
    """ special class for 3D images, that are (x, x, y), and they are converted in (x, y) with x-depth images
    """

    def __init__(self, filename, labels):
        super().__init__(filename, labels)
        assert len(self.labels)   == 1  # only on1 3D image allowed
        assert len(self.xs.shape) == 5  # checl that there are 3D images
        xxs     = np.swapaxes(self.xs, 1, 4)
        self.xs = np.squeeze(xxs, 4)
        zzs     = np.swapaxes(self.zs, 3, 4)
        self.zs = np.swapaxes(zzs, 2, 3)

dataset = {'GoDataset'      : GoDataset, 
           'GoDatasetInv'   : GoDatasetInv,
           'GoDataset3D'    : GoDataset3D,
           'GoDatasetTest'  : GoDatasetTest}

#----- PyTorch data operations


def _index(nsize, fractions):
    index = [int(i*nsize) for i in fractions]
    index[1] = sum(index)
    return index

def subsets(dataset, fractions = (0.7, 0.2), batch_size = 200, shuffle = False):
    nsize = len(dataset)
    index = _index(nsize, fractions)

    train_ = torch.utils.data.Subset(dataset, range(0, index[0]))
    train  = DataLoader(train_, batch_size = batch_size, shuffle = shuffle)

    test_  = torch.utils.data.Subset(dataset,range(index[0], index[1]) )
    test   = DataLoader(test_, batch_size = index[1], shuffle = shuffle)
    
    val_   = torch.utils.data.Subset(dataset, range(index[1], nsize))
    val    = DataLoader(val_, batch_size = batch_size, shuffle = shuffle)
    
    return train, test, val, index

def full_sample(dataset):
    nsize = len(dataset)
    sample_ = torch.utils.data.Subset(dataset, range(0, nsize))
    sample  = DataLoader(sample_, batch_size = 1, shuffle = False)
    return sample


#---------------------
# Model
#----------------------


class GoCNN(nn.Module):
    """ A simple binary classification CNN starting from a (n_width, n_widht, n_depth) 
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
        def _sshape(x):
            return str(x.size())[11: -1]
        if (self.debug): s = 'CNN : \n   ' + _sshape(x) 
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        if (self.debug): s = s + ' > ' + _sshape(x) 
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        if (self.debug): s = s + ' > ' + _sshape(x) 
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        if (self.debug): s = s + ' > ' + _sshape(x) 
        x = x.flatten(start_dim=1)
        if (self.debug): s = s + ' > ' + _sshape(x) 
        #x = self.drop(x)
        x = self.fc0(x)
        if (self.debug): s = s + ' > ' + _sshape(x) + '\n'
        x = self.fc1(x)
        if (self.debug): s = s + ' > ' + _sshape(x) + '\n'
        if (self.debug): print(s)
        self.debug = False
        return x


#--------------------
# Fit
#---------------------


def chi2_loss(ys_pred, ys):
    squared_diffs = (ys_pred - ys) ** 2
    return squared_diffs.mean()

def chi2weight_loss(ys_pred, ys):
    w0, w1 = 1., 0.5 
    ydelta = ys_pred - ys
    ydelta2 = ydelta * ydelta
    ydelta2[ys == 0] = ydelta2[ys == 0]/w0**2
    ydelta2[ys == 1] = ydelta2[ys == 1]/w1**2
    return  ydelta2.mean()

def tstat_loss(ys_pred, ys):
    ydelta = ys_pred - ys
    ysig   = 2.*ys - 1.
    yval   = ydelta * ysig
    yval2  = yval * yval
    yval2[yval > 0.] = 1e-6
    return  yval2.mean()

def tsig_loss(ys_pred, ys):
    ydelta  = ys_pred - ys
    yref0   = torch.quantile(ys_pred[ys == 0], 0.5)
    yref1   = torch.quantile(ys_pred[ys == 1], 0.5)
    ydelta0 = ys_pred - yref0
    ydelta1 = ys_pred - yref1
    ydelta2  = ydelta0
    ydelta2[ys == 1] = ydelta1[ys == 1]
    ypm     = 2*ys - 1
    ysig    = ydelta2 * ypm
    yval2  = ydelta * ydelta
    yval2[ysig > 0.] = 1e-6
    return  yval2.mean()


loss_function = chi2_loss

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
        loss    = loss_function(ys_pred, ys)
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
            loss    = loss_function(ys_pred, ys)
            losses.append(loss.data.item())
    return losses


def _epoch(model, optimizer, train, val):

    losses_train = _training(model, optimizer, train)
    losses_val   = _validation(model, val)

    _sum = lambda x: (np.mean(x), np.std(x))

    sum =  (_sum(losses_train), _sum(losses_val))

    print('Epoch:  train {:3.3f} +- {:3.3f}  validation {:3.3f} +- {:3.3f}'.format(*sum[0], *sum[1]))
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


#-------------
# Run
#-------------


def run(dataset, nepochs = 10, ofilename = ''):

    NNType = GoCNN
    print(dataset)

    train, test, val, index = subsets(dataset)
    assert len(dataset.xs.shape) == 4
    n_depth, n_width, _ = dataset.xs[0].shape
    print('Event Image tensor ', dataset.xs[0].shape)

    model     = NNType(n_depth, n_width)
    model     = in_cuda(model)
    learning_rate = 0.001 # default (tested 0.01, 0.0001 with no improvements)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    epochs    = train_model(model, optimizer, train, val, nepochs = nepochs)

    ys, yps = prediction(model, test)
    ybin    =  yps >= 0.5
    acc = 100.*np.sum(ys == ybin)/len(ys)
    print('Test  accuracy {:4.2f}'.format(acc))

    if (ofilename != ''):
        print('save cnn results at ', ofilename)
        np.savez(ofilename, epochs = epochs, index = index, y = ys, yp = yps)

    return GoCNNBox(model, dataset, epochs, index, ys, yps)

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

#-----------
# Ana
#------------

def roc_vals(y, yp):
    y0, _  = np.histogram(yp[y == 0], bins = 100, range = (-1.5, 1.5), density = True) 
    y1, _  = np.histogram(yp[y == 1], bins = 100, range = (-1.5, 1.5), density = True) 
    y0c    = np.cumsum(y0)/np.sum(y0)
    y1c    = 1. - np.cumsum(y1)/np.sum(y1)
    return y0c, y1c

def to_df(index, y, yp):
    ids = np.arange(*index)
    y   = y.flatten()
    yp  = yp.flatten()
    df = pd.DataFrame({'ids':ids, 'y': y, 'yp':yp})
    return df

def roc_value(y, yp, epsilon = 0.9):
    yp0   = np.percentile(yp[y==0], 100 * epsilon) 
    seff  = np.sum(yp[y==1] >= yp0)/len(yp[y==1])
    return yp0, seff

def false_positives_indices(y, yp, yp0, index0 = 0):
    ids = index0 + np.argwhere(np.logical_and(y == 0, yp >= yp0)).flatten()
    return ids


#--------
# Tests
#---------

def test_godataset(ifilename, labels, DSet = GoDataset):

    odata  = dp.load(ifilename)

    def _test(labels):
        dataset = DSet(ifilename, labels)
        assert dataset.xs.shape[1] == len(labels)
        nsize = len(dataset)
        assert dataset.xs.shape[0] == nsize
        xdic = odata.xdic if DSet == GoDataset else odata.zdic
        #zdic = odata.zdic if DSet == GoDataset else odata.xdic
        i = random.choice(range(nsize))
        for j, label in enumerate(labels):
            assert np.all(xdic[label][i] == dataset.xs[i, j])
        assert odata.y[i] == dataset.ys[i]
        assert dataset.zs.shape[0] == nsize
        assert dataset.zs.shape[1] == len(dataset.zlabels)
    
    _test(labels)
    _test((labels[0],))
    return True


def test_box_index(box):
    index = box.index

    ys  = np.array(box.dataset[index[0]:index[1]][1], dtype = int).flatten()
    ys0 = box.dataset.ys[index[0]:index[1]].flatten()
    ys1 = np.array(box.y, dtype = int).flatten()
    assert np.all(ys == ys0)
    assert np.all(ys == ys1)
    return True

def test_box_save(box, ofile):
    cnndata = np.load(ofile)

    assert np.all(np.array(box.epochs) ==  cnndata['epochs'])
    assert np.all(box.y == cnndata['y'])
    assert np.all(box.yp == cnndata['yp'])
    assert np.all(np.array(box.index) == cnndata['index'])
    return True


def test(ifilename):

    print('input filename ', ifilename)
    ofilename = dp.prepend_filename(ifilename, 'test_cnn')
    print('output filename ', ofilename)

    # print('--- data set ---')
    # labels = ['esum', 'emax', 'ecount']
    # test_godataset(ifilename, labels)
    # dset = GoDataset(ifilename, labels)
    # box = run(dset, ofilename = ofilename, nepochs = 4)
    # test_box_index(box)
    # test_box_save(box, ofilename)

    # print('--- data set inv ---')
    # labels = ['seg', 'ext']
    # test_godataset(ifilename, labels, GoDatasetInv)
    # dset = GoDatasetInv(ifilename, labels)
    # box = run(dset, ofilename = ofilename, nepochs = 4)
    # test_box_index(box)
    # test_box_save(box, ofilename)

    print('--- data set test ---')
    labels = ['test']
    #test_godataset(ifilename, labels, GoDatasetTest)
    dset = GoDatasetTest(ifilename, labels)
    box  = run(dset, ofilename = ofilename, nepochs = 4)
    test_box_index(box)
    test_box_save(box, ofilename)

#    print('input filename ', ifilename_xyz)
#    ofilename = dp.prepend_filename(ifilename_xyz, 'test_cnn')
#    print('output filename ', ofilename)
#
#    #test_godataset(ifilename, labels)
#    labels = ['esum',]
#    dset   = GoDataset3DImg(ifilename, labels)
#    box    = run(dset, ofilename = ofilename, nepochs = 2)
#    test_box_index(box)
#    test_box_save(box, ofilename)

    return True