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

import xyimg.utils    as ut 
import xyimg.dataprep as dp

#----------------------
#  General
#----------------------

from collections import namedtuple
CNNResult = namedtuple('CNNResult' , ['model', 'dataset', 'epochs', 'index', 'y', 'yp'])


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
        odata         = dp.godata_load(filename)
        self.y        = odata.y[:, np.newaxis]
        #self.y        = odata.y
        self.id       = odata.id

        ok = [(label in odata.xdic.keys()) for label in labels]
        ok = (np.sum(ok) == len(ok))
        xdic = odata.xdic if ok else odata.zdic
        self.x = _xs(xdic, labels)
        
    def filter(self, mask):
        self.x   = self.x [mask]
        self.y   = self.y [mask]
        self.id  = self.id[mask]
        self.mask = self.mask

    def __str__(self):
        s  = 'Dataset : \n'
        s += '   labels   : ' + str(self.labels)   + '\n'
        s += '   x shape  : ' + str(self.x.shape) + '\n'
        s += '   y shape  : ' + str(self.y.shape) + '\n'
        return s

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xi = self.x[idx]
        yi = self.y[idx]
        xi = torch.tensor(xi, dtype = torch.float) # Add channel dimension
        # if (len(self.labels) == 1): xi.unsqueeze(0) #TODO
        yi = torch.tensor(yi, dtype = torch.float)
        return xi, yi

#----- PyTorch data operations


def _index(nsize, fractions):
    index = [int(i*nsize) for i in fractions]
    index[1] = sum(index)
    return index

def subsets(dataset, fractions = (0.7, 0.2), batch_size = 100, shuffle = False):
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
# Models
#----------------------

    
class KCNN(nn.Module):
    """
    Defines a convolutional network with a basic architecture:
    4 - Conv layers that increased the depth 
    convolution (3x3) , reLU batch norm and MaxPool,
    drop (optional) 
    linear layer 512 => 2

    Input to the network are the pixels of the pictures, output (x,y,z)

    """
    def __init__(self, depth, width, kernel = 3, expansion = 2, padding = 1,
                 pool = 2, dropout_fraction = 0.1, use_sigmoid = False):
        super().__init__()
        chi          = depth * expansion
        self.dropout = dropout_fraction > 0
        self.conv1   = nn.Conv2d(depth, chi, kernel, padding = padding) 
        self.bn1     = nn.BatchNorm2d(chi)
        self.conv2   = nn.Conv2d(chi, chi*2, kernel, padding = padding)
        self.bn2     = nn.BatchNorm2d(chi*2)
        self.conv3   = nn.Conv2d(chi*2, chi*4, kernel, padding = padding)
        self.bn3     = nn.BatchNorm2d(chi*4)
        self.conv4   = nn.Conv2d(chi*4, chi*8, kernel, padding = padding)
        self.bn4     = nn.BatchNorm2d(chi*8)
        self.pool    = nn.MaxPool2d(pool, pool)
        self.fc0     = nn.Linear(chi*8, 1)
        #self.bnp     = nn.BatchNorm1d(2 * depth)
        #self.fc1     = nn.Linear(2 * depth, 1)
        self.drop1   = nn.Dropout(p=dropout_fraction)
        self.smoid   = nn.Sigmoid()

        self.use_sig = use_sigmoid
        self.debug   = True

    def forward(self, x):

        if(self.debug) : print(f"input data shape =>{x.shape}")
        # convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (8,8,64)
        x = self.pool(self.bn1(F.leaky_relu(self.conv1(x))))

        if(self.debug) : print(f"conv 1 =>{x.shape}")
        x = self.pool(self.bn2(F.leaky_relu(self.conv2(x))))
        
        if (self.debug) : print(f"conv 2 =>{x.shape}")
        x = self.pool(self.bn3(F.leaky_relu(self.conv3(x))))
        
        if (self.debug) : print(f"conv 3 =>{x.shape}")
        x = self.pool(self.bn4(F.leaky_relu(self.conv4(x))))
        
        if (self.debug) : print(f"conv 4 =>{x.shape}")
        x = x.flatten(start_dim=1)
        # Flatten

        if (self.debug): print(f"lin input =>{x.shape}")
        if self.dropout: x = self.drop1(x)  # drop
        
        #x = self.bnp(self.fc0(x))
        #if (self.debug): print(f"lin 0 =>{x.shape}")

        #x = x[:, 1] - x[:, 0]
        #x = x[:, np.newaxis]
        x = self.fc0(x)
        if (self.debug): print(f"lin 0 =>{x.shape}")

        if (self.use_sig): x = self.smoid(x)
        #x -= x.min()
        #x /= x.max()
        #x = x.squeeze(1)

        self.debug = False

        return x


def conv_dimensiones(width, depth, filters = 3, kernel = 3, stride = 1, padding = 0, pool = 1):

        depth = filters * depth
        width = ((width - kernel + 2 * padding)/stride + 1)/pool
        return width, depth


class TestGoCNN(nn.Module):
    """ A simple binary classification CNN starting from a (n_width, n_widht, n_depth) 
    """

    # WARNING: you always have to set a layer in the self!
    def __init__(self, depth, width, kmax = 20, kfactor = 2, padding = 0):
        super().__init__()
    
        self.debug = True
        self.flow  = []

        def _add_next_conv(m, n, i):
            k = max(min(int(n/kfactor) + 1, kmax), 2)
            m0, n0 = m, n
            m, n   = kfactor * m, n - k + 1
            i      = i +1
            if (n <= 0): return m0, n0, i, True
            conv = nn.Conv2d(m0, m, k, padding = padding)
            bn   = nn.BatchNorm2d(m)
            setattr(self, 'conv'+str(i), conv)
            setattr(self, 'bn'+str(i), bn)
            self.flow.append(conv)
            self.flow.append(F.leaky_relu)
            self.flow.append(bn)
            print('conv : ', i, ' init ', (m0, n0), ', next ', (m, n), ', kernel ', k)
            return m, n, i, False

        # convolutions
        m, n, i, stop = depth, width, 0, False
        while not stop:
            m, n, i, stop = _add_next_conv(m, n, i)

        # linear
        ndim1 = m * n * n
        ndim2 = max(kfactor * depth, 2)
        print('lin  : init ', ndim1, ', next', ndim2, ', next ', 1)
        flat  = lambda x : x.flatten(start_dim = 1)
        fc1   = nn.Linear(ndim1, ndim2)
        setattr(self, 'fc1', fc1)
        fc2   = nn.Linear(ndim2, 1)
        setattr(self, 'fc1', fc2)
        smoid  = nn.Sigmoid()
        self.flow.append(flat)
        self.flow.append(fc1)
        self.flow.append(fc2)
        self.flow.append(smoid)

    def forward(self, x):

        def _sshape(x):
            si = str(x.size())[11: -1] + '-> '
            return si
        #if (self.debug): s = 'CNN: ' + _sshape(x)
        for op in self.flow:
            #if (self.debug):  s = s + _sshape(x)
            x = op(x)

        #if (self.debug):
        #    print(s)
        #    self.debug = False
        return x


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
        self.smoid  = nn.Sigmoid()
        n_out = n_width - (k1+k2+k3) + 2*(p1+p2+p2) + 3
        self.fc0    = nn.Linear(n_out * n_out * m3, m2)
        self.fc1    = nn.Linear(m2, 1)

    def forward(self, x):
        def _sshape(x):
            si = str(x.size())[11: -1]
            print(si)
            return si
        if (self.debug): s = 'CNN : \n   ' + _sshape(x) 
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        if (self.debug): s = s + ' => ' + _sshape(x) 
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        if (self.debug): s = s + ' => ' + _sshape(x) 
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        if (self.debug): s = s + '=> ' + _sshape(x) 
        x = x.flatten(start_dim=1)
        if (self.debug): s = s + ' => ' + _sshape(x) 
        #x = self.drop(x)
        x = self.fc0(x)
        if (self.debug): s = s + ' => ' + _sshape(x) + '\n'
        x = self.smoid(self.fc1(x))
        #x = self.fc1(x)
        if (self.debug): s = s + ' => ' + _sshape(x) + '\n'
        if (self.debug): print(s)
        self.debug = False
        return x


#--------------------
# CNN Fit
#---------------------

config = {'loss_function' : 'MSELoss',
          'learning_rate' : 0.001} # default 0.001


loss_functions = {'MSELoss' : nn.MSELoss(),    # regression (chi2)
                  'BCELoss' : nn.BCELoss(),    # Binary classification (output must be 0-1)
                  'CrossEntropyLoss'  : nn.CrossEntropyLoss()  # n-classification
                  }
                  #'chi2'    : chi2_loss}


def chi2_loss(ys_pred, ys):
    squared_diffs = (ys_pred - ys) ** 2
    return squared_diffs.mean()

def in_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
        return x
    return x

def _training(model, optimizer, train, loss_function):
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

def _validation(model, val, loss_function):
    losses = []
    with torch.no_grad():
        model.eval()
        for xs, ys in val:
            xs, ys = in_cuda(xs), in_cuda(ys)
            ys_pred = model(xs)
            loss    = loss_function(ys_pred, ys)
            losses.append(loss.data.item())
    return losses


def _epoch(model, optimizer, train, val, loss_function):

    losses_train = _training(model, optimizer, train, loss_function)
    losses_val   = _validation(model, val, loss_function)

    _sum = lambda x: (np.mean(x), np.std(x))
    sum  = (_sum(losses_train), _sum(losses_val))
    acc  = accuracy(model, val)

    print('Epoch:  train {:1.2e} +- {:1.2e}  validation {:1.2e} +- {:1.2e}'.format(*sum[0], *sum[1]))
    print('        accuracy {:4.2f}'.format(acc))

    return sum 


def train_model(model, optimizer, train, val, loss_function, nepochs = 20):

    sums = [_epoch(model, optimizer, train, val, loss_function) for i in range(nepochs)]
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


def accuracy(model, val, y0 = 0.5):
    ys, yps   = prediction(model, val)
    ybin      = yps >= y0
    acc       = np.sum(ys == ybin)/len(ys)
    return acc

#===========================
# Run
#===========================


def run(dataset, model, nepochs = 10, ofilename = '', config = config):

    print(dataset)
    print(config)
    loss_function = loss_functions[config['loss_function']]

    train, test, val, index = subsets(dataset)
    assert len(dataset.x.shape) == 4
    print('Event Image sample : ', dataset.x.shape)

    model     = in_cuda(model)
    learning_rate = config['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    epochs    = train_model(model, optimizer, train, val, loss_function, nepochs = nepochs)

    ys, yps = prediction(model, test)

    if (ofilename != ''):
        print('save cnn results at ', ofilename)
        np.savez(ofilename, epochs = epochs, index = index, y = ys, yp = yps)

    return CNNResult(model, dataset, epochs, index, ys, yps)

#---------------------
# Production
#---------------------

def cnn_config_name(cnnname, config):

    labels    = config['labels']
    expansion = 'f'+str(int(config['expansion']))
    nepochs   = 'e'+str(int(config['nepochs']))
    eloss     = str(config['loss_function'])

    slabels = ut.str_concatenate(labels, '+')
    sconfig = ut.str_concatenate((expansion, nepochs), '')
    ss = ut.str_concatenate((cnnname, slabels, sconfig, eloss),'_')
    return ss


def filename_cnn(ifilename, cnnname, config):
    """ return the formated data files for cnn-input and cnn-output
    """
    fname   = ifilename.split('.')[0]
    sname   = cnn_config_name(cnnname, config)
    ofile   = ut.str_concatenate((fname, sname)) +  '.npz'
    return ofile

def production(ifile, ofile, config):
    
    print('input file  : ', ifile)
    print('output file : ', ofile)
    print('config      : ', config)

    labels    = config['labels']
    expansion = config['expansion']
    eloss     = config['loss_function']
    nepochs   = config['nepochs']

    print('loading data ', ifile)
    idata  = GoDataset(ifile, labels)
    print('Input shape ', idata.x.shape)
    _, depth, width, _ = idata.x.shape
    kernel = 3
    print('configurate cnn (kernel, expansion)', kernel, expansion)
    use_sigmoid = eloss = 'BCELoss'
    kcnn = KCNN(depth, width, expansion = expansion, kernel = kernel, use_sigmoid = use_sigmoid)
    print('run cnn (epochs) ', nepochs)
    rcnn = run(idata, kcnn, ofilename = ofile, nepochs = nepochs, config = config)
    return rcnn 


#--------------------
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


def plot_event(idata, odata, labels, zlabels = [], ievt = -1):
    i0   = odata['index'][0]
    size = odata['index'][1] - i0
    ievt = int(np.random.choice(size, 1)) if ievt == -1 else ievt
    kevt = i0 + ievt
    print('event ', kevt)
    y0   = idata.ys[kevt]
    yt0  = odata['y'][ievt]
    ytp  = odata['yp'][ievt]
    print('target test      ', int(y0))
    assert int(y0) == int(yt0)
    print('target test pred ', float(ytp))
    success = (ytp >= 0.5) == y0
    print('success          ', bool(success))
    dp.plot_imgs(idata.xs, kevt, labels)
    for i, label in enumerate(labels):
        print('total    ', label, np.sum(idata.xs[kevt][i]))
    dp.plot_imgs(idata.zs, kevt, zlabels)
    return

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


test_xlabel = ['xy_E_sum']
test_zlabel = ['zy_segclass_mean']

def test(ifilename, xlabels = test_xlabel, zlabels = test_zlabel):

    print('input filename ', ifilename)
    ofilename = dp.prepend_filename(ifilename, 'test_cnn')
    print('output filename ', ofilename)

    print('--- data set ---')
    test_godataset(ifilename, xlabels)
    dset = GoDataset(ifilename, xlabels)
    box  = run(dset, ofilename = ofilename, nepochs = 4)
    test_box_index(box)
    test_box_save(box, ofilename)

    print('--- data set inv ---')
    test_godataset(ifilename, xlabels, GoDatasetInv)
    dset = GoDatasetInv(ifilename, zlabels)
    box  = run(dset, ofilename = ofilename, nepochs = 4)
    test_box_index(box)
    test_box_save(box, ofilename)

    # print('--- data set test ---')
    # labels = ['test']
    # #test_godataset(ifilename, labels, GoDatasetTest)
    # dset = GoDatasetTest(ifilename, labels)
    # box  = run(dset, ofilename = ofilename, nepochs = 4)
    # test_box_index(box)
    # test_box_save(box, ofilename)

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