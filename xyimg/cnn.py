import random            as random
import glob              as glob
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
CNNResult = namedtuple('CNNResult' , ['model', 'dataset', 'losses', 'accuracies', 'index', 'y', 'yp'])



#-------------------
# data
#-------------------


class ImgDataset(Dataset):

    def __init__(self, imgdispatch, transform = None):
        self.imgdispatch = imgdispatch
        self.transform   = transform 
        
    def __len__(self):
        return len(self.imgdispatch)

    def __getitem__(self, index):
        x, y = self.imgdispatch[index]
        x = x if self.transform == None else self.transform(x)
        xi = torch.tensor(x, dtype = torch.float)
        yi = torch.tensor(y)
        return xi, yi


# def _xs(dic, labels):
#     """ select from the dictionary the list of arrays with labels with m-size
#         and convert them into a numpy array with first dimension has m-size.
#     """
#     xs = np.array([dic[label] for label in labels])
#     xs = np.swapaxes(xs, 0, 1)
#     return xs


# class GoDataset(Dataset):

#     def __init__(self, filename, labels, black = False, img_scale = 1.):
#         self.filename = filename
#         self.labels   = labels
#         odata         = dp.godata_load(filename)
#         self.y        = odata.y
#         #self.y        = odata.y
#         self.id       = odata.id

#         ok = [(label in odata.xdic.keys()) for label in labels]
#         ok = (np.sum(ok) == len(ok))
#         xdic = odata.xdic if ok else odata.zdic
#         self.x     = _xs(xdic, labels)
        
#         # creata a digital (white and black only image)
#         self.black = black
#         if (black):
#             print('black data ')
#             self.x[self.x > 0] = 1 

#         self.factor = img_scale
#         self.x     = img_scale * self.x
#         if (img_scale != 1.):
#             print('scale image by factor ', img_scale)
        
#     def filter(self, mask):
#         self.x   = self.x [mask]
#         self.y   = self.y [mask]
#         self.id  = self.id[mask]
#         self.mask = self.mask

#     def __str__(self):
#         s  = 'Dataset : \n'
#         s += '   labels   : ' + str(self.labels)   + '\n'
#         s += '   x shape  : ' + str(self.x.shape) + '\n'
#         s += '   y shape  : ' + str(self.y.shape) + '\n'
#         s += '   black    : ' + str(self.black) + '\n'
#         s += '   scale    : ' + str(self.factor)
#         return s

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         xi = self.x[idx]
#         yi = int(self.y[idx])
#         xi = torch.tensor(xi, dtype = torch.float) # Add channel dimension
#         # if (len(self.labels) == 1): xi.unsqueeze(0) #TODO
#         yi = torch.tensor(yi)
#         return xi, yi

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
    4 - Conv layers that increased the depth and decreasing width
         convolution (3x3) , reLU batch norm and MaxPool,
    2 - linear with drop drop (optional) 
      - previous to last layer has 2-neurons for binnary classification
      - last layer depends on the loss function: options 'sigmoid, softmax'
      
    """
    def __init__(self, depth, width, kernel = 3, expansion = 2, padding = 1,
                 pool = 2, dropout_fraction = 0.1, noutput = 2):
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
        self.fc0     = nn.Linear(chi*8, noutput)
        self.drop1   = nn.Dropout(p=dropout_fraction)
        self.noutput = noutput
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
        
        x = self.fc0(x)
        if (self.debug): print(f"lin 0 =>{x.shape}")

        if (self.noutput == 1): 
            x = torch.squeeze(x)
            if (self.debug): print(f"lin 0 =>{x.shape}")

        self.debug = False

        return x

class HCNN(nn.Module):
    """
    Defines a convolutional network with a basic architecture:
    4 - Conv layers that increased the depth and decreasing width half-width kernel
         convolution (HW kernel) , reLU batch norm,
    2 - linear with drop drop (optional) 
      - previous to last layer has 2-neurons for binnary classification
      - last layer depends on the loss function: options 'sigmoid, softmax'
      
    """
    def __init__(self, depth, width, kernel = 3, expansion = 2, padding = 1, dropout_fraction = 0.1, noutput = 2):
        super().__init__()
        chi          = depth * expansion
        self.dropout = dropout_fraction > 0
        
        ikernel = lambda width: max(int(width/2), 2)
        iwitdh  = lambda width, k: width - k + 2 * padding + 1

        kernel = ikernel(width)
        print('int conv : width = ', width)
        print('1st conv : width = ', width, ', kernel = ', kernel)
        self.conv1   = nn.Conv2d(depth, chi, kernel, padding = padding) 
        self.bn1     = nn.BatchNorm2d(chi)
        
        width        = iwitdh(width, kernel)
        kernel       = ikernel(width)
        print('2nd conv : width = ', width, ', kernel = ', kernel)
        self.conv2   = nn.Conv2d(chi, chi*2, kernel, padding = padding)
        self.bn2     = nn.BatchNorm2d(chi*2)

        width        = iwitdh(width, kernel)
        kernel       = ikernel(width)
        print('3rd conv : width = ', width, ', kernel = ', kernel)
        self.conv3   = nn.Conv2d(chi*2, chi*4, kernel, padding = padding)
        self.bn3     = nn.BatchNorm2d(chi*4)

        width        = iwitdh(width, kernel)
        kernel       = ikernel(width)
        print('4th conv : width = ', width, ', kernel = ', kernel)
        self.conv4   = nn.Conv2d(chi*4, chi*8, kernel, padding = padding)
        self.bn4     = nn.BatchNorm2d(chi*8)

        width        = iwitdh(width, kernel)
        print('out conv : width = ', width)

        self.fc0     = nn.Linear(chi*8 * (width * width), noutput)
        #self.pool    = nn.MaxPool2d(pool, pool)
        self.drop1   = nn.Dropout(p=dropout_fraction)
        self.noutput = noutput
        self.debug   = True

    def forward(self, x):

        if(self.debug) : print(f"input data shape =>{x.shape}")
        # convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (8,8,64)
        x = self.bn1(F.leaky_relu(self.conv1(x)))

        if(self.debug) : print(f"conv 1 =>{x.shape}")
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        
        if (self.debug) : print(f"conv 2 =>{x.shape}")
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        
        if (self.debug) : print(f"conv 3 =>{x.shape}")
        x = self.bn4(F.leaky_relu(self.conv4(x)))
        
        if (self.debug) : print(f"conv 4 =>{x.shape}")
        x = x.flatten(start_dim=1)
        # Flatten

        if (self.debug): print(f"lin input =>{x.shape}")
        if self.dropout: x = self.drop1(x)  # drop
        
        x = self.fc0(x)
        if (self.debug): print(f"lin 0 =>{x.shape}")

        if (self.noutput == 1): 
            x = torch.squeeze(x)
            if (self.debug): print(f"lin 0 =>{x.shape}")

        self.debug = False

        return x


class HKCNN(nn.Module):
    """
    Defines a convolutional network with a basic architecture:
    4 - Conv layers that increased the depth and decreasing width half-width kernel
         convolution (HW kernel) , reLU batch norm,
    2 - linear with drop drop (optional) 
      - previous to last layer has 2-neurons for binnary classification
      - last layer depends on the loss function: options 'sigmoid, softmax'
      
    """
    def __init__(self, depth, width, kernel = 3, expansion = 2, padding = 1, dropout_fraction = 0.1, noutput = 2):
        super().__init__()
        chi          = depth * expansion
        self.dropout = dropout_fraction > 0
        
        ikernel = lambda width: max(int(width/2), 2)
        iwitdh  = lambda width, k: width - k + 2 * padding + 1

        kernel = ikernel(width)
        print('int conv : width = ', width)
        print('1st conv : width = ', width, ', kernel = ', kernel)
        self.conv1   = nn.Conv2d(depth, chi, kernel, padding = padding) 
        self.bn1     = nn.BatchNorm2d(chi)
        
        width        = iwitdh(width, kernel)
        kernel       = 3
        print('2nd conv : width = ', width, ', kernel = ', kernel)
        self.conv2   = nn.Conv2d(chi, chi*2, kernel, padding = padding)
        self.bn2     = nn.BatchNorm2d(chi*2)

        width        = iwitdh(width, kernel)
        print('3rd conv : width = ', width, ', kernel = ', kernel)
        self.conv3   = nn.Conv2d(chi*2, chi*4, kernel, padding = padding)
        self.bn3     = nn.BatchNorm2d(chi*4)

        width        = iwitdh(width, kernel)
        print('4th conv : width = ', width, ', kernel = ', kernel)
        self.conv4   = nn.Conv2d(chi*4, chi*8, kernel, padding = padding)
        self.bn4     = nn.BatchNorm2d(chi*8)

        width        = iwitdh(width, kernel)
        print('out conv : width = ', width)

        self.fc0     = nn.Linear(chi*8 * (width * width), noutput)
        
        pool         = 2
        self.pool    = nn.MaxPool2d(pool, pool)
        self.drop1   = nn.Dropout(p=dropout_fraction)
        self.noutput = noutput
        self.debug   = True

    def forward(self, x):

        if(self.debug) : print(f"input data shape =>{x.shape}")
        # convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (8,8,64)
        x = self.bn1(F.leaky_relu(self.conv1(x)))

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
        
        x = self.fc0(x)
        if (self.debug): print(f"lin 0 =>{x.shape}")

        if (self.noutput == 1): 
            x = torch.squeeze(x)
            if (self.debug): print(f"lin 0 =>{x.shape}")

        self.debug = False

        return x


def conv_dimensiones(width, depth, filters = 3, kernel = 3, stride = 1, padding = 0, pool = 1):

        depth = filters * depth
        width = ((width - kernel + 2 * padding)/stride + 1)/pool
        return width, depth


# class TestGoCNN(nn.Module):
#     """ A simple binary classification CNN starting from a (n_width, n_widht, n_depth) 
#     """

#     # WARNING: you always have to set a layer in the self!
#     def __init__(self, depth, width, kmax = 20, kfactor = 2, padding = 0):
#         super().__init__()
    
#         self.debug = True
#         self.flow  = []

#         def _add_next_conv(m, n, i):
#             k = max(min(int(n/kfactor) + 1, kmax), 2)
#             m0, n0 = m, n
#             m, n   = kfactor * m, n - k + 1
#             i      = i +1
#             if (n <= 0): return m0, n0, i, True
#             conv = nn.Conv2d(m0, m, k, padding = padding)
#             bn   = nn.BatchNorm2d(m)
#             setattr(self, 'conv'+str(i), conv)
#             setattr(self, 'bn'+str(i), bn)
#             self.flow.append(conv)
#             self.flow.append(F.leaky_relu)
#             self.flow.append(bn)
#             print('conv : ', i, ' init ', (m0, n0), ', next ', (m, n), ', kernel ', k)
#             return m, n, i, False

#         # convolutions
#         m, n, i, stop = depth, width, 0, False
#         while not stop:
#             m, n, i, stop = _add_next_conv(m, n, i)

#         # linear
#         ndim1 = m * n * n
#         ndim2 = max(kfactor * depth, 2)
#         print('lin  : init ', ndim1, ', next', ndim2, ', next ', 1)
#         flat  = lambda x : x.flatten(start_dim = 1)
#         fc1   = nn.Linear(ndim1, ndim2)
#         setattr(self, 'fc1', fc1)
#         fc2   = nn.Linear(ndim2, 1)
#         setattr(self, 'fc1', fc2)
#         smoid  = nn.Sigmoid()
#         self.flow.append(flat)
#         self.flow.append(fc1)
#         self.flow.append(fc2)
#         self.flow.append(smoid)

#     def forward(self, x):

#         def _sshape(x):
#             si = str(x.size())[11: -1] + '-> '
#             return si
#         #if (self.debug): s = 'CNN: ' + _sshape(x)
#         for op in self.flow:
#             #if (self.debug):  s = s + _sshape(x)
#             x = op(x)

#         #if (self.debug):
#         #    print(s)
#         #    self.debug = False
#         return x


# class GoCNN(nn.Module):
#     """ A simple binary classification CNN starting from a (n_width, n_widht, n_depth) 
#     """

#     def __init__(self, n_depth, n_width):
#         super().__init__()
#         m1, k1, p1 = 2 * n_depth, int(n_width/2)+1, 0
#         m2, k2, p2 = 2 * m1, int(n_width/4) + 1, 0
#         m3, k3, p3 = 2 * m2, int(n_width/8) + 1, 0
#         self.debug  = True
#         self.conv1  = nn.Conv2d(n_depth, m1, k1, padding = p1)
#         self.bn1    = nn.BatchNorm2d(m1)
#         self.conv2  = nn.Conv2d(m1, m2, k2, padding = p2)
#         self.bn2    = nn.BatchNorm2d(m2)
#         self.conv3  = nn.Conv2d(m2, m3, k3, padding = p3)
#         self.bn3    = nn.BatchNorm2d(m3)
#         self.pool   = nn.MaxPool2d(2, 2)
#         self.smoid  = nn.Sigmoid()
#         n_out = n_width - (k1+k2+k3) + 2*(p1+p2+p2) + 3
#         self.fc0    = nn.Linear(n_out * n_out * m3, m2)
#         self.fc1    = nn.Linear(m2, 1)

#     def forward(self, x):
#         def _sshape(x):
#             si = str(x.size())[11: -1]
#             print(si)
#             return si
#         if (self.debug): s = 'CNN : \n   ' + _sshape(x) 
#         x = self.bn1(F.leaky_relu(self.conv1(x)))
#         if (self.debug): s = s + ' => ' + _sshape(x) 
#         x = self.bn2(F.leaky_relu(self.conv2(x)))
#         if (self.debug): s = s + ' => ' + _sshape(x) 
#         x = self.bn3(F.leaky_relu(self.conv3(x)))
#         if (self.debug): s = s + '=> ' + _sshape(x) 
#         x = x.flatten(start_dim=1)
#         if (self.debug): s = s + ' => ' + _sshape(x) 
#         #x = self.drop(x)
#         x = self.fc0(x)
#         if (self.debug): s = s + ' => ' + _sshape(x) + '\n'
#         x = self.smoid(self.fc1(x))
#         #x = self.fc1(x)
#         if (self.debug): s = s + ' => ' + _sshape(x) + '\n'
#         if (self.debug): print(s)
#         self.debug = False
#         return x


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

def to_numpy(y):
    if torch.cuda.is_available():
        y = y.detach().cpu().numpy()
        return y
    return y.numpy()


def _training(model, optimizer, train, loss_function):
    losses = []
    model.train()
    for xs, ys in train:
        xs, ys = in_cuda(xs), in_cuda(ys)
        ys_pred = model(xs)
        loss    = loss_function(ys_pred, ys)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.data.item())
    return losses

def _validation(model, val, loss_function):
    losses = []
    model.eval()
    with torch.no_grad():
        for xs, ys in val:
            xs, ys  = in_cuda(xs), in_cuda(ys)
            ys_pred = model(xs)
            loss    = loss_function(ys_pred, ys)
            losses.append(loss.data.item())
    return losses


def _epoch(model, optimizer, train, val, loss_function):

    losses_train = _training(model, optimizer, train, loss_function)
    losses_val   = _validation(model, val, loss_function)

    _sum = lambda x: (np.mean(x), np.std(x))
    sum  = (_sum(losses_train), _sum(losses_val))
    acc  = (accuracy(model, train), accuracy(model, val))

    print('Epoch:  loss train {:1.2e} +- {:1.2e}  validation {:1.2e} +- {:1.2e}'.format(*sum[0], *sum[1]))
    print('        accuracy train {:4.3f} validation {:4.3f}'.format(*acc))

    return sum, acc


def train_model(model, optimizer, train, val, loss_function, nepochs = 20):

    model.train()
    losses, accus = [], []
    for i in range(nepochs):
        loss, acc = _epoch(model, optimizer, train, val, loss_function)
        losses.append(loss)
        accus .append(acc)

    return losses, accus
    

#---------------------
# Prediction
#-----------------------

def prediction_scale(digits):
    """ from the output of the NN returns an scalar
    """
    ddim = digits.shape
    if (len(ddim) == 1): return digits
    if ddim[-1] == 2:
        zs = nn.functional.softmax(digits, dim = 1)
        return zs[:, 1]
    return digits

def prediction_class(digits, y0 = 0.5):
    """ from the output of the NN returns 0,1 for the 2 classes
    """
    yscale = prediction_scale(digits)
    return yscale > y0
    
def prediction(model, test, type = 'scale'):
    _prediction = prediction_class if type == 'class' else prediction_scale
    with torch.no_grad():
        model.eval()
        for xs, ys in test:
            xs, ys  = in_cuda(xs), in_cuda(ys)
            ys_pred = _prediction(model(xs))
    return to_numpy(ys), to_numpy(ys_pred)

def accuracy(model, val):
    ys, yps   = prediction(model, val, 'class')
    acc       = np.sum(ys == yps)/len(ys)
    return acc

#===========================
# Run
#===========================

def device():
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available()
           else "cpu")
    
    print(f"Using {dev} device ")
    return dev


def run(dataset, model, nepochs = 10, ofilename = '', config = config):

    print(dataset)
    print(config)
    loss_function = loss_functions[config['loss_function']]
    learning_rate = config['learning_rate']

    train, test, val, index = subsets(dataset)
    assert len(dataset.x.shape) == 4
    print('Event Image sample : ', dataset.x.shape)

    device()
    #model  = model.to(dev)
    model  = in_cuda(model)
    print(model)
    ok =  torch.cuda.is_available()
    print(f"Is CUDA avialable? {ok} ")

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    losses, accus = train_model(model, optimizer, train, val, loss_function, nepochs = nepochs)
    ys, yps       = prediction(model, test)

    if (ofilename != ''):
        print('save cnn results at ', ofilename)
        np.savez(ofilename, losses = losses, accuracies = accus, index = index, y = ys, yp = yps)

    return CNNResult(model, dataset, losses, accus, index, ys, yps)

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
    if config['black']: sconfig += 'B'
    if config['img_scale'] != 1:
        sconfig += 'F'+str(int(config['img_scale']))

    ss = ut.str_concatenate((cnnname, slabels, sconfig, eloss),'_')

    return ss


def filename_cnn(ifilename, config):
    """ return the formated data files for cnn-input and cnn-output
    """
    fname   = ifilename.split('.')[0]
    cnnname = config['cnnname']
    sname   = cnn_config_name(cnnname, config)
    ofile   = ut.str_concatenate((fname, sname)) +  '.npz'
    return ofile

def production(root, ofile, config):
    
    print('root        : ', root)
    print('output file : ', ofile)
    print('config      : ', config)

    label  = config['label']
    width  = config['width']
    frame  = config['frame']

    evtdis = dp.EvtDispatch(root)
    imgdis = dp.ImgDispach(evtdis, label, width, frame)
    idata  = ImgDataset(imgdis)
 
    expansion = config['expansion']
    nepochs   = config['nepochs']
    cnnname   = config['cnnname']
    CNN       = HCNN if cnnname == 'HCNN' else HKCNN if cnnname == 'HKCNN' else KCNN
    padding   = 0    if cnnname == 'HCNN' else 1

    x, _ = imgdis[0]

    color, width, _ = x.shape
    print('Input shape (C, W, H) ', color, width, width)
    print('CNN model ', cnnname)
    kernel = 3
    print('configurate cnn (kernel, expansion, padding)', kernel, expansion, padding)
    kcnn = CNN(color, width, expansion = expansion, kernel = kernel, padding = padding)
    print('run cnn (epochs) ', nepochs)
    rcnn = run(idata, kcnn, ofilename = ofile, nepochs = nepochs, config = config)
    return rcnn 


#--------------------
# Plot
#--------------------

def plot_epochs(losses, accus):
    plt.figure()
    plt.subplot(2, 2, 1)
    us  = [sum[0][0] for sum in losses]
    eus = [sum[0][1] for sum in losses]
    vs  = [sum[1][0] for sum in losses]
    evs = [sum[1][1] for sum in losses]
    plt.errorbar(range(len(us)), us, yerr = eus, alpha = 0.5, label = "train")
    plt.errorbar(0.1+np.arange(len(vs)), vs, yerr = evs, alpha = 0.5, label = "validation")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend();
    plt.subplot(2, 2, 2)
    us  = [acc[0] for acc in accus]
    vs  = [acc[1] for acc in accus]
    plt.plot(range(len(us)), us, label = 'train')
    plt.plot(range(len(vs)), vs, label = 'validation')
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend();
    plt.tight_layout()
    return 



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

def false_positives_indices(y, yp, yp0 = 0.95, index0 = 0):
    ids = index0 + np.argwhere(np.logical_and(y == 0, yp >= yp0)).flatten()
    return ids

def false_negatives_indices(y, yp, yp0 = 0.05, index0 = 0):
    ids = index0 + np.argwhere(np.logical_and(y == 1, yp <= yp0)).flatten()
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