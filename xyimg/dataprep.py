#  Module to produce images of NEXT tracks for NN
#  JA Hernando 15/04/24

import numpy             as np
import pandas            as pd
import random            as random
import sys
import os
import time              as time
from   scipy       import stats
from   collections import namedtuple

import matplotlib.pyplot as plt
import xyimg.utils       as ut



#-------------
#   GoData Structurea
#------------

"""
GoData are 2D images with a target value, true images, and and id
contains:
    xdic : a dictionary with the images 
    zdic : a dictionary with the true images (extremes and segmentation (0-track, 1-other, 2-blob))
    y    : the target (0-bkg, 1-signal)
    id   : id of the event (file, event-number)
data is saved in a npz file where the dictionaries are stored as values (x, z) and labels (xlabel, zlabel)
"""
GoData      = namedtuple('GoData', ['xdic', 'y', 'zdic', 'id'])

frames = {'20bar': 67, '13bar' : 100, '5bar' : 304, '2bar' : 587, '1bar' : 593}

def godata_init(xlabel, zlabel):
    xdic, zdic, y, id = {}, {}, [], []
    for label in xlabel: xdic[label] = []
    for label in zlabel: zdic[label] = []
    return GoData(xdic, y, zdic, id)

def godata_append(data0, data1):
    for label in data0.xdic.keys():
        data0.xdic[label].append(data1.xdic[label])
    for label in data0.zdic.keys():
        data0.zdic[label].append(data1.zdic[label])
    data0.y.append(data1.y)
    data0.id.append(data1.id)
    return data0

def godata_save(odata, ofilename):
    def _ofile(ofilename, extension = 'npz'):
        words = ofilename.split('.')
        if (words[-1] != extension): return ofilename
        return ut.str_concatenate(words[:-1], '.')
    # print('Output file : ', ofilename)
    xlabel = list(odata.xdic.keys())
    x = np.array([np.array(odata.xdic[label]) for label in xlabel])
    y = np.array(odata.y)
    zlabel = list(odata.zdic.keys())
    z = np.array([np.array(odata.zdic[label]) for label in zlabel])
    id = np.array(odata.id)
    ofile = _ofile(ofilename)
    print('save file without extension ', ofile)
    np.savez_compressed(ofile, x = x, y = y, z = z, id = id,
             xlabel = np.array(xlabel), zlabel = np.array(zlabel))
    return

def godata_load(ifilename):
    
    data = np.load(ifilename)
    x, y, z, id = data['x'], data['y'], data['z'], data['id']
    xlabel, zlabel = list(data['xlabel']), list(data['zlabel'])

    xdic = {}
    for i, label in enumerate(xlabel): xdic[label] = x[i]
    zdic = {}
    for i, label in enumerate(zlabel): zdic[label] = z[i]
    odata = GoData(xdic, y, zdic, id )
    return odata


def godata_shuffle(data0, data1):
    """ 
    mix two GoData objsect and produce a Godats with shuffle contents
    """

    _swap = lambda x : np.swapaxes(x, 0 , 1)

    x0, x1 = _swap(data0['x']), _swap(data1['x'])
    y0, y1 = data0['y'], data1['y']
    z0, z1 = _swap(data0['z']), _swap(data1['z'])
    xlabel = list(data0['xlabel'])
    zlabel = list(data0['zlabel'])
    id0, id1 = data0['id'], data1['id']

    def _list(a, b):
        return list(a) + list(b)

    def _dic(vals, labels):
        xdic = {}
        for i, label in enumerate(labels):  xdic[label] = vals[i]
        return xdic

    xs  = _list(x0, x1)
    ys  = _list(y0, y1)
    zs  = _list(z0, z1)
    ids = _list(id0, id1)

    ww = list(zip(xs, ys, zs, ids))
    random.shuffle(ww)

    xs  = np.array([wi[0] for wi in ww])
    ys  = np.array([wi[1] for wi in ww])
    zs  = np.array([wi[2] for wi in ww])
    ids = np.array([wi[3] for wi in ww])

    xs = _dic(_swap(xs), xlabel)
    zs = _dic(_swap(zs), zlabel)

    odata = GoData(xs, ys, zs, ids)
    return odata

#---------------------
#  Input Output files
#---------------------

def filename_voxel(pressure, sample, prefix = 'voxel_dataset', ext = '.h5'):
    """ return the voxel filename
    inputs:
        pressure: '13bar', '5bar', '2bar', '1bar'
        sample  : '1eroi', '0nubb'
        prefix  : default 'voxel_dataset'
        ext     : default '.h5'
    """
    filename = ut.str_concatenate((prefix, pressure, sample)) + ext
    return filename


#--------------
# Frame
#--------------

def get_frame(idata, track_id = 0):
    """ returns the size in each dimention of the main track of all the events in the h5 file voxel data.
    """
    sel   = idata.track_id == track_id
    dmax  = idata[sel].groupby(['file_id', 'event']).max()
    dmin  = idata[sel].groupby(['file_id', 'event']).min()
    delta = dmax - dmin
    dx, dy, dz  = np.max(delta.x), np.max(delta.y), np.max(delta.z)
    return dx, dy, dz

#-----------
#   Algorithm
#-----------

def evt_preparation(evt, track_id = 0):
    """ locate the main track in the center of the frame, and normalize total energy to 1.
    segclass and ext are modified too:
        - segclass: (1-track, 2-delta electron, 3-blob)
        - ext     : (1-track, 2-minor extreme,  3-mayor extreme)
    Input:
        - event: DF with (x, y, z, E, track_id, segclass, ext)
        - frams: float, the size of the frame (mm)
    """
    sel = evt.track_id == track_id
    sevt = evt[sel]

    labels = ('x', 'y', 'z')
    xs     = [sevt[label].values for label in labels]
    x0s    = [np.mean(x) for x in xs]
    #print('event len    ', len(evt))
    #print('track len    ', len(sevt))
    #print('track center ', x0s)

    xevt   = evt.copy()
    for label, x0 in zip(labels, x0s):
        xevt[label] = xevt[label] - x0

    ene = np.sum(evt['E'].values)
    xevt['E'] = xevt['E']/ene

    _seg  = np.array([2, 1, 3])
    xevt['segclass'] = [_seg[x] for x in xevt.segclass.values]

    def _ext():
        xi  = (xevt.track_id.values == 0).astype(int)
        xi += xevt.ext.values
        return xi
    
    xevt['ext'] = _ext()

    return xevt


def evt_godata(evt, xlabel, zlabel, bins):

    def _coors(evt, labels):
        return [evt[label].values for label in labels]

    def _img(label):
        proyection, varname, statistic = label.split('_') 
        coors      = _coors(evt, proyection)
        var        = evt[varname].values
        img , _, _ = stats.binned_statistic_dd(coors, var,  bins = bins, statistic = statistic)
        img        = np.nan_to_num(img, 0) 
        return img
    
    xdic = {}
    for label in xlabel: xdic[label] = _img(label)
    y     = evt.binclass.unique()[0]

    zdic = {}
    for label in zlabel: zdic[label] = _img(label)

    evtid = (evt['file_id'].unique()[0], evt['event'].unique()[0])

    gdata = GoData(xdic, y, zdic, evtid) 

    return gdata

track_id = 0

def run(ifilename,
        ofilename,
        width      = (10, 10),
        frame      = 100.,
        projection = ['xy', 'xz', 'zy'],
        xlabel     = ['E_sum', 'E_count'],
        zlabel     = ['segclass_max', 'ext_max'],
        nevents    = 10, 
        verbose    = True):
    
    t0 = time.time()
    if (verbose):
        print('input  filename ', ifilename)
        print('output filename ', ofilename)
        print('projection      ', projection)
        print('widths     (mm) ', width)
        print('frame      (mm) ', frame)
        print('xlabel          ', xlabel)
        print('zlabel          ', zlabel)
        print('events          ', nevents)

    def _label(label):
        label = [[p + '_' + k for k in label] for p in projection]
        label = [j for i in label for j in i]
        return label
    xlabel = _label(xlabel)
    zlabel = _label(zlabel)

    print('x labels ', xlabel)
    print('z labels ', zlabel)

    bins   = [np.arange(-frame, frame, w) for w in width]

    def _evt(evt):
        evt   = evt_preparation(evt) 
        gdata = evt_godata(evt, xlabel, zlabel, bins)
        return gdata

    ta = time.time()
    i = 0
    ifilename = [ifilename,] if isinstance(ifilename, str) else ifilename
    odata = godata_init(xlabel, zlabel)
    for k, ifile in enumerate(ifilename):
        print('opening ', ifile)
        idata = pd.read_hdf(ifile, 'voxels')
        for evtid, evt in idata.groupby(['file_id', 'event']):
            i += 1
            if (nevents > 0) & (i > nevents): break
            if i % 100 == 0: print('processing event ', i, ', id ', evtid)
            godata_append(odata, _evt(evt))
    print('save godata at ', ofilename)
    godata_save(odata, ofilename)

    t1 = time.time()
    print('event processed   {:d} '.format(i))
    print('time per event    {:4.2f} s'.format((t1-ta)/i))
    print('time execution    {:8.1f}  s'.format(t1-t0))
    print('done!')

    return

#---------------
# Plot
#---------------

def plot_imgs(xs, ievt = -1, labels = -1):
    """ plots the images stored in a dictonary.
    inputs:
        - xs    : dict{label, list or image}
        - i     : index of the image in the dictionary to plot (if -1, there is only one image in the dict)
        - label : label of the image in the dictionary to plot
    """

    def _img(ki):
        label = labels[ki]
        if (isinstance(xs, dict)): 
            if (ievt >= 0): return xs[label][ievt]
            return xs[label]
        return xs[ievt][ki]

    labels = list(xs.keys()) if labels == -1 else labels
    n = len(labels)
    m = int(n/4) +  n % 4
    for k in range(m):
        plt.figure()
        for i in range(4):
            ki = 4 * k + i
            if (ki >= n): break
            plt.subplot(2, 2, i + 1)
            label = labels[ki]
            plt.imshow(_img(ki)); 
            plt.title(label); plt.colorbar()
        plt.tight_layout()
    return

def plot_godata(gdata, ievt = -1, labels = -1):
    y  = gdata.y[ievt]  if ievt >= 0 else gdata.y
    id = gdata.id[ievt] if ievt >= 0 else gdata.id
    print('y  ', y)
    print('id ', id)
    plot_imgs(gdata.xdic, ievt = ievt, labels = labels)
    plot_imgs(gdata.zdic, ievt = ievt, labels = labels)
    return

#--------------------------------------------------------


#--------------
# Tests
#--------------

def test_godata():
    xdic  = {'a' : (10, 2, 2), 'b': (10, 3, 2)}
    y     = [1, 1]
    zdic  = {'1' : (10, 2, 2), '2': (10, 2, 2)}
    id    = [0, 1]
    idata = GoData(xdic, y, zdic, id)
    ofile = 'temp/temp'
    godata_save(idata, ofile)
    odata = godata_load(ofile+'.npz')
    for label in xdic.keys():
        assert np.all(idata.xdic[label] == odata.xdic[label])
    for label in zdic.keys():
        assert np.all(idata.zdic[label] == odata.zdic[label])
    assert np.all(idata.y  == odata.y)
    assert np.all(idata.id == odata.id)
    return True

def test_voxel_filename(pressure, sample):
    filename = filename_voxel(pressure, sample)
    assert filename == "voxel_dataset_" + pressure + "_" + sample + ".h5"
    return True

#------------

def test_frame(ifilename):
    idata  = pd.read_hdf(ifilename, "voxels") 
    delta  = np.max(get_frame(idata))
    for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if (i >= 10): break
        sel = evt.track_id == 0
        us  = (evt[sel].x, evt[sel].y, evt[sel].z)
        dus = [np.max(x) - np.min(x) for x in us]
        assert np.all(np.array(dus) <= np.array(delta))
    return True


def test_run(ifilename, ofilename):
    projection = ['xy', 'xz', 'zy']
    xlabel     = ['E_sum', 'E_count', 'E_mean']
    width      = (10, 10)
    frame      = 60
    odata      = run(ifilename, ofilename, width = width, frame = frame,
                     projection = projection, xlabel = xlabel , nevents= 10)
    xdata      = godata_load(ofilename)

    assert np.all(xdata.y  == odata.y)
    assert np.all(xdata.id == odata.id)
    for label in xdata.xdic.keys():
        assert np.all(xdata.xdic[label] == odata.xdic[label])
    for label in xdata.zdic.keys():
        assert np.all(xdata.zdic[label] == odata.zdic[label])

    for evt in range(10):
        for pro in projection:
            esum   = odata.xdic[pro+'_E_sum'][evt]
            emean  = odata.xdic[pro+'_E_count'][evt]
            ecount = odata.xdic[pro+'_E_mean'][evt]
            assert np.all(np.isclose(esum, emean * ecount))
    return True


def test_godata_shuffle(data0, data1):

    return True

def tests(path):

    pressure = '13bar'
    sample1  = '0nubb'
    sample2  = '1eroi'

    test_voxel_filename(pressure, sample1)

    test_godata()
    
    ifilename1 = path + filename_voxel(pressure, sample1)
    test_frame(ifilename1)

    test_run(ifilename1, 'temp/sample1')

    ifilename2 = path + filename_voxel(pressure, sample2)
    test_run(ifilename2, 'temp/sample2')

    print('Passed all tests!')
    return True
