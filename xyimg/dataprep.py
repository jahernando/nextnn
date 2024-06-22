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


#------------
# Parameters
#-------------

wi        = 25*(1e-6)   # MeV (ionization threshold)
track_id  = 0           # ID of the main track
hit_width = (2, 2, 2) # size of the MC hits

frames = {'20bar': 67, '13bar' : 100, '5bar' : 304, '2bar' : 587, '1bar' : 593}

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

def test_voxel_filename(pressure, sample):
    filename = filename_voxel(pressure, sample)
    assert filename == "voxel_dataset_" + pressure + "_" + sample + ".h5"
    return True

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


#-----------
#   Algorithm
#-----------

def evt_preparation(evt, track_id = track_id, wi = wi):
    """ Maniputales the initiall DF of MC hits and does:
    DF should have 'x', 'y', 'z' (mm), 'E' (MeV), segclass, ext
        - locate the hits at the center of the main track (with label track_id)
        - set the number of ielectrons in each hit (divided E/wi)
        - normalize the total energy of the event to 1.
        - set segclass :(1-track, 2-delta electron, 3-blob)
        - set ext      :(1-track, 2-main blob, 3-minor blob)
    Input:
        - event: DF with (x, y, z, E, track_id, segclass, ext)
    """
    sel    = evt.track_id == track_id
    labels = ('x', 'y', 'z')
    x0s    = [np.mean(evt[sel][label]) for label in labels]
    #print('event len    ', len(evt))
    #print('track len    ', len(evt[sel]))
    #print('track center ', x0s)

    # center the event around the main track center
    xevt   = evt.copy()
    for label, x0 in zip(labels, x0s):
        xevt[label] = evt[label].values - x0

    # compute the number of ielectrons
    ene = evt['E'].values
    xevt['nie'] = np.maximum(np.round(ene/wi), 1).astype(int)

    # normalize the hit energy to the total energy of the event (sure?)
    sumene = np.sum(ene)
    xevt['Enorma'] = ene/sumene

    # change the segmentation (1-track, 2-delta electron, 3-blob)
    _seg  = np.array([2, 1, 3])
    xevt['segclass'] = [_seg[x] for x in xevt.segclass.values]

    # change the ext (1-deposition, 2-main extreme, 3-minor extreme)
    ext   = evt['ext'].values
    trk   = (evt['segclass'].values >= 0).astype(int)
    xevt['ext'] = ext + trk

    return xevt

def coors(df, labels = ('x', 'y', 'z')):
    return [df[label].values for label in labels]


def test_evt_preparation(evt, track_id = track_id, wi = wi):

    xevt = evt_preparation(evt, track_id, wi)
    sel  = evt.track_id == 0
    ys, xs = coors(xevt), coors(evt)
    x0s    = [np.mean(x) for x in coors(evt[sel])]
    diffs = [yi - xi + xi0 for yi, xi, xi0 in zip(ys, xs, x0s)] 
    assert np.all(np.isclose(diffs, 0.))

    assert np.all(evt.E.values == xevt.E.values)
    ene0 = np.sum(evt.E)
    ene  = np.sum(xevt.E)
    assert np.isclose(ene0, ene)

    ni0  = int(round(ene0/wi))
    nie  = np.sum(xevt.nie)
    assert np.isclose(ni0, nie, atol = np.sqrt(ni0))

    assert np.isclose(np.sum(xevt.Enorma), 1.)

    segs = xevt.segclass.unique()
    for i in (1, 2, 3): assert (i in segs)
    
    ext  = xevt.ext.unique()
    for i in (1, 2, 3): assert (i in ext)

    assert len(evt) == len(xevt)

    return True

def evt_ielectrons(evt, width = hit_width, wi = wi, transfer_info = True):

    pos = coors(evt)
    ene = evt.E
    nie = np.maximum(np.round(ene/wi), 1).astype(int)

    def _distribute(xi, wi):
        def _idistribute(xii, nii):
            x  = xii * np.ones(nii)
            x +=  wi * np.random.uniform(-0.5, 0.5, size = nii)
            return x
        xs = [_idistribute(xii, nii) for xii, nii in zip(xi, nie)]
        return np.concatenate(xs)

    def _ene():
        def _iene(eneii, nii):
            return eneii/nii * np.ones(nii)
        es = [_iene(eneii, nii) for eneii, nii in zip(ene, nie)]
        return np.concatenate(es)
        
    # associate to each electron the index of the original hit
    ieid = np.concatenate([i * np.ones(ni) for i, ni in enumerate(nie)]).astype(int)
    # distribute the position of the electrion inside the box uniformly
    iepos = [_distribute(xi, wi) for xi, wi in zip(pos, width)]
    # share the energy of the hit among the ielectrons 
    ieene = _ene()

    dd = {'hit_id': ieid, 'x': iepos[0], 'y' : iepos[1], 'z' : iepos[2], 'E' : ieene}

    for label in ('segclass', 'ext', 'track_id'):
        var = evt[label].values
        dd[label] = (np.concatenate([v * np.ones(ni) for v,ni in zip(var, nie)])).astype(int)

    nsize = len(ieene)
    for label in ('file_id', 'event', 'binclass'):
        dd[label] = (evt[label].values[0] * np.ones(nsize)).astype(int)

    return pd.DataFrame(dd)

def test_evt_ielectron(evt):

    dfie = evt_ielectrons(evt)
    nie  = np.maximum(np.round(evt.E.values/wi), 1).astype(int)

    assert len(dfie) == np.sum(nie)

    for i in range(len(evt)):
        sel = (dfie['hit_id'] == i)
        dii = dfie[sel]
        #print(' hit id ', i, ', size ', len(dii), nie[i])
        #print('Energy ', np.sum(dii.E), evt.E.values[i])
        #print('x pos ', np.min(dii.x), np.max(dii.x),  evt.x.values[i])
        #print('y pos ', np.min(dii.y), np.max(dii.y),  evt.y.values[i])
        assert len(dii) == nie[i]
        #assert np.isclose(np.sum(dii.E), evt.E[i], atol = wi)
        for k, label in enumerate(('x', 'y', 'z')):
            assert np.all(np.isclose(np.mean(dii[label]), evt[label].values[i], atol = 0.5*hit_width[k]))
            assert np.all(np.isclose(np.max(dii[label]),  evt[label].values[i], atol = 0.5*hit_width[k]))
            assert np.all(np.isclose(np.min(dii[label]),  evt[label].values[i], atol = 0.5*hit_width[k]))

    return True


def evt_ielectrons_diffuse(dfie, sigma, copy = False):

    xdfie = dfie.copy() if copy else dfie

    nsize = len(xdfie)
    def _smear(x, s):
        return x + s * np.random.normal(0, 1, size = nsize)

    for label, s in zip(('x', 'y', 'z'), sigma):
        xdfie[label] = _smear(xdfie[label].values, s)

    return xdfie

def test_evt_ielectrons_diffuse(df, sigma = (1, 2, 3)):

    sdf = evt_ielectrons_diffuse(df, sigma, copy = True)

    for label, s in zip(('x', 'y', 'z'), sigma):
        dif =  sdf[label] - df[label] 
        assert np.all(np.isclose(np.mean(dif), 0.,  atol = s/10.))
        assert np.all(np.isclose(np.std(dif ),  s,  atol = s/10.))

    return True


def evt_shot(xdf, zdf = None, xlabel = ('xy_E_sum',),
             zlabel = ('xy_segclass_mean', 'xy_ext_max'),
             bins = -1, width = (10, 10), frame = 100):

    if (bins == -1):
        bins = [np.arange(-frame - w, frame + w, w) for w in width]

    dbins = {}
    for label, bin in zip(('x', 'y', 'z'), bins): dbins[label] = bin
        
    def _img(df, label):
        proyection, varname, statistic = label.split('_') 
        xcoors     = coors(df, proyection)
        xbins      = [dbins[label] for label in proyection]
        var        = df[varname].values
        if (varname == 'E'): var = var/np.sum(var) # normalize the energy to the total of the event
        img , _, _ = stats.binned_statistic_dd(xcoors, var,  bins = xbins, statistic = statistic)
        img        = np.nan_to_num(img, 0) 
        return img
    
    xdic = {}
    for label in xlabel: xdic[label] = _img(xdf, label)
    y    =  xdf.binclass.unique()[0]
    zdic = {}
    zdf =  zdf if isinstance(zdf, pd.DataFrame) else xdf
    for label in zlabel: zdic[label] = _img(zdf, label)
    evtid = (xdf['file_id'].unique()[0], xdf['event'].unique()[0])

    gdata = GoData(xdic, y, zdic, evtid) 

    return gdata


def evt_voxelize(evt, bins = -1, frame = 100, width = (10, 10)):

    return


# def evt_godata(evt, xlabel, zlabel, bins):

#     def _coors(evt, labels):
#         return [evt[label].values for label in labels]

#     def _img(label):
#         proyection, varname, statistic = label.split('_') 
#         coors      = _coors(evt, proyection)
#         var        = evt[varname].values
#         img , _, _ = stats.binned_statistic_dd(coors, var,  bins = bins, statistic = statistic)
#         img        = np.nan_to_num(img, 0) 
#         return img
    
#     xdic = {}
#     for label in xlabel: xdic[label] = _img(label)
#     y     = evt.binclass.unique()[0]

#     zdic = {}
#     for label in zlabel: zdic[label] = _img(label)

#     evtid = (evt['file_id'].unique()[0], evt['event'].unique()[0])

#     gdata = GoData(xdic, y, zdic, evtid) 

#     return gdata

xlabel = ['xy_E_sum', 'yz_E_sum', 'zx_E_sum']
zlabel = ['xy_segclass_mean', 'xy_ext_max',
          'yz_segclass_mean', 'yz_ext_max',
          'zx_segclass_mean', 'zx_ext_max']


def run(ifilename,
        ofilename,
        sigma      = (0, 0, 0),
        width      = (10, 10, 10),
        frame      = 100.,
        xlabel     = xlabel,
        zlabel     = zlabel,
        nevents    = 10, 
        verbose    = True):

    # check inputs    
    assert len(sigma) == 3
    assert len(width) == 3
    for label in xlabel : assert label.split('_')[0] in ['xy', 'yz', 'zx', 'xyz', 'yzx', 'zxy']
    for label in xlabel : assert label.split('_')[1] in ['x', 'y', 'z', 'E']
    for label in xlabel : assert label.split('_')[2] in ['min', 'max', 'mean', 'sum', 'std', 'count']
    for label in zlabel : assert label.split('_')[0] in ['xy', 'yz', 'zx', 'xyz', 'yzx', 'zxy']
    for label in zlabel : assert label.split('_')[1] in ['segclass', 'ext']
    for label in zlabel : assert label.split('_')[2] in ['min', 'max', 'mean']
    assert np.min(sigma) >= 0.
    assert np.min(width) > 0.
    assert frame > 2. * np.min(width)

    t0 = time.time()
    if (verbose):
        print('input  filename      ', ifilename)
        print('output filename      ', ofilename)
        print('sigma diffusion (mm) ', sigma)
        print('widths          (mm) ', width)
        print('frame           (mm) ', frame)
        print('xlabel               ', xlabel)
        print('zlabel               ', zlabel)
        print('events               ', nevents)

    do_smearing = np.sum(sigma) > 0

    # def _label(label):
    #     label = [[p + '_' + k for k in label] for p in projection]
    #     label = [j for i in label for j in i]
    #     return label
    # xlabel = _label(xlabel)
    # zlabel = _label(zlabel)
    # print('x labels      : ', xlabel)
    # print('z labels      : ', zlabel)

    bins   = [np.arange(-frame, frame, w) for w in width]

    def _evt(evt):
        evt   = evt_preparation(evt)
        dfie  = evt_ielectrons(evt, width = hit_width)
        if (do_smearing):
            dfie = evt_ielectrons_diffuse(dfie, sigma = sigma)
        shot  = evt_shot(xdf = dfie, zdf = evt,
                         xlabel = xlabel, zlabel = zlabel,
                         bins = bins)
        return shot

    ta = time.time()
    i = 0
    ifilename = [ifilename,] if isinstance(ifilename, str) else ifilename
    odata = godata_init(xlabel, zlabel)
    for k, ifile in enumerate(ifilename):
        print('opening voxel file : ', ifile)
        idata = pd.read_hdf(ifile, 'voxels')
        for evtid, evt in idata.groupby(['file_id', 'event']):
            i += 1
            if (nevents > 0) & (i > nevents): break
            if i % 100 == 0: print('processing event ', i, ', id ', evtid)
            godata_append(odata, _evt(evt))
    print('save godata filename :', ofilename)
    godata_save(odata, ofilename)

    t1 = time.time()
    print('event processed   {:d} '.format(i))
    print('time per event    {:4.2f} s'.format((t1-ta)/i))
    print('time execution    {:8.1f}  s'.format(t1-t0))
    print('done!')

    return odata

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


#------------

def test_run(ifilename, ofilename):
    projection = ['xy', 'xz', 'zy']
    xlabel     = ['E_sum', 'E_count', 'E_mean']
    zlabel     = ['segclass_mean', 'ext_max']
    sigma      = (2, 2, 2)
    width      = (10, 10, 10)
    frame      = 60
    odata      = run(ifilename, ofilename, sigma = sigma, width = width, frame = frame,
                     projection = projection, xlabel = xlabel , zlavel = zlabel, nevents= 10)
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
