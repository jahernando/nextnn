#  Module to produce images of NEXT tracks for NN
#  JA Hernando 15/04/24

import numpy             as np
import pandas            as pd
import random            as random
import sys
import os
import time              as time

from   scipy       import stats

import matplotlib.pyplot as plt

from collections import namedtuple


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
    # print('Output file : ', ofilename)
    xlabel = list(odata.xdic.keys())
    x = np.array([np.array(odata.xdic[label]) for label in xlabel])
    y = np.array(odata.y)
    zlabel = list(odata.zdic.keys())
    z = np.array([np.array(odata.zdic[label]) for label in zlabel])
    id = np.array(odata.id)
    np.savez_compressed(ofilename, x = x, y = y, z = z, id = id,
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


#---------------------
#  Input Output files
#---------------------


def voxel_filename(pressure, sample, prefix = 'voxel_dataset', ext = '.h5'):
    """ return the voxel filename
    inputs:
        pressure: '13bar', '5bar', '2bar', '1bar'
        sample  : '1eroi', '0nubb'
        prefix  : default 'voxel_dataset'
        ext     : default '.h5'
    """
    filename = str_concatenate((prefix, pressure, sample)) + ext
    return filename


def xymm_filename(projection, widths, frame, prefix = 'xymm', ext = '.npz'):
    """ return the xymm images filename
    inputs:
        projection : ('x', 'y'), ('x', 'y', 'z')
        widths     : i.e (5, 5) (only integers)
        frame      : 100 (int)
        prefix     : default 'xymm'
        ext        : default '.npz'
    """
    sproj   = str_concatenate(projection, '')
    swidths = str_concatenate([str(int(w)) for w in widths], 'x')
    ofile   = str_concatenate((prefix, sproj, swidths, int(frame)), '_') + ext
    return ofile

def prepend_filename(ifilename, name , link = '_'):
    words = ifilename.split('/')
    fname = words[-1]
    ofname = str_concatenate((name, fname), link)
    ofname = str_concatenate(words[:-1], '/') + '/'+ofname
    return ofname

#---------------
# Utils
#----------------

def str_concatenate(words, link = '_'):
    ss = ''
    for w in words: ss += str(w) + link
    nn = len(link)
    if nn == 0: return ss 
    return ss[:-nn]

def urange(var : np.array) -> np.array:
    """ set the variable in the range [0, 1]
    input:
     - var: np.array(float)
    """
    vmin, vmax = np.min(var), np.max(var)
    if vmax <= vmin: return np.zeros(len(var))
    return (var-vmin)/(vmax-vmin)

def uframecentered(var, width):
    """
    center the [min, max] of the variable centered in a frame of a given width
    """
    vmin, vmax = np.min(var), np.max(var)
    assert (vmax > vmin)
    v0   = (width - vmax + vmin)/2.
    assert (v0 >= 0.) # values should be contained in the width
    ovar = var - vmin + v0   
    return ovar

def arange_include_endpoint(start, stop, step):
    u = np.arange(start, stop, step)
    uu = list(u)
    if (uu[-1] < stop): 
        uu.append(uu[-1] + step)
    return np.array(uu)


def image(coors, data, varname, statistics, bins):
    """ return the image of the label of data in coors with given bins
    """
    var          = data[varname]
    xyimg , _, _ = stats.binned_statistic_dd(coors, var,  bins = bins, statistic = statistics)
    xyimg        = np.nan_to_num(xyimg, 0) 
    return xyimg

## Test images

def good_ttimage(segimg, extrimg, yy, strict = False):
    bbimg   = np.logical_and(extrimg, (segimg >= 3))
    nsize   = int(np.sum(bbimg))
    ok      = bool((nsize >= 1) and (nsize <= yy + 1))
    if (strict): ok = ok & (nsize == yy + 1)
    return ok


def ttimage(segimg, extrimg, y, blob_fraction = 0.2):
    """ generate a test track image using the tru segmentation and extremes image
    it produces an image where the energy is distributed alongn the track uniformely and some blob energy is added on the blobs
    """
    nblobs  = (int(y) + 1) 
    eblobs  = nblobs * blob_fraction
    ttimg   = (segimg >= 1) 
    nsize   = np.sum(ttimg) 
    ttimg   = ttimg * (1. - eblobs) / nsize
    bbimg   = np.logical_and(extrimg, (segimg >= 3))
    nsize   = np.sum(bbimg)
    #if (nsize == 0): print('Warning no blobs in the event!')
    nsize = max(1, nsize)
    bbimg   = bbimg * eblobs / nsize
    return ttimg + bbimg


#--------------
# Physical Frame
#--------------

def get_frame(idata):
    """ returns the size in each dimention of the main track of all the events in the h5 file voxel data.
    """
    sel   = idata.track_id == 0
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

def run_algo(ifilename,
             ofilename,
             width      = (10, 10),
             frame      = 100.,
             projection = ['xy', 'xz', 'zy'],
             xlabel     = ['E_sum',],
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
    for k, ifile in enumerate(ifilename):
        odata = godata_init(xlabel, zlabel)
        print('opening ', ifile)
        idata = pd.read_hdf(ifile, 'voxels')
        for evtid, evt in idata.groupby(['file_id', 'event']):
            i += 1
            if (nevents > 0) & (i > nevents): break
            if i % 100 == 0: print('processing event ', i, ', id ', evtid)
            godata_append(odata, _evt(evt))
        ofile = ofilename.split('.')[0]+'.'+str(k)
        print('save godata at ', ofile)
        godata_save(odata, ofile)

    t1 = time.time()
    print('event processed   {:d} '.format(i))
    print('time per event    {:4.2f} s'.format((t1-ta)/i))
    print('time execution    {:8.1f}  s'.format(t1-t0))
    print('done!')

    return


#----------
#   Run
#----------


def run(ifilename,
        ofilename,
        projection = ('x', 'y'),
        widths     = (10, 10),
        frame      = 100.,
        labels     = ['esum', 'ecount', 'emean', 'emax', 'estd'],
        nevents    = -1, 
        verbose    = True):

    ofilename = xymm_filename(projection, widths, frame, prefix = ofilename)
    
    if (verbose):
        print('input  filename ', ifilename)
        print('output filename ', ofilename)
        print('projection      ', projection)
        print('widths     (mm) ', widths)
        print('frame      (mm) ', frame)
        print('labels          ', labels)
        print('events          ', nevents)

    assert (len(projection) >= 2) # 2d projection
    assert (len(projection) <= 3) # 3d projection
    assert len(projection) == len(widths)
    assert len(labels) >= 1

    idata   = pd.read_hdf(ifilename, "voxels") 
    
    delta  = np.max(get_frame(idata))
    print('maximum window frame {:4.2f} mm'.format(delta))
    if (delta > frame):
        print('Error: Unsificient frame width', frame, ', must be ', delta, 'mm')
        assert delta <= frame

    bins   = [arange_include_endpoint(0, frame, width) for width in widths]
    if (verbose):
        print('image shape in bins ', [len(b)-1 for b in bins])

    def _dinit(labels):
        xdic = {}
        for label in labels: xdic[label] = []
        return xdic
    
    def _dappend(dic, vals):
        for i, label in enumerate(dic.keys()): dic[label].append(vals[i])

    zlabels = ['seg', 'ext']
    xdic    = _dinit(labels) # images
    zdic    = _dinit(zlabels) # true images
    y , id  = [], [] # target, id

    _seg  = np.array([2, 1, 3])
    def _data(evt):
        E0         = np.sum(evt.E)
        sel        = evt.track_id == 0
        xs, ys, zs = [uframecentered(x, delta) for x in [evt[sel].x, evt[sel].y, evt[sel].z]]
        es         = evt[sel].E/E0
        seg        = [_seg[x] for x in evt[sel].segclass.values]
        ext        = evt[sel].ext.values > 0
        data       = {'x':xs, 'y': ys, 'z':zs, 'e':es, 'seg': seg, 'ext': ext}
        return data

    def _coors(data, coorsnames):
        coors      = tuple(data[c] for c in coorsnames)
        return coors


    for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if ((nevents >= 1) & (i >= nevents)): break

        data  = _data(evt)
        coors = _coors(data, projection)

        xi  = [image(coors, data, label[:1], label[1:], bins) for label in labels]
        yi  = evt.binclass.unique()
        zi  = [image(coors, data, label, 'max', bins) for label in zlabels]
        idi = (evt.file_id.unique(), evt.event.unique())
        _dappend(xdic, xi)
        _dappend(zdic, zi)
        y.append(yi)
        id.append(idi)

    odata = GoData(xdic, y, zdic, id)

    ofile = ofilename.split('.')[0]
    save(odata, ofile)
    if verbose:
        print('saved output file ', ofile+'.npz')

    return odata


#-----------
# Mix samples
#------------

def mix_godata(signal_filename, bkg_filename, ofilename):
    """ create a GoData using the signal (0nubb) and bkg (1eroi) of a given pressure and image bin (width)
    events are shuffle (there are not ordered, and they are signal if y = 1, bkg if y = 0)
    """

    print('input file 1 ', signal_filename)
    print('input file 2 ', bkg_filename)

    data0 = np.load(signal_filename)
    data1 = np.load(bkg_filename)

    _swap = lambda x : np.swapaxes(x, 0 , 1)

    x0, x1 = _swap(data0['x']), _swap(data1['x'])
    y0, y1 = data0['y'], data1['y']
    z0, z1 = _swap(data0['z']), _swap(data1['z'])
    xlabel = list(data0['xlabel'])
    zlabel = list(data0['zlabel'])
    #print('x labels ', xlabel)
    #print('z labels ', zlabel)
    id0, id1 = data0['id'], data1['id']
    #xf1, xf2 = data0['xf'], data1['xf']
    #zf1, zf2 = data0['zf'], data1['zf']

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
    #xfs = _list(xf1, xf2)
    #zfs = _list(zf1, zf2)

    ww = list(zip(xs, ys, zs, ids))
    random.shuffle(ww)

    xs  = np.array([wi[0] for wi in ww])
    ys  = np.array([wi[1] for wi in ww])
    zs  = np.array([wi[2] for wi in ww])
    ids = np.array([wi[3] for wi in ww])
    #xfs = np.array([wi[4] for wi in ww])
    #zfs = np.array([wi[5] for wi in ww])

    xs = _dic(_swap(xs), xlabel)
    zs = _dic(_swap(zs), zlabel)

    odata = GoData(xs, ys, zs, ids)
    save(odata, ofilename)
    print('output file  ', ofilename+'.npz')
    return odata


#---------------
# Plot
#---------------

def plot_imgs(xs, ievt = -1, labels = -1):
    """ plots the images of xdic for the event ievt
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
    filename = voxel_filename(pressure, sample)
    assert filename == "voxel_dataset_" + pressure + "_" + sample + ".h5"
    return True

def test_xymm_filename(projection, widths, frame, prefix = 'xymm'):
    sproj  = str_concatenate(projection, '')
    swidth = str_concatenate([int(w) for w in widths], 'x')
    ofile0 = xymm_filename(projection, widths, frame, prefix)
    ofile1 = str_concatenate((prefix, sproj, swidth, frame))+'.npz'
    assert ofile0 == ofile1
    return True

def test_str_concatenate():
    words = ('x', 'y')
    ss    = str_concatenate(words, '')
    assert len(ss) == sum([len(x) for x in words])
    words = np.arange(10)
    ss    = str_concatenate(words, '_')
    assert len(ss.split('_')) == len(words)
    words = ('a', 'b', 'c')
    ss    = str_concatenate(words, 'x')
    assert len(ss.split('x')) == len(words)
    return True

def test_urange(x):
    uz = urange(x)
    assert (np.min(uz) >= 0) & (np.max(uz) <= 1)
    iar = np.argmax(x)
    assert uz[iar] == 1
    iar = np.argmin(x)
    assert uz[iar] == 0
    return True

def test_uframecentered(x, delta):
    ux   = uframecentered(x, delta)
    umin, umax = np.min(ux), np.max(ux)
    udelta = umax - umin
    d0, d1 = umin, delta - umax
    assert np.isclose(d0, d1)
    return True

def test_arange_include_endpoint(start, stop, step):
    u = arange_include_endpoint(start, stop, step)
    assert u[-1] >= stop
    assert u[-2] <  stop
    return True

#------------

def test_frame(ifilename):
    idata   = pd.read_hdf(ifilename, "voxels") 
    delta  = np.max(get_frame(idata))
    for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if (i >= 10): break
        sel = evt.track_id == 0
        us  = (evt[sel].x, evt[sel].y, evt[sel].z)
        dus = [np.max(x) - np.min(x) for x in us]
        assert np.all(np.array(dus) <= np.array(delta))
    return True


def test_run(ifilename, ofilename, coors, widths, frame):
    labels = ['esum', 'ecount', 'emean']
    odata  = run(ifilename, ofilename, coors, widths, frame, labels , nevents= 10)
    ofile  = xymm_filename(coors, widths, frame, ofilename)
    xdata  = load(ofile)

    assert np.all(xdata.y  == odata.y)
    assert np.all(xdata.id == odata.id)
    for label in xdata.xdic.keys():
        assert np.all(xdata.xdic[label] == odata.xdic[label])
    for label in xdata.zdic.keys():
        assert np.all(xdata.zdic[label] == odata.zdic[label])

    for evt in range(10):
        esum   = odata.xdic['esum'][evt]
        emean  = odata.xdic['ecount'][evt]
        ecount = odata.xdic['emean'][evt]
        assert np.all(np.isclose(esum, emean * ecount))
    return True


def test_mix_godata(ifilename1, ifilename2, ofilename):

    odata = mix_godata(ifilename1, ifilename2, ofilename)

    def _test(y):
        nsig = np.sum(y == 1)
        nbkg = np.sum(y == 0)
        assert (nsig >0) & (nbkg > 0)

    _test(odata.y)
    nsize = len(odata.y)
    ni = int(nsize/2)
    _test(odata.y[0  : ni])
    _test(odata.y[ni :   ])

    for evt in range(min(10, nsize)):
        esum   = odata.xdic['esum'][evt]
        emean  = odata.xdic['ecount'][evt]
        ecount = odata.xdic['emean'][evt]
        assert np.all(np.isclose(esum, emean * ecount))

    for evt in range(min(10, nsize)):
        next  = np.sum(odata.zdic['ext'][evt] > 0)
        assert (next >= 1) & (next <= 2)

    return True

def _test_ttimage(seg, ext, y, tt, bf):
    assert np.isclose(np.sum(tt), 1.)
    nsize       = np.sum(tt > 0.)
    nblobs      = int(y) + 1
    emean       = (1 - nblobs * bf) / nsize
    mask_blobs  = np.logical_and(seg == 3, ext >= 1)
    mblobs      = np.sum(mask_blobs)
    eblobs      = np.sum(tt[mask_blobs])
    assert np.isclose(eblobs, nblobs * bf + mblobs * emean)
    return True


def test_ttimages(ifile):
    idata = load(ifile)

    for i in range(min(len(idata.y), 10)):
        seg, ext, yi = idata.zdic['seg'][i], idata.zdic['ext'][i], idata.y[i]
        tt = ttimage(seg, ext, yi, 0.2)
        _test_ttimage(seg, ext, yi, tt, 0.2)
    return True


def tests(path):

    pressure = '13bar'
    sample1  = '0nubb'
    sample2  = '1eroi'
    
    coords   = ('x', 'y')
    widths   = (10, 10)
    frame    = 100

    test_str_concatenate()
    test_voxel_filename(pressure, sample1)
    test_xymm_filename(coords, widths, frame)

    test_godata()
    test_urange(np.arange(30))
    test_arange_include_endpoint(0, 10, 0.3)
    test_uframecentered(np.arange(30), 50)

    ifilename1 = path + voxel_filename(pressure, sample1)
    test_frame(ifilename1)

    test_run(ifilename1, 'temp/sample1', ('x', 'y'), (10., 10.), 100.,)
    test_run(ifilename1, 'temp/sample1', ('x', 'z'), (10., 10.), 100.)
    test_run(ifilename1, 'temp/sample1', ('x', 'y', 'z'), (10., 10., 10.), 100.)

    ifilename2 = path + voxel_filename(pressure, sample2)
    test_run(ifilename2, 'temp/sample2', ('x', 'y'), (10., 10.), 100.)

    ofile = 'temp/test_'+pressure
    test_mix_godata('temp/sample1_xy_10x10_100.npz', 'temp/sample2_xy_10x10_100.npz', ofile)
    
    test_ttimages(ofile + '.npz')

    print('Passed all tests!')
    return True
