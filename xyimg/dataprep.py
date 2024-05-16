#  Module to produce images of NEXT tracks for NN
#  JA Hernando 15/04/24

import numpy             as np
import pandas            as pd
import random            as random
import os

from   scipy       import stats

import matplotlib.pyplot as plt

from collections import namedtuple

#-------------
#   Data
#------------

"""
GoData is the data prepared for the NN
contains:
    xdic : a dictionary with the images 
    zdic : a dictionary with the true images (extremes and segmentation (0-track, 1-other, 2-blob))
    y    : the target (0-bkg, 1-signal)
    id   : id of the event (file, event-number)
    xf   : np.array with relative physical information (delta-x, delta-yy, delta-z (mm), fraction of E) of the main track
    zf   : np.array with absolute physical information (x0, y0, z0 (mm), E-track (MeV)) of the main track
data is saved in a npz file where the dictionaries are stored as values (x, z) and labels (xlabel, zlabel)
"""
GoData = namedtuple('GoData', ['xdic', 'y', 'zdic', 'id'])

def save(odata, ofilename):
    # print('Output file : ', ofilename)
    xlabel = list(odata.xdic.keys())
    x = np.array([np.array(odata.xdic[label]) for label in xlabel])
    y = np.array(odata.y)
    zlabel = list(odata.zdic.keys())
    z = np.array([np.array(odata.zdic[label]) for label in zlabel])
    id = np.array(odata.id)
    #xf = np.array(odata.xf)
    #zf = np.array(odata.zf)
    np.savez_compressed(ofilename, x = x, y = y, z = z, id = id,
             xlabel = np.array(xlabel), zlabel = np.array(zlabel))
#             xf = xf, zf = zf)
    return

def load(ifilename):
    
    data = np.load(ifilename)
    x, y, z, id = data['x'], data['y'], data['z'], data['id']
    xlabel, zlabel = list(data['xlabel']), list(data['zlabel'])

    xdic = {}
    for i, label in enumerate(xlabel): xdic[label] = x[i]
    zdic = {}
    for i, label in enumerate(zlabel): zdic[label] = z[i]
    odata = GoData(xdic, y, zdic, id )
    return odata


#---------------
# Utils
#----------------

def voxel_filename(pressure, sample):
    """ return the voxel filename
    """
    filename = "voxel_dataset_" + pressure + "_" + sample + ".h5"
    return filename

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

# def _track_eframe(evt):
#     etot  = np.sum(evt.E)
#     sel   = evt.track_id <= 1
#     etrk  = np.sum(evt[sel].E)
#     return etrk/etot, etrk

# def _track_xyzframe(evt):
#     sel   = evt.track_id <= 1
#     x0s   = np.array([np.min(x) for x in (evt[sel].x, evt[sel].y, evt[sel].z)])
#     x1s   = np.array([np.max(x) for x in (evt[sel].x, evt[sel].y, evt[sel].z)])
#     dxs   = x1s - x0s
#     return dxs, x0s

# def _track_frame(evt):
#     fe , etrk = _track_eframe(evt)
#     dxs, x0s  = _track_xyzframe(evt)
#     xf = np.array(list(dxs) + list((fe,)))
#     zf = np.array(list(x0s) + list((etrk,)))
#     return xf, zf


def image(coors, data, varname, statistics, bins):
    """ return the image of the label of data in coors with given bins
    """
    var          = data[varname]
    xyimg , _, _ = stats.binned_statistic_dd(coors, var,  bins = bins, statistic = statistics)
    xyimg        = np.nan_to_num(xyimg, 0) 
    return xyimg

#-------------
# Algorithms
#-------------


# def xyimg_projections(bins, labels, _data):
#     """ creates 3 projections (x, y), (x, z) and (z, y) 
#     each projection contains the images of labels, i,e label = 'esum'
#     """

#     def _func(event):

#         sel = event.track_id <= track_id
#         data    = _data(event[sel])
#         xycoors = _coors(data, ('x', 'y'))
#         xzcoors = _coors(data, ('x', 'z'))
#         zycoors = _coors(data, ('z', 'y'))

#         xs, zs = [], []
#         for coors in (xycoors, xzcoors, zycoors):
#             xs += [_xyimg(coors, data, label, bins) for label in labels]
#             zs += _true_xyimgs(coors, event[sel], bins)

#         ys = event[sel].binclass.unique()

#         return (xs, ys, zs)
    
#     return _func


# _algorithm = {'levels'      : xyimg_levels,
#               'z'           : xyimg_z,
#               'projections' : xyimg_projections}


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

def arange_include_endpoint(start, stop, step):
    u = np.arange(start, stop, step)
    uu = list(u)
    if (uu[-1] < stop): 
        uu.append(uu[-1] + step)
    return np.array(uu)

#----------
#   Run
#----------

def _ofile(ofile, projection, widths, frame):
    ss = '';
    for si in projection: ss += si
    windows = ''
    for xi in widths: windows += str(int(xi))+'x'
    windows = windows[:-1] 
    return ofile+'_'+ss+'_'+windows+'_'+str(int(frame))


def run(ifilename,
        ofilename,
        projection = ('x', 'y'),
        widths     = (10, 10),
        frame      = 100.,
        labels     = ['esum', 'ecount', 'emean', 'emax', 'estd'],
        nevents    = -1, 
        verbose    = True):



    if (verbose):
        print('input  filename ', ifilename)
        print('output filename ', ofilename)
        print('projections     ', projection)
        print('widths          ', widths)
        print('frame           ', frame)
        print('labels          ', labels)
        print('events          ', nevents)

    assert (len(projection) >= 2) # 2d projection
    assert (len(projection) <= 3) # 3d projection
    assert len(projection) == len(widths)
    assert len(labels) >= 1

    idata   = pd.read_hdf(ifilename, "voxels") 
    
    delta  = np.max(get_frame(idata))
    print('maximum window frame {:4.2f}'.format(delta))
    if (delta > frame):
        print('Error: Unsificient frame width', frame, ', must be ', delta)
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

    ofile = _ofile(ofilename, projection, widths, frame)
    if (verbose):
        print('output file ', ofile)

    save(odata, ofile)

    return odata


#-----------
# Mix
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
    return odata


#---------------
# Plot
#---------------

def plot_imgs(xs, ievt, labels = -1):
    """ plots the images of xdic for the event ievt
    """

    def _img(ki):
        label = labels[ki]
        if (isinstance(xs, dict)): 
            return xs[label][ievt]
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
    save(idata, ofile)
    odata = load(ofile+'.npz')
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
    xdata  = load(_ofile(ofilename, coors, widths, frame)+'.npz')

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

# def test_run_xyimg_projections(ifilename, ofilename = 'temp'):
#     bins  = (8, 8) 
#     odata = run(ifilename, ofilename, xyimg_type = 'projections', bins = bins, nevents= 10)
#     xdata = load(ofilename+'.npz')

#     assert np.all(xdata.y == odata.y)
#     assert np.all(xdata.id == odata.id)
#     for label in xdata.xdic.keys():
#         assert np.all(xdata.xdic[label] == odata.xdic[label])
#     for label in xdata.zdic.keys():
#         assert np.all(xdata.zdic[label] == odata.zdic[label])

#     for evt in range(10):
#         for proj in ('xy', 'xz', 'zy'):
#             esum   = odata.xdic[proj+'_esum'][evt]
#             emean  = odata.xdic[proj+'_ecount'][evt]
#             ecount = odata.xdic[proj+'_emean'][evt]
#             assert np.all(np.isclose(esum, emean * ecount))

#     return True

# def test_run_xyimg_z(ifilename, ofilename = 'temp'):

#     bins  = (8, 8, 4) 
#     odata = run(ifilename, ofilename, xyimg_type = 'z', bins = bins, nevents= 10)
#     xdata = load(ofilename+'.npz')

#     assert np.all(xdata.y == odata.y)
#     assert np.all(xdata.id == odata.id)
#     for label in xdata.xdic.keys():
#         assert np.all(xdata.xdic[label] == odata.xdic[label])
#     for label in xdata.zdic.keys():
#         assert np.all(xdata.zdic[label] == odata.zdic[label])

#     for evt in range(10):
#         esum   = odata.xdic['esum'][evt]
#         emean  = odata.xdic['ecount'][evt]
#         ecount = odata.xdic['emean'][evt]
#         assert np.all(np.isclose(esum, emean * ecount))
#     return True

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


path   = os.environ['LPRDATADIR']

def tests(path = path, pressure = '13bar'):
    sample1  = '0nubb'
    sample2  = '1eroi'
    
    test_voxel_filename(pressure, sample1)
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

    test_mix_godata('temp/sample1_xy_10x10_100.npz', 'temp/sample2_xy_10x10_100.npz', 'temp/test_'+pressure)
    
    print('Passed all tests!')
    return True
