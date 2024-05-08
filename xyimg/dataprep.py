#  Module to produce images of NEXT tracks for NN
#  JA Hernando 15/04/24

import numpy             as np
import pandas            as pd
import random            as random
import os

from   scipy       import stats

import matplotlib.pyplot as plt

from collections import namedtuple

GoData = namedtuple('GoData', ['xdic', 'y', 'zdic', 'id'])

path   = os.environ['LPRDATADIR']
#path = "/Users/hernando/work/investigacion/NEXT/data/NEXT100/pressure_topology/"

def voxel_filename(pressure, sample):
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

def normarange(var: np.array) -> np.array:
    """ set the variable into a fraction sum(var)
    """
    norma = sum(var)
    return var/norma

def _data(evt):
    xs, ys, zs = [urange(x) for x in [evt.x, evt.y, evt.z]]
    es         = normarange(evt.E)
    data       = {'x':xs, 'y': ys, 'z':zs, 'e':es}
    return data

def _coors(data, coorsnames):
    coors      = tuple(data[c] for c in coorsnames)
    return coors

def _xyimg(coors, data, label, bins):
    varname      = label[0]
    statistics   = label[1:]
    #print('label ', label, ', varname', varname, ', statistics ', statistics)
    var          = data[varname]
    xyimg , _, _ = stats.binned_statistic_dd(coors, var,  bins = bins, statistic = statistics)
    xyimg        = np.nan_to_num(xyimg, 0) 
    return xyimg

def _true_xyimgs(coors, event, bins):
    _seg = np.array([2, 1, 3])
    seg = [_seg[x] for x in event.segclass.values]
    ext = event.ext.values > 0
    text, _, _ = stats.binned_statistic_dd(coors, ext,  bins = bins, statistic = 'max')
    tseg, _, _ = stats.binned_statistic_dd(coors, seg,  bins = bins, statistic = 'max')
    timgs = [np.nan_to_num(img, 0).astype(int) for img in (text, tseg)]
    return timgs

#-------------
# Algorithms
#-------------

def xyimg_levels(bins, labels, track_id = 0):
    """ created (x, y) images (1 projection only) of labels (i.e 'esum', 'emax', ...)
    """

    def _func(event):

        sel   = event.track_id <= track_id
        data  = _data(event[sel])
        coors = _coors(data, ('x', 'y'))

        xs = [_xyimg(coors, data, label, bins) for label in labels]
        ys = event.binclass.unique()
        zs = _true_xyimgs(coors, event[sel], bins)

        return (xs, ys, zs)

    return _func

def xyimg_z(bins, labels, track_id = 0):
    """ creates a 3D voxelized 'image', the depth of the (x, y) image is z
    bins must be a 3-element tuple of ints
    """

    def _func(event):

        sel   = event.track_id <= track_id
        data  = _data(event[sel])
        coors = (data['x'], data['y'], data['z'])

        xs = [_xyimg(coors, data, label, bins) for label in labels]
        ys = event.binclass.unique()
        zs = _true_xyimgs(coors, event[sel], bins)

        return (xs, ys, zs)

    return _func

def xyimg_projections(bins, labels, track_id = 0):
    """ creates 3 projections (x, y), (x, z) and (z, y) 
    each projection contains the images of labels, i,e label = 'esum'
    """

    def _func(event):

        sel = event.track_id <= track_id
        data    = _data(event[sel])
        xycoors = _coors(data, ('x', 'y'))
        xzcoors = _coors(data, ('x', 'z'))
        zycoors = _coors(data, ('z', 'y'))

        xs, zs = [], []
        for coors in (xycoors, xzcoors, zycoors):
            xs += [_xyimg(coors, data, label, bins) for label in labels]
            zs += _true_xyimgs(coors, event[sel], bins)

        ys = event[sel].binclass.unique()

        return (xs, ys, zs)
    
    return _func


_algorithm = {'levels'      : xyimg_levels,
              'z'           : xyimg_z,
              'projections' : xyimg_projections}

# def get_func_event_xyimg(bins, coorsnames, labels, track_id = 0):

#     def _data(evt):
#         xs, ys, zs = [urange(x) for x in [evt.x, evt.y, evt.z]]
#         es         = normarange(evt.E)
#         data       = {'x':xs, 'y': ys, 'z':zs, 'e':es}
#         coors      = tuple(data[c] for c in coorsnames)
#         return coors, data

#     def _xyimg(coors, data, label):
#         varname      = label[0]
#         statistics   = label[1:]
#         #print('label ', label, ', varname', varname, ', statistics ', statistics)
#         var          = data[varname]
#         xyimg , _, _ = stats.binned_statistic_dd(coors, var,  bins = bins, statistic = statistics)
#         xyimg        = np.nan_to_num(xyimg, 0) 
#         return xyimg

#     def _true_xyimgs(coors, event):
#         _seg = np.array([2, 1, 3])
#         seg = [_seg[x] for x in event.segclass.values]
#         ext = event.ext.values > 0
#         text, _, _ = stats.binned_statistic_dd(coors, ext,  bins = bins, statistic = 'max')
#         tseg, _, _ = stats.binned_statistic_dd(coors, seg,  bins = bins, statistic = 'max')
#         timgs = [np.nan_to_num(img, 0).astype(int) for img in (text, tseg)]
#         return timgs

#     def get_event_xyimg(event):

#         sel = event.track_id <= track_id
#         coors, data = _data(event[sel])

#         xs = [_xyimg(coors, data, label) for label in labels]
#         ys = event.binclass.unique()
#         zs = _true_xyimgs(coors, event[sel])

#         return (xs, ys, zs)

#     return get_event_xyimg


#----------
#   Run
#----------

def _xlabels(xyimg_type, labels):
    if (xyimg_type == 'projections') : 
        xlabels = []
        for proj in ('xy', 'xz', 'zy'):
            xlabels += [proj +'_'+label for label in labels]
        return xlabels
    return labels
         
def _zlabels(xyimg_type):
    return _xlabels(xyimg_type, ('ext', 'seg'))

def run(ifilename,
        ofilename,
        xyimg_type  =  "levels",
        bins        = (8, 8),
        labels      = ['esum', 'ecount', 'emean', 'emax', 'estd'],
        track_id    = 0,
        nevents     = -1):

    print('Input voxel file:', ifilename)
    idata        = pd.read_hdf(ifilename, "voxels") 
    assert xyimg_type in _algorithm.keys()
    algorithm    = _algorithm[xyimg_type]
    event_xyimgs = algorithm(bins, labels, track_id)

    def _dinit(labels):
        xdic = {}
        for label in labels: xdic[label] = []
        return xdic
    
    def _dappend(dic, vals):
        for i, label in enumerate(dic.keys()): dic[label].append(vals[i])

    xdic  = _dinit(_xlabels(xyimg_type, labels))
    zdic  = _dinit(_zlabels(xyimg_type))
    y, id = [], []

    for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if ((nevents >= 1) & (i >= nevents)): break
        xi, yi, zi = event_xyimgs(evt)
        file_id    = int(evt.file_id.unique())
        event_id   = int(evt.event.unique())
        idi        = np.array((file_id, event_id))
        _dappend(xdic, xi)
        _dappend(zdic, zi)
        y.append(yi)
        id.append(idi)

    odata = GoData(xdic, y, zdic, id)

    save(odata, ofilename)

    return odata

#----------
# Save and Load
#----------

def save(odata, ofilename):
    print('Output file : ', ofilename)
    xlabel = list(odata.xdic.keys())
    x = np.array([np.array(odata.xdic[label]) for label in xlabel])
    y = np.array(odata.y)
    zlabel = list(odata.zdic.keys())
    z = np.array([np.array(odata.zdic[label]) for label in zlabel])
    id = np.array(odata.id)
    np.savez(ofilename, x = x, y = y, z = z, id = id, 
             xlabel = np.array(xlabel), zlabel = np.array(zlabel))
    return

def load(ifilename):
    
    data = np.load(ifilename)
    x, y, z, id = data['x'], data['y'], data['z'], data['id']
    xlabel, zlabel = list(data['xlabel']), list(data['zlabel'])

    xdic = {}
    for i, label in enumerate(xlabel): xdic[label] = x[i]
    zdic = {}
    for i, label in enumerate(zlabel): zdic[label] = z[i]

    odata = GoData(xdic, y, zdic, id)
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
    save(odata, ofilename)
    return odata


#---------------
# Plot
#---------------

def plot_imgs(xs, ievt, labels = -1):

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

def test_normarange(x):
    ux   = normarange(x)
    sum  = np.sum(x)
    assert  np.sum(ux) == 1
    iar  = np.argmax(x)
    assert np.isclose(ux[iar], x[iar]/sum)
    iar  = np.argmin(x)
    assert np.isclose(ux[iar], x[iar]/sum)
    return True

def test_xyimg_levels(ifilename):
    bins   = (8, 8)
    labels = ('esum', 'ecount', 'emean')
    print('Input voxel file:', ifilename)
    idata        = pd.read_hdf(ifilename, "voxels") 
    event_xyimgs = xyimg_levels(bins, labels)
    for i, (_, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if (i >= 5): break
        xi, _, zi = event_xyimgs(evt)
        assert np.all(np.isclose(xi[0], xi[1] * xi[2]))
        assert (np.sum(zi[0]) >= 1) & (np.sum(zi[0]) <= 2)
        assert np.any(zi[1] == 3)
        assert np.max(zi[1]) <= 3
    return True

def test_xyimg_projections(ifilename):
    bins   = (8, 8)
    labels = ('esum', 'ecount', 'emean')
    print('Input voxel file:', ifilename)
    idata        = pd.read_hdf(ifilename, "voxels") 
    event_xyimgs = xyimg_projections(bins, labels)
    for i, (_, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if (i >= 5): break
        xi, _, zi = event_xyimgs(evt)
        for i in range(3):
            k = 3*i
            assert np.all(np.isclose(xi[k+0], xi[k+1] * xi[k+2]))
        for i in range(3):
            assert (np.sum(zi[2*i]) >= 1) & (np.sum(zi[2*i]) <= 2)
            assert np.any(zi[2*i + 1] == 3)
            assert np.max(zi[2*i +1]) <= 3
    return True

def test_xyimg_z(ifilename):
    bins   = (8, 8, 4)
    labels = ('esum', 'emax', 'emean')
    print('Input voxel file:', ifilename)
    idata        = pd.read_hdf(ifilename, "voxels") 
    event_xyimgs = xyimg_z(bins, labels)
    for i, (_, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if (i >= 5): break
        xi, _, zi = event_xyimgs(evt)
        assert np.isclose(np.sum(xi[0]), 1.)
        assert np.sum(xi[1]) <= 1.
        assert np.sum(xi[2]) <= 1.
        assert np.all(xi[0] >= xi[1])
        assert np.all(xi[0] >= xi[2])
        assert np.all(xi[1] >= xi[2])
        assert (np.sum(zi[0]) >= 1) & (np.sum(zi[0]) <= 2)
        assert np.any(zi[1] == 3)

    return True

def test_run_xyimg_levels(ifilename, ofilename = 'temp'):
    bins  = (8, 8) 
    odata = run(ifilename, ofilename, bins = bins, nevents= 10)
    xdata = load(ofilename+'.npz')

    assert np.all(xdata.y == odata.y)
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

def test_run_xyimg_projections(ifilename, ofilename = 'temp'):
    bins  = (8, 8) 
    odata = run(ifilename, ofilename, xyimg_type = 'projections', bins = bins, nevents= 10)
    xdata = load(ofilename+'.npz')

    assert np.all(xdata.y == odata.y)
    assert np.all(xdata.id == odata.id)
    for label in xdata.xdic.keys():
        assert np.all(xdata.xdic[label] == odata.xdic[label])
    for label in xdata.zdic.keys():
        assert np.all(xdata.zdic[label] == odata.zdic[label])

    for evt in range(10):
        for proj in ('xy', 'xz', 'zy'):
            esum   = odata.xdic[proj+'_esum'][evt]
            emean  = odata.xdic[proj+'_ecount'][evt]
            ecount = odata.xdic[proj+'_emean'][evt]
            assert np.all(np.isclose(esum, emean * ecount))

    return True

def test_run_xyimg_z(ifilename, ofilename = 'temp'):

    bins  = (8, 8, 4) 
    odata = run(ifilename, ofilename, xyimg_type = 'z', bins = bins, nevents= 10)
    xdata = load(ofilename+'.npz')

    assert np.all(xdata.y == odata.y)
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


def tests(path = path, pressure = '13bar'):
    sample1  = '0nubb'
    sample2  = '1eroi'
    
    test_voxel_filename(pressure, sample1)
    test_urange(np.arange(30))
    test_normarange(np.arange(30))
    
    ifilename1 = path + voxel_filename(pressure, sample1)
    test_xyimg_levels(ifilename1)
    test_xyimg_projections(ifilename1)
    test_xyimg_z(ifilename1)

    test_run_xyimg_levels(ifilename1, 'test_levels_sample1')
    test_run_xyimg_levels(ifilename1, 'test_projections_sample1')
    test_run_xyimg_z(ifilename1, 'test_z_sample1')

    ifilename2 = path + voxel_filename(pressure, sample2)
    test_run_xyimg_levels(ifilename2, 'test_levels_sample2')
    test_mix_godata('test_levels_sample1.npz', 'test_levels_sample2.npz', 'test_'+pressure)
    print('Passed all tests!')
    return True
