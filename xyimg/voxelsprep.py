#  Module to produce pandas files with voxels (centered) and with mixed classes
#  JA Hernando 02/08/24

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

#-----------
#   I/O
#-----------

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

def test_filename_voxel(pressure, sample):
    filename = filename_voxel(pressure, sample)
    assert filename == "voxel_dataset_" + pressure + "_" + sample + ".h5"
    return True

#----------
# Generator
#----------

def get_keys(gs, nevents = -1):
    """ get a list of (i, group_key) for a list of grouped data-frames
    """
    keys = [(i, g[0]) for i, gi in enumerate(gs) for g in gi] if nevents <= 0 else []
    if (nevents <= 0): return keys
    for i, gi in enumerate(gs):
        for n, g in enumerate(gi):
            if (n >= nevents):  break
            keys.append( (i, g[0]) )
    return keys

def evt_iter(gs, nevents = -1, shuffle = False):
    keys  = get_keys(gs, nevents = nevents)
    if shuffle: random.shuffle(keys)
    nsize = len(keys)
    i = 0
    while i < nsize:
        key = keys[i]
        k, kkey = key
        i += 1
        data = gs[k].get_group(kkey) 
        yield key, data


#-----------
#   Algorithms
#-----------

coor_labels = ('x', 'y', 'z')

def coors(df, labels = ('x', 'y', 'z')):
    return [df[label].values for label in labels]

track_id = 0

def evt_coors(evt):
    return [evt[x].values for x in coor_labels]


def evt_preparation(evt      : pd.DataFrame,
                    track_id : int = track_id) -> pd.DataFrame:
    """ Center the x, y, z position of the hits in the center of the track with track_id
        re-arrange the segclass values: returns 1-track, 2-other (i.e delta e), 3-blob
        re-arrange ext values: returns 1 for main extreme (blob) and 2 for second extreme (initial part of the track for e, blob for bb)
    Arguments:
        - evt:  DF, it should have 'x', 'y', 'z' (mm), 'E' (MeV), segclass, ext columns
        - track_id, int, default 0 is the main track
        - segclass
        - ext
    Return:
        _ evt: DataFrame
    """
    sel    = evt.track_id == track_id
    x0s    = [np.mean(evt[sel][label]) for label in coor_labels]

    # center the event around the main track center
    xevt   = evt.copy()
    for label, x0 in zip(coor_labels, x0s):
        xevt[label] = evt[label].values - x0

    # change the segmentation (1-track, 2-delta electron, 3-blob)
    _seg  = np.array([2, 1, 3])
    xevt['segclass'] = [_seg[x] for x in xevt.segclass.values]

    # change the ext (1-deposition, 2-main extreme, 3-minor extreme)
    ext   = evt['ext'].values
    trk   = (evt['segclass'].values >= 0).astype(int)
    xevt['ext'] = ext + trk

    return xevt


def test_evt_preparation(evt, track_id = track_id):

    sel  = evt.track_id == 0
    xs   = evt_coors(evt[sel])
    xs   = [np.mean(x) for x in xs]
    assert np.all(np.isclose(xs, 0.))

    segs = evt.segclass.unique()
    assert np.all(segs >= 1) and np.all(segs <= 3)
    
    ext  = evt.ext.unique()
    assert np.all(ext >= 1) and np.all(ext <= 3)

    return True


def evt_image(df    : pd.DataFrame, 
              label : list[str],
              width : float = 5,
              frame : float = 100,
              bins  : int = -1) -> np.array:
    """ 
    """
    # define the bins if the client has not defined
    if (bins == -1):
        bins = [np.arange(-frame - w, frame + w, w) for w in (width, width, width)]

    dbins = {}
    for xvar, bin in zip(coor_labels, bins): dbins[xvar] = bin
        
    # create a image of a label (if label == 'E' normalize to the total event)
    # the label has tree words separared by '_' i.e 'xy_E_sum', in general 'projection_var_statistic'
    #   xy: inficates the projections
    #   E:  indicates the variable
    #   sum: indicates the statistics
    def _img(df, ilabel):
        projection, varname, statistic = ilabel.split('_') 
        xcoors     = [df[xvar].values for xvar in projection]
        xbins      = [dbins[xvar]     for xvar in projection]
        var        = df[varname].values
        img , _, _ = stats.binned_statistic_dd(xcoors, var,  bins = xbins, statistic = statistic)
        img        = np.nan_to_num(img, 0) 
        return img

    x       = np.array([_img(df, ilabel) for ilabel in label])
    return x


#---- ----
# RUN
#---------

def run(ifilename, 
        ofilename, 
        shuffle    = False,
        nbunch     = 10000,
        nevents    = 10, 
        verbose    = True):

    # check inputs    

    t0 = time.time()
    if (verbose):
        print('input  filename      ', ifilename)
        print('output filename      ', ofilename)
        print('shuffle              ', shuffle)
        print('nbunch               ', nbunch)
        print('events               ', nevents)

    def _save(kdf, ibunch, k):
        print('proceesed bunch ', ibunch)
        print('events in the bunch ', k)
        words = ofilename.split('.')
        smain, stail = words
        ofile = smain + '_bunch' + str(ibunch)+'.'+stail
        kdf.to_hdf(ofile, 'voxels')
        print('saved processed bunch data at:', ofile)
        ibunch += 1
        return ibunch

    def _concat(kdf, kevt, k):
        kdf = pd.concat((kdf, kevt)) if k > 0 else kevt
        k  += 1
        return kdf, k

    # load the data
    assert len(ifilename) == 2
    dfs = [pd.read_hdf(ifile, 'voxels') for ifile in ifilename]
    gs  = [df.groupby(['file_id', 'event']) for df in dfs]

    # loop in the events
    ta = time.time()
    ibunch = 0
    kdf, k = None, 0
    for i, ievt in enumerate(evt_iter(gs, nevents = nevents, shuffle = shuffle)):

        idevt, evt  = ievt

        if  (k > 0) & (k % nbunch == 0): 
            ibunch = _save(kdf, ibunch, k)
            kdf, k = None, 0
        if (i >=0) & (i % 100 == 0):  print('processed event ', i, ', id ', idevt)

        kevt        = evt_preparation(evt)
        kevt['idx'] = i
        kdf, k      = _concat(kdf, kevt, k)
    
    if  (k > 0): 
        ibunch = _save(kdf, ibunch, k)

    t1 = time.time()

    print('events processed   {:d} '.format(i))
    print('bunches processed  {:d} '.format(ibunch))
    print('time per event    {:4.2f}  s'.format((t1-ta)/i))
    print('time execution    {:8.1f}  s'.format(t1-t0))
    print('done!')

    return

#-----------------------
# Plots
#------------------------

def scatter_evt(evt, var, title = '', alpha = 0.1):
    color = var
    plt.figure(); plt.title(title)
    plt.subplot(2, 2, 1); plt.scatter(evt.x, evt.y, c = color, alpha = alpha); plt.colorbar(); plt.xlabel('x'); plt.ylabel('y'); plt.title(title)
    plt.subplot(2, 2, 2); plt.scatter(evt.y, evt.z, c = color, alpha = alpha); plt.colorbar(); plt.xlabel('y'); plt.ylabel('z'); plt.title(title)
    plt.subplot(2, 2, 3); plt.scatter(evt.z, evt.x, c = color, alpha = alpha); plt.colorbar(); plt.xlabel('z'); plt.ylabel('z'); plt.title(title)
    plt.tight_layout()