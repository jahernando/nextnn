#  Module: study the extremes of the tracks
#  JA Hernando 04/06/24

import numpy             as np
import pandas            as pd
import random            as random
#import os

import xyimg.dataprep          as dp

#from   scipy       import stats

#import matplotlib.pyplot as plt

#from collections import namedtuple



#---------------------
#  Input Output files
#---------------------

#---------------
# Utils
#----------------


#----------
#   Run
#----------

def _voxel_position(evt, i):
    return np.array((evt.x.values[i], evt.y.values[i], evt.z.values[i]))

def _voxel_blob_index(evt, i = 1):
    id1 = np.argwhere(evt.ext.values == i)
    if len(id1) != 1: return -1
    return int(id1[0])

def _voxels_in_radius(evt, x0, radius):
    dx, dy, dz = evt.x.values - x0[0], evt.y.values - x0[1], evt.z.values - x0[2]
    dd  = np.sqrt(dx*dx + dy*dy + dz*dz)
    sel = dd <= radius
    return sel

def _voxels_data(evt, sel, prefix = ''):
    dd = {}
    dd[prefix + 'nhits'] = sum(sel)
    dd[prefix + 'ene']   = sum(evt.E[sel])
    return dd

def _evt_blobs(evt, radius):

    dd = {}

    i1 = _voxel_blob_index(evt, 1)
    i2 = _voxel_blob_index(evt, 2)
    if (i1 < 0) or (i1 < 0): return dd

    dd['ext1seg'] = evt.segclass.values[i1]
    dd['ext2seg'] = evt.segclass.values[i2]
    
    for i, id in enumerate((i1, i2)): 
        x    = _voxel_position(evt, id)
        sel  = _voxels_in_radius(evt, x, radius)
        dd   = {**dd, **_voxels_data(evt, sel, 'blob'+str(i+1))}

    xx  = _voxel_position(evt, i2) - _voxel_position(evt, i1)
    dis = np.sqrt(np.sum(xx*xx))
    d2  = np.sqrt(xx[0]**2 + xx[1]**2)
    dz  = xx[2]
    dd['blobsdist'] = dis
    dd['blobsdxy']  = d2
    dd['blobsdz']   = np.abs(dz)
                 
    return dd

def _evt_general(evt):

    dd = {}

    dd['file_id'] = evt.file_id.unique()[0]
    dd['event']   = evt.event.unique()[0]
    nsize         = len(evt.event)
    dd            = {**dd, **_voxels_data(evt, np.ones(nsize, bool), 'evt')}
    sel           = evt.track_id == 0
    dd            = {**dd, **_voxels_data(evt, sel, 'trk')}

    return dd

def run(ifilename,
        ofilename,
        radius     = 6,
        nevents    = -1, 
        verbose    = True):
    
    if (verbose):
        print('input  filename ', ifilename)
        print('output filename ', ofilename)
        print('radius          ', radius)
        print('events          ', nevents)

    idata   = pd.read_hdf(ifilename, "voxels") 
    nsize   = len(idata.groupby(['file_id', 'event']).size())
    if (verbose):
        print('input  number of events : ', nsize)

    nevents = nsize if nevents >= nsize else nevents 
    nsize   = nsize if nevents <= -1 else nevents
    if (verbose):
        print('output number of events : ', nsize)

    df = {}
    def _fill(data, i):
        for name in data.keys():
            if (name not in df.keys()):
                df[name] = np.zeros(nsize, type(data[name]))
        for name in data.keys(): df[name][i] = data[name]
        return
     
    for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if ((nevents >= 1) & (i >= nevents)): break

        edata  = _evt_general(evt)
        _fill(edata, i)
        bdata = _evt_blobs(evt, radius)
        _fill(bdata, i)

    df = pd.DataFrame(df)
    if (verbose):
        print('save data at ', ofilename)
    df.to_hdf(ofilename, 'df', mode = 'w')    

    return df


#-----------
# Production
#------------

def production(path, pressure, radius,  nevents = -1):
    ofiles = []
    for sample in ('0nubb', '1eroi'):
        ifile = path + dp.voxel_filename(pressure, sample)
        ofile = path + 'extana/' + dp.str_concatenate(('extana', pressure, sample, 'radius' + str(radius))) + '.h5'
        run(ifile, ofile, radius, nevents) 
    return True

#---------------
# Plot
#---------------



#--------------------------------------------------------


#--------------
# Tests
#--------------


def tests(path):


    print('Passed all tests!')
    return True
