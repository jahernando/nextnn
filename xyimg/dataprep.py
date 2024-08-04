#  Module to produce images of NEXT tracks for NN
#  JA Hernando 15/04/24

import numpy             as np
import pandas            as pd
import random            as random
import glob              as glob
#import sys
#import os
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

#-----------
#  Event Dispatch
#-----------

def _df_nevents_in_file(dfile, label = 'idx'):
    return len(dfile[label].unique())

def _df_generator(filename, key = 'voxels', label = 'idx'):
    df  = pd.read_hdf(filename, key)
    #label = [label,] if type(label) != list else label
    gdf = df.groupby(label)
    return gdf

def evt_generator(root, key = 'voxels', label = 'idx'):
    files      = glob.glob(root)
    nbunches   = len(files)
    print('number of files ', nbunches)
    def _giter(ifile):
        fname = root.replace('*', str(ifile)) if nbunches > 1 else files[0]
        gen = _df_generator(fname, key, label)
        return iter(gen)
    ifile  = 0
    giter  = _giter(ifile)
    while ifile < nbunches:
        try:
            yield next(giter)
        except StopIteration:
            ifile += 1
            if (ifile < nbunches): giter = _giter(ifile)
     
class EvtDispatch:
    """ Dispatches events using the 'idx' index in the DF.
    It computes the total number of events in the *root* files.
    Files are expected to have the same number of events except the last one.
    The 'idx' index should be in increasing order in the files
    Root should have '*' in it and the '*' should be an integer in order of the files, that, is with 3 files '*' can be [0, 1, 2]
    """

    def __init__(self, root):
        self.root     = root
        files         = glob.glob(root)
        nbunches      = len(files)
        self.nbunches = nbunches
        print('number of files ', nbunches)
        if (nbunches == 1):
            self._set_file(0, files[0])
            self.nevents = _df_nevents_in_file(self._file)
            self.nevents_batch = self.nevents
        else:
            nevents_last_bunch = _df_nevents_in_file(self._set_file(nbunches-1))
            nevents_bunch      = _df_nevents_in_file(self._set_file(0))
            self.nevents = (nbunches-1) * nevents_bunch + nevents_last_bunch
        bins = list(range(0, self.nevents, nevents_bunch))
        if (bins[-1] < self.nevents): bins.append(self.nevents)
        self.bins = np.array(bins, dtype = int)
        print('events ', self.nevents)
        print('range  ', self.bins[0], self.bins[-1], ', nbins ', len(self.bins))
        assert len(self.bins) == nbunches + 1
        #self.bins = np.linspace(0, self.nevents, nbunches, endpoint = True, dtype = int)
        return

    def _set_file(self, ifile, fname = ''):
        if (ifile < 0) or (ifile >= self.nbunches):
            raise IndexError('EventDispatch not valid index file '+str(ifile))
        self._ifile = ifile
        fname = self.root.replace('*', str(ifile)) if fname == '' else fname
        self._file  = _df_generator(fname, 'voxels', 'idx')
        return self._file

    def _set_ifile_by_index(self, index):
        ibin = np.digitize(index, self.bins) - 1 
        #print(ibin)
        if (self._ifile != ibin): self._set_file(ibin)
        return ibin

    def __len__(self):
        return self.nevents

    def __getitem__(self, index):
        if (index >= self.nevents) or (index < 0):
            raise IndexError('Evtdispatch not valid index '+str(index))
        self._set_ifile_by_index(index)
        kevt   = self._file.get_group(index)
        if len(kevt) <= 0: 
            raise IndexError('EvtDispatch empty event with index '+str(index))
        return kevt

#-----------
#  Image Dispatch
#-----------

coor_labels = ('x', 'y', 'z')

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


class ImgDispatch():

    def __init__(self, evtdispatch, label, width, frame):
        self.evtdispatch = evtdispatch
        self.label = label
        self.bins  = [np.arange(-frame - w, frame + w, w) for w in (width, width, width)]

    def __len__(self):
        return len(self.evtdispatch)
    
    def __getitem__(self, index):

        evt = self.evtdispatch[index]
        x   = evt_image(evt, self.label, bins = self.bins)
        y   = int(evt['binclass'].unique())
        return x, y

def plot_img(x, y, label):

    print('target ', y)
    plt.figure()
    for i, lab in enumerate(label):
        plt.subplot(2, 2, i+1); plt.imshow(x[i]); plt.colorbar(); plt.title(lab)
    plt.tight_layout()

    return


# #--------------
# # Frame
# #--------------

# def get_frame(idata, track_id = 0):
#     """ returns the size in each dimention of the main track of all the events in the h5 file voxel data.
#     """
#     sel   = idata.track_id == track_id
#     dmax  = idata[sel].groupby(['file_id', 'event']).max()
#     dmin  = idata[sel].groupby(['file_id', 'event']).min()
#     delta = dmax - dmin
#     dx, dy, dz  = np.max(delta.x), np.max(delta.y), np.max(delta.z)
#     return dx, dy, dz

# def test_frame(ifilename):
#     idata  = pd.read_hdf(ifilename, "voxels") 
#     delta  = np.max(get_frame(idata))
#     for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
#         if (i >= 10): break
#         sel = evt.track_id == 0
#         us  = (evt[sel].x, evt[sel].y, evt[sel].z)
#         dus = [np.max(x) - np.min(x) for x in us]
#         assert np.all(np.array(dus) <= np.array(delta))
#     return True


#------

# labels = ['x', 'y', 'z', 'E', 'nie', 'file_id', 'event', 'track_id', 'hit_id', 'binclass', 'segclass', 'ext']

# def evt_ielectrons(evt, wi = wi, width = hit_width):

#     ene         = evt.E.values
#     nie         = np.maximum(np.round(ene/wi), 1).astype(int)
#     evt['nie']  = nie
#     evt['hit_id'] = np.arange(len(ene))

#     dd = {}
#     for label in labels:
#         dd[label] = np.repeat(evt[label].values, nie)

#     niesum = np.sum(nie)

#     for label, wd in zip(('x', 'y', 'z'), hit_width):
#         dd[label] = dd[label] + wd * np.random.uniform(-0.5, 0.5, size = niesum)

#     dd['E']   = dd['E']/dd['nie']
#     dd['nie'] = 1

#     return pd.DataFrame(dd)


# def test_evt_ielectron(evt):

#     dfie = evt_ielectrons(evt)
#     nie  = np.maximum(np.round(evt.E.values/wi), 1).astype(int)

#     assert len(dfie) == np.sum(nie)

#     for i in range(len(evt)):
#         ievt = evt[evt.hit_id   == i]
#         dii  = dfie[dfie.hit_id == i]

#         assert np.isclose(np.sum(dii.E), ievt.E)
#         assert ievt.nie.values[0] == len(dii)

#         def _check_label(label):
#             assert ievt[label].unique()[0] == dii[label].unique()[0]
#         for label in ['file_id', 'event', 'track_id', 'hit_id', 'segclass', 'ext']:
#             _check_label(label)

#         for label, wd in zip(('x', 'y', 'z'), hit_width):
#             diff = dii[label].values - ievt[label].values
#             assert np.all(np.abs(diff) <= 0.5 * wd)

#     return True


# def evt_ielectrons_diffuse(dfie, sigma, copy = False):

#     xdfie = dfie.copy() if copy else dfie

#     nsize = len(xdfie)

#     for label, s in zip(('x', 'y', 'z'), sigma):
#         xdfie[label] += s * np.random.normal(0, 1, size = nsize)

#     return xdfie

# def test_evt_ielectrons_diffuse(df, sigma = (1, 2, 3)):

#     sdf = evt_ielectrons_diffuse(df, sigma, copy = True)

#     for label, s in zip(('x', 'y', 'z'), sigma):
#         dif =  sdf[label] - df[label] 
#         assert np.all(np.isclose(np.mean(dif), 0.,  atol = s/10.))
#         assert np.all(np.isclose(np.std(dif ),  s,  atol = s/10.))

#     return True

# def evt_voxelize(evt, bins = -1, frame = 100, width = (10, 10)):

#     return


# xlabel = ['xy_E_sum', 'yz_E_sum', 'zx_E_sum']
# zlabel = ['xy_segclass_max', 'xy_ext_max',
#           'yz_segclass_max', 'yz_ext_max',
#           'zx_segclass_max', 'zx_ext_max']


# def run(ifilename,
#         ofilename,
#         hit_width  = (0, 0, 0),
#         sigma      = (0, 0, 0),
#         width      = (10, 10, 10),
#         frame      = 100.,
#         xlabel     = xlabel,
#         zlabel     = zlabel,
#         nevents    = 10, 
#         verbose    = True):

#     # check inputs    
#     assert len(sigma) == 3
#     assert len(width) == 3
#     for label in xlabel : assert label.split('_')[0] in ['xy', 'yz', 'zx', 'xyz', 'yzx', 'zxy']
#     for label in xlabel : assert label.split('_')[1] in ['x', 'y', 'z', 'E']
#     for label in xlabel : assert label.split('_')[2] in ['min', 'max', 'mean', 'sum', 'std', 'count']
#     for label in zlabel : assert label.split('_')[0] in ['xy', 'yz', 'zx', 'xyz', 'yzx', 'zxy']
#     for label in zlabel : assert label.split('_')[1] in ['segclass', 'ext']
#     for label in zlabel : assert label.split('_')[2] in ['min', 'max', 'mean']
#     assert np.min(hit_width) >= 0.
#     assert np.min(sigma) >= 0.
#     assert np.min(width) > 0.
#     assert frame > 2. * np.min(width)

#     t0 = time.time()
#     if (verbose):
#         print('input  filename      ', ifilename)
#         print('output filename      ', ofilename)
#         print('hit widths      (mm) ', hit_width)
#         print('wi              (eV) ', wi * 1e5)
#         print('sigma diffusion (mm) ', sigma)
#         print('widths          (mm) ', width)
#         print('frame           (mm) ', frame)
#         print('xlabel               ', xlabel)
#         print('zlabel               ', zlabel)
#         print('events               ', nevents)

#     do_ie_dist     = np.sum(hit_width) > 0
#     do_ie_smearing = np.sum(sigma)     > 0
#     do_ie          = do_ie_dist or do_ie_smearing

#     print('do ie            ', do_ie)
#     print('do ie inside hit ', do_ie_dist)
#     print('do ie diffusion  ', do_ie_smearing)

#     bins   = [np.arange(-frame, frame, w) for w in width]
#     print('image pixel size in frame ', [len(b) for b in bins])

#     def _evt_ie(evt):
#         evt   = evt_preparation(evt)
#         dfie  = evt_ielectrons(evt, width = hit_width)
#         if (do_ie_smearing):
#             dfie = evt_ielectrons_diffuse(dfie, sigma = sigma)
#         shot  = evt_shot(xdf = dfie, zdf = evt,
#                          xlabel = xlabel, zlabel = zlabel,
#                          bins = bins)
#         return shot

#     def _evt_hit(evt):
#         evt   = evt_preparation(evt)
#         shot  = evt_shot(xdf = evt, zdf = evt,
#                          xlabel = xlabel, zlabel = zlabel,
#                          bins = bins)
#         return shot

#     _evt = _evt_ie if do_ie else _evt_hit

#     ta = time.time()
#     i = 0
#     ifilename = [ifilename,] if isinstance(ifilename, str) else ifilename
#     odata = godata_init(xlabel, zlabel)
#     for k, ifile in enumerate(ifilename):
#         print('opening voxel file : ', ifile)
#         idata = pd.read_hdf(ifile, 'voxels')
#         for evtid, evt in idata.groupby(['file_id', 'event']):
#             i += 1
#             if (nevents > 0) & (i > nevents): break
#             if i % 100 == 0: print('processing event ', i, ', id ', evtid)
#             godata_append(odata, _evt(evt))
#     print('save godata filename :', ofilename)
#     godata_save(odata, ofilename)

#     t1 = time.time()
#     print('event processed   {:d} '.format(i))
#     print('time per event    {:4.2f} s'.format((t1-ta)/i))
#     print('time execution    {:8.1f}  s'.format(t1-t0))
#     print('done!')

#     return odata


#--------------------------------------------------------


#--------------
# Tests
#--------------


# #------------

# def test_run(ifilename, ofilename):
#     projection = ['xy', 'xz', 'zy']
#     xlabel     = ['E_sum', 'E_count', 'E_mean']
#     zlabel     = ['segclass_mean', 'ext_max']
#     sigma      = (2, 2, 2)
#     width      = (10, 10, 10)
#     frame      = 60
#     odata      = run(ifilename, ofilename, sigma = sigma, width = width, frame = frame,
#                      projection = projection, xlabel = xlabel , zlavel = zlabel, nevents= 10)
#     xdata      = godata_load(ofilename)

#     assert np.all(xdata.y  == odata.y)
#     assert np.all(xdata.id == odata.id)
#     for label in xdata.xdic.keys():
#         assert np.all(xdata.xdic[label] == odata.xdic[label])
#     for label in xdata.zdic.keys():
#         assert np.all(xdata.zdic[label] == odata.zdic[label])

#     for evt in range(10):
#         for pro in projection:
#             esum   = odata.xdic[pro+'_E_sum'][evt]
#             emean  = odata.xdic[pro+'_E_count'][evt]
#             ecount = odata.xdic[pro+'_E_mean'][evt]
#             assert np.all(np.isclose(esum, emean * ecount))
#     return True

