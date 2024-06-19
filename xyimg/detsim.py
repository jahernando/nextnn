#  DetSim
#  Module to smeare the MC hits along the tracks and re-voxel
#  Krisham Mistry, JA Hernando 18/06/24

import numpy             as np
import pandas            as pd
from   scipy       import stats
#import os

import xyimg.dataprep          as dp


wi         = 25.6 # eV Scintillation threshold
voxel_size =  2.0 # mm (voxel size)  
DL         = 0.278 # mm / sqrt(cm)
DT         = 0.272 # mm / sqrt(cm)
sigma_L    = 5. # mm
sigma_T    = 4. # mm


def ielectrons(position, energy, widths, sigmas, wi = wi):

    def _ie(pos, ene):
        nsize = int(np.round(1e6*ene/wi)) # E is in MeV and wi in eV
        def _smear(xi, width, sigma): 
            x  = xi * np.ones(nsize)
            x += width * np.random.uniform(-0.5, 0.5, size = nsize)
            x += sigma * np.random.normal(size = nsize)
            return x
        spos = [_smear(x, width, sigma) for x, width, sigma in zip(pos, widths, sigmas)]
        sene = ene * np.ones(nsize) / nsize
        return spos, sene

    vals   = [_ie((x, y, z), ene) for x, y, z, ene in zip(*position, energy)]
    ie_pos = [np.concatenate([v[0][i] for v in vals]) for i in range(3)]
    ie_ene =  np.concatenate([v[1] for v in vals])

    return ie_pos, ie_ene

        
def voxelize(position, ene, widths):
    bins        = [np.arange(np.min(x) - 2.*width, np.max(x) + 2.*width, width) for x, width in zip(position, widths)]
    img , _, _  =  stats.binned_statistic_dd(position, ene,  bins = bins, statistic = 'sum')
    ximg        = [stats.binned_statistic_dd(position,   x,  bins = bins, statistic = 'mean')[0] for x in position]
           
    sel    = img > 0
    xene   = img[sel].flatten()
    xs     = [x[sel].flatten() for x in ximg]
    return xs, xene, bins, sel
    
def _segclass(seg):
    _seg  = np.array([2, 1, 3])
    return [_seg[x] for x in seg]

def val_in_frame(coors, val, bins, sel, statistic = 'max'):
    img , _, _  =  stats.binned_statistic_dd(coors, val,  bins = bins, statistic = statistic)
    xval = img[sel].flatten()
    np.nan_to_num(xval, 0)
    return xval


def event_smear(evt, width0, sigma, width1, wi = wi):

    pos      = [evt[label].values for label in ('x', 'y', 'z')]
    ene      =  evt['E'].values
    segclass = _segclass(evt['segclass'].values)
    ext      = evt['ext'].values
    trkid    = evt['track_id'].values
    nhits    = evt['nhits'].values

    ie_pos, ie_ene        = ielectrons(pos, ene, width0, sigma, wi)
    xpos, xene, bins, sel = voxelize(ie_pos, ie_ene, width1)
    xsegclass             = val_in_frame(pos, segclass, bins, sel).astype(int)
    xext                  = val_in_frame(pos, ext, bins, sel).astype(int)
    xtrkid                = val_in_frame(pos, trkid, bins, sel, 'min').astype(int)
    xnhits                = val_in_frame(pos, nhits, bins, sel, 'sum').astype(int)

    dd = {'x' : xpos[0], 'y' : xpos[1], 'z' : xpos[2], 'E' : xene,
         'segclass' : xsegclass, 'ext' : xext, 'track_id' : xtrkid, 'nhits' : xnhits}

    nsize = len(xene)
    for label in ('file_id', 'event', 'binclass'):
        dd[label] = np.ones(nsize, int) * int(np.unique(evt[label].values))

    return dd


def run(ifilename, ofilename, width0, sigma, width1, wi = wi, verbose = True, nevents = -1):

    
    if (verbose):
        print('input  filename ', ifilename)
        print('output filename ', ofilename)
        print('width-0    (mm) ', width0)
        print('sigmas     (mm) ', sigma)
        print('widths-1   (mm) ', width1)
        print('wi         (eV) ', wi)
        print('events          ', nevents)

    idata   = pd.read_hdf(ifilename, "voxels") 

    def _dinit(labels):
        xdic = {}
        for label in labels: xdic[label] = []
        return xdic
    
    def _dappend(dic, idic):
        for label in dic.keys(): dic[label].append(idic[label])
        
    def _dconcat(dic):
        for label in dic.keys(): dic[label] = np.concatenate(dic[label])
        return dic

    labels = ('file_id', 'event', 'binclass', 'x', 'y', 'z', 'E', 'track_id', 'nhits', 'ext', 'segclass')
    dd     = _dinit(labels)

    for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if ((nevents >= 1) & (i >= nevents)): break

        ddi = event_smear(evt, width0, sigma, width1, wi)
        _dappend(dd, ddi)
    
    df = pd.DataFrame(_dconcat(dd))

    df.to_hdf(ofilename, 'voxels')

    return df

    








                


