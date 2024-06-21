#  DetSim
#  Module to smeare the MC hits along the tracks and re-voxel
#  Krisham Mistry, JA Hernando 18/06/24

import time              as time
import numpy             as np
import pandas            as pd
from   scipy       import stats

import matplotlib.pyplot as plt
import os

import xyimg.dataprep          as dp


wi         = 25.6 # eV Scintillation threshold

#   DataFrame verion
#--------------------

def df_ielectrons(dfhit, width, wi = wi):
    """ from a hit DataFrame creates a ionized electrons DF, maintaining the keys
    inputs:
        dfhits: DF, hit information (x, y, z, E) (assumes E is in MeV)
        widht : tuple(float), withs in the (x, y, z) coordinates (mm)
        wi    : float, ionization energy (eV) 25 eV
    """

    df = dfhit.copy()
    df['E'] = 1e6 * df['E']

    nielectron = np.maximum((df.E.values/wi).astype(int), 1)
    df['nielectron'] = nielectron
    labels = list(df.columns) 
    dfie   = pd.DataFrame(np.repeat(df[labels].values, nielectron, axis=0), columns = labels)

    for w, label in zip(width, ('x', 'y', 'z')):
        x0          = dfie[label].values
        dfie[label] = x0 + 0.5 * w * np.random.uniform(0, 1., size = len(x0))

    dfie['E'] = dfie['E']/dfie['nielectron']

    #int_labels = ('file_id', 'event', 'binclass', 'track_id', 'ext', 'segclass', 'nhits', 'nielectron')
    for label, dtype in zip(labels, df.dtypes):
        if (label in ('x', 'y', 'z')): continue
        dfie[label] = dfie[label].values.astype(dtype)

    return dfie

def df_event(df, file_id, event):
    return df[(df.file_id == file_id) & (df.event == event)]

def df_norma(df, sigma, labels = ('x', 'y', 'z')):

    for s, label in zip(sigma, labels):
        x0 = df[label].values
        df[label] = x0 + s * np.random.normal(size = len(x0))

    return df 

def mmbins(x, width):
    """ bins from the minimum to the maximum + width, with width binning
    """
    return np.arange(np.min(x), np.max(x) + width)



def df_voxalize(df, width):

    labels = ('x', 'y', 'z')
    xs     = [df[label].values for label in labels]
    bins   = [mmbins(x, w) for x, w in zip(xs, width)]

    ilabels = ['i'+label for label in labels]
    for ilabel, x, bin in zip(ilabels, xs, bins):
        df[ilabel] = np.digitize(x, bin) 

    dfgroup = df.groupby(ilabels)
    dfvoxel = dfgroup.sum()

    def max_frequency(x):
        vals, counts = np.unique(x, return_counts = True)
        imax = np.argmax(counts)
        return vals[imax]

    def unique(x): return np.unique(x)[0]
    def mean(x)  : return np.mean(x)
    def max(x)   : return np.max(x)
    def count(x) : return len(x)

    opers = {'file_id'  : unique,
             'event'    : unique,
             'track_id' : unique,
             'binclass' : unique,
             'segclass' : max_frequency,
             'nielectron' : count,
             'nhits'    : mean,
             'x'        : mean,
             'y'        : mean,
             'z'        : mean,
             'ix'       : unique,
             'iy'       : unique,
             'iz'       : unique,
             'ext'      : max}

    for label in opers.keys():
        dfvoxel[label]  = dfgroup[label].apply(opers[label])

    return dfvoxel

def df_diff_ievoxel(evt, width0, sigma, width1):
    
    dfie = df_ielectrons(evt, width0)
    dfie = df_norma(dfie, sigma) 
    dvox = df_voxalize(dfie, width1)
    return dvox


# numpy version
#---------------

def ielectrons(pos, ene, width):

    nie = np.maximum(np.round(ene/wi), 1).astype(int)

    def _distribute(xi, wi):
        def _idistribute(xii, nii):
            x  = xii * np.ones(nii)
            x += 0.5 * wi * np.random.uniform(0, 1, size = nii)
            return x
        xs = [_idistribute(xii, nii) for xii, nii in zip(xi, nie)]
        return np.concatenate(xs)

    def _ene():
        def _iene(eneii, nii):
            return eneii/nii * np.ones(nii)
        es = [_iene(eneii, nii) for eneii, nii in zip(ene, nie)]
        return np.concatenate(es)
        
    ie_pos = [_distribute(xi, wi) for xi, wi in zip(pos, width)]
    ie_ene = _ene()

    return ie_pos, ie_ene


def ielectrons_diffuse(pos, sigma):

    nsize = len(pos[0])
    def _smear(x, s):
        return x + s * np.random.normal(0, 1, size = nsize)
    spos = [_smear(x, s) for x, s in zip(pos, sigma)]

    return spos

_stats_binned = stats.binned_statistic_dd

def voxelize(pos, ene, widths):
    bins        = [mmbins(x, w) for x, w in zip(pos, widths)]
    img , _, _  =  _stats_binned(pos, ene,  bins = bins, statistic = 'sum')
    ximg        = [_stats_binned(pos,   x,  bins = bins, statistic = 'mean')[0] for x in pos]
           
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

def diff_ievoxel(evt, width0, sigma, width1):

    labels = ('x', 'y', 'z')
    pos      = [evt[label].values for label in labels]
    ene      =  1.e6*evt['E'].values # convert to eV
    segclass = _segclass(evt['segclass'].values)
    ext      = evt['ext'].values
    trkid    = 1 + evt['track_id'].values
    nhits    = evt['nhits'].values

    ie_pos, ie_ene  = ielectrons(pos, ene, width0)
    ie_pos          = ielectrons_diffuse(ie_pos, sigma)
    xpos, xene, bins, sel = voxelize(ie_pos, ie_ene, width1)
    xnielectron     = val_in_frame(xpos, xene, bins, sel, 'count').astype(int)
    xsegclass       = val_in_frame(pos, segclass, bins, sel).astype(int)
    xext            = val_in_frame(pos, ext, bins, sel).astype(int)
    xtrkid          = val_in_frame(pos, trkid, bins, sel, 'min').astype(int)
    xnhits          = val_in_frame(pos, nhits, bins, sel, 'sum').astype(int)
    nsize           = len(xene)
    file_id         = evt['file_id'].unique()[0]  * np.ones(nsize, int)
    event           = evt['event'].unique()[0]    * np.ones(nsize, int)
    binclass        = evt['binclass'].unique()[0] * np.ones(nsize, int)

    dd = {'file_id' : file_id, 'event' : event,
          'x' : xpos[0], 'y' : xpos[1], 'z' : xpos[2], 'E' : xene,
          'binclass' : binclass, 'segclass' : xsegclass, 'ext' : xext,
          'track_id' : xtrkid, 'nhits' : xnhits, 'nielectron' : xnielectron}
    
    return pd.DataFrame(dd)

memory_limit = 1000e6 # b

def run(ifilename, ofilename, width0, sigma, width1, use_df = True,
        verbose = True, nevents = -1):

    t0 = time.time()
    if (verbose):
        print('input  filename ', ifilename)
        print('output filename ', ofilename)
        print('width-0    (mm) ', width0)
        print('sigmas     (mm) ', sigma)
        print('widths-1   (mm) ', width1)
        print('wi         (eV) ', wi)
        print('use DF          ', use_df)
        print('events          ', nevents)

    idata = pd.read_hdf(ifilename, "voxels") 
    odata = None

    _algo = df_diff_ievoxel if use_df else diff_ievoxel

    def _odata(init_data, odata, dvox):
        odata = dvox if init_odata == True else pd.concat((odata, dvox), ignore_index = True)
        return odata

    def _save_odata(kfile, odata):
        ofile = ofilename.split('.')[0] + '.' +str(kfile) + '.h5'
        print('saving data      ', ofile)
        odata.to_hdf(ofile, 'voxels', complevel = 9)

    ta  = time.time()
    odata = None
    init_odata, kfile = True, 0
    for i, (ievt, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if ((nevents >= 1) & (i >= nevents)): break

        if i % 100 == 0: print('processing event ', i, ', id ', ievt)

        dvox  = _algo(evt, width0, sigma, width1)
        odata = _odata(init_odata, odata, dvox)
        init_odata = False

        mem = np.sum(odata.memory_usage())
#        print('memory... {:.2e} b'.format(mem))
        if (mem >= memory_limit): 
            _save_odata(kfile, odata)
            kfile += 1; init_odata = True

    _save_odata(kfile, odata)

    t1 = time.time()
    print('event processed   {:d} '.format(i))
    print('time per event    {:4.2f} s'.format((t1-ta)/i))
    print('time execution    {:8.1f}  s'.format(t1-t0))
    print('done!')

    return odata

    
#-----------
#  Plot
#-----------

def plot_event(df, scatter = False, seg = -1, ext = -1,  bins = None):
    marker = '.'; mcolor = 'black'
    cmap   = 'cool' # spring, autumn, binary
    x, y, z, ene = [df[label].values for label in ('x', 'y', 'z', 'E')]
    width = 2.
    if bins == None:
        bins = [np.arange(np.min(xi) - 2 * width, np.max(xi) + 2 * width, width) for xi in (x, y, z)]
    def _plot(x, y, bins):
        plt.hist2d(x, y, weights = 1e3*ene, bins = (bins[0], bins[1]), cmap = cmap)
        if (scatter):
            plt.scatter(x, y, alpha = 0.2, c= mcolor, marker = marker)
        if (seg > 0):
            sel = df.segclass >= seg
            if (np.sum(sel) > 0):
                plt.scatter(x[sel], y[sel], alpha = 0.2, c= mcolor, marker = '+')
        if (ext > 0):
            sel = df.ext  >= ext
            if (np.sum(sel) > 0):
                plt.scatter(x[sel], y[sel], alpha = 1., c= mcolor, marker = '*')
    plt.subplot(2, 2, 1)
    _plot(x, y, (bins[0], bins[1]))
    plt.xlabel('x'); plt.ylabel('y')
    plt.subplot(2, 2, 2)
    _plot(x, z, (bins[0], bins[2]))
    plt.xlabel('x'); plt.ylabel('z')
    plt.subplot(2, 2, 3)
    _plot(z, y, (bins[2], bins[1]))
    plt.xlabel('z'); plt.ylabel('y')
    plt.tight_layout()
    return


#-----------
#  Test
#------------


def test_df_ielectrons(evt, width, plot = False):

    dfie = df_ielectrons(evt, width)

    Etot = np.sum(evt.E)
    #print(Etot, np.sum(dfie.E))
    assert np.isclose(Etot, 1e-6*np.sum(dfie.E))

    nie  = len(dfie)
    ntot = np.sum(np.maximum(np.round(1e6*evt.E/wi), 1))
    #print(ntot, nie, np.sqrt(nie))
    assert np.isclose(ntot, nie, atol = np.sqrt(nie))

    labels = ('x', 'y', 'z')
    xs    = [evt [label].values for label in labels]
    iexs  = [dfie[label].values for label in labels]
    bins  = [mmbins(xi, wi) for xi, wi in zip(xs, width)]

    k = 1
    for i, j in ((0, 1), (0, 2), (1, 2)):
        c0, _, _ = np.histogram2d(xs[i]  , xs[j]  , weights = 1e6*evt.E , bins = (bins[i], bins[j])) 
        c1, _, _ = np.histogram2d(iexs[i], iexs[j], weights = dfie.E, bins = (bins[i], bins[j])) 
        if plot:
            plt.subplot(3, 2, k); plt.imshow(c0); k += 1
            plt.subplot(3, 2, k); plt.imshow(c1); k += 1 
        ok = np.isclose(c0.flatten(), c1.flatten(), atol = wi)
        #print(np.sum(ok), len(ok))
        assert (np.sum(ok) == len(ok))

    return True

def test_df_norma(dfie, sigma = (5, 4, 2), plot = False):

    dfie0 = dfie.copy()
    dfie  = df_norma(dfie, sigma)

    mtol  = 40
    labels = ('x', 'y', 'z') 
    diffs  = [dfie0[label].values - dfie[label].values for label in labels]
    for i, diff in enumerate(diffs):
        m, rms, n = np.mean(diff), np.std(diff), len(diff)
        if (plot):
            plt.subplot(2, 2, i + 1)
            plt.hist(diff, 100, label = ' mean {:4.2f} \n std {:4.2f}'.format(m, rms))
            plt.legend();

        atol = mtol * 1./np.sqrt(n)
        #print(m, 0, atol, m/atol)
        assert np.isclose(m, 0., atol = atol) 
        #print(rms, sigma[i], rms * atol, (rms-sigma[i])/(rms*atol))
        assert np.isclose(rms, sigma[i], atol = rms * atol)

    return True


def test_df_voxalize(df, width):

    dv = df_voxalize(df,width)

    assert len(df) >= len(dv)

    assert np.isclose(np.sum(df.E), np.sum(dv.E), atol = wi)

    ilabels = ('ix', 'iy', 'iz')
    for ilabel in ilabels:
        index = np.unique(df[ilabel].values)
        for ii in index:
            x0, x1 = df[df[ilabel] == ii][ilabel[1:]].min(), df[df[ilabel] == ii][ilabel[1:]].max()
            v0, v1 = dv[dv[ilabel] == ii][ilabel[1:]].min(), dv[dv[ilabel] == ii][ilabel[1:]].max()
            assert (v0 >= x0) & (v0 <= x1)
            assert (v1 >= x0) & (v1 <= x1)
            Ex, Ev = df[df[ilabel] == ii]['E'].sum(), dv[dv[ilabel] == ii]['E'].sum()
            assert np.isclose(Ex, Ev, wi)

    return True


def test(ifilename):

    voxels = pd.read_hdf(ifilename, 'voxels')

    width0 = (2, 2, 2)
    for i in range(3):
        evt = df_event(voxels, 1, i)
        test_df_ielectrons(evt, width0)
        sigma = (1, 2*i, 3*i)
        dfie = df_ielectrons(evt, width0)
        test_df_norma(dfie, sigma)
        dfie = df_norma(dfie, sigma)
        width = (1, 2*i, 3*i)
        test_df_voxalize(dfie, width)

    return True











                


