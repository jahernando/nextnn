#  DetSim
#  Module to smeare the MC hits along the tracks and re-voxel
#  Krisham Mistry, JA Hernando 18/06/24

import numpy             as np
import pandas            as pd
from   scipy       import stats

import matplotlib.pyplot as plt
import os

import xyimg.dataprep          as dp


wi         = 25.6 # eV Scintillation threshold
voxel_size =  2.0 # mm (voxel size)  
DL         = 0.278 # mm / sqrt(cm)
DT         = 0.272 # mm / sqrt(cm)
sigma_L    = 5. # mm
sigma_T    = 4. # mm


# def ielectrons(position, energy, widths, sigmas, wi = wi):

#     def _ie(pos, ene):
#         nsize = int(np.round(1e6*ene/wi)) # E is in MeV and wi in eV
#         def _smear(xi, width, sigma): 
#             x  = xi * np.ones(nsize)
#             x += width * np.random.uniform(-0.5, 0.5, size = nsize)
#             x += sigma * np.random.normal(size = nsize)
#             return x
#         spos = [_smear(x, width, sigma) for x, width, sigma in zip(pos, widths, sigmas)]
#         sene = ene * np.ones(nsize) / nsize
#         return spos, sene

#     vals   = [_ie((x, y, z), ene) for x, y, z, ene in zip(*position, energy)]
#     ie_pos = [np.concatenate([v[0][i] for v in vals]) for i in range(3)]
#     ie_ene =  np.concatenate([v[1] for v in vals])

#     return ie_pos, ie_ene

def create_df_ielectrons(dfhit, width, wi = wi):
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

        
# def voxelize(position, ene, widths):
#     bins        = [np.arange(np.min(x) - 2.*width, np.max(x) + 2.*width, width) for x, width in zip(position, widths)]
#     img , _, _  =  stats.binned_statistic_dd(position, ene,  bins = bins, statistic = 'sum')
#     ximg        = [stats.binned_statistic_dd(position,   x,  bins = bins, statistic = 'mean')[0] for x in position]
           
#     sel    = img > 0
#     xene   = img[sel].flatten()
#     xs     = [x[sel].flatten() for x in ximg]
#     return xs, xene, bins, sel
    
def _segclass(seg):
    _seg  = np.array([2, 1, 3])
    return [_seg[x] for x in seg]

# def val_in_frame(coors, val, bins, sel, statistic = 'max'):
#     img , _, _  =  stats.binned_statistic_dd(coors, val,  bins = bins, statistic = statistic)
#     xval = img[sel].flatten()
#     np.nan_to_num(xval, 0)
#     return xval


# def event_smear(evt, width0, sigma, width1, wi = wi):

#     pos      = [evt[label].values for label in ('x', 'y', 'z')]
#     ene      =  evt['E'].values
#     segclass = _segclass(evt['segclass'].values)
#     ext      = evt['ext'].values
#     trkid    = 1 + evt['track_id'].values
#     nhits    = evt['nhits'].values

#     ie_pos, ie_ene        = ielectrons(pos, ene, width0, sigma, wi)
#     xpos, xene, bins, sel = voxelize(ie_pos, ie_ene, width1)
#     xsegclass             = val_in_frame(pos, segclass, bins, sel).astype(int)
#     xext                  = val_in_frame(pos, ext, bins, sel).astype(int)
#     xtrkid                = val_in_frame(pos, trkid, bins, sel, 'min').astype(int)
#     xnhits                = val_in_frame(pos, nhits, bins, sel, 'sum').astype(int)

#     dd = {'x' : xpos[0], 'y' : xpos[1], 'z' : xpos[2], 'E' : xene,
#          'segclass' : xsegclass, 'ext' : xext, 'track_id' : xtrkid, 'nhits' : xnhits}

#     nsize = len(xene)
#     for label in ('file_id', 'event', 'binclass'):
#         dd[label] = np.ones(nsize, int) * int(np.unique(evt[label].values))

#     return dd


def run(ifilename, ofilename, width0, sigma, width1, verbose = True, nevents = -1):

    
    if (verbose):
        print('input  filename ', ifilename)
        print('output filename ', ofilename)
        print('width-0    (mm) ', width0)
        print('sigmas     (mm) ', sigma)
        print('widths-1   (mm) ', width1)
        print('wi         (eV) ', wi)
        print('events          ', nevents)

    idata = pd.read_hdf(ifilename, "voxels") 
    odata = []

    for i, (evtid, evt) in enumerate(idata.groupby(['file_id', 'event'])):
        if ((nevents >= 1) & (i >= nevents)): break

        dfie = create_df_ielectrons(evt, width0)
        dfie = df_norma(dfie, sigma) 
        dvox = df_voxalize(dfie, width1)
        print(dvox.columns)
        odata.append(dvox)
    
    df = pd.concat(odata, ignore_index = True)
    df.to_hdf(ofilename, 'voxels')

    return df
    
#-----------
#  Plot
#-----------

def plot_event(df, scatter = True, bins = None):
    marker = '.'; mcolor = 'black'
    cmap   = 'cool' # spring, autumn, binary
    x, y, z, ene = [df[label].values for label in ('x', 'y', 'z', 'E')]
    plt.subplot(2, 2, 1)
    width = 2.
    if bins == None:
        bins = [np.arange(np.min(xi), np.max(xi) + width, width) for xi in (x, y, z)]
    plt.hist2d(x, y, weights = 1e3*ene, bins = (bins[0], bins[1]), cmap = cmap)
    if (scatter):
        plt.scatter(x, y, alpha = 0.2, c= mcolor, marker = marker)
    plt.xlabel('x'); plt.ylabel('y')
    plt.subplot(2, 2, 2)
    plt.hist2d(x, z, weights = 1e3*ene, bins = (bins[0], bins[2]), cmap = cmap)
    if (scatter):
        plt.scatter(x, z, alpha = 0.2, c = mcolor, marker = marker)
    plt.xlabel('x'); plt.ylabel('z')
    plt.subplot(2, 2, 3)
    plt.hist2d(z, y, weights = 1e3*ene, bins = (bins[2], bins[1]), cmap = cmap)
    if (scatter):
        plt.scatter(z, y, alpha = 0.1, c = mcolor, marker = marker)
    plt.xlabel('z'); plt.ylabel('y')
    plt.tight_layout()
    return



#-----------
#  Test
#------------


def test_create_df_ielectrons(evt, width, plot = False):

    dfie = create_df_ielectrons(evt, width)

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
        test_create_df_ielectrons(evt, width0)
        sigma = (1, 2*i, 3*i)
        dfie = create_df_ielectrons(evt, width0)
        test_df_norma(dfie, sigma)
        dfie = df_norma(dfie, sigma)
        width = (1, 2*i, 3*i)
        test_df_voxalize(dfie, width)

    return True











                


