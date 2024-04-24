
import xyimg.dataprep as dp
import os

path  = os.environ["LPRDATADIR"]
opath = path+'prod/'

def production(pressures, type, bins, labels, nevents = -1):
    for pressure in pressures:
        ofiles = []
        for sample in ('0nubb', '1eroi'):
            print('processing : ', type, ', ', pressure, ', ', sample, ', ', str(bins), ', ', labels)
            ifile = path + dp.voxel_filename(pressure, sample)
            ofile = opath+'xyimg_'+type+'_'+str(pressure)+'_'+str(bins[0])+'_'+sample
            ofiles.append(ofile+'.npz')
            dp.run(ifile, ofile, xyimg_type = type, bins = bins, labels = labels, nevents = nevents)
        ofile = opath+'xyimg_'+type+'_'+str(pressure)+'_'+str(bins[0])
        dp.mix_godata(*ofiles, ofile)
    return

pressures = ['13bar', '5bar']
type      = 'levels'
bins      = (8, 8)
labels    = ['esum', 'ecount', 'emax', 'emean', 'estd', 'zmean', 'zstd']
nevents   = 5

production(pressures, type, bins, labels, nevents)


