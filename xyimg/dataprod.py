
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
opath = path+'xyimg/'

def production(pressure, type, bins, labels, nevents = -1):
    ofiles = []
    sbin = str(bins[0])
    for i in bins[1:]: sbin+='x'+str(i)
    for sample in ('0nubb', '1eroi'):
        print('processing : ', type, ', ', pressure, ', ', sample, ', ', str(bins), ', ', labels)
        ifile = path + dp.voxel_filename(pressure, sample)
        ofile = opath+'xyimg_'+type+'_'+str(pressure)+'_'+sbin+'_'+sample
        ofiles.append(ofile+'.npz')
        dp.run(ifile, ofile, xyimg_type = type, bins = bins, labels = labels, nevents = nevents)
    ofile = opath+'xyimg_'+type+'_'+str(pressure)+'_'+sbin
    dp.mix_godata(*ofiles, ofile)
    for ofile in ofiles: os.remove(ofile)
    return True

bins   = (8, 8)
labels = ['esum', 'ecount', 'emax', 'emean', 'estd', 'zmean', 'zstd'] 

parser = argparse.ArgumentParser(description='xyimg data preparation: from voxels to xyimgs')
parser.add_argument('-pressure', type = str, help ="pressure, i.e '13bar'")
parser.add_argument('-type', type = str, help ="xyimg type, select one: 'levels', 'projections', 'z'")
parser.add_argument('-bins', metavar = 'N', type = int, nargs = '+',
                     help = "xyimg bins, i.e (8, 8)", default= bins)
parser.add_argument('-labels', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'esum', 'emax'", default= labels)
parser.add_argument('-events', type = int, help="number of events, (all -1)", default = 5)
                    
args = parser.parse_args()

print('path', path)
print('args', args)
production(pressure = args.pressure, type = args.type, bins = args.bins, labels = args.labels, nevents = args.events)
