
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
opath = path+'xymm/'


def production(pressure, coors, widths, frame, labels, nevents = -1):
    ofiles = []
    for sample in ('0nubb', '1eroi'):
        ifile = path + dp.voxel_filename(pressure, sample)
        ofile = opath+sample+'_'+str(pressure)
        dp.run(ifile, ofile, coors, widths = widths, frame = frame, labels = labels, nevents = nevents)
        ofiles.append(dp._ofile(ofile, coors, widths, frame)+'.npz')
    ofile = dp._ofile(opath+'xymm_'+str(pressure), coors, widths, frame)
    dp.mix_godata(*ofiles, ofile)
    for ofile in ofiles: os.remove(ofile)
    return True

#--- parser

coors  = ('x', 'y')
widths = (10, 10)
frame  = 100
labels = ['esum', 'ecount', 'emax', 'emean', 'estd', 'zmean', 'zstd'] 


parser = argparse.ArgumentParser(description='img data preparation: from voxels to imgs')

parser.add_argument('-pressure', type = str, help ="pressure, i.e '13bar'")

parser.add_argument('-coors', metavar = 'N', type = str, nargs = '+',
                     help = "projections, i.e ('x', 'y')", default = coors)

parser.add_argument('-widths', metavar = 'N', type = int, nargs = '+',
                     help = "bin widths, i.e (10, 10)", default = widths)

parser.add_argument('-frame', type = int, help ="frame size (int)")

parser.add_argument('-labels', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'esum', 'emax'", default= labels)

parser.add_argument('-pressure', type = str, help ="pressure, i.e '13bar'")

parser.add_argument('-events', type = int, help="number of events, (all -1)", default = 5)
                    
args = parser.parse_args()

#--- Run

print('path', path)
print('args', args)

production(pressure = args.pressure, 
           coors    = args.coors,
           widths   = args.widths,
           frame    = args.frame,
           labels   = args.labels,
           nevents = args.events)
