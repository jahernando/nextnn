
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
opath = path+'xymm/'


def production(pressure, projection, widths, frame, labels, nevents = -1):
    ofiles = []
    for sample in ('0nubb', '1eroi'):
        ifile = path + dp.voxel_filename(pressure, sample)
        ofile = path + sample+'_'+str(pressure)
        #if (nevents >0): ofile += ofile + '_' + str(int(nevents)) + 'events_test'
        dp.run(ifile, ofile, projection, widths = widths, frame = frame, labels = labels, nevents = nevents)
        ofiles.append(dp.xymm_filename(projection, widths, frame, prefix = ofile))
    ofile = dp.xymm_filename(projection, widths, frame, prefix = opath+'xymm_'+str(pressure))
    dp.mix_godata(*ofiles, ofile.split('.')[0])
    for ofile in ofiles: os.remove(ofile)
    return True


#--- parser

pressure = '13bar'
coors    = ('x', 'y')
widths   = (10, 10)
labels   = ['esum', 'ecount', 'emax', 'emean', 'zmean'] 
nevents  = 10


parser = argparse.ArgumentParser(description='img data preparation: from voxels to imgs')

parser.add_argument('-pressure', type = str, help ="pressure, i.e '13bar'", default = pressure)

parser.add_argument('-projection', metavar = 'N', type = str, nargs = '+',
                     help = "projections, i.e ('x', 'y')", default = coors)

parser.add_argument('-widths', metavar = 'N', type = int, nargs = '+',
                     help = "bin widths, i.e (10, 10)", default = widths)

#parser.add_argument('-frame', type = int, help ="frame size (int)", default=frame)

parser.add_argument('-labels', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'esum', 'emax'", default= labels)

parser.add_argument('-events', type = int, help="number of events, (all -1)", default = nevents)
                    
args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)
frame = dp.frames[args.pressure]

production(pressure   = args.pressure, 
           projection = args.projection,
           widths     = args.widths,
           frame      = frame,
           labels     = args.labels,
           nevents    = args.events)
