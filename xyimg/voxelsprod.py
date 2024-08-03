
import xyimg.voxelsprep as vp
import os
import argparse

path  = os.environ["LPRDATADIR"]
opath = path+'cvoxels/'


pressure  = '5bar'
nbunch    =  100
nevents   = 1000

#--- parser

parser = argparse.ArgumentParser(description='prepare voxels events ')

parser.add_argument('-pressure', type = str, help = "pressure, i.e '13bar'", default = pressure)

parser.add_argument('-nbunch', type = int, help="number of events in bunch", default = nbunch)

parser.add_argument('-nevents', type = int, help="number of events, (all -1)", default = nevents)
                    
args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)

bkgfile = path + vp.filename_voxel(pressure, '1eroi')
sigfile = path + vp.filename_voxel(pressure, '0nubb') 

ofile       = opath+pressure+'.h5'

vp.run([bkgfile, sigfile], ofile, nbunch = args.nbunch, nevents = args.nevents, shuffle = True)
