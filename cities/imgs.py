import xyimg.utils    as ut
import xyimg.dataprep as dp

import numpy as np

import os
import argparse
import time as time

#--- configure

path  = os.environ["LPRDATADIR"]
ipath = path+  'cvoxels/'
opath = path + 'imgs/'

pressure  = '20bar'
label     = 'xy_E_sum', 'yz_E_sum', 'zx_E_sum'
width     =  5
frame     = 40

#--- parser

parser = argparse.ArgumentParser(description='create images')
parser.add_argument('-pressure', type = str, help = "pressure, i.e '13bar'", default = pressure)
parser.add_argument('-label', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'xy_E_sum' ", default= label)
parser.add_argument('-width', type = float, help = 'image pixel width ', default = width)
parser.add_argument('-frame', type = float, help = 'image scale factor ', default = frame)
args = parser.parse_args()

#--- Run

print('args : ', args)

root   = ipath+pressure+'_bunch*.h5'
print('root :', root)

slabs  = ut.str_concatenate(label, '+')
sframe = 'w'+str(int(width))+'f'+str(int(frame))
ofile  = opath+ut.str_concatenate((pressure, slabs, sframe), '_')+'.npz'
print('ofile :', ofile)

bins  = dp.get_bins(width, frame)

def oper(evt):
     x = dp.evt_image(evt, label, bins = bins)
     y = int(evt['binclass'].unique())
     return x, y

evtgen = dp.evt_generator(root)
t0 = time.time()
imgs = [oper(evt[1]) for evt in evtgen]

xs = [x[0] for x in imgs]
ys = [x[1] for x in imgs]
t1 = time.time()

np.savez_compressed(ofile, x = xs, y = ys)

print(f"time per img : {(t1 - t0)/len(ys)} s")
print("done!")

