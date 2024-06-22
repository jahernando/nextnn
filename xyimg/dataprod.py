
import xyimg.dataprep as dp
import xyimg.utils    as ut
import os
import argparse

path  = os.environ["LPRDATADIR"]
opath = path+'shots/'


frames    = {'5bar' : 160, '13bar': 80, '20bar' : 40}

pressure  = '5bar'
sample    = '0nubb'
sigma     = 0 # mm
width     = 2 # mm
xlabel    = ('xy_E_sum', 'xy_z_mean', 'yz_E_sum', 'yz_x_mean', 'zx_E_sum', 'zx_y_mean')
zlabel    = ('xy_segclass_mean', 'xy_ext_max', 'yz_segclass_mean', 'yz_ext_max', 'zx_segclass_mean', 'zx_ext_max')
nevents   = 100

def _ofile(pressure, sample, sigma, width):
    ofile = 'shots/' + ut.str_concatenate((pressure, sample, 's'+str(int(sigma))+'mm', 'w'+str(int(width))+'mm'))+'.npz'
    print(ofile)
    return ofile

def production(pressure, sample, width, sigma, nevents = -1):
    ifile = path + dp.filename_voxel(pressure, sample)
    ofile = path + _ofile(pressure, sample, sigma, width)
    frame = frames[pressure]
    sigma = (sigma, sigma, sigma)
    width = (width, width, width)
    dp.run(ifilename = ifile, 
           ofilename = ofile,
           sigma     = sigma,
           width     = width,
           frame     = frame,
           xlabel    = xlabel,
           zlabel    = zlabel,
           nevents   = nevents)
    return True


#--- parser


parser = argparse.ArgumentParser(description='from voxels to images vis detsim')

parser.add_argument('-pressure', type = str, help = "pressure, i.e '13bar'", default = pressure)

parser.add_argument('-sample', type = str, help = "sample, ile, '1eroi', '0nubb' ", default = sample)

parser.add_argument('-sigma', type = float, help = "sigma - normal diffusion ", default = sigma)

parser.add_argument('-width', type = float, help = "width - image pixel size ", default = width)

parser.add_argument('-events', type = int, help="number of events, (all -1)", default = nevents)
                    
args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)
frame = dp.frames[args.pressure]

production(pressure   = args.pressure, 
           sample     = args.sample,
           sigma      = args.sigma,
           width      = args.width,
           nevents    = args.events)
