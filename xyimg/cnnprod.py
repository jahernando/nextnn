import xyimg.cnn      as cnn
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
ipath = path+'shots/'
opath = path+'kcnn/'

pressure  = '5bar'
hit_width =  0
sigma     =  0
width     = 10

cnnname   = 'k3cnn'
labels    = ['xy_segclass_max',]
expansion = 4 
nepochs   = 20
eloss     = 'MSELoss'
lrate     = 0.001

parser = argparse.ArgumentParser(description='cnn')

parser.add_argument('-cnnname', type = str, help = 'name of the cnn', default = cnnname)

parser.add_argument('-labels', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'xy_E_sum' ", default= labels)

parser.add_argument('-eloss', type = str, help = 'energy loss function', default = eloss)

parser.add_argument('-lrate', type = float, help = 'learning rate', default = lrate)

parser.add_argument('-expansion', type = int, help = "sigma - normal diffusion ", default = expansion)

parser.add_argument('-nepochs', type = int, help = "width - image pixel size ", default = nepochs)

args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)

config                   = cnn.config
config['labels']         = args.labels
config['expansion']      = args.expansion
config['nepochs']        = args.nepochs
config['learning_rate']  = args.lrate
config['loss_function']  = args.eloss

ifile  = dp.filename_godata(pressure, 'shuffle', hit_width, sigma, width)
ofile  = cnn.filename_cnn('5bar_h0s0w10', args.cnnname, config)

_ = cnn.production(ipath + ifile, opath + ofile, config)
print('Done!')
