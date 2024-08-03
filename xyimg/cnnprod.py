import xyimg.cnn      as cnn
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
opath = path+'CNN/'

pressure  = '5bar'
width     =  10
frame     = 150


root = path + 'cvoxels/'+pressure+'_bunch*.h5'

cnnname   = 'HCNN'
label     = ['xy_E_sum', 'yz_E_sum', 'zx_E_sum']
expansion = 8 
nepochs   = 20
eloss     = 'CrossEntropyLoss'
lrate     = 0.01

parser = argparse.ArgumentParser(description='cnn')

parser.add_argument('-cnnname', type = str, help = 'name of the cnn', default = cnnname)

parser.add_argument('-label', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'xy_E_sum' ", default= label)

parser.add_argument('-width', type = float, help = 'image pixel width ', default = width)

parser.add_argument('-frame', type = float, help = 'image scale factor ', default = frame)

parser.add_argument('-expansion', type = int, help = "sigma - normal diffusion ", default = expansion)

parser.add_argument('-nepochs', type = int, help = "width - image pixel size ", default = nepochs)

parser.add_argument('-eloss', type = str, help = 'energy loss function', default = eloss)

parser.add_argument('-lrate', type = float, help = 'learning rate', default = lrate)

args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)

config                   = cnn.config
config['label']          = args.label
config['width']          = args.width
config['frame']          = args.frame

config['cnnname']        = args.cnnname
config['expansion']      = args.expansion
config['loss_function']  = args.eloss
config['learning_rate']  = args.lrate
config['nepochs']        = args.nepochs

ofile = pressure + '_img_w'+str(int(width))+'f'+str(int(frame))+'.npz'

_ = cnn.production(root, opath + ofile, config)
print('Done!')
