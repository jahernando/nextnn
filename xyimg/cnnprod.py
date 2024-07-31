import xyimg.cnn      as cnn
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
ipath = path+'shots/'
opath = path+'CNN/'

pressure  = '5bar'
hit_width =  2
sigma     =  2
width     = 10

def _hname():
    ss = pressure+'_h' +str(hit_width)+'s'+str(sigma)+'w'+str(width)
    return ss

cnnname   = 'HCNN'
labels    = ['xy_E_sum', 'yz_E_sum', 'zx_E_sum']
expansion = 8 
nepochs   = 20
eloss     = 'CrossEntropyLoss'
lrate     = 0.001

parser = argparse.ArgumentParser(description='cnn')

parser.add_argument('-cnnname', type = str, help = 'name of the cnn', default = cnnname)

parser.add_argument('-labels', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'xy_E_sum' ", default= labels)

parser.add_argument('-black', type = bool, help = 'digitial image ', default = False)

parser.add_argument('-img_scale', type = float, help = 'image scale factor ', default = 1.)

parser.add_argument('-expansion', type = int, help = "sigma - normal diffusion ", default = expansion)

parser.add_argument('-nepochs', type = int, help = "width - image pixel size ", default = nepochs)

parser.add_argument('-eloss', type = str, help = 'energy loss function', default = eloss)

parser.add_argument('-lrate', type = float, help = 'learning rate', default = lrate)

args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)

config                   = cnn.config
config['labels']         = args.labels
config['black']          = args.black
config['img_scale']      = args.img_scale

config['cnnname']        = args.cnnname
config['expansion']      = args.expansion
config['loss_function']  = args.eloss
config['learning_rate']  = args.lrate
config['nepochs']        = args.nepochs

ifile  = dp.filename_godata(pressure, 'shuffle', hit_width, sigma, width)
ofile  = cnn.filename_cnn(_hname(), config)

_ = cnn.production(ipath + ifile, opath + ofile, config)
print('Done!')
