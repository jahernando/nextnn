import xyimg.utils    as ut
import xyimg.cnn      as cnn
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
ipath = path+'imgs/'
opath = path+'cnn/'

pressure  = '20bar'
label     = 'xy_E_sum', 'yz_E_sum', 'zx_E_sum'
width     =  5

frames     = {'5bar' : 155, '13bar' : 50, '20bar' : 35}

cnnname   = 'HCNN'
label     = ['xy_E_sum', 'yz_E_sum', 'zx_E_sum']
expansion = 8 
nepochs   = 20
eloss     = 'CrossEntropyLoss'
lrate     = 0.001

parser = argparse.ArgumentParser(description='cnn')

parser.add_argument('-cnnname', type = str, help = 'name of the cnn', default = cnnname)

parser.add_argument('-label', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'xy_E_sum' ", default= label)

parser.add_argument('-width', type = float, help = 'image pixel width ', default = width)

parser.add_argument('-frame', type = float, help = 'image scale factor ', default = frame)

parser.add_argument('-expansion', type = int, help = "sigma - normal diffusion ", default = expansion)

parser.add_argument('-nepochs', type = int, help = "width - image pixel size ", default = nepochs)

parser.add_argument('-tag', type = str, help = 'tag cnn name', default = '')

parser.add_argument('-eloss', type = str, help = 'energy loss function', default = eloss)

parser.add_argument('-lrate', type = float, help = 'learning rate', default = lrate)

args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)

config                   = cnn.config
config['pressure']       = args.pressure
config['label']          = args.label
config['width']          = args.width

config['cnnname']        = args.cnnname
config['expansion']      = args.expansion
config['loss_function']  = args.eloss
config['learning_rate']  = args.lrate
config['nepochs']        = args.nepochs

# Input-Ouput

slabs  = ut.str_concatenate(label, '+')
sframe = 'w'+str(int(width))
ifile  = ut.str_concatenate((pressure, slabs, sframe), '_')

cnnname    = args.cnnname
expansion  = args.expansion
nepochs    = args.nepochs
tag        = args.tag

scnn = 'exp'+str(expansion)+'epochs'+str(nepochs)+tag

ofile  = ut.str_concatenate((ifile,cnnname, scnn, '_'))
print('ifile : ', ifile + '.npz')
print('ofile : ', ofile + '.npz')

#--- Preparation

imgdis = dp.ImgDispatchNP(ifile)
idata  = cnn.ImgDataset(imgdis)
 
CNN       = cnn.HCNN if cnnname == 'HCNN' else cnn.HKCNN if cnnname == 'HKCNN' else cnn.KCNN
padding   = 0    if cnnname == 'HCNN' else 1

shape = imgdis.x.shape
print('Image spahe ', shape)

print('CNN model ', cnnname)
kernel = 3
color  = shape[1]
print('configurate cnn (kernel, expansion, padding)', kernel, expansion, padding)
kcnn = CNN(color, width, expansion = expansion, kernel = kernel, padding = padding)

#--- Run

print('run cnn (epochs) ', nepochs)
rcnn = cnn.run(idata, kcnn, ofilename = ofile, nepochs = nepochs, config = config)

print('done!')
