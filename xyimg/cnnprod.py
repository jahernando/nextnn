import xyimg.cnn as cnn
import os
import argparse

path  = os.environ["LPRDATADIR"]
ipath = path+'xyimg/'
opath = path+'cnn/'

def production(pressure, type, sbins, labels, name = ''):

    ifile   = ipath + cnn.xyimg_filename(type, pressure, sbins)
    ofile   = opath + cnn.cnn_filename(type, pressure, sbins, name)
    Dset    = cnn.GoDataset3DImg if type == 'z' else cnn.GoDataset  
    dataset = Dset(ifile, labels)
    box    = cnn.run(dataset, ofilename = ofile)
    print('efficiency {:2.1f}% at 80% rejection'.format(100.*cnn.roc_value(box.y, box.yp, 0.8)[1]))
    return box


type     = 'levels'
pressure = '13bar' 
sbins    = '8x8'
labels   = ['esum', 'ecount', 'emax'] 

parser = argparse.ArgumentParser(description='cnn')
parser.add_argument('-pressure', type = str, help ="pressure, i.e '13bar'", default = pressure)
parser.add_argument('-type'    , type = str, help ="type, select one: 'levels', 'projections', 'z'", default = type)
parser.add_argument('-sbins'   , type = str,  help = "bins string, i.e '8x8'", default= sbins)
parser.add_argument('-labels'  , metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'esum', 'emax'", default= labels)
parser.add_argument('-name'   , type = str,  help = "name of the cnn", default= '')

args = parser.parse_args()

print('path', path)
print('args', args)
box = production(pressure = args.pressure, 
                 type      = args.type, 
                 sbins     = args.sbins, 
                 labels    = args.labels, 
                 name      = args.name);
print('Done!')
