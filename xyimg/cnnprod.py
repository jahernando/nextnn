import xyimg.cnn     as cnn
import xyimg.dataprep as dp
import os
import argparse

path  = os.environ["LPRDATADIR"]
ipath = path+'xymm/'
opath = path+'cnn/'

def production(ipath, opath, pressure, projection, widths, frame, labels, nepochs = 20, name = 'cnn', rejection = 0.95):

    xname   = dp.str_concatenate((name, dp.str_concatenate(labels, '+'), '_'))
    ifile   = dp.xymm_filename(projection, widths, frame, 'xymm_'+pressure)
    ofile   = dp.prepend_filename(ifile, xname)
    print('input  filename ', ifile)
    print('output filename ', ofile)
    Dset    = cnn.GoDataset3DImg if len(projection) == 3 else cnn.GoDataset  
    dataset = Dset(ipath + ifile, labels)
    box     = cnn.run(dataset, ofilename = opath + ofile, nepochs = nepochs)
    print('efficiency {:2.1f}% at {:2.1f}% rejection'.format(100.*cnn.roc_value(box.y, box.yp, rejection)[1], 100 *rejection))
    return box, ifile, ofile


pressure = '13bar'
projection = ('x', 'y')
widths   = (10, 10)
labels   = ['esum',] 
nepochs  = 20

parser = argparse.ArgumentParser(description='cnn')

parser.add_argument('-pressure', type = str, help ="pressure, i.e '13bar'", default = pressure)

parser.add_argument('-projection', metavar = 'N', type = str, nargs = '+',
                     help = "projections, i.e ('x', 'y')", default = projection)

parser.add_argument('-widths', metavar = 'N', type = int, nargs = '+',
                     help = "bin widths, i.e (10, 10)", default = widths)

#parser.add_argument('-frame', type = int, help ="frame size (int)", default=frame)

parser.add_argument('-labels', metavar = 'N', type = str, nargs='+',
                    help = "list of images, i.e 'esum', 'emax'", default= labels)

parser.add_argument('-nepochs', type = int, help ="number of epochs (int)", default=nepochs)

#parser.add_argument('-name'   , type = str,  help = "name of the cnn", default= '')

args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)
frame  = dp.frames[args.pressure]

_ = production(ipath, opath, 
               pressure   = args.pressure, 
               projection = args.projection,
               widths     = args.widths,
               frame      = frame,
               labels     = args.labels,
               nepochs    = args.nepochs)
print('Done!')
