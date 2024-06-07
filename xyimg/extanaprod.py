
import xyimg.dataprep as dp
import xyimg.extana   as extana
import os
import argparse

path  = os.environ["LPRDATADIR"]



#--- parser

pressure = '13bar'
radius   = 16
nevents  = 10


parser = argparse.ArgumentParser(description='ntuple the track extremes information')

parser.add_argument('-pressure', type = str, help ="pressure, i.e '13bar'", default = pressure)

parser.add_argument('-radius', type = int, help="radius (default 16 mm)", default = radius)

parser.add_argument('-events', type = int, help="number of events, (all -1)", default = nevents)
                    
args = parser.parse_args()

#--- Run

print('path : ', path)
print('args : ', args)

extana.production(path       = path,
                  pressure   = args.pressure, 
                  radius     = args.radius,
                  nevents    = args.events)
