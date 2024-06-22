#----------------
#  Utils : string, numpy and panda helping functions
#----------------

import numpy             as np
import pandas            as pd



# string
#-------------

def str_concatenate(words, link = '_'):
    ss = ''
    for w in words: ss += str(w) + link
    nn = len(link)
    if nn == 0: return ss 
    return ss[:-nn]


def prepend_filename(ifilename, name , link = '_'):
    words = ifilename.split('/')
    fname = words[-1]
    ofname = str_concatenate((name, fname), link)
    ofname = str_concatenate(words[:-1], '/') + '/'+ofname
    return ofname

# numpy
#------------

def urange(var : np.array) -> np.array:
    """ set the variable in the range [0, 1]
    input:
     - var: np.array(float)
    """
    vmin, vmax = np.min(var), np.max(var)
    if vmax <= vmin: return np.zeros(len(var))
    return (var-vmin)/(vmax-vmin)

def ucenter(var, width):
    """
    return the center of the variable inside a frame with a given width
    Inputs:
        - var   : np.array(ndim = 1)
        - width : float
    """
    vmin, vmax = np.min(var), np.max(var)
    assert (vmax > vmin)
    v0   = (width - vmax + vmin)/2.
    assert (v0 >= 0.) # values should be contained in the width
    return vmin + v0   


#-------------
#    Tests
#-------------

def test_str_concatenate():
    words = ('x', 'y')
    ss    = str_concatenate(words, '')
    assert len(ss) == sum([len(x) for x in words])
    words = np.arange(10)
    ss    = str_concatenate(words, '_')
    assert len(ss.split('_')) == len(words)
    words = ('a', 'b', 'c')
    ss    = str_concatenate(words, 'x')
    assert len(ss.split('x')) == len(words)
    return True

def test_urange(x):
    uz = urange(x)
    assert (np.min(uz) >= 0) & (np.max(uz) <= 1)
    iar = np.argmax(x)
    assert uz[iar] == 1
    iar = np.argmin(x)
    assert uz[iar] == 0
    return True

def test_center(x, delta):
    x0 = ucenter(x, delta)
    assert np.mean(np.mean(x), 0.5*delta)
    return True

