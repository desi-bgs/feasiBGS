'''

SV target selection 


'''
import os 
import h5py 
import numpy as np 
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False

dir_dat = os.path.join(UT.dat_dir(), 'survey_validation')


def _1400deg2_test(): 
    '''
    '''
    # read in targets in the 1400deg2 test region 
    ftarg = h5py.File(os.path.join(dir_dat, 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    targets = {} 
    for k in ftarg.keys(): 
        targets[k] = ftarg[k][...] 

    # read in random files 
    random = np.load(os.path.join(dir_dat, 'bgs.1400deg2.random.npy'))

    # read in tycho2 stars to test masking of randoms 
    leg = Cat.Legacy() 
    tycho = leg._Tycho(ra_lim=[160., 230.], dec_lim=[-2., 18.]) 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.scatter(random['RA'], random['DEC'], c='C0', s=0.5) 
    sub.scatter(targets['ra'], targets['dec'], c='k', s=1) 
    sub.scatter(tycho['RA'], tycho['DEC'], c='r', s=10) 
    sub.set_xlim(160., 161.)
    sub.set_ylim(-2, -1.)
    fig.savefig(os.path.join(dir_dat, '_1400deg2_test.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    _1400deg2_test()
