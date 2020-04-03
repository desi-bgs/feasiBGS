#!/bin/python
'''

test SV tiles 

'''
import os 
import numpy as np 
# --- desi --- 
import healpy as hp 
from astropy.io import fits
# -- feasibgs -- 
from feasibgs import util as UT
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


def SV_tiles(): 
    ''' 
    '''
    fsv = fits.open(os.path.join(dir_dat, 'BGS_SV_30_3x_superset60_Sep2019.fits')) 
    sv = fsv[1].data 
    ra, dec = sv['RA'], sv['DEC']
    
    # plot the SV tiles 
    fig = plt.figure(figsize=(6,3))
    sub = fig.add_subplot(111)
    sub.grid(True)

    #gs = mpl.gridspec.GridSpec(1,1, figure=fig) 
    #sub = plt.subplot(gs[0], projection='mollweide')
    #sub.grid(True) 
    #_ra = np.remainder(ra+360 - 120., 360) # shift RA values
    #_ra[_ra > 180] -=360    # scale conversion to [-180, 180]
    #_ra = -_ra    # reverse the scale: East to the left
    #sub.scatter(np.radians(_ra), np.radians(dec), s=20, lw=0, c='C1')

    sub.scatter(ra, dec, s=20, lw=0, c='C0')
    sub.set_xlim(360, 0)
    
    hp_pixs = hp.pixelfunc.ang2pix(2, np.radians(90. - dec), np.radians(360. - ra))
    print('healpix', hp_pixs) 
    print('%i unique healpix' % len(np.unique(hp_pixs)))
    print(np.unique(hp_pixs)) 

    ffig = os.path.join(dir_dat, 'test_sv_tiles.png') 
    fig.savefig(ffig, bbox_inches='tight')
    return None 


if __name__=="__main__": 
    SV_tiles()
