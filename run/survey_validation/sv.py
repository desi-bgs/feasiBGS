#!/bin/python 
'''
redshift success rate calculations necessary for SV preparation 
'''
import os 
import h5py 
import pickle 
import numpy as np 
# --- desi --- 
import astropy.units as u
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import skymodel as Sky 
from feasibgs import catalogs as Cat
from feasibgs import forwardmodel as FM 
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


def G15_rmag_rfibermag(): 
    ''' plots examining r-mag selection versus r fiber mag selection
    '''
    # read in GAMA-Legacy catalog with galaxies in both GAMA and Legacy surveys
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7, silent=True)  
    
    # extract meta-data of galaxies 
    redshift    = gleg['gama-spec']['z']
    
    g_mag   = UT.flux2mag(gleg['legacy-photo']['flux_g'], method='log')
    r_mag   = UT.flux2mag(gleg['legacy-photo']['flux_r'], method='log')
    z_mag   = UT.flux2mag(gleg['legacy-photo']['flux_z'], method='log')
    w1_mag  = UT.flux2mag(gleg['legacy-photo']['flux_w1'], method='log')
    
    # probe of emission line galaxies based on WISE and opitcal colors 
    _emline = (z_mag - w1_mag) - 3./2.5 * (g_mag - r_mag) + 1.2 

    r_fiber_mag = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1], method='log') # aperture flux
    r_mag_gama  = gleg['gama-photo']['r_petro'] # r-band magnitude from GAMA (SDSS) photometry

    fig = plt.figure(figsize=(10, 4))
    sub = fig.add_subplot(121) 
    sub.scatter(r_mag_gama, r_fiber_mag, s=1, c='k', 
            label=r'$r < 19.8~(%.f/{\rm deg}^2)$' % (float(len(r_mag_gama))/60.))
    for i, _rmag in enumerate([19.5]): 
        rmag_lim = (r_mag_gama < _rmag) 
        sub.scatter(r_mag_gama[rmag_lim], r_fiber_mag[rmag_lim], s=1, c='C%i' % i, 
                label=r'$r < %.1f~(%.f/{\rm deg}^2)$' % (_rmag, float(np.sum(rmag_lim))/60.)) 
    sub.plot([16, 23], [16, 23], c='k', ls='--') 
    sub.set_xlabel('$r$ magnitude', fontsize=25) 
    sub.set_xlim(16., 23.) 
    sub.set_ylabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_ylim(16., 23.) 
    sub.legend(loc='lower right', handletextpad=0., markerscale=5, fontsize=15) 

    sub = fig.add_subplot(122) 
    sub.scatter(r_fiber_mag, _emline, c='k', s=1)
    for i, _rmag in enumerate([19.5]): 
        rmag_lim = (r_mag_gama < _rmag) 
        sub.scatter(r_fiber_mag[rmag_lim], _emline[rmag_lim], s=1, c='C%i' % i)
    sub.vlines(21, -1., 2., color='k', linestyle='--', linewidth=1)
    sub.set_xlabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 23.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=20) 
    sub.set_ylim(-1., 2.) 

    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(dir_dat, 'G15_rmag_rfibermag.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    
    fig = plt.figure(figsize=(10, 4))
    sub = fig.add_subplot(121) 
    sub.scatter(r_mag_gama, r_fiber_mag, s=1, c='k') 
    for i, _rmag in enumerate([22, 21]): 
        rmag_lim = (r_fiber_mag < _rmag) 
        sub.scatter(r_mag_gama[rmag_lim], r_fiber_mag[rmag_lim], s=1, c='C%i' % i, 
                label=r'$r_{\rm fiber} < %.1f~(%.f/{\rm deg}^2)$' % (_rmag, float(np.sum(rmag_lim))/60.)) 
    sub.plot([16, 23], [16, 23], c='k', ls='--') 
    sub.set_xlabel('$r$ magnitude', fontsize=25) 
    sub.set_xlim(16., 23.) 
    sub.set_ylabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_ylim(16., 23.) 
    sub.legend(loc='lower right', handletextpad=0., markerscale=5, fontsize=15) 

    sub = fig.add_subplot(122) 
    sub.scatter(r_fiber_mag, _emline, c='k', s=1)
    for i, _rmag in enumerate([22, 21]): 
        rmag_lim = (r_fiber_mag < _rmag) 
        sub.scatter(r_fiber_mag[rmag_lim], _emline[rmag_lim], s=1, c='C%i' % i)
    sub.vlines(21, -1., 2., color='k', linestyle='--', linewidth=1)
    sub.set_xlabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 23.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=20) 
    sub.set_ylim(-1., 2.) 

    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(dir_dat, 'G15_rfibermag_rmag.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


if __name__=='__main__': 
    G15_rmag_rfibermag()
