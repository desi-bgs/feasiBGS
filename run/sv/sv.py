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
from astropy.io import fits
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

    # r-mag selection 
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
    
    # r fiber mag selection 
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
    
    # r fiber mag + emline selection 
    selection = ~(_emline < 1.5*(r_fiber_mag - 20.75))
    fig = plt.figure(figsize=(10, 4))
    sub = fig.add_subplot(121) 
    sub.scatter(r_mag_gama, r_fiber_mag, s=1, c='k') 
    sub.scatter(r_mag_gama[selection], r_fiber_mag[selection], s=1, c='C1', 
            label=r'$(%.f/{\rm deg}^2)$' % (float(np.sum(selection))/60.)) 
    sub.vlines(19.5, 16., 23., color='k', linestyle='--', linewidth=1)
    sub.plot([16, 23], [16, 23], c='k', ls='--') 
    sub.set_xlabel('$r$ magnitude', fontsize=25) 
    sub.set_xlim(16., 23.) 
    sub.set_ylabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_ylim(16., 23.) 
    sub.legend(loc='lower right', handletextpad=0., markerscale=5, fontsize=15) 

    sub = fig.add_subplot(122) 
    sub.scatter(r_fiber_mag, _emline, c='k', s=1)
    sub.scatter(r_fiber_mag[selection], _emline[selection], s=1, c='C1')
    sub.vlines(21, -1., 2., color='k', linestyle='--', linewidth=1)
    sub.set_xlabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 23.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=20) 
    sub.set_ylim(-1., 2.) 

    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(dir_dat, 'G15_rmag_rfibermag.color_sbright.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def G15_zsuccess(): 
    ''' calculations using G15 galaxies 
    '''
    # compile all the G15 meta-data and redshift success 
    keys = ['gama-spec/z', 'legacy-photo/flux_g', 'legacy-photo/flux_r', 'legacy-photo/flux_z', 'legacy-photo/flux_w1', 'legacy-photo/apflux_r', 'gama-photo/r_petro']
    data, redrock = [], [] 
    for iexp in range(1,14): 
        # meta data 
        fexp = h5py.File(os.path.join(dir_dat, 'GALeg.g15.sourceSpec.%iof13.hdf5' % iexp), 'r') 
        datum = [] 
        for k in keys: 
            if 'apflux_r' in k: 
                datum.append(fexp[k][...][:,1]) 
            else: 
                datum.append(fexp[k][...]) 
        data.append(np.array(datum))

        # redrock outputs 
        rr = fits.open(os.path.join(dir_dat, 'GALeg.g15.bgsSpec.%iof13.default_exp.rr.fits' % iexp))[1].data
        redrock.append(np.array([rr['Z'], rr['DELTACHI2'], rr['ZWARN']]))
    data = np.concatenate(data, axis=1)
    redrock = np.concatenate(redrock, axis=1) 
    
    # unpack values 
    redshift = data[0]

    g_mag   = UT.flux2mag(data[1], method='log')
    r_mag   = UT.flux2mag(data[2], method='log')
    z_mag   = UT.flux2mag(data[3], method='log')
    w1_mag  = UT.flux2mag(data[4], method='log')
    
    r_fiber_mag = UT.flux2mag(data[5], method='log') # aperture flux
    r_mag_gama  = data[6]  # r-band magnitude from GAMA (SDSS) photometry

    # color probe of emission line galaxies based on WISE and opitcal colors 
    _emline = (z_mag - w1_mag) - 3./2.5 * (g_mag - r_mag) + 1.2 

    z_rr    = redrock[0] 
    dchi2   = redrock[1]
    zwarn   = redrock[2]
    
    # calculate redshift success
    zsuccess = UT.zsuccess(z_rr, redshift, zwarn, deltachi2=dchi2, min_deltachi2=40.) 

    #wmean, rate, err_rate = UT.zsuccess_rate(rmag, zsuccess_exp, range=[15,20], nbins=28, bin_min=10) 
    wmean, rate, err_rate = UT.zsuccess_rate(r_fiber_mag, zsuccess | (r_mag_gama < 18.), 
            range=[18,23], nbins=28, bin_min=10) 

    # r-mag selection 
    fig = plt.figure(figsize=(10, 4))
    sub = fig.add_subplot(121) 
    sub.scatter(r_mag_gama, r_fiber_mag, s=1, c='k', label='GAMA G15 $(r < 19.8)$')
    sub.scatter(r_mag_gama[(r_mag_gama > 18.0) & ~zsuccess], r_fiber_mag[(r_mag_gama > 18.0) & ~zsuccess], s=1, c='C1', 
            label='$z$ failure$^*$')
    sub.plot([16, 23], [16, 23], c='k', ls='--') 
    sub.set_xlabel('$r$ magnitude', fontsize=25) 
    sub.set_xlim(16., 23.) 
    sub.set_ylabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_ylim(16., 23.) 
    sub.legend(loc='lower right', handletextpad=0., markerscale=5, fontsize=15) 

    sub = fig.add_subplot(122) 
    sub.scatter(r_fiber_mag, _emline, c='k', s=1)
    sub.scatter(r_fiber_mag[(r_mag_gama > 18.0) & ~zsuccess], _emline[(r_mag_gama > 18.0) & ~zsuccess], c='C1', s=1)
    sub.set_xlabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 23.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=20) 
    sub.set_ylim(-1., 2.) 
    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(dir_dat, 'G15.zsuccess.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    
    fig = plt.figure(figsize=(10, 4))
    sub = fig.add_subplot(121) 
    sub.plot([15., 23.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(wmean, rate, err_rate, fmt='.k', elinewidth=2, markersize=10, label='GAMA G15 ($r < 19.8$)') 
    sub.scatter([0.], [0.], s=1, c='C1', label='$z$ failure') 
    #sub.vlines(21, 0., 2., color='C1', linestyle='--', linewidth=1)
    sub.set_xlabel(r'$r$ fiber magnitude', fontsize=25)
    sub.set_xlim([18., 22.5]) 
    sub.set_ylabel(r'$z$ success rate', fontsize=25)
    sub.set_ylim(0.6, 1.1) 
    sub.legend(loc='lower left', handletextpad=0., fontsize=15) 

    sub = fig.add_subplot(122) 
    sub.scatter(r_fiber_mag, _emline, c='k', s=1)
    sub.scatter(r_fiber_mag[(r_mag_gama > 18.0) & ~zsuccess], _emline[(r_mag_gama > 18.0) & ~zsuccess], c='C1', s=1)
    #sub.vlines(21, -1., 2., color='C1', linestyle='--', linewidth=1)
    sub.set_xlabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 23.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=20) 
    sub.set_ylim(-1., 2.) 
    
    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(dir_dat, 'G15.zsuccess.fibermag.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def G15_zsuccess_rfibermag(): 
    ''' calculations using G15 galaxies 
    '''
    # compile all the G15 meta-data and redshift success 
    keys = ['gama-spec/z', 'legacy-photo/flux_g', 'legacy-photo/flux_r', 'legacy-photo/flux_z', 'legacy-photo/flux_w1', 'legacy-photo/apflux_r', 'gama-photo/r_petro']
    data, redrock = [], [] 
    for iexp in range(1,14): 
        # meta data 
        fexp = h5py.File(os.path.join(dir_dat, 'GALeg.g15.sourceSpec.%iof13.hdf5' % iexp), 'r') 
        datum = [] 
        for k in keys: 
            if 'apflux_r' in k: 
                datum.append(fexp[k][...][:,1]) 
            else: 
                datum.append(fexp[k][...]) 
        data.append(np.array(datum))

        # redrock outputs 
        rr = fits.open(os.path.join(dir_dat, 'GALeg.g15.bgsSpec.%iof13.default_exp.rr.fits' % iexp))[1].data
        redrock.append(np.array([rr['Z'], rr['DELTACHI2'], rr['ZWARN']]))
    data = np.concatenate(data, axis=1)
    redrock = np.concatenate(redrock, axis=1) 
    
    # unpack values 
    redshift = data[0]

    g_mag   = UT.flux2mag(data[1], method='log')
    r_mag   = UT.flux2mag(data[2], method='log')
    z_mag   = UT.flux2mag(data[3], method='log')
    w1_mag  = UT.flux2mag(data[4], method='log')
    
    r_fiber_mag = UT.flux2mag(data[5], method='log') # aperture flux
    r_mag_gama  = data[6]  # r-band magnitude from GAMA (SDSS) photometry

    # color probe of emission line galaxies based on WISE and opitcal colors 
    _emline = (z_mag - w1_mag) - 3./2.5 * (g_mag - r_mag) + 1.2 

    z_rr    = redrock[0] 
    dchi2   = redrock[1]
    zwarn   = redrock[2]
    
    # calculate redshift success
    zsuccess = UT.zsuccess(z_rr, redshift, zwarn, deltachi2=dchi2, min_deltachi2=40.) 

    wmean, rate, err_rate = UT.zsuccess_rate(r_fiber_mag, zsuccess | (r_mag_gama < 18.), 
            range=[18,23], nbins=28, bin_min=10) 

    # r-mag selection 
    fig = plt.figure(figsize=(15, 4))
    sub = fig.add_subplot(121) 
    sub.plot([15., 23.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(wmean, rate, err_rate, fmt='.k', elinewidth=2, markersize=10, label='GAMA G15 ($r < 19.8$)') 
    sub.set_xlabel(r'$r$ fiber magnitude', fontsize=25)
    sub.set_xlim([18., 22.5]) 
    sub.set_ylabel(r'$z$ success rate', fontsize=25)
    sub.set_ylim(0.6, 1.1) 
    sub.legend(loc='lower left', handletextpad=0., markerscale=2, fontsize=15) 

    sub = fig.add_subplot(122) 
    sub.scatter(r_fiber_mag, _emline, c='k', s=1)
    sub.scatter(r_fiber_mag[(r_mag_gama > 18.0) & ~zsuccess], _emline[(r_mag_gama > 18.0) & ~zsuccess], c='C1', s=1)
    sub.set_xlabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 23.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=20) 
    sub.set_ylim(-1., 2.) 

    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(dir_dat, 'G15.zsuccess.rfibermag.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def G15_zsuccess_low_surface_brightness(): 
    ''' calculations using G15 galaxies 
    '''
    # compile all the G15 meta-data and redshift success 
    keys = ['gama-spec/z', 'legacy-photo/flux_g', 'legacy-photo/flux_r', 'legacy-photo/flux_z', 'legacy-photo/flux_w1', 'legacy-photo/apflux_r', 'gama-photo/r_petro']
    data, redrock = [], [] 
    for iexp in range(1,14): 
        # meta data 
        fexp = h5py.File(os.path.join(dir_dat, 'GALeg.g15.sourceSpec.%iof13.hdf5' % iexp), 'r') 
        datum = [] 
        for k in keys: 
            if 'apflux_r' in k: 
                datum.append(fexp[k][...][:,1]) 
            else: 
                datum.append(fexp[k][...]) 
        data.append(np.array(datum))

        # redrock outputs 
        rr = fits.open(os.path.join(dir_dat, 'GALeg.g15.bgsSpec.%iof13.default_exp.rr.fits' % iexp))[1].data
        redrock.append(np.array([rr['Z'], rr['DELTACHI2'], rr['ZWARN']]))
    data = np.concatenate(data, axis=1)
    redrock = np.concatenate(redrock, axis=1) 
    
    # unpack values 
    redshift = data[0]

    g_mag   = UT.flux2mag(data[1], method='log')
    r_mag   = UT.flux2mag(data[2], method='log')
    z_mag   = UT.flux2mag(data[3], method='log')
    w1_mag  = UT.flux2mag(data[4], method='log')
    
    r_fiber_mag = UT.flux2mag(data[5], method='log') # aperture flux
    r_mag_gama  = data[6]  # r-band magnitude from GAMA (SDSS) photometry

    # color probe of emission line galaxies based on WISE and opitcal colors 
    _emline = (z_mag - w1_mag) - 3./2.5 * (g_mag - r_mag) + 1.2 

    z_rr    = redrock[0] 
    dchi2   = redrock[1]
    zwarn   = redrock[2]

    # calculate redshift success
    zsuccess = UT.zsuccess(z_rr, redshift, zwarn, deltachi2=dchi2, min_deltachi2=40.) 

    sample_cut = (r_mag_gama < 19.5) 
    
    # r-mag selection 
    fig = plt.figure(figsize=(10, 4))
    sub = fig.add_subplot(121) 
    sub.scatter(r_mag_gama[sample_cut], r_fiber_mag[sample_cut], s=1, c='k')
    sub.scatter(r_mag_gama[sample_cut & (r_mag_gama > 18.5) & ~zsuccess], r_fiber_mag[sample_cut & (r_mag_gama > 18.5) & ~zsuccess], s=1, c='C1')
    sub.plot([16, 23], [16, 23], c='k', ls='--') 
    sub.set_xlabel('$r$ magnitude', fontsize=25) 
    sub.set_xlim(16., 23.) 
    sub.set_ylabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_ylim(16., 23.) 
    sub.legend(loc='lower right', handletextpad=0., markerscale=5, fontsize=15) 

    sub = fig.add_subplot(122) 
    sub.scatter(r_fiber_mag[sample_cut], _emline[sample_cut], c='k', s=1)
    sub.scatter(r_fiber_mag[sample_cut & (r_mag_gama > 18.5) & ~zsuccess], 
            _emline[sample_cut & (r_mag_gama > 18.5) & ~zsuccess], c='C1', s=1)
    #sub.vlines(21, -1., 2., color='k', linestyle='--', linewidth=1)
    sub.set_xlabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 23.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=20) 
    sub.set_ylim(-1., 2.) 

    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(dir_dat, 'G15.zsuccess.low_surface_brightness.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


if __name__=='__main__': 
    #G15_rmag_rfibermag()
    G15_zsuccess()
    #G15_zsuccess_rfibermag()
    #G15_zsuccess_low_surface_brightness()
