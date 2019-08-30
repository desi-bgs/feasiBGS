#!/bin/python 
'''
redshift success rate calculations necessary for SV preparation 
'''
import os 
import h5py 
import pickle 
import numpy as np 
import scipy as sp 
import corner as DFM 
from itertools import product 
# --- desi --- 
import specsim.config 
from desisurvey import etc as ETC
# --- astropy --- 
import astropy.units as u
from astropy.io import fits
from astropy.table import Table as aTable
# --- sklearn ---
from sklearn.gaussian_process import GaussianProcessRegressor as GPR 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression as LR 
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


def zsuccess_surveysimExposures(specfile='GALeg.g15.sourceSpec.3000.hdf5', expfile=None, seed=0, min_deltachi2=40.):
    ''' plot the compiled redshift success rate for the redrock output 
    of BGS-like spectra for the nexp observing conditions

    :param spec_flag: 
        noiseless source spectra file. (default: 'GALeg.g15.sourceSpec.3000.hdf5') 
    '''
    # read in noiseless spectra (for true redshift and r-band magnitude) 
    _fspec = os.path.join(dir_dat, specfile)
    fspec = h5py.File(_fspec, 'r') 
    ztrue = fspec['gama-spec']['z'].value 
    r_mag_legacy = UT.flux2mag(fspec['legacy-photo']['flux_r'].value, method='log')
    r_fibermag = fspec['r_mag_apflux'][...]

    # read in sampled exposures
    _fexp = os.path.join(dir_dat, expfile)
    fexps = h5py.File(_fexp.replace('.fits', '.sample.seed%i.hdf5' % seed), 'r') 
    nexps = len(fexps['airmass'][...]) 
    
    # read in nominal dark sky 
    config = specsim.config.load_config('desi')
    atm_config = config.atmosphere
    surface_brightness_dict = config.load_table(
        atm_config.sky, 'surface_brightness', as_dict=True)
    _wave    = config.wavelength # wavelength 
    _Idark   = surface_brightness_dict['dark'].copy().value
    
    ncol = 4
    nrow = int(np.ceil(float(nexps)/ncol)) 

    for rname, rmag in zip(['r_mag', 'r_fibermag'], [r_mag_legacy, r_fibermag]): 
        fig = plt.figure(figsize=(4*ncol, 4*nrow))
        for iexp in range(nexps): 
            print('--- exposure %i ---' % iexp) 
            print('%s' % ', '.join(['%s = %.2f' % (k, fexps[k][iexp]) 
                for k in ['texp_total', 'airmass', 'moon_alt', 'moon_ill', 'moon_sep', 'sun_alt', 'sun_sep']]))
                
            # read in redrock outputs
            f_bgs   = _fspec.replace('sourceSpec', 'bgsSpec').replace('.hdf5', '.sample%i.seed%i.rr.fits' % (iexp, seed))
            rr      = fits.open(f_bgs)[1].data
            zrr     = rr['Z']
            dchi2   = rr['DELTACHI2']
            zwarn   = rr['ZWARN']

            # redshift success 
            zsuccess_exp = UT.zsuccess(zrr, ztrue, zwarn, deltachi2=dchi2, min_deltachi2=min_deltachi2) 
            if rname == 'r_mag': 
                wmean, rate, err_rate = UT.zsuccess_rate(rmag, zsuccess_exp, range=[15,22], nbins=28, bin_min=10) 
            elif rname == 'r_fibermag': 
                wmean, rate, err_rate = UT.zsuccess_rate(rmag, zsuccess_exp, range=[18,22], nbins=28, bin_min=10) 
            
            sub = fig.add_subplot(nrow, ncol, iexp+1)
            sub.plot([15., 22.], [1., 1.], c='k', ls='--', lw=2)
            sub.errorbar(wmean, rate, err_rate, fmt='.C0', elinewidth=2, markersize=10)
            sub.vlines(19.5, 0., 1.2, color='k', linestyle=':', linewidth=1)
            if rname == 'r_mag': 
                sub.set_xlim([16.5, 21.]) 
            elif rname == 'r_fibermag': 
                sub.set_xlim([18., 22.]) 
            sub.set_ylim([0.6, 1.1])
            sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
            if iexp == ncol-1: 
                sub.legend(loc='lower right', markerscale=0.5, handletextpad=-0.7, prop={'size': 20})
            if (iexp % ncol) != 0:  
                sub.set_yticklabels([]) 
            if (iexp // ncol) != nrow-1: 
                sub.set_xticklabels([]) 

            #wlim = (fexps['wave'][...] > 6800.) & (fexps['wave'][...] < 7200.) 
            #_wlim = (_wave.value > 6800.) & (_wave.value < 7200.) 
            #print('sky is %.2fx brighter than nominal at 7000A' % 
            #        (np.median(fexps['sky'][iexp][wlim])/np.median(_Idark[_wlim])))
            #fbright = ETC.bright_exposure_factor(fexps['moon_ill'][iexp], fexps['moon_alt'][iexp], np.array(fexps['moon_sep'][iexp]),
            #        fexps['sun_alt'][iexp], np.array(fexps['sun_sep'][iexp]), np.array(fexps['airmass'][iexp]))
            #print('bright factor = %.1f' % fbright) 
            #_ETC = ETC.ExposureTimeCalculator() 
            #fweather = _ETC.weather_factor(fexps['seeing'][iexp], fexps['transp'][iexp])
            #print('weather factor = %.1f' % fweather) 
            #fairmass = ETC.airmass_exposure_factor(fexps['airmass'][iexp]) 
            #print('airmass factor = %.1f' % fairmass) 

            sub.text(0.05, 0.05, ('%i.' % (iexp+1)), ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
            sub.text(0.95, 0.4, r'$t_{\rm exp} = %.f$' % (fexps['texp_total'][iexp]), 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=10) 
            #sub.text(0.95, 0.275, r'exp factor = %.1f, airmass = %.2f' % (fbright, fexps['airmass'][iexp]), 
            #        ha='right', va='bottom', transform=sub.transAxes, fontsize=10) 
            sub.text(0.95, 0.15, r'moon ill=%.2f, alt=%.f, sep=%.f' % 
                    (fexps['moon_ill'][iexp], fexps['moon_alt'][iexp], fexps['moon_sep'][iexp]), 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=10) 
            sub.text(0.95, 0.025, r'sun alt=%.f, sep=%.f' % 
                    (fexps['sun_alt'][iexp], fexps['sun_sep'][iexp]), 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=10) 

        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        if rname == 'r_mag': 
            bkgd.set_xlabel(r'Legacy DR7 $r$ magnitude', labelpad=10, fontsize=30)
        elif rname == 'r_fibermag':
            bkgd.set_xlabel(r'Legacy DR7 $r$ 1" aperture magnitude', labelpad=10, fontsize=30)
        bkgd.set_ylabel(r'redrock redshift success', labelpad=10, fontsize=30)

        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        ffig = os.path.join(dir_dat, 
                'GALeg.g15%s.zsuccess.%s.min_deltachi2_%.f.png' % (expfile.replace('.fits', '.sample.seed%i' % seed), rname, min_deltachi2))
        fig.savefig(ffig, bbox_inches='tight') 
    return None 


def zsuccess_fibermag_halpha(specfile='GALeg.g15.sourceSpec.3000.hdf5', expfile=None, seed=0, min_deltachi2=40.):
    ''' plot the compiled redshift success rate for the redrock output 
    of BGS-like spectra for the nexp observing conditions

    :param spec_flag: 
        noiseless source spectra file. (default: 'GALeg.g15.sourceSpec.3000.hdf5') 
    '''
    # read in noiseless spectra (for true redshift and r-band magnitude) 
    _fspec = os.path.join(dir_dat, specfile)
    fspec = h5py.File(_fspec, 'r') 
    ztrue = fspec['gama-spec']['z'][...]
    r_mag_legacy = UT.flux2mag(fspec['legacy-photo']['flux_r'].value)
    
    # r-band fiber magnitude 
    r_fibermag = fspec['r_mag_apflux'][...]
    
    # halpha probe 
    g_mag = UT.flux2mag(fspec['legacy-photo']['flux_g'][...], method='log')
    r_mag = UT.flux2mag(fspec['legacy-photo']['flux_r'][...], method='log')
    z_mag = UT.flux2mag(fspec['legacy-photo']['flux_z'][...], method='log')
    w1_mag = UT.flux2mag(fspec['legacy-photo']['flux_w1'][...], method='log')
    
    _halpha = (z_mag - w1_mag) - 3./2.5 * (g_mag - r_mag) + 1.2 

    # read in sampled exposures
    _fexp = os.path.join(dir_dat, expfile)
    fexps = h5py.File(_fexp.replace('.fits', '.sample.seed%i.hdf5' % seed), 'r') 
    nexps = len(fexps['airmass'][...]) 
    
    for iexp in range(nexps): 
        print('--- exposure %i ---' % iexp) 
        print('%s' % ', '.join(['%s = %.2f' % (k, fexps[k][iexp]) 
            for k in ['texp_total', 'airmass', 'moon_alt', 'moon_ill', 'moon_sep', 'sun_alt', 'sun_sep']]))
            
        # read in redrock outputs
        f_bgs   = _fspec.replace('sourceSpec', 'bgsSpec').replace('.hdf5', '.sample%i.seed%i.rr.fits' % (iexp, seed))
        rr      = fits.open(f_bgs)[1].data
        zrr     = rr['Z']
        dchi2   = rr['DELTACHI2']
        zwarn   = rr['ZWARN']

        # redshift success 
        zsuccess_exp = UT.zsuccess(zrr, ztrue, zwarn, deltachi2=dchi2, min_deltachi2=min_deltachi2) 
        
        fig = plt.figure(figsize=(5,5))
        sub = fig.add_subplot(111)
        sub.scatter(r_fibermag[zsuccess_exp], _halpha[zsuccess_exp], c='C0', s=2, label='$z$ success') 
        sub.scatter(r_fibermag[~zsuccess_exp], _halpha[~zsuccess_exp], c='C1', s=2, label='$z$ fail') 
        sub.set_xlabel('$r$-band fiber magnitude', fontsize=25) 
        sub.set_xlim(18., 22.) 
        sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=25) 
        sub.set_ylim(-1., 2.) 

        ffig = os.path.join(dir_dat, 
                'GALeg.g15%s.zsuccess.min_deltachi2_%.f.fibermag_halpha.png' % (expfile.replace('.fits', '.sample%i.seed%i' % (iexp, seed)), min_deltachi2))
        fig.savefig(ffig, bbox_inches='tight') 
    return None 
