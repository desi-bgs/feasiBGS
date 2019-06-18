'''

functions for taking redrock outputs and calculating redshift
success rate 


'''
import os 
import h5py 
import numpy as np 
# -- astropy -- 
from astropy.io import fits
from astropy import units as u
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


def zsuccess_iexps(nexp, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=9.):
    ''' plot the compiled redshift success rate for the redrock output 
    of BGS-like spectra for the nexp observing conditions

    :param nexp: 
        nubmer of total observing condition sampled from `surveysim`
        exposures

    :param nsub: (default: 3000)  
        number of spectra in the G15 subsample. 
    
    :param spec_flag: (default: '') 
        string specifying different spectra runs 
    '''
    nrow = 3 
    ncol = (nexp + nrow - 1) // nrow
    fig = plt.figure(figsize=(18,9))#6*ncol, 6*nrow))

    for iexp in range(nexp): 
        # read in noiseless spectra (for true redshift and r-band magnitude) 
        fspec = h5py.File(''.join([UT.dat_dir(), 
            'bgs_zsuccess/', 'g15.simSpectra.', str(nsub), spec_flag, '.v2.hdf5']), 'r') 
        ztrue = fspec['gama-spec']['z'].value 
        r_mag_legacy = UT.flux2mag(fspec['legacy-photo']['flux_r'].value)

        # read in sampled exposures
        fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
            'bgs_survey_exposures.subset.', str(nexp), method, '.hdf5']), 'r')
        
        # read in redrock outputs
        f_bgs_old = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
            'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
            '.old_sky.v2.rr.fits'])
        f_bgs_new = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
            'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
            '.new_sky.v2.rr.fits'])
        rr_old = fits.open(f_bgs_old)[1].data
        rr_new = fits.open(f_bgs_new)[1].data
        zrr_old = rr_old['Z']
        zrr_new = rr_new['Z']
        dchi2_old = rr_old['DELTACHI2']
        dchi2_new = rr_new['DELTACHI2']
        zwarn_old = rr_old['ZWARN']
        zwarn_new = rr_new['ZWARN']

        # redshift success 
        zsuccess_old = zsuccess(zrr_old, ztrue, zwarn_old, deltachi2=dchi2_old, min_deltachi2=min_deltachi2) 
        zsuccess_new = zsuccess(zrr_new, ztrue, zwarn_new, deltachi2=dchi2_new, min_deltachi2=min_deltachi2) 
        
        sub = fig.add_subplot(nrow, ncol, iexp+1)
        sub.plot([15., 22.], [1., 1.], c='k', ls='--', lw=2)
        for i, zs, col, lbl in zip(range(2), [zsuccess_old, zsuccess_new], ['k', 'C1'], ['old sky', 'new sky']): 
            wmean, rate, err_rate = zsuccess_rate(r_mag_legacy, zs, range=[15,22], nbins=28, bin_min=10) 
            sub.errorbar(wmean, rate, err_rate, fmt='.'+col, elinewidth=(2-i), markersize=5*(2-i), label=lbl)
        sub.vlines(19.5, 0., 1.2, color='k', linestyle=':', linewidth=1)
        sub.set_xlim([16., 21.]) 
        sub.set_ylim([0.6, 1.1])
        sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
        if iexp == ncol-1: 
            sub.legend(loc='lower right', markerscale=0.5, handletextpad=-0.7, prop={'size': 20})
        if (iexp % ncol) != 0:  
            sub.set_yticklabels([]) 
        if (iexp // ncol) != nrow-1: 
            sub.set_xticklabels([]) 
        sub.text(0.05, 0.05, ('%i.' % iexp), ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
        
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel("$k$ [$h$/Mpc]", labelpad=10, fontsize=25) 
    bkgd.set_xlabel(r'Legacy DR7 $r$ magnitude', labelpad=10, fontsize=30)
    bkgd.set_ylabel(r'redrock redshift success', labelpad=10, fontsize=30)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    ffig = os.path.join(UT.dat_dir(), 'bgs_zsuccess', 
            'g15.simSpectra.%i%s.texp_default.%i%s.zsuccess.min_deltachi2_%.f.v2.png' % (nsub, spec_flag, nexp, method, min_deltachi2))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def zsuccess_iexps_fibermag(nexp, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=9.):
    ''' plot the compiled redshift success rate as a function of fiber mag (surface 
    brightness) for the redrock output of BGS-like spectra for the nexp observing 
    conditions. 

    :param nexp: 
        nubmer of total observing condition sampled from `surveysim`
        exposures

    :param nsub: (default: 3000)  
        number of spectra in the G15 subsample. 
    
    :param spec_flag: (default: '') 
        string specifying different spectra runs 
    '''
    nrow = 3 
    ncol = (nexp + nrow - 1) // nrow
    fig = plt.figure(figsize=(18,9))#6*ncol, 6*nrow))

    for iexp in range(nexp): 
        # read in noiseless spectra (for true redshift and r-band magnitude) 
        fspec = h5py.File(''.join([UT.dat_dir(), 
            'bgs_zsuccess/', 'g15.simSpectra.', str(nsub), spec_flag, '.v2.hdf5']), 'r') 
        ztrue = fspec['gama-spec']['z'].value 
        r_mag_legacy = fspec['r_mag_apflux'].value #UT.flux2mag(fspec['legacy-photo']['flux_r'].value)

        # read in sampled exposures
        fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
            'bgs_survey_exposures.subset.', str(nexp), method, '.hdf5']), 'r')
        
        # read in redrock outputs
        f_bgs_old = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
            'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
            '.old_sky.v2.rr.fits'])
        f_bgs_new = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
            'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
            '.new_sky.v2.rr.fits'])
        rr_old = fits.open(f_bgs_old)[1].data
        rr_new = fits.open(f_bgs_new)[1].data
        zrr_old = rr_old['Z']
        zrr_new = rr_new['Z']
        dchi2_old = rr_old['DELTACHI2']
        dchi2_new = rr_new['DELTACHI2']
        zwarn_old = rr_old['ZWARN']
        zwarn_new = rr_new['ZWARN']

        # redshift success 
        zsuccess_old = zsuccess(zrr_old, ztrue, zwarn_old, deltachi2=dchi2_old, min_deltachi2=min_deltachi2) 
        zsuccess_new = zsuccess(zrr_new, ztrue, zwarn_new, deltachi2=dchi2_new, min_deltachi2=min_deltachi2) 
        
        sub = fig.add_subplot(nrow, ncol, iexp+1)
        sub.plot([18., 24.], [1., 1.], c='k', ls='--', lw=2)
        for i, zs, col, lbl in zip(range(2), [zsuccess_old, zsuccess_new], ['k', 'C1'], ['old sky', 'new sky']): 
            wmean, rate, err_rate = zsuccess_rate(r_mag_legacy, zs, range=[18,24], nbins=28, bin_min=10) 
            sub.errorbar(wmean, rate, err_rate, fmt='.'+col, elinewidth=(2-i), markersize=5*(2-i), label=lbl)
        sub.set_xlim([18., 24.]) 
        sub.set_ylim([0.4, 1.1])
        #sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
        if iexp == ncol-1: 
            sub.legend(loc='lower right', markerscale=0.5, handletextpad=-0.7, prop={'size': 20})
        if (iexp % ncol) != 0:  
            sub.set_yticklabels([]) 
        if (iexp // ncol) != nrow-1: 
            sub.set_xticklabels([]) 
        sub.text(0.05, 0.05, ('%i.' % iexp), ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
        
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel("$k$ [$h$/Mpc]", labelpad=10, fontsize=25) 
    bkgd.set_xlabel(r'Legacy DR7 $r$ fiber-magnitude', labelpad=10, fontsize=30)
    bkgd.set_ylabel(r'redrock redshift success', labelpad=10, fontsize=30)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    ffig = os.path.join(UT.dat_dir(), 'bgs_zsuccess', 
            'g15.simSpectra.%i%s.texp_default.%i%s.fibermag.zsuccess.min_deltachi2_%.f.v2.png' % 
            (nsub, spec_flag, nexp, method, min_deltachi2))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def zsuccess_iexps_gamapetro(nexp, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=9.):
    ''' plot the compiled redshift success rate as a function of fiber mag (surface 
    brightness) for the redrock output of BGS-like spectra for the nexp observing 
    conditions. 

    :param nexp: 
        nubmer of total observing condition sampled from `surveysim`
        exposures

    :param nsub: (default: 3000)  
        number of spectra in the G15 subsample. 
    
    :param spec_flag: (default: '') 
        string specifying different spectra runs 
    '''
    nrow = 3 
    ncol = (nexp + nrow - 1) // nrow
    fig = plt.figure(figsize=(18,9))#6*ncol, 6*nrow))

    for iexp in range(nexp): 
        # read in noiseless spectra (for true redshift and r-band magnitude) 
        fspec = h5py.File(''.join([UT.dat_dir(), 
            'bgs_zsuccess/', 'g15.simSpectra.', str(nsub), spec_flag, '.v2.hdf5']), 'r') 
        ztrue = fspec['gama-spec']['z'].value 
        r_mag = fspec['gama-photo']['r_petro'].value 

        # read in sampled exposures
        fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
            'bgs_survey_exposures.subset.', str(nexp), method, '.hdf5']), 'r')
        
        # read in redrock outputs
        f_bgs_old = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
            'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
            '.old_sky.v2.rr.fits'])
        f_bgs_new = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
            'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
            '.new_sky.v2.rr.fits'])
        rr_old = fits.open(f_bgs_old)[1].data
        rr_new = fits.open(f_bgs_new)[1].data
        zrr_old = rr_old['Z']
        zrr_new = rr_new['Z']
        dchi2_old = rr_old['DELTACHI2']
        dchi2_new = rr_new['DELTACHI2']
        zwarn_old = rr_old['ZWARN']
        zwarn_new = rr_new['ZWARN']

        # redshift success 
        zsuccess_old = zsuccess(zrr_old, ztrue, zwarn_old, deltachi2=dchi2_old, min_deltachi2=min_deltachi2) 
        zsuccess_new = zsuccess(zrr_new, ztrue, zwarn_new, deltachi2=dchi2_new, min_deltachi2=min_deltachi2) 
        
        sub = fig.add_subplot(nrow, ncol, iexp+1)
        sub.plot([16., 20.], [1., 1.], c='k', ls='--', lw=2)
        for i, zs, col, lbl in zip(range(2), [zsuccess_old, zsuccess_new], ['k', 'C1'], ['old sky', 'new sky']): 
            wmean, rate, err_rate = zsuccess_rate(r_mag, zs, range=[16,20.], nbins=28, bin_min=10) 
            sub.errorbar(wmean, rate, err_rate, fmt='.'+col, elinewidth=(2-i), markersize=5*(2-i), label=lbl)
        sub.set_xlim([16., 20.]) 
        sub.set_ylim([0.4, 1.1])
        #sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
        if iexp == ncol-1: 
            sub.legend(loc='lower right', markerscale=0.5, handletextpad=-0.7, prop={'size': 20})
        if (iexp % ncol) != 0:  
            sub.set_yticklabels([]) 
        if (iexp // ncol) != nrow-1: 
            sub.set_xticklabels([]) 
        sub.text(0.05, 0.05, ('%i.' % iexp), ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
        
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel("$k$ [$h$/Mpc]", labelpad=10, fontsize=25) 
    bkgd.set_xlabel(r'GAMA $r$ petro mag', labelpad=10, fontsize=30)
    bkgd.set_ylabel(r'redrock redshift success', labelpad=10, fontsize=30)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    ffig = os.path.join(UT.dat_dir(), 'bgs_zsuccess', 
            'g15.simSpectra.%i%s.texp_default.%i%s.gamapetro.zsuccess.min_deltachi2_%.f.v2.png' % 
            (nsub, spec_flag, nexp, method, min_deltachi2))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def zsuccess_iexp(iexp, nexp=15, method='spacefill', nsub=3000, spec_flag=''):
    ''' plot the redshift success rate for the redrock output of 
    BGS-like spectra for the iexp (of nexp) observing condition 

    :param iexp: 
        index of the observing condition 
    
    :param nexp: (default: 15) 
        nubmer of total observing condition sampled from `surveysim`
        exposures

    :param nsub: (default: 3000)  
        number of spectra in the G15 subsample. 
    
    :param spec_flag: (default: '') 
        string specifying different spectra runs 
    '''
    # read in noiseless spectra (for true redshift and r-band magnitude) 
    fspec = h5py.File(''.join([UT.dat_dir(), 
        'bgs_zsuccess/', 'g15.simSpectra.', str(nsub), spec_flag, '.hdf5']), 'r') 
    ztrue = fspec['gama-spec']['z'].value 
    r_mag_legacy = UT.flux2mag(fspec['legacy-photo']['flux_r'].value)

    # read in sampled exposures
    fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
        'bgs_survey_exposures.subset.', str(nexp), method, '.hdf5']), 'r')
    
    # read in redrock outputs
    f_bgs_old = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
        'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
        '.old_sky.rr.fits'])
    f_bgs_new = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
        'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
        '.new_sky.rr.fits'])
    rr_old = fits.open(f_bgs_old)[1].data
    rr_new = fits.open(f_bgs_new)[1].data
    zrr_old = rr_old['Z']
    zrr_new = rr_new['Z']
    zwarn_old = rr_old['ZWARN']
    zwarn_new = rr_new['ZWARN']
    
    # redshift success 
    zsuccess_old = zsuccess(zrr_old, ztrue, zwarn_old) 
    zsuccess_new = zsuccess(zrr_new, ztrue, zwarn_new) 
    
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot([15., 22.], [1., 1.], c='k', ls='--', lw=2)
    for i, zs, col, lbl in zip(range(2), [zsuccess_old, zsuccess_new], ['k', 'C1'], ['old sky', 'new sky']): 
        wmean, rate, err_rate = zsuccess_rate(r_mag_legacy, zs, range=[15,22], nbins=28, bin_min=10) 
        sub.errorbar(wmean, rate, err_rate, fmt='.'+col, elinewidth=(2-i), markersize=5*(2-i), label=lbl)
    sub.vlines(19.5, 0., 1.2, color='k', linestyle=':', linewidth=1)
    sub.set_xlabel(r'$r$ magnitude Legacy DR7', fontsize=20)
    sub.set_xlim([15., 21.]) 
    sub.set_ylabel(r'redrock redshift success', fontsize=20)
    sub.set_ylim([0.5, 1.2])
    sub.legend(loc='lower left', handletextpad=0., prop={'size': 20})
    sub.set_title(("$t_\mathrm{exp}=%.0f$sec, airmass=%.1f\nMoon Ill=%.2f, Alt=%.0f, Sep=%.0f \nSun Alt=%.0f, Sep=%.f" % 
        (fexps['exptime'][iexp], fexps['airmass'][iexp], fexps['moonfrac'][iexp], fexps['moonalt'][iexp], fexps['moonsep'][iexp], 
            fexps['sunalt'][iexp], fexps['sunsep'][iexp])), fontsize=20)
    fig.savefig(''.join([UT.dat_dir(), 'bgs_zsuccess/',
        'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
        '.zsuccess.png']), bbox_inches='tight') 
    return None 


def zsuccess_rate(prop, zsuccess_cond, range=None, nbins=20, bin_min=2):
    ''' measure the redshift success rate along with property `prop`

    :params prop: 
        array of properties (i.e. Legacy r-band magnitude) 

    :params zsuccess_cond:
        boolean array indicating redshift success 

    :params range: (default: None) 
        range of the `prop` 

    :params nbins: (default: 20) 
        number of bins to divide `prop` by 
    
    :params bin_min: (default: 2)  
        minimum number of objects in bin to exlcude it 

    :return wmean: 
        weighted mean of `prop` in the bins 

    :return e1: 
        redshift success rate in the bins

    :return ee1: 
        simple poisson error on the success rate
    '''
    h0, bins = np.histogram(prop, bins=nbins, range=range)
    hv, _ = np.histogram(prop, bins=bins, weights=prop)
    h1, _ = np.histogram(prop[zsuccess_cond], bins=bins)
    
    good = h0 > bin_min
    hv = hv[good]
    h0 = h0[good]
    h1 = h1[good]

    wmean = hv / h0 # weighted mean 
    rate = h1.astype("float") / (h0.astype('float') + (h0==0))
    e_rate = np.sqrt(rate * (1 - rate)) / np.sqrt(h0.astype('float') + (h0 == 0))
    return wmean, rate, e_rate


def zsuccess(zrr, ztrue, zwarn, deltachi2=None, min_deltachi2=9.):
    ''' apply redshift success crition

    |z_redrock - z_true|/(1+z_true) < 0.003 and ZWARN flag = 0 

    :params zrr: 
        redrock best-fit redshift

    :params ztrue: 
        true redshift 

    :params zwarn: 
        zwarn flag value 

    :return crit: 
        boolean array indiciate which redshifts were successfully
        measured by redrock 
    '''
    dz_1pz = np.abs(ztrue - zrr)/(1.+ztrue)
    if deltachi2 is None: 
        crit = (dz_1pz < 0.003) & (zwarn == 0)
    else: 
        crit = (dz_1pz < 0.003) & (zwarn == 0) & (deltachi2 > min_deltachi2) 
    return crit


if __name__=="__main__": 
    #zsuccess_iexps(15, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=9.)
    #zsuccess_iexps(15, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=25.)
    zsuccess_iexps(15, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=40.) # 40 is boss limit 
    zsuccess_iexps_fibermag(15, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=40.)
    zsuccess_iexps_gamapetro(15, method='spacefill', nsub=3000, spec_flag='', min_deltachi2=40.)

    #for i in range(1, 15): 
    #    zsuccess_iexp(i, nexp=15, method='spacefill', nsub=3000, spec_flag='')
