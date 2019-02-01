'''

functions for taking redrock outputs and calculating redshift
success rate 


'''
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


def zsuccess(zrr, ztrue, zwarn):
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
    crit = (dz_1pz < 0.003) & (zwarn == 0)
    return crit


if __name__=="__main__": 
    for i in range(1, 11): 
        zsuccess_iexp(i, nexp=15, method='spacefill', nsub=3000, spec_flag='')
