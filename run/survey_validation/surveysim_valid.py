#!/bin/python 
'''
scripts to validate surveysim outputs
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


def buildGP_bright_exposure_factor(expfile='exposures_surveysim_fork_150sv0p4.fits'): 
    ''' build Gaussian process emulator for the bright exposure factor 
    and validate. 1) select training sample that fully encompasses 
    the range of observing conditions of surveysim output exposures.
    2) train GP. 3) validate on test sample of observing conditions. 

    :param expfile: 
        surveysim output exposure file. (default: 'exposures_surveysim_fork_150sv0p4.fits') 
    '''
    # read in surveysim output file 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
    nexp = len(exps['airmass']) 

    # read in precomputed exposure factors
    exp_factors = np.load(os.path.join(dir_dat, 'exposure_factor.BGS.%s' % expfile.replace('.fits', '.npy'))) 

    # select training sample that fully encompasses the surveysim output exposures 
    
    # split into twilight and non twilight 
    for cond in ['twilight', 'not_twilight']: 
        if cond == 'twilight': 
            cut = (exps['sun_alt'] >= -20.) 
            props = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep']
            nbins = [2, 5, 4, 3, 3, 3]
        elif cond == 'not_twilight': 
            cut = (exps['sun_alt'] < -20.) 
            props = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep']
            nbins = [4, 4, 4, 4]
        print('%i exposures in %s' % (np.sum(cut), cond)) 
        #n_test = int(np.floor(0.1 * float(np.sum(cut)) * 0.3)) 
        #print('%i exposures in %s test sample' % (n_test, cond)) 
        
        prop_range = np.zeros((len(props), 2)) 
        for i, prop in enumerate(props): 
            prop_range[i,0] = exps[prop][cut].min() 
            prop_range[i,1] = exps[prop][cut].max()
            print('%s: %f - %f' % (prop, prop_range[i,0], prop_range[i,1])) 
        
        # randomly select 100 exposures from surveysim output and calculate the exposure factors  
        i_test = np.arange(nexp)[cut]#np.random.choice(np.arange(nexp)[cut], size=n_test, replace=False) 
        theta_test, exp_factor_test = [], []
        for iexp in i_test: 
            # compile test parameters 
            theta_test.append(np.array([exps[k][iexp] for k in props])) 
            # exposure factor 
            exp_factor_test.append(exp_factors[iexp]) 
        theta_test = np.array(theta_test) 
        exp_factor_test = np.array(exp_factor_test)
    
        # training sample on a grid
        prop_bins = tuple([np.linspace(_prop_range[0], _prop_range[1], nbin)  for _prop_range, nbin in zip(prop_range, nbins)]) 
        #if cond == 'not_twilight': 
        #    prop_bins[2] = np.array([0., 5., 10., 20., 40., 60., 80.]) 
        
        # calculate exposure factors for training set 
        theta_train, exp_factor_train = [], [] 
        if cond == 'twilight':  
            for airmass, moon_ill, moon_alt, moon_sep, sun_alt, sun_sep in product(*prop_bins):  
                # compile training parameters 
                theta_train.append(np.array([airmass, moon_ill, moon_alt, moon_sep, sun_alt, sun_sep])) 
                # exposure factor 
                _exp_factor = ETC.bright_exposure_factor(moon_ill, moon_alt, np.array(moon_sep), 
                        sun_alt, np.array(sun_sep), np.array(airmass))
                exp_factor_train.append(_exp_factor) 
        elif cond == 'not_twilight':  
            for airmass, moon_ill, moon_alt, moon_sep in product(*prop_bins):  
                theta_train.append(np.array([airmass, moon_ill, moon_alt, moon_sep])) 
                # exposure factor 
                _exp_factor = ETC.bright_exposure_factor(moon_ill, moon_alt, np.array(moon_sep), 
                        -30., np.array(180.), np.array(airmass))
                exp_factor_train.append(_exp_factor) 

        if cond == 'twilight':         
            i_train = np.random.choice(np.arange(nexp)[cut], 500, replace=False) 
        elif cond == 'not_twilight': 
            i_train = np.random.choice(np.arange(nexp)[cut], 1000, replace=False) 

        for iexp in i_train: 
            # compile training parameters 
            theta_train.append(np.array([exps[k][iexp] for k in props])) 
            # exposure factor 
            exp_factor_train.append(exp_factors[iexp]) 
        
        if cond == 'not_twilight': 
            i_train = np.random.choice(np.setdiff1d(np.arange(nexp)[cut & (exps['moon_alt'] < 10.)], np.array(i_train)), 
                    200, replace=False) 
            for iexp in i_train: 
                # compile training parameters 
                theta_train.append(np.array([exps[k][iexp] for k in props])) 
                # exposure factor 
                exp_factor_train.append(exp_factors[iexp]) 

        theta_train = np.array(theta_train) 
        exp_factor_train = np.array(exp_factor_train)

        # train fits 
        _length_scale = np.ones(theta_train.shape[1])
        _length_scale[2] = 10. 
        kern = ConstantKernel(1.0, (1e-4, 1e4)) + ConstantKernel(1.0, (1e-4, 1e4)) * RBF(_length_scale, (1e-4, 1e4)) # kernel
        gp = GPR(kernel=kern, alpha=np.std(exp_factor_train)**2, n_restarts_optimizer=10) # instanciate a GP model
        gp.fit(theta_train, exp_factor_train)
        exp_factor_test_gp = gp.predict(theta_test)
        print(exp_factor_test_gp) 
        print(exp_factor_test) 
        
        # store GP 
        pickle.dump(gp, open(os.path.join(dir_dat, 'GP_bright_exp_factor.%s.p' % cond), 'wb')) # entire GP (for my purposes) 
        # store more memory efficiently (for surveysim) 
        f_gp_param = h5py.File(os.path.join(dir_dat, 'GP_bright_exp_factor.%s.params.hdf5' % cond), 'w') 
        f_gp_param.create_dataset('Xtrain', data=gp.X_train_) 
        f_gp_param.create_dataset('alpha', data=gp.alpha_) 
        f_gp_param.close() 
        f_gp_kernel = os.path.join(dir_dat, 'GP_bright_exp_factor.%s.kernel.p' % cond)
        pickle.dump(gp.kernel_, open(f_gp_kernel, 'wb'))

        if cond == 'twilight': 
            theta_train_twi = theta_train
            exp_factor_train_twi = exp_factor_train
            theta_test_twi = theta_test
            exp_factor_test_gp_twi = exp_factor_test_gp 
            exp_factor_test_twi = exp_factor_test 
        elif cond == 'not_twilight': 
            theta_train_notwi = theta_train
            exp_factor_train_notwi = exp_factor_train
            theta_test_notwi = theta_test
            exp_factor_test_gp_notwi = exp_factor_test_gp 
            exp_factor_test_notwi = exp_factor_test 

    badfit_twi = (np.abs((exp_factor_test_gp_twi / exp_factor_test_twi) - 1.) > 0.25) 
    badfit_notwi = (np.abs((exp_factor_test_gp_notwi / exp_factor_test_notwi) - 1.) > 0.25) 
    print('%i bad fits in twilight' % np.sum(badfit_twi)) 
    print('%i bad fits in not twilight' % np.sum(badfit_notwi)) 
    print(theta_test[badfit_notwi]) 

    # exposure time vs various properties 
    fig = plt.figure(figsize=(10,10)) 
    props   = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep']
    lims    = [(0.9, 2.1), (0.4, 1.), (-10., 90.), (40., 180.), (-90., -10.), (30., 180.)]
    lbls    = ['airmass', 'moon ill.', 'moon alt.', 'moon sep.', 'sun alt.', 'sun sep.']
    for _i, i, j in zip(range(4), [0, 2, 3, 4], [1, 1, 1, 5]):
        sub = fig.add_subplot(2,2,_i+1) 
        sub.scatter(exps[props[i]], exps[props[j]], s=1, c='k') 
        sub.scatter(theta_train_twi[:,i], theta_train_twi[:,j], s=2, c='C0', label='training set')
        sub.scatter(theta_test_twi[badfit_twi,i], theta_test_twi[badfit_twi,j], s=3, c='C1', label='fit is worse than $25\%$')
        if i < 4: 
            sub.scatter(theta_train_notwi[:,i], theta_train_notwi[:,j], s=2, c='C0')
            sub.scatter(theta_test_notwi[badfit_notwi,i], theta_test_notwi[badfit_notwi,j], s=3, c='C1')
        sub.set_xlabel(lbls[i], fontsize=20) 
        sub.set_xlim(lims[i]) 
        sub.set_ylabel(lbls[j], fontsize=20) 
        sub.set_ylim(lims[j]) 
    sub.legend(loc='lower left', markerscale=5, handletextpad=0.1, fontsize=15) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm exp}$ (sec)', fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.3) 
    fig.savefig(os.path.join(dir_dat, 'GP_bright_exp_factor.traintest.png'), bbox_inches='tight') 
    
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.scatter(exp_factor_test_notwi, exp_factor_test_gp_notwi, c='C0', s=1) 
    sub.scatter(exp_factor_test_twi, exp_factor_test_gp_twi, c='C1', s=1, label='twilight') 
    sub.scatter(exp_factor_test_notwi[badfit_notwi], exp_factor_test_gp_notwi[badfit_notwi], c='C3', s=1) 
    sub.plot([1., 50], [1., 50.], c='k', ls='--') 
    sub.legend(loc='upper left', markerscale=5, handletextpad=0.2, fontsize=20) 
    sub.set_xlabel('bright exposure factor', fontsize=20) 
    sub.set_xlim(0., 20) 
    sub.set_ylabel('predicted bright exposure factor', fontsize=20) 
    sub.set_ylim(0., 20) 
    fig.savefig(os.path.join(dir_dat, 'GP_bright_exp_factor.png'), bbox_inches='tight') 
    return None 


def _test_loadGP(): 
    ''' test the best way to store and load GP emulator for bright_exposure_factor 
    '''
    # read in surveysim output file 
    expfile = 'exposures_surveysim_fork_150sv0p4.fits'
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
    nexp = len(exps['airmass']) 

    for cond in ['twilight', 'not_twilight']: 
        if cond == 'twilight': 
            cut = (exps['sun_alt'] >= -20.) 
            props = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep']
        elif cond == 'not_twilight': 
            cut = (exps['sun_alt'] < -20.) 
            props = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep']
    
        # read in fully pickled GP 
        gp_true     = pickle.load(open(os.path.join(dir_dat, 'GP_bright_exp_factor.%s.p' % cond), 'rb')) 

        # read in stored GP parameters 
        f_gp_param = h5py.File(os.path.join(dir_dat, 'GP_bright_exp_factor.%s.params.hdf5' % cond), 'r') 
        _alpha_true = f_gp_param['alpha'][...]
        _Xtrain_true = f_gp_param['Xtrain'][...]
        # read in pickled GP kernel  
        _kern_true = pickle.load(open(os.path.join(dir_dat, 'GP_bright_exp_factor.%s.kernel.p' % cond), 'rb'))

        gp_load = GPR()
        gp_load.alpha_ = _alpha_true
        gp_load.kernel_ = _kern_true
        gp_load.X_train_ = _Xtrain_true
        gp_load._y_train_mean = [0] 

        iexps = np.random.choice(np.arange(nexp)[cut], 10, replace=False)
        for iexp in iexps: 
            theta_i = np.array([exps[k][iexp] for k in props]) 
            print('true GP = %f' % gp_true.predict(np.atleast_2d(theta_i))) 
            print('load GP = %f' % gp_load.predict(np.atleast_2d(theta_i)))
    return None 


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


def _GALeg_sourceSpec_fibermag_halpha(nsub): 
    ''' plot the Halpha probed by (z - W1) - 3/2.5 (g - r) + 1.2  to fiber mag
    for the source spectra


    :param nsub: 
        number of galaxies to randomly select from the GAMALegacy 
        joint catalog 

    :param spec_flag: (default: '') 
        string that specifies what type of spectra options are
        '',  '.lowHalpha', '.noEmline'

    :param validate: (default: False) 
        if True make some plots that validate the chosen spectra
    '''
    # read in source spectra data
    fspec = os.path.join(dir_dat, 'GALeg.g15.sourceSpec.%i.hdf5' % nsub)
    fmeta = h5py.File(fspec, 'r') 
    # r-band fiber magnitude 
    r_fibermag = fmeta['r_mag_apflux'][...]
    
    # halpha probe 
    g_mag = UT.flux2mag(fmeta['legacy-photo']['flux_g'][...], method='log')
    r_mag = UT.flux2mag(fmeta['legacy-photo']['flux_r'][...], method='log')
    z_mag = UT.flux2mag(fmeta['legacy-photo']['flux_z'][...], method='log')
    w1_mag = UT.flux2mag(fmeta['legacy-photo']['flux_w1'][...], method='log')
    
    _halpha = (z_mag - w1_mag) - 3./2.5 * (g_mag - r_mag) + 1.2 
    fmeta.close()

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.scatter(r_fibermag, _halpha, c='k', s=1) 
    sub.set_xlabel('$r$-band fiber magnitude', fontsize=25) 
    sub.set_xlim(18., 22.) 
    sub.set_ylabel('$(z - W1) - 3/2.5 (g - r) + 1.2$', fontsize=25) 
    sub.set_ylim(-1., 2.) 
    fig.savefig(fspec.replace('.hdf5', '.fibermag_halpha.png'), bbox_inches='tight') 
    return None 


def sample_surveysimExposures(expfile, seed=0): 
    '''sample BGS exposures of the surveysim output to cover a wide set of parameter 
    combinations. We only selet tiles where SNR2FRAC is achieved with a single exposure
    due to the fact that the observing conditions are updated when a tile is revsited.  
    '''
    np.random.seed(seed)
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
    n_exps = len(exps['moon_ill'])
    
    # cut out some of the BGS expsoures
    cut = (exps['texp'] > 100) # bug when twilight=True
    
    # sample along moon ill, alt, and sun alt 
    moonill_bins = [0.4, 0.75, 1.]
    moonalt_bins = [0.0, 40., 90.] 
    sun_alt_bins = [0.0, -20., -90.]
    
    texp_total, airmass = [], [] 
    moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = [], [], [], [], [] 
    seeing, transp = [], []
    for i0 in range(len(moonill_bins)-1): 
        for i1 in range(len(moonalt_bins)-1): 
            for i2 in range(len(sun_alt_bins)-1): 
                inbin = (
                        (exps['moon_ill'] > moonill_bins[i0]) & 
                        (exps['moon_ill'] < moonill_bins[i0+1]) & 
                        (exps['moon_alt'] > moonalt_bins[i1]) & 
                        (exps['moon_alt'] < moonalt_bins[i1+1]) & 
                        (exps['sun_alt'] < sun_alt_bins[i2]) & 
                        (exps['sun_alt'] > sun_alt_bins[i2+1]) 
                        )
                found_exposures = False
                while not found_exposures:
                    # randomly choose an exposure observed at this condition 
                    _i_samples = np.random.choice(np.arange(n_exps)[cut & inbin])
                    # find all the exposures of this tile  
                    same_tile = (exps['tileid'] == exps['tileid'][_i_samples]) 
                    if (np.sum(same_tile) == 1): 
                        # SNR2FRAC > 1 was achieved with a single exposure 
                        _texp_total = np.sum(exps['texp'][same_tile])
                        _airmass    = exps['airmass'][_i_samples]
                        _moon_ill   = exps['moon_ill'][_i_samples]
                        _moon_alt   = exps['moon_alt'][_i_samples]
                        _moon_sep   = exps['moon_sep'][_i_samples]
                        _sun_alt    = exps['sun_alt'][_i_samples]
                        _sun_sep    = exps['sun_sep'][_i_samples]
                        _seeing     = exps['seeing'][_i_samples]
                        _transp     = exps['transp'][_i_samples]

                        found_exposures=True 

                print('total t_exp=%.f' % _texp_total)
                print('airmass = %.1f' % _airmass)
                fairmass = ETC.airmass_exposure_factor(_airmass) 
                print('airmass factor = %.1f' % fairmass) 

                print('moon ill=%.2f, alt=%.f' % (_moon_ill, _moon_alt))
                print('sun alt=%.f' % _sun_alt)
                fbright = ETC.bright_exposure_factor(_moon_ill, _moon_alt, np.array(_moon_sep),
                        _sun_alt, _sun_sep, np.array(_airmass))
                print('bright factor = %.1f' % fbright) 

                print('seeing=%.1f, transp=%.1f' % (_seeing, _transp))
                _ETC = ETC.ExposureTimeCalculator() 
                fweather = _ETC.weather_factor(_seeing, _transp) 
                print('weather factor = %.1f' % fweather) 
                print('f_total = %.1f' % (fairmass * fbright / fweather))
                print('t_exp x f_total = %.1f' % (150. * fairmass * fbright / fweather))
                print('----------------------------') 

                texp_total.append(_texp_total) 
                airmass.append(_airmass) 
                moon_ill.append(_moon_ill)
                moon_alt.append(_moon_alt)
                moon_sep.append(_moon_sep)
                sun_alt.append(_sun_alt)
                sun_sep.append(_sun_sep)
                seeing.append(_seeing)
                transp.append(_transp)

    texp_total = np.array(texp_total)
    airmass    = np.array(airmass)
    moon_ill   = np.array(moon_ill)
    moon_alt   = np.array(moon_alt)
    moon_sep   = np.array(moon_sep)
    sun_alt    = np.array(sun_alt)
    sun_sep    = np.array(sun_sep)
    seeing     = np.array(seeing)
    transp     = np.array(transp)

    # compute sky brightness of the sampled exposures 
    Iskys = [] 
    for i in range(len(texp_total)): 
        wave, _Isky = Sky.sky_KSrescaled_twi(airmass[i], moon_ill[i], moon_alt[i], moon_sep[i], sun_alt[i], sun_sep[i])
        Iskys.append(_Isky)
    Iskys = np.array(Iskys)
    
    # save to file 
    _fsample = fexp.replace('.fits', '.sample.seed%i.hdf5' % seed)
    fsample = h5py.File(_fsample, 'w') 
    fsample.create_dataset('texp_total', data=texp_total)
    fsample.create_dataset('airmass', data=airmass)  
    fsample.create_dataset('moon_ill', data=moon_ill) 
    fsample.create_dataset('moon_alt', data=moon_alt) 
    fsample.create_dataset('moon_sep', data=moon_sep)
    fsample.create_dataset('sun_alt', data=sun_alt) 
    fsample.create_dataset('sun_sep', data=sun_sep)
    fsample.create_dataset('seeing', data=seeing)  
    fsample.create_dataset('transp', data=transp) 
    # save sky brightnesses
    fsample.create_dataset('wave', data=wave) 
    fsample.create_dataset('sky', data=Iskys) 
    fsample.close() 

    fig = plt.figure(figsize=(21,5))
    sub = fig.add_subplot(141)
    sub.scatter(exps['moon_alt'], exps['moon_ill'], c='k', s=1)
    scat = sub.scatter(moon_alt, moon_ill, c='C1', s=10)
    sub.set_xlabel('Moon Altitude', fontsize=20)
    sub.set_xlim([-90., 90.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(142)
    sub.scatter(exps['moon_sep'], exps['moon_ill'], c='k', s=1)
    scat = sub.scatter(moon_sep, moon_ill, c='C1', s=10) 
    sub.set_xlabel('Moon Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(143)
    sub.scatter(exps['airmass'], exps['moon_ill'], c='k', s=1)
    scat = sub.scatter(airmass, moon_ill, c='C1', s=10)  
    sub.set_xlabel('Airmass', fontsize=20)
    sub.set_xlim([1., 2.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(144)
    sub.scatter(exps['sun_sep'], exps['sun_alt'], c='k', s=1)
    scat = sub.scatter(sun_sep, sun_alt, c='C1', s=10)
    sub.set_xlabel('Sun Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylabel('Sun Altitude', fontsize=20)
    sub.set_ylim([-90., 0.])
    ffig = _fsample.replace('.hdf5', '.png')  
    fig.savefig(ffig, bbox_inches='tight')

    # plot some of the sky brightnesses
    fig = plt.figure(figsize=(15,5))
    bkgd = fig.add_subplot(111, frameon=False) 
    for isky in range(Iskys.shape[0]):
        sub = fig.add_subplot(111)
        sub.plot(wave, Iskys[isky,:])
    sub.set_xlim([3500., 9500.]) 
    sub.set_ylim([0., 20]) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel('wavelength [Angstrom]', fontsize=25) 
    bkgd.set_ylabel('sky brightness [$erg/s/cm^2/A/\mathrm{arcsec}^2$]', fontsize=25) 

    ffig = _fsample.replace('.hdf5', '.sky.png')  
    fig.savefig(ffig, bbox_inches='tight')
    return None 


def surveysim_BGS_texp(expfile): 
    ''' a closer examination of the exposure times that are coming otu of surveysim 
    '''
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
        
    u_tileid = np.unique(exps['tileid'])  
    
    texp_total = np.zeros(len(u_tileid))
    exp_factor = np.zeros(len(u_tileid)) 
    same_day = np.zeros(len(u_tileid)).astype(bool) 
    single_exp = np.zeros(len(u_tileid)).astype(bool) 
    for i, tileid in enumerate(u_tileid): 
        sametile = (exps['tileid'] == tileid) 
        
        # total exposure time 
        texp_total[i] = np.sum(exps['texp'][sametile]) 
        
        # all exposures were on the same night 
        if ((exps['mjd'][sametile].max() - exps['mjd'][sametile].min()) < 1.): 
            same_day[i] = True

        if np.sum(sametile) == 1: 
            single_exp[i] = True
        
        # observing conditions of first exposure (not reflect of all the exposures but oh well)
        iexp = np.arange(len(exps['texp']))[sametile][0]
        fdust = ETC.dust_exposure_factor(exps['ebv'][iexp]) 
        fairmass = ETC.airmass_exposure_factor(exps['airmass'][iexp]) 
        fbright = ETC.bright_exposure_factor(
                exps['moon_ill'][iexp], exps['moon_alt'][iexp], np.array(exps['moon_sep'][iexp]),
                exps['sun_alt'][iexp], np.array(exps['sun_sep'][iexp]), np.array(exps['airmass'][iexp]))
        _ETC = ETC.ExposureTimeCalculator() 
        fweather = _ETC.weather_factor(exps['seeing'][iexp], exps['transp'][iexp])
        exp_factor[i] = fdust * fairmass * fbright / fweather 
        if np.sum(sametile) == 1: 
            print(texp_total[i], 150.*exp_factor[i], exps['snr2frac'][sametile].max())

    # exposure time vs various properties 
    fig = plt.figure(figsize=(5,5)) 
    sub = fig.add_subplot(111) 
    sub.scatter(texp_total, 150. * exp_factor, s=1, c='C0') 
    sub.scatter(texp_total[same_day], 150. * exp_factor[same_day], s=1, c='C1') 
    sub.scatter(texp_total[single_exp], 150. * exp_factor[single_exp], s=1, c='C2') 
    sub.plot([0., 5000.], [0., 5000.], c='k', ls='--', zorder=10) 
    sub.set_xlabel(r'total $t_{\rm exp}$ (sec)', fontsize=20) 
    sub.set_xlim(-100., 2500) 
    sub.set_ylabel(r'(150 sec) $\times f_{\rm exp}$', fontsize=20) 
    sub.set_ylim(-100., 2500) 

    ffig = os.path.join(dir_dat, 'texp_test.BGS.%s' % expfile.replace('.fits', '.png'))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


# --- surveysim output --- 
def surveysim_BGS(expfile): 
    ''' read in surveysim output exposures and plot the exposure time distributions 
    for BGS exposures. Also check the BGS exposure time vs exposure properties.
    '''
    # master-branch surveysim output (for comparison) 
    fmaster = os.path.join(dir_dat, 'exposures_surveysim_master.fits')
    exps_master = extractBGS(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
        
    twilight = (exps['sun_alt'] >= -20.) 
    
    # compile total exposures 
    texp_tot_master = [] 
    u_tileid = np.unique(exps_master['tileid'])
    for tileid in u_tileid: 
        istile = (exps_master['tileid'] == tileid) 
        texp_tot_master.append(np.sum(exps_master['texp'][istile]))

    texp_tot = [] 
    u_tileid = np.unique(exps['tileid'])
    for tileid in u_tileid: 
        istile = (exps['tileid'] == tileid) 
        texp_tot.append(np.sum(exps['texp'][istile]))
    
    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(texp_tot_master, bins=50, density=True, range=(0, 2500), color='C0', label='master branch')
    sub.hist(texp_tot, bins=50, density=True, range=(0, 2500), alpha=0.75, color='C1', label=r'surveysim fork')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'total $t_{\rm exp}$ (sec) of tiles', fontsize=20) 
    sub.set_xlim(0.,2500) 
    fig.savefig(os.path.join(dir_dat, 'texp_total.BGS.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    
    # exposure time vs various properties 
    fig = plt.figure(figsize=(15,15)) 
    props   = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep', 'seeing', 'transp']
    lbls    = ['airmass', 'moon ill.', 'moon alt.', 'moon sep.', 'sun alt.', 'sun sep.', 'seeing', 'transp.']
    for i, k in enumerate(props):
        sub = fig.add_subplot(3,3,i+1) 
        sub.scatter(exps['texp'], exps[k], s=1, c='k') 
        # highlight twilight exposures 
        sub.scatter(exps['texp'][twilight], exps[k][twilight], s=2, c='C1', zorder=10) 
        sub.set_xlim(-100., 2500) 
        sub.set_ylabel(lbls[i], fontsize=20) 

    # exposure time vs exposure time correction factor
    from desisurvey import etc as ETC
    exp_factor = np.zeros(len(exps['texp'])) 
    for iexp in range(len(exps['texp'])): 
        exp_factor[iexp] = ETC.bright_exposure_factor(
                exps['moon_ill'][iexp], exps['moon_alt'][iexp], np.array(exps['moon_sep'][iexp]),
                exps['sun_alt'][iexp], np.array(exps['sun_sep'][iexp]), np.array(exps['airmass'][iexp]))
    sub = fig.add_subplot(3,3,i+2) 
    sub.scatter(exps['texp'], exp_factor, s=1, c='k') 
    sub.scatter(exps['texp'][twilight], exp_factor[twilight], s=2, c='C1', zorder=10, label='twilight') 
    sub.set_xlim(-100., 2500) 
    sub.set_ylabel('Bright Exposure Factor', fontsize=20) 
    sub.legend(loc='upper right', fontsize=15, handletextpad=0.2, markerscale=5) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm exp}$ (sec)', fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.3) 
    ffig = os.path.join(dir_dat, 'texp_condition.BGS.%s.png' % expfile.replace('.fits', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def exposure_factor_surveysim_BGS(expfile): 
    ''' calculate the exposure factor for all BGS exposures of the specified
    surveysim output. 
    '''
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
        
    # exposure time vs exposure time correction factor
    from desisurvey import etc as ETC
    exp_factor = np.zeros(len(exps['texp'])) 
    for iexp in range(len(exps['texp'])): 
        exp_factor[iexp] = ETC.bright_exposure_factor(
                exps['moon_ill'][iexp], exps['moon_alt'][iexp], np.array(exps['moon_sep'][iexp]),
                exps['sun_alt'][iexp], np.array(exps['sun_sep'][iexp]), np.array(exps['airmass'][iexp]))
    np.save(os.path.join(dir_dat, 'exposure_factor.BGS.%s' % expfile), exp_factor) 
    return None 


def surveysim_All(expfile): 
    ''' read in surveysim output exposures and plot the exposure time distributions 
    '''
    # master-branch surveysim output (for comparison) 
    fmaster = os.path.join(dir_dat, 'exposures_surveysim_master.fits')
    exps_master = extractAll(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractAll(fexp) # get BGS exposures only 
    
    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(exps_master['texp'], bins=50, density=True, range=(0, 2500), color='C0', label='master branch')
    sub.hist(exps['texp'], bins=50, density=True, range=(0, 2500), alpha=0.75, color='C1', label=r'surveysim fork')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'$t_{\rm exp}$ (sec) ', fontsize=20) 
    sub.set_xlim(0.,2500) 
    fig.savefig(os.path.join(dir_dat, 'texp.All.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    return None 


def surveysim_Weird(expfile):
    ''' examine the odd surveysim output exposures 
    '''
    # master-branch surveysim output (for comparison) 
    fmaster = os.path.join(dir_dat, 'exposures_surveysim_master.fits')
    exps_master = extractAll(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractAll(fexp) # get BGS exposures only 

    isweird = (exps['texp'] < 300.)
    
    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(exps_master['texp'], bins=50, density=True, range=(-200, 2500), color='C0', label='master branch')
    sub.hist(exps['texp'], bins=50, density=True, range=(-200, 2500), alpha=0.75, color='C1', label=r'surveysim fork')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'$t_{\rm exp}$ (sec) ', fontsize=20) 
    sub.set_xlim(-200.,500) 
    fig.savefig(os.path.join(dir_dat, 'texp.Weird.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    
    # exposure time vs various properties 
    fig = plt.figure(figsize=(15,15)) 
    props   = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep', 'seeing', 'transp']
    lbls    = ['airmass', 'moon ill.', 'moon alt.', 'moon sep.', 'sun alt.', 'sun sep.', 'seeing', 'transp.']
    for i, k in enumerate(props):
        sub = fig.add_subplot(3,3,i+1) 
        sub.scatter(exps['texp'][isweird], exps[k][isweird], s=1, c='k') 
        sub.set_xlim(-200., 300) 
        sub.set_ylabel(lbls[i], fontsize=20) 
        if k == 'sun_alt': 
            sub.plot([-100., 2500.], [-20., -20.], c='r', ls='--') 
            sub.set_ylim(-25., None) 

    # exposure time vs exposure time correction factor
    from desisurvey import etc as ETC
    exp_factor = np.zeros(len(exps['texp'])) 
    for iexp in np.arange(len(exps['texp']))[isweird]: 
        exp_factor[iexp] = ETC.bright_exposure_factor(
                exps['moon_ill'][iexp], exps['moon_alt'][iexp], np.array(exps['moon_sep'][iexp]),
                exps['sun_alt'][iexp], np.array(exps['sun_sep'][iexp]), np.array(exps['airmass'][iexp]))
    sub = fig.add_subplot(3,3,i+2) 
    sub.scatter(exps['texp'][isweird], exp_factor[isweird], s=1, c='k') 
    sub.set_xlim(-200., 300) 
    sub.set_ylabel('Bright Exposure Factor', fontsize=20) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm exp}$ (sec)', fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.3) 
    ffig = os.path.join(dir_dat, 'texp_condition.Weird.%s.png' % expfile.replace('.fits', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def extractBGS(fname): 
    """ extra data on bgs exposures from surveysim output 

    no cosmics split adds 20% to margin
    total BGS time: 2839 hours
    total BGS minus twilight: 2372
    assuming 7.5 deg^2 per field
    assumiong 68% open-dome fraction
    """
    fbgs = fname.replace('.fits', '.bgs.hdf5') 
    print(fbgs)
    
    if not os.path.isfile(fbgs): 
        ssout = fits.open(fname)[1].data # survey sim output 
        tiles = fits.open(os.path.join(UT.dat_dir(), 'bright_exposure', 'desi-tiles.fits'))[1].data # desi-tiles
        
        isbgs = (tiles['PROGRAM'] == 'BRIGHT') # only bgs 
        
        uniq_tiles, iuniq = np.unique(ssout['TILEID'], return_index=True)  
        print('%i unique tiles out of %i total exposures' % (len(uniq_tiles), len(ssout['TILEID'])))

        _, ssbgs, bgsss = np.intersect1d(ssout['TILEID'][iuniq], tiles['TILEID'][isbgs], return_indices=True)  
        print('%i total BGS fields: ' % len(ssbgs))
        print('approx. BGS coverage [#passes]: %f' % (float(len(ssbgs)) * 7.5 / 14000.)) 
        
        tileid, mjd, RAs, DECs = [], [], [], [] 
        airmasses, moon_ills, moon_alts, moon_seps, sun_alts, sun_seps, texps = [], [], [], [], [], [], []
        snr2frac, ebv, seeings, transps = [], [], [], [] 
        nexps = 0 
        for i in range(len(ssbgs)): 
            isexps  = (ssout['TILEID'] == ssout['TILEID'][iuniq][ssbgs[i]]) 
            nexp    = np.sum(isexps)
            ra      = tiles['RA'][isbgs][bgsss[i]]
            dec     = tiles['DEC'][isbgs][bgsss[i]]
            _ebv    = tiles['EBV_MED'][isbgs][bgsss[i]]
            _mjd    = ssout['MJD'][isexps]

            # get sky parameters for given ra, dec, and mjd
            moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = UT.get_thetaSky(np.repeat(ra, nexp), np.repeat(dec, nexp), _mjd)
             
            tileid.append(ssout['TILEID'][isexps]) 
            mjd.append(_mjd)
            texps.append(ssout['EXPTIME'][isexps]) 
            RAs.append(np.repeat(ra, nexp)) 
            DECs.append(np.repeat(dec, nexp)) 
            airmasses.append(ssout['AIRMASS'][isexps]) 
            moon_ills.append(moon_ill)
            moon_alts.append(moon_alt) 
            moon_seps.append(moon_sep)
            sun_alts.append(sun_alt) 
            sun_seps.append(sun_sep) 
            snr2frac.append(ssout['SNR2FRAC'][isexps])
            ebv.append(np.repeat(_ebv, nexp)) 
            
            # atmospheric seeing and transp
            seeings.append(ssout['SEEING'][isexps]) 
            transps.append(ssout['TRANSP'][isexps]) 

            nexps += nexp 

        exps = {
            'tileid':   np.concatenate(tileid), 
            'mjd':      np.concatenate(mjd),
            'texp':     np.concatenate(texps), 
            'ra':       np.concatenate(RAs),
            'dec':      np.concatenate(DECs),
            'airmass':  np.concatenate(airmasses),
            'moon_ill': np.concatenate(moon_ills), 
            'moon_alt': np.concatenate(moon_alts), 
            'moon_sep': np.concatenate(moon_seps), 
            'sun_alt':  np.concatenate(sun_alts), 
            'sun_sep':  np.concatenate(sun_seps),
            'seeing':   np.concatenate(seeings),
            'transp':   np.concatenate(transps),
            'snr2frac': np.concatenate(snr2frac), 
            'ebv':      np.concatenate(ebv)
        }
        _bgs = h5py.File(fbgs, 'w') 
        for k in exps.keys(): 
            _bgs.create_dataset(k, data=exps[k]) 
        _bgs.close() 
    else: 
        exps = {} 
        _bgs = h5py.File(fbgs, 'r') 
        for k in _bgs.keys(): 
            exps[k] = _bgs[k][...]
        _bgs.close() 
    return exps 


def extractAll(fname, notwilight=True): 
    """ extra data on all exposures from surveysim output 

    no cosmics split adds 20% to margin
    total BGS time: 2839 hours
    total BGS minus twilight: 2372
    assuming 7.5 deg^2 per field
    assumiong 68% open-dome fraction
    """
    total_hours = 2839
    if notwilight: total_hours = 2372

    open_hours = total_hours*0.68
    
    ssout = fits.open(fname)[1].data # survey sim output 
    tiles = fits.open(os.path.join(UT.dat_dir(), 'bright_exposure', 'desi-tiles.fits'))[1].data # desi-tiles
    
    uniq_tiles, iuniq = np.unique(ssout['TILEID'], return_index=True)  

    _, ssbgs, bgsss = np.intersect1d(ssout['TILEID'][iuniq], tiles['TILEID'], return_indices=True)  
    
    RAs, DECs = [], [] 
    airmasses, moon_ills, moon_alts, moon_seps, sun_alts, sun_seps, texps = [], [], [], [], [], [], []
    seeings, transps = [], [] 
    nexps = 0 
    for i in range(len(ssbgs)): 
        isexps  = (ssout['TILEID'] == ssout['TILEID'][iuniq][ssbgs[i]]) 
        nexp    = np.sum(isexps)
        ra      = tiles['RA'][bgsss[i]]
        dec     = tiles['DEC'][bgsss[i]]
        mjd     = ssout['MJD'][isexps]
        # get sky parameters for given ra, dec, and mjd
        moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = UT.get_thetaSky(np.repeat(ra, nexp), np.repeat(dec, nexp), mjd)

        texps.append(ssout['EXPTIME'][isexps]) 
        RAs.append(np.repeat(ra, nexp)) 
        DECs.append(np.repeat(dec, nexp)) 
        airmasses.append(ssout['AIRMASS'][isexps]) 
        moon_ills.append(moon_ill)
        moon_alts.append(moon_alt) 
        moon_seps.append(moon_sep)
        sun_alts.append(sun_alt) 
        sun_seps.append(sun_sep) 
        
        # atmospheric seeing and transp
        seeings.append(ssout['SEEING'][isexps]) 
        transps.append(ssout['TRANSP'][isexps]) 

        nexps += nexp 

    exps = {
        'texp':     np.concatenate(texps), 
        'ra':       np.concatenate(RAs),
        'dec':      np.concatenate(DECs),
        'airmass':  np.concatenate(airmasses),
        'moon_ill': np.concatenate(moon_ills), 
        'moon_alt': np.concatenate(moon_alts), 
        'moon_sep': np.concatenate(moon_seps), 
        'sun_alt':  np.concatenate(sun_alts), 
        'sun_sep':  np.concatenate(sun_seps),
        'seeing':   np.concatenate(seeings),
        'transp':   np.concatenate(transps)
    }
    return exps 


if __name__=="__main__": 
    #surveysim_BGS_texp('exposures_surveysim_fork_150sv0p5.fits')
    #surveysim_BGS('exposures_surveysim_fork_150sv0p5.fits') 
    #surveysim_All('exposures_surveysim_fork_150sv0p4.fits') 
    #surveysim_Weird('exposures_surveysim_fork_150sv0p4.fits') 

    extractBGS(os.path.join(dir_dat, 'exposures_surveysim_fork_150sv0p5.fits'))
    #sample_surveysimExposures('exposures_surveysim_fork_150sv0p5.fits', seed=0)
    #GALeg_bgsSpec(
    #        specfile='GALeg.g15.sourceSpec.3000.hdf5', 
    #        expfile='exposures_surveysim_fork_150sv0p5.fits', 
    #        flag='v0p5', 
    #        seed=0)
    #zsuccess_surveysimExposures(
    #        specfile='GALeg.g15.sourceSpec.3000.hdf5', 
    #        expfile='exposures_surveysim_fork_150sv0p4.fits', 
    #        seed=0, 
    #        min_deltachi2=40.)
    #zsuccess_fibermag_halpha(
    #        specfile='GALeg.g15.sourceSpec.3000.hdf5', 
    #        expfile='exposures_surveysim_fork_150sv0p4.fits', 
    #        seed=0, 
    #        min_deltachi2=60.)
    #exposure_factor_surveysim_BGS('exposures_surveysim_fork_150sv0p4.fits')
    #buildGP_bright_exposure_factor(expfile='exposures_surveysim_fork_150sv0p4.fits')
    #_test_loadGP() 
    #_GALeg_sourceSpec_fibermag_halpha(3000)
