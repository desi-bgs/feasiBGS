'''
'''
import os 
import h5py 
import pickle 
import numpy as np 
import scipy as sp 
# -- astropy -- 
from astropy import units as u
# --- sklearn ---
from sklearn.gaussian_process import GaussianProcessRegressor as GPR 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
# -- desi --
import desisim.simexp 
import specsim.config 
from desispec.io import read_spectra
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


def texp_factor(validate=False, silent=True): 
    ''' Calculate the exposure time correction factor using the `surveysim` 
    exposure list from Jeremy: `bgs_survey_exposures.withsun.hdf5', which 
    supplemented the observing conditions with sun observing conditions. 
    Outputs a file that contains the observing conditions (parameters) and 
    the ratio between (new sky flux)/(nominal dark time sky flux) 


    :param validate: (default: False)
        If True, generate some figures to validate things 

    :param silent: (default: True) 
        if False, code will print statements to indicate progress 
    '''
    # read surveysim BGS exposures 
    fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_survey_exposures.withsun.hdf5']), 'r')
    bgs_exps = {}
    for k in fexps.keys():
        bgs_exps[k] = fexps[k].value
    n_exps = len(bgs_exps['RA']) # number of exposures
        
    # read in pre-computed old and new sky brightness (this takes a bit) 
    if not silent: print('reading in sky brightness') 
    fnew = ''.join([UT.dat_dir(), 'newKSsky_twi_brightness.bgs_survey_exposures.withsun.p'])
    wave, sky_bright = pickle.load(open(fnew, 'rb'))

    # nominal dark sky brightness 
    config = desisim.simexp._specsim_config_for_wave(wave, dwave_out=None, specsim_config_file='desi')
    atm_config = config.atmosphere
    surface_brightness_dict = config.load_table(atm_config.sky, 'surface_brightness', as_dict=True)
    sky_dark= surface_brightness_dict['dark'] 

    # calculate (new sky brightness)/(nominal dark sky brightness), which is the correction
    # factor for the exposure time. 
    wlim = ((wave > 4000.) & (wave < 5000.)) # ratio over 4000 - 5000 A  
    f_exp = np.zeros(n_exps)
    for i_exp in range(n_exps): 
        f_exp[i_exp] = np.median((sky_bright[i_exp] / sky_dark.value)[wlim])
    print np.median(f_exp) 

    # write exposure subsets out to file 
    ff = os.path.join(UT.dat_dir(), 'bright_exposure', 'texp_factor_exposures.hdf5')
    fpick = h5py.File(ff, 'w')
    for k in ['AIRMASS', 'MOONFRAC', 'MOONALT', 'MOONSEP', 'SUNALT', 'SUNSEP', 'EXPTIME']: # write observing conditions  
        fpick.create_dataset(k.lower(), data=bgs_exps[k]) 
    # save sky brightnesses
    fpick.create_dataset('f_exp', data=f_exp) 
    fpick.close() 

    if validate: 
        fig = plt.figure(figsize=(15,25))
        sub = fig.add_subplot(611)
        sub.scatter(bgs_exps['MOONALT'], f_exp, c='k', s=1)
        sub.set_xlabel('Moon Altitude', fontsize=20)
        sub.set_xlim([-90., 90.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])

        sub = fig.add_subplot(612)
        sub.scatter(bgs_exps['MOONFRAC'], f_exp, c='k', s=1)
        sub.set_xlabel('Moon Illumination', fontsize=20)
        sub.set_xlim([0.5, 1.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])

        sub = fig.add_subplot(613)
        sub.scatter(bgs_exps['MOONSEP'], f_exp, c='k', s=1)
        sub.set_xlabel('Moon Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])
        
        sub = fig.add_subplot(614)
        sub.scatter(bgs_exps['SUNALT'], f_exp, c='k', s=1)
        sub.set_xlabel('Sun Altitude', fontsize=20)
        sub.set_xlim([-90., 0.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])
        
        sub = fig.add_subplot(615)
        sub.scatter(bgs_exps['SUNSEP'], f_exp, c='k', s=1)
        sub.set_xlabel('Sun Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])

        sub = fig.add_subplot(616)
        sub.scatter(bgs_exps['AIRMASS'], f_exp, c='k', s=1)
        sub.set_xlabel('Airmass', fontsize=20)
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_xlim([1., 2.])
        sub.set_ylim([0.5, 15.])
        fig.savefig(ff.replace('.hdf5', '.png'), bbox_inches='tight')

        # plot some of the sky brightnesses
        fig = plt.figure(figsize=(15,20))
        bkgd = fig.add_subplot(111, frameon=False) 
        for ii, isky in enumerate(np.random.choice(range(n_exps), 4, replace=False)):
            sub = fig.add_subplot(4,1,ii+1)
            sub.plot(wave, sky_bright[isky,:], c='C1', label='bright sky')
            sub.plot(wave, sky_dark, c='k', label='nomnial dark sky')
            sub.set_xlim([3500., 9500.]) 
            sub.set_ylim([0., 20]) 
            if ii == 0: sub.legend(loc='upper left', fontsize=20) 
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('wavelength [Angstrom]', fontsize=25) 
        bkgd.set_ylabel('sky brightness [$erg/s/cm^2/A/\mathrm{arcsec}^2$]', fontsize=25) 
        fig.savefig(ff.replace('.hdf5', '.sky.png'), bbox_inches='tight')
    return None 


def texp_factor_GP(validate=False): 
    ''' fit GP for exposure time correction factor as a function of 6 parameters
    'AIRMASS', 'MOONFRAC', 'MOONALT', 'MOONSEP', 'SUNALT', 'SUNSEP'
    '''
    ff = os.path.join(UT.dat_dir(), 'bright_exposure', 'texp_factor_exposures.hdf5')
    ffexp = h5py.File(ff, 'r')
    
    f_exp  = ffexp['f_exp'].value 
    bgs_exps = {} 
    thetas = np.zeros((len(f_exp), 6))
    for i_k, k in enumerate(['AIRMASS', 'MOONFRAC', 'MOONALT', 'MOONSEP', 'SUNALT', 'SUNSEP']): 
        bgs_exps[k] = ffexp[k.lower()].value 
        thetas[:,i_k] = ffexp[k.lower()].value 

    kern = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4)) # kernel
    gp = GPR(kernel=kern, n_restarts_optimizer=10) # instanciate a GP model
    gp.fit(thetas[::2], f_exp[::2])
    print gp.get_params(deep=True) 

    _thetas = np.zeros((5000, 6))
    for i in range(6): 
        _thetas[:,i] = np.random.uniform(thetas[:,i].min(), thetas[:,i].max(), 5000)
    print('convex hull') 
    param_hull = sp.spatial.Delaunay(thetas[::10,:])
    inhull = (param_hull.find_simplex(_thetas) >= 0) 
    thetas_test = _thetas[inhull]
    print('GP predicting')  
    mu_theta_test = gp.predict(thetas_test)
    
    if validate: 
        fig = plt.figure(figsize=(15,25))
        sub = fig.add_subplot(611)
        sub.scatter(bgs_exps['MOONALT'], f_exp, c='k', s=1)
        sub.scatter(thetas_test[:,2], mu_theta_test, c='C1', s=5)
        sub.set_xlabel('Moon Altitude', fontsize=20)
        sub.set_xlim([-90., 90.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])

        sub = fig.add_subplot(612)
        sub.scatter(bgs_exps['MOONFRAC'], f_exp, c='k', s=1)
        sub.scatter(thetas_test[:,1], mu_theta_test, c='C1', s=5)
        sub.set_xlabel('Moon Illumination', fontsize=20)
        sub.set_xlim([0.5, 1.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])

        sub = fig.add_subplot(613)
        sub.scatter(bgs_exps['MOONSEP'], f_exp, c='k', s=1)
        sub.scatter(thetas_test[:,3], mu_theta_test, c='C1', s=5)
        sub.set_xlabel('Moon Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])
        
        sub = fig.add_subplot(614)
        sub.scatter(bgs_exps['SUNALT'], f_exp, c='k', s=1)
        sub.scatter(thetas_test[:,4], mu_theta_test, c='C1', s=5)
        sub.set_xlabel('Sun Altitude', fontsize=20)
        sub.set_xlim([-90., 0.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])
        
        sub = fig.add_subplot(615)
        sub.scatter(bgs_exps['SUNSEP'], f_exp, c='k', s=1)
        sub.scatter(thetas_test[:,5], mu_theta_test, c='C1', s=5)
        sub.set_xlabel('Sun Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_ylim([0.5, 15.])

        sub = fig.add_subplot(616)
        sub.scatter(bgs_exps['AIRMASS'], f_exp, c='k', s=1)
        sub.scatter(thetas_test[:,0], mu_theta_test, c='C1', s=5)
        sub.set_xlabel('Airmass', fontsize=20)
        sub.set_ylabel('exposure time factor', fontsize=20)
        sub.set_xlim([1., 2.])
        sub.set_ylim([0.5, 15.])
        fig.savefig(ff.replace('.hdf5', '.GP.png'), bbox_inches='tight')
    return None 


if __name__=="__main__": 
    #texp_factor(validate=True)
    texp_factor_GP(validate=True)
