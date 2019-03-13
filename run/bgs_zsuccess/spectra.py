'''
'''
import os 
import h5py 
import numpy as np 
# -- astropy -- 
from astropy import units as u
# -- desi --
from desispec.io import read_spectra
# -- feasibgs -- 
from feasibgs import util as UT
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


def gleg_simSpec(nsub, spec_flag='', validate=False): 
    '''generate noiseless simulated spectra for a subset of GAMAlegacy 
    galaxies. The output hdf5 file will also contain all the galaxy
    properties 

    :param nsub: 
        number of galaxies to randomly select from the GAMALegacy 
        joint catalog 

    :param spec_flag: (default: '') 
        string that specifies what type of spectra options are
        '',  '.lowHalpha', '.noEmline'

    :param validate: (default: False) 
        if True make some plots that validate the chosen spectra
    '''
    # read in GAMA-Legacy catalog
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7, silent=False) # these values shouldn't change 

    redshift = gleg['gama-spec']['z']
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    r_mag_gama = gleg['gama-photo']['r_model'] # r-band magnitude from GAMA (SDSS) photometry

    ha_gama = gleg['gama-spec']['ha_flux'] # halpha line flux

    ngal = len(redshift) # number of galaxies
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg)
    hasmatch = (match != -999)
    
    if spec_flag == '': 
        criterion = hasmatch 
    elif spec_flag == '.lowHalpha':
        low_halpha = (ha_gama < 10.) 
        criterion = hasmatch & low_halpha 
    else: 
        raise ValueError
    
    # randomly pick a few more than nsub galaxies from the catalog
    subsamp = np.random.choice(np.arange(ngal)[criterion], int(1.1 * nsub), replace=False) 

    # generate noiseless spectra for these galaxies 
    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=15000) 
    emline_flux = s_bgs.EmissionLineFlux(gleg, index=subsamp, dr_gama=3, silent=True) # emission lines from GAMA 
    flux, wave, _, magnorm_flag = s_bgs.Spectra(r_mag_apflux[subsamp], redshift[subsamp],
                                                    vdisp[subsamp], seed=1, templateid=match[subsamp],
                                                    emflux=emline_flux, mag_em=r_mag_gama[subsamp], 
                                                    silent=True)
    # some of the galaxies will have issues where the emission line is brighter  
    # than the photometric magnitude. Lets make sure we take nsub galaxies that 
    # do not include these. 
    isubsamp = np.random.choice(np.arange(len(subsamp))[magnorm_flag], nsub, replace=False) 
    subsamp = subsamp[isubsamp]
    
    fspec = os.path.join(UT.dat_dir(), 'bgs_zsuccess', 'g15.simSpectra.%i%s.v2.hdf5' % (nsub, spec_flag))
    fsub = h5py.File(fspec, 'w') 
    fsub.create_dataset('zred', data=redshift[subsamp])
    fsub.create_dataset('absmag_ugriz', data=absmag_ugriz[:,subsamp]) 
    fsub.create_dataset('r_mag_apflux', data=r_mag_apflux[subsamp]) 
    fsub.create_dataset('r_mag_gama', data=r_mag_gama[subsamp]) 
    fsub.create_dataset('flux', data=flux[isubsamp, :])
    fsub.create_dataset('wave', data=wave)
    for grp in gleg.keys(): 
        group = fsub.create_group(grp) 
        for key in gleg[grp].keys(): 
            group.create_dataset(key, data=gleg[grp][key][subsamp])
    fsub.close()

    if validate: 
        fig = plt.figure(figsize=(10,8))
        sub = fig.add_subplot(111)
        for i in range(10): #np.random.choice(isubsamp, 10, replace=False): 
            wave_rest = wave / (1.+redshift[subsamp][i])
            sub.plot(wave_rest, flux[isubsamp[i],:]) 
        emline_keys = ['oiib', 'oiir', 'hb',  'oiiib', 'oiiir', 'ha', 'siib', 'siir']
        emline_lambda = [3727.092, 3729.874, 4862.683, 4960.295, 5008.239, 6564.613, 6718.294, 6732.673]
        for k, l in zip(emline_keys, emline_lambda): 
            if k == 'ha': 
                sub.vlines(l, 0., 20, color='k', linestyle='--', linewidth=1)
            else: 
                sub.vlines(l, 0., 20, color='k', linestyle=':', linewidth=0.5)
        sub.set_xlabel('rest-frame wavelength [Angstrom]', fontsize=25) 
        sub.set_xlim([3e3, 1e4]) 
        sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=25) 
        sub.set_ylim([0., 20.]) 
        fig.savefig(fspec.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def gleg_simSpec_mockexp(nsub, iexp, nexp=15, method='spacefill', spec_flag='', silent=True, validate=False): 
    ''' Given noiseless spectra, simulate DESI BGS noise based on observing
    conditions provided by iexp of nexp sampled observing conditions 

    :param nsub: 
        number of no noise spectra 

    :param iexp: 
        index of nexp observing conditions sampled using `method`

    :param nexp: (default: 15) 
        number of observing conditions sampled from `surveysim` 

    :param method: (default: 'spacefill') 
        method used for sampling `nexp` observing conditions 

    :param spec_flag: (default: '') 
        string that specifies what type of spectra options are
        '',  '.lowHalpha', '.noEmline'

    :param silent: (default: True)

    :param validate: (default: False) 
        if True generate some plots 
    '''
    # read in no noise spectra
    _fspec = os.path.join(UT.dat_dir(), 'bgs_zsuccess', 'g15.simSpectra.%i%s.v2.hdf5' % (nsub, spec_flag))
    fspec = h5py.File(_fspec, 'r') 
    wave = fspec['wave'].value 
    flux = fspec['flux'].value 

    # read in sampled exposures
    fexps = h5py.File(os.path.join(UT.dat_dir(), 'bgs_zsuccess', 
        'bgs_survey_exposures.subset.%i%s.hdf5' % (nexp, method)), 'r')
    texp = fexps['exptime'].value
    airmass = fexps['airmass'].value 
    wave_old = fexps['wave_old'].value
    wave_new = fexps['wave_new'].value
    u_sb = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second
    sky_old = fexps['sky_old'].value
    sky_new = fexps['sky_new'].value
    if not silent: print('t_exp=%f, airmass=%f' % (texp[iexp], airmass[iexp]))

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()
    if not silent: print('simulate exposures with old sky model') 
    f_bgs_old = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
        'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method,  
        '.old_sky.v2.fits'])
    bgs_old = fdesi.simExposure(wave, flux, 
            exptime=texp[iexp], 
            airmass=airmass[iexp],
            skycondition={'name': 'input', 'sky': np.clip(sky_old[iexp,:], 0, None) * u_sb, 'wave': wave_old}, 
            filename=f_bgs_old)

    if not silent: print('simulate exposures with new sky model') 
    f_bgs_new = ''.join([UT.dat_dir(), 'bgs_zsuccess/',
        'g15.simSpectra.', str(nsub), spec_flag, '.texp_default.iexp', str(iexp), 'of', str(nexp), method, 
        '.new_sky.v2.fits'])
    bgs_new = fdesi.simExposure(wave, flux, 
            exptime=texp[iexp], 
            airmass=airmass[iexp],
            skycondition={'name': 'input', 'sky': np.clip(sky_new[iexp,:], 0, None) * u_sb, 'wave': wave_new}, 
            filename=f_bgs_new)

    if validate: 
        fig = plt.figure(figsize=(10,20))
        sub = fig.add_subplot(411) 
        sub.plot(wave_new, sky_new[iexp], c='C1', label='new sky brightness') 
        sub.plot(wave_old, sky_old[iexp], c='C0', label='old sky brightness') 
        sub.text(0.05, 0.95, 'texp=%.0f, airmass=%.2f\nmoon ill=%.2f, alt=%.0f, sep=%.0f\nsun alt=%.0f, sep=%.f' % 
                (texp[iexp], airmass[iexp], fexps['moonfrac'][iexp], fexps['moonalt'][iexp], fexps['moonsep'][iexp], 
                    fexps['sunalt'][iexp], fexps['sunsep'][iexp]), 
                ha='left', va='top', transform=sub.transAxes, fontsize=15)
        sub.legend(loc='upper right', frameon=True, fontsize=20) 
        sub.set_xlim([3e3, 1e4]) 
        sub.set_ylim([0., 20.]) 
        for i in range(3): 
            sub = fig.add_subplot(4,1,i+2)
            for band in ['b', 'r', 'z']: 
                lbl = None  
                if band == 'b': lbl = 'new sky'
                sub.plot(bgs_new.wave[band], bgs_new.flux[band][i], c='C1', label=lbl) 
            for band in ['b', 'r', 'z']: 
                lbl = None  
                if band == 'b': lbl = 'old sky'
                sub.plot(bgs_old.wave[band], bgs_old.flux[band][i], c='C0', label=lbl) 
            sub.plot(wave, flux[i], c='k', ls=':', lw=1, label='no noise')
            if i == 0: sub.legend(loc='upper right', fontsize=20)
            sub.set_xlim([3e3, 1e4]) 
            sub.set_ylim([0., 15.]) 
        bkgd = fig.add_subplot(111, frameon=False) 
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('rest-frame wavelength [Angstrom]', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', labelpad=10, fontsize=25) 
        fig.savefig(os.path.join(UT.dat_dir(), 'bgs_zsuccess',
            'g15.simSpectra.%i%s.texp_default.iexp%iof%i%s.v2.png' % (nsub, spec_flag, iexp, nexp, method)),
            bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #gleg_simSpec(3000, validate=True)
    for iexp in [0]: #range(10,16): 
        gleg_simSpec_mockexp(3000, iexp, nexp=15, method='spacefill', validate=True)
    #gleg_simSpec(3000, spec_flag='.lowHalpha', validate=True)
    #gleg_simSpec_mockexp(3000, 0, spec_flag='.lowHalpha', nexp=15, method='spacefill', validate=True)
