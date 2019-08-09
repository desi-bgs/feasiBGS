#!/bin/python 
'''
scripts to validate surveysim outputs
'''
import os 
import h5py 
import numpy as np 
import scipy as sp 
import corner as DFM 
import desisurvey.etc as detc
# --- astropy --- 
import astropy.units as u
from astropy.io import fits
from astropy.table import Table as aTable
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


def GALeg_bgsSpec(specfile='GALeg.g15.sourceSpec.3000.hdf5', expfile=None, seed=0): 
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
    _fspec = os.path.join(dir_dat, specfile) 
    fspec = h5py.File(_fspec, 'r') 
    wave = fspec['wave'].value 
    flux = fspec['flux'].value 

    # read in sampled exposures
    _fsample = os.path.join(dir_dat, expfile.replace('.fits', '.sample.seed%i.fits' % seed)) 
    fexps       = h5py.File(_fsample, 'r')
    texp        = fexps['texp'][...]
    airmass     = fexps['airmass'][...]
    moon_ill    = fexps['moon_ill'][...]
    moon_alt    = fexps['moon_alt'][...]
    moon_sep    = fexps['moon_sep'][...]
    sun_alt     = fexps['sun_alt'][...]
    sun_sep     = fexps['sun_sep'][...]
    seeing      = fexps['seeing'][...]
    transp      = fexps['transp'][...]
    n_sample    = len(texp) 
    # sky brightness 
    wave_sky    = fexps['wave'][...]
    u_sb        = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second
    sky         = fexps['sky'][...]

    print('--- simulate exposures with sky model ---') 
    for iexp in range(n_sample): 
        print('t_exp=%f' % texp[iexp])
        print('airmass=%f' % airmass[iexp])
        print('moon ill=%.2f alt=%.f, sep=%.f' % (moon_ill[iexp], moon_alt[iexp], moon_sep[iexp]))
        print('sun alt=%.f, sep=%.f' % (sun_alt[iexp], sun_sep[iexp]))
        print('seeing=%.2f, transp=%.2f' % (seeing[iexp], transp[iexp]))

        # simulate the exposures 
        fdesi = FM.fakeDESIspec()
        f_bgs = os.path.join(dir_dat, 
                specfile.replace('sourceSpec', 'bgsSpec').replace('.hdf5', '.sample%i.seed%i.fits' % (iexp, seed)))

        bgs = fdesi.simExposure(wave, flux, 
                exptime=texp[iexp], 
                airmass=airmass[iexp],
                skycondition={'name': 'input', 'sky': np.clip(sky[iexp,:], 0, None) * u_sb, 'wave': wave_sky}, 
                filename=f_bgs)

        fig = plt.figure(figsize=(10,20))
        sub = fig.add_subplot(411) 
        sub.plot(wave_sky, sky[iexp], c='C1') 
        sub.text(0.05, 0.95, 
                'texp=%.f, airmass=%.2f\nmoon ill=%.2f, alt=%.f, sep=%.f\nsun alt=%.f, sep=%.f\nseeing=%.1f, transp=%.1f' % 
                (texp[iexp], airmass[iexp], moon_ill[iexp], moon_alt[iexp], moon_sep[iexp], 
                    sun_alt[iexp], sun_sep[iexp], seeing[iexp], transp[iexp]), 
                ha='left', va='top', transform=sub.transAxes, fontsize=15)
        sub.legend(loc='upper right', frameon=True, fontsize=20) 
        sub.set_xlim([3e3, 1e4]) 
        sub.set_ylim([0., 20.]) 
        for i in range(3): 
            sub = fig.add_subplot(4,1,i+2)
            for band in ['b', 'r', 'z']: 
                sub.plot(bgs.wave[band], bgs.flux[band][i], c='C1') 
            sub.plot(wave, flux[i], c='k', ls=':', lw=1, label='no noise')
            if i == 0: sub.legend(loc='upper right', fontsize=20)
            sub.set_xlim([3e3, 1e4]) 
            sub.set_ylim([0., 15.]) 
        bkgd = fig.add_subplot(111, frameon=False) 
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('rest-frame wavelength [Angstrom]', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', labelpad=10, fontsize=25) 
        fig.savefig(f_bgs.replace('.fits', '.png'), bbox_inches='tight') 
    return None 


def GALeg_sourceSpec(nsub, validate=False): 
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
    criterion = hasmatch 
    
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
    
    fspec = os.path.join(dir_dat, 'GALeg.g15.metadata.%i.hdf5' % nsub)
    fmeta = h5py.File(fspec, 'w') 
    fmeta.create_dataset('zred', data=redshift[subsamp])
    fmeta.create_dataset('absmag_ugriz', data=absmag_ugriz[:,subsamp]) 
    fmeta.create_dataset('r_mag_apflux', data=r_mag_apflux[subsamp]) 
    fmeta.create_dataset('r_mag_gama', data=r_mag_gama[subsamp]) 
    for grp in gleg.keys(): 
        group = fsub.create_group(grp) 
        for key in gleg[grp].keys(): 
            group.create_dataset(key, data=gleg[grp][key][subsamp])
    fmeta.close()
    
    fspec = os.path.join(dir_dat, 'GALeg.g15.sourceSpec.%i.hdf5' % nsub)
    fsub = h5py.File(fspec, 'w') 
    fsub.create_dataset('zred', data=redshift[subsamp])
    fsub.create_dataset('absmag_ugriz', data=absmag_ugriz[:,subsamp]) 
    fsub.create_dataset('r_mag_apflux', data=r_mag_apflux[subsamp]) 
    fsub.create_dataset('r_mag_gama', data=r_mag_gama[subsamp]) 
    for grp in gleg.keys(): 
        group = fsub.create_group(grp) 
        for key in gleg[grp].keys(): 
            group.create_dataset(key, data=gleg[grp][key][subsamp])
    fsub.create_dataset('flux', data=flux[isubsamp, :])
    fsub.create_dataset('wave', data=wave)
    fsub.close()

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


def sample_surveysimExposures(expfile, seed=0): 
    '''sample BGS exposures of the surveysim output to cover a wide set of parameter combinations
    '''
    np.random.seed(seed)
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
    n_exps = len(exps['moon_ill'])
    
    # cut out some of the BGS expsoures
    cut = (exps['texp'] > 100) # bug when twilight=True
    
    # sample along moon ill, alt, and sun alt 
    moonill_bins = [0.4, 0.7, 1.]
    moonalt_bins = [0.0, 40., 90.] 
    sun_alt_bins = [0.0, -20., -90.]
    
    i_samples = [] 
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
                _i_samples = np.random.choice(np.arange(n_exps)[cut & inbin])
                i_samples.append(_i_samples) 
                print('moon ill=%.2f, alt=%.f' % (exps['moon_ill'][_i_samples], exps['moon_alt'][_i_samples]))
                print('sun alt=%.f' % exps['sun_alt'][_i_samples])
    i_samples = np.array(i_samples) 

    # compute sky brightness of the sampled exposures 
    Iskys = [] 
    for i in i_samples: 
        wave, _Isky = Sky.sky_KSrescaled_twi(
                exps['airmass'][i], 
                exps['moon_ill'][i], 
                exps['moon_alt'][i], 
                exps['moon_sep'][i], 
                exps['sun_alt'][i], 
                exps['sun_sep'][i])
        Iskys.append(_Isky)
    Iskys = np.array(Iskys)
    
    # save to file 
    _fsample = fexp.replace('.fits', '.sample.seed%i.fits' % seed)
    fsample = h5py.File(_fsample, 'w') 
    for k in ['ra', 'dec', 'airmass', 'moon_ill', 'moon_alt', 'moon_sep', 
            'sun_alt', 'sun_sep', 'texp', 'seeing', 'transp']:  
        fsample.create_dataset(k.lower(), data=exps[k][i_samples]) 
    # save sky brightnesses
    fsample.create_dataset('wave', data=wave) 
    fsample.create_dataset('sky', data=Iskys) 
    fsample.close() 

    fig = plt.figure(figsize=(21,5))
    sub = fig.add_subplot(141)
    sub.scatter(exps['moon_alt'], exps['moon_ill'], c='k', s=1)
    scat = sub.scatter(exps['moon_alt'][i_samples], exps['moon_ill'][i_samples], c='C1', s=10)
    sub.set_xlabel('Moon Altitude', fontsize=20)
    sub.set_xlim([-90., 90.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(142)
    sub.scatter(exps['moon_sep'], exps['moon_ill'], c='k', s=1)
    scat = sub.scatter(exps['moon_sep'][i_samples], exps['moon_ill'][i_samples], c='C1', s=10) 
    sub.set_xlabel('Moon Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(143)
    sub.scatter(exps['airmass'], exps['moon_ill'], c='k', s=1)
    scat = sub.scatter(exps['airmass'][i_samples], exps['moon_ill'][i_samples], c='C1', s=10)  
    sub.set_xlabel('Airmass', fontsize=20)
    sub.set_xlim([1., 2.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(144)
    sub.scatter(exps['sun_sep'], exps['sun_alt'], c='k', s=1)
    scat = sub.scatter(exps['sun_sep'][i_samples], exps['sun_alt'][i_samples], c='C1', s=10)
    sub.set_xlabel('Sun Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylabel('Sun Altitude', fontsize=20)
    sub.set_ylim([-90., 0.])
    ffig = _fsample.replace('.fits', '.png')  
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

    ffig = _fsample.replace('.fits', '.sky.png')  
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

    print('master: %.f < texp < %.f' % (exps_master['texp'].min(), exps_master['texp'].max()))
    print('forked: %.f < texp < %.f' % (exps['texp'].min(), exps['texp'].max()))
    
    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(exps_master['texp'], bins=50, density=True, range=(0, 2500), color='C0', label='master branch')
    sub.hist(exps['texp'], bins=50, density=True, range=(0, 2500), alpha=0.75, color='C1', label=r'surveysim fork')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'$t_{\rm exp}$ (sec) ', fontsize=20) 
    sub.set_xlim(0.,2500) 
    fig.savefig(os.path.join(dir_dat, 'texp.BGS.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    
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


def extractBGS(fname, notwilight=True): 
    """ extra data on bgs exposures from surveysim output 

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
    
    isbgs = (tiles['PROGRAM'] == 'BRIGHT') # only bgs 
    
    uniq_tiles, iuniq = np.unique(ssout['TILEID'], return_index=True)  
    print('%i unique tiles out of %i total exposures' % (len(uniq_tiles), len(ssout['TILEID'])))

    _, ssbgs, bgsss = np.intersect1d(ssout['TILEID'][iuniq], tiles['TILEID'][isbgs], return_indices=True)  
    print('%i total BGS fields: ' % len(ssbgs))
    print('approx. BGS coverage [#passes]: %f' % (float(len(ssbgs)) * 7.5 / 14000.)) 
    
    RAs, DECs = [], [] 
    airmasses, moon_ills, moon_alts, moon_seps, sun_alts, sun_seps, texps = [], [], [], [], [], [], []
    seeings, transps = [], [] 
    nexps = 0 
    for i in range(len(ssbgs)): 
        isexps  = (ssout['TILEID'] == ssout['TILEID'][iuniq][ssbgs[i]]) 
        nexp    = np.sum(isexps)
        ra      = tiles['RA'][isbgs][bgsss[i]]
        dec     = tiles['DEC'][isbgs][bgsss[i]]
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
    #sample_surveysimExposures('exposures_surveysim_fork_150sv0p4.fits', seed=0)
    GALeg_bgsSpec(
            specfile='GALeg.g15.sourceSpec.3000.hdf5', 
            expfile='exposures_surveysim_fork_150sv0p4.fits', 
            seed=0)
    #surveysim_BGS('exposures_surveysim_fork_150sv0p4.fits') 
    #surveysim_All('exposures_surveysim_fork_150sv0p4.fits') 
    #surveysim_Weird('exposures_surveysim_fork_150sv0p4.fits') 
