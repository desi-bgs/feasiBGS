'''
script for generating spectra (noiseless, BGS-like, etc) for SV prep work 
'''
import os 
import h5py 
import numpy as np 
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


def GALeg_sourceSpec(nsub, flag=None): 
    '''generate noiseless simulated spectra for a subset of GAMAlegacy 
    galaxies. The output hdf5 file will also contain all the galaxy
    properties 

    :param nsub: 
        number of galaxies to randomly select from the GAMALegacy 
        joint catalog 

    :param flag: (optional) 
        flag specifies additional sample selection criteria. (default: None) 
    '''
    # read in GAMA-Legacy catalog with galaxies in both GAMA and Legacy surveys
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7, silent=True)  
    
    # extract meta-data of galaxies 
    redshift        = gleg['gama-spec']['z']
    absmag_ugriz    = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux    = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1]) # aperture flux
    r_mag_gama      = gleg['gama-photo']['r_model'] # r-band magnitude from GAMA (SDSS) photometry
    ha_gama         = gleg['gama-spec']['ha_flux'] # halpha line flux

    ngal = len(redshift) # number of galaxies
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg)
    hasmatch = (match != -999)
    criterion = hasmatch 
    
    # randomly pick a few more than nsub galaxies from the catalog that have 
    # matching templates because some of the galaxies will have issues where the 
    # emission line is brighter than the photometric magnitude.  
    subsamp = np.random.choice(np.arange(ngal)[criterion], int(1.1 * nsub), replace=False) 

    # generate noiseless spectra for these galaxies 
    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=15000) 
    # emission line fluxes from GAMA data  
    emline_flux = s_bgs.EmissionLineFlux(gleg, index=subsamp, dr_gama=3, silent=True) # emission lines from GAMA 

    flux, wave, _, magnorm_flag = s_bgs.Spectra(
            r_mag_apflux[subsamp], 
            redshift[subsamp],
            vdisp[subsamp], 
            seed=1, 
            templateid=match[subsamp], 
            emflux=emline_flux, 
            mag_em=r_mag_gama[subsamp], 
            silent=True)
    # only keep nsub galaxies
    isubsamp = np.random.choice(np.arange(len(subsamp))[magnorm_flag], nsub, replace=False) 
    subsamp = subsamp[isubsamp]
    
    # save to file  
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


def GALeg_noisySpec(specfile, exptime, airmass, Isky, filename=None): 
    ''' Given noiseless spectra, simulate noisy exposure with Isky 
    sky brightness, exptime sec exposure time, and airmass. Wrapper for 
    FM.fakeDESIspec().simExposure  
    
    :param specfile: 
        file name of noiseless source spectra to run through BGS exposure simulation. 

    '''
    # read in noiseless source spectra
    fspec = h5py.File(specfile, 'r') 
    wave = fspec['wave'][...]
    flux = fspec['flux'][...] 

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()
    noisy = fdesi.simExposure(wave, flux, exptime=exptime, airmass=airmass, Isky=Isky, filename=filename) 
    return noisy


def GALeg_noisySpec_surveysim(specfile, expfile): 
    ''' 
    ***FIX THIS UP***
    ***FIX THIS UP***
    ***FIX THIS UP***
    ***FIX THIS UP***
    Given noiseless spectra, simulate DESI BGS noise based on observing
    conditions provided by iexp of nexp sampled observing conditions 
    '''
    # read in no noise spectra
    fspec = h5py.File(specfile, 'r') 
    wave = fspec['wave'][...]
    flux = fspec['flux'][...] 

    # read in sampled exposures
    _fsample = os.path.join(dir_dat, expfile) 
    fexps       = h5py.File(_fsample, 'r')
    texp        = fexps['texp_total'][...]
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
        print('t_exp=%.f' % texp[iexp])
        print('airmass=%.2f' % airmass[iexp])
        print('moon ill=%.2f alt=%.f, sep=%.f' % (moon_ill[iexp], moon_alt[iexp], moon_sep[iexp]))
        print('sun alt=%.f, sep=%.f' % (sun_alt[iexp], sun_sep[iexp]))
        print('seeing=%.2f, transp=%.2f' % (seeing[iexp], transp[iexp]))

        # simulate the exposures 
        fdesi = FM.fakeDESIspec()
        f_bgs = os.path.join(dir_dat, 
                specfile.replace('sourceSpec', 'bgsSpec').replace('.hdf5', '%s.sample%i.seed%i.fits' % (str_flag, iexp, seed)))
        print(wave.shape)  
        print(flux.shape)  
        print(sky[iexp,:].shape) 
        print(wave_sky.shape) 
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


def GALeg_noisySpec_TSreview(specfile): 
    ''' Generate noisy spectra for the TS review document. Given noiseless spectra, 
    simulate noisy exposure for 3 different observing conditions: 
    1) not twilight, low moon ill.
    2) not twilight, high moon ill.
    3) yes twilight, medium moon ill. 
    for multiple exposure times. 
    '''
    # read in sampled exposures from `surveysim` output.
    # this is hardcoded to an output from a recent `surveysim` 
    # branch. since we're only interested in getting some realistic
    # bgs-esque observing conditions, this not terribly important. 
    _fsample = os.path.join(dir_dat, 'exposures_surveysim_fork_150sv0p5.bgs.hdf5') 
    fexps       = h5py.File(_fsample, 'r')
    texp        = fexps['texp'][...]
    airmass     = fexps['airmass'][...]
    moon_ill    = fexps['moon_ill'][...]
    moon_alt    = fexps['moon_alt'][...]
    moon_sep    = fexps['moon_sep'][...]
    sun_alt     = fexps['sun_alt'][...]
    sun_sep     = fexps['sun_sep'][...]
    n_sample    = len(texp) 

    # some cuts in determining the 3 different observing conditions 
    nottwilight = (sun_alt < -20.) 
    yestwilight = ~nottwilight 
    airmass_cut = (airmass < 1.5) # not too high airmass 
    bright_moon = (moon_ill > 0.8) 
    faint_moon  = (moon_ill < 0.7) 
    medium_moon = (moon_ill > 0.7) & (moon_ill < 0.8) 

    np.random.seed(0)
    exp1 = np.random.choice(np.arange(n_sample)[airmass_cut & nottwilight & faint_moon])
    exp2 = np.random.choice(np.arange(n_sample)[airmass_cut & nottwilight & bright_moon])
    exp3 = np.random.choice(np.arange(n_sample)[airmass_cut & yestwilight & medium_moon])
    for iexp, exp in enumerate([exp1, exp2, exp3]): 
        print('------ exp #%i -------' % iexp)
        print('airmass=%.1f' % airmass[exp])
        print('moon ill=%.2f, alt=%.1f, sep=%.1f' % (moon_ill[exp], moon_alt[exp], moon_sep[exp]))
        print('sun alt=%.2f, sep=%.1f' % (sun_alt[exp], sun_sep[exp]))
        
    texps = 60. * np.array([3, 5, 8, 12, 15]) # 3 to 15 min 
    
    for iexp, exp in enumerate([exp1, exp2, exp3]): 
        for texp in texps: 
            _fexp = specfile.replace('sourceSpec', 'bgsSpec').replace('.hdf5', '.TSreview.exp%i.texp_%.f.fits' % (iexp, texp))
            Isky = Sky.Isky_newKS_twi(airmass[exp], moon_ill[exp], moon_alt[exp], moon_sep[exp], sun_alt[exp], sun_sep[exp])
            bgs = GALeg_noisySpec(specfile, texp, airmass[exp], Isky, filename=_fexp)

            fig = plt.figure(figsize=(10,20))
            sub = fig.add_subplot(411) 
            sub.plot(Isky[0], Isky[1], c='C1') 
            sub.text(0.05, 0.95, 
                    'texp=%.f, airmass=%.2f\nmoon ill=%.2f, alt=%.f, sep=%.f\nsun alt=%.f, sep=%.f' % 
                    (texp, airmass[exp], moon_ill[exp], moon_alt[exp], moon_sep[exp], sun_alt[exp], sun_sep[exp]), 
                    ha='left', va='top', transform=sub.transAxes, fontsize=15)
            sub.legend(loc='upper right', frameon=True, fontsize=20) 
            sub.set_xlim([3e3, 1e4]) 
            sub.set_ylim([0., 20.]) 

            for i in range(3): 
                sub = fig.add_subplot(4,1,i+2)
                for band in ['b', 'r', 'z']: 
                    sub.plot(bgs['wave_%s' % band], bgs['flux_%s' % band][i], c='C1') 
                sub.plot(wave, flux[i], c='k', ls=':', lw=1, label='no noise')
                if i == 0: sub.legend(loc='upper right', fontsize=20)
                sub.set_xlim([3e3, 1e4]) 
                sub.set_ylim([0., 15.]) 
            bkgd = fig.add_subplot(111, frameon=False) 
            bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            bkgd.set_xlabel('rest-frame wavelength [Angstrom]', labelpad=10, fontsize=25) 
            bkgd.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', labelpad=10, fontsize=25) 
            fig.savefig(_fexp.replace('.fits', '.png'), bbox_inches='tight') 
    return None 


if __name__=='__main__': 
    #GALeg_sourceSpec(5000)
    #GALeg_noisySpec_TSreview(os.path.join(dir_dat, 'GALeg.g15.sourceSpec.5000.hdf5')) 
