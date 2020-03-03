'''


script for generating completeness simulations and using them to make redshift
success rate calculations


'''
import os 
import h5py 
import fitsio 
import numpy as np 
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


dir_dat = os.path.join(UT.dat_dir(), 'srp')



def GALeg_G15_sourceSpec5000(seed=0): 
    '''generate noiseless simulated spectra for a subset of 5000 GAMAlegacy 
    galaxies in the G15 field. The output hdf5 file will also contain all 
    the galaxy properties.
    '''
    np.random.seed(seed) 
    # read in GAMA-Legacy catalog with galaxies in both GAMA and Legacy surveys
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7, silent=True)  
    
    # extract meta-data of galaxies 
    redshift        = gleg['gama-spec']['z']
    absmag_ugriz    = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux    = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1]) # aperture flux
    r_mag_gama      = gleg['gama-photo']['r_petro'] # r-band magnitude from GAMA (SDSS) photometry
    ha_gama         = gleg['gama-spec']['ha_flux'] # halpha line flux

    ngal = len(redshift) # number of galaxies
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg)
    hasmatch = (match != -999)
    criterion = hasmatch 
    
    # randomly pick a few more than 5000 galaxies from the catalog that have 
    # matching templates because some of the galaxies will have issues where the 
    # emission line is brighter than the photometric magnitude.  
    subsamp = np.random.choice(np.arange(ngal)[criterion], int(1.1 * 5000), replace=False) 

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
    # only keep 5000 galaxies
    isubsamp = np.random.choice(np.arange(len(subsamp))[magnorm_flag], 5000, replace=False) 
    subsamp = subsamp[isubsamp]
    
    # save to file  
    fspec = os.path.join(dir_dat, 'GALeg.g15.sourceSpec.5000.seed%i.hdf5' % seed)
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


def GALeg_G15_noisySpec5000(t_fid=150.): 
    ''' Construct BGS spectral simulations for 5000 galaxies in the GAMA G15 field using
    observing conditions sampled from surveysim output exposures.

    :param t_fid: 
        fiducial dark exposure time in seconds. This exposure time is scaled up 
        by the ETC to calculate all the exposure times. (default: 150)
    '''
    # read in 8 sampled bright time exposures
    fexps       = h5py.File(os.path.join(dir_dat, 'exposures_surveysim_fork_150sv0p5.sample.seed0.hdf5'), 'r')
    airmass     = fexps['airmass'][...]
    moon_ill    = fexps['moon_ill'][...]
    moon_alt    = fexps['moon_alt'][...]
    moon_sep    = fexps['moon_sep'][...]
    sun_alt     = fexps['sun_alt'][...]
    sun_sep     = fexps['sun_sep'][...]
    seeing      = fexps['seeing'][...]
    transp      = fexps['transp'][...]
    n_sample    = len(airmass) 

    # read in sky brightness 
    wave_sky    = fexps['wave'][...]
    u_sb        = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second
    sky_sbright = fexps['sky'][...]
    
    # scale up the exposure times
    texp        = fexps['texp_total'][...] / 150. * t_fid 
        
    # read in noiseless spectra
    specfile = os.path.join(dir_dat, 'GALeg.g15.sourceSpec.5000.seed0.hdf5')
    fspec = h5py.File(specfile, 'r') 
    wave = fspec['wave'][...]
    flux = fspec['flux'][...] 
    
    for iexp in range(len(airmass)):
        _fexp = specfile.replace('sourceSpec', 'bgsSpec').replace('.hdf5', 
                '.exposures_surveysim_fork_150sv0p5.sample.seed0.tfid%.f.exp%i.fits' % (t_fid, iexp))
        print('--- constructing %s ---' % _fexp) 
        print('t_exp=%.f (unscaled %.f)' % (texp[iexp], fexps['texp_total'][...][iexp]))
        print('airmass=%.2f' % airmass[iexp])
        print('moon ill=%.2f alt=%.f, sep=%.f' % (moon_ill[iexp], moon_alt[iexp], moon_sep[iexp]))
        print('sun alt=%.f, sep=%.f' % (sun_alt[iexp], sun_sep[iexp]))
        print('seeing=%.2f, transp=%.2f' % (seeing[iexp], transp[iexp]))
        
        # iexp-th sky spectra 
        Isky = [wave_sky, sky_sbright[iexp]]

        # simulate the exposures 
        fdesi = FM.fakeDESIspec()
        bgs = fdesi.simExposure(wave, flux, exptime=texp[iexp], airmass=airmass[iexp], Isky=Isky, filename=_fexp) 

        # --- some Q/A plots --- 
        fig = plt.figure(figsize=(10,20))
        sub = fig.add_subplot(411) 
        sub.plot(wave_sky, sky_sbright[iexp], c='C1') 
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
        fig.savefig(_fexp.replace('.fits', '.png'), bbox_inches='tight') 
    return None 


def zsuccess(): 
    ''' compare the redshift completeness 
    '''
    # get true redshifts 
    specfile = os.path.join(dir_dat, 'GALeg.g15.sourceSpec.5000.seed0.hdf5')
    fspec = h5py.File(specfile, 'r') 
    ztrue = fspec['zred'][...]
    r_mag = UT.flux2mag(fspec['legacy-photo']['flux_r'][...], method='log') 

    f_rr = lambda tfid, iexp: os.path.join(dir_dat,
            'GALeg.g15.bgsSpec.5000.seed0.exposures_surveysim_fork_150sv0p5.sample.seed0.tfid%i.exp%i.rr.fits'
            % (tfid, iexp))

    fig = plt.figure(figsize=(20,10))
    for iexp in range(8): 
        sub = fig.add_subplot(2,4,iexp+1) 
        sub.plot([15., 22.], [1., 1.], c='k', ls='--', lw=2)
        
        _plts = [] 
        for i, tfid in enumerate([130, 150, 200]): 
            if not os.path.isfile(f_rr(tfid, iexp)): 
                continue 

            rr      = fitsio.read(f_rr(tfid, iexp)) # read redrock file
            zrr     = rr['Z']
            dchi2   = rr['DELTACHI2']
            zwarn   = rr['ZWARN']
            _zsuc   = UT.zsuccess(zrr, ztrue, zwarn, deltachi2=dchi2, min_deltachi2=40.)
            wmean, rate, err_rate = UT.zsuccess_rate(r_mag, _zsuc, range=[15,22], nbins=28, bin_min=10) 

            _plt = sub.errorbar(wmean, rate, err_rate, fmt='.C%i' % i, elinewidth=2, markersize=10)
            _plts.append(_plt) 

        sub.vlines(19.5, 0., 1.2, color='k', linestyle=':', linewidth=1)
        sub.set_xlim([16., 21.]) 
        if iexp < 4: sub.set_xticklabels([]) 
        sub.set_ylim([0.6, 1.1])
        if iexp in [0,4]: sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
        else: sub.set_yticklabels([]) 
    sub.legend(_plts, 
            [r'$t_{\rm fid} = 130s$', r'$t_{\rm fid} = 150s$', r'$t_{\rm fid} = 200s$'], 
            fontsize=25, loc='lower right') 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'Legacy DR7 $r$ magnitude', labelpad=10, fontsize=30)
    bkgd.set_ylabel(r'redrock $z$ success rate', labelpad=10, fontsize=30)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(os.path.join(dir_dat, 'comp_sim.zsuccess.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #GALeg_G15_noisySpec5000(t_fid=130.)
    #GALeg_G15_noisySpec5000(t_fid=200.)
    zsuccess()
