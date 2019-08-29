__all__ = ['test_fmSpec'] 

import pytest
import numpy as np 
# --- gqp_mc --- 
from feasibgs import util as UT
from feasibgs import skymodel as Sky 
from feasibgs import catalogs as Cat
from feasibgs import forwardmodel as FM 


def test_fmSpec(): 
    # read in GAMA-Legacy catalog
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7, silent=True) # these values shouldn't change 

    redshift = gleg['gama-spec']['z']
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    r_mag_gama = gleg['gama-photo']['r_model'] # r-band magnitude from GAMA (SDSS) photometry

    ngal    = len(redshift) # number of galaxies
    vdisp   = np.repeat(100.0, ngal) # velocity dispersions [km/s]
    igal    = np.array([np.random.choice(np.arange(ngal))]) 

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg, index=igal)
    print(bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'].data[match]) 
    print(absmag_ugriz[:,igal].flatten()) 
    assert np.mean(np.abs(bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'].data[match] - absmag_ugriz[:,igal].flatten())) < 1.
    
    # generate noiseless spectra for these galaxies 
    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=15000) 
    # emission lines from GAMA 
    emline_flux = s_bgs.EmissionLineFlux(gleg, index=igal, dr_gama=3, silent=True) 
    print('emission lines', emline_flux.min(), emline_flux.max())
    assert emline_flux.min() >= 0. 

    flux, wave, _, magnorm_flag = s_bgs.Spectra(r_mag_apflux[igal], redshift[igal],
            vdisp[igal], seed=1, templateid=match,
            emflux=emline_flux, mag_em=r_mag_gama[igal], 
            silent=True)
    print('noiseless spectra', flux)
    assert np.median(flux) > 0.

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()
    dark = fdesi.simExposure(wave, flux, exptime=1000, airmass=1.1, seeing=1.1, Isky=None, filename=None)
    print('noisy spectra (fiducial dark)', dark.flux) 
    assert np.var(dark.flux['b']) > np.var(flux[:,(wave > dark.wave['b'].min()) & (wave < dark.wave['b'].max())]) 

    # get sky brightness for BGS-like conditions 
    airmass = 1.1
    moonill = 0.7
    moonalt = 60.
    moonsep = 80.

    wsky, Isky_notwi = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -30., 180.)
    bright_notwi = fdesi.simExposure(wave, flux, exptime=1000, airmass=1.1, seeing=1.1, Isky=[wsky, Isky_notwi], filename=None)
    print('noisy spectra (non-twilight bright)', bright_notwi.flux) 

    wsky, Isky_twi = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, -10., 70.)
    bright_twi = fdesi.simExposure(wave, flux, exptime=1000, airmass=1.1, seeing=1.1, Isky=[wsky, Isky_twi], filename=None)
    print('noisy spectra (twilight bright)', bright_twi.flux) 

    assert np.var(bright_notwi.flux['b']) > np.var(dark.flux['b']) 
    assert np.var(bright_twi.flux['b']) > np.var(dark.flux['b']) 
    assert np.var(bright_twi.flux['b']) > np.var(bright_notwi.flux['b'])
