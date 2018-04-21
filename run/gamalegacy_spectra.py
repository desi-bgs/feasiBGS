#!/bin/usr/python 
import numpy as np 
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat
from feasibgs import forwardmodel as FM 

from redrock.external.desi import rrdesi


def _test_NERSC(): 
    ''' test that every part of the code is functional with a quick version of the 
    actual run
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    i_rand = np.random.choice(range(len(redshift)), size=10) 
    
    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 
    
    # r-band aperture magnitude from Legacy photometry 
    r_mag = UT.flux2mag(gleg['legacy-photo']['apflux_r'][i_rand,1])  

    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]

    flux, wave, meta = bgstemp.Spectra(
            r_mag, redshift[i_rand], vdisp,
            seed=1, templateid=match, silent=False) 
    flux0 = flux.copy()  
    wave, flux_eml = bgstemp.addEmissionLines(wave, flux, gleg, i_rand, silent=False) 

    # simulate exposure using 
    fdesi = FM.fakeDESIspec() 
    f_bright =  UT.dat_dir()+'spectra_tmp_bright.fits'
    bgs_spectra_bright = fdesi.simExposure(wave, flux_eml, skycondition='bright', filename=f_bright) 

    rrdesi(options=['--zbest', ''.join([f_bright.split('.fits')[0]+'_zbest.fits']) , f_bright]) 
    return None 


def expSpectra(skycondition='bright', seed=1):
    ''' spectra with simulated exposure for the galaxies in the 
    Gama-Legacy survey.
    '''
    if skycondition not in ['bright', 'dark']: raise ValueError
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 
    ngal = len(redshift) # number of galaxies

    # match galaxies in the catalog to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg) 
    hasmatch = (match != -999) 
    print('%i galaxies do not have matches' % (len(match) - np.sum(hasmatch)))
    
    # r-band aperture magnitude from Legacy photometry 
    r_mag = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    flux, wave, meta = bgstemp.Spectra(r_mag[hasmatch], redshift[hasmatch], vdisp[hasmatch],
            seed=seed, templateid=match[hasmatch], silent=False) 
    flux0 = flux.copy()  
    wave, flux_eml = bgstemp.addEmissionLines(wave, flux, gleg, np.arange(ngal)[hasmatch], 
            silent=False) 

    # simulate exposure using 
    fdesi = FM.fakeDESIspec() 

    f = ''.join([UT.dat_dir(), 
        'gama_legacy.expSpectra.', skycondition, 'sky.seed', str(seed), '.fits']) 
    bgs_spectra = fdesi.simExposure(wave, flux_eml, skycondition=skycondition, 
            filename=f) 
    # save indices for future reference 
    f_index = ''.join([UT.dat_dir(), 
        'gama_legacy.expSpectra.', skycondition, 'sky.seed', str(seed), '.index']) 
    np.savetxt(f_indx, np.arange(ngal)[hasmatch], fmt='%i')
    return None 


if __name__=='__main__':
    expSpectra(skycondition='bright', seed=1)
