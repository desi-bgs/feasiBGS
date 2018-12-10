#!/bin/usr/python 
import os
import sys
import h5py 
import time 
import pickle 
import numpy as np 
# -- astropy -- 
from astropy import units as u
# -- desi --
from desispec.io import read_spectra
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat
from feasibgs import forwardmodel as FM 


def mockexp_gleg_simSpec_texp(iblock, iexp, texp=480.): 
    ''' simulate DESI BGS spectra with specified exposure time.
    '''
    # mock exposures
    exps = mockexp() 
    
    # GAMA-Legacy catalog
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7)

    redshift = gleg['gama-spec']['z']
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    r_mag_gama = gleg['gama-photo']['modelmag_r'] # r-band magnitude from GAMA (SDSS) photometry

    ngal = len(redshift) # number of galaxies
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg)
    hasmatch = (match != -999)
    
    # only the block 
    nblock = int(np.ceil(float(ngal) / 1000.))
    in_block = (hasmatch & (np.arange(ngal) >= (iblock-1) * 1000) & (np.arange(ngal) < iblock * 1000))

    # get source spectra  
    fblock = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.source_spectra.mockexp_block.', str(iblock), 'of', str(nblock), '.p'])
    if os.path.isfile(fblock): 
        flux_eml, wave, magnorm_flag = pickle.load(open(fblock, 'rb'))
    else: 
        s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
        emline_flux = s_bgs.EmissionLineFlux(gleg, index=np.arange(ngal)[in_block], dr_gama=3, silent=True)
        flux_eml, wave, _, magnorm_flag = s_bgs.Spectra(r_mag_apflux[in_block], redshift[in_block],
                                                        vdisp[in_block], seed=1, templateid=match[in_block],
                                                        emflux=emline_flux, mag_em=r_mag_gama[in_block], 
                                                        silent=False)
        pickle.dump([flux_eml, wave, magnorm_flag], open(fblock, 'wb'))

    # write GLeg catalog block for future reference
    cata_block = cata.select(index=in_block)
    cata_block['forwardmodel'] = {}
    cata_block['forwardmodel']['magnorm_flag'] = magnorm_flag

    fblock = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/', 
        'g15.mockexp_block.', str(iblock), 'of', str(nblock), '.hdf5'])
    if not os.path.isfile(fblock):
        cata.write(cata_block, fblock)
    
    # read in sky surface brightness
    w_sky, skybright_old = mockexp_skyBright('old')
    _, skybright_new = mockexp_skyBright('new')

    skybright_old = skybright_old[iexp,:] 
    skybright_new = skybright_new[iexp,:]
    u_surface_brightness = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()

    # with old sky model
    f_simspec_old = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.', str(iblock), 'of', str(nblock), 
        '.', str(int(texp)), '.iexp', str(iexp), '.KSsky.fits'])
    if os.path.isfile(f_simspec_old): 
        bgs_spectra = read_spectra(f_simspec_old)
    else: 
        bgs_spectra = fdesi.simExposure(wave, flux_eml, exptime=texp, 
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input', 
                    'sky': np.clip(skybright_old, 0, None) * u_surface_brightness, 
                    'wave': w_sky}, 
                filename=f_simspec_old)

    # with new sky model 
    f_simspec_new = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.', str(iblock), 'of', str(nblock), 
        '.', str(int(texp)), '.iexp', str(iexp), '.newKSsky.fits'])
    if os.path.isfile(f_simspec_new):
        bgs_spectra = read_spectra(f_simspec_new)
    else:
        bgs_spectra = fdesi.simExposure(wave, flux_eml, exptime=texp,
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input',
                    'sky': np.clip(skybright_new, 0, None) * u_surface_brightness,
                    'wave': w_sky},
                filename=f_simspec_new)
    return None 


def mockexp_gleg_simSpec(iblock, iexp): 
    ''' simulate DESI BGS spectra with exposure time from survey sim mock exposure
    '''
    # mock exposures
    exps = mockexp() 
    
    # GAMA-Legacy catalog
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7)

    redshift = gleg['gama-spec']['z']
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    r_mag_gama = gleg['gama-photo']['modelmag_r'] # r-band magnitude from GAMA (SDSS) photometry

    ngal = len(redshift) # number of galaxies
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg)
    hasmatch = (match != -999)
    
    # only the block 
    nblock = int(np.ceil(float(ngal) / 1000.))
    in_block = (hasmatch & (np.arange(ngal) >= (iblock-1) * 1000) & (np.arange(ngal) < iblock * 1000))

    # get source spectra  
    fblock = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.source_spectra.mockexp_block.', str(iblock), 'of', str(nblock), '.p'])
    if os.path.isfile(fblock): 
        flux_eml, wave, magnorm_flag = pickle.load(open(fblock, 'rb'))
    else: 
        s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
        emline_flux = s_bgs.EmissionLineFlux(gleg, index=np.arange(ngal)[in_block], dr_gama=3, silent=True)
        flux_eml, wave, _, magnorm_flag = s_bgs.Spectra(r_mag_apflux[in_block], redshift[in_block],
                                                        vdisp[in_block], seed=1, templateid=match[in_block],
                                                        emflux=emline_flux, mag_em=r_mag_gama[in_block], 
                                                        silent=False)
        pickle.dump([flux_eml, wave, magnorm_flag], open(fblock, 'wb'))

    # write GLeg catalog block for future reference
    cata_block = cata.select(index=in_block)
    cata_block['forwardmodel'] = {}
    cata_block['forwardmodel']['magnorm_flag'] = magnorm_flag

    fblock = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/', 
        'g15.mockexp_block.', str(iblock), 'of', str(nblock), '.hdf5'])
    if not os.path.isfile(fblock):
        cata.write(cata_block, fblock)
    
    # read in sky surface brightness
    w_sky, skybright_old = mockexp_skyBright('old')
    _, skybright_new = mockexp_skyBright('new')

    skybright_old = skybright_old[iexp,:] 
    skybright_new = skybright_new[iexp,:]
    u_surface_brightness = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()

    # with old sky model
    f_simspec_old = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.', str(iblock), 'of', str(nblock), 
        '.texp_default.iexp', str(iexp), '.KSsky.fits'])
    if os.path.isfile(f_simspec_old): 
        bgs_spectra = read_spectra(f_simspec_old)
    else: 
        bgs_spectra = fdesi.simExposure(wave, flux_eml, 
                exptime=exps['EXPTIME'][iexp], 
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input', 
                    'sky': np.clip(skybright_old, 0, None) * u_surface_brightness, 
                    'wave': w_sky}, 
                filename=f_simspec_old)

    # with new sky model 
    f_simspec_new = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.', str(iblock), 'of', str(nblock), 
        '.texp_default.iexp', str(iexp), '.newKSsky.fits'])
    if os.path.isfile(f_simspec_new):
        bgs_spectra = read_spectra(f_simspec_new)
    else:
        bgs_spectra = fdesi.simExposure(wave, flux_eml, 
                exptime=exps['EXPTIME'][iexp], 
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input',
                    'sky': np.clip(skybright_new, 0, None) * u_surface_brightness,
                    'wave': w_sky},
                filename=f_simspec_new)
    return None 


def mockexp_gleg_simSpec_lowHA(iexp): 
    ''' simulate DESI BGS spectra with exposure time from survey sim mock exposure
    '''
    # mock exposures
    exps = mockexp() 
    
    # GAMA-Legacy catalog
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7)

    redshift = gleg['gama-spec']['z']
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    r_mag_gama = gleg['gama-photo']['modelmag_r'] # r-band magnitude from GAMA (SDSS) photometry
    
    ha_gama = gleg['gama-spec']['ha_flux'] # halpha line flux 

    ngal = len(redshift) # number of galaxies
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg)
    hasmatch = (match != -999)
    
    # only the block 
    nblock = int(np.ceil(float(ngal) / 1000.))
    in_block = np.random.choice(np.arange(ngal)[hasmatch & (ha_gama < 10.)], 5000, replace=False) 
    #in_block = (hasmatch & (np.arange(ngal) >= (iblock-1) * 1000) & (np.arange(ngal) < iblock * 1000))

    # get source spectra  
    fblock = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.source_spectra.mockexp_block.lowHalpha.p'])
    if os.path.isfile(fblock): 
        flux_eml, wave, magnorm_flag = pickle.load(open(fblock, 'rb'))
    else: 
        s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
        emline_flux = s_bgs.EmissionLineFlux(gleg, index=np.arange(ngal)[in_block], dr_gama=3, silent=True)
        flux_eml, wave, _, magnorm_flag = s_bgs.Spectra(r_mag_apflux[in_block], redshift[in_block],
                                                        vdisp[in_block], seed=1, templateid=match[in_block],
                                                        emflux=emline_flux, mag_em=r_mag_gama[in_block], 
                                                        silent=False)
        pickle.dump([flux_eml, wave, magnorm_flag], open(fblock, 'wb'))

    # write GLeg catalog block for future reference
    cata_block = cata.select(index=in_block)
    cata_block['forwardmodel'] = {}
    cata_block['forwardmodel']['magnorm_flag'] = magnorm_flag

    fblock = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/', 'g15.mockexp_block.lowHalpha.hdf5'])
    if not os.path.isfile(fblock):
        cata.write(cata_block, fblock)
    
    # read in sky surface brightness
    w_sky, skybright_old = mockexp_skyBright('old')
    _, skybright_new = mockexp_skyBright('new')

    skybright_old = skybright_old[iexp,:] 
    skybright_new = skybright_new[iexp,:]
    u_surface_brightness = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()

    # with old sky model
    f_simspec_old = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.lowHalpha.texp_default.iexp', str(iexp), '.KSsky.fits'])
    if not os.path.isfile(f_simspec_old): 
        #bgs_spectra = read_spectra(f_simspec_old)
        bgs_spectra = fdesi.simExposure(wave, flux_eml, 
                exptime=exps['EXPTIME'][iexp], 
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input', 
                    'sky': np.clip(skybright_old, 0, None) * u_surface_brightness, 
                    'wave': w_sky}, 
                filename=f_simspec_old)

    # with new sky model 
    f_simspec_new = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.lowHalpha.texp_default.iexp', str(iexp), '.newKSsky.fits'])
    if not os.path.isfile(f_simspec_new):
        # bgs_spectra = read_spectra(f_simspec_new)
        bgs_spectra = fdesi.simExposure(wave, flux_eml, 
                exptime=exps['EXPTIME'][iexp], 
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input',
                    'sky': np.clip(skybright_new, 0, None) * u_surface_brightness,
                    'wave': w_sky},
                filename=f_simspec_new)
    return None 


def mockexp(): 
    # read exposures from file
    fexp = h5py.File(''.join([UT.dat_dir(), 'bgs_survey_exposures.withsun.hdf5']), 'r')
    exps = {} 
    for k in fexp.keys(): 
        exps[k] = fexp[k].value 
    fexp.close()
    return exps


def mockexp_skyBright(skymodel): 
    if skymodel == 'old': 
        # read in sky surface brightness using KS model
        w, sbright = pickle.load(open(''.join([UT.dat_dir(), 
            'KSsky_brightness.bgs_survey_exposures.withsun.p']), 'rb'))
    elif skymodel == 'new':
        w, sbright = pickle.load(open(''.join([UT.dat_dir(), 
            'newKSsky_twi_brightness.bgs_survey_exposures.withsun.p']), 'rb'))
    else: 
        raise ValueError
    return w, sbright


if __name__=="__main__": 
    iblock = sys.argv[1]
    iexp = int(sys.argv[2])
    if iblock == 'lowHA': 
        t0 = time.time() 
        mockexp_gleg_simSpec_lowHA(iexp)
        print('--- took %f mins ---' % ((time.time() - t0)/60.))
    else: 
        iblock = int(iblock) 
        texp = sys.argv[3]
        t0 = time.time() 
        if str(texp) == 'default': 
            mockexp_gleg_simSpec(iblock, iexp)
        else: 
            texp = float(texp) 
            mockexp_gleg_simSpec_texp(iblock, iexp, texp=texp)
        print('--- took %f mins ---' % ((time.time() - t0)/60.))
