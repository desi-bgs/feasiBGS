'''
'''
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


def gleg_simSpec(nsub, validate=False): 
    '''generate noiseless simulated spectra for a subset of GAMAlegacy 
    galaxies. The output hdf5 file will also contain all the galaxy
    properties 

    :params ngal: 
        number of galaxies to randomly select from the GAMALegacy 
        joint catalog 
    '''
    # read in GAMA-Legacy catalog
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7) # these values shouldn't change 

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
    
    # randomly pick a few more than nsub galaxies from the catalog
    subsamp = np.random.choice(np.arange(ngal)[hasmatch], int(1.1 * nsub), replace=False) 

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
    
    fsub = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 'g15.simSpectra.', str(nsub), '.p']), 'w') 
    fsub.create_dataset('zred', data=redshift[subsamp])
    fsub.create_dataset('absmag_ugriz', data=absmag_ugriz[:,subsamp]) 
    fsub.create_dataset('r_mag_apflux', data=r_mag_apflux[subsamp]) 
    fsub.create_dataset('r_mag_gama', data=r_mag_gama[subsamp]) 
    fsub.create_dataset('flux', data=flux[isubsamp, :])
    fsub.create_dataset('wave', data=wave)
    fsub.close()

    if validate: 
        fig = plt.figure(figsize=(10,8))
        sub = fig.add_subplot(111)
        for i in range(10): #np.random.choice(isubsamp, 10, replace=False): 
            wave_rest = wave / (1.+redshift[subsamp][i])
            sub.plot(wave_rest, flux[isubsamp[i],:]) 
        sub.set_xlabel('rest-frame wavelength [Angstrom]', fontsize=25) 
        sub.set_xlim([3e3, 1e4]) 
        sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=25) 
        sub.set_ylim([0., 20.]) 
        fig.savefig(''.join([UT.dat_dir(), 'bgs_zsuccess/', 'g15.simSpectra.', str(nsub), '.png']), bbox_inches='tight') 
    return None 


def gleg_simSpec_mockexp(nsub, iexp, method='spacefill'): 
    ''' Given noiseless spectra, simulate DESI BGS noise based on
    iexp 
    '''
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
    u_sb = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()

    # with old sky model
    f_simspec_old = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.', str(iblock), 'of', str(nblock), 
        '.texp_default.iexp', str(iexp), '.KSsky.fits'])
    if not os.path.isfile(f_simspec_old): 
        bgs_spectra = fdesi.simExposure(wave, flux_eml, 
                exptime=exps['EXPTIME'][iexp], 
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input', 
                    'sky': np.clip(skybright_old, 0, None) * u_sb, 
                    'wave': w_sky}, 
                filename=f_simspec_old)

    # with new sky model 
    f_simspec_new = ''.join([UT.dat_dir(), 'spectra/gamadr3_legacydr7/',
        'g15.sim_spectra.mockexp_block.', str(iblock), 'of', str(nblock), 
        '.texp_default.iexp', str(iexp), '.newKSsky.fits'])
    if not os.path.isfile(f_simspec_new):
        bgs_spectra = fdesi.simExposure(wave, flux_eml, 
                exptime=exps['EXPTIME'][iexp], 
                airmass=exps['AIRMASS'][iexp],
                skycondition={'name': 'input',
                    'sky': np.clip(skybright_new, 0, None) * u_sb,
                    'wave': w_sky},
                filename=f_simspec_new)
    return None 

if __name__=="__main__": 
    gleg_simSpec(3000, validate=True)
