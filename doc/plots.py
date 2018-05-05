'''

make interesting plots 

'''
import os 
import h5py 
import numpy as np 
import scipy.integrate as integ
from pylab import cm
import healpy as HP
import fitsio as FitsIO
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from pydl.pydlutils.spheregroup import spherematch
# -- local -- 
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


def DESI_GAMA(): 
    ''' overplot the GAMA DR2 footprint onto the DESI 
    footprint.  
    '''
    # read in desi healpix weights
    w_hp = FitsIO.read(UT.dat_dir()+'desi-healpix-weights.fits') 
    
    gama = Cat.GAMA() 
    data = gama.Read(silent=True)
    theta_gama = 0.5 * np.pi - np.deg2rad(data['photo']['dec']) 
    phi_gama = np.deg2rad(data['photo']['ra'])
    print('GAMA theta: %f - %f' % (theta_gama.min(), theta_gama.max()))
    print('GAMA phi: %f - %f' % (phi_gama.min(), phi_gama.max()))

    fig = plt.figure(1, figsize=(10, 7.5))
    cmap = cm.Blues
    cmap.set_under('w')
    HP.mollview(w_hp, cmap=cmap, title='', min=0, max=1, nest=True, fig=1)
    HP.graticule()
    HP.projscatter(theta_gama, phi_gama, color='C1', s=1, linewidth=0) 
    HP.projtext(15., 38., 'DESI', color='navy', fontsize=20, lonlat=True) 
    HP.projtext(250., 10., 'GAMA DR2', color='C1', fontsize=20, lonlat=True) 
    fig.delaxes(fig.axes[1])
    fig.savefig(UT.doc_dir()+"figs/DESI_GAMA.pdf", bbox_inches='tight')
    return None


def GAMALegacy_Halpha_color(): 
    ''' color versus Halpha line flux relation for the GAMA-Legacy matched 
    legacy. 
    '''
    bands = ['g', 'r', 'z'] 
    # read in GAMA-Legacy objects
    gamaleg = Cat.GamaLegacy() 
    gleg = gamaleg.Read(silent=True)

    # GAMA Halpha line flux:  
    gama_ha = gleg['gama-spec']['ha'] 
    
    # legacy g,r,z model fluxes in nMgy
    legacy_photo = np.array([gleg['legacy-photo']['flux_'+band] for band in bands]) 
    #legacy_modelmag = np.array([UT.flux2mag(legacy_photo[i], bands=bands[i]) for i in range(len(bands))]) 
    legacy_modelmag = np.array([22.5 - 2.5*np.log10(legacy_photo[i]) for i in range(len(bands))]) 
    # legacy photometry color 
    legacy_gr = legacy_modelmag[0] - legacy_modelmag[1]
    legacy_rz = legacy_modelmag[1] - legacy_modelmag[2]
    
    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(121) # halpha vs (g - r)
    sub.scatter(legacy_gr[::10], gama_ha[::10], c='k', s=1) 
    sub.set_xlabel(r'$(g - r)$ color from Legacy DR5', fontsize=20) 
    sub.set_xlim([-0.5, 2.5])
    sub.set_xticks([0., 1., 2.]) 
    sub.set_ylabel(r'$H_\alpha$ line flux GAMA DR2 $[10^{-17}erg/s/cm^2]$', fontsize=20)
    sub.set_yscale('log')
    sub.set_ylim([1e-2, 1e4])

    sub = fig.add_subplot(122) # halpha vs (r - z)
    sub.scatter(legacy_rz[::10], gama_ha[::10], c='k', s=1) 
    sub.set_xlabel(r'$(r - z)$ color Legacy DR5', fontsize=20) 
    sub.set_xlim([-0.5, 1.5])
    sub.set_xticks([0., 1.]) 
    sub.set_yscale('log')
    sub.set_ylim([1e-2, 1e4])
    sub.set_yticklabels([]) 
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(UT.doc_dir()+"figs/GAMALegacy_Halpha_color.pdf", bbox_inches='tight')
    plt.close() 
    return None 


def BGStemplates(): 
    ''' plot the redshift distribution and M_0.1r vs ^0.1(g-r) relation of 
    BGS templates 
    '''
    # read in GAMA-Legacy objects
    legacy = Cat.GamaLegacy() 
    legacy_data = legacy.Read(silent=True)

    # GAMA Halpha 
    gama_z = legacy_data['gama-spec']['z_helio']

    bgs3 = FM.BGStree() 
    
    fig = plt.figure(figsize=(12,6))
    # redshift distribution of the templates 
    sub1 = fig.add_subplot(121) 
    _ = sub1.hist(gama_z, bins=25, range=(0., 1.), histtype='stepfilled', label='GAMA DR2')
    _ = sub1.hist(bgs3.meta['Z'], bins=25, range=(0., 1.), histtype='stepfilled', label='Templates')#, normed=True)
    sub1.legend(loc='upper right', prop={'size': 20}) 
    sub1.set_xlabel('Redshift', fontsize=20) 
    sub1.set_xlim([0., 0.8]) 
    
    # M_r0.1 vs (g-r)0.1 of the templates
    Mabs = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # absolute magnitude 
    print('Number of templates = %i' % Mabs.shape[0]) 

    sub2 = fig.add_subplot(122)
    sub2.scatter(Mabs[:,2], Mabs[:,1] - Mabs[:,2], c='k', s=2) 
    sub2.set_xlabel(r'$M_{0.1r}$', fontsize=20)
    sub2.set_xlim([-14., -24.]) 
    sub2.set_ylabel(r'$^{0.1}(g - r)$', fontsize=20)
    sub2.set_ylim([-0.2, 1.3]) 
    fig.subplots_adjust(wspace=0.3) 
    fig.savefig(UT.doc_dir()+"figs/BGStemplates.pdf", bbox_inches='tight')
    plt.close() 
    return None


def GamaLegacy_matchSpectra(): 
    ''' match galaxies from the GAMA-Legacy catalog to BGS templates based on 
    meta data and then plot their spectra
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio']  # redshift
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 
    
    # BGS templates
    bgs3 = FM.BGStree() 
    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    # pick 10 random galaxies from the GAMA-legacy sample
    # and then find the closest template
    x_bins = np.linspace(-24., -14., 4) 
    y_bins = np.linspace(-0.2, 1.2, 4) 
    i_rand = [] 
    for ix in range(len(x_bins)-1): 
        for iy in range(len(y_bins)-1): 
            inbin = np.where((y_bins[ix] < absmag_ugriz[1,:] - absmag_ugriz[2,:]) 
                    & (y_bins[ix+1] > absmag_ugriz[1,:] - absmag_ugriz[2,:])
                    & (x_bins[iy] < absmag_ugriz[2,:]) 
                    & (x_bins[iy+1] > absmag_ugriz[2,:])) 
            if len(inbin[0]) > 0: 
                i_rand.append(np.random.choice(inbin[0], size=1)[0]) 
    assert len(i_rand) > 5
    i_rand = np.array(i_rand)
    
    # meta data of [z, M_r0.1, 0.1(g-r)]
    gleg_meta = np.vstack([
        redshift[i_rand], 
        absmag_ugriz[2,i_rand], 
        absmag_ugriz[1,i_rand] - absmag_ugriz[2,i_rand]]).T
    match, _ = bgs3.Query(gleg_meta)
    
    # velocity dispersion 
    vdisp = np.repeat(100.0, len(i_rand)) # [km/s]
    
    flux, wave, meta = bgstemp.Spectra(
            gleg['gama-photo']['modelmag_r'][i_rand], 
            redshift[i_rand], 
            vdisp,
            seed=1, templateid=match, silent=False) 
    
    fig = plt.figure(figsize=(12,6))
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    sub1.scatter(absmag_ugriz[2,:], absmag_ugriz[1,:] - absmag_ugriz[2,:], c='k', s=1) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, edgecolors='k', marker='s', label='GAMA object')
        if ii == 0: 
            sub1.legend(loc='upper left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot template spectra
        sub2.plot(wave, np.log10(flux[ii]), c='C'+str(ii)) 
    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylabel(r'$^{0.1}(g-r)$ color', fontsize=20) 
    sub1.set_ylim([-0.2, 1.6])
    sub1.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    
    sub2.text(0.9, 0.9, 'Template Spectra', ha='right', va='center', transform=sub2.transAxes, fontsize=20)
    sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub2.set_xlim([1.5e3, 2e4]) 
    sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub2.set_yscale('log') 
    sub2.set_ylim([1.e-1, 3.]) 
    fig.subplots_adjust(wspace=0.3) 
    fig.savefig(UT.doc_dir()+"figs/GamaLegacy_matchedtempSpectra.pdf", bbox_inches='tight')
    plt.close() 
    return None


def GamaLegacy_emlineSpectra(): 
    ''' Emission lines are added to template spectra based on GAMA emission line data
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    i_rand = np.random.choice(range(len(redshift)), size=1) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]

    flux, wave, meta = bgstemp.Spectra(
            gleg['gama-photo']['modelmag_r'][i_rand], 
            redshift[i_rand], 
            vdisp,
            seed=1, templateid=match, silent=False) 
    flux0 = flux.copy()  
    wave, flux_eml = bgstemp.addEmissionLines(wave, flux, gleg, i_rand, silent=False) 
    
    emline_keys = ['[OII]', r'$\mathrm{H}_\beta$',  r'[OIII]$_b$', r'[OIII]$_r$', 'NII', r'$\mathrm{H}_\alpha$', '[NII]', '[SII]']
    emline_lambda = [3727., 4861., 4959., 5007., 6548., 6563., 6584., 6716.]
    emline_zlambda = (1.+redshift[i_rand]) * np.array(emline_lambda)

    fig = plt.figure(figsize=(12,6))
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    sub1.scatter(absmag_ugriz[2,:], absmag_ugriz[1,:] - absmag_ugriz[2,:], c='k', s=1) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, edgecolors='k', marker='s', label='GAMA object')
        if ii == 0: 
            sub1.legend(loc='upper left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot template spectra w/ emission lines
        sub2.plot(wave, flux_eml[ii], c='C'+str(ii), label='Template w/ Em.Lines') 
        # plot template spectra
        sub2.plot(wave, flux0[ii], c='k', ls=':', lw=0.5, label='Template') 
    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylabel(r'$^{0.1}(g-r)$ color', fontsize=20) 
    sub1.set_ylim([-0.2, 1.6])
    sub1.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    for i_l, zlambda in enumerate(emline_zlambda): 
        sub2.vlines(zlambda, 0., 2.*flux_eml[0].max(), color='k', linestyle=':', linewidth=1) 
        sub2.text(zlambda, flux_eml[0].max()*float(28-i_l)/20., emline_keys[i_l], ha='left', va='top', fontsize=12) 
    sub2.legend(loc='upper right', prop={'size': 15})
    #sub2.text(0.9, 0.9, 'Template Spectra', ha='right', va='center', transform=sub2.transAxes, fontsize=20)
    sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub2.set_xlim([3.5e3, 1e4]) 
    sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub2.set_ylim([0., 1.8*flux_eml[0].max()]) 
    fig.subplots_adjust(wspace=0.3) 
    fig.savefig(UT.doc_dir()+"figs/GLeg_EmLineSpectra.pdf", bbox_inches='tight')
    plt.close() 
    return None


def skySurfaceBrightness():
    ''' compare the sky surface brightness from the method
    `feasibgs.forwardmodel.fakeDESIspec.skySurfBright` to the 
    sky fluxes provided by Parker.
    '''
    # ccd wavelength limit
    wavemin, wavemax = 3533., 9913. 

    # read in bright sky flux measured in BOSS by Parker
    f = UT.dat_dir()+'sky/moon_sky_spectrum.hdf5'
    f_hdf5 = h5py.File(f, 'r')
    ws, ss = [], []
    for i in range(4):
        ws.append(f_hdf5['sky'+str(i)+'/wave'].value)
        ss.append(f_hdf5['sky'+str(i)+'/sky'].value)
    # convert sky fluxes to surface brightnesses by dividing by the 
    # fiber area of pi arcsec^2
    bright_sky_sbright0 = [10.*ws[2], ss[2]/np.pi]
    bright_sky_sbright1 = [10.*ws[3], ss[3]/np.pi]
    
    # this is just to get wavelenghts 
    ww, _ = np.loadtxt(UT.dat_dir()+'sky/spec-sky.dat', 
            unpack=True, skiprows=2, usecols=[0,1])
    waves = ww[(ww > wavemin) & (ww < wavemax)] * u.Angstrom
    
    # get our fit sky surface brightness
    fDESI = FM.fakeDESIspec()
    dark_sbright = fDESI.skySurfBright(waves, cond='dark')
    bright_sbright = fDESI.skySurfBright(waves, cond='bright') 

    # now lets compare
    fig = plt.figure(figsize=(8,4))
    sub = fig.add_subplot(111)
    sub.scatter(bright_sky_sbright0[0], bright_sky_sbright0[1], c='C1', lw=0, s=1., label='Bright Sky')
    sub.scatter(bright_sky_sbright1[0], bright_sky_sbright1[1], c='C1', lw=0, s=1.)

    sub.scatter(waves.value, dark_sbright, c='k', lw=0, s=0.5, label="Model Dark Sky")
    sub.scatter(waves.value, bright_sbright, c='C0', lw=0, s=0.5, label="Model Bright Sky")
    sub.set_xlabel('Wavelength', fontsize=20)
    sub.set_xlim([3600., 9800.])
    sub.set_ylabel("Surface Brightness [$10^{-17} erg/\AA/cm^2/s/''^2$]", fontsize=15)
    sub.set_yscale("log")
    sub.set_ylim([0.5, 3e1])
    sub.legend(loc='upper right', frameon=True, markerscale=10, prop={'size':20})
    fig.savefig(UT.doc_dir()+"figs/model_skySurfBright.pdf", bbox_inches='tight')
    plt.close() 
    return None


def rMag_normalize(): 
    ''' Compare the normalization of the template spectra using model r-magnitude 
    and apflux derived magnitude
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    i_rand = np.random.choice(range(len(redshift)), size=1) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # gama model r-band magnitude
    flux_mr, wave_mr, _ = bgstemp.Spectra(
            gleg['gama-photo']['modelmag_r'][i_rand], 
            redshift[i_rand], 
            vdisp,
            seed=1, templateid=match, silent=False) 
    flux_mr0 = flux_mr.copy()  
    wave_mr, flux_mr_eml = bgstemp.addEmissionLines(wave_mr, flux_mr0, gleg, i_rand, silent=False) 
    
    # derive r-band aperture magnitude from Legacy photometry 
    r_apflux = gleg['legacy-photo']['apflux_r'][:,1]  # nanomaggies
    r_mag = UT.flux2mag(r_apflux, method='log')  # convert to mag 
    r_mag_max = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,-1], method='log')

    flux, wave, meta = bgstemp.Spectra(
            r_mag[i_rand], 
            redshift[i_rand], 
            vdisp,
            seed=1, templateid=match, silent=False) 
    flux0 = flux.copy()  
    wave, flux_eml = bgstemp.addEmissionLines(wave, flux, gleg, i_rand, silent=False) 
    print('model magnitude: %f' % gleg['gama-photo']['modelmag_r'][i_rand]) 
    print('r aperture magnitude 0.75 arcsec: %f' % r_mag[i_rand]) 
    print('r aperture magnitude 7.0 arcsec: %f' % r_mag_max[i_rand]) 

    fig = plt.figure(figsize=(12,6))
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    sub1.scatter(absmag_ugriz[2,:], absmag_ugriz[1,:] - absmag_ugriz[2,:], c='k', s=1) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, edgecolors='k', marker='s', label='GAMA object')
        if ii == 0: 
            sub1.legend(loc='upper left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot template spectra w/ emission lines
        sub2.plot(wave, flux_mr_eml[ii], c='C'+str(ii), label='GAMA model $r$-band mag') 
        # plot template spectra
        sub2.plot(wave, flux_eml[ii], c='k', ls=':', lw=0.5, label="Legacy $r$-band $0.75''$ apflux") 
    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylabel(r'$^{0.1}(g-r)$ color', fontsize=20) 
    sub1.set_ylim([-0.2, 1.6])
    sub1.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    sub2.legend(loc='upper right', prop={'size': 15})
    sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub2.set_xlim([3.5e3, 1e4]) 
    sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub2.set_ylim([0., 1.8*flux_eml[0].max()]) 
    fig.subplots_adjust(wspace=0.3) 
    fig.savefig(UT.doc_dir()+"figs/GLeg_rMag_norm.pdf", bbox_inches='tight')
    plt.close() 
    return None


def expSpectra():
    ''' exposured spectra of DESI spectrograph simulated using dark and bright sky models. 
    Source flux is some templates with emission line data from GAMA DR2. 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 
    
    # pick a random galaxy from the GAMA-legacy sample
    #i_rand = np.random.choice(range(len(redshift)), size=1) 
    i_rand = [34893]
    print('i_rand = %i' % i_rand[0]) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # r-band aperture magnitude from Legacy photometry 
    r_mag = UT.flux2mag(gleg['legacy-photo']['apflux_r'][i_rand,1], method='log')  
    assert np.isfinite(r_mag)
    print('r_mag = %f' % r_mag)

    flux, wave, meta = bgstemp.Spectra(
            r_mag, 
            redshift[i_rand], 
            vdisp,
            seed=1, templateid=match, silent=False) 
    flux0 = flux.copy()  
    wave, flux_eml = bgstemp.addEmissionLines(wave, flux, gleg, i_rand, silent=False) 

    # simulate exposure using 
    fdesi = FM.fakeDESIspec() 
    bgs_spectra_bright = fdesi.simExposure(wave, flux_eml, skycondition='bright') 
    bgs_spectra_dark = fdesi.simExposure(wave, flux_eml, skycondition='dark') 
    # write out simulated spectra
    for b in ['b', 'r', 'z']: 
        tbl_bright = Table([bgs_spectra_bright.wave[b], bgs_spectra_bright.flux[b][0].flatten()], names=('lambda', 'flux')) 
        tbl_bright.write('obj'+str(i_rand[0])+'_brightsky_'+b+'.fits', format='fits', overwrite=True) 
        tbl_dark = Table([bgs_spectra_dark.wave[b], bgs_spectra_dark.flux[b][0].flatten()], names=('lambda', 'flux'))
        tbl_dark.write('obj'+str(i_rand[0])+'darksky_'+b+'.fits', format='fits', overwrite=True) 
    
    # plot the spectra
    fig, (sub1, sub2) = plt.subplots(1,2, figsize=(12,4), gridspec_kw={'width_ratios':[1,3]})
    sub1.scatter(absmag_ugriz[2,:][::10], absmag_ugriz[1,:][::10] - absmag_ugriz[2,:][::10], c='k', s=2) 
    i = i_rand[0]
    sub1.scatter(mabs_temp[match[0],2], mabs_temp[match[0],1] - mabs_temp[match[0],2],
            color='C0', s=30, edgecolors='k', marker='^', label='Template')
    sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
            color='C0', s=30, edgecolors='k', marker='s', label='GAMA object')
    sub1.legend(loc='lower left', markerscale=3, handletextpad=0., prop={'size':15})

    # plot exposed spectra of the three CCDs
    for b in ['b', 'r', 'z']: 
        lbl0, lbl1 = None, None
        if b == 'z': lbl0, lbl1 = 'Simulated Exposure (Dark Sky)', 'Bright Sky'
        sub2.plot(bgs_spectra_bright.wave[b], bgs_spectra_bright.flux[b][0].flatten(), 
                c='C1', lw=0.2, alpha=0.7, label=lbl1) 
        sub2.plot(bgs_spectra_dark.wave[b], bgs_spectra_dark.flux[b][0].flatten(), 
                c='C0', lw=0.2, alpha=0.7, label=lbl0) 
    # plot template spectra
    sub2.plot(wave, flux[0], c='k', lw=0.3, ls=':', label='Template')
    
    sub2.text(0.95, 0.9, r'$z_\mathrm{GAMA} = '+str(redshift[i_rand[0]])+'$', 
            ha='right', va='center', transform=sub2.transAxes, fontsize=20)

    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylabel(r'$^{0.1}(g-r)$ color', fontsize=20) 
    sub1.set_ylim([-0.2, 1.6])
    sub1.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub2.set_xlim([3600., 9800.]) 
    sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub2.set_ylim([0., 3*flux0[0].max()]) 
    sub2.legend(loc='upper left', prop={'size': 15}) 
    fig.savefig(UT.doc_dir()+"figs/Gleg_expSpectra.pdf", bbox_inches='tight')
    plt.close() 
    return None


def expSpectra_emline():
    ''' exposured spectra of DESI spectrograph simulated using dark and bright sky models, 
    focusing on the emission lines. 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    #i_rand = np.random.choice(range(len(redshift)), size=1) 
    i_rand = [34893]
    print('i_rand = %i' % i_rand[0]) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # r-band aperture magnitude from Legacy photometry 
    r_mag = UT.flux2mag(gleg['legacy-photo']['apflux_r'][i_rand,1], method='log')  
    print('r_mag = %f' % r_mag)

    flux, wave, meta = bgstemp.Spectra(r_mag, redshift[i_rand], vdisp, seed=1, templateid=match, silent=False) 
    wave, flux_eml = bgstemp.addEmissionLines(wave, flux, gleg, i_rand, silent=False) 

    # simulate exposure using 
    fdesi = FM.fakeDESIspec() 
    bgs_spectra_bright = fdesi.simExposure(wave, flux_eml, skycondition='bright') 
    bgs_spectra_dark = fdesi.simExposure(wave, flux_eml, skycondition='dark') 
    
    # plot the spectra
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(111)
    # plot exposed spectra of the three CCDs
    for b in ['b', 'r', 'z']: 
        lbl0, lbl1 = None, None
        if b == 'z': lbl0, lbl1 = 'Simulated Exposure (Dark Sky)', 'Bright Sky'
        sub.plot(bgs_spectra_bright.wave[b], bgs_spectra_bright.flux[b][0].flatten(), 
                c='C1', lw=0.2, alpha=0.7, label=lbl1) 
        sub.plot(bgs_spectra_dark.wave[b], bgs_spectra_dark.flux[b][0].flatten(), 
                c='C0', lw=0.2, alpha=0.7, label=lbl0) 
    
    emline_labels = ['[OII]', r'$\mathrm{H}_\beta$',  r'[OIII]$_b$', r'[OIII]$_r$', r'[NII]$_b$', r'$\mathrm{H}_\alpha$', 
            r'[NII]$_r$', '[SII]']
    emline_lambda = [3727., 4861., 4959., 5007., 6548., 6563., 6584., 6716.]
    emline_keys = ['oiib', 'hb',  'oiiib', 'oiiir', 'niib', 'ha', 'niir', 'siib']

    emline_zlambda = (1.+redshift[i_rand][0]) * np.array(emline_lambda)
    
    for i_l, zlambda in enumerate(emline_zlambda): 
        # mark the redshifted wavelength of the emission line 
        sub.vlines(zlambda, 0., 2.*flux_eml[0].max(), color='k', linestyle=':', linewidth=1) 
        sub.text(zlambda, flux[0].max()*float(14-i_l)/10., emline_labels[i_l], ha='left', va='top', fontsize=12) 

        # lineflux of the emissionline 
        emlineflux = gleg['gama-spec'][emline_keys[i_l]][i_rand][0]
        # width of emline 
        emlinesig = gleg['gama-spec'][emline_keys[i_l]+'sig'][i_rand][0]
        if (emlineflux == -99.) or (emlinesig <= 0.): continue 
        A = emlineflux/np.sqrt(2.*np.pi*emlinesig**2)

        f_eml = lambda ww: A*np.exp(-0.5*(ww-zlambda)**2/emlinesig**2)

        sub.plot(wave, f_eml(wave), c='k', linestyle=':', linewidth=2)
        print('-- %s --' % emline_keys[i_l]) 
        print('%f vs %f' % (emlineflux, integ.simps(f_eml(wave), x=wave)))

    sub.text(0.95, 0.9, r'$z_\mathrm{GAMA} = '+str(redshift[i_rand[0]])+'$', 
            ha='right', va='center', transform=sub.transAxes, fontsize=20)

    sub.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub.set_ylim([0., 3*flux[0].max()]) 
    sub.legend(loc='upper left', prop={'size': 15}) 
    fig.savefig(UT.doc_dir()+"figs/Gleg_expSpectra_emline.pdf", bbox_inches='tight')
    plt.close() 
    
    # plot the spectra
    fig = plt.figure(figsize=(20,8))
    # plot exposed spectra of the three CCDs
    for i_l, zlambda in enumerate(emline_zlambda): 
        sub = fig.add_subplot(2,int(np.ceil(0.5*len(emline_zlambda))),i_l+1)
        for b in ['b', 'r', 'z']: 
            lbl0, lbl1 = None, None
            if b == 'z': lbl0, lbl1 = 'Simulated Exposure (Dark Sky)', 'Bright Sky'
            sub.plot(bgs_spectra_bright.wave[b], bgs_spectra_bright.flux[b][0].flatten(), 
                    c='C1', lw=1, alpha=0.7, label=lbl1) 
            sub.plot(bgs_spectra_dark.wave[b], bgs_spectra_dark.flux[b][0].flatten(), 
                    c='C0', lw=1, alpha=0.7, label=lbl0) 
    
        # mark the redshifted wavelength of the emission line 
        sub.vlines(zlambda, 0., 2.*flux_eml[0].max(), color='k', linestyle=':', linewidth=0.5) 
        sub.text(0.9, 0.85, emline_labels[i_l]+'\n $\sigma='+str(gleg['gama-spec'][emline_keys[i_l]+'sig'][i_rand][0])+'$', 
                ha='right', va='center', transform=sub.transAxes, fontsize=20)

        # lineflux of the emissionline 
        emlineflux = gleg['gama-spec'][emline_keys[i_l]][i_rand][0]
        # width of emline 
        emlinesig = gleg['gama-spec'][emline_keys[i_l]+'sig'][i_rand][0]
        if (emlineflux == -99.) or (emlinesig <= 0.): continue 
        A = emlineflux/np.sqrt(2.*np.pi*emlinesig**2)

        f_eml = lambda ww: A*np.exp(-0.5*(ww-zlambda)**2/emlinesig**2)

        sub.plot(wave, f_eml(wave), c='k', linestyle=':', linewidth=1)
        sub.set_xlim([zlambda-50., zlambda+50.]) 
        sub.set_ylim([0., 3*flux[0].max()]) 

    fig.savefig(UT.doc_dir()+"figs/Gleg_expSpectra_emline_zoom.pdf", bbox_inches='tight')
    plt.close() 
    return None


def expSpectra_redshift(seed=1): 
    ''' check out the redshifts, measured using redrock, of the exposed spectra 
    simulated with dark and bright sky models. Check the impact of bright sky. 
    Make the plot of color vs Halpha 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio'] # redshift 
    ngal = len(redshift) # number of galaxies

    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    n_block = (ngal // 1000) + 1
        
    fig, (sub1, sub2) = plt.subplots(1,2, figsize=(8,4), gridspec_kw={'width_ratios':[1,1]})
    sub1.plot([0.0, 0.4], [0.0, 0.4], c='k', lw=1, ls='--') 
    
    for i_sky, skycondition in enumerate(['dark', 'bright']): 
        for i_block in range(n_block):  
            # read in redshift files 
            f_redrock = ''.join([UT.dat_dir(), 'spectra/',
                'gama_legacy.expSpectra.', skycondition, 'sky.seed', str(seed), 
                '.', str(i_block+1), 'of', str(n_block), 'blocks.zbest.fits']) 
            if not os.path.isfile(f_redrock): 
                continue 
            redrock_data = fits.open(f_redrock)[1].data

            # read in index files 
            f_index = ''.join([UT.dat_dir(), 'spectra/',
                'gama_legacy.expSpectra.', skycondition, 'sky.seed', str(seed), 
                '.', str(i_block+1), 'of', str(n_block), 'blocks.index']) 
            index_block = np.loadtxt(f_index, unpack=True, usecols=[0], dtype='i') 
            if i_block == 0: 
                indices = index_block
                zbest = redrock_data['Z']
                zwarn = redrock_data['ZWARN']
            else: 
                indices = np.append(indices, index_block) 
                zbest = np.append(zbest, redrock_data['Z']) 
                zwarn = np.append(zwarn, redrock_data['ZWARN']) 
    
        # gama redshift 
        z_gama = redshift[indices]
    
        # plot template spectra
        sub1.scatter(z_gama, zbest, c='C'+str(i_sky), s=30, label=skycondition+' sky') 
        
        dz_1pz = (zbest - z_gama)/(1.+z_gama)
        #sub2.scatter(z_gama, (zbest - z_gama)/(1.+z_gama), c='C'+str(i_sky), s=30, label=skycondition+' sky') 
        _ = sub2.hist(dz_1pz, color='C'+str(i_sky), histtype='step', range=(0.,1.), bins=50) 

        print('%s : %f percent zwarn = 0' % (skycondition, float(np.sum(zwarn == 0))/float(len(zwarn))) )
        print('%s : %f percent zwarn = 0, dz/(1+z) < 0.003' % 
                (skycondition, float(np.sum((zwarn == 0) & ((zbest - z_gama)/(1.+z_gama) < 0.003)))/float(len(zwarn))))

    sub1.legend(loc='upper left', markerscale=2, handletextpad=0., prop={'size':15}) 
    sub1.set_xlabel(r'$z_\mathrm{GAMA}$', fontsize=20) 
    sub1.set_xlim([0.,0.4]) 
    sub1.set_ylabel(r'$z_\mathrm{redrock}$', fontsize=20) 
    sub1.set_ylim([0.,0.4]) 

    sub2.plot([0., 0.4], [0.0, 0.0], c='k', ls='--', lw=1) 
    sub2.legend(loc='upper left', markerscale=2, handletextpad=0., prop={'size':15})
    #sub2.set_xlabel(r'$z_\mathrm{GAMA}$', fontsize=20) 
    sub2.set_xlim([0., 1.]) 
    sub2.set_xlabel(r'$\Delta z/(1+z_\mathrm{GAMA})$', fontsize=20) 
    fig.subplots_adjust(wspace=0.4) 
    fig.savefig(UT.doc_dir()+"figs/Gleg_expSpectra_redshift.pdf", bbox_inches='tight')
    plt.close() 
    return None          


def expSpectra_SDSScomparison(): 
    '''
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()
    cataid = gleg['gama-photo']['cataid'] # GAMA catalog ID 
    redshift = gleg['gama-spec']['z_helio'] # redshift 
    ngal = len(redshift) # number of galaxies

    # sdss spAll
    f_sdss = h5py.File(''.join([UT.dat_dir(), 'sdss/spAll-v5_7_0.zcut.hdf5']), 'r') 
    ra_sdss = f_sdss['ra'].value
    dec_sdss = f_sdss['dec'].value
    plate = f_sdss['plate'].value
    mjd = f_sdss['mjd'].value
    fiberid = f_sdss['fiberid'].value
    
    m_sdss, m_gleg, d_match = spherematch(ra_sdss, dec_sdss, 
            gleg['gama-photo']['ra'], gleg['gama-photo']['dec'], 0.000277778) # spherematch 
    
    # plot the spectra
    fig = plt.figure(figsize=(12,12))
    for i, igal in enumerate([2,3,7]): 
        i_obj = [m_gleg[igal]]
        i_sdss = m_sdss[igal]
        f_spec_local = fits.open(''.join([UT.dat_dir(), 'gama/spectra/',
                    'spec-', str(plate[i_sdss]), '-', str(mjd[i_sdss]), '-', str(fiberid[i_sdss]).zfill(4), '.fits']))
        sdss_spec = f_spec_local[1].data
        print('z_gama = %f' % redshift[i_obj]) 

        # match random galaxy to BGS templates
        bgs3 = FM.BGStree() 
        match = bgs3._GamaLegacy(gleg, index=i_obj) 
        mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

        bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
        vdisp = np.repeat(100.0, len(i_obj)) # velocity dispersions [km/s]
        
        # r-band aperture magnitude from Legacy photometry 
        r_mag = UT.flux2mag(gleg['legacy-photo']['apflux_r'][i_obj,1], method='log')  
        print('r_mag = %f' % r_mag)

        flux, wave, meta = bgstemp.Spectra(r_mag, redshift[i_obj], vdisp, seed=1, templateid=match, silent=False) 
        wave, flux_eml = bgstemp.addEmissionLines(wave, flux, gleg, i_obj, silent=False) 

        # simulate exposure using 
        fdesi = FM.fakeDESIspec() 
        bgs_spectra_bright = fdesi.simExposure(wave, flux_eml, skycondition='bright') 
        bgs_spectra_dark = fdesi.simExposure(wave, flux_eml, skycondition='dark') 
        
        sub = fig.add_subplot(3,1,i+1)
        sub.plot(10.**sdss_spec['loglam'], sdss_spec['flux'], c='k', lw=0.1, label='SDSS DR 7 spectra')  
        # plot exposed spectra of the three CCDs
        for b in ['b', 'r', 'z']: 
            lbl0, lbl1 = None, None
            if b == 'z': lbl0, lbl1 = 'Dark Sky', 'Simulated Exposure (Bright Sky)'
            sub.plot(bgs_spectra_bright.wave[b], bgs_spectra_bright.flux[b][0].flatten(), 
                    c='C1', lw=0.2, alpha=0.7, label=lbl1) 
            sub.plot(bgs_spectra_dark.wave[b], bgs_spectra_dark.flux[b][0].flatten(), 
                    c='C0', lw=0.2, alpha=0.7, label=lbl0) 

        if i == 2: sub.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
        sub.set_xlim([3600., 9800.]) 
        if i == 1: sub.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
        sub.set_ylim([-5., 15]) 
        #sub.set_ylim([0., 3*flux[0].max()]) 
        if i == 0: sub.legend(loc='upper left', prop={'size': 15}) 
    fig.savefig(UT.doc_dir()+"figs/Gleg_expSpectra_SDSScomparison.pdf", bbox_inches='tight')
    plt.close() 
    return None


def SDSS_emlineComparison(): 
    '''
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()
    redshift = gleg['gama-spec']['z_helio'] # redshift 
    ngal = len(redshift) # number of galaxies

    # sdss spAll
    f_sdss = h5py.File(''.join([UT.dat_dir(), 'sdss/spAll-v5_7_0.zcut.hdf5']), 'r') 
    ra_sdss = f_sdss['ra'].value
    dec_sdss = f_sdss['dec'].value
    plate = f_sdss['plate'].value
    mjd = f_sdss['mjd'].value
    fiberid = f_sdss['fiberid'].value
    
    f_sdss_emline = h5py.File(''.join([UT.dat_dir(), 'sdss/spAllLine-v5_7_0.zcut.hdf5']), 'r') #'linecontlevel'

    m_sdss, m_gleg, d_match = spherematch(ra_sdss, dec_sdss, 
            gleg['gama-photo']['ra'], gleg['gama-photo']['dec'], 0.000277778) # spherematch 
    
    igal = 2 
    i_sdss = m_sdss[igal]
    i_gleg = m_gleg[igal]
    f_spec_local = fits.open(''.join([UT.dat_dir(), 'gama/spectra/',
                'spec-', str(plate[i_sdss]), '-', str(mjd[i_sdss]), '-', str(fiberid[i_sdss]).zfill(4), '.fits']))
    sdss_spec = f_spec_local[1].data
    wave = 10.**sdss_spec['loglam']
    flux = sdss_spec['flux']

    emline_sdss_index = [6, 7, 15, 16, 17, 25, 26, 27, 28, 29]
    emline_gleg_key = ['oiib', 'oiir', 'hb', 'oiiib', 'oiiir', 'niib', 'ha', 'niir', 'siib', 'siir']
    emline_lambda = [3726., 3728., 4861., 4959., 5007., 6548., 6563., 6584., 6716., 6731.]
    emlambda_red = (1. + redshift[i_gleg])*np.array(emline_lambda)

    emline = np.zeros(len(wave))
    for i_k, k in enumerate(emline_gleg_key):
        if (gleg['gama-spec'][k][i_gleg] == -99.) or (gleg['gama-spec'][k+'sig'][i_gleg] < 0.): continue
        print('%s' % k)
        lineflux = gleg['gama-spec'][k][i_gleg]
        linesig = gleg['gama-spec'][k+'sig'][i_gleg]

        A = lineflux / np.sqrt(2.*np.pi*linesig**2)
        emline_flux = A * np.exp(-0.5*(wave-emlambda_red[i_k])**2/linesig**2)
    
        #wlim = (wave > emline_lambda[i_k] - 50.) & (wave < emline_lambda[i_k] + 50.)
        emline += emline_flux
        #emline[wlim] = emline_flux[wlim]# + f_sdss_emline['linecontlevel'].value[i_sdss, emline_sdss_index[i_k]]

    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(111)
    sub.plot(wave, flux, c='k', lw=0.5) 
    sub.plot(wave, emline, c='C1', lw=1, ls=':') 
    sub.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub.set_xlim([3550, 10325])
    sub.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub.set_ylim([-5., 15]) 
    #sub.legend(loc='upper left', prop={'size': 15}) 
    fig.savefig(UT.doc_dir()+"figs/SDSS_emlineComparison.pdf", bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    #GAMALegacy_Halpha_color()
    #BGStemplates()
    #GAMALegacy()
    #GamaLegacy_matchSpectra()
    #GamaLegacy_emlineSpectra()
    #skySurfaceBrightness()
    #rMag_normalize()
    #expSpectra()
    #expSpectra_emline()
    #expSpectra_redshift(seed=1)
    expSpectra_SDSScomparison()
    #SDSS_emlineComparison()
