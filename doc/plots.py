'''

make interesting plots 

'''
import os 
import h5py 
import pickle
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
from datetime import datetime
from astroplan import Observer
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, GCRS, get_sun, get_moon, BaseRADecFrame
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

    fig = plt.figure(1, figsize=(10, 7.5))
    cmap = cm.Blues
    cmap.set_under('w')
    HP.mollview(w_hp, cmap=cmap, title='', min=0, max=1, nest=True, fig=1)
    HP.graticule()
    for i_f, field in enumerate(['g02', 'g09', 'g12', 'g15']): 
        if field == 'g02': 
            ras = np.concatenate([np.tile(30.2, 100), np.linspace(30.2, 38.8, 100), np.tile(38.8, 100), np.linspace(30.2, 38.8, 100)])
            decs = np.concatenate([np.linspace(-6., -4., 100), np.tile(-6., 100), np.linspace(-6., -4., 100), np.tile(-4., 100)])
        elif field == 'g09': 
            ras = np.concatenate([np.tile(129., 100), np.linspace(129., 141., 100), np.tile(141., 100), np.linspace(129., 141, 100)])
            decs = np.concatenate([np.linspace(-2., 3., 100), np.tile(-2., 100), np.linspace(-2., 3., 100), np.tile(3., 100)])
        elif field == 'g12': 
            ras = np.concatenate([np.tile(174., 100), np.linspace(174., 186, 100), np.tile(186., 100), np.linspace(174., 186., 100)])
            decs = np.concatenate([np.linspace(-3., 2., 100), np.tile(-3., 100), np.linspace(-3., 2., 100), np.tile(2., 100)])
        elif field == 'g15': 
            ras = np.concatenate([np.tile(211.5, 100), np.linspace(211.5, 223.5, 100), np.tile(223.5, 100), np.linspace(211.5, 223.5, 100)])
            decs = np.concatenate([np.linspace(-2., 3., 100), np.tile(-2., 100), np.linspace(-2., 3., 100), np.tile(3., 100)])
        theta_gama = 0.5 * np.pi - np.deg2rad(decs) 
        phi_gama = np.deg2rad(ras) 
        #data = gama.Read(field, data_release=3, silent=True)
        #theta_gama = 0.5 * np.pi - np.deg2rad(data['photo']['dec']) 
        #phi_gama = np.deg2rad(data['photo']['ra'])
        print('GAMA theta: %f - %f' % (theta_gama.min(), theta_gama.max()))
        print('GAMA phi: %f - %f' % (phi_gama.min(), phi_gama.max()))
        HP.projscatter(theta_gama, phi_gama, color='C'+str(i_f), s=2, linewidth=0) 
        HP.projtext(250., 10.+10.*i_f, field.upper(), color='C'+str(i_f), 
                fontsize=20, lonlat=True) 
    HP.projtext(15., 38., 'DESI', color='navy', fontsize=20, lonlat=True) 
    fig.delaxes(fig.axes[1])
    fig.savefig(UT.doc_dir()+"figs/DESI_GAMA.png", bbox_inches='tight')
    return None


def GAMALegacy_Halpha_color(): 
    ''' color versus Halpha line flux relation for the GAMA-Legacy catalog
    for each of the GAMA DR3 regions. 
    '''
    bands = ['g', 'r', 'z'] 
    # read in GAMA-Legacy objects
    gamaleg = Cat.GamaLegacy() 
    
    fig = plt.figure(figsize=(12,6))
    for i_f, field in enumerate(['g09', 'g12', 'g15']): 
        gleg = gamaleg.Read(field, dr_gama=3, silent=True)

        # GAMA Halpha line flux:  
        gama_ha = gleg['gama-spec']['ha_flux'] 
        print('%i galaxies in GAMA+Legacy' % len(gama_ha))
        
        # legacy g,r,z model fluxes in nMgy
        legacy_photo = np.array([gleg['legacy-photo']['flux_'+band] for band in bands]) 
        #legacy_modelmag = np.array([UT.flux2mag(legacy_photo[i], bands=bands[i]) for i in range(len(bands))]) 
        legacy_modelmag = np.array([22.5 - 2.5*np.log10(legacy_photo[i]) for i in range(len(bands))]) 
        # legacy photometry color 
        legacy_gr = legacy_modelmag[0] - legacy_modelmag[1]
        legacy_rz = legacy_modelmag[1] - legacy_modelmag[2]
    
        no_ha = (gama_ha < 0.) 
        gama_ha[no_ha] = 10**(-1.9)
        print('%i galaxies with no Halpha' % np.sum(no_ha))
        
        sub1 = fig.add_subplot(121) # halpha vs (g - r)
        sub1.scatter(legacy_gr[::10], gama_ha[::10], s=0.5, c='C'+str(i_f)) 
    
        sub2 = fig.add_subplot(122) # halpha vs (r - z)
        sub2.scatter(legacy_rz[::10], gama_ha[::10], s=0.5, c='C'+str(i_f), 
                label=field.upper())
    
    sub1.set_xlabel(r'$(g - r)$', fontsize=25) 
    sub1.set_xlim([-0.5, 2.5])
    sub1.set_xticks([0., 1., 2.]) 
    sub1.set_ylabel(r'$H_\alpha$ line flux $[10^{-17}erg/s/cm^2]$', fontsize=20)
    sub1.set_yscale('log')
    sub1.set_ylim([1e-2, 1e4])

    sub2.set_xlabel(r'$(r - z)$ color Legacy DR5', fontsize=20) 
    sub2.set_xlim([-0.5, 1.5])
    sub2.set_xticks([0., 1.]) 
    sub2.set_yscale('log')
    sub2.set_ylim([1e-2, 1e4])
    sub2.set_yticklabels([]) 
    sub2.legend(loc='lower left', markerscale=20, handletextpad=0, prop={'size': 20}) 
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(UT.doc_dir()+"figs/GAMALegacy_Halpha_color.pdf", bbox_inches='tight')
    plt.close() 
    return None 


def BGStemplates(): 
    ''' compare the redshift distribution and M_0.1r vs ^0.1(g-r) relation of 
    BGS templates to the GAMA properties 
    '''
    # read in GAMA-Legacy objects
    gleg = Cat.GamaLegacy() 
    gleg_data = gleg.Read('g15', silent=True)

    bgs3 = FM.BGStree() 
    
    fig = plt.figure(figsize=(12,6))
    # redshift distribution of the templates 
    sub1 = fig.add_subplot(121) 
    _ = sub1.hist(gleg_data['gama-spec']['z'], bins=25, range=(0., 1.), histtype='stepfilled', label='GAMA DR3 G15')
    _ = sub1.hist(bgs3.meta['Z'], bins=25, range=(0., 1.), histtype='stepfilled', label='Templates')#, normed=True)
    sub1.legend(loc='upper right', prop={'size': 20}) 
    sub1.set_xlabel('Redshift', fontsize=20) 
    sub1.set_xlim([0., 0.8]) 
    
    # absolute magnitude of the templates
    Mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] 
    print('%i templates' % Mabs_temp.shape[0]) 
    # absolute magnitude of GAMA-Legacy objects
    Mabs = gleg.AbsMag(gleg_data, kcorr=0.1, H0=70, Om0=0.3) 
    print('%i GAMA G15 objects' % Mabs.shape[1]) 

    sub2 = fig.add_subplot(122)
    sub2.scatter(Mabs[2,:], Mabs[1,:] - Mabs[2,:], s=1) 
    sub2.scatter(Mabs_temp[:,2], Mabs_temp[:,1] - Mabs_temp[:,2], s=1) 
    sub2.set_xlabel(r'$M_{0.1r}$', fontsize=20)
    sub2.set_xlim([-14., -24.]) 
    sub2.set_ylabel(r'$^{0.1}(g - r)$', fontsize=20)
    sub2.set_ylim([-0.2, 1.3]) 
    fig.subplots_adjust(wspace=0.3) 
    fig.savefig(UT.doc_dir()+"figs/BGStemplates.pdf", bbox_inches='tight')
    plt.close() 
    return None


def GamaLegacy_matchSpectra(): 
    ''' match galaxies from the GAMA-Legacy catalog G15 field to BGS templates based on 
    their meta data and then plot their spectra
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15')

    redshift = gleg['gama-spec']['z']  # redshift
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 
    
    # BGS templates
    bgs3 = FM.BGStree() 
    bgstemp = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
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
    #gleg_meta = np.vstack([
    #    redshift[i_rand], 
    #    absmag_ugriz[2,i_rand], 
    #    absmag_ugriz[1,i_rand] - absmag_ugriz[2,i_rand]]).T
    #match, _ = bgs3.Query(gleg_meta)
    match = bgs3._GamaLegacy(gleg, index=i_rand)
    
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
    sub1.scatter(absmag_ugriz[2,:][::10], absmag_ugriz[1,:][::10] - absmag_ugriz[2,:][::10], c='k', s=0.1) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=60, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=60, edgecolors='k', marker='s', label='GAMA G15')
        if ii == 0: 
            sub1.legend(loc='upper left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot template spectra
        sub2.plot(wave, np.log10(flux[ii]), c='C'+str(ii), lw=0.5) 
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
    gleg = cata.Read('g15')

    redshift = gleg['gama-spec']['z'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    i_rand = np.random.choice(range(len(redshift)), size=1) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # generate emission line flux from GAMA data 
    emline_flux = s_bgs.EmissionLineFlux(gleg, index=i_rand, dr_gama=3, silent=True)
    # generate spectra with emission lines and template spectra that's 
    # flux calibrated to just GAMA (SDSS) photometry.
    flux, wave, _ = s_bgs.Spectra(
            gleg['gama-photo']['modelmag_r'][i_rand], 
            redshift[i_rand], 
            vdisp,
            seed=1, templateid=match, 
            emflux=None,# emline_flux, 
            mag_em=None,#gleg['gama-photo']['modelmag_r'][i_rand], 
            silent=False) 
    flux_eml, wave_eml, _ = s_bgs.Spectra(
            gleg['gama-photo']['modelmag_r'][i_rand], 
            redshift[i_rand], 
            vdisp,
            seed=1, templateid=match, 
            emflux=emline_flux, 
            mag_em=gleg['gama-photo']['modelmag_r'][i_rand], 
            silent=False) 

    emline_keys = [r'[OII]$_b$', r'[OII]$_r$', r'$\mathrm{H}_\beta$',  r'[OIII]$_b$', r'[OIII]$_r$', 
            r'[OI]$_b$', r'[OI]$_r$', r'[NII]$_b$', r'$\mathrm{H}_\alpha$', r'[NII]$_r$', r'[SII]$_b$', r'[SII]$_r$']
    emline_lambda = [3726., 3729., 4861., 4959., 5007., 6300., 6364., 6548., 6563., 6583., 6717., 6731.]
    emline_zlambda = (1.+redshift[i_rand]) * np.array(emline_lambda)

    #fig = plt.figure(figsize=(12,6))
    #sub2 = fig.add_subplot(111)
    #for ii, i in enumerate(i_rand): 
    #    # plot template spectra w/ emission lines
    #    sub2.scatter(wave, flux_eml[ii], c='C'+str(ii), s=1, label='Template w/ Em.Lines') 
    #    # plot template spectra
    #    sub2.plot(wave, flux0[ii], c='k', ls=':', lw=1, label='Template') 

    #for i_l, zlambda in enumerate(emline_zlambda): 
    #    sub2.vlines(zlambda, 0., 2.*flux_eml[0].max(), color='k', linestyle=':', linewidth=1) 
    #    sub2.text(zlambda, flux_eml[0].max()*float(20-i_l)/20., emline_keys[i_l], ha='left', va='top', fontsize=12) 
    #sub2.legend(loc='upper right', markerscale=10, prop={'size': 15})
    ##sub2.text(0.9, 0.9, 'Template Spectra', ha='right', va='center', transform=sub2.transAxes, fontsize=20)
    #sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    #sub2.set_xlim([3.5e3, 1e4]) 
    #sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    #sub2.set_ylim([0., 1.1*flux_eml[0].max()]) 
    #fig.subplots_adjust(wspace=0.3) 
    #fig.savefig(UT.doc_dir()+"figs/GLeg_EmLineSpectra.png", bbox_inches='tight')
    fig = plt.figure(figsize=(18,6))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1,3])
    sub1 = plt.subplot(gs[0]) #fig.add_subplot(121)
    sub2 = plt.subplot(gs[1]) #fig.add_subplot(122)
    sub1.scatter(absmag_ugriz[2,:][::10], absmag_ugriz[1,:][::10] - absmag_ugriz[2,:][::10], c='k', s=0.1) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, edgecolors='k', marker='s', label='GAMA G15')
        if ii == 0: 
            sub1.legend(loc='upper left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot template spectra w/ emission lines
        sub2.plot(wave_eml, flux_eml[ii], c='C'+str(ii), label='Template w/ Em.Lines') 
        # plot template spectra
        sub2.plot(wave, flux[ii], c='k', ls=':', lw=0.5, label='Template') 
        sub2.plot(s_bgs.basewave.astype(float)*(1.+redshift[i_rand]), 1e17*emline_flux.flatten(), 
                c='C1', lw=1, label='Emission Lines') 
    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylabel(r'$^{0.1}(g-r)$ color', fontsize=20) 
    sub1.set_ylim([-0.2, 1.6])
    sub1.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    for i_l, zlambda in enumerate(emline_zlambda): 
        sub2.vlines(zlambda, 0., 2.*flux_eml[0].max(), color='k', linestyle=':', linewidth=1) 
        sub2.text(zlambda, flux_eml[0].max()*float(28-i_l)/20., emline_keys[i_l], ha='left', va='top', fontsize=12) 
    sub2.legend(loc='upper right', prop={'size': 15})
    sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub2.set_xlim([3.5e3, 1e4]) 
    sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub2.set_ylim([0., 1.8*flux_eml[0].max()]) 
    fig.subplots_adjust(wspace=0.2) 
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


def skyBrightness(n_az=32, n_alt=18, method='pf', units='flux'): 
    ''' Sky brightness in (alt and az) of night sky

    method = 'pf' (parker's model), 'ks' Krisciunas & Schaefer
    '''
    # get moon at some night when moon altitutde ~0.3 and illumination ~ 0.7
    utc_time = Time(datetime(2019, 3, 25, 9, 0, 0)) 
    moon = get_moon(utc_time)

    # kpno
    kpno = EarthLocation.of_site('kitt peak')
    kpno_altaz = AltAz(obstime=utc_time, location=kpno)

    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    if units == 'ratio': 
        import desimodel.io
        import desisim.simexp
        params = desimodel.io.load_desiparams()
        wavemin = params['ccd']['b']['wavemin']
        wavemax = params['ccd']['z']['wavemax']
        waves = np.arange(wavemin, wavemax, 0.2) * u.angstrom
        config = desisim.simexp._specsim_config_for_wave((waves).to('Angstrom').value, specsim_config_file='desi')
        surface_brightness_dict = config.load_table(config.atmosphere.sky, 'surface_brightness', as_dict=True)
        norm_dark = np.average(surface_brightness_dict['dark'][(waves.value > 4000.) & (waves.value < 4500.)])
    else:
        norm_dark = 1.
    
    if method == 'pf': # parker's sky
        phi_grid, r_grid, totsky = _Isky_AltAz(n_az=n_az, n_alt=n_alt, key='I_cont', obstime=utc_time, location=kpno)
        totsky /= np.pi
    elif method == 'ks': 
        az_bins = np.linspace(0., 360., n_az+1)
        alt_bins = np.linspace(0., 90., n_alt+1)
        phi_grid, r_grid = np.meshgrid(az_bins/180.*np.pi, 90.-alt_bins)
        totsky = _Kris_AltAz(n_az=n_az, n_alt=n_alt, band='blue', moon=moon, obstime=utc_time, location=kpno)

    skymask = _sky_mask(n_az=n_az, n_alt=n_alt, moon=moon, obstime=utc_time, location=kpno)
    totsky[~skymask] = np.NaN

    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    if units == 'flux': 
        c = sub.pcolormesh(phi_grid, r_grid, totsky / norm_dark, cmap='viridis', vmin=0., vmax=20.)
    elif units == 'ratio': 
        c = sub.pcolormesh(phi_grid, r_grid, totsky / norm_dark, cmap='viridis', vmin=1., vmax=10.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=250, label='Moon')
    sub.annotate(r'moon',
            xy=(moon_altaz.az.deg/180.*np.pi, 90.-moon_altaz.alt.deg),  # theta, radius
            xytext=(moon_altaz.az.deg/180.*np.pi, 90.-moon_altaz.alt.deg),  # theta, radius
            fontsize=20, horizontalalignment='center', verticalalignment='bottom')
    sub.text(0., -0.05, 'moon illumination: 0.7', 
            ha='left', va='center', transform=sub.transAxes, fontsize=20)
    sub.text(0., -0.1, 'moon altitude: '+str(round(moon_alt,1)),
            ha='left', va='center', transform=sub.transAxes, fontsize=20)
    cbar = plt.colorbar(c)
    if units == 'flux': 
        cbar.set_label('sky brightness [$10^{-17} erg/cm^{2}/s/\AA/arcsec^2$]', rotation=270, labelpad=30, fontsize=20)
    elif units == 'ratio': 
        cbar.set_label('(sky brightness) / (UVES dark sky)', rotation=270, labelpad=30, fontsize=20)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 70.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, ''])#, '', 0])
    sub.grid(True, which='major')
    if method == 'pf': 
        sub.set_title(r"Sky Model at $4000 \AA$ (from BOSS sky fibers)", fontsize=25) 
    elif method == 'ks': 
        sub.set_title(r"Sky Model at $4000 \AA$ (Krisciunas $\&$ Schaefer 1991)", fontsize=25) 
    fig.savefig(UT.doc_dir()+"figs/skyBrightness."+method+"."+units+".pdf", bbox_inches='tight')
    return None

def _Kris_AltAz(n_az=8, n_alt=3, band='blue', moon=None, obstime=None, location=None): 
    import desimodel.io
    import desisim.simexp
    params = desimodel.io.load_desiparams() 
    wavemin = params['ccd']['b']['wavemin']
    wavemax = params['ccd']['z']['wavemax']
    if band == 'blue': 
        wmin, wmax = 4000., 4500. 
    waves = np.arange(wavemin, wavemax, 0.2) * u.angstrom
    wlim = ((waves > wmin*u.angstrom) & (waves < wmax*u.angstrom)) 
    config = desisim.simexp._specsim_config_for_wave((waves).to('Angstrom').value, specsim_config_file='desi')
    desi = FM.SimulatorHacked(config, num_fibers=1, camera_output=True)

    extinction_coefficient = config.load_table(config.atmosphere.extinction, 'extinction_coefficient')
    #surface_brightness_dict = config.load_table(config.atmosphere.sky, 'surface_brightness', as_dict=True)
    # kpno
    kpno_altaz = AltAz(obstime=obstime, location=location)
    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    sun = get_sun(obstime) 
            
    elongation = sun.separation(moon)
    phase = np.arctan2(sun.distance * np.sin(elongation),
            moon.distance - sun.distance*np.cos(elongation))
    desi.atmosphere.moon.moon_phase = phase.value/np.pi #moon_phase/np.pi #np.arccos(2*moonfrac-1)/np.pi
    desi.atmosphere.moon.moon_zenith = (90. - moon_alt) * u.deg
    
    #altitude and azimuth bins 
    az_bins = np.linspace(0., 360., n_az+1)
    alt_bins = np.linspace(0., 90., n_alt+1)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
   
    Bmoons = np.zeros(az_grid.shape)
    for i in range(az_grid.shape[0]): 
        for j in range(az_grid.shape[1]): 
            sky_aa = AltAz(az=az_grid[i,j]*u.deg, alt=alt_grid[i,j]*u.deg, obstime=obstime, location=location)
            sky = SkyCoord(sky_aa)
            sep = moon.separation(sky).deg 
            
            desi.atmosphere.airmass = sky_aa.secz 
            desi.atmosphere.moon.separation_angle = sep * u.deg
            Bmoon = desi.atmosphere.moon.surface_brightness.value * 1e17
            Bmoons[i,j] = np.average(Bmoon[wlim]) 
    return Bmoons 


def _Isky_AltAz(n_az=8, n_alt=3, key='Icont', obstime=None, location=None, overwrite=False):
    # altitude and azimuth bins 
    az_bins = np.linspace(0., 360., n_az+1)
    alt_bins = np.linspace(0., 90., n_alt+1)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
    
    f = ''.join([UT.dat_dir(), 'Isky.altazgrid.az', str(n_az), '.alt', str(n_alt), '.p']) 
    if os.path.isfile(f) and not overwrite: 
        phi_grid, r_grid, totsky = pickle.load(open(f, 'rb'))
    else: 
        # binning for the plot  
        phi_grid, r_grid = np.meshgrid(az_bins/180.*np.pi, 90.-alt_bins)

        # calculate sky brightness
        keys = ['I_cont', 'I_airmass', 'I_zodiacal', 'I_isl', 'I_solar_flux', 'I_seashour', 'dT', 'I_twilight', 
                'I_moon', 'I_moon_noexp', 'I_add']
        totsky = {} 
        for k in keys: 
            totsky[k] = np.zeros(az_grid.shape)
        for i in range(az_grid.shape[0]): 
            for j in range(az_grid.shape[1]): 
                Isky = skyflux_onthesky(alt_grid[i,j], az_grid[i,j], band='blue', obstime=obstime, location=location)
                for k in keys: 
                    totsky[k][i,j] = Isky[k]
        pickle.dump([phi_grid, r_grid, totsky], open(f, 'wb'))
    return phi_grid, r_grid, totsky[key] 


def _skyflux(alt, az, band='blue', obstime=None, location=None): 
    ''' return the sky flux on a given point (alt, az) on the sky  
    '''
    if band == 'blue': 
        wmin, wmax = 4000., 4500. 

    sky_aa = AltAz(az=az*u.deg, alt=alt*u.deg, obstime=obstime, location=location) #utc_time, location=kpno)
    sky = SkyCoord(sky_aa)
    pt = Sky.skySpec(sky.icrs.ra.deg, sky.icrs.dec.deg, obstime)

    w, Icont = pt.get_Icontinuum()
    wlim = ((w > wmin) & (w < wmax))
    out = {} 
    out['I_cont'] = np.average(Icont[wlim]) 
    out['I_airmass'] = np.average(pt._Iairmass[wlim]) 
    out['I_zodiacal'] = np.average(pt._Izodiacal[wlim]) 
    out['I_isl'] = np.average(pt._Iisl[wlim]) 
    out['I_solar_flux'] = np.average(pt._Isolar_flux[wlim]) 
    out['I_seashour'] = np.average(pt._Iseasonal[wlim] + pt._Ihourly[wlim]) 
    out['dT'] = np.average(pt._dT[wlim]) 
    out['I_twilight'] = np.average(pt._Itwilight[wlim]) 
    out['I_moon'] = np.average(pt._Imoon[wlim]) 
    out['I_moon_noexp'] = np.average((pt._Imoon * np.exp(pt.coeffs['m6'] * pt.X))[wlim]) 
    out['I_add'] = np.average(pt._Iadd_continuum[wlim]) 
    return out  


def _sky_mask(n_az=8, n_alt=3, moon=None, obstime=None, location=None): 
    # a mask in alt az grid for sky conditions where Parker's model is not well calibrated.
    # < 30 deg separation; > 30 deg altitude (airmass < 2)  

    # altitude and azimuth bins 
    az_bins = np.linspace(0., 360., n_az+1)
    alt_bins = np.linspace(0., 90., n_alt+1)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
    
    # apply some mask 
    totsky = np.ones(az_grid.shape).astype(bool) 
    for i in range(az_grid.shape[0]): 
        for j in range(az_grid.shape[1]): 
            sky_aa = AltAz(az=az_grid[i,j]*u.deg, alt=alt_grid[i,j]*u.deg, obstime=obstime, location=location)
            sky = SkyCoord(sky_aa)
            sep = moon.separation(sky).deg 
            if sep < 30. or alt_grid[i,j] < 30.: 
                totsky[i,j] = False 
    return totsky

def rMag_normalize(): 
    ''' Compare the normalization of the template spectra using model r-magnitude 
    and apflux derived magnitude
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15')

    redshift = gleg['gama-spec']['z'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    i_rand = np.random.choice(range(len(redshift)), size=1) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 
    
    # using the template generate source spectra *without* emission lines 
    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # normalize by gama model r-band magnitude
    flux_gama, wave_gama, _ = s_bgs.Spectra(
            gleg['gama-photo']['modelmag_r'][i_rand], redshift[i_rand], vdisp, 
            seed=1, templateid=match, emflux=None, mag_em=None) 

    # derive r-band aperture magnitude from Legacy photometry 
    r_apflux = gleg['legacy-photo']['apflux_r'][:,1]  # nanomaggies
    r_mag = UT.flux2mag(r_apflux, method='log')  # convert to mag 
    flux_eml, wave_eml, _ = s_bgs.Spectra(
            r_mag[i_rand], redshift[i_rand], vdisp, seed=1, templateid=match, 
            emflux=None, mag_em=None, silent=False) 
    print('model magnitude: %f' % gleg['gama-photo']['modelmag_r'][i_rand]) 
    print('r aperture magnitude 0.75 arcsec: %f' % r_mag[i_rand]) 

    fig = plt.figure(figsize=(18,6))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1,3])
    sub1 = plt.subplot(gs[0]) #fig.add_subplot(121)
    sub2 = plt.subplot(gs[1]) #fig.add_subplot(122)
    sub1.scatter(absmag_ugriz[2,:][::10], absmag_ugriz[1,:][::10] - absmag_ugriz[2,:][::10], c='k', s=1) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, edgecolors='k', marker='s', label='GAMA object')
        if ii == 0: 
            sub1.legend(loc='upper left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot template spectra w/ emission lines
        sub2.plot(wave_gama, flux_gama[ii], c='C'+str(ii), label='GAMA model $r$-band mag') 
        # plot template spectra
        sub2.plot(wave_eml, flux_eml[ii], c='k', ls=':', lw=0.5, label="Legacy $r$-band $0.75''$ apflux") 
    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylabel(r'$^{0.1}(g-r)$ color', fontsize=20) 
    sub1.set_ylim([-0.2, 1.6])
    sub1.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    sub2.legend(loc='upper right', prop={'size': 15})
    sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub2.set_xlim([3.5e3, 1e4]) 
    sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub2.set_ylim([0., 1.25*flux_gama[0].max()]) 
    fig.subplots_adjust(wspace=0.2) 
    fig.savefig(UT.doc_dir()+"figs/GLeg_rMag_norm.pdf", bbox_inches='tight')
    plt.close() 
    return None


def rMag_normalize_emline(): 
    ''' Compare the normalization of the template spectra using model r-magnitude 
    and apflux derived magnitude with emission lines. The emission lines are included 
    in the normalization
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15')

    redshift = gleg['gama-spec']['z'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    i_rand = np.random.choice(range(len(redshift)), size=1) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 
    
    # using the template generate source spectra *without* emission lines 
    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # emission line flux 
    emline_flux = s_bgs.EmissionLineFlux(gleg, index=i_rand, dr_gama=3, silent=True)
    print 'emline flux max ', emline_flux.max()
    # normalize by gama model r-band magnitude
    flux_gama, wave_gama, _ = s_bgs.Spectra(
            gleg['gama-photo']['modelmag_r'][i_rand], redshift[i_rand], vdisp, 
            seed=1, templateid=match, 
            emflux=emline_flux, mag_em=gleg['gama-photo']['modelmag_r'][i_rand]) 

    emline_flux = s_bgs.EmissionLineFlux(gleg, index=i_rand, dr_gama=3, silent=True)
    print 'emline flux max ', emline_flux.max()
    # derive r-band aperture magnitude from Legacy photometry 
    r_apflux = gleg['legacy-photo']['apflux_r'][:,1]  # nanomaggies
    r_mag = UT.flux2mag(r_apflux, method='log')  # convert to mag 
    flux_eml, wave_eml, _ = s_bgs.Spectra(
            r_mag[i_rand], redshift[i_rand], vdisp, seed=1, templateid=match, 
            emflux=emline_flux, mag_em=gleg['gama-photo']['modelmag_r'][i_rand])
    print('model magnitude: %f' % gleg['gama-photo']['modelmag_r'][i_rand]) 
    print('r aperture magnitude 0.75 arcsec: %f' % r_mag[i_rand]) 
    
    emline_keys = [r'[OII]$_b$', r'[OII]$_r$', r'$\mathrm{H}_\beta$',  
            r'[OIII]$_b$', r'[OIII]$_r$', r'[OI]$_b$', r'[OI]$_r$', 
            r'[NII]$_b$', r'$\mathrm{H}_\alpha$', r'[NII]$_r$', 
            r'[SII]$_b$', r'[SII]$_r$']
    emline_lambda = [3726., 3729., 4861., 4959., 5007., 6300., 6364., 6548., 6563., 6583., 6717., 6731.]
    emline_zlambda = (1.+redshift[i_rand]) * np.array(emline_lambda)


    fig = plt.figure(figsize=(18,6))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1,3])
    sub1 = plt.subplot(gs[0]) #fig.add_subplot(121)
    sub2 = plt.subplot(gs[1]) #fig.add_subplot(122)
    sub1.scatter(absmag_ugriz[2,:][::10], absmag_ugriz[1,:][::10] - absmag_ugriz[2,:][::10], 
            c='k', s=1) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, edgecolors='k', marker='s', label='GAMA object')
        if ii == 0: 
            sub1.legend(loc='upper left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot template spectra w/ emission lines
        sub2.plot(wave_gama, flux_gama[ii], c='C'+str(ii), label='GAMA model $r$-band mag') 
        # plot template spectra
        sub2.plot(wave_eml, flux_eml[ii], c='k', lw=0.5, label="Legacy $r$-band $0.75''$ apflux") 
    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylabel(r'$^{0.1}(g-r)$ color', fontsize=20) 
    sub1.set_ylim([-0.2, 1.6])
    sub1.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    for i_l, zlambda in enumerate(emline_zlambda): 
        sub2.vlines(zlambda, 0., 2.*flux_gama[0].max(), color='k', linestyle=':', linewidth=0.5)
        sub2.text(zlambda, flux_gama[0].max()*float(28-i_l)/20., emline_keys[i_l], ha='left', 
                va='top', fontsize=12) 

    sub2.legend(loc='upper right', prop={'size': 15})
    sub2.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub2.set_xlim([3.5e3, 1e4]) 
    sub2.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub2.set_ylim([0., 2.*flux_gama[0].max()]) 
    fig.subplots_adjust(wspace=0.2) 
    fig.savefig(UT.doc_dir()+"figs/GLeg_rMag_norm_emline.pdf", bbox_inches='tight')
    plt.close() 
    return None


def expSpectra():
    ''' exposured spectra of DESI spectrograph simulated using dark and bright sky models. 
    Source flux is some templates with emission line data from GAMA DR2. 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15')

    redshift = gleg['gama-spec']['z'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 
    
    # pick a random galaxy from the GAMA-legacy sample
    #i_rand = np.random.choice(range(len(redshift)), size=1) 
    i_rand = np.array([38164])
    print('i_rand = %i' % i_rand[0]) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # r-band modelmag magnitude from GAMA photometry
    r_mag_gama = gleg['gama-photo']['modelmag_r'][i_rand]
    # r-band aperture magnitude from Legacy photometry 
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][i_rand,1], method='log')  
    assert np.isfinite(r_mag_apflux)
    print('r_mag = %f' % r_mag_apflux)
    
    emline_flux = s_bgs.EmissionLineFlux(gleg, index=i_rand, dr_gama=3, silent=True)

    flux_eml, wave, _ = s_bgs.Spectra(r_mag_apflux, redshift[i_rand], vdisp,
            seed=1, templateid=match, emflux=emline_flux, mag_em=r_mag_gama) 

    # simulate exposure using 
    fdesi = FM.fakeDESIspec() 
    bgs_spectra_bright = fdesi.simExposure(wave, flux_eml, skycondition='bright') 
    bgs_spectra_dark = fdesi.simExposure(wave, flux_eml, skycondition='dark') 
    # write out simulated spectra
    for b in ['b', 'r', 'z']: 
        tbl_bright = Table([bgs_spectra_bright.wave[b], bgs_spectra_bright.flux[b][0].flatten()], names=('lambda', 'flux')) 
        tbl_bright.write(UT.dat_dir()+'spectra/obj'+str(i_rand[0])+'_brightsky_'+b+'.fits', format='fits', overwrite=True) 
        tbl_dark = Table([bgs_spectra_dark.wave[b], bgs_spectra_dark.flux[b][0].flatten()], names=('lambda', 'flux'))
        tbl_dark.write(UT.dat_dir()+'spectra/obj'+str(i_rand[0])+'darksky_'+b+'.fits', format='fits', overwrite=True) 
    
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
    #sub2.plot(wave, flux[0], c='k', lw=0.3, ls=':', label='Template')
    
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
    sub2.set_ylim([0., 1.5*flux_eml[0].max()]) 
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
    gleg = cata.Read('g15')

    redshift = gleg['gama-spec']['z'] # redshift 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # pick a random galaxy from the GAMA-legacy sample
    #i_rand = np.random.choice(range(len(redshift)), size=1) 
    i_rand = np.array([38164])
    print('i_rand = %i' % i_rand[0]) 

    # match random galaxy to BGS templates
    bgs3 = FM.BGStree() 
    match = bgs3._GamaLegacy(gleg, index=i_rand) 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
    vdisp = np.repeat(100.0, len(i_rand)) # velocity dispersions [km/s]
    
    # r-band aperture magnitude from Legacy photometry 
    r_mag_gama = gleg['gama-photo']['modelmag_r'][i_rand]
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][i_rand,1], method='log')  
    print('GAMA r_mag = %f' % r_mag_gama)
    print('apflux r_mag = %f' % r_mag_apflux)

    emline_flux = s_bgs.EmissionLineFlux(gleg, index=i_rand, dr_gama=3, silent=True)

    flux_eml, wave, _ = s_bgs.Spectra(r_mag_apflux, redshift[i_rand], vdisp, seed=1, 
            templateid=match, emflux=emline_flux, mag_em=r_mag_gama, silent=False) 

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
                c='C1', lw=0.5, label=lbl1) 
        sub.plot(bgs_spectra_dark.wave[b], bgs_spectra_dark.flux[b][0].flatten(), 
                c='C0', lw=0.5, label=lbl0) 
    
    emline_keys = ['oiib', 'oiir', 'hb',  'oiiib', 'oiiir', 'oib', 'oir', 'niib', 
            'ha', 'niir', 'siib', 'siir']
    emline_labels = [r'[OII]$_b$', r'[OII]$_r$', r'$\mathrm{H}_\beta$',  
            r'[OIII]$_b$', r'[OIII]$_r$', r'[OI]$_b$', r'[OI]$_r$', 
            r'[NII]$_b$', r'$\mathrm{H}_\alpha$', r'[NII]$_r$', 
            r'[SII]$_b$', r'[SII]$_r$']
    emline_lambda = [3726., 3729., 4861., 4959., 5007., 6300., 6364., 6548., 6563., 6583., 6717., 6731.]
    emline_zlambda = (1.+redshift[i_rand][0]) * np.array(emline_lambda)
    
    for i_l, zlambda in enumerate(emline_zlambda): 
        # mark the redshifted wavelength of the emission line 
        sub.vlines(zlambda, 0., 2.*flux_eml[0].max(), color='k', linestyle=':', linewidth=0.5) 
        sub.text(1.05*zlambda, flux_eml[0].max()*float(15-i_l)/11., emline_labels[i_l],
                ha='left', va='top', fontsize=12) 

        # lineflux of the emissionline 
        emlineflux = gleg['gama-spec'][emline_keys[i_l]+'_flux'][i_rand][0]
        # width of emline 
        emlinesig = gleg['gama-spec']['sig_'+emline_keys[i_l]][i_rand][0]
        if (emlineflux == -99.) or (emlinesig <= 0.): continue 
        A = emlineflux/np.sqrt(2.*np.pi*emlinesig**2)

        f_eml = lambda ww: A*np.exp(-0.5*(ww-zlambda)**2/emlinesig**2)

        #sub.plot(wave, f_eml(wave), c='k', linestyle=':', linewidth=2)
        assert np.abs(emlineflux - integ.simps(f_eml(wave), x=wave)) < 0.01

    sub.text(0.95, 0.9, r'$z_\mathrm{GAMA} = '+str(redshift[i_rand[0]])+'$', 
            ha='right', va='center', transform=sub.transAxes, fontsize=20)

    sub.set_xlabel('Wavelength [$\AA$] ', fontsize=20) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylabel('$f(\lambda)\,\,[10^{-17}erg/s/cm^2/\AA]$', fontsize=20) 
    sub.set_ylim([0., 1.5*flux_eml[0].max()]) 
    #sub.legend(loc='upper left', prop={'size': 15}) 
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
        sub.text(0.9, 0.85, emline_labels[i_l]+'\n $\sigma='+str(gleg['gama-spec']['sig_'+emline_keys[i_l]][i_rand][0])+'$', 
                ha='right', va='center', transform=sub.transAxes, fontsize=20)

        # lineflux of the emissionline 
        emlineflux = gleg['gama-spec'][emline_keys[i_l]+'_flux'][i_rand][0]
        # width of emline 
        emlinesig = gleg['gama-spec']['sig_'+emline_keys[i_l]][i_rand][0]
        if (emlineflux == -99.) or (emlinesig <= 0.): continue 
        A = emlineflux/np.sqrt(2.*np.pi*emlinesig**2)

        f_eml = lambda ww: A*np.exp(-0.5*(ww-zlambda)**2/emlinesig**2)

        sub.plot(wave, f_eml(wave), c='k', linestyle=':', linewidth=1)
        sub.set_xlim([zlambda-50., zlambda+50.]) 
        sub.set_ylim([0., 3*flux_eml[0].max()]) 

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
    gleg = cata.Read('g15')

    redshift = gleg['gama-spec']['z'] # redshift 
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


def expSpectra_redrock(exptime=300): 
    ''' Redrock redshift success rate of simulated exposures of 
    simulated spectra with dark versus bright sky of 1000 randomly 
    selected galaxies.
    
    Creates a three panel plot: Halpha line flux vs g-r legacy color, z_redrock vs z_true, 
    and redshift success rate vs apflux r-mag  
    '''
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15')
    redshift = gleg['gama-spec']['z']
    ngal = len(redshift)
    iblock = 5
    print('%i galaxies in the GAMA-Legacy G15 region' % ngal)

    # apparent magnitude from Legacy photometry aperture flux
    g_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_g'][:,1])
    r_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    # H-alpha line flux from GAMA spectroscopy
    gama_ha = gleg['gama-spec']['ha_flux']

    fzdata = lambda s, et: ''.join([UT.dat_dir(), 'redrock/', 
        'GamaLegacy.g15.expSpectra.', s, 'sky.seed1.exptime', str(exptime), '.', str(iblock), 'of64blocks.redrock.fits']) 
    findex = lambda s, et: ''.join([UT.dat_dir(), 'spectra/', 
        'GamaLegacy.g15.expSpectra.', s, 'sky.seed1.exptime', str(exptime), '.', str(iblock), 'of64blocks.index']) 
    
    # read in redrock redshifts
    zdark_data = fits.open(fzdata('dark', exptime))[1].data
    i_dark = np.loadtxt(findex('dark', exptime), unpack=True, usecols=[0], dtype='i')

    zbright_data = fits.open(fzdata('bright', exptime))[1].data
    i_bright = np.loadtxt(findex('bright', exptime), unpack=True, usecols=[0], dtype='i')
    assert np.array_equal(i_dark, i_bright)
    print('%i redshifts' % len(zdark_data['Z']))
    print('%i spectra w/ dark sky have ZWARN != 0' % np.sum(zdark_data['ZWARN'] != 0))
    print('%i spectra w/ bright sky have ZWARN != 0' % np.sum(zbright_data['ZWARN'] != 0))
    
    # calculate delta z / (1+z)  for dark and brigth skies 
    dz_1pz_dark = (zdark_data['Z'] - redshift[i_dark])/(1.+redshift[i_dark])
    dz_1pz_bright = (zbright_data['Z'] - redshift[i_dark])/(1.+redshift[i_dark])

    fig = plt.figure(figsize=(15, 4))
    # panel 1 Halpha line flux versus g-r apflux legacy color 
    sub = fig.add_subplot(131)
    hasha = (gama_ha > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[hasha], gama_ha[hasha], s=1, c='k')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[~hasha], np.repeat(1e-2, np.sum(~hasha)), s=1, c='k')
    hasha = (gama_ha[i_dark] > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], gama_ha[i_dark][hasha], c='C1', s=0.5)
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha], np.repeat(1e-2, np.sum(~hasha)), c='C1', s=0.5)
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylabel(r'$H_\alpha$ line flux GAMA DR3 $[10^{-17}erg/s/cm^2]$', fontsize=15)
    sub.set_ylim([5e-3, 2e4])
    sub.set_yscale('log')

    # panel 2 z_redrock vs r_true
    sub = fig.add_subplot(132)
    sub.scatter(redshift[i_dark], dz_1pz_bright, c='C1', s=10)
    sub.scatter(redshift[i_dark], dz_1pz_dark, c='C0', s=2)
    sub.set_xlabel(r"$z_\mathrm{true}$ GAMA", fontsize=25)
    sub.set_xlim([0., 0.36])
    sub.set_ylabel(r"$\Delta z / (1+z_\mathrm{true})$", fontsize=20)
    sub.set_ylim([-0.05, 1.])

    # panel 3 
    sub = fig.add_subplot(133)
    mm_dark, e1_dark, ee1_dark = _z_successrate(r_mag_legacy_apflux[i_dark], #range=(18, 22), 
            condition=((zdark_data['ZWARN'] == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(r_mag_legacy_apflux[i_dark], #range=(18, 22), 
            condition=((zbright_data['ZWARN'] == 0) & (dz_1pz_bright < 0.003)))
    sub = fig.add_subplot(133)
    sub.plot([17., 22.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$r_\mathrm{apflux}$ magnitude', fontsize=20)
    sub.set_xlim([17.5, 22.])
    sub.set_ylabel(r'fraction of $\Delta z /(1+z_\mathrm{GAMA}) < 0.003$', fontsize=20)
    sub.set_ylim([0., 1.2])
    sub.legend(loc='lower left', handletextpad=0., prop={'size': 20})
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_g15_expSpectra_redrock_exptime"+str(exptime)+".pdf", 
        bbox_inches='tight')
    plt.close() 
    return None


def expSpectra_blocks_zsuccess(field, iblocks=30, exptime=300): 
    ''' Redrock redshift success rate of simulated exposures of 
    simulated spectra with dark versus bright sky of 1000 randomly 
    selected galaxies.
    
    Creates a three panel plot: Halpha line flux vs g-r legacy color, z_redrock vs z_true, 
    and redshift success rate vs apflux r-mag  
    '''
    if field == 'g15': nblocks = 64
    else: raise ValueError
    cata = Cat.GamaLegacy()
    gleg = cata.Read(field)
    redshift = gleg['gama-spec']['z']
    ngal = len(redshift)
    print('%i galaxies in the GAMA-Legacy G15 region' % ngal)
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # apparent magnitude from Legacy photometry aperture flux
    r_mag_legacy = UT.flux2mag(gleg['legacy-photo']['flux_r'])
    g_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_g'][:,1])
    r_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    # H-alpha line flux from GAMA spectroscopy
    gama_ha = gleg['gama-spec']['ha_flux']
  
    i_dark, i_bright, zbest_d, zbest_b, zwarn_d, zwarn_b = _read_expSpectra_blocks(field, iblocks=iblocks, exptime=exptime)
    print('%i redshifts' % len(zbest_d))
    print('%i spectra w/ dark sky have ZWARN != 0' % np.sum(zwarn_d != 0))
    print('%i spectra w/ bright sky have ZWARN != 0' % np.sum(zwarn_b != 0))
    
    # calculate delta z / (1+z)  for dark and brigth skies 
    dz_1pz_dark = np.abs(zbest_d - redshift[i_dark])/(1.+redshift[i_dark])
    dz_1pz_bright = np.abs(zbest_b - redshift[i_dark])/(1.+redshift[i_dark])

    fig = plt.figure(figsize=(5,4))
    sub = fig.add_subplot(111)
    rmag = r_mag_legacy
    print rmag.min(), rmag.max()
    rmag_range = [15, 22]
    mm_dark, e1_dark, ee1_dark = _z_successrate(rmag[i_dark], range=rmag_range, 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(rmag[i_dark], range=rmag_range, 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([15., 25.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$r$ magnitude Legacy', fontsize=20)
    sub.set_xlim(rmag_range)
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])
    sub.legend(loc='lower left', handletextpad=0., prop={'size': 20})
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_rmag.png", 
        bbox_inches='tight')
    plt.close() 
    
    fig = plt.figure(figsize=(20, 4))
    # panel 1 Halpha line flux versus g-r apflux legacy color 
    sub = fig.add_subplot(141)
    hasha = (gama_ha > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[hasha], gama_ha[hasha], 
            s=1, c='k', label='GAMA '+field.upper()+' $\cup$ Legacy DR5')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[~hasha], np.repeat(1e-2, np.sum(~hasha)), s=1, c='k')
    hasha = (gama_ha[i_dark] > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], gama_ha[i_dark][hasha], 
            c='C0', s=0.5, alpha=0.33, label=r'$\sim 30,000$ galaxies')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha], np.repeat(1e-2, np.sum(~hasha)), 
            c='C0', s=0.5, alpha=0.33)
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylabel(r'$H_\alpha$ line flux GAMA $[10^{-17}erg/s/cm^2]$', fontsize=17)
    sub.set_ylim([5e-3, 2e4])
    sub.set_yscale('log')
    sub.legend(loc='lower left', bbox_to_anchor=(0.0, 0.05), markerscale=10, 
            handletextpad=0, frameon=True, fancybox=True, framealpha=0.8, prop={'size':15}) 
    # panel 2 z_redrock vs r_true
    sub = fig.add_subplot(142)
    sub.scatter(redshift[i_dark], dz_1pz_bright, c='C1', s=10)
    sub.scatter(redshift[i_dark], dz_1pz_dark, c='C0', s=2)
    sub.set_xlabel(r"$z_\mathrm{true}$ GAMA", fontsize=25)
    sub.set_xlim([0., 0.36])
    sub.set_ylabel(r"$\Delta z / (1+z_\mathrm{true})$", fontsize=20)
    sub.set_ylim([-0.05, 1.])
    print('%i spectra w/ dark sky fail' % np.sum(~((zwarn_d == 0) & (dz_1pz_dark < 0.003))))
    print('%i spectra w/ bright sky fail' % np.sum(~((zwarn_b == 0) & (dz_1pz_bright < 0.003))))
    # panel 3 
    sub = fig.add_subplot(143)
    rmag = r_mag_legacy_apflux
    rmag_range = [16, 23]
    mm_dark, e1_dark, ee1_dark = _z_successrate(rmag[i_dark], range=rmag_range, 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(rmag[i_dark], range=rmag_range, 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([15., 25.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$r_\mathrm{ap.flux}$ magnitude', fontsize=20)
    sub.set_xlim(rmag_range)
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])
    # panel 4 
    sub = fig.add_subplot(144)
    prop = np.log10(gama_ha)
    prop[gama_ha <= 0.] = -2
    prop_range = [-2., 4.]
    mm_dark, e1_dark, ee1_dark = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([prop.min()-0.5*np.abs(prop.min()), 1.5*prop.max()], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$\log \mathrm{H}\alpha$ line flux', fontsize=20)
    sub.set_xlim(prop_range) 
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])
    sub.legend(loc='lower left', handletextpad=0., prop={'size': 20})
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess.png", 
        bbox_inches='tight')
    plt.close() 

    # --- H-alpha vs (g-r) color ---
    fig = plt.figure(figsize=(16,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(131)
    hasha = (gama_ha[i_dark] > 0.) 
    gr = np.concatenate([(g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], (g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha]])
    ha = np.concatenate([gama_ha[i_dark][hasha], np.repeat(1e-2, np.sum(~hasha))])
    w = np.ones(len(ha))
    hb = sub.hexbin(gr, ha, C=w, reduce_C_function=np.sum, vmax=1000., mincnt=20, gridsize=50, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'$N_\mathrm{gal}$', fontsize=20)
    sub.text(0.1, 0.1, r'$N_{bin} > 20$', ha='left', va='bottom', 
            transform=sub.transAxes, fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    # --- H-alpha vs (g-r) color : Dark Sky ---
    sub = fig.add_subplot(132)
    hasha = (gama_ha[i_dark] > 0.) 
    dz_1pz_dark[zwarn_d != 0] = 1.
    dz_1pz_d = np.concatenate([dz_1pz_dark[hasha], dz_1pz_dark[~hasha]]) 
    w_dz_1pz_d = np.zeros(len(dz_1pz_d))
    w_dz_1pz_d[dz_1pz_d < 0.003] = 1. 
    hb = sub.hexbin(gr, ha, C=w_dz_1pz_d, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    sub.set_title('w/ Dark Sky', fontsize=20) 
    # --- H-alpha vs (g-r) color : Dark Sky ---
    sub = fig.add_subplot(133)
    hasha = (gama_ha[i_dark] > 0.) 
    dz_1pz_bright[zwarn_b != 0] = 1.
    dz_1pz_b = np.concatenate([dz_1pz_bright[hasha], dz_1pz_bright[~hasha]]) 
    w_dz_1pz_b = np.zeros(len(dz_1pz_b))
    w_dz_1pz_b[dz_1pz_b < 0.003] = 1. 
    hb = sub.hexbin(gr, ha, C=w_dz_1pz_b, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    sub.set_title('w/ Bright Sky', fontsize=20) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$(g-r)$ from Legacy ap. flux', labelpad=7, fontsize=20)
    bkgd.set_ylabel(r'$H_\alpha$ line flux GAMA $[10^{-17}erg/s/cm^2]$', labelpad=7, fontsize=20)

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_HA_gr.png", 
        bbox_inches='tight')
    plt.close() 
    
    # M_r vs (g-r) color
    fig = plt.figure(figsize=(16,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(131)
    w = np.ones(len(i_dark))
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w, reduce_C_function=np.sum, vmax=1000., mincnt=20., gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'$N_\mathrm{gal}$', fontsize=20)
    sub.text(0.1, 0.1, r'$N_{bin} > 20$', ha='left', va='bottom', 
            transform=sub.transAxes, fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    sub = fig.add_subplot(132)
    dz_1pz_dark[zwarn_d != 0] = 1.
    w_dz_1pz_dark = np.zeros(len(dz_1pz_dark))
    w_dz_1pz_dark[dz_1pz_dark < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_dark, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Dark Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    
    sub = fig.add_subplot(133)
    dz_1pz_bright[zwarn_b != 0] = 1.
    w_dz_1pz_bright = np.zeros(len(dz_1pz_bright))
    w_dz_1pz_bright[dz_1pz_bright < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_bright, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Bright Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'GAMA $M_{0.1r}$', labelpad=5, fontsize=20)
    bkgd.set_ylabel(r'GAMA $^{0.1}(g-r)$ color', labelpad=7, fontsize=20)

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_Mr_gr.png", 
        bbox_inches='tight')
    plt.close() 
    return None


def expSpectra_blocks_faintemline_zsuccess(field, iblocks=30, exptime=480): 
    ''' Redrock redshift success rate of simulated exposures of 
    simulated spectra with dark versus bright sky of 1000 randomly 
    selected galaxies.
    
    Creates a three panel plot: Halpha line flux vs g-r legacy color, z_redrock vs z_true, 
    and redshift success rate vs apflux r-mag  
    '''
    if field == 'g15': nblocks = 64
    else: raise ValueError
    cata = Cat.GamaLegacy()
    gleg = cata.Read(field)
    redshift = gleg['gama-spec']['z']
    ngal = len(redshift)
    print('%i galaxies in the GAMA-Legacy G15 region' % ngal)
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # apparent magnitude from Legacy photometry aperture flux
    g_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_g'][:,1])
    r_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    # H-alpha line flux from GAMA spectroscopy
    gama_ha = gleg['gama-spec']['ha_flux']
    
    i_dark, i_bright, zbest_d, zbest_b, zwarn_d, zwarn_b = _read_expSpectra_blocks(field, iblocks=iblocks, exptime=exptime)
    
    faintemline = (gama_ha[i_dark] < 10.) 
    i_dark = i_dark[faintemline]
    i_bright = i_bright[faintemline]
    zbest_d = zbest_d[faintemline]
    zbest_b = zbest_b[faintemline]
    zwarn_d = zwarn_d[faintemline]
    zwarn_b = zwarn_b[faintemline]

    print('%i galaxies' % len(zbest_d))
    print('%i spectra w/ dark sky have ZWARN != 0' % np.sum(zwarn_d != 0))
    print('%i spectra w/ bright sky have ZWARN != 0' % np.sum(zwarn_b != 0))
    
    # calculate delta z / (1+z)  for dark and brigth skies 
    dz_1pz_dark = np.abs(zbest_d - redshift[i_dark])/(1.+redshift[i_dark])
    dz_1pz_bright = np.abs(zbest_b - redshift[i_dark])/(1.+redshift[i_dark])
    
    print('%i spectra w/ dark sky fail' % np.sum(~((zwarn_d == 0) & (dz_1pz_dark < 0.003))))
    print('%i spectra w/ bright sky fail' % np.sum(~((zwarn_b == 0) & (dz_1pz_bright < 0.003))))
    
    fig = plt.figure(figsize=(20, 4))
    # panel 1 Halpha line flux versus g-r apflux legacy color 
    sub = fig.add_subplot(141)
    hasha = (gama_ha > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[hasha], gama_ha[hasha], 
            s=1, c='k', label='GAMA '+field.upper()+' $\cup$ Legacy DR5')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[~hasha], np.repeat(1e-2, np.sum(~hasha)), 
            s=1, c='k')
    hasha = (gama_ha[i_dark] > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], gama_ha[i_dark][hasha], 
            c='C0', s=0.5, alpha=0.33, label=r'H$\alpha$ line flux $< 10$')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha], np.repeat(1e-2, np.sum(~hasha)), 
            c='C0', s=0.5, alpha=0.33)
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylabel(r'$H_\alpha$ line flux GAMA $[10^{-17}erg/s/cm^2]$', fontsize=15)
    sub.set_ylim([5e-3, 2e4])
    sub.set_yscale('log')
    sub.legend(loc='lower left', bbox_to_anchor=(0.0, 0.05), markerscale=10, 
            handletextpad=0, frameon=True, fancybox=True, framealpha=0.8, prop={'size':15}) 
    # panel 2 z_redrock vs r_true
    sub = fig.add_subplot(142)
    sub.scatter(redshift[i_dark], dz_1pz_bright, c='C1', s=10)
    sub.scatter(redshift[i_dark], dz_1pz_dark, c='C0', s=2)
    sub.set_xlabel(r"$z_\mathrm{true}$ GAMA", fontsize=25)
    sub.set_xlim([0., 0.36])
    sub.set_ylabel(r"$\Delta z / (1+z_\mathrm{true})$", fontsize=20)
    sub.set_ylim([-0.05, 1.])
    # panel 3 
    sub = fig.add_subplot(143)
    mm_dark, e1_dark, ee1_dark = _z_successrate(r_mag_legacy_apflux[i_dark], #range=(18, 22), 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(r_mag_legacy_apflux[i_dark], #range=(18, 22), 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([15., 25.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$r_\mathrm{apflux}$ magnitude', fontsize=20)
    sub.set_xlim([16., 23.])
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])
    
    sub = fig.add_subplot(144)
    #prop = np.log10(gama_ha)
    #prop[gama_ha <= 0.] = -2
    prop = (g_mag_legacy_apflux - r_mag_legacy_apflux)
    prop_range = [-0.2, 2.2]
    #prop_range = [-2.1, 1.2]
    mm_dark, e1_dark, ee1_dark = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([prop.min()-0.5*np.abs(prop.min()), 1.5*prop.max()], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    #sub.set_xlabel(r'$\log \mathrm{H}\alpha$ line flux', fontsize=20)
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim(prop_range) 
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])
    sub.legend(loc='lower right', handletextpad=-0.2, prop={'size': 20})

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_faintemline_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess.png", 
        bbox_inches='tight')
    plt.close() 

    # --- H-alpha vs (g-r) color ---
    fig = plt.figure(figsize=(16,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(131)
    hasha = (gama_ha[i_dark] > 0.) 
    gr = np.concatenate([(g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], (g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha]])
    ha = np.concatenate([gama_ha[i_dark][hasha], np.repeat(1e-2, np.sum(~hasha))])
    w = np.ones(len(ha))
    hb = sub.hexbin(gr, ha, C=w, reduce_C_function=np.sum, vmax=500., mincnt=20, gridsize=50, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    sub.text(0.1, 0.9, r'$N_{bin} > 20$', ha='left', va='center', 
            transform=sub.transAxes, fontsize=20)
    sub.text(0.1, 0.1, r'Faint H$\alpha$ Galaxies', ha='left', va='bottom', 
            transform=sub.transAxes, fontsize=20)
    cb.set_label(r'$N_\mathrm{gal}$', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    sub.set_title(r'', fontsize=20)
    # --- H-alpha vs (g-r) color : dark sky ---
    sub = fig.add_subplot(132)
    dz_1pz_dark[zwarn_d != 0] = 1.
    dz_1pz_d = np.concatenate([dz_1pz_dark[hasha], dz_1pz_dark[~hasha]]) 
    w_dz_1pz_d = np.zeros(len(dz_1pz_d))
    w_dz_1pz_d[dz_1pz_d < 0.003] = 1. 
    hb = sub.hexbin(gr, ha, C=w_dz_1pz_d, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    sub.set_title('w/ Dark Sky', fontsize=20)
    # --- H-alpha vs (g-r) color : bright sky ---
    sub = fig.add_subplot(133)
    hasha = (gama_ha[i_dark] > 0.) 
    dz_1pz_bright[zwarn_b != 0] = 1.
    dz_1pz_b = np.concatenate([dz_1pz_bright[hasha], dz_1pz_bright[~hasha]]) 
    w_dz_1pz_b = np.zeros(len(dz_1pz_b))
    w_dz_1pz_b[dz_1pz_b < 0.003] = 1. 
    hb = sub.hexbin(gr, ha, C=w_dz_1pz_b, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    sub.set_title('w/ Bright Sky', fontsize=20)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$(g-r)$ from Legacy ap. flux', labelpad=5, fontsize=20)
    bkgd.set_ylabel(r'$H_\alpha$ line flux GAMA $[10^{-17}erg/s/cm^2]$', labelpad=7, fontsize=20)

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_faintemline_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_HA_gr.png", 
        bbox_inches='tight')
    plt.close() 
    
    # --- H-alpha vs (g-r) color ---
    fig = plt.figure(figsize=(16,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(131)
    w = np.ones(len(i_dark))
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w, reduce_C_function=np.sum, vmax=1000., mincnt=20., gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'$N_\mathrm{gal}$', fontsize=20)
    sub.text(0.1, 0.9, r'$N_{bin} > 20$', ha='left', va='center', 
            transform=sub.transAxes, fontsize=20)
    sub.text(0.1, 0.1, r'Faint H$\alpha$ Galaxies', ha='left', va='bottom', 
            transform=sub.transAxes, fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    # --- H-alpha vs (g-r) color : Dark Sky ---
    sub = fig.add_subplot(132)
    dz_1pz_dark[zwarn_d != 0] = 1.
    w_dz_1pz_dark = np.zeros(len(dz_1pz_dark))
    w_dz_1pz_dark[dz_1pz_dark < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_dark, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Dark Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    # --- H-alpha vs (g-r) color : Bright Sky ---
    sub = fig.add_subplot(133)
    dz_1pz_bright[zwarn_b != 0] = 1.
    w_dz_1pz_bright = np.zeros(len(dz_1pz_bright))
    w_dz_1pz_bright[dz_1pz_bright < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_bright, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Bright Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'GAMA $M_{0.1r}$', labelpad=5, fontsize=20)
    bkgd.set_ylabel(r'GAMA $^{0.1}(g-r)$ color', labelpad=7, fontsize=20)

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_faintemline_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_Mr_gr.png", 
        bbox_inches='tight')
    plt.close() 
    return None


def expSpectra_blocks_noHalpha_zsuccess(field, iblocks=30, exptime=480): 
    ''' Redrock redshift success rate of simulated exposures of 
    simulated spectra with dark versus bright sky of 1000 randomly 
    selected galaxies.
    
    Creates a three panel plot: Halpha line flux vs g-r legacy color, z_redrock vs z_true, 
    and redshift success rate vs apflux r-mag  
    '''
    if field == 'g15': nblocks = 64
    else: raise ValueError
    cata = Cat.GamaLegacy()
    gleg = cata.Read(field)
    redshift = gleg['gama-spec']['z']
    ngal = len(redshift)
    print('%i galaxies in the GAMA-Legacy G15 region' % ngal)
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # apparent magnitude from Legacy photometry aperture flux
    g_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_g'][:,1])
    r_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    # H-alpha line flux from GAMA spectroscopy
    gama_ha = gleg['gama-spec']['ha_flux']
    
    i_dark, i_bright, zbest_d, zbest_b, zwarn_d, zwarn_b = _read_expSpectra_blocks(field, iblocks=iblocks, exptime=exptime)
    
    faintemline = (gama_ha[i_dark] <= 0.) 
    i_dark = i_dark[faintemline]
    i_bright = i_bright[faintemline]
    zbest_d = zbest_d[faintemline]
    zbest_b = zbest_b[faintemline]
    zwarn_d = zwarn_d[faintemline]
    zwarn_b = zwarn_b[faintemline]

    print('%i galaxies' % len(zbest_d))
    print('%i spectra w/ dark sky have ZWARN != 0' % np.sum(zwarn_d != 0))
    print('%i spectra w/ bright sky have ZWARN != 0' % np.sum(zwarn_b != 0))
    
    # calculate delta z / (1+z)  for dark and brigth skies 
    dz_1pz_dark = np.abs(zbest_d - redshift[i_dark])/(1.+redshift[i_dark])
    dz_1pz_bright = np.abs(zbest_b - redshift[i_dark])/(1.+redshift[i_dark])
    
    print('%i spectra w/ dark sky fail' % np.sum(~((zwarn_d == 0) & (dz_1pz_dark < 0.003))))
    print('%i spectra w/ bright sky fail' % np.sum(~((zwarn_b == 0) & (dz_1pz_bright < 0.003))))
    
    fig = plt.figure(figsize=(20,4))
    # panel 1 Halpha line flux versus g-r apflux legacy color 
    sub = fig.add_subplot(141)
    hasha = (gama_ha > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[hasha], gama_ha[hasha], 
            s=1, c='k', label='GAMA '+field.upper()+' $\cup$ Legacy DR5')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[~hasha], np.repeat(1e-2, np.sum(~hasha)), 
            s=1, c='k')
    hasha = (gama_ha[i_dark] > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], gama_ha[i_dark][hasha], 
            c='C0', s=0.5, alpha=0.33, label=r'no H$\alpha$')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha], np.repeat(1e-2, np.sum(~hasha)), 
            c='C0', s=0.5, alpha=0.33)
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylabel(r'$H_\alpha$ line flux GAMA $[10^{-17}erg/s/cm^2]$', fontsize=15)
    sub.set_ylim([5e-3, 2e4])
    sub.set_yscale('log')
    sub.legend(loc='lower left', bbox_to_anchor=(0.0, 0.05), markerscale=10, 
            handletextpad=0, frameon=True, fancybox=True, framealpha=0.8, prop={'size':15}) 
    # panel 2 z_redrock vs r_true
    sub = fig.add_subplot(142)
    sub.scatter(redshift[i_dark], dz_1pz_bright, c='C1', s=10)
    sub.scatter(redshift[i_dark], dz_1pz_dark, c='C0', s=2)
    sub.set_xlabel(r"$z_\mathrm{true}$ GAMA", fontsize=25)
    sub.set_xlim([0., 0.36])
    sub.set_ylabel(r"$\Delta z / (1+z_\mathrm{true})$", fontsize=20)
    sub.set_ylim([-0.05, 1.])
    # panel 3 
    sub = fig.add_subplot(143)
    mm_dark, e1_dark, ee1_dark = _z_successrate(r_mag_legacy_apflux[i_dark], #range=(18, 22), 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(r_mag_legacy_apflux[i_dark], #range=(18, 22), 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([15., 25.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$r_\mathrm{apflux}$ magnitude', fontsize=20)
    sub.set_xlim([16., 23.])
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])

    sub = fig.add_subplot(144)
    prop = (g_mag_legacy_apflux - r_mag_legacy_apflux)
    prop_range = [-0.2, 2.2]
    mm_dark, e1_dark, ee1_dark = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([prop.min()-0.5*np.abs(prop.min()), 1.5*prop.max()], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim(prop_range) 
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])
    sub.legend(loc='lower right', handletextpad=-0.2, prop={'size': 20})
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_noHalpha_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess.png", 
        bbox_inches='tight')
    plt.close() 

    # --- M_r vs (g-r) color ---
    fig = plt.figure(figsize=(16,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(131)
    w = np.ones(len(i_dark))
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w, reduce_C_function=np.sum, vmax=1000., mincnt=20., gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'$N_\mathrm{gal}$', fontsize=20)
    sub.text(0.1, 0.9, r'$N_{bin} > 20$', ha='left', va='center', 
            transform=sub.transAxes, fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    sub.set_title(r'Faint H$\alpha$ Galaxies', fontsize=20)
    # --- M_r vs (g-r) color : Dark Sky ---
    sub = fig.add_subplot(132)
    dz_1pz_dark[zwarn_d != 0] = 1.
    w_dz_1pz_dark = np.zeros(len(dz_1pz_dark))
    w_dz_1pz_dark[dz_1pz_dark < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_dark, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Dark Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    # --- M_r vs (g-r) color : Bright Sky ---
    sub = fig.add_subplot(133)
    dz_1pz_bright[zwarn_b != 0] = 1.
    w_dz_1pz_bright = np.zeros(len(dz_1pz_bright))
    w_dz_1pz_bright[dz_1pz_bright < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_bright, vmin=0.0, vmax=1.0, mincnt=20, gridsize=50)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Bright Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'GAMA $M_{0.1r}$', labelpad=5, fontsize=20)
    bkgd.set_ylabel(r'GAMA $^{0.1}(g-r)$ color', labelpad=7, fontsize=20)

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_noHalpha_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_Mr_gr.png", 
        bbox_inches='tight')
    plt.close() 
    return None


def _expSpectra_blocks_rapflux19_zsuccess(field, iblocks=30, exptime=480): 
    ''' Redrock redshift success rate of simulated exposures of 
    simulated spectra with dark versus bright sky of 1000 randomly 
    selected galaxies.
    
    Creates a three panel plot: Halpha line flux vs g-r legacy color, z_redrock vs z_true, 
    and redshift success rate vs apflux r-mag  
    '''
    if field == 'g15': nblocks = 64
    else: raise ValueError
    cata = Cat.GamaLegacy()
    gleg = cata.Read(field)
    redshift = gleg['gama-spec']['z']
    ngal = len(redshift)
    print('%i galaxies in the GAMA-Legacy G15 region' % ngal)
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1 

    # apparent magnitude from Legacy photometry aperture flux
    g_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_g'][:,1])
    r_mag_legacy_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    # H-alpha line flux from GAMA spectroscopy
    gama_ha = gleg['gama-spec']['ha_flux']
    
    i_dark, i_bright, zbest_d, zbest_b, zwarn_d, zwarn_b = _read_expSpectra_blocks(field, iblocks=iblocks, exptime=exptime)
    
    faintemline = (gama_ha[i_dark] < 10.) 

    rapflux_cut = ((r_mag_legacy_apflux[i_dark] < 19.2) & (r_mag_legacy_apflux[i_dark] > 18.8)) 
    i_dark = i_dark[rapflux_cut]
    i_bright = i_bright[rapflux_cut]
    zbest_d = zbest_d[rapflux_cut]
    zbest_b = zbest_b[rapflux_cut]
    zwarn_d = zwarn_d[rapflux_cut]
    zwarn_b = zwarn_b[rapflux_cut]

    print('%i redshifts' % len(zbest_d))
    print('%i spectra w/ dark sky have ZWARN != 0' % np.sum(zwarn_d != 0))
    print('%i spectra w/ bright sky have ZWARN != 0' % np.sum(zwarn_b != 0))
    
    # calculate delta z / (1+z)  for dark and brigth skies 
    dz_1pz_dark = np.abs(zbest_d - redshift[i_dark])/(1.+redshift[i_dark])
    dz_1pz_bright = np.abs(zbest_b - redshift[i_dark])/(1.+redshift[i_dark])
    print('%i spectra w/ dark sky fail' % np.sum(~((zwarn_d == 0) & (dz_1pz_dark < 0.003))))
    print('%i spectra w/ bright sky fail' % np.sum(~((zwarn_b == 0) & (dz_1pz_bright < 0.003))))
    
    fig = plt.figure(figsize=(20,4))
    # panel 1 Halpha line flux versus g-r apflux legacy color 
    sub = fig.add_subplot(141)
    hasha = (gama_ha > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[hasha], gama_ha[hasha], 
            s=1, c='k', label='GAMA '+field.upper()+' $\cup$ Legacy DR5')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[~hasha], np.repeat(1e-2, np.sum(~hasha)), s=1, c='k')
    hasha = (gama_ha[i_dark] > 0.) 
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], gama_ha[i_dark][hasha], 
            c='C0', s=0.5, alpha=0.33, label=r'$18.8 < r_\mathrm{apflux} < 19.2$')
    sub.scatter((g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha], np.repeat(1e-2, np.sum(~hasha)), 
            c='C0', s=0.5, alpha=0.33)
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylabel(r'$H_\alpha$ line flux GAMA $[10^{-17}erg/s/cm^2]$', fontsize=15)
    sub.set_ylim([5e-3, 2e4])
    sub.set_yscale('log')
    sub.legend(loc='lower left', bbox_to_anchor=(0.0, 0.05), markerscale=10, 
            handletextpad=0, frameon=True, fancybox=True, framealpha=0.8, prop={'size':15}) 
    # panel 2 z_redrock vs r_true
    sub = fig.add_subplot(142)
    sub.scatter(redshift[i_dark], dz_1pz_bright, c='C1', s=10)
    sub.scatter(redshift[i_dark], dz_1pz_dark, c='C0', s=2)
    sub.set_xlabel(r"$z_\mathrm{true}$ GAMA", fontsize=25)
    sub.set_xlim([0., 0.36])
    sub.set_ylabel(r"$\Delta z / (1+z_\mathrm{true})$", fontsize=20)
    sub.set_ylim([-0.05, 1.])
    # panel 3 
    sub = fig.add_subplot(143)
    prop = (g_mag_legacy_apflux - r_mag_legacy_apflux)
    prop_range = [-0.2, 2.2]
    mm_dark, e1_dark, ee1_dark = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([prop.min()-0.5*np.abs(prop.min()), 1.5*prop.max()], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$(g-r)$ from Legacy ap. flux', fontsize=20)
    sub.set_xlim(prop_range) 
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])

    sub = fig.add_subplot(144)
    prop = np.log10(gama_ha)
    prop[gama_ha <= 0.] = -2
    prop_range = [-2., 4.]
    mm_dark, e1_dark, ee1_dark = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_d == 0) & (dz_1pz_dark < 0.003)))
    mm_bright, e1_bright, ee1_bright = _z_successrate(prop[i_dark], range=prop_range, 
            condition=((zwarn_b == 0) & (dz_1pz_bright < 0.003)))
    sub.plot([prop.min()-0.5*np.abs(prop.min()), 1.5*prop.max()], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(mm_dark, e1_dark, ee1_dark, c='C0', fmt='o', label="w/ Dark Sky")
    sub.errorbar(mm_bright, e1_bright, ee1_bright, fmt='.C1', label="w/ Bright Sky")
    sub.set_xlabel(r'$\log \mathrm{H}\alpha$ line flux', fontsize=20)
    sub.set_xlim(prop_range) 
    sub.set_ylabel(r'Redshift Success Rate', fontsize=20)
    sub.set_ylim([0., 1.2])
    sub.legend(loc='lower left', handletextpad=0., prop={'size': 20})

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_rapflux19_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess.png", 
        bbox_inches='tight')
    plt.close() 

    # --- H-alpha vs (g-r) color ---
    fig = plt.figure(figsize=(16,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(131)
    hasha = (gama_ha[i_dark] > 0.) 
    gr = np.concatenate([(g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][hasha], (g_mag_legacy_apflux - r_mag_legacy_apflux)[i_dark][~hasha]])
    ha = np.concatenate([gama_ha[i_dark][hasha], np.repeat(1e-2, np.sum(~hasha))])
    w = np.ones(len(ha))
    hb = sub.hexbin(gr, ha, C=w, reduce_C_function=np.sum, vmax=100., mincnt=10, gridsize=20, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    sub.text(0.05, 0.9, r'$N_{bin} > 10$', ha='left', va='center', 
            transform=sub.transAxes, fontsize=20)
    sub.text(0.05, 0.1, r'$18.8 < r_\mathrm{apflux} < 19.2$', ha='left', va='bottom', 
            transform=sub.transAxes, fontsize=20)
    cb.set_label(r'$N_\mathrm{gal}$', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    # --- H-alpha vs (g-r) color : dark sky ---
    sub = fig.add_subplot(132)
    dz_1pz_dark[zwarn_d != 0] = 1.
    dz_1pz_d = np.concatenate([dz_1pz_dark[hasha], dz_1pz_dark[~hasha]]) 
    w_dz_1pz_d = np.zeros(len(dz_1pz_d))
    w_dz_1pz_d[dz_1pz_d < 0.003] = 1. 
    hb = sub.hexbin(gr, ha, C=w_dz_1pz_d, vmin=0.0, vmax=1.0, mincnt=10, gridsize=20, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    sub.set_title('w/ Dark Sky', fontsize=20)
    # --- H-alpha vs (g-r) color : bright sky ---
    sub = fig.add_subplot(133)
    hasha = (gama_ha[i_dark] > 0.) 
    dz_1pz_bright[zwarn_b != 0] = 1.
    dz_1pz_b = np.concatenate([dz_1pz_bright[hasha], dz_1pz_bright[~hasha]]) 
    w_dz_1pz_b = np.zeros(len(dz_1pz_b))
    w_dz_1pz_b[dz_1pz_b < 0.003] = 1. 
    hb = sub.hexbin(gr, ha, C=w_dz_1pz_b, vmin=0.0, vmax=1.0, mincnt=10, gridsize=20, yscale='log')
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-0.2, 2.])
    sub.set_ylim([5e-3, 2e4])
    sub.set_title('w/ Bright Sky', fontsize=20)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$(g-r)$ from Legacy ap. flux', labelpad=5, fontsize=20)
    bkgd.set_ylabel(r'$H_\alpha$ line flux GAMA DR3 $[10^{-17}erg/s/cm^2]$', labelpad=7, fontsize=20)

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_rapflux19_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_HA_gr.png", 
        bbox_inches='tight')
    plt.close() 
    
    # --- H-alpha vs (g-r) color ---
    fig = plt.figure(figsize=(16,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(131)
    w = np.ones(len(i_dark))
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w, reduce_C_function=np.sum, vmax=100., mincnt=10, gridsize=20)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'$N_\mathrm{gal}$', fontsize=20)
    sub.text(0.05, 0.9, r'$N_{bin} > 10$', ha='left', va='center', 
            transform=sub.transAxes, fontsize=20)
    sub.text(0.05, 0.1, r'$18.8 < r_\mathrm{apflux} < 19.2$', ha='left', va='bottom', 
            transform=sub.transAxes, fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    # --- H-alpha vs (g-r) color : Dark Sky ---
    sub = fig.add_subplot(132)
    dz_1pz_dark[zwarn_d != 0] = 1.
    w_dz_1pz_dark = np.zeros(len(dz_1pz_dark))
    w_dz_1pz_dark[dz_1pz_dark < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_dark, vmin=0.0, vmax=1.0, mincnt=10, gridsize=20)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Dark Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 
    # --- H-alpha vs (g-r) color : Bright Sky ---
    sub = fig.add_subplot(133)
    dz_1pz_bright[zwarn_b != 0] = 1.
    w_dz_1pz_bright = np.zeros(len(dz_1pz_bright))
    w_dz_1pz_bright[dz_1pz_bright < 0.003] = 1. 
    hb = sub.hexbin(absmag_ugriz[2,i_dark], absmag_ugriz[1,i_dark] - absmag_ugriz[2,i_dark], 
            C=w_dz_1pz_bright, vmin=0.0, vmax=1.0, mincnt=10, gridsize=20)
    cb = plt.colorbar(hb, ax=sub)
    cb.set_label(r'Redshift Success Rate', fontsize=20)
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.6])
    sub.set_title('w/ Bright Sky', fontsize=20) 
    sub.set_yticks([-0.2, 0.2, 0.6, 1.0, 1.4]) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'GAMA $M_{0.1r}$', labelpad=5, fontsize=20)
    bkgd.set_ylabel(r'GAMA $^{0.1}(g-r)$ color', labelpad=7, fontsize=20)

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(UT.doc_dir()+"figs/gLeg_"+field+"_expSpectra_rapflux19_"+str(iblocks)+"blocks_exptime"+str(exptime)+"_zsuccess_Mr_gr.png", 
        bbox_inches='tight')
    plt.close() 
    return None


def _read_expSpectra_blocks(field, iblocks=30, exptime=300): 
    ''' read in the blocks of expSpectra given the field, number of blocks, 
    and exposure time 
    '''
    if field == 'g15': nblocks = 64
    else: raise ValueError
    i_dark, i_bright = [], [] 
    zbest_d, zbest_b = [], [] 
    zwarn_d, zwarn_b = [], [] 
    for iblock in range(1,iblocks+1):
        print('%i th block' % iblock)
        fzdata = lambda s, et: ''.join([UT.dat_dir(), 'redrock/', 
            'GamaLegacy.', field, '.expSpectra.', s, 'sky.seed1.exptime', str(exptime), '.', str(iblock), 'of64blocks.redrock.fits']) 
        findex = lambda s, et: ''.join([UT.dat_dir(), 'spectra/', 
            'GamaLegacy.', field, '.expSpectra.', s, 'sky.seed1.exptime', str(exptime), '.', str(iblock), 'of64blocks.index']) 
    
        # read in redrock redshifts
        zdark_data = fits.open(fzdata('dark', exptime))[1].data
        i_dark_i = np.loadtxt(findex('dark', exptime), unpack=True, usecols=[0], dtype='i')

        zbright_data = fits.open(fzdata('bright', exptime))[1].data
        i_bright_i = np.loadtxt(findex('bright', exptime), unpack=True, usecols=[0], dtype='i')

        assert np.array_equal(i_dark_i, i_bright_i)
        # all the block indices
        i_dark.append(i_dark_i) 
        i_bright.append(i_bright_i) 

        # all the redrock z estimates 
        zbest_d.append(zdark_data['Z'])
        zbest_b.append(zbright_data['Z'])
        # zwarnings 
        zwarn_d.append(zdark_data['ZWARN'])
        zwarn_b.append(zbright_data['ZWARN'])

    i_dark = np.concatenate(i_dark) 
    i_bright = np.concatenate(i_bright) 
    zbest_d = np.concatenate(zbest_d)
    zbest_b = np.concatenate(zbest_b)
    zwarn_d = np.concatenate(zwarn_d)
    zwarn_b = np.concatenate(zwarn_b)
    return i_dark, i_bright, zbest_d, zbest_b, zwarn_d, zwarn_b


def _z_successrate(var, range=None, condition=None, nbins=20): 
    ''' 
    '''
    if condition is None: raise ValueError
    s1 = condition
    
    h0, bins = np.histogram(var, bins=nbins, range=range)
    hv, _ = np.histogram(var, bins=bins, weights=var)
    h1, _ = np.histogram(var[s1], bins=bins)
    
    good = h0 > 2
    hv = hv[good]
    h0 = h0[good]
    h1 = h1[good]
    vv = hv / h0 # weighted mean of var
    
    def _eff(k, n):
        eff = k.astype("float") / (n.astype('float') + (n==0))
        efferr = np.sqrt(eff * (1 - eff)) / np.sqrt(n.astype('float') + (n == 0))
        return eff, efferr
    
    e1, ee1 = _eff(h1, h0)
    return vv, e1, ee1


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
    #DESI_GAMA()
    #GAMALegacy_Halpha_color()
    #BGStemplates()
    #GamaLegacy_matchSpectra()
    #GamaLegacy_emlineSpectra()
    #skySurfaceBrightness()
    #rMag_normalize()
    #rMag_normalize_emline()
    #expSpectra()
    #expSpectra_emline()
    #expSpectra_redrock(exptime=300)
    #expSpectra_redshift(seed=1)
    #expSpectra_SDSScomparison()
    #SDSS_emlineComparison()
    #for et in [300, 480, 1000]: 
    #expSpectra_redrock(exptime=300)
    #expSpectra_blocks_zsuccess('g15', iblocks=30, exptime=300)
    #expSpectra_blocks_faintemline_zsuccess('g15', iblocks=30, exptime=300)
    #expSpectra_blocks_noHalpha_zsuccess('g15', iblocks=30, exptime=300)
    #_expSpectra_blocks_rapflux19_zsuccess('g15', iblocks=30, exptime=300)
    skyBrightness(n_az=32, n_alt=18, method='ks', units='flux')
    skyBrightness(n_az=32, n_alt=18, method='ks', units='ratio')
    #skyBrightness(n_az=32, n_alt=18, method='pf', units='flux')
    #skyBrightness(n_az=32, n_alt=18, method='pf', units='ratio')
