'''

Test `feasibgs.forwardmodel`


'''
import h5py
import numpy as np 
import astropy.units as u 
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from redrock import plotspec as RRplotspec
from redrock.external.desi import rrdesi

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


def GamaLegacy_skyflux(obvs): 
    ''' take a random galaxy from the GAMA-Legacy catalog, match it to 
    BGS templates, then simulate exposure on the templates with varying 
    atmospheric conditions. This is just to get a taste of what each of 
    the atmospheric condition parameters do to the spectra. 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio']  # redshift
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3) # ABSMAG k-correct to z=0.1 
    
    # BGS templates
    bgs3 = FM.BGStree() 
    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    # pick a random galaxy from the GAMA-legacy sample and then 
    # find the closest template
    i_rand = [1765]# np.random.choice(range(absmag_ugriz.shape[1]), size=1) 
    
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

    # what conditions do we want to expose it in
    if obvs == 'exptime': 
        conditions = [{'exptime': val} for val in np.linspace(100, 1000, 4)]
    elif obvs == 'airmass': 
        conditions = [{'airmass': val} for val in np.linspace(1., 10., 4)]
    elif obvs == 'moonfrac': 
        conditions = [{'moonfrac': val} for val in np.linspace(0., 1., 4)]
    elif obvs == 'moonalt': 
        conditions = [{'moonalt': val, 'moonfrac':1.} for val in [-60., -30., 0., 30., 60.]]
    elif obvs == 'moonsep': 
        conditions = [{'moonsep': val, 'moonalt': 20, 'moonfrac':1.} for val in [0., 30., 60., 90.]]
    
    # plot the spectra
    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(111)
    for i_c, cond in enumerate(conditions): 
        # simulate exposure using the dark time default 
        # observing condition parameters
        waves, fluxes, skyfluxes  = bgstemp._skyflux(wave, flux, skyerr=1., **cond) 
        
        for i in range(len(waves)): 
            if i == 0: 
                lbl = '$'+str(cond[obvs])+'$'
            else: 
                lbl = None
            sub.plot(waves[i], skyfluxes[i], c='C'+str(i_c), lw=1, label=lbl) 
    sub.legend(frameon=True, loc='upper right', prop={'size':20}) 
    sub.set_xlabel('wavelength', fontsize=20) 
    sub.set_xlim([3600., 9800.]) 
    #sub.set_ylim([-50., 50.]) 
    fig.savefig(UT.fig_dir()+"GamaLegacy_skyflux."+obvs+".png", bbox_inches='tight')
    plt.close() 
    return None


def GamaLegacy_residualSpectraCond(obvs): 
    ''' take a galaxy from the GAMA-Legacy catalog, match it to 
    BGS templates, then simulate exposure on the templates with varying 
    atmospheric conditions. This is just to get a taste of what each of 
    the atmospheric condition parameters do to the spectra. 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio']  # redshift
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3) # ABSMAG k-correct to z=0.1 
    
    # BGS templates
    bgs3 = FM.BGStree() 
    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    # pick a random galaxy from the GAMA-legacy sample and then 
    # find the closest template
    i_rand = [1765]# np.random.choice(range(absmag_ugriz.shape[1]), size=1) 
    
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

    # what conditions do we want to expose it in
    if obvs == 'exptime': 
        conditions = [{'exptime': val} for val in np.linspace(100, 1000, 4)]
    elif obvs == 'airmass': 
        conditions = [{'airmass': val} for val in np.linspace(1., 10., 4)]
    elif obvs == 'moonfrac': 
        conditions = [{'moonfrac': val} for val in np.linspace(0., 1., 4)]
    
    noiseless_spectra = bgstemp.simExposure(wave, flux, skyerr=1., nonoise=True) 

    # plot the spectra
    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(111)
    for i_c, cond in enumerate(conditions): 
        # simulate exposure using the dark time default 
        # observing condition parameters
        bgs_spectra = bgstemp.simExposure(wave, flux, skyerr=1., **cond) 

        # plot exposed spectra of the three CCDs
        for b in ['b', 'r', 'z']: 
            if b == 'z': lbl = '$'+str(cond[obvs])+'$'
            else: lbl = None
            sub.plot(bgs_spectra.wave[b], 
                    bgs_spectra.flux[b].flatten() - noiseless_spectra.flux[b].flatten(), 
                    c='C'+str(i_c), lw=0.2, alpha=0.5, label=lbl) 
    sub.legend(frameon=True, loc='upper right', prop={'size':20}) 
    sub.set_xlabel('wavelength', fontsize=20) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylim([-50., 50.]) 
    fig.savefig(UT.fig_dir()+"GamaLegacy_residualSpectra."+obvs+".png", bbox_inches='tight')
    plt.close() 
    return None


def GamaLegacy_SpectraCond(obvs): 
    ''' take a random galaxy from the GAMA-Legacy catalog, match it to 
    BGS templates, then simulate exposure on the templates with varying 
    atmospheric conditions. This is just to get a taste of what each of 
    the atmospheric condition parameters do to the spectra. 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio']  # redshift
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3) # ABSMAG k-correct to z=0.1 
    
    # BGS templates
    bgs3 = FM.BGStree() 
    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    # pick a random galaxy from the GAMA-legacy sample and then 
    # find the closest template
    i_rand = [1765]# np.random.choice(range(absmag_ugriz.shape[1]), size=1) 
    
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

    # what conditions do we want to expose it in
    if obvs == 'exptime': 
        conditions = [{'exptime': val} for val in np.linspace(100, 1000, 4)]
    elif obvs == 'airmass': 
        conditions = [{'airmass': val} for val in np.linspace(1., 10., 4)]
    elif obvs == 'moonfrac': 
        conditions = [{'moonfrac': val} for val in np.linspace(0., 1., 4)]
    
    # plot the spectra
    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(111)
    for i_c, cond in enumerate(conditions): 
        # simulate exposure using the dark time default 
        # observing condition parameters
        bgs_spectra = bgstemp.simExposure(wave, flux, **cond) 

        # plot exposed spectra of the three CCDs
        for b in ['b', 'r', 'z']: 
            if b == 'z': lbl = '$'+str(cond[obvs])+'$'
            else: lbl = None
            sub.plot(bgs_spectra.wave[b], bgs_spectra.flux[b].flatten(), 
                    c='C'+str(i_c), lw=0.2, alpha=0.5, label=lbl) 
    # overplot template spectra
    sub.plot(wave, flux.flatten(), c='k', lw=0.3, label='Template')
    sub.legend(frameon=True, loc='upper right', prop={'size':20}) 
    sub.set_xlabel('wavelength', fontsize=20) 
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylim([0., 50.]) 
    fig.savefig(UT.fig_dir()+"GamaLegacy_Spectra."+obvs+".png", bbox_inches='tight')
    plt.close() 
    return None


def GamaLegacy_expSpectra(): 
    ''' match galaxies from the GAMA-Legacy catalog to 
    BGS templates and simulate exposure on the templates.
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio']  # redshift
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3) # ABSMAG k-correct to z=0.1 
    
    # BGS templates
    bgs3 = FM.BGStree() 
    bgstemp = FM.BGStemplates(wavemin=1500.0, wavemax=2e4)
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # template absolute magnitude 

    # pick 5 random galaxies from the GAMA-legacy sample
    # and then find the closest template
    i_rand = np.random.choice(range(absmag_ugriz.shape[1]), size=5) 
    
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
    
    # simulate exposure using the dark time default 
    # observing condition parameters
    bgs_spectra = bgstemp.simExposure(wave, flux) 
    
    # plot the spectra
    fig = plt.figure(figsize=(12,6))
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    sub1.scatter(absmag_ugriz[2,:], absmag_ugriz[1,:] - absmag_ugriz[2,:], c='k', s=2) 
    for ii, i in enumerate(i_rand): 
        sub1.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, edgecolors='k', marker='^', label='Template')
        sub1.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, edgecolors='k', marker='s', label='GAMA object')
        if ii == 0: 
            sub1.legend(loc='lower left', markerscale=3, handletextpad=0., prop={'size':20})

        # plot exposed spectra of the three CCDs
        for b in ['b', 'r', 'z']: 
            sub2.plot(bgs_spectra.wave[b], bgs_spectra.flux[b][ii].flatten(), c='C'+str(ii), lw=0.2, alpha=0.5) 
        # overplot template spectra
        sub2.plot(wave, flux[ii], c='k', lw=0.3, label='Template')
    sub1.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub1.set_xlim([-14., -24]) 
    sub1.set_ylim([-0.2, 1.2])
    sub2.set_xlabel('wavelength', fontsize=20) 
    sub2.set_xlim([3600., 9800.]) 
    sub2.set_ylim([0., 50.]) 
    fig.savefig(UT.fig_dir()+"GamaLegacy_expSpectra.png", bbox_inches='tight')
    plt.close() 
    return None


def matchGamaLegacy(): 
    ''' match galaxies from the GAMA-Legacy catalog to 
    BGS templates and compare the meta data 
    '''
    # read in GAMA-Legacy catalog 
    cata = Cat.GamaLegacy()
    gleg = cata.Read()

    redshift = gleg['gama-spec']['z_helio']  # redshift

    # calculate ABSMAG k-correct to z=0.1 
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3) 
    
    # BGS templates
    bgs3 = FM.BGStree() 
    mabs_temp = bgs3.meta['SDSS_UGRIZ_ABSMAG_Z01'] # absolute magnitude 

    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(121)
    sub.scatter(absmag_ugriz[2,:], absmag_ugriz[1,:] - absmag_ugriz[2,:], 
            c='k', s=2, label='GAMA object') 
    sub.scatter(mabs_temp[:,2], mabs_temp[:,1] - mabs_temp[:,2], 
            c='C0', s=2, label='Template') 
    sub.legend(loc='lower left', markerscale=5, handletextpad=0, prop={'size': 15}) 
    sub.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub.set_xlim([-14., -24]) 
    sub.set_ylabel('$^{0.1}(g-r)$', fontsize=20) 
    sub.set_ylim([-0.2, 1.2])

    # pick 10 random galaxies from the GAMA-legacy sample
    # and then find the closest template
    i_rand = np.random.choice(range(absmag_ugriz.shape[1]), size=10) 
    
    # meta data of [z, M_r0.1, 0.1(g-r)]
    gleg_meta = np.vstack([
        redshift[i_rand], 
        absmag_ugriz[2,i_rand], 
        absmag_ugriz[1,i_rand] - absmag_ugriz[2,i_rand]]).T
    match, _ = bgs3.Query(gleg_meta)
    
    sub = fig.add_subplot(122)
    for ii, i in enumerate(i_rand): 
        sub.scatter(mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2],
                color='C'+str(ii), s=30, linewidth=0, marker='^', label='Template')
        sub.scatter(absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i], 
                color='C'+str(ii), s=30, linewidth=0, marker='s', label='GAMA object')
        sub.plot([mabs_temp[match[ii],2], mabs_temp[match[ii],1] - mabs_temp[match[ii],2]],
                [absmag_ugriz[2,i], absmag_ugriz[1,i] - absmag_ugriz[2,i]], 
                color='C'+str(ii))
        if ii == 0: sub.legend(loc='lower left', prop={'size':20})
    sub.set_xlabel('$M_{0.1r}$', fontsize=20) 
    sub.set_xlim([-14., -24]) 
    sub.set_ylim([-0.2, 1.2])
    fig.savefig(UT.fig_dir()+"matchGamaLegacy.png", bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    pass
