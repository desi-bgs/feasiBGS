'''

make interesting plots 

'''
import numpy as np 
from pylab import cm
import healpy as HP
import fitsio as FitsIO
from astropy.cosmology import FlatLambdaCDM

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
    #i_rand = np.random.choice(range(absmag_ugriz.shape[1]), size=10) 
    
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


if __name__=="__main__": 
    #GAMALegacy_Halpha_color()
    #BGStemplates()
    #GAMALegacy()
    GamaLegacy_matchSpectra()
