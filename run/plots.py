'''

Significant plots 

'''
import numpy as np 

# -- local --  
import env
from feasibgs import util as UT
from feasibgs import catalogs as Cat

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


def GAMA_Legacy_zdist(): 
    ''' Plot the redshift distribution of the GAMA-Legacy catalog
    '''
    # read in GAMA-Legacy objects
    legacy = Cat.GamaLegacy() 
    legacy_data = legacy.Read(silent=True)

    # GAMA Halpha 
    gama_z = legacy_data['gama-spec']['z_helio']

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    _ = sub.hist(gama_z, bins=40, range=[0.0, 0.4], normed=True, histtype='step') 
    sub.set_xlabel(r'Redshift', fontsize=20) 
    sub.set_xlim([0., 0.4])
    fig.savefig(UT.fig_dir()+"GAMALegacy_zdist.pdf", bbox_inches='tight')
    plt.close() 
    return None 


def GAMA_Legacy_color(): 
    ''' Plot the relation between GAMA Halpha versus color from Legacy imaging
    '''
    # read in GAMA-Legacy objects
    legacy = Cat.GamaLegacy() 
    legacy_data = legacy.Read(silent=True)

    # GAMA Halpha 
    gama_halpha = legacy_data['gama-spec']['ha']
   
    # legacy g,r,z model fluxes in nMgy
    legacy_photo = np.array([legacy_data['legacy-photo']['flux_'+band] for band in ['g', 'r', 'z']]) 
    # legacy photometry color 
    legacy_gr = UT.flux2mag(legacy_photo[0,:], band='g') - UT.flux2mag(legacy_photo[1,:], band='r')
    legacy_rz = UT.flux2mag(legacy_photo[1,:], band='r') - UT.flux2mag(legacy_photo[2,:], band='z')

    # gama photometry (for reference) 
    gama_photo = np.array([legacy_data['gama-photo']['modelmag_'+band] for band in ['g', 'r', 'z']]) 
    gama_gr = gama_photo[0,:] - gama_photo[1,:]
    gama_rz = gama_photo[1,:] - gama_photo[2,:]

    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(121) # halpha vs (g - r)
    sub.scatter(gama_gr[::10], gama_halpha[::10], c='k', s=2) 
    sub.scatter(legacy_gr[::10], gama_halpha[::10], c='C1', s=2) 
    sub.set_xlabel(r'$(g - r)$ color', fontsize=20) 
    sub.set_xlim([-0.5, 2.5])
    sub.set_xticks([0., 1., 2.]) 
    sub.set_ylabel(r'$H_\alpha$ line flux $[10^{-17}erg/s/cm^2]$', fontsize=20)
    sub.set_yscale('log')
    sub.set_ylim([1e-2, 1e4])

    sub = fig.add_subplot(122) # halpha vs (r - z)
    sub.scatter(gama_rz[::10], gama_halpha[::10], c='k', s=2, label='GAMA photo.') 
    sub.scatter(legacy_rz[::10], gama_halpha[::10], c='C1', s=2, label='Legacy photo.') 
    sub.legend(loc='lower right', markerscale=5, handletextpad=0., prop={'size': 18})
    sub.set_xlabel(r'$(r - z)$ color', fontsize=20) 
    sub.set_xlim([-0.5, 2.5])
    sub.set_xticks([0., 1., 2.]) 
    #sub.set_ylabel(r'$H_\alpha$ line flux $[10^{-17}erg/s/cm^2]$', fontsize=20)
    sub.set_yscale('log')
    sub.set_ylim([1e-2, 1e4])
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(UT.fig_dir()+"GAMALegacy_Halpha_color.pdf", bbox_inches='tight')
    plt.close() 
    return None 


def GAMA_Legacy_rmag_z():
    ''' Plot r magnitude versus redshift (z). 
    '''
    ''' Plot the redshift distribution of the GAMA-Legacy catalog
    '''
    # read in GAMA-Legacy objects
    legacy = Cat.GamaLegacy() 
    legacy_data = legacy.Read(silent=True)

    # GAMA Halpha 
    gama_z = legacy_data['gama-spec']['z_helio']
    
    # legacy g,r,z model fluxes in nMgy
    legacy_rflux = legacy_data['legacy-photo']['flux_g']
    # legacy photometry color 
    #legacy_rmag = UT.flux2mag(legacy_data['legacy-photo']['flux_g'], band='g') 
    legacy_rmag = 22.5 - 2.5*np.log10(legacy_rflux) 

    # gama photometry (for reference) 
    gama_rmag = legacy_data['gama-photo']['modelmag_r'] 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.scatter(gama_z, gama_rmag, s=3, c='C0') 
    sub.scatter(gama_z, legacy_rmag, s=2, c='C1') 
    sub.plot([0., 1.], [20., 20.], c='k', ls='--', lw=2) 
    sub.set_xlabel(r'Redshift', fontsize=20) 
    sub.set_xlim([0., 0.4])
    sub.set_ylabel('$r$ (AB mag)', fontsize=20) 
    sub.set_ylim([13., 21.]) 
    fig.savefig(UT.fig_dir()+"GAMALegacy_rmag_z.pdf", bbox_inches='tight')
    plt.close() 
    return None 


if __name__=="__main__":
    GAMA_Legacy_rmag_z()
