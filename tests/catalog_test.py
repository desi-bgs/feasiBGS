'''

testing `feasibgs.catalogs`

'''
import numpy as np 
from pydl.pydlutils.spheregroup import spherematch

# -- local -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat 
from ChangTools.fitstables import mrdfits

# -- plotting --
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def GAMA_test():
    ''' Tests GAMA object
    '''
    gama = Cat.GAMA() 
    data = gama.Read(silent=False)
    assert 'kcorr_z0.0' in data.keys() 
    assert 'kcorr_z0.1' in data.keys() 
    assert np.array_equal(data['photo']['cataid'], data['kcorr_z0.0']['cataid'])

    fig = plt.figure(figsize=(6,6))
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('RA', labelpad=10, fontsize=25)
    bkgd.set_ylabel('Dec', labelpad=10, fontsize=25)

    sub = fig.add_subplot(111)
    sub.scatter(data['photo']['ra'], data['photo']['dec'], c='C1', s=1, label='GAMA photo+spec overlap')
    sub.set_xlim([110., 240.])
    sub.set_ylim([-3.5, 3.5])
    sub.legend(loc='lower left', markerscale=5, prop={'size':20})
    fig.savefig(UT.fig_dir()+"GAMA_test.png", bbox_inches='tight')
    plt.close() 
    return None 


def GAMA_fields():
    ''' Tests GAMA object for fields is working properly
    '''
    gama = Cat.GAMA() 

    fig = plt.figure(figsize=(12,6))
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('RA', labelpad=10, fontsize=25)
    bkgd.set_ylabel('Dec', labelpad=10, fontsize=25)
       
    for i in range(2): 
        sub = fig.add_subplot(1,2,i+1)
        gama._Build(data_release=3, silent=False) 
        data = gama.Read('all', data_release=3, silent=False)
        sub.scatter(data['photo']['ra'], data['photo']['dec'], c='k', s=1+i*5)

        for field in ['g09', 'g12', 'g15']: 
            data = gama.Read(field, data_release=3, silent=False)
            assert 'kcorr_z0.0' in data.keys() 
            assert 'kcorr_z0.1' in data.keys() 
            assert np.array_equal(data['photo']['cataid'], data['kcorr_z0.0']['cataid'])

            sub.scatter(data['photo']['ra'], data['photo']['dec'], s=1+i, label=field.upper())
        if i == 0:
            sub.set_xlim([110., 240.])
            sub.set_ylim([-3.5, 3.5])
        else: 
            sub.set_xlim([128.8, 130.])
            sub.set_ylim([-2.1, -1.8])
        sub.legend(loc='lower left', markerscale=5, prop={'size':20})
    fig.savefig(UT.fig_dir()+"GAMA_fields.png", bbox_inches='tight')
    plt.close() 
    return None 


def GAMA_Legacy_sweep():
    ''' check that appropriate sweep files are listed for the GAMA 
    survey
    '''
    gama = Cat.GAMA() 

    fig = plt.figure(figsize=(6,6))
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('RA', labelpad=10, fontsize=25)
    bkgd.set_ylabel('Dec', labelpad=10, fontsize=25)
    for field in ['g09', 'g12', 'g15']: 
        data = gama.Read(field, data_release=3, silent=False)
        assert 'kcorr_z0.0' in data.keys() 
        assert 'kcorr_z0.1' in data.keys() 
        assert np.array_equal(data['photo']['cataid'], data['kcorr_z0.0']['cataid'])
        
        gleg = Cat.GamaLegacy()
        ra_mins, dec_mins = gleg._getSweeps(field, silent=False)

        sub = fig.add_subplot(111)
        sub.scatter(data['photo']['ra'], data['photo']['dec'], s=1, label=field.upper())
        for i in range(len(ra_mins)): 
            for j in range(len(dec_mins)): 
                sub.add_patch(patches.Rectangle((ra_mins[i], dec_mins[j]), 9.95, 4.95, 
                    fill=None, alpha=1))
    sub.set_xlim([110., 240.])
    sub.set_ylim([-12.5, 12.5])
    sub.legend(loc='lower left', markerscale=5, prop={'size':20})
    fig.savefig(UT.fig_dir()+"GAMA_Legacy_sweep.png", bbox_inches='tight')
    plt.close() 
    return None 


def Legacy_test():  
    ''' Test that the Legacy object is sensible
    '''
    legacy = Cat.GamaLegacy() 
    legacy_data = legacy.Read(silent=False)
    
    # some sanity check on the data by comparing it to 
    # the GAMA footprint 
    gama = Cat.GAMA() # read in gama
    gama_data = gama.Read(silent=False)

    fig = plt.figure(figsize=(12,6))
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('RA', labelpad=10, fontsize=25)
    bkgd.set_ylabel('Dec', labelpad=10, fontsize=25)

    sub = fig.add_subplot(121)
    sub.scatter(gama_data['photo']['ra'][::10], gama_data['photo']['dec'][::10], 
            c='k', s=2, label='GAMA photo+spec')
    sub.scatter(legacy_data['ra'], legacy_data['dec'], c='C1', s=1, label='Legacy Survey DR5')  
    sub.set_xlim([110., 240.])
    sub.set_ylim([-3.5, 3.5])
    sub.legend(loc='lower left', markerscale=5, prop={'size':20})
    
    sub = fig.add_subplot(122)
    sub.scatter(gama_data['photo']['ra'], gama_data['photo']['dec'], 
            c='k', s=10, label='GAMA photo+spec')
    sub.scatter(legacy_data['ra'], legacy_data['dec'], 
            c='C1', s=5, label='Legacy Survey DR5')  
    sub.set_xlim([179., 181.])
    sub.set_ylim([0.9, 1.1])

    fig.savefig(UT.fig_dir()+"Legacy_test.png", bbox_inches='tight')
    plt.close() 
    return None


def Legacy_mismatch(): 
    '''*** Tested: randomly distributed ***
    Plot the Legacy objects that do not match any GAMA objects to make sure 
    I'm not missing any bricks
    '''
    # read in GAMA-Legacy objects
    legacy = Cat.GamaLegacy() 
    ld = legacy.Read(silent=False)
    legacy_data = ld['legacy-photo']

    # read in objects from the GAMA footprint 
    gama = Cat.GAMA() 
    gama_data = gama.Read(silent=False)
    
    # now match!
    match = spherematch(gama_data['photo']['ra'], gama_data['photo']['dec'], 
            legacy_data['ra'], legacy_data['dec'], 0.000277778)
    
    outside = {'ra': [], 'dec': []} 
    for i in range(len(gama_data['photo']['ra'])): 
        if i not in match[0]: 
            outside['ra'].append(gama_data['photo']['ra'][i]) 
            outside['dec'].append(gama_data['photo']['dec'][i]) 

    fig = plt.figure(figsize=(6,6))
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('RA', labelpad=10, fontsize=25)
    bkgd.set_ylabel('Dec', labelpad=10, fontsize=25)

    sub = fig.add_subplot(111)
    sub.scatter(gama_data['photo']['ra'], gama_data['photo']['dec'], 
            c='k', s=2, label='GAMA photo+spec')
    sub.scatter(outside['ra'], outside['dec'], c='C1', s=2, label='No Legacy') 
    sub.set_xlim([110., 240.])
    sub.set_ylim([-3.5, 3.5])
    sub.legend(loc='lower left', markerscale=5, prop={'size':20})
    
    fig.savefig(UT.fig_dir()+"GAMA_Legacy_outlier.png", bbox_inches='tight')
    plt.close() 
    return None


def GAMA_Legacy_photo_discrepancy(): 
    ''' Compare the photometry of GAMA with the photometry of 
    the Legacy Survey
    '''
    # read in GAMA-Legacy objects
    legacy = Cat.GamaLegacy() 
    legacy_data = legacy.Read(silent=False)

    # read in GAMA extinctinon data
    ext = mrdfits(UT.dat_dir()+'gama/GalacticExtinction.fits')
    
    # gama u,g,r,i,z
    bands = ['u', 'g', 'r', 'i', 'z']
    mag_gama_photo = np.array([legacy_data['gama-photo']['modelmag_'+b] for b in bands]) 
    mag_gama_kcorr0= np.array([legacy_data['gama-kcorr-z0.0'][b+'_model'] for b in bands]) 
    mag_gama_kcorr1= np.array([legacy_data['gama-kcorr-z0.1'][b+'_model'] for b in bands]) 
    
    # bruteforce match 
    ext_index = np.zeros(len(ext.cataid), dtype=bool)
    for icat in legacy_data['gama-photo']['cataid']: 
        match = (ext.cataid == icat) 
        if np.sum(match) != 1: 
            raise ValueError
        ext_index[match] = True
    
    fig = plt.figure(figsize=(20,4))
    for i, b in enumerate(bands): 
        sub = fig.add_subplot(1,5,i+1)
        sub.scatter(mag_gama_photo[i,:]-getattr(ext, 'a_'+b)[ext_index], mag_gama_kcorr0[i,:], 
                c='k', s=2) 
        sub.scatter(mag_gama_photo[i,:]-getattr(ext, 'a_'+b)[ext_index], mag_gama_kcorr1[i,:], 
                c='C0', s=2) 
        sub.plot([15, 25], [15, 25], c='k', ls='--') 
        sub.set_xlabel('$'+b+'$ mag from GAMA InputCatA', fontsize=15)
        sub.set_xlim([15, 25]) 
        if i == 0: 
            sub.set_ylabel('mag from GAMA kcorr', fontsize=15)
        sub.set_ylim([15, 25]) 
    fig.savefig(UT.fig_dir()+"GAMA_Legacy_photo_discrepancy.png", bbox_inches='tight')
    plt.close() 
    return None


def GAMA_Legacy_photo(): 
    ''' Resolve the discrepancy in the magnitudes in 'gama-photo' and 'gama-kcorr'
    '''
    # read in GAMA-Legacy objects
    legacy = Cat.GamaLegacy() 
    legacy_data = legacy.Read(silent=False)
    
    # gama g,r,z
    gama_photo = np.array([
            legacy_data['gama-photo']['modelmag_g'], 
            legacy_data['gama-photo']['modelmag_r'], 
            legacy_data['gama-photo']['modelmag_z']]) 
    
    # legacy g,r,z fluxes in nMgy
    legacy_photo = np.array([
            legacy_data['legacy-photo']['flux_g'], 
            legacy_data['legacy-photo']['flux_r'], 
            legacy_data['legacy-photo']['flux_z']])
    
    fig = plt.figure(figsize=(12,6))

    # g-r color comparison
    sub = fig.add_subplot(121)
    gama_gr = gama_photo[0,:] - gama_photo[1,:]
    # convert the legacy survey model_flux to model_mags? 
    # m = 22.5 - 2.5 log10(f)... this may be wrong because SDSS uses
    # m = -2.5/ln(10) * [asinh((f/f0)/(2b)) + ln(b)].
    # asinh mag 
    legacy_gr = UT.flux2mag(legacy_photo[0,:], band='g') - UT.flux2mag(legacy_photo[1,:], band='r')
    sub.scatter(gama_gr, legacy_gr, s=2) 
    sub.plot([-0.5, 4.5], [-0.5, 4.5], c='k', ls='--') 
    sub.set_xlabel('GAMA $g-r$', fontsize=20)
    sub.set_xlim([-0.5, 4.5]) 
    sub.set_ylabel('Legacy Survey $g-r$', fontsize=20)
    sub.set_ylim([-0.5, 4.5]) 

    # r-z color comparison
    sub = fig.add_subplot(122)
    gama_rz = gama_photo[1,:] - gama_photo[2,:]
    legacy_rz = UT.flux2mag(legacy_photo[1,:], band='r') - UT.flux2mag(legacy_photo[2,:], band='z')
    sub.scatter(gama_rz, legacy_rz, s=2) 
    sub.plot([-0.5, 4.5], [-0.5, 4.5], c='k', ls='--') 
    sub.set_xlabel('GAMA $r-z$', fontsize=20)
    sub.set_xlim([-0.5, 1.5]) 
    sub.set_ylabel('Legacy Survey $r-z$', fontsize=20)
    sub.set_ylim([-0.5, 1.5]) 
    fig.savefig(UT.fig_dir()+"GAMA_Legacy_photometry.png", bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    GAMA_fields()
    #GAMA_Legacy_sweep()
