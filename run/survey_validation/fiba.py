'''

script for testing the output of fiber assign

'''
import os 
import glob
import numpy as np 
# --- 
import fitsio
from astropy.io import fits
# -- feasibgs -- 
from feasibgs import util as UT
# -- desitarget -- 
from desitarget.sv1.sv1_targetmask import bgs_mask
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ['NERSC_HOST'] != 'cori': 
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


dir_dat = os.path.join(UT.dat_dir(), 'survey_validation')


def testFA_singletile(ftile): 
    ''' examine the different target classes for a single tile 
    '''
    # read in tile
    tile = fits.open(ftile)[1].data

    # bgs bitmasks
    bitmask_bgs     = tile['SV1_BGS_TARGET']
    bgs_bright      = (bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool)
    bgs_faint       = (bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool)
    bgs_extfaint    = (bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool) # extended faint
    bgs_fibmag      = (bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool) # fiber magnitude limited
    bgs_lowq        = (bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool) # low quality
    #2203 Bright
    #787 Faint
    #597 Ext.Faint
    #218 Fib.Mag.
    #67 Low Q.
    n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq = \
            bgs_targetclass(tile['SV1_BGS_TARGET'])
    
    print('---------------------------------')
    print('total n_bgs = %i' % n_bgs)
    print('%i' % (n_bgs_bright + n_bgs_faint + n_bgs_extfaint + n_bgs_fibmag + n_bgs_lowq))
    print('BGS Bright %i %.3f' % (n_bgs_bright, n_bgs_bright/n_bgs))
    print('BGS Faint %i %.3f' % (n_bgs_faint, n_bgs_faint/n_bgs))
    print('BGS Ext.Faint %i %.3f' % (n_bgs_extfaint, n_bgs_extfaint/n_bgs))
    print('BGS Fib.Mag %i %.3f' % (n_bgs_fibmag, n_bgs_fibmag/n_bgs))
    print('BGS Low Q. %i %.3f' % (n_bgs_lowq, n_bgs_lowq/n_bgs))
    
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)
    sub.scatter(tile['TARGET_RA'], tile['TARGET_DEC'], c='k', s=1) 
    # BGS BRIGHT
    sub.scatter(tile['TARGET_RA'][bgs_bright], tile['TARGET_DEC'][bgs_bright], c='C0', s=3,
            label='Bright %i (%.2f)' % (np.sum(bgs_bright), np.float(np.sum(bgs_bright))/n_bgs)) 
    # BGS FAINT 
    sub.scatter(tile['TARGET_RA'][bgs_faint], tile['TARGET_DEC'][bgs_faint], c='C1', s=3, 
            label='Faint %i (%.2f)' % (np.sum(bgs_faint), np.float(np.sum(bgs_faint))/n_bgs)) 
    # BGS EXTFAINT 
    sub.scatter(tile['TARGET_RA'][bgs_extfaint], tile['TARGET_DEC'][bgs_extfaint], c='C4', s=5,
            label='Ext.Faint %i (%.2f)' % (np.sum(bgs_extfaint), np.float(np.sum(bgs_extfaint))/n_bgs)) 
    # BGS fibmag 
    sub.scatter(tile['TARGET_RA'][bgs_fibmag], tile['TARGET_DEC'][bgs_fibmag], c='C5', s=7, 
            label='Fib.Mag. %i (%.2f)' % (np.sum(bgs_fibmag), np.sum(bgs_fibmag)/n_bgs)) 
    # BGS Low quality 
    sub.scatter(tile['TARGET_RA'][bgs_lowq], tile['TARGET_DEC'][bgs_lowq], c='C3', s=9, 
            label='Low Q. %i (%.2f)' % (np.sum(bgs_lowq), np.sum(bgs_lowq)/n_bgs)) 
    
    sub.legend(loc='upper right', handletextpad=0.2, markerscale=5, fontsize=15) 
    sub.set_xlabel('RA', fontsize=20)
    sub.set_xlim(tile['TARGET_RA'].min(), tile['TARGET_RA'].max())
    sub.set_ylabel('Dec', fontsize=20)
    #sub.set_ylim(22., 26) 
    sub.set_ylim(tile['TARGET_DEC'].min(), tile['TARGET_DEC'].max())
    fig.savefig(os.path.join(dir_dat, ftile.replace('.fits', '.png')), bbox_inches='tight')  
    return None 


def testFA_tiles(): 
    '''
    '''
    n_zero = 0
    _flags = [] 
    for i, f in enumerate(glob.glob(os.path.join(dir_dat, 'fba_dr8.0.34.0.hp-comb', 'fiberassign*.fits'))): 
        # read in tile
        tile_i = fits.open(f)[1].data 
        if i == 0: 
            tile = tile_i
        else: 
            tile = np.concatenate([tile, tile_i]) 

        _n_bgs, _n_bgs_bright, _n_bgs_faint, _n_bgs_extfaint, _n_bgs_fibmag, _n_bgs_lowq = \
                bgs_targetclass(tile_i['SV1_BGS_TARGET'])
        print('---------------------------------')
        print('field %i, n_bgs = %f' % (i, _n_bgs))
        print('BGS Bright %.3f (0.45)' % (_n_bgs_bright/_n_bgs))
        print('BGS Faint %.3f (0.25)' % (_n_bgs_faint/_n_bgs))
        print('BGS Ext.Faint %.3f (0.125)' % (_n_bgs_extfaint/_n_bgs))
        print('BGS Fib.Mag %.3f (0.125)' % (_n_bgs_fibmag/_n_bgs))
        print('BGS Low Q. %.3f (0.05)' % (_n_bgs_lowq/_n_bgs))
        if _n_bgs == 0: n_zero += 1

        # high LOW Q
        if _n_bgs_lowq/_n_bgs > 0.1: 
            _flags.append(np.ones(len(tile_i['TARGET_RA'])))
        else: 
            _flags.append(np.zeros(len(tile_i['TARGET_RA'])))
    _flags = np.concatenate(_flags).astype(bool)

    n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq = \
            bgs_targetclass(tile['SV1_BGS_TARGET'])
    print('---------------------------------')
    print('%i tiles with zero BGS targets' % n_zero)
    print('---------------------------------')
    print('total n_bgs = %i' % n_bgs)
    print('BGS Bright %.3f (0.45)' % (n_bgs_bright/n_bgs))
    print('BGS Faint %.3f (0.25)' % (n_bgs_faint/n_bgs))
    print('BGS Ext.Faint %.3f (0.125)' % (n_bgs_extfaint/n_bgs))
    print('BGS Fib.Mag %.3f (0.125)' % (n_bgs_fibmag/n_bgs))
    print('BGS Low Q. %.3f (0.05)' % (n_bgs_lowq/n_bgs))
    
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(tile['TARGET_RA'], tile['TARGET_DEC'], c='k', s=2)
    sub.scatter(tile['TARGET_RA'][_flags], tile['TARGET_DEC'][_flags], c='C1', s=2)

    #sub.legend(loc='upper right', handletextpad=0.2, markerscale=5, fontsize=15) 
    sub.set_xlabel('RA', fontsize=20)
    sub.set_xlim(360., 0.)#tile['TARGET_RA'].min(), tile['TARGET_RA'].max())
    sub.set_ylabel('Dec', fontsize=20)
    #sub.set_ylim(22., 26) 
    sub.set_ylim(-30., 80.)#tile['TARGET_DEC'].min(), tile['TARGET_DEC'].max())
    fig.savefig(os.path.join(dir_dat, 'fba_dr8.0.34.0.hp-comb', 'fiberassign_outliers.png'), bbox_inches='tight') 
    return None 

def testFA_tiles_targetclass(): 
    ''' plot the target class fractions of the fiberassign output 
    '''
    # SV tiles 
    sv = fitsio.read(os.path.join(dir_dat, 'BGS_SV_30_3x_superset60_Sep2019.fits')) 
    phi = np.deg2rad(sv['RA'])
    theta = 0.5 * np.pi - np.deg2rad(sv['DEC'])

    fig = plt.figure(figsize=(20,20))
    gs = mpl.gridspec.GridSpec(3,2, figure=fig)

    classes = ['BGS_BRIGHT', 'BGS_FAINT', 'BGS_FAINT_EXT', 'BGS_FIBMAG', 'BGS_LOWQ', 'IN_SPECTRUTH']
    subs = [plt.subplot(gs[i], projection='mollweide') for i in range(6)]

    for cl, sub in zip(classes, subs):  
        tras, tdecs, fclasses = [], [], [] 
        for ftile in glob.glob(os.path.join(dir_dat, 'fba_dr8.0.34.0.hp-comb', 'fiberassign*.fits')):
            # read in tile
            tile = fits.open(ftile)[1].data 
            tid     = int(ftile.split('-')[-1].replace('.fits', ''))  # tile ID 
            tra     = sv['RA'][sv['TILEID'] == tid] # tile phi 
            tdec    = sv['DEC'][sv['TILEID'] == tid] # tile theta 

            # bgs bitmask
            bitmask_bgs = tile['SV1_BGS_TARGET'] 
            n_bgs = float(np.sum((bitmask_bgs).astype(bool))) 
            
            if cl != 'IN_SPECTRUTH': 
                bgs_class = (bitmask_bgs & bgs_mask.mask(cl)).astype(bool)
            else:
                # fraction of galaxies in spectroscopic truth tables 
                bgs_class = (bitmask_bgs.astype(bool) & tile['IN_SPECTRUTH'])

            # calculate target fraction in tile  
            fclass = float(np.sum(bgs_class)) / n_bgs

            tras.append(tra)
            tdecs.append(tdec)
            fclasses.append(fclass) 

        tras = np.array(tras).flatten()
        tdecs = np.array(tdecs).flatten() 
        sub.scatter((tras - 180.) * np.pi/180., tdecs * np.pi/180., c=np.array(fclasses), 
                cmap='viridis', vmin=0., vmax=1.) 
        sub.set_title(cl, fontsize=20) 

    fig.savefig(os.path.join(dir_dat, 'fba_dr8.0.34.0.hp-comb', 'fiberassign_targetclass.png'), bbox_inches='tight') 
    return None 


def bgs_targetclass(bitmask_bgs): 
    n_bgs = np.float(np.sum(bitmask_bgs.astype(bool))) 
    n_bgs_bright    = np.sum((bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool))
    n_bgs_faint     = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool))
    n_bgs_extfaint  = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool)) # extended faint
    n_bgs_fibmag    = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool)) # fiber magnitude limited
    n_bgs_lowq      = np.sum((bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool)) # low quality
    return n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq


if __name__=="__main__": 
    #for f in glob.glob(os.path.join(dir_dat, 'fba_dr8.0.34.0.hp-comb', 'fiberassign*.fits')): 
    #    testFA_singletile(f)
    testFA_tiles()
    #testFA_tiles_targetclass()
    #for f in glob.glob(os.path.join(dir_dat, 'subpriority_test', 'fiberassign-test0*.fits')): 
    #    testFA_singletile(f)
