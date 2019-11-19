'''

script for testing the output of fiber assign

'''
import os 
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


def testFA_singletile(itile): 
    ''' examine the different target classes for a single tile 
    '''
    # read in tile
    ftile = fits.open(os.path.join(dir_dat, 'fiberassign-0740%s.fits' % str(itile).zfill(2)))
    tile = ftile[1].data 

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
    n_bgs = np.float(np.sum(bitmask_bgs.astype(bool)))
    print(tile.shape) 
    print(n_bgs)
    
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)
    sub.scatter(tile['TARGET_RA'], tile['TARGET_DEC'], c='k', s=1) 
    # BGS BRIGHT
    sub.scatter(tile['TARGET_RA'][bgs_bright], tile['TARGET_DEC'][bgs_bright], c='C1', s=5,
            label='Bright %i (%.2f)' % (np.sum(bgs_bright), np.float(np.sum(bgs_bright))/n_bgs)) 
    # BGS FAINT 
    sub.scatter(tile['TARGET_RA'][bgs_faint], tile['TARGET_DEC'][bgs_faint], c='C0', s=5, 
            label='Faint %i (%.2f)' % (np.sum(bgs_faint), np.float(np.sum(bgs_faint))/n_bgs)) 
    # BGS EXTFAINT 
    sub.scatter(tile['TARGET_RA'][bgs_extfaint], tile['TARGET_DEC'][bgs_extfaint], c='C4', s=5,
            label='Ext.Faint %i (%.2f)' % (np.sum(bgs_extfaint), np.float(np.sum(bgs_extfaint))/n_bgs)) 
    # BGS fibmag 
    sub.scatter(tile['TARGET_RA'][bgs_fibmag], tile['TARGET_DEC'][bgs_fibmag], c='C5', s=5, 
            label='Fib.Mag. %i (%.2f)' % (np.sum(bgs_fibmag), np.sum(bgs_fibmag)/n_bgs)) 
    # BGS Low quality 
    sub.scatter(tile['TARGET_RA'][bgs_lowq], tile['TARGET_DEC'][bgs_lowq], c='C3', s=5, 
            label='Low Q. %i (%.2f)' % (np.sum(bgs_lowq), np.sum(bgs_lowq)/n_bgs)) 
    
    sub.legend(loc='upper right', handletextpad=0.2, markerscale=5, fontsize=15) 
    sub.set_xlabel('RA', fontsize=20)
    sub.set_xlim(235.5, 239.9) 
    sub.set_ylabel('Dec', fontsize=20)
    sub.set_ylim(22., 26) 
    fig.savefig(os.path.join(dir_dat, 'fiberassign-0740%s.png' % str(itile).zfill(2)), bbox_inches='tight')  
    return None 


def testFA_tiles(): 
    n_zero = 0
    for i in range(60): 
        # read in tile
        ftile_i = fits.open(os.path.join(dir_dat, 'fiberassign-0740%s.fits' % str(i).zfill(2)))
        tile_i = ftile_i[1].data 
        if i == 0: 
            tile = tile_i
        else: 
            tile = np.concatenate([tile, tile_i]) 

        _n_bgs, _n_bgs_bright, _n_bgs_faint, _n_bgs_extfaint, _n_bgs_fibmag, _n_bgs_lowq = \
                bgs_targetclass(tile_i['SV1_BGS_TARGET'])
        print('field %i, n_bgs = %f' % (i, _n_bgs))
        print('BGS Bright %.3f' % (_n_bgs_bright/_n_bgs))
        print('BGS Faint %.3f' % (_n_bgs_faint/_n_bgs))
        print('BGS Ext.Faint %.3f' % (_n_bgs_extfaint/_n_bgs))
        print('BGS Fib.Mag %.3f' % (_n_bgs_fibmag/_n_bgs))
        print('BGS Low Q. %.3f' % (_n_bgs_lowq/_n_bgs))
        if _n_bgs == 0: n_zero += 1

    n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq = \
            bgs_targetclass(tile['SV1_BGS_TARGET'])

    print('---------------------------------')
    print('total n_bgs = %i' % n_bgs)
    print('BGS Bright %.3f' % (n_bgs_bright/n_bgs))
    print('BGS Faint %.3f' % (n_bgs_faint/n_bgs))
    print('BGS Ext.Faint %.3f' % (n_bgs_extfaint/n_bgs))
    print('BGS Fib.Mag %.3f' % (n_bgs_fibmag/n_bgs))
    print('BGS Low Q. %.3f' % (n_bgs_lowq/n_bgs))
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
    #testFA_singletile(1)
    testFA_tiles()
