'''

calculaions for SV target selection 


'''
import os 
import h5py 
import numpy as np 
# -- feasibgs -- 
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

dir_dat = os.path.join(UT.dat_dir(), 'survey_validation')


def _1400deg2_rmag_limit(): 
    '''
    '''
    # read in random files 
    random = np.load(os.path.join(dir_dat, 'bgs.1400deg2.random.npy'))
    
    # get spatial masking fraction 
    leg = Cat.Legacy() 
    _spatial = leg.spatial_mask(random['MASKBITS'], [random['NOBS_G'], random['NOBS_R'], random['NOBS_Z']])
    f_mask = float(np.sum(_spatial))/float(random.shape[0])
    print('f_mask=%.3f' % f_mask) 
    
    # get area and eff. area in square degrees 
    area = leg._1400deg2_area() 
    area_eff = area * f_mask
    print('area = %.f deg2' % area) 
    print('eff. area = %.f deg2' % area_eff) 

    # read in targets in the 1400deg2 test region 
    ftarg = h5py.File(os.path.join(dir_dat, 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    targets = {} 
    for k in ftarg.keys(): 
        targets[k] = ftarg[k][...] 
    # r-magnitude
    r_mag = leg.flux_to_mag(targets['flux_r']/targets['mw_transmission_r']) 
    rfiber_mag = leg.flux_to_mag(targets['fiberflux_r']/targets['mw_transmission_r']) 

    for rlim in [19.42, 19.44, 19.46, 19.48, 19.5, 20., 20.5]: 
        rlimit = (r_mag < rlim) 
        print('r < %.2f' % rlim) 
        print('density = %f' % (np.sum(rlimit)/area_eff))

    extended_faint = (r_mag > 20.1) & (r_mag < 20.5) 
    print('extended faint sample')  
    print('density = %f' % (np.sum(extended_faint)/area_eff))
    
    fibermag_limited = (r_mag > 20.1) & (rfiber_mag < 21.051) 
    print('fiber magnitude limited sample')  
    print('density = %f' % (np.sum(fibermag_limited)/area_eff))
    
    print('fiber magnitude limited sample but not extended faint')  
    print('density = %f' % (np.sum(fibermag_limited & ~extended_faint)/area_eff))
    return None 


def _1400deg2_test(): 
    '''
    '''
    # read in targets in the 1400deg2 test region 
    ftarg = h5py.File(os.path.join(dir_dat, 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    targets = {} 
    for k in ftarg.keys(): 
        targets[k] = ftarg[k][...] 

    # read in random files 
    random = np.load(os.path.join(dir_dat, 'bgs.1400deg2.random.npy'))

    # read in tycho2 stars to test masking of randoms 
    leg = Cat.Legacy() 
    tycho = leg._Tycho(ra_lim=[160., 230.], dec_lim=[-2., 18.]) 

    _spatial = leg.spatial_mask(random['MASKBITS'], [random['NOBS_G'], random['NOBS_R'], random['NOBS_Z']])

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.scatter(random['RA'], random['DEC'], c='C0', s=0.5) 
    sub.scatter(random['RA'][_spatial], random['DEC'][_spatial], c='k', s=0.5) 
    sub.scatter(targets['ra'], targets['dec'], c='C1', s=0.5) 
    sub.scatter(tycho['RA'], tycho['DEC'], c='r', s=10) 
    sub.set_xlim(160., 162.)
    sub.set_ylim(-2, 0.)
    fig.savefig(os.path.join(dir_dat, '_1400deg2_test.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #_1400deg2_test()
    _1400deg2_rmag_limit()
