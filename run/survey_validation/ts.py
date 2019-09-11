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

    
    # plot baseline target selection surface density as a function of r-mag 
    rlims = np.linspace(18.6, 20., 100)
    densities = [] 
    for rlim in rlims:# [19.42, 19.44, 19.46, 19.48, 19.5, 20., 20.5]: 
        rlimit = (r_mag < rlim) 
        #print('r < %.2f' % rlim) 
        #print('density = %f' % (np.sum(rlimit)/area_eff))
        densities.append(np.sum(rlimit)/area_eff)

    rlimit = (r_mag < 19.5) 
    density_195 = np.sum(rlimit)/area_eff

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.plot(rlims, densities, c='k') 
    sub.plot([19.5, 19.5], [200, density_195], c='k', ls=':') 
    sub.plot([18.6, 19.5], [density_195, density_195], c='k', ls=':') 
    sub.set_xlabel('$r$ magnitude', fontsize=20) 
    sub.set_xlim(18.6, 20.0) 
    sub.set_ylabel(r'target density (${\rm deg}^2$)', fontsize=20) 
    sub.set_ylim(200, 1500) 
    fig.savefig(os.path.join(dir_dat, '_1400deg2.densities_rmag.png'), bbox_inches='tight') 
    return None 


def _1400deg2_sv_targetclass(): 
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

    main_faint = (r_mag > 19.5) & (r_mag < 20.1) 
    print('main faint sample')  
    print('density = %f' % (np.sum(main_faint)/area_eff))
    print('total = %f' % (np.sum(main_faint)/area_eff*60.))
    
    extended_faint = (r_mag > 20.1) & (r_mag < 20.5) 
    print('extended faint sample')  
    print('density = %f' % (np.sum(extended_faint)/area_eff))
    print('total = %f' % (np.sum(extended_faint)/area_eff*60.))
    
    fibermag_limited = (r_mag > 20.1) & (rfiber_mag < 21.051) 
    print('fiber magnitude limited sample')  
    print('density = %f' % (np.sum(fibermag_limited)/area_eff))
    print('total = %f' % (np.sum(fibermag_limited)/area_eff*60.))
    
    print('fiber magnitude limited sample but not extended faint')  
    print('density = %f' % (np.sum(fibermag_limited & ~extended_faint)/area_eff))
    print('total = %f' % (np.sum(main_faint)/area_eff*60.))
    
    print('extended faint sample but not fiber mag limited')  
    print('density = %f' % (np.sum(~fibermag_limited & extended_faint)/area_eff))
    print('total = %f' % (np.sum(main_faint)/area_eff*60.))
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


def _1400deg2_fibermag(): 
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
    r_fiber_mag = leg.flux_to_mag(targets['fiberflux_r']/targets['mw_transmission_r']) 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.scatter(r_mag, r_fiber_mag, c='k', s=0.2) 

    #bright_high_surf_bright = (r_mag < 19.5) & (r_fiber_mag < 21.) 
    #print(np.sum(bright_high_surf_bright)/area_eff)
    #sub.scatter(r_mag[bright_high_surf_bright], r_fiber_mag[bright_high_surf_bright], 
    #        c='C0', s=0.2) 
    #sub.text(18., 20.5, r'$%.f~{\rm deg}^2$' % (np.sum(bright_high_surf_bright)/area_eff), 
    #        ha='left', va='top', fontsize=15) 

    faint_high_surf_bright = (r_mag > 19.5) & (r_mag <= 20.) & (r_fiber_mag < 21.) 
    print(np.sum(faint_high_surf_bright)/area_eff)
    sub.scatter(r_mag[faint_high_surf_bright], r_fiber_mag[faint_high_surf_bright], c='C0', s=0.2)
    #sub.text(19.6, 20.75, r'$%.f~{\rm deg}^2$' % (np.sum(faint_high_surf_bright)/area_eff), 
    #        ha='left', va='top', fontsize=15) 
    
    #bright_mid_surf_bright = (r_mag < 19.5) & (r_fiber_mag > 21.) & (r_fiber_mag < 21.5) 
    #print(np.sum(bright_mid_surf_bright)/area_eff)
    #sub.scatter(r_mag[bright_mid_surf_bright], r_fiber_mag[bright_mid_surf_bright], c='C2', s=0.2)
    #sub.text(18., 21., r'$%.f~{\rm deg}^2$' % (np.sum(bright_mid_surf_bright)/area_eff), 
    #        ha='left', va='bottom', fontsize=15) 

    faint_mid_surf_bright = (r_mag > 19.5) & (r_mag <= 20.) & (r_fiber_mag > 21.) & (r_fiber_mag < 21.5) 
    print(np.sum(faint_mid_surf_bright)/area_eff)
    sub.scatter(r_mag[faint_mid_surf_bright], r_fiber_mag[faint_mid_surf_bright], c='C1', s=0.2)
    #sub.text(19.6, 21., r'$%.f~{\rm deg}^2$' % (np.sum(faint_mid_surf_bright)/area_eff), 
    #        ha='left', va='bottom', fontsize=15) 
    
    #bright_low_surf_bright = (r_mag < 19.5) & (r_fiber_mag > 21.5)
    #print(np.sum(bright_low_surf_bright)/area_eff)
    #sub.scatter(r_mag[bright_low_surf_bright], r_fiber_mag[bright_low_surf_bright], c='C4', s=0.2)
    #sub.text(18.5, 22., r'$%.f~{\rm deg}^2$' % (np.sum(bright_low_surf_bright)/area_eff), 
    #        ha='left', va='bottom', fontsize=15) 

    faint_low_surf_bright = (r_mag > 19.5) & (r_mag <= 20.) & (r_fiber_mag > 21.5)
    print(np.sum(faint_low_surf_bright)/area_eff)
    sub.scatter(r_mag[faint_low_surf_bright], r_fiber_mag[faint_low_surf_bright], c='C2', s=0.2)
    #sub.text(19.6, 22., r'$%.f~{\rm deg}^2$' % (np.sum(faint_low_surf_bright)/area_eff), 
    #        ha='left', va='bottom', fontsize=15) 
    
    faint_high_surf_bright = (r_mag > 20.) & (r_fiber_mag < 21.) 
    print(np.sum(faint_high_surf_bright)/area_eff)
    sub.scatter(r_mag[faint_high_surf_bright], r_fiber_mag[faint_high_surf_bright], c='C3', s=0.2)
    
    faint_mid_surf_bright = (r_mag > 20.) & (r_fiber_mag > 21.) & (r_fiber_mag < 21.5) 
    print(np.sum(faint_mid_surf_bright)/area_eff)
    sub.scatter(r_mag[faint_mid_surf_bright], r_fiber_mag[faint_mid_surf_bright], c='C4', s=0.2)
    
    faint_low_surf_bright = (r_mag > 20.) & (r_fiber_mag > 21.5)
    print(np.sum(faint_low_surf_bright)/area_eff)
    sub.scatter(r_mag[faint_low_surf_bright], r_fiber_mag[faint_low_surf_bright], c='C5', s=0.2)

    sub.plot([16, 23], [16, 23], c='k', ls='--') 
    sub.set_xlabel('$r$ magnitude', fontsize=25) 
    sub.set_xlim(16., 21.5) 
    sub.set_ylabel('$r$ fiber magnitude', fontsize=25) 
    sub.set_ylim(16., 24.) 
    fig.savefig(os.path.join(dir_dat, '_1400deg2.fibermag.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #_1400deg2_test()
    #_1400deg2_rmag_limit()
    #_1400deg2_sv_targetclass()
    _1400deg2_fibermag()
