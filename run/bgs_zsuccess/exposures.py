'''

scripts for generating a small sample of exposures that 
reasonably span the observing conditions of BGS

'''
import os 
import h5py
import pickle 
import numpy as np 
from scipy.interpolate import interp1d
# --- astropy --- 
import astropy.units as u
from astropy.io import fits
# -- feasibgs -- 
from feasibgs import util as UT 
from feasibgs import skymodel as Sky
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


specsim_sky = Sky.specsim_initialize('desi')


def pickExposures(nsub, method='random', validate=False, expfile=None, silent=True): 
    ''' Pick nsub subset of exposures from `surveysim` BGS exposure list
    Outputs a file that contains the exposure indices, observing conditions, 
    and old and new sky brightness of the chosen subset. 

    :param nsub: 
        number of exposures to pick. 

    :param method: (default: 'random') 
        method for picking the exposures. either spacefill or random 

    :param validate: (default: False)
        If True, generate some figures to validate things 

    :param silent: (default: True) 
        if False, code will print statements to indicate progress 
    '''
    # read surveysim BGS exposures 
    bgs_exps = extractBGSsurveysim(expfile)
    n_exps = len(bgs_exps['ra']) # number of exposures
    airmass = bgs_exps['airmass'] 
    moonill = bgs_exps['moon_ill'] 
    moonalt = bgs_exps['moon_alt'] 
    moonsep = bgs_exps['moon_sep'] 
    sun_alt = bgs_exps['sun_alt']
    sun_sep = bgs_exps['sun_sep']
        
    # pick a small subset
    if method == 'random': # randomly pick exposures
        if not silent: print('randomly picking exposures')
        iexp_sub = np.random.choice(np.arange(n_exps), nsub)
    elif method == 'spacefill': # pick exposures that span the observing conditions
        if not silent: print('picking exposures to span observing conditions')
        obs = np.zeros((n_exps, 5))
        #obs[:,0] = airmass 
        obs[:,0] = moonill
        obs[:,1] = moonalt 
        obs[:,2] = moonsep 
        obs[:,3] = sun_alt 
        obs[:,4] = sun_sep 
        histmd, edges = np.histogramdd(obs, 2)
        _hasexp = histmd > 0.
        has_exp = np.where(_hasexp)
        iexp_sub = []
        for i in range(np.sum(histmd > 0.)):
            in_bin = np.ones(n_exps).astype(bool)
            for i_dim in range(obs.shape[1]):
                in_bin = (in_bin & 
                        (obs[:,i_dim] > edges[i_dim][has_exp[i_dim]][i]) & 
                        (obs[:,i_dim] <= edges[i_dim][has_exp[i_dim]+1][i])) 
            iexp_sub.append(np.random.choice(np.arange(n_exps)[in_bin], 1)[0])
        iexp_sub = np.array(iexp_sub)
        nsub = len(iexp_sub) 
        if not silent: print('%i exposures in the subset' % nsub)
    
    # computed sky brightness (this takes a bit) 
    if not silent: print('computing sky brightness') 
    Iskys = [] 
    for i in iexp_sub: 
        wave, _Isky = sky_KSrescaled_twi(airmass[i], moonill[i], moonalt[i], moonsep[i], sun_alt[i], sun_sep[i])
        Iskys.append(_Isky)
    print(wave.min(), wave.max())
    Iskys = np.array(Iskys)
    # write exposure subsets out to file 
    fpick = h5py.File(os.path.join(UT.dat_dir(), 'bgs_zsuccess/', 
        '%s.subset.%i%s.hdf5' % (os.path.splitext(os.path.basename(expfile))[0], nsub, method)), 'w')
    # write observing conditions 
    for k in ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep', 'texp']:  
        fpick.create_dataset(k.lower(), data=bgs_exps[k][iexp_sub]) 
    # save sky brightnesses
    fpick.create_dataset('wave', data=wave) 
    fpick.create_dataset('sky', data=Iskys) 
    fpick.close() 

    if validate: 
        fig = plt.figure(figsize=(21,5))
        sub = fig.add_subplot(141)
        sub.scatter(bgs_exps['moon_alt'], bgs_exps['moon_ill'], c='k', s=1)
        scat = sub.scatter(bgs_exps['moon_alt'][iexp_sub], bgs_exps['moon_ill'][iexp_sub], c='C1', s=10)
        sub.set_xlabel('Moon Altitude', fontsize=20)
        sub.set_xlim([-90., 90.])
        sub.set_ylabel('Moon Illumination', fontsize=20)
        sub.set_ylim([0.5, 1.])

        sub = fig.add_subplot(142)
        sub.scatter(bgs_exps['moon_sep'], bgs_exps['moon_ill'], c='k', s=1)
        scat = sub.scatter(bgs_exps['moon_sep'][iexp_sub], bgs_exps['moon_ill'][iexp_sub], c='C1', s=10) 
        sub.set_xlabel('Moon Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylim([0.5, 1.])

        sub = fig.add_subplot(143)
        sub.scatter(bgs_exps['airmass'], bgs_exps['moon_ill'], c='k', s=1)
        scat = sub.scatter(bgs_exps['airmass'][iexp_sub], bgs_exps['moon_ill'][iexp_sub], c='C1', s=10)  
        sub.set_xlabel('Airmass', fontsize=20)
        sub.set_xlim([1., 2.])
        sub.set_ylim([0.5, 1.])

        sub = fig.add_subplot(144)
        sub.scatter(bgs_exps['sun_sep'], bgs_exps['sun_alt'], c='k', s=1)
        scat = sub.scatter(bgs_exps['sun_sep'][iexp_sub], bgs_exps['sun_alt'][iexp_sub], c='C1', s=10)
        sub.set_xlabel('Sun Separation', fontsize=20)
        sub.set_xlim([40., 180.])
        sub.set_ylabel('Sun Altitude', fontsize=20)
        sub.set_ylim([-90., 0.])
        ffig = os.path.join(UT.dat_dir(), 'bgs_zsuccess/', 
                '%s.subset.%i%s.png' % (os.path.splitext(os.path.basename(expfile))[0], nsub, method))
        fig.savefig(ffig, bbox_inches='tight')

        # plot some of the sky brightnesses
        fig = plt.figure(figsize=(15,5))
        bkgd = fig.add_subplot(111, frameon=False) 
        for isky in range(Iskys.shape[0]):
            sub = fig.add_subplot(111)
            sub.plot(wave, Iskys[isky,:])
        sub.set_xlim([3500., 9500.]) 
        sub.set_ylim([0., 20]) 
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('wavelength [Angstrom]', fontsize=25) 
        bkgd.set_ylabel('sky brightness [$erg/s/cm^2/A/\mathrm{arcsec}^2$]', fontsize=25) 
        ffig = os.path.join(UT.dat_dir(), 'bgs_zsuccess/', 
                '%s.subset.%i%s.skybright.png' % (os.path.splitext(os.path.basename(expfile))[0], nsub, method))
        fig.savefig(ffig, bbox_inches='tight')
    return None 


def plotExposures(nsub, method='random'): 
    ''' Plot shwoing the subset of exposures picked from `surveysim` exposure 
    list from Jeremy: `bgs_survey_exposures.withsun.hdf5', which  
    supplemented the observing conditions with sun observing conditions.
    Outputs a file that contains the exposure indices, observing conditions, 
    and old and new sky brightness of the chosen subset. 

    :param nsub: 
        number of exposures to pick. 

    :param method: (default: 'random') 
        method for picking the exposures. either spacefill or random 
    '''
    # read surveysim BGS exposures 
    fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_survey_exposures.withsun.hdf5']), 'r')
    bgs_exps = {}
    for k in fexps.keys():
        bgs_exps[k] = fexps[k].value

    # read exposure subsets out to file 
    fpick = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
        'bgs_survey_exposures.subset.', str(nsub), method, '.hdf5']), 'r')
    iexp_sub = fpick['iexp'].value

    fig = plt.figure(figsize=(21,5))
    sub = fig.add_subplot(141)
    sub.scatter(bgs_exps['MOONALT'], bgs_exps['MOONFRAC'], c='k', s=1)
    scat = sub.scatter(bgs_exps['MOONALT'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10)
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['MOONALT'][iexp_sub][i], 1.02*bgs_exps['MOONFRAC'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Moon Altitude', fontsize=20)
    sub.set_xlim([-90., 90.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(142)
    sub.scatter(bgs_exps['MOONSEP'], bgs_exps['MOONFRAC'], c='k', s=1)
    scat = sub.scatter(bgs_exps['MOONSEP'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10) 
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['MOONSEP'][iexp_sub][i], 1.02*bgs_exps['MOONFRAC'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Moon Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(143)
    sub.scatter(bgs_exps['AIRMASS'], bgs_exps['MOONFRAC'], c='k', s=1)
    scat = sub.scatter(bgs_exps['AIRMASS'][iexp_sub], bgs_exps['MOONFRAC'][iexp_sub], c='C1', s=10)  
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['AIRMASS'][iexp_sub][i], 1.02*bgs_exps['MOONFRAC'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Airmass', fontsize=20)
    sub.set_xlim([1., 2.])
    sub.set_ylim([0.5, 1.])

    sub = fig.add_subplot(144)
    sub.scatter(bgs_exps['SUNSEP'], bgs_exps['SUNALT'], c='k', s=1)
    scat = sub.scatter(bgs_exps['SUNSEP'][iexp_sub], bgs_exps['SUNALT'][iexp_sub], c='C1', s=10)
    for i in range(len(iexp_sub)): 
        sub.text(1.02*bgs_exps['SUNSEP'][iexp_sub][i], 1.02*bgs_exps['SUNALT'][iexp_sub][i], 
                str(i), ha='left', va='bottom', fontsize=15, color='k', bbox=dict(facecolor='w', alpha=0.7))
    sub.set_xlabel('Sun Separation', fontsize=20)
    sub.set_xlim([40., 180.])
    sub.set_ylabel('Sun Altitude', fontsize=20)
    sub.set_ylim([-90., 0.])
    fig.subplots_adjust(wspace=0.36) 
    fig.savefig(''.join([UT.dat_dir(), 'bgs_zsuccess/', 'bgs_survey_exposures.subset.', str(nsub), method, '.order.png']), 
            bbox_inches='tight')
    return None 


def tableExposures(nsub, method='spacefill'): 
    ''' write out exposure information to latex table 

    :param nsub: 
        number of exposures to pick. 

    :param method: (default: 'random') 
        method for picking the exposures. either spacefill or random 
    '''
    # read exposure subsets out to file 
    fpick = h5py.File(''.join([UT.dat_dir(), 'bgs_zsuccess/', 
        'bgs_survey_exposures.subset.', str(nsub), method, '.hdf5']), 'r')
    fexps = {} 
    for k in fpick.keys(): 
        fexps[k] = fpick[k].value 
    
    ftex = open(os.path.join(UT.dat_dir(), 'bgs_zsuccess', 'bgs_survey_exposures.subset.%i%s.tex' % (nsub, method)), 'w') 
    hdr = '\n'.join([ 
        r'\documentclass{article}', 
        r'\begin{document}', 
        r'\begin{table}', 
        r'\begin{center}', 
        (r'\caption{%i exposures sampled from surveysim exposures}' % nsub), 
        r'\begin{tabular}{|cccccccc|}', 
        r'\hline', 
        ' & '.join(['', '$t_\mathrm{exp}$', 'airmass', 'moon frac.', 'moon alt.', 'moon sep.', 'sun alt.', r'sun sep.\\[0.5ex]'])])
    ftex.write(hdr) 
    ftex.write(r'\hline') 
     
    for iexp in range(nsub): 
        str_iexp = (r'%i. & %.f & %.2f & %.2f & %.2f & %.2f & %.f & %.f \\' % 
                (iexp, fexps['exptime'][iexp], fexps['airmass'][iexp], 
                    fexps['moonfrac'][iexp], fexps['moonalt'][iexp], fexps['moonsep'][iexp], 
                    fexps['sunalt'][iexp], fexps['sunsep'][iexp]))
        ftex.write(str_iexp+'\n')  
    ftex.write(r'\hline') 
    end = '\n'.join([
        r'\end{tabular}', 
        r'\end{center}', 
        r'\end{table}', 
        r'\end{document}']) 
    ftex.write(end)
    ftex.close() 
    return None 


def extractBGSsurveysim(fname): 
    """ extra data on bgs exposures from surveysim output 

    no cosmics split adds 20% to margin
    total BGS time: 2839 hours
    total BGS minus twilight: 2372
    assuming 7.5 deg^2 per field
    assumiong 68% open-dome fraction
    """
    ssout = fits.open(fname)[1].data # survey sim output 
    tiles = fits.open(os.path.join(UT.dat_dir(), 'bright_exposure', 'desi-tiles.fits'))[1].data # desi-tiles
    
    isbgs = (tiles['PROGRAM'] == 'BRIGHT') # only bgs 
    
    uniq_tiles, iuniq = np.unique(ssout['TILEID'], return_index=True)  
    _, ssbgs, bgsss = np.intersect1d(ssout['TILEID'][iuniq], tiles['TILEID'][isbgs], return_indices=True)  
    
    RAs, DECs = [], [] 
    airmasses, moon_ills, moon_alts, moon_seps, sun_alts, sun_seps, texps = [], [], [], [], [], [], []
    seeings, transps = [], [] 
    nexps = 0 
    for i in range(len(ssbgs)): 
        isexps  = (ssout['TILEID'] == ssout['TILEID'][iuniq][ssbgs[i]]) 
        nexp    = np.sum(isexps)
        ra      = tiles['RA'][isbgs][bgsss[i]]
        dec     = tiles['DEC'][isbgs][bgsss[i]]
        mjd     = ssout['MJD'][isexps]
        # get sky parameters for given ra, dec, and mjd
        moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = get_thetaSky(np.repeat(ra, nexp), np.repeat(dec, nexp), mjd)

        texps.append(ssout['EXPTIME'][isexps]) 
        RAs.append(np.repeat(ra, nexp)) 
        DECs.append(np.repeat(dec, nexp)) 
        airmasses.append(ssout['AIRMASS'][isexps]) 
        moon_ills.append(moon_ill)
        moon_alts.append(moon_alt) 
        moon_seps.append(moon_sep)
        sun_alts.append(sun_alt) 
        sun_seps.append(sun_sep) 
        
        # atmospheric seeing and transp
        seeings.append(ssout['SEEING'][isexps]) 
        transps.append(ssout['TRANSP'][isexps]) 

        nexps += nexp 

    exps = {
        'texp':     np.concatenate(texps), 
        'ra':       np.concatenate(RAs),
        'dec':      np.concatenate(DECs),
        'airmass':  np.concatenate(airmasses),
        'moon_ill': np.concatenate(moon_ills), 
        'moon_alt': np.concatenate(moon_alts), 
        'moon_sep': np.concatenate(moon_seps), 
        'sun_alt':  np.concatenate(sun_alts), 
        'sun_sep':  np.concatenate(sun_seps)
    }
    return exps 


def sky_KSrescaled_twi(airmass, moonill, moonalt, moonsep, sun_alt, sun_sep):
    ''' calculate sky brightness using rescaled KS coefficients plus a twilight
    factor from Parker. 

    :return specsim_wave, Isky: 
        returns wavelength [Angstrom] and sky surface brightness [$10^{-17} erg/cm^{2}/s/\AA/arcsec^2$]
    '''
    #specsim_sky = Sky.specsim_initialize('desi')
    specsim_wave = specsim_sky._wavelength # Ang
    specsim_sky.airmass = airmass
    specsim_sky.moon.moon_phase = np.arccos(2.*moonill - 1)/np.pi
    specsim_sky.moon.moon_zenith = (90. - moonalt) * u.deg
    specsim_sky.moon.separation_angle = moonsep * u.deg
    
    # updated KS coefficients 
    specsim_sky.moon.KS_CR = 458173.535128
    specsim_sky.moon.KS_CM0 = 5.540103
    specsim_sky.moon.KS_CM1 = 178.141045

    _sky = specsim_sky._surface_brightness_dict['dark'].copy()
    _sky *= specsim_sky.extinction

    I_ks_rescale = specsim_sky.surface_brightness
    Isky = I_ks_rescale.value
    if sun_alt > -20.: # adding in twilight
        w_twi, I_twi = cI_twi(sun_alt, sun_sep, airmass)
        I_twi /= np.pi
        I_twi_interp = interp1d(10. * w_twi, I_twi, fill_value='extrapolate')
        Isky += np.clip(I_twi_interp(specsim_wave), 0, None) 
    return specsim_wave, Isky


def cI_twi(alpha, delta, airmass):
    ''' twilight contribution

    :param alpha: 

    :param delta: 

    :param airmass: 

    :return twi: 

    '''
    ftwi = os.path.join(UT.dat_dir(), 'sky', 'twilight_coeffs.p')
    twi_coeffs = pickle.load(open(ftwi, 'rb'))
    twi = (
        twi_coeffs['t0'] * np.abs(alpha) +      # CT2
        twi_coeffs['t1'] * np.abs(alpha)**2 +   # CT1
        twi_coeffs['t2'] * np.abs(delta)**2 +   # CT3
        twi_coeffs['t3'] * np.abs(delta)        # CT4
    ) * np.exp(-twi_coeffs['t4'] * airmass) + twi_coeffs['c0']
    return twi_coeffs['wave'], np.array(twi)


def _twilight_coeffs(): 
    ''' save twilight coefficients from Parker
    '''
    import pandas as pd
    f = os.path.join(UT.code_dir(), 'dat', 'sky', 'MoonResults.csv')

    coeffs = pd.DataFrame.from_csv(f)
    coeffs.columns = [
        'wl', 'model', 'data_var', 'unexplained_var',' X2', 'rX2',
        'c0', 'c_am', 'tau', 'tau2', 'c_zodi', 'c_isl', 'sol', 'I',
        't0', 't1', 't2', 't3', 't4', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6',
        'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'c2', 'c3', 'c4', 'c5', 'c6']
    # keep moon models
    twi_coeffs = coeffs[coeffs['model'] == 'twilight']
    coeffs = coeffs[coeffs['model'] == 'moon']
    # order based on wavelengths for convenience
    wave_sort = np.argsort(np.array(coeffs['wl']))

    twi = {} 
    twi['wave'] = np.array(coeffs['wl'])[wave_sort] 
    for k in ['t0', 't1', 't2', 't3', 't4', 'c0']:
        twi[k] = np.array(twi_coeffs[k])[wave_sort]
    
    # save to file 
    ftwi = os.path.join(UT.dat_dir(), 'sky', 'twilight_coeffs.p')
    pickle.dump(twi, open(ftwi, 'wb'))
    return None 


def get_thetaSky(ra, dec, mjd): 
    ''' given RA, Dec, and mjd time return sky parameters at kitt peak 
    '''
    import ephem 
    from astropy.time import Time
    import desisurvey.config
    import desisurvey.utils as dutils
    config = desisurvey.config.Configuration()

    mayall = ephem.Observer()
    mayall.lat = config.location.latitude().to(u.rad).value
    mayall.lon = config.location.longitude().to(u.rad).value
    mayall.elevation = config.location.elevation().to(u.m).value
    # Configure atmospheric refraction model for rise/set calculations.
    mayall.pressure = 1e3 * config.location.pressure().to(u.bar).value
    mayall.temp = config.location.temperature().to(u.C).value

    # observed time (MJD) 
    mjd_time = Time(mjd, format='mjd')

    moon_alt    = np.zeros(len(mjd))
    moon_ra     = np.zeros(len(mjd))
    moon_dec    = np.zeros(len(mjd)) 
    moon_ill    = np.zeros(len(mjd))
    sun_alt     = np.zeros(len(mjd))
    sun_ra      = np.zeros(len(mjd))
    sun_dec     = np.zeros(len(mjd)) 
    for i in range(len(mjd)):
        mayall.date = mjd_time.datetime[i] 
        _moon = ephem.Moon()
        _moon.compute(mayall) 
        _sun = ephem.Sun()
        _sun.compute(mayall) 
        
        moon_alt[i] = 180./np.pi*_moon.alt
        moon_ra[i]  = 180./np.pi*_moon.ra
        moon_dec[i] = 180./np.pi*_moon.dec
        moon_ill[i] = _moon.moon_phase
        sun_alt[i] = 180./np.pi*_sun.alt
        sun_ra[i]  = 180./np.pi*_sun.ra
        sun_dec[i] = 180./np.pi*_sun.dec

    moon_sep    = np.diag(dutils.separation_matrix(moon_ra, moon_dec, np.atleast_1d(ra), np.atleast_1d(dec)))
    sun_sep     = np.diag(dutils.separation_matrix(sun_ra, sun_dec, np.atleast_1d(ra), np.atleast_1d(dec)))
    return moon_ill, moon_alt, moon_sep, sun_alt, sun_sep


def validate_desisurvey_etc(): 
    ''' running desisurvey.etc and validating with implementaiton of sky model in this script. 
    This doesn't really belong here -- oh well. 
    '''
    import time 
    #t0 = time.time() 
    #w, Isky0 = sky_KSrescaled_twi(1.2, 0.7, 60., 100., -30., 100.)
    #print('sky model takes %f sec' % ((time.time() - t0)))
    #t0 = time.time() 
    #_, Isky1 = Sky._Isky(1.2, 0.7, 60., 100.)
    #print('light sky model takes %f sec' % ((time.time() - t0)))
    #print(Isky0)
    #print(Isky1)
    from desisurvey import etc 
    #wlim = (w.value > 4000.) & (w.value < 5000.) 
    #print np.median(Isky0[wlim])/1.519
    #wlim = (w > 4000.) & (w < 5000.) 
    #print np.median(Isky1.value[wlim])/1.519
    #print etc.texp_factor_bright_notwi(1.2, 0.7, 60., 100.) 

    #w, Isky0 = sky_KSrescaled_twi(1.2, 0.7, 60., 100., -15., 100.)
    #wlim = (w.value > 4000.) & (w.value < 5000.) 
    print(etc.texp_factor_bright_notwi(1.2, 0.7, 60., 100.))
    t0 = time.time() 
    print(etc.texp_factor_bright_notwi(
        np.array([1.2, 1.2]), 
        np.array([0.7, 0.7]), 
        np.array([60., 60.]), 
        np.array([100., 100.]))
        ) 
    print('texp_factor_bright_notwi takes %f sec' % ((time.time() - t0)))
    #print(np.median(Isky0[wlim])/1.519)
    print(etc.texp_factor_bright_twi(1.2, 0.7, 60., 100., -15., 100.)) 
    t0 = time.time() 
    print(etc.texp_factor_bright_twi(
        np.array([1.2, 1.2]), 
        np.array([0.7, 0.7]), 
        np.array([60., 60.]), 
        np.array([100., 100.]), 
        np.array([-15., -15]), 
        np.array([100., 100.]))) 
    print('texp_factor_bright_notwi takes %f sec' % ((time.time() - t0)))
    #t0 = time.time() 
    #sky_KSrescaled_twi(1.2, 0.7, 60., 100., -15., 100.)
    #print('sky model w/ twi takes %f sec' % ((time.time() - t0)))
    return None 


if __name__=="__main__": 
    #plotExposures(15, method='spacefill') 
    #tableExposures(15, method='spacefill')
    fexp = os.path.join(UT.dat_dir(), 'bright_exposure', 'exposures_surveysim_fork_150sv0p4.fits') 
    pickExposures(15, method='spacefill', validate=True, expfile=fexp, silent=False)
