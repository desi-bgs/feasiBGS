'''

validating the sky model

'''
import os
import pickle
import numpy as np 
from scipy.interpolate import interp1d
from speclite import filters
# -- astropy --
import astropy.time
import astropy.coordinates
from astropy.io import fits 
from astropy import units as u
from astropy.table import Table as aTable
# -- eso sky --
from skycalc_cli import skycalc 
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
#########################################################
# BOSS 
#########################################################
def boss_sky(): 
    ''' read in BOSS sky data -- fluxes and meta-data 

    :notes: 
         BOSS sky fiber treats the sky as a point source and therefore corrects for 
         fiber-loss accordingly. Therefore to do it correctly would involve backing 
         out the uncorrected sky flux, which is complicated.
    '''
    fboss = os.path.join(UT.dat_dir(), 'sky', 'Bright_BOSS_Sky_blue.fits')
    boss = aTable.read(fboss)
    return boss


def BOSS_sky_validate(): 
    ''' validate sky models again BOSS sky data 
    '''
    boss = boss_sky() # read in BOSS sky data 
    n_sky = len(boss) 
    print('%i sky fibers' % n_sky)

    # calcuate the different sky models
    for i in range(n_sky): 
        w_ks, ks_i      = sky_KS(boss['AIRMASS'][i], boss['MOON_ILL'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i])
        w_nks, newks_i  = sky_KSrescaled_twi(boss['AIRMASS'][i], boss['MOON_ILL'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i], 
                boss['SUN_ALT'][i], boss['SUN_SEP'][i])
        w_eso, eso_i    = sky_pseudoESO(boss['AIRMASS'][i], boss['MOON_ILL'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i], 
                boss['SUN_ALT'][i], boss['SUN_SEP'][i])
        #w_eso, eso_i    = sky_ESO(boss['AIRMASS'][i], boss['SUN_MOON_SEP'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i])

        if i == 0: 
            Isky_ks     = np.zeros((n_sky, len(w_ks)))
            Isky_newks  = np.zeros((n_sky, len(w_nks)))
            Isky_eso    = np.zeros((n_sky, len(w_eso)))
        Isky_ks[i,:]    = ks_i
        Isky_newks[i,:] = newks_i
        Isky_eso[i,:]   = eso_i

    # --- plot the sky brightness for a few of the BOSS skys ---
    fig = plt.figure(figsize=(10,10))
    for i,ispec in enumerate(np.random.choice(np.arange(n_sky), size=3, replace=False)): 
        sub = fig.add_subplot(3,1,i+1) 
        sub.plot(w_ks, Isky_ks[i,:], label='original KS') 
        sub.plot(w_nks, Isky_newks[i,:], label='rescaled KS + twi') 
        sub.plot(w_eso, Isky_eso[i,:], label='pseudo ESO') 
        sub.plot(boss['WAVE'][i] * 10., boss['SKY'][i]/np.pi, c='k', ls='--', label='BOSS sky') 
        if i == 0: sub.legend(loc='upper left', ncol=2, fontsize=15) 
        if i < 2: sub.set_xticklabels([])
        sub.set_xlim(3600., 6350)
        sub.set_ylim(0., 12.) 
        sub.set_title(('airmass=%.2f, moon ill=%.2f, alt=%.2f, sep=%.2f, sun alt=%.2f, sep=%.2f' % 
            (boss['AIRMASS'][i], boss['MOON_ILL'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i], 
                boss['SUN_ALT'][i], boss['SUN_SEP'][i])), fontsize=15)

    bkgd = fig.add_subplot(111, frameon=False) # x,y labels
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel('Wavelength [Angstrom]', fontsize=20) 
    bkgd.set_ylabel('Sky Brightness [$10^{-17} erg/s/cm^2/\AA/arcsec^2$]', fontsize=20) 
    fig.savefig(os.path.join(UT.code_dir(), 'figs', 'BOSS_sky_validate.sky_brightness.png'), 
            bbox_inches='tight') 

    # --- plot sky at 4100A as a function of varioius parameters---
    for ww in [4100., 4600]: 
        ks_wlim     = (w_ks > ww-50.) & (w_ks < ww+50.)
        nks_wlim    = (w_nks > ww-50.) & (w_nks < ww+50.)
        eso_wlim    = (w_eso > ww-50.) & (w_eso < ww+50.) 
        
        boss_ww, ks_ww, nks_ww, eso_ww = np.zeros(n_sky), np.zeros(n_sky), np.zeros(n_sky), np.zeros(n_sky)
        for i in range(n_sky): 
            boss_wlim= (boss['WAVE'][i] > ww/10.-5.) & (boss['WAVE'][i] < ww/10.+5.) 
            boss_ww[i]= np.median(boss['SKY'][i][boss_wlim])/np.pi
            ks_ww[i]  = np.median(Isky_ks[i,:][ks_wlim])
            nks_ww[i] = np.median(Isky_newks[i,:][nks_wlim])
            eso_ww[i] = np.median(Isky_eso[i,:][eso_wlim])

        fig = plt.figure(figsize=(10, 25))
        for i, k in enumerate(['MOON_ILL', 'MOON_ALT', 'MOON_SEP', 'AIRMASS', 'SUN_ALT']): 
            sub = fig.add_subplot(5,1,i+1)
            sub.scatter(boss[k][:n_sky], boss_ww/ks_ww, c='C0', s=3, label='original KS')
            sub.scatter(boss[k][:n_sky], boss_ww/nks_ww, c='C1', s=3, label='rescaled KS + twi')
            sub.scatter(boss[k][:n_sky], boss_ww/eso_ww, c='C2', s=3, label='ESO') 
            sub.set_xlabel(' '.join(k.split('_')).lower(), fontsize=25)
            if i == 0:
                sub.legend(loc='upper right', handletextpad=0, markerscale=10, fontsize=20)
                sub.set_xlim([0.,1.])
            elif i == 1:
                sub.set_xlim([0., 90.])
            elif i == 2:
                sub.set_xlim([0., 180.])
            elif i == 3:
                sub.set_xlim([1., 2.])
            elif i == 4:
                sub.set_xlim([-90., 0.])
            sub.set_ylim(0, 15) 
            sub.plot(sub.get_xlim(), [1., 1.], color='k', linestyle='--')

        bkgd = fig.add_subplot(111, frameon=False) # x,y labels
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_ylabel('(BOSS Sky)/(sky model) at %.fA' % ww, fontsize=25)
        fig.savefig(os.path.join(UT.code_dir(), 'figs', 'BOSS_sky_validate.sky%.f.png' % ww), 
                bbox_inches='tight') 

        # histogram of the devations 
        fig = plt.figure(figsize=(6,6)) 
        sub = fig.add_subplot(111)
        sub.hist(boss_ww/ks_ww, range=(0., 5.), bins=50, color='C0', label='original KS') 
        sub.hist(boss_ww/nks_ww, range=(0., 5.), bins=50, color='C1', label='rescaled KS + twi') 
        sub.hist(boss_ww/eso_ww, range=(0., 5.), bins=50, color='C2', label='pesudo ESO') 
        _, ymax = sub.get_ylim()
        sub.vlines(1., 0, ymax, color='k', linestyle='--')
        sub.set_xlim(0., 5.) 
        sub.set_ylabel('(BOSS Sky)/(sky model) at %.fA' % ww, fontsize=20)
        fig.savefig(os.path.join(UT.code_dir(), 'figs', 'BOSS_sky_validate.sky%.f.hist.png' % ww), 
                bbox_inches='tight') 
    return None

#########################################################
# DECam 
#########################################################

def decam_sky(overwrite=False): 
    ''' read in decam sky data 
    '''
    fpickle = os.path.join(UT.dat_dir(), 'decam_sky.p')
    if not os.path.isfile(fpickle) or overwrite: 
        fdecam = fits.open(os.path.join(UT.dat_dir(), 'decalsobs-zpt-dr3-allv2.fits'))
        _decam = fdecam[1].data
        
        keep = ((_decam['AIRMASS'] != 0.0) & 
                (_decam['TRANSP'] > .75) & 
                (_decam['TRANSP'] < 1.3)) 
        
        decam = {} 
        for k in _decam.names: 
            decam[k.lower()] = _decam[k][keep]
        
        # calculate moon altitude and moon separation 
        time = astropy.time.Time(decam['date'], format='jd')
        location = astropy.coordinates.EarthLocation.from_geodetic(
                lat='-30d10m10.78s', lon='-70d48m23.49s', height=2241.4*u.m)

        moon_position = astropy.coordinates.get_moon(time, location)
        moon_ra = moon_position.ra.value
        moon_dec = moon_position.dec.value
        
        moon_position_altaz = moon_position.transform_to(astropy.coordinates.AltAz(obstime=time, location=location))
        decam['moon_alt']   = moon_position_altaz.alt.value

        sun_position = astropy.coordinates.get_sun(time) 
        decam['moon_sun_sep'] = sun_position.separation(moon_position).deg
        pickle.dump(decam, open(fpickle, 'wb'))
    else: 
        decam = pickle.load(open(fpickle, 'rb'))
    return decam 


def DECam_sky_validate(): 
    ''' compare the sky models to the DECam sky magnitudes 
    '''
    decam = decam_sky() 
    n_sky = len(decam['airmass'])

    for i in range(n_sky): 
        w_ks, ks_i      = sky_KS(decam['airmass'][i], decam['moonphase'][i], 
                decam['moon_alt'][i], decam['moonsep'][i])
        w_nks, newks_i  = sky_KSrescaled_twi(decam['airmass'][i], decam['moonphase'][i], 
                decam['moon_alt'][i], decam['moonsep'][i], decam['sunalt'][i], decam['sunsep'][i])
        w_eso, eso_i  = sky_pseudoESO(decam['airmass'][i], decam['moonphase'][i], 
                decam['moon_alt'][i], decam['moonsep'][i], decam['sunalt'][i], decam['sunsep'][i])
        #w_eso, eso_i    = sky_ESO(decam['airmass'][i], decam['moon_sun_sep'][i], decam['moon_alt'][i], decam['moonsep'][i])
    
        if i == 0: 
            grz_ks  = np.zeros(n_sky)
            grz_nks = np.zeros(n_sky)
            grz_eso = np.zeros(n_sky)
            
        grz_ks[i]   = get_mag(w_ks, ks_i, decam['filter'][i]) 
        grz_nks[i]  = get_mag(w_nks, newks_i, decam['filter'][i]) 
        grz_eso[i]  = get_mag(w_eso, eso_i, decam['filter'][i]) 
    
    hasg = (decam['filter'] == 'g')
    hasr = (decam['filter'] == 'r')
    hasz = (decam['filter'] == 'z')
    
    # --- sky photometry comparison --- 
    fig = plt.figure(figsize=(17,5))
    
    sub = fig.add_subplot(131)
    sub.scatter(decam['skybr'][hasg], grz_ks[hasg],  c='C0', s=10)
    sub.scatter(decam['skybr'][hasg], grz_nks[hasg], c='C1', s=10)
    sub.scatter(decam['skybr'][hasg], grz_eso[hasg], c='C2', s=10)
    sub.plot([16, 22], [16, 22], c='k', ls='--')
    sub.set_xlabel('DECam $g$ band', fontsize=20)
    sub.set_xlim([20., 22])
    sub.set_ylabel('sky model magnitude', fontsize=20)
    sub.set_ylim([20., 22])

    sub = fig.add_subplot(132)
    sub.scatter(decam['skybr'][hasr], grz_ks[hasr],  c='C0', s=10)
    sub.scatter(decam['skybr'][hasr], grz_nks[hasr], c='C1', s=10)
    sub.scatter(decam['skybr'][hasr], grz_eso[hasr], c='C2', s=10)
    sub.plot([16, 22], [16, 22], c='k', ls='--')
    sub.set_xlabel('DECam $r$ band', fontsize=20)
    sub.set_xlim([19, 22])
    sub.set_ylim([19, 22])

    sub = fig.add_subplot(133)
    sub.scatter(decam['skybr'][hasz], grz_ks[hasz],  c='C0', s=10, label='original KS')
    sub.scatter(decam['skybr'][hasz], grz_nks[hasz], c='C1', s=10, label='rescaled KS+twi')
    sub.scatter(decam['skybr'][hasz], grz_eso[hasz], c='C2', s=10, label='pseudo ESO')
    sub.plot([16, 22], [16, 22], c='k', ls='--')
    sub.legend(loc='upper left', markerscale=2, handletextpad=0, fontsize=20)
    sub.set_xlabel('DECam $z$ band', fontsize=20)
    sub.set_xlim([16, 22])
    sub.set_ylim([16, 22])
    fig.savefig(os.path.join(UT.code_dir(), 'figs', 'DECAM_sky_validate.sky_photo.png'), 
            bbox_inches='tight') 

    # --- sky photometry comparison --- 
    fig = plt.figure(figsize=(17,5))
    sub = fig.add_subplot(131)
    sub.hist(decam['skybr'][hasg] - grz_ks[hasg],  range=(-3., 3.), bins=20, color='C0')
    sub.hist(decam['skybr'][hasg] - grz_nks[hasg], range=(-3., 3.), bins=20, color='C1')
    sub.hist(decam['skybr'][hasg] - grz_eso[hasg], range=(-3., 3.), bins=20, color='C2')
    sub.set_xlabel('DECam mag - (sky mag) $g$', fontsize=20)
    sub.set_xlim(-5., 5.)

    sub = fig.add_subplot(132)
    sub.hist(decam['skybr'][hasr] - grz_ks[hasr],  range=(-3., 3.), bins=20, color='C0')
    sub.hist(decam['skybr'][hasr] - grz_nks[hasr], range=(-3., 3.), bins=20, color='C1')
    sub.hist(decam['skybr'][hasr] - grz_eso[hasr], range=(-3., 3.), bins=20, color='C2')
    sub.set_xlabel('DECam mag - (sky mag) $r$', fontsize=20)
    sub.set_xlim(-5., 5.)

    sub = fig.add_subplot(133)
    sub.hist(decam['skybr'][hasz] - grz_ks[hasz],  range=(-3., 3.), bins=20, color='C0', label='original KS')
    sub.hist(decam['skybr'][hasz] - grz_nks[hasz], range=(-3., 3.), bins=20, color='C1', label='rescaled KS+twi')
    sub.hist(decam['skybr'][hasz] - grz_eso[hasz], range=(-3., 3.), bins=20, color='C2', label='pseudo ESO')
    sub.legend(loc='upper left', markerscale=2, handletextpad=0, fontsize=20)
    sub.set_xlabel('DECam mag - (sky mag) $z$', fontsize=20)
    sub.set_xlim(-5., 5.)
    fig.savefig(os.path.join(UT.code_dir(), 'figs', 'DECAM_sky_validate.sky_photo.hist.png'), 
            bbox_inches='tight') 
    return None


def get_mag(wsky, Isky, band): 
    ''' get magnitude of sky surface brightness
    '''
    Isky *= 1e-17 # erg/s/cm^2/A

    filter_response = filters.load_filter('decam2014-{}'.format(band))
    moon_flux, sky_wlen = filter_response.pad_spectrum(Isky, wsky)
    sky_brightness = filter_response.get_ab_maggies(moon_flux, sky_wlen)
    return flux_to_mag(sky_brightness) 


def flux_to_mag(flux):
    return 22.5 - 2.5*np.log10(flux*10**9)

#########################################################
# sky models
#########################################################
def sky_KS(airmass, moonill, moonalt, moonsep):
    ''' calculate original KS sky model 
    '''
    specsim_wave = specsim_sky._wavelength # Ang
    specsim_sky.airmass = airmass
    specsim_sky.moon.moon_phase = np.arccos(2.*moonill - 1)/np.pi
    specsim_sky.moon.moon_zenith = (90. - moonalt) * u.deg
    specsim_sky.moon.separation_angle = moonsep * u.deg
    # original KS coefficient 
    specsim_sky.moon.KS_CR = 10**5.36 
    specsim_sky.moon.KS_CM0 = 6.15 
    specsim_sky.moon.KS_CM1 = 40.
    return specsim_wave.value, specsim_sky.surface_brightness.value


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
    
    I_ks_rescale = specsim_sky.surface_brightness
    Isky = I_ks_rescale.value
    if sun_alt > -20.: # adding in twilight
        w_twi, I_twi = cI_twi(sun_alt, sun_sep, airmass)
        I_twi /= np.pi
        I_twi_interp = interp1d(10. * w_twi, I_twi, fill_value='extrapolate')
        Isky += I_twi_interp(specsim_wave.value)
    return specsim_wave.value, Isky


def sky_pseudoESO(airmass, moonill, moonalt, moonsep, sun_alt, sun_sep):
    ''' sky brightness using the KS parameterization with coefficients of the 
    ESO sky model. 
    '''
    #specsim_sky = Sky.specsim_initialize('desi')
    specsim_wave = specsim_sky._wavelength # Ang
    specsim_sky.airmass = airmass
    specsim_sky.moon.moon_phase = np.arccos(2.*moonill - 1)/np.pi
    specsim_sky.moon.moon_zenith = (90. - moonalt) * u.deg
    specsim_sky.moon.separation_angle = moonsep * u.deg
    
    # updated KS coefficients 
    specsim_sky.moon.KS_CR = 10**5.70 
    specsim_sky.moon.KS_CM0 = 7.15
    specsim_sky.moon.KS_CM1 = 40.
    
    Isky = specsim_sky.surface_brightness.value
    return specsim_wave.value, Isky


def sky_ESO(airmass, moon_sun_sep, moonalt, moonsep, observatory='2400'): 
    ''' calculate sky brightness using ESO sky calc 

    :param airmass: 
        airmass ranging from [1., 3.]

    :param moon_sun_sep:
        Separation in deg of Sun and Moon as seen from Earth

    :param moonalt: 
        Moon Altitude over Horizon in deg

    :param moonsep: 
        Moon-Target Separation in deg

    :references: 
    - https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC
    - https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html
    '''
    airmassp = 1./(1./airmass + 0.025*np.exp(-11./airmass)) # correct sec(z) airmass to Rozenberg (1966) airmass
    dic = {'airmass': round(airmass,5), 
            'incl_moon': 'Y', 
            'moon_sun_sep': moon_sun_sep, 
            'moon_target_sep': moonsep, 
            'moon_alt': moonalt, 
            'wmin': 355.,
            'wmax': 985., 
            'observatory': observatory}
    skyModel = skycalc.SkyModel()
    skyModel.callwith(dic)
    ftmp = os.path.join(UT.dat_dir(), 'sky', '_tmp.fits')
    skyModel.write(ftmp) # the clunkiest way to deal with this ever. 

    f = fits.open(ftmp) 
    fdata = f[1].data 
    wave = fdata['lam'] * 1e4       # wavelength in Ang 
    radiance = fdata['flux']        # photons/s/m^2/microm/arcsec^2 (radiance -- fuck)
    radiance *= 1e-8                # photons/s/cm^2/Ang/arcsec^2 
    # photons/s/cm^2/Ang/arcsec^2 * h * c / lambda 
    Isky = 1.99 * 1e-8 * radiance / wave * 1e17 # 10^-17 erg/s/cm^2/Ang/arcsec^2
    return wave, Isky


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


def _Noll_sky_ESO(): 
    ''' try to reproduce plots in Noll et al. (2012) using the ESO sky model 
    '''
    # Noll+(2012) Figure 1 generated from parameters in Table 1 
    dic = {
            'airmass': 1.0366676717,   # skysim.zodiacal.airmass_zodi(90 - 85.1) (based on alitutde) 
            'moon_sun_sep': 77.9,       # separation of sun and moon 
            'moon_target_sep': 51.3,    # separation of moon and target 
            'moon_alt': 41.3,           # altitude of the moon above the horizon
            'moon_earth_dist': 1.,      # relative distance to the moon  
            'ecl_lon': -124.5,          # heliocentric eclipitic longitude
            'ecl_lat': -31.6,           # heliocentric eclipitic latitude
            'msolflux': 205.5,          # monthly-averaged solar radio flux at 10.7 cm
            'pwv_mode': 'season',       # pwv or season
            'season': 4, 
            'time': 3, 
            'vacair': 'air', 
            'wmin': 300.,
            'wmax': 4200., 
            'wdelta': 5.,
            'observatory': '2640'}
    skyModel = skycalc.SkyModel()
    skyModel.callwith(dic)
    ftmp = os.path.join(UT.dat_dir(), 'sky', '_tmp.fits')
    skyModel.write(ftmp) # the clunkiest way to deal with this ever. 

    f = fits.open(ftmp) 
    fdata = f[1].data 
    wave = fdata['lam']             # wavelength in Ang 
    radiance = fdata['flux']        # photons/s/m^2/microm/arcsec^2 (radiance -- fuck)
    print fdata.names 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.plot(wave, np.log10(radiance), c='k', lw=0.5, label='composite') 
    for k, lbl, clr in zip(['flux_sml', 'flux_ssl', 'flux_zl', 'flux_tie', 'flux_tme', 'flux_ael', 'flux_arc'], 
            ['moon', 'stars', 'zodiacal', 'telescope', 'lower atmos.', 'airglow lines', 'resid. cont.'], ['b', 'C0', 'g', 'r', 'y', 'm', 'cyan']): 
        sub.plot(wave, np.log10(fdata[k]), lw=0.5, c=clr, label=lbl) 
    sub.legend(loc='lower right', frameon=True, fontsize=15) 
    sub.set_xlabel(r'Wavelength [$\mu m$]', fontsize=20) 
    sub.set_xlim(0.2, 4.2) 
    sub.set_ylabel(r'Radiance [dex]', fontsize=20) 
    sub.set_ylim(-1.5, 7.5) 
    fig.savefig(os.path.join(UT.code_dir(), 'figs', '_Nollfig1_sky_ESO.png'), bbox_inches='tight') 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    w0p5 = np.abs(wave - 0.5).argmin() 
    sub.plot(wave, (fdata['flux_sml']/fdata['flux_sml'][w0p5]), lw=0.5, c='b', label='Moon') 
    sub.plot(wave, (fdata['flux_ssl']/fdata['flux_ssl'][w0p5]), lw=0.5, c='g', ls='--', label='stars')  
    sub.plot(wave, (fdata['flux_zl']/fdata['flux_zl'][w0p5]), lw=0.5, c='r', ls='-.', label='zodiacal light') 
    sub.legend(loc='lower right', fontsize=15) 
    sub.set_xlabel(r'Wavelength [$\mu m$]', fontsize=20) 
    sub.set_xlim(0.36, 0.885) 
    sub.set_ylabel(r'rel.radiance', fontsize=20) 
    sub.set_ylim(0.,1.8) 
    fig.savefig(os.path.join(UT.code_dir(), 'figs', '_Nollfig6_sky_ESO.png'), bbox_inches='tight') 
    return None 


def _sky_ESOvsKSvband(): 
    '''
    '''
    # get default ESO moon scatterlight brightness 
    dic = {
            'airmass': 1.0366676717,   # skysim.zodiacal.airmass_zodi(90 - 85.1) (based on alitutde) 
            'moon_sun_sep': 77.9,       # separation of sun and moon 
            'moon_target_sep': 51.3,    # separation of moon and target 
            'moon_alt': 41.3,           # altitude of the moon above the horizon
            'moon_earth_dist': 1.,      # relative distance to the moon  
            'ecl_lon': -124.5,          # heliocentric eclipitic longitude
            'ecl_lat': -31.6,           # heliocentric eclipitic latitude
            'msolflux': 205.5,          # monthly-averaged solar radio flux at 10.7 cm
            'pwv_mode': 'season',       # pwv or season
            'season': 4, 
            'time': 3, 
            'vacair': 'air', 
            'wmin': 300.,
            'wmax': 4200., 
            'wdelta': 5.,
            'observatory': '2640'}
    
    skyModel = skycalc.SkyModel()
    skyModel.callwith(dic)
    ftmp = os.path.join(UT.dat_dir(), 'sky', '_tmp.fits')
    skyModel.write(ftmp) # the clunkiest way to deal with this ever. 

    f = fits.open(ftmp) 
    fdata = f[1].data 
    wave_eso = fdata['lam']         # wavelength in Ang 
    radiance = fdata['flux_sml']    # photons/s/m^2/microm/arcsec^2 (radiance -- fuck)
    radiance *= 1e-8                # photons/s/cm^2/Ang/arcsec^2 
    Im_eso = 1.99 * 1e-8 * radiance / (wave_eso * 1e4) * 1e17 # 10^-17 erg/s/cm^2/Ang/arcsec^2
    
    # get moon brightness where some moon spectra is caled by KS V-band 
    specsim_sky = Sky.specsim_initialize('desi')
    specsim_wave = specsim_sky._wavelength # Ang
    specsim_sky.airmass = 1.0366676717 
    specsim_sky.moon.moon_phase = 77.9/180. #np.arccos(2.*moonill - 1)/np.pi
    specsim_sky.moon.moon_zenith = (90. - 41.3) * u.deg
    specsim_sky.moon.separation_angle = 51.3 * u.deg
    
    # updated KS coefficients 
    specsim_sky.moon.KS_CR = 10**5.70 
    specsim_sky.moon.KS_CM0 = 7.15
    specsim_sky.moon.KS_CM1 = 40.
    
    Im_ks = specsim_sky.moon.surface_brightness.value
    
    rho = specsim_sky.moon.separation_angle.to(u.deg).value
    fR = 10**5.36*(1.06 + np.cos(np.radians(rho))**2)
    fM = 10 ** (6.15 -  rho/ 40.)
    fRp = 10**5.70*(1.06 + np.cos(np.radians(rho))**2)
    fMp = 10 ** (7.15 -  rho/ 40.)
    fRn = 10**5.66*(1.06 + np.cos(np.radians(rho))**2)
    fMn = 10 ** (5.54 -  rho/ 178.)
    tRS = np.interp(specsim_wave/1e4, wave_eso, fdata['trans_rs']) 
    tMS = np.interp(specsim_wave/1e4, wave_eso, fdata['trans_ms']) 
    tall = np.interp(specsim_wave/1e4, wave_eso, fdata['trans']) 
    Xo = (1 - 0.96 * np.sin(specsim_sky.moon.obs_zenith)**2)**(-0.5)
    Xm = (1 - 0.96 * np.sin(specsim_sky.moon.moon_zenith)**2)**(-0.5) 
    tkso = 10 ** (-0.4 * (specsim_sky.moon.vband_extinction * Xo))
    tksm = 10 ** (-0.4 * (specsim_sky.moon.vband_extinction * Xm))
    fcorr = ((fRp*(1.-tRS**Xo) + fMp*(1.-tMS**Xo))*tall**Xm)/((fRp+fMp) * (1-tkso) * tksm)
    f_eso_ks = ((fRp*(1.-tRS**Xo) + fMp*(1.-tMS**Xo))*tall**Xm)/((fR+fM) * (1-tkso) * tksm)
    f_nks_ks = (fRn + fMn)/(fR + fM) 
    f_peso_ks = (fRp + fMp)/(fR + fM) 
    
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.plot(wave_eso, Im_eso, c='k', label='ESO refit KS')
    sub.plot(specsim_wave/1e4, fcorr*Im_ks * 1e17, c='C1', label='V-band scale ESO coeff.') 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'Wavelength [$\mu m$]', fontsize=20) 
    sub.set_xlim(0.3, 1.) 
    sub.set_ylabel(r'Moon Brightness', fontsize=20) 
    fig.savefig(os.path.join(UT.code_dir(), 'figs', '_sky_ESOvsKSvband.png'), bbox_inches='tight') 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.plot(specsim_wave, f_eso_ks, c='k', label='ESO')
    sub.plot(specsim_wave, np.repeat(f_nks_ks, len(specsim_wave)), c='C0', label='refit KS')
    sub.plot(specsim_wave, np.repeat(f_peso_ks, len(specsim_wave)), c='C1', label='pseudo ESO')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'Wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3.4e3, 9.8e3) 
    sub.set_ylabel(r'(moon model)/(KS moon)', fontsize=20) 
    fig.savefig(os.path.join(UT.code_dir(), 'figs', '_sky_ESOoverKS.png'), bbox_inches='tight') 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.plot(wave_eso, fdata['trans_rs'], c='k', label='Rayleigh')
    sub.plot(wave_eso, fdata['trans_ms'], c='C1', label='Mie') 
    sub.plot(wave_eso, np.repeat(10**(-0.4*specsim_sky.moon.vband_extinction*dic['airmass']), len(wave_eso)))
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'Wavelength [$\mu m$]', fontsize=20) 
    sub.set_xlim(0.3, 1.) 
    sub.set_ylabel(r'transmission', fontsize=20) 
    fig.savefig(os.path.join(UT.code_dir(), 'figs', '_sky_ESOvsKSvband.trans.png'), bbox_inches='tight') 
    return None 


def _test_sky_ESO(): 
    boss = boss_sky() # read in BOSS sky data 

    fig = plt.figure(figsize=(10,10)) 
    sub = fig.add_subplot(311) 
    for obs in ['2400', '2640', '3060']: 
        w_eso, eso_i = sky_ESO(boss['AIRMASS'][0], boss['SUN_MOON_SEP'][0], boss['MOON_ALT'][0], boss['MOON_SEP'][0], 
                observatory=obs)
        sub.plot(w_eso, eso_i, lw=1, label='%s Observatory' % obs) 
    sub.legend(loc='upper left', fontsize=15) 
    sub.set_xlim(3550, 9850) 
    sub.set_ylim(0., 10)

    sub = fig.add_subplot(312) 
    for am in [1.2, 1.5, 1.7]: 
        print('airmass = %f' % am) 
        w_eso, eso_i = sky_ESO(am, boss['SUN_MOON_SEP'][0], boss['MOON_ALT'][0], boss['MOON_SEP'][0])
        sub.plot(w_eso, eso_i, lw=1, label='airmass = %.1f' % am) 
    sub.legend(loc='upper left', fontsize=15) 
    sub.set_xlim(3550, 9850) 
    sub.set_ylim(0., 10)

    sub = fig.add_subplot(313) 
    for ms in [20, 40, 60, 80]: 
        print('moon sep = %f' % ms) 
        w_eso, eso_i = sky_ESO(boss['AIRMASS'][0], boss['SUN_MOON_SEP'][0], boss['MOON_ALT'][0], ms)
        sub.plot(w_eso, eso_i, lw=1, label='moon sep. = %.f' % ms) 
    sub.legend(loc='upper left', fontsize=15) 
    sub.set_xlim(3550, 9850) 
    sub.set_ylim(0., 10)

    bkgd = fig.add_subplot(111, frameon=False) # x,y labels
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel('Wavelength [Angstrom]', fontsize=20) 
    bkgd.set_ylabel('Sky Brightness [$10^{-17} erg/s/cm^2/\AA/arcsec^2$]', fontsize=20) 
    fig.savefig(os.path.join(UT.code_dir(), 'figs', '_test_sky_ESO.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #twilight_coeffs()
    BOSS_sky_validate()
    #decam_sky(overwrite=True)
    #DECam_sky_validate()
    #_Noll_sky_ESO()
    #_sky_ESOvsKSvband()
    #_test_sky_ESO()
