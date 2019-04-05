'''

validating the sky model

'''
import os
import pickle
import numpy as np 
from scipy.interpolate import interp1d
import speclite
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
    n_sky = 50 # len(boss) 
    print('%i sky fibers' % n_sky)

    # calcuate the different sky models
    for i in range(n_sky): 
        print('%i of %i' % (i+1, n_sky))
        w_ks, ks_i      = sky_KS(boss['AIRMASS'][i], boss['MOON_ILL'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i])
        w_nks, newks_i  = sky_KSrescaled_twi(boss['AIRMASS'][i], boss['MOON_ILL'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i], 
                boss['SUN_ALT'][i], boss['SUN_SEP'][i])
        w_eso, eso_i    = sky_ESO(boss['AIRMASS'][i], boss['SUN_MOON_SEP'][i], boss['MOON_ALT'][i], boss['MOON_SEP'][i])

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
        sub.plot(w_eso, Isky_eso[i,:], label='ESO') 
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
    ks_4100lim      = (w_ks > 4050.) & (w_ks > 4150.)
    nks_4100lim     = (w_nks > 4050.) & (w_nks > 4150.)
    eso_4100lim     = (w_eso > 4050.) & (w_eso > 4150.) 
    
    boss_4100, ks_4100, nks_4100, eso_4100 = np.zeros(n_sky), np.zeros(n_sky), np.zeros(n_sky), np.zeros(n_sky)
    for i in range(n_sky): 
        boss_4100lim= (boss['WAVE'][i] > 405.) & (boss['WAVE'][i] < 415.) 
        boss_4100[i]= np.median(boss['SKY'][i][boss_4100lim])/np.pi
        ks_4100[i]  = np.median(Isky_ks[i,:][ks_4100lim])
        nks_4100[i] = np.median(Isky_newks[i,:][nks_4100lim])
        eso_4100[i] = np.median(Isky_eso[i,:][eso_4100lim])

    fig = plt.figure(figsize=(10, 25))
    for i, k in enumerate(['MOON_ILL', 'MOON_ALT', 'MOON_SEP', 'AIRMASS', 'SUN_ALT']): 
        sub = fig.add_subplot(5,1,i+1)
        sub.scatter(boss[k][:n_sky], boss_4100/ks_4100, c='C0', s=3, label='original KS')
        sub.scatter(boss[k][:n_sky], boss_4100/nks_4100, c='C1', s=3, label='rescaled KS + twi')
        sub.scatter(boss[k][:n_sky], boss_4100/eso_4100, c='C2', s=3, label='ESO') 
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
    bkgd.set_ylabel('(BOSS Sky)/(sky model) at 4100A', fontsize=25)
    fig.savefig(os.path.join(UT.code_dir(), 'figs', 'BOSS_sky_validate.sky4100.png'), 
            bbox_inches='tight') 
    return None


#########################################################
# DECam 
#########################################################

def decam_sky(): 
    ''' read in decam sky data 
    '''
    fpickle = os.path.join(UT.dat_dir(), 'decam_sky.p')
    if not os.path.isfile(fpickle): 
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
        decam['moon_az']    = moon_position_altaz.az.value
        decam['moon_sep']   = RADec_separation(decam['ra'], decam['dec'], ra2=moon_ra, dec2=moon_dec)
        pickle.dump(decam, open(fpickle, 'wb'))
    else: 
        decam = pickle.load(open(fpickle, 'rb'))
    return decam 


def DECam_sky_validate(): 
    '''
    '''
    decam = decam_sky() 
    n_sky = 10# len(decam['airmass'])

    for i in range(n_sky): 
        print('%i of %i' % (i, n_sky))
        w_ks, ks_i      = sky_KS(decam['airmass'][i], decam['moonphase'][i], decam['moon_alt'][i], decam['moon_sep'][i])
        w_nks, newks_i  = sky_KSrescaled_twi(decam['airmass'][i], decam['moonphase'][i], decam['moon_alt'][i], decam['moon_sep'][i], 
                decam['sun_alt'][i], decam['sun_sep'][i])
        w_eso, eso_i    = sky_ESO(decam['airmass'][i], decam['sun_moon_sep'][i], decam['moon_alt'][i], decam['moon_sep'][i])
    
        if i == 0: 
            grz_ks  = np.zeros(len(w_ks))
            grz_nks = np.zeros(len(w_nks))
            grz_eso = np.zeros(len(w_eso))
            
        grz_ks[i]   = get_mag(w_ks, ks_i, decam['filter']) 
        grz_nks[i]  = get_mag(w_nks, newks_i, decam['filter']) 
        grz_eso[i]  = get_mag(w_eso, eso_i, decam['filter']) 
    
    hasg = (decam['filter'] == 'g')
    hasr = (decam['filter'] == 'r')
    hasz = (decam['filter'] == 'z')
    
    # --- sky photometry comparison --- 
    fig = plt.figure(figsize=(17,5))
    
    sub = fig.add_subplot(131)
    sub.scatter(decam['skybr'][hasg], grz_ks[hasg], s=10, lw=1)
    sub.scatter(decam['skybr'][hasg], grz_nks[hasg], s=10, lw=1)
    sub.scatter(decam['skybr'][hasg], grz_eso[hasg], s=10, lw=1)
    sub.plot([16, 22], [16, 22], c='k', ls='--')
    sub.set_xlabel('DECam $g$ band', fontsize=20)
    sub.set_xlim([20., 22])
    sub.set_ylabel('sky model magnitude', fontsize=20)
    sub.set_ylim([20., 22])

    sub = fig.add_subplot(132)
    sub.scatter(decam['skybr'][hasr], grz_ks[hasr], s=10, lw=1)
    sub.scatter(decam['skybr'][hasr], grz_nks[hasr], s=10, lw=1)
    sub.scatter(decam['skybr'][hasr], grz_eso[hasr], s=10, lw=1)
    sub.plot([16, 22], [16, 22], c='k', ls='--')
    sub.set_xlabel('DECam $r$ band', fontsize=20)
    sub.set_xlim([19, 22])
    sub.set_ylim([19, 22])

    sub = fig.add_subplot(133)
    sub.scatter(decam['skybr'][hasz], grz_ks[hasz], s=10, lw=1)
    sub.scatter(decam['skybr'][hasz], grz_nks[hasz], s=10, lw=1)
    sub.scatter(decam['skybr'][hasz], grz_eso[hasz], s=10, lw=1)
    sub.plot([16, 22], [16, 22], c='k', ls='--')
    sub.legend(loc='upper left', markerscale=10, handletextpad=0, fontsize=20)
    sub.set_xlabel('DECam $z$ band', fontsize=20)
    sub.set_xlim([16, 22])
    sub.set_ylim([16, 22])
    fig.savefig(os.path.join(UT.code_dir(), 'figs', 'DECAM_sky_validate.sky_photo.png'), 
            bbox_inches='tight') 
    return None


def RADec_separation(ra1, dec1, ra2, dec2):
    pi2 = np.radians(90)
    alpha = np.cos(np.radians(ra1)-np.radians(ra2))
    first = np.cos(pi2-np.radians(dec1))*np.cos(pi2-np.radians(dec2))
    second = np.sin(pi2-np.radians(dec1))*np.sin(pi2-np.radians(dec2))*alpha
    return np.arccos(first+second)*180/np.pi


def get_mag(wsky, Isky, band): 
    ''' get magnitude of sky surface brightness
    '''
    Isky *= 1e-17 # erg/s/cm^2/A

    filter_response = speclite.filters.load_filter('decam2014-{}'.format(band))
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
    specsim_sky = Sky.specsim_initialize('desi')
    specsim_wave = specsim_sky._wavelength # Ang
    specsim_sky.airmass = airmass
    specsim_sky.moon.moon_phase = np.arccos(2.*moonill - 1)/np.pi
    specsim_sky.moon.moon_zenith = (90. - moonalt) * u.deg
    specsim_sky.moon.separation_angle = moonsep * u.deg
    return specsim_wave.value, specsim_sky.surface_brightness.value


def sky_KSrescaled_twi(airmass, moonill, moonalt, moonsep, sun_alt, sun_sep):
    ''' calculate sky brightness using rescaled KS coefficients plus a twilight
    factor from Parker. 

    :return specsim_wave, Isky: 
        returns wavelength [Angstrom] and sky surface brightness [$10^{-17} erg/cm^{2}/s/\AA/arcsec^2$]
    '''
    specsim_sky = Sky.specsim_initialize('desi')
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


def sky_ESO(airmass, moon_sun_sep, moonalt, moonsep): 
    ''' calculate sky brightness using ESO sky calc 

    :references: 
    - https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC
    - https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html
    '''
    dic = {'airmass': round(airmass,3), 
            'incl_moon': 'Y', 
            'moon_sun_sep': moon_sun_sep, 
            'moon_target_sep': moonsep, 
            'moon_alt': moonalt, 
            'observatory': '2400'}
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
    Isky = 3. * 6.6 * 1e-9 * radiance / wave * 1e17 # 10^-17 erg/s/cm^2/Ang/arcsec^2
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


if __name__=="__main__": 
    #twilight_coeffs()
    BOSS_sky_validate()
    #DECam_sky_validate()
