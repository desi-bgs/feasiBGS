'''

validating the sky model

'''
import os
import pickle
import numpy as np 
# -- astropy --
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


def boss_sky(): 
    ''' read in BOSS sky data -- fluxes and meta-data 
    '''
    fboss = os.path.join(UT.dat_dir(), 'sky', 'Bright_BOSS_Sky_blue.fits')
    boss = aTable.read(fboss)
    return boss


def BOSS_sky_validate(): 
    ''' validate sky models again BOSS sky data 
    '''
    boss = boss_sky() # read in BOSS sky data 
    n_sky = len(boss) 

    # calcuate the different sky models
    for i in range(3):#n_sky): 
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
    for i in range(3): 
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
    return None

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
        I_twi = cI_twi(sun_alt, sun_sep, airmass)/np.pi
        I_twi_interp = interp1d(10.*np.array(coeffs['wl']), I_twi, fill_value='extrapolate')
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
    return np.array(twi)


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
