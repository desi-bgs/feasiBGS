'''
'''
import os 
import pickle
import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
# -- astropy --
import astropy.units as u
from astropy.time import Time
# -- astroplan -- 
from astroplan import Observer
from astroplan import download_IERS_A
# -- specsim -- 
import specsim
from specsim.atmosphere import Moon 
# -- feasibgs -- 
from . import util as UT 


def Isky_newKS_twi(airmass, moonill, moonalt, moonsep, sunalt, sunsep):
    ''' Sky surface brightness as a function of airmass, moon parameters, and sun parameters.
    The sky surface brightness uses the KS model scaling with coefficients re-fit to match
    BOSS sky data and includes a twilight contribution from Parker's thesis. 

    :param airmass: 
        airmass 
    
    :param moonill:  
        moon illumination fraction: 0 - 1 
    
    :param moonalt:  
        moon altitude: 0 - 90 deg 
    
    :param moonsep:  
        moon separation angle: 0 - 180 deg 
    
    :param sunalt:
        sun altitude: 0 - 90 deg 
    
    :param sunsep: 
        sun separation: 0 - 90 deg 

    :return specsim_wave, Isky: 
        returns wavelength [Angstrom] and sky surface brightness [$10^{-17} erg/cm^{2}/s/\AA/arcsec^2$]
    '''
    # initialize atmosphere model using hacked version of specsim.atmosphere.initialize 
    specsim_sky     = _specsim_initialize('desi')
    specsim_wave    = specsim_sky._wavelength # Ang
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
    
    # twilight contribution 
    if sunalt > -20.: 
        w_twi, I_twi = _cI_twi(sunalt, sunsep, airmass)
        I_twi /= np.pi
        I_twi_interp = interp1d(10. * w_twi, I_twi, fill_value='extrapolate')
        Isky += np.clip(I_twi_interp(specsim_wave), 0, None) 
    return specsim_wave, Isky


def Isky_parker(airmass, ecl_lat, gal_lat, gal_lon, tai, sun_alt, sun_sep, moon_phase, moon_ill, moon_alt, moon_sep): 
    ''' Parker's sky model, which is a function of: 

    :param airmass: 
        airmass

    :param ecl_lat: 
        ecliptic latitude (used for zodiacal light contribution) 

    :param gal_lat: 
        galactic latitude (used for ISL contribution) 
    
    :param gal_lon: 
        galactic longitude (used for ISL contribution) 

    :param tai: 
        time in seconds 
    
    :param sunalt:
        sun altitude: 0 - 90 deg 
    
    :param sunsep: 
        sun separation: 0 - 90 deg 
    
    :param moonill:  
        moon illumination fraction: 0 - 1 
    
    :param moonalt:  
        moon altitude: 0 - 90 deg 
    
    :param moonsep:  
        moon separation angle: 0 - 180 deg 
    
    '''
    from astropy.coordinates import EarthLocation
    X = airmass    # air mass 
    beta = ecl_lat # ecliptic latitude ( used for zodiacal light contribution ) 
    l = gal_lat    # galactic latitude ( used for ISL contribution ) 
    b = gal_lon    # galactic longitude ( used for ISL contribution ) 

    _kpno = EarthLocation.of_site('kitt peak')
    obs_time = Time(tai/86400., scale='tai', format='mjd', location=_kpno)
    mjd = obs_time.mjd

    # fractional months ( used for seasonal contribution) 
    month_frac = obs_time.datetime.month + obs_time.datetime.day/30. 
    
    # fractional hour ( used for hourly contribution) 
    kpno = Observer(_kpno)
    sun_rise    = kpno.sun_rise_time(obs_time, which='next')
    sun_set     = kpno.sun_set_time(obs_time, which='previous')
    hour        = ((obs_time - sun_set).sec)/3600.
    hour_frac   = hour/((Time(sun_rise, format='mjd') - Time(sun_set,format = 'mjd')).sec/3600.)

    alpha = sun_alt    # sun altitude
    delta = sun_sep    # sun separation (separation between the target and the sun's location)
    
    # used for scattered moonlight
    g = moon_phase     # moon phase 
    altm = moon_alt
    illm = moon_ill
    delm = moon_sep
    
    # get coefficients 
    coeffs = _read_parkerCoeffs()

    # sky continuum 
    _w, _Icont = _parker_Icontinuum(coeffs, X, beta, l, b, mjd, month_frac, hour_frac, alpha, delta, altm, illm, delm, g)
    S_continuum = _Icont / np.pi  # BOSS has 2 arcsec diameter

    # sky emission from the UVES continuum subtraction
    w_uves, S_uves = np.loadtxt(''.join([UT.code_dir(), 'dat/sky/UVES_sky_emission.dat']), 
            unpack=True, usecols=[0,1]) 
    f_uves = interp1d(w_uves, S_uves, bounds_error=False, fill_value='extrapolate')
    S_emission = f_uves(_w)

    return _w, S_continuum + S_emission 


def Isky_parker_radecobs(ra, dec, obs_time): 
    ''' wrapper for Isky_parker, where the input parameters are calculated based
    on RA, Dec, and obs_time 
    '''
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon
 
    download_IERS_A()
    # target coordinates 
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg) 
    # observed time (UTC)          
    utc_time = Time(obs_time)
    kpno = EarthLocation.of_site('kitt peak')

    kpno_altaz = AltAz(obstime=utc_time, location=kpno) 
    coord_altaz = coord.transform_to(kpno_altaz)

    airmass = coord_altaz.secz
    elc_lat = coord.barycentrictrueecliptic.lat.deg
    gal_lat = coord.galactic.l.deg   # galactic latitude ( used for ISL contribution ) 
    gal_lon = coord.galactic.b.deg   # galactic longitude ( used for ISL contribution ) 

    tai = utc_time.tai   

    # sun altitude (degrees)
    sun = get_sun(utc_time) 
    sun_altaz   = sun.transform_to(kpno_altaz) 
    sunalt      = sun_altaz.alt.deg
    # sun separation
    sunsep      = sun.separation(coord).deg

    # used for scattered moonlight
    moon = get_moon(utc_time)
    moon_altaz = moon.transform_to(kpno_altaz) 
    moon_alt = moon_altaz.alt.deg 
    moon_sep = moon.separation(coord).deg #coord.separation(self.moon).deg
            
    elongation  = sun.separation(moon)
    phase       = np.arctan2(sun.distance * np.sin(elongation), moon.distance - sun.distance*np.cos(elongation))
    moon_phase  = phase.value
    moon_ill    = (1. + np.cos(phase))/2.
    return Isky_parker(airmass, ecl_lat, gal_lat, gal_lon, tai, sun_alt, sun_sep, moon_phase, moon_ill, moon_alt, moon_sep)


def _specsim_initialize(config, model='regression'): 
    ''' hacked version of specsim.atmosphere.initialize, which initializes the 
    atmosphere model from configuration parameters.
    '''
    if specsim.config.is_string(config):
        config = specsim.config.load_config(config)

    atm_config = config.atmosphere

    # Load tabulated data.
    surface_brightness_dict = config.load_table(
        atm_config.sky, 'surface_brightness', as_dict=True)
    extinction_coefficient = config.load_table(
        atm_config.extinction, 'extinction_coefficient')

    # Initialize an optional atmospheric seeing PSF.
    psf_config = getattr(atm_config, 'seeing', None)
    if psf_config:
        seeing = dict(
            fwhm_ref=specsim.config.parse_quantity(psf_config.fwhm_ref),
            wlen_ref=specsim.config.parse_quantity(psf_config.wlen_ref),
            moffat_beta=float(psf_config.moffat_beta))
    else:
        seeing = None

    # Initialize an optional lunar scattering model.
    moon_config = getattr(atm_config, 'moon', None)
    if moon_config:
        moon_spectrum = config.load_table(moon_config, 'flux')
        c = config.get_constants(moon_config,
            ['moon_zenith', 'separation_angle', 'moon_phase'])
        moon = _Moon(
            config.wavelength, moon_spectrum, extinction_coefficient,
            atm_config.airmass, c['moon_zenith'], c['separation_angle'],
            c['moon_phase'], model=model)
    else:
        moon = None

    atmosphere = specsim.atmosphere.Atmosphere(
        config.wavelength, surface_brightness_dict, extinction_coefficient,
        atm_config.extinct_emission, atm_config.sky.condition,
        atm_config.airmass, seeing, moon)

    if config.verbose:
        print(
            "Atmosphere initialized with condition '{0}' from {1}."
            .format(atmosphere.condition, atmosphere.condition_names))
        if seeing:
            print('Seeing is {0} at {1} with Moffat beta {2}.'
                  .format(seeing['fwhm_ref'], seeing['wlen_ref'],
                          seeing['moffat_beta']))
        if moon:
            print(
                'Lunar V-band extinction coefficient is {0:.5f}.'
                .format(moon.vband_extinction))

    return atmosphere


class _Moon(Moon): 
    ''' specimsim.atmosphere.Moon object hacked to work with a Krisciunas & Schaefer (1991)
    model with extra free parameters
    '''
    def __init__(self, wavelength, moon_spectrum, extinction_coefficient,
            airmass, moon_zenith, separation_angle, moon_phase,
            model='regression'):
        # initialize via super function 
        super().__init__(wavelength, moon_spectrum, extinction_coefficient,
                airmass, moon_zenith, separation_angle, moon_phase)
        
        self.model = model
        # default KS coefficients 
        self.KS_CR = 10**5.36 # proportionality constant in the Rayleigh scattering function 
        # constants for the Mie scattering function term 
        self.KS_CM0 = 6.15 
        self.KS_CM1 = 40.

        self.KS_M0 = -12.73
        self.KS_M1 = 0.026
        self.KS_M2 = 4.

    def _update(self):
        """Update the model based on the current parameter values.
        """
        self._update_required = False

        # Calculate the V-band surface brightness of scattered moonlight.
        if self.model == 'refit_ks': 
            self._scattered_V = krisciunas_schaefer_free(
                self.obs_zenith, self.moon_zenith, self.separation_angle,
                self.moon_phase, self.vband_extinction, self.KS_CR, self.KS_CM0,
                self.KS_CM1, self.KS_M0, self.KS_M1, self.KS_M2)
        elif self.model == 'regression': 
            self._scattered_V = _scattered_V_regression(
                    self.airmass, 
                    0.5 * (np.cos(np.pi * self.moon_phase) + 1.), 
                    90 - self.moon_zenith, 
                    self.separation_angle) 
        else: 
            raise NotImplementedError 

        # Calculate the wavelength-dependent extinction of moonlight
        # scattered once into the observed field of view.
        scattering_airmass = (
            1 - 0.96 * np.sin(self.moon_zenith) ** 2) ** (-0.5)
        extinction = (
            10 ** (-self._extinction_coefficient * scattering_airmass / 2.5) *
            (1 - 10 ** (-self._extinction_coefficient * self.airmass / 2.5)))
        self._surface_brightness = self._moon_spectrum * extinction

        # Renormalized the extincted spectrum to the correct V-band magnitude.
        raw_V = self._vband.get_ab_magnitude(
            self._surface_brightness, self._wavelength) * u.mag

        area = 1 * u.arcsec ** 2
        self._surface_brightness *= 10 ** (
            -(self._scattered_V * area - raw_V) / (2.5 * u.mag)) / area

    @property
    def KS_CR(self):
        return self._KS_CR

    @KS_CR.setter
    def KS_CR(self, ks_cr):
        self._KS_CR = ks_cr 
        self._update_required = True

    @property
    def KS_CM0(self):
        return self._KS_CM0

    @KS_CM0.setter
    def KS_CM0(self, ks_cm0):
        self._KS_CM0 = ks_cm0 
        self._update_required = True
    
    @property
    def KS_CM1(self):
        return self._KS_CM1

    @KS_CM1.setter
    def KS_CM1(self, ks_cm1):
        self._KS_CM1 = ks_cm1 
        self._update_required = True

    @property
    def KS_M0(self):
        return self._KS_M0

    @KS_M0.setter
    def KS_M0(self, ks_m0):
        self._KS_M0 = ks_m0 
        self._update_required = True

    @property
    def KS_M1(self):
        return self._KS_M1

    @KS_M1.setter
    def KS_M1(self, ks_m1):
        self._KS_M1 = ks_m1 
        self._update_required = True

    @property
    def KS_M2(self):
        return self._KS_M2

    @KS_M2.setter
    def KS_M2(self, ks_m2):
        self._KS_M2 = ks_m2 
        self._update_required = True


reg_model_coeffs = np.array([-1.72523347e-02, -1.31361477e+02,  6.63730935e+01,
    -9.41030359e-02, 8.48961915e-01,  2.17497311e+02, -3.72597558e+02,
    1.00758992e+00, 3.08696159e-01,  2.67636483e+02,  8.35770985e-03,
    2.23588689e-01, 2.50034199e-03, -1.31082188e-02, -1.78405038e-02,
    -1.33394254e+02, 3.59991736e+02, -1.20301999e+00, -1.15591783e+00,
    -2.08943993e+02, 3.16669904e-01,  4.73848984e-01,  2.19641193e-03,
    1.03678764e-02, 1.05647754e-02,  5.97639216e+01, -3.29778699e+00,
    -3.53949915e+00, 1.15924972e-02,  2.26256920e-02,  1.10537908e-02,
    -9.30849402e-05, -9.24348638e-05, 6.08290452e-05,  7.47429090e-05,
    2.87743547e+01, -1.06004742e+02, 3.49781288e-01,  4.15966437e-01,
    5.02510051e+01, 2.33158177e-01, -8.99504191e-02, -1.55945957e-03,
    -1.67962218e-03, -1.48167087e-03, -6.79449418e+01,  9.24990594e-01,
    1.74126599e+00, -6.98329328e-03, -1.67072349e-02, -9.48908504e-03,
    1.71830097e-05, 1.53753423e-05, -1.20825222e-05, -1.10402948e-05,
    -6.82883124e+00, 3.42104080e-01,  1.15313005e-01,  1.08055005e-02,
    1.62673968e-02, 6.95873457e-03, -3.37508143e-05, -1.23513959e-04,
    -7.94721140e-05, -1.98438188e-05,  3.43658925e-07,  8.88223639e-07,
    5.96446942e-07, -1.29296893e-07, -1.62605517e-07]) 
reg_model_intercept = 35.87660110628718


def _scattered_V_regression(airmass, moon_frac, moon_alt, moon_sep):
    ''' 4th degree polynomial regression fit to the V-band scattered moonlight
    from BOSS and DESI CMX data. 
    '''
    theta = np.atleast_2d(np.array([airmass, moon_frac, moon_alt, moon_sep]).T)

    combs = chain.from_iterable(combinations_with_replacement(range(4), i) 
            for i in range(0, n_order+1))
    theta_transform = np.empty((theta.shape[0], len(reg_model_coeffs)))
    for i, comb in enumerate(combs):
        theta_transform[:, i] = theta[:, comb].prod(1)

    return np.dot(theta_transform, reg_model_coeffs.T) + reg_model_intercept


def krisciunas_schaefer_free(obs_zenith, moon_zenith, separation_angle, moon_phase,
                        vband_extinction, C_R, C_M0, C_M1, M0, M1, M2):
    """Calculate the scattered moonlight surface brightness in V band.

    Based on Krisciunas and Schaefer, "A model of the brightness of moonlight",
    PASP, vol. 103, Sept. 1991, p. 1033-1039 (http://dx.doi.org/10.1086/132921).
    Equation numbers in the code comments refer to this paper.

    The function :func:`plot_lunar_brightness` provides a convenient way to
    plot this model's predictions as a function of observation pointing.

    Units are required for the angular inputs and the result has units of
    surface brightness, for example:

    >>> sb = krisciunas_schaefer(20*u.deg, 70*u.deg, 50*u.deg, 0.25, 0.15)
    >>> print(np.round(sb, 3))
    19.855 mag / arcsec2

    The output is automatically broadcast over input arrays following the usual
    numpy rules.

    This method has several caveats but the authors find agreement with data at
    the 8% - 23% level.  See the paper for details.

    Parameters
    ----------
    obs_zenith : astropy.units.Quantity
        Zenith angle of the observation in angular units.
    moon_zenith : astropy.units.Quantity
        Zenith angle of the moon in angular units.
    separation_angle : astropy.units.Quantity
        Opening angle between the observation and moon in angular units.
    moon_phase : float
        Phase of the moon from 0.0 (full) to 1.0 (new), which can be calculated
        as abs((d / D) - 1) where d is the time since the last new moon
        and D = 29.5 days is the period between new moons.  The corresponding
        illumination fraction is ``0.5*(1 + cos(pi * moon_phase))``.
    vband_extinction : float
        V-band extinction coefficient to use.

    Returns
    -------
    astropy.units.Quantity
        Observed V-band surface brightness of scattered moonlight.
    """
    moon_phase = np.asarray(moon_phase)
    if np.any((moon_phase < 0) | (moon_phase > 1)):
        raise ValueError(
            'Invalid moon phase {0}. Expected 0-1.'.format(moon_phase))
    # Calculate the V-band magnitude of the moon (eqn. 9).
    abs_alpha = 180. * moon_phase
    #m = -12.73 + 0.026 * abs_alpha + 4e-9 * abs_alpha ** 4 (default value)
    m = M0 + M1 * abs_alpha + M2 * 1e-9 * abs_alpha ** 4
    # Calculate the illuminance of the moon outside the atmosphere in
    # foot-candles (eqn. 8).
    Istar = 10 ** (-0.4 * (m + 16.57))
    # Calculate the scattering function (eqn.21).
    rho = separation_angle.to(u.deg).value
    f_scatter = (C_R * (1.06 + np.cos(separation_angle) ** 2) +
                 10 ** (C_M0 - rho / C_M1))
    # Calculate the scattering airmass along the lines of sight to the
    # observation and moon (eqn. 3).
    X_obs = (1 - 0.96 * np.sin(obs_zenith) ** 2) ** (-0.5)
    X_moon = (1 - 0.96 * np.sin(moon_zenith) ** 2) ** (-0.5)
    # Calculate the V-band moon surface brightness in nanoLamberts.
    B_moon = (f_scatter * Istar *
        10 ** (-0.4 * vband_extinction * X_moon) *
        (1 - 10 ** (-0.4 * (vband_extinction * X_obs))))
    # Convert from nanoLamberts to to mag / arcsec**2 using eqn.19 of
    # Garstang, "Model for Artificial Night-Sky Illumination",
    # PASP, vol. 98, Mar. 1986, p. 364 (http://dx.doi.org/10.1086/131768)
    return ((20.7233 - np.log(B_moon / 34.08)) / 0.92104 *
            u.mag / (u.arcsec ** 2))


def _cI_twi(alpha, delta, airmass):
    ''' twilight contribution

    :param alpha: 

    :param delta: 

    :param airmass: 
    
    :retrun wave: 

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


##########################################################################
# contributions to parker's sky surface brightness model  
##########################################################################
def _read_parkerCoeffs(): 
    ''' read the coefficients of parker's model 
    '''
    f = ''.join([UT.code_dir(), 'dat/sky/MoonResults.csv']) 

    _coeffs = pd.DataFrame.from_csv(f)
    _coeffs.columns = [
            'wl', 'model', 'data_var', 'unexplained_var',' X2', 'rX2', 
            'c0', 'c_am', 'tau', 'tau2', 'c_zodi', 'c_isl', 'sol', 'I', 
            't0', 't1', 't2', 't3', 't4', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 
            'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 
            'c2', 'c3', 'c4', 'c5', 'c6'
            ]
    # keep moon models
    coeffs = _coeffs[coeffs['model'] == 'moon']

    # order based on wavelengths for convenience
    wave_sort = np.argsort(np.array(coeffs['wl']))  
    
    for k in coeffs.keys(): 
        coeffs[k] = np.array(coeffs[k])[wave_sort]

    return coeffs


def _parker_Icontinuum(coeffs, X, beta, l, b, mjd, month_frac, hour_frac, alpha, delta, altm, illm, delm, g): 
    ''' sky continuum (Fragelius thesis Eq. 4.23)
    '''
    # airmass contrib.  
    _Iairmass = coeffs['c_am'] * X  

    # zodiacal contrib. (func. of ecliptic latitude) 
    _Izodiacal = coeffs['c_zodi'] * _parker_Izodi(beta)
    
    _Iisl = coeffs['c_isl'] * _parker_Iisl(l, b) 

    _Isolar_flux = coeffs['sol'] * _parker_Isf(mjd - coeffs['I']) 

    _Iseasonal = _parker_cI_seas(month_frac, coeffs) 

    _Ihourly = _parker_cI_hour(hour_frac, coeffs)

    _dT = _parker_deltaT(X, coeffs) 
    
    # When the sun is above -20 altitude, some of its light will back-scatter 
    # off the atmosphere into the field of view. (Fragelius thesis Eq. 4.27)
    _Itwilight = _parker_cI_twi_exp(alpha, delta, X, coeffs) 

    # light from the moon that is scattered into our field of view (Fragelius thesis Eq. 4.28, 4.29)
    _Imoon = _parker_cI_moon_exp(altm, illm, delm, g, X, coeffs)

    _Iadd_continuum = coeffs['c0']

    # I_continuum(lambda)
    Icont = (_Iairmass + _Izodiacal + _Iisl + _Isolar_flux + _Iseasonal + _Ihourly + _Iadd_continuum) * _dT + _Itwilight + _Imoon

    return 10*coeffs['wl'], np.array(Icont)


def _parker_cI_moon_exp(altm, illm, deltam, g, airmass, coeffs): 
    ''' light from the moon that is scattered into our field of view (Fragelius thesis Eq. 4.28, 4.29)
    '''
    Alambda = _parker_albedo(g, coeffs) # albedo factor 

    moon = (coeffs['m0'] * altm**2 + 
            coeffs['m1'] * altm + 
            coeffs['m2'] * illm**2 + 
            coeffs['m3'] * illm + 
            coeffs['m4'] * deltam**2 + 
            coeffs['m5'] * deltam 
            ) * Alambda * np.exp(-coeffs['m6'] * airmass) 
    return moon


def _parker_albedo(g, coeffs): 
    ''' albedo, i.e. reflectivity of the moon (Fragelius thesis Eq. 4.28)
    g is the lunar phase (g = 0 for full moon and 180 for new moon) 
    '''
    albedo_table = pd.read_csv(''.join([UT.code_dir(), 'dat/sky/albedo_constants.csv']), 
            delim_whitespace=True) 
    albedo_constants = {}
    for col in list(albedo_table):
        line = interp1d(albedo_table['WAVELENGTH'], albedo_table[col], 
                bounds_error=False, fill_value=0)
        albedo_constants[col] = line 

    p1 = 4.06054
    p2 = 12.8802
    p3 = -30.5858
    p4 = 16.7498
    A = []
    for i in range(4):
        A.append(albedo_constants['a%d'%i](coeffs['wl'])*(g**i))
    A.append(albedo_constants['d1'](coeffs['wl']) * np.exp(-g/p1))
    A.append(albedo_constants['d2'](coeffs['wl']) * np.exp(-g/p2))
    A.append(albedo_constants['d3'](coeffs['wl']) * np.cos((g - p3)/p4))
    lnA = np.sum(A, axis=0)
    Al  = np.exp(lnA)
    return Al


def _parker_cI_twi_exp(alpha, delta, airmass, coeffs): 
    ''' When the sun is above -20 altitude, some of its light will back-scatter 
    off the atmosphere into the field of view. (Fragelius thesis Eq. 4.27)
    no observations are made when sun is above -14 altitude.
    '''
    if alpha > -20.: 
        twi = (
                coeffs['t0'] * np.abs(alpha) + # CT2
                coeffs['t1'] * alpha**2 +      # CT1
                coeffs['t2'] * delta**2 +      # CT3
                coeffs['t3'] * delta           # CT4
                ) * np.exp(-coeffs['t4'] * airmass)
    else: 
        twi = np.zeros(len(coeffs['t0'])) 
    return twi


def _parker_deltaT(airmass, coeffs): 
    '''effective transmission curve that accounts for the additional extinction 
    for observing at higher airmass (Fragelius thesis Eq. 4.24)
    '''
    zen_ext = np.loadtxt(''.join([UT.code_dir(), 'dat/sky/ZenithExtinction-KPNO.dat']))
    zen_wave = zen_ext[:,0]/10.
    ext = zen_ext[:,1]
    zext = interp1d(zen_wave, ext, bounds_error=False, fill_value='extrapolate')
    k = zext(coeffs['wl'])
    return 1 - (10**(-0.4*k) - 10**(-0.4*k*airmass))


def _parker_cI_hour(hour_frac, coeffs): 
    ''' Fragelius thesis Eq. 4.26
    '''
    levels = np.linspace(0,1,7)
    idx = np.argmin(np.abs(levels - hour_frac))

    _hours = np.zeros(6)
    _hours[idx] = 1

    for i in range(1,6): 
        if i == 1: 
            hours = coeffs['c'+str(i+1)] * _hours[i]
        else: 
            hours += coeffs['c'+str(i+1)] * _hours[i]
    return hours 


def _parker_cI_seas(month_frac, coeffs): 
    # Fragelius thesis Eq. 4.25 
    mm = np.rint(month_frac)
    if mm == 13: mm = 1
    
    _months = np.zeros(12) 
    _months[int(mm-1)] = 1
    
    month_names = ['feb', 'mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec']

    for i, mon in zip(range(1,12),  month_names): 
        if i == 1: 
            months = coeffs[mon] * _months[i]
        else: 
            months += coeffs[mon] * _months[i]
    return months 


def _parker_Isf(mjd): 
    # solar flux as a function of MJD 
    solar_data = np.load(''.join([UT.code_dir(), 'dat/sky/solar_flux.npy'])) 
    solar_flux = interp1d(solar_data['MJD'], solar_data['fluxobsflux'], bounds_error=False, fill_value=0)
    return solar_flux(mjd) 


def _parker_Iisl(gal_lat, gal_long): 
    # returns float 
    isl_data = pickle.load(open(''.join([UT.code_dir(), 'dat/sky/isl_map.p']),'rb'))
    return isl_data(gal_long, gal_lat)[0]


def _parker_Izodi(ecl_lat): 
    zodi_data = pickle.load(open(''.join([UT.code_dir(), 'dat/sky/s10_zodi.p']),'rb'))
    return zodi_data(np.abs(ecl_lat))
