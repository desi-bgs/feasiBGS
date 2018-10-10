'''
'''
import pickle
import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
# -- astropy --
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.coordinates import CartesianRepresentation, HeliocentricTrueEcliptic
# -- feasibgs -- 
from . import util as UT 

# -- astroplan -- 
from astroplan import Observer
from astroplan import download_IERS_A
download_IERS_A()

class skySpec(object): 
    def __init__(self, ra, dec, obs_time, airmass=None, ecl_lat=None, sun_alt=None, sun_sep=None, moon_phase=None, moon_sep=None, moon_alt=None):
        ''' Given airmass, ra (deg), dec (deg), and observed time (UTC datetime) 
        '''
        # target coordinates 
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg) 
        # observed time (UTC)          
        utc_time = Time(obs_time)
        # kitt peak  
        kpno = EarthLocation.of_site('kitt peak')
        kpno_altaz = AltAz(obstime=utc_time, location=kpno) 
        coord_altaz = coord.transform_to(kpno_altaz)
        self.objalt = coord_altaz.alt.deg
        if self.objalt < 0.: 
            raise ValueError('object is below the horizon') 

        if airmass is None: 
            self.X = coord_altaz.secz
        else: 
            self.X = airmass    # air mass 
        
        if ecl_lat is None: # ecliptic latitude ( used for zodiacal light contribution ) 
            self.beta = coord.barycentrictrueecliptic.lat.deg
        else: 
            self.beta = ecl_lat 
        self.l = coord.galactic.l.deg   # galactic latitude ( used for ISL contribution ) 
        self.b = coord.galactic.b.deg   # galactic longitude ( used for ISL contribution ) 

        #obs_time = tai/86400.       # used to calculate mjd 
        self.mjd = utc_time.mjd   # mjd ( used for solar flux contribution ) 
        # fractional months ( used for seasonal contribution) 
        self.month_frac = utc_time.datetime.month + utc_time.datetime.day/30. 
        
        # fractional hour ( used for hourly contribution) 
        self.site = Observer(kpno, timezone='UTC')
        sun_rise = self.site.sun_rise_time(utc_time, which='next')
        sun_set = self.site.sun_set_time(utc_time, which='previous')
        hour = ((utc_time - sun_set).sec)/3600.
        self.hour_frac = hour/((Time(sun_rise, format='mjd') - Time(sun_set,format = 'mjd')).sec/3600.)

        # sun altitude (degrees)
        sun = get_sun(utc_time) 
        if sun_alt is None:
            sun_altaz = sun.transform_to(kpno_altaz) 
            self.alpha = sun_altaz.alt.deg
        else: 
            self.alpha = sun_alt
        if self.alpha > -13.: raise ValueError("sun is higher than BGS limit") 

        # sun separation (separation between the target and the sun’s location)
        if sun_sep is None: 
            self.delta = coord.separation(sun).deg
        else: 
            self.delta = sun_sep           
        
        # used for scattered moonlight
        moon = get_moon(utc_time)  
        if moon_alt is None: 
            moon_altaz = moon.transform_to(kpno_altaz) 
            self.altm = moon_altaz.alt.deg 
        else: 
            self.altm = moon_alt 

        if moon_sep is None: 
            self.delm = coord.separation(moon).deg
        else: 
            self.delm = moon_sep
    
        if moon_phase is None:  
            #from https://astroplan.readthedocs.io/en/latest/_modules/astroplan/moon.html
            elongation = sun.separation(moon)
            phase = np.arctan2(sun.distance * np.sin(elongation),
                    moon.distance - sun.distance*np.cos(elongation))
            self.g = phase.value    # in radians 
            self.illm = (1. + np.cos(phase))/2.
        else: 
            self.g = moon_phase     # moon phase angle 
            self.illm = (1. + np.cos(moon_phase))/2.

        self._readCoeffs()

    def surface_brightness(self, wave): 
        ''' return the surface brightness of the sky 
        '''
        # read in sky emission from the UVES continuum subtraction
        w_uves, S_uves = np.loadtxt(''.join([UT.code_dir(), 'dat/sky/UVES_sky_emission.dat']), 
                unpack=True, usecols=[0,1]) 
        # interpolate 
        f_uves = interp1d(w_uves, S_uves, bounds_error=False, fill_value='extrapolate')
        S_emission = f_uves(wave)

        flux_continuum = self.Icont(wave) 
        S_continuum = flux_continuum / np.pi  # BOSS has 2 arcsec diameter
        return S_continuum + S_emission 

    def Icont(self, w): 
        ''' interpolate 
        '''
        wave, Icont = self.get_Icontinuum() 
        f_Icont = interp1d(wave, Icont, bounds_error=False, fill_value='extrapolate')
        return f_Icont(w) 

    def get_Icontinuum(self): 
        ''' calculate the continuum of the sky (Fragelius thesis Eq. 4.23)
        '''
        self._Iairmass = self.coeffs['c_am'] * self.X  
    
        self._Izodiacal = self.coeffs['c_zodi'] * self.I_zodi(self.beta)

        self._Iisl = self.coeffs['c_isl'] * self.I_isl(self.l, self.b) 

        self._Isolar_flux = self.coeffs['sol'] * self.I_sf(self.mjd - self.coeffs['I']) 

        self._Iseasonal = self.cI_seas(self.month_frac) 

        self._Ihourly = self.cI_hour(self.hour_frac)

        self._dT = self.deltaT(self.X) 
        
        self._Itwilight = self.cI_twi_exp(self.alpha, self.delta, self.X) 
    
        self._Imoon = self.cI_moon_exp(self.altm, self.illm, self.delm, self.g, self.X)

        self._Iadd_continuum = self.coeffs['c0']
    
        # I_continuum(lambda)
        Icont = (self._Iairmass + self._Izodiacal + self._Iisl + self._Isolar_flux + self._Iseasonal + self._Ihourly + self._Iadd_continuum) * self._dT + self._Itwilight + self._Imoon
    
        wave = np.array(self.coeffs['wl'])
        wave_sort = np.argsort(wave) 
        return 10*wave[wave_sort], np.array(Icont)[wave_sort]

    def cI_moon_exp(self, altm, illm, deltam, g, airmass): 
        # light from the moon that is scattered into our field of view 
        # (Fragelius thesis Eq. 4.28, 4.29)
        Alambda = self._albedo(g) # albedo factor 

        moon = (
                self.coeffs['m0'] * altm**2 + 
                self.coeffs['m1'] * altm + 
                self.coeffs['m2'] * illm**2 + 
                self.coeffs['m3'] * illm + 
                self.coeffs['m4'] * deltam**2 + 
                self.coeffs['m5'] * deltam 
                ) * Alambda * np.exp(-self.coeffs['m6'] * airmass) 
        return moon

    def _albedo(self, g): 
        # albedo (i.e. reflectivity of the moon)
        # g is the lunar phase (g = 0◦ for full moon and 180◦ for new moon)
        # (Fragelius thesis Eq. 4.28)
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
            A.append(albedo_constants['a%d'%i](self.coeffs['wl'])*(g**i))
        A.append(albedo_constants['d1'](self.coeffs['wl']) * np.exp(-g/p1))
        A.append(albedo_constants['d2'](self.coeffs['wl']) * np.exp(-g/p2))
        A.append(albedo_constants['d3'](self.coeffs['wl']) * np.cos((g - p3)/p4))
        lnA = np.sum(A, axis=0)
        Al = np.exp(lnA)
        return Al

    def cI_twi_exp(self, alpha, delta, airmass): 
        # When the sun is above −20◦ altitude, some of its light will back-scatter 
        # off the atmosphere into the field of view. (Fragelius thesis Eq. 4.27)
        # no observations are made when sun is above -14o altitude.
        if alpha > -20.: 
            twi = (
                    self.coeffs['t0'] * np.abs(alpha) + # CT2
                    self.coeffs['t1'] * alpha**2 +      # CT1
                    self.coeffs['t2'] * delta**2 +      # CT3
                    self.coeffs['t3'] * delta           # CT4
                    ) * np.exp(-self.coeffs['t4'] * airmass)
        else: 
            twi = np.zeros(len(self.coeffs['t0'])) 
        return twi

    def deltaT(self, airmass): 
        # effective transmission curve that accounts for the additional extinction 
        # for observing at higher airmass (Fragelius thesis Eq. 4.24)
        zen_ext = np.loadtxt(''.join([UT.code_dir(), 'dat/sky/ZenithExtinction-KPNO.dat']))
        zen_wave = zen_ext[:,0]/10.
        ext = zen_ext[:,1]
        zext = interp1d(zen_wave, ext, bounds_error=False, fill_value='extrapolate')
        k = zext(self.coeffs['wl'])
        return 1 - (10**(-0.4*k) - 10**(-0.4*k*airmass))
    
    def cI_hour(self, hour_frac): 
        # Fragelius thesis Eq. 4.26
        levels = np.linspace(0,1,7)
        idx = np.argmin(np.abs(levels - hour_frac))

        _hours = np.zeros(6)
        _hours[idx] = 1

        for i in range(1,6): 
            if i == 1: 
                hours = self.coeffs['c'+str(i+1)] * _hours[i]
            else: 
                hours += self.coeffs['c'+str(i+1)] * _hours[i]
        return hours 

    def cI_seas(self, month_frac): 
        # Fragelius thesis Eq. 4.25 
        mm = np.rint(month_frac)
        if mm == 13: mm = 1
        
        _months = np.zeros(12) 
        _months[int(mm-1)] = 1
        
        month_names = ['feb', 'mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec']
    
        for i, mon in zip(range(1,12),  month_names): 
            if i == 1: 
                months = self.coeffs[mon] * _months[i]
            else: 
                months += self.coeffs[mon] * _months[i]
        return months 

    def I_sf(self, mjd): 
        # solar flux as a function of MJD 
        solar_data = np.load(''.join([UT.code_dir(), 'dat/sky/solar_flux.npy'])) 
        solar_flux = interp1d(solar_data['MJD'], solar_data['fluxobsflux'], bounds_error=False, fill_value=0)
        return solar_flux(mjd) 

    def I_isl(self, gal_lat, gal_long): 
        isl_data = pickle.load(open(''.join([UT.code_dir(), 'dat/sky/isl_map.p']),'rb'))
        return isl_data(gal_long, gal_lat)[0]

    def I_zodi(self, ecl_lat): 
        zodi_data = pickle.load(open(''.join([UT.code_dir(), 'dat/sky/s10_zodi.p']),'rb'))
        return zodi_data(np.abs(ecl_lat))

    def _readCoeffs(self): 
        ''' read the coefficients of the model 
        '''
        f = ''.join([UT.code_dir(), 'dat/sky/MoonResults.csv']) 

        coeffs = pd.DataFrame.from_csv(f)
        coeffs.columns = [
                'wl', 'model', 'data_var', 'unexplained_var',' X2', 'rX2', 
                'c0', 'c_am', 'tau', 'tau2', 'c_zodi', 'c_isl', 'sol', 'I', 
                't0', 't1', 't2', 't3', 't4', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 
                'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 
                'c2', 'c3', 'c4', 'c5', 'c6'
                ]
        # keep moon models
        self.coeffs = coeffs[coeffs['model'] == 'moon']
        return None 
