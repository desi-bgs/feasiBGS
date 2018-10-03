'''
'''
import pickle
import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
# -- astropy --
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
# -- feasibgs -- 
from . import util as UT 

# -- astroplan -- 
from astroplan import Observer
from astroplan import download_IERS_A
download_IERS_A()

class skySpec(object): 
    def __init__(self, airmass, ecl_lat, gal_lat, gal_lon, tai, sun_alt, sun_sep, moon_phase, moon_ill, moon_sep, moon_alt):
        # prase the input parameters 
        self.X = airmass    # air mass 
        self.beta = ecl_lat # ecliptic latitude ( used for zodiacal light contribution ) 
        self.l = gal_lat    # galactic latitude ( used for ISL contribution ) 
        self.b = gal_lon    # galactic longitude ( used for ISL contribution ) 
        obs_time = tai/86400.       # used to calculate mjd 
        _apache = EarthLocation.of_site('Apache Point')
        start_time = Time(obs_time, scale='tai', format='mjd', location=_apache)
        self.mjd = start_time.mjd   # mjd ( used for solar flux contribution ) 
        # fractional months ( used for seasonal contribution) 
        self.month_frac = start_time.datetime.month + start_time.datetime.day/30. 
        
        # fractional hour ( used for hourly contribution) 
        self.apache = Observer(_apache)
        sun_rise = self.apache.sun_rise_time(start_time, which='next')
        sun_set = self.apache.sun_set_time(start_time, which='previous')
        hour = ((start_time - sun_set).sec)/3600.
        self.hour_frac = hour/((Time(sun_rise, format='mjd') - Time(sun_set,format = 'mjd')).sec/3600.)

        self.alpha = sun_alt    # sun altitude
        self.delta = sun_sep    # sun separation (separation between the target and the sun’s location)
        
        # used for scattered moonlight
        self.g = moon_phase     # moon phase 
        self.altm = moon_alt
        self.illm = moon_ill
        self.delm = moon_sep
        self._readCoeffs()

    def Icont(self, wave): 
        ''' interpolate 
        '''
        wave, Icont = self.get_Icontinuum() 
        f_Icont = interp1d(wave, Icont, bounds_error=False, fill_value='extrapolate')
        return f_Icont(wave) 

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
        return wave[wave_sort], np.array(Icont)[wave_sort]

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
        albedo_table = pd.read_csv(''.join([UT.code_dir(), 'dat/sky/albedo_constants.csv']), delim_whitespace=True) 
        albedo_constants = {}
        for col in list(albedo_table):
            line = interp1d(albedo_table['WAVELENGTH'], albedo_table[col], bounds_error=False, fill_value=0)
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



