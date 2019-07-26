'''

script to generate data files to be read by the sky model (`sky._Isky`) 
to speed up calculations. This mainly saves to file data from 
`sky.specsim_initialize`

'''
import os 
import pickle 
import numpy as np 
import specsim.config 
from feasibgs import util as UT 

config = specsim.config.load_config('desi')
atm_config = config.atmosphere
    
surface_brightness_dict = config.load_table(
    atm_config.sky, 'surface_brightness', as_dict=True)
# dark sky surface brightness 
wave    = config.wavelength # wavelength 
Idark   = surface_brightness_dict['dark'].copy()  

extinction_coefficient = config.load_table(atm_config.extinction, 'extinction_coefficient')

extinction_array = 10**(-extinction_coefficient * np.linspace(1., 5., 101)[:,None] / 2.5)
print(extinction_array.shape)
print(extinction_array[0]) 
print(extinction_array[-1]) 
    
psf_config = getattr(atm_config, 'seeing', None)
seeing = dict(
    fwhm_ref=specsim.config.parse_quantity(psf_config.fwhm_ref),
    wlen_ref=specsim.config.parse_quantity(psf_config.wlen_ref),
    moffat_beta=float(psf_config.moffat_beta))
    
moon_config = getattr(atm_config, 'moon', None)
moon_spectrum = config.load_table(moon_config, 'flux')

wlim = (wave.value > 4000.) & (wave.value < 5000.) 
Idark4500 = np.median(Idark.value[wlim])

# lets also save twilgith data 
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

data = {
        'wavelength': wave, 
        'darksky_surface_brightness': Idark, 
        'extinction_coefficient': extinction_coefficient, 
        'extinction_array': extinction_array, 
        'seeing': seeing, 
        'moon_spectrum': moon_spectrum, 
        'darksky_4500a': Idark4500,
        'wavelength_twi': twi['wave'],  
        't0': twi['t0'],  
        't1': twi['t1'],  
        't2': twi['t2'],  
        't3': twi['t3'],  
        't4': twi['t4'],  
        'c0': twi['c0']
        } 

f_skymodel = os.path.join(UT.dat_dir(), 'data4skymodel.p') 
pickle.dump(data, open(f_skymodel, 'wb')) 
