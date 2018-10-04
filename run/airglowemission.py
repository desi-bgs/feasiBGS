'''
'''
import os 
import numpy as np
import astropy.units as u
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
# -- feasibgs --
from feasibgs import util as UT


def UVESsky_continuum(emline_mask_width=5., kernel_size=51, sigma=100): 
    ''' Get the continuum of the UVES sky surface brightness
    '''
    # read in UVES sky spectrum 
    dir_desi = os.environ.get('DESIMODEL') 
    fsky = ''.join([dir_desi, '/data/spectra/spec-sky.dat']) 
    
    wave, sky = np.loadtxt(fsky, unpack=True, usecols=[0,1], skiprows=2)
    # this surface brightness has a wavelength resolution of 0.1 Ang
    
    # read in the lines identified by Crosby et al.(2006)
    wave_emlines, emfluxes = [], []
    for n in ['346', '437', '580L', '580U', '800U', '860L', '860U']:
        f = ''.join([UT.code_dir(), 'dat/sky/UVES_ident/', 'gident_', n, '.dat'])
        wl, _, emflux = np.loadtxt(open(f, 'rt').readlines()[:-1], skiprows=3, unpack=True, usecols=[1, 3, 4])
        wave_emlines.append(wl)
        emfluxes.append(emflux)
    wave_emlines = np.concatenate(wave_emlines)
    emfluxes = np.concatenate(emfluxes)

    # mask airglow emission lines 
    keep = (emfluxes > 0.5) & (wave_emlines >= 3600.) & (wave_emlines <= 10400.)
    lines_mask = np.ones(len(wave)).astype(bool)
    for w in wave_emlines[keep]: 
        nearline = ((wave > w * n_edlen(w) - emline_mask_width) & (wave < w * n_edlen(w) + emline_mask_width))
        lines_mask = lines_mask & ~nearline
    
    # emission line from lamps Table 4.4. from Parker's thesis
    lamp = np.array([4047, 4048, 4165, 4168, 4358, 4420, 4423, 4665, 4669, 4827, 
        4832, 4983, 5461, 5683, 5688, 5770, 5791, 5893, 6154, 6161]) 
    for w in lamp: 
        nearline = ((wave > w - emline_mask_width) & (wave < w + emline_mask_width))
        lines_mask = lines_mask & ~nearline
    
    # apply a median filter with a kernel size that corresponds to 
    # kernel_size * 0.1 Ang
    sky_filtered = medfilt(sky[lines_mask], kernel_size)

    # finally apply a Gaussian filter with standard deviation 
    # sigma * 0.1 Ang 
    sky_gauss = gaussian_filter(sky_filtered, sigma)
    
    # interpolate 
    f_continuum = interp1d(wave[lines_mask], sky_gauss, bounds_error=False, fill_value='extrapolate')
    sky_continuum = f_continuum(wave) 
    
    # UVES emission lines (i.e. sky spectrum - continuum) 
    sky_emission = sky - sky_continuum 

    # save the UVES sky continuum
    f_skycont = ''.join([UT.code_dir(), 'dat/sky/UVES_sky_continuum.dat']) 
    np.savetxt(f_skycont, np.array([wave, sky_continuum]).T)
    
    # save the UVES sky continuum
    f_skyem = ''.join([UT.code_dir(), 'dat/sky/UVES_sky_emission.dat']) 
    np.savetxt(f_skyem, np.array([wave, sky_emission]).T)
    return None 


def n_edlen(ll):
    # wavelength conversion from air to vacuum Edlen (1966)
    return 1. + 10**-8 * (8432.13 + 2406030./(130.-(1/ll)**2) + 15977/(38.9 - (1/ll)**2))


if __name__=="__main__": 
    UVESsky_continuum() 
