'''

script for validating the sky model with CMX and SV data


'''
import os 
import numpy as np 
# -- astropy -- 
import astropy.units as u
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


dir_dat = os.path.join(UT.dat_dir(), 'survey_validation')


def sky_brightness(airmass, moonill, moonalt, moonsep, sunalt, sunsep):
    ''' sky surface brightness as a function of observing conditions
    '''
    w, Isky = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, sunalt, sunsep)
    return w, Isky 


def sky_photons(exptime, airmass, moonill, moonalt, moonsep, sunalt, sunsep):
    '''
    '''
    import desisim.simexp
    from specsim.simulator import Simulator 
    
    # get sky surface brightness 
    wave, _Isky = sky_brightness(airmass, moonill, moonalt, moonsep, sunalt, sunsep)
    Isky = _Isky * 1e-17 * u.erg / (u.Angstrom * u.arcsec**2 * u.cm**2 * u.s) 

    # Generate specsim config object for a given wavelength grid
    config = desisim.simexp._specsim_config_for_wave(wave.value, dwave_out=None, specsim_config_file='desi')

    #- Create simulator
    desi = Simulator(config, camera_output=True)

    # Calculate the on-sky fiber areas at each focal-plane location.
    focal_x, focal_y = np.tile(desi.source.focal_xy, [1, 1]).T
    focal_r = np.sqrt(focal_x ** 2 + focal_y ** 2)

    radial_fiber_size       = (0.5 * desi.instrument.fiber_diameter / desi.instrument.radial_scale(focal_r))
    azimuthal_fiber_size    = (0.5 * desi.instrument.fiber_diameter / desi.instrument.azimuthal_scale(focal_r))
    fiber_area = np.pi * radial_fiber_size * azimuthal_fiber_size

    sky_fiber_flux = (Isky[:, np.newaxis] * fiber_area).to(desi.simulated['sky_fiber_flux'].unit)
        
    num_sky_photons = (sky_fiber_flux * desi.instrument.photons_per_bin[:, np.newaxis] * (exptime * u.s)).to(1).value
    return wave, num_sky_photons


def plot_sky_photons(exptime, airmass, moonill, moonalt, moonsep, sunalt, sunsep):
    '''
    '''
    wave, skyphoto = sky_photons(exptime, airmass, moonill, moonalt, moonsep, sunalt, sunsep)

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(wave, skyphoto, c='k', s=2) 
    sub.set_xlabel('Wavelength [Angstrom]', fontsize=20) 
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylabel('Sky flux [photons]', fontsize=20) 
    sub.set_ylim(0, 100)
    fig.savefig(os.path.join(dir_dat, 'sky_photons.png'), bbox_inches='tight')  
    return None 


if __name__=='__main__': 
    plot_sky_photons(150, 1., 0.7, 60., 60., -30., 80.)
