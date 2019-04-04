import numpy as np 
from astropy.io import fits 
# eso sky calc
from skycalc_cli import skycalc 
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


def testsky(): 
    ''' purely testing 
    '''
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    dic = {'airmass': 1.01, 'incl_moon': 'Y', 'moon_target_sep': 55.33, 'moon_alt': 30., 'observatory': '2400'}
    skyModel = skycalc.SkyModel()
    skyModel.callwith(dic)
    skyModel.write('test.fits')
    
    f = fits.open('test.fits') 
    fdata = f[1].data 
    wave = fdata['lam'] * 1e4       # wavelength in Ang 
    flux = fdata['flux']            # photons/s/m^2/microm/arcsec2 (surface brightness)
    flux *= 2.483 * 1e-12 * 1e-8    # ergs/s/cm^2/Ang/arcsec^2

    sub.plot(wave, flux*1e17) 
    sub.set_xlim(3e3, 1e4)
    sub.set_ylim(-1, 10) 
    fig.savefig('test.png', bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    testsky()
