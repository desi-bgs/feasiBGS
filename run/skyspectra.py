import os 
import time 
import pickle 
import numpy as np 
import scipy.optimize as sciop
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
# -- astropy -- 
import astropy.units as u
from astropy.table import Table as aTable
# -- feasibgs --
from feasibgs import util as UT
from feasibgs import skymodel as Sky


def refit_KSsky():  
    _t0 = time.time() 
    # read boss data 
    boss_meta = aTable.read(''.join([UT.dat_dir(), 'sky/', 'Bright_BOSS_Sky_blue.fits']))
    boss_airmass = boss_meta['AIRMASS'] 
    boss_moonill = boss_meta['MOON_ILL'] 
    boss_moonalt = boss_meta['MOON_ALT'] 
    boss_moonsep = boss_meta['MOON_SEP'] 
     
    # get BOSS sky surface brightness continuum at 380, 410, and 460 um
    boss_380 = BOSS_cont(380.) 
    print("BOSS continuum @ 380 calculated") 
    boss_410 = BOSS_cont(410.) 
    print("BOSS continuum @ 410 calculated") 
    boss_460 = BOSS_cont(460.) 
    print("BOSS continuum @ 460 calculated") 

    specsim_sky = Sky.specsim_initialize('desi')
    specsim_wave = specsim_sky._wavelength # Ang
    cr_def = specsim_sky.moon.KS_CR
    cm0_def = specsim_sky.moon.KS_CM0
    cm1_def = specsim_sky.moon.KS_CM1
    
    ks_elmask = airglow_emline_mask(specsim_wave.value)
    ks_wlim380 = (specsim_wave.value > 3600.) & (specsim_wave.value < 4000.) & ks_elmask 
    ks_wlim410 = (specsim_wave.value > 3900.) & (specsim_wave.value < 4300.) & ks_elmask 
    ks_wlim460 = (specsim_wave.value > 4400.) & (specsim_wave.value < 4800.) & ks_elmask 

    def skyKS(airmass, moonill, moonalt, moonsep, cr, cm0, cm1): 
        specsim_sky.airmass = airmass
        specsim_sky.moon.moon_phase = np.arccos(2.*moonill - 1)/np.pi
        specsim_sky.moon.moon_zenith = (90. - moonalt) * u.deg
        specsim_sky.moon.separation_angle = moonsep * u.deg
        specsim_sky.moon.KS_CR = cr
        specsim_sky.moon.KS_CM0 = cm0 
        specsim_sky.moon.KS_CM1 = cm1 
        return specsim_sky.surface_brightness.value

    def L2_skycont(cs): 
        '''calculate the total L2 norm between the predicted 
        UVES+KSsky(theta, CR, CM0, CM1) and the BOSSsky(theta)
        '''
        cr, cm0, cm1 = cs
        # calculate UVES + KS sky (theta, C_R, C_M0, C_M1)
        i_s = np.arange(len(boss_meta))[::10]
        skycont380 = np.zeros(len(i_s))
        skycont410 = np.zeros(len(i_s))
        skycont460 = np.zeros(len(i_s))
        for ii, i in enumerate(i_s): 
            ks_i = skyKS(boss_airmass[i], boss_moonill[i], boss_moonalt[i], boss_moonsep[i], cr, cm0, cm1)
            #ks_i_cont = gaussian_filter(medfilt(ks_i, 51), 100)
            #skycont380[ii] = np.median(ks_i_cont[ks_wlim380])   
            #skycont410[ii] = np.median(ks_i_cont[ks_wlim410])   
            #skycont460[ii] = np.median(ks_i_cont[ks_wlim460])   
            #skycont380[ii] = np.median(gaussian_filter(medfilt(ks_i[ks_wlim380], 21), 80))
            skycont410[ii] = np.median(gaussian_filter(medfilt(ks_i[ks_wlim410], 21), 80))   
            #skycont460[ii] = np.median(gaussian_filter(medfilt(ks_i[ks_wlim460], 21), 80))   
        #L2 = np.sum(
        #    (boss_380[i_s] - skycont380)**2 + 
        #    (boss_410[i_s] - skycont410)**2 + 
        #    (boss_460[i_s] - skycont460)**2)
        L2 = np.sum((boss_410[i_s] - skycont410)**2)
        return L2

    t0 = [2.*cr_def, cm0_def, cm1_def]
    theta_min = sciop.minimize(L2_skycont, t0)
    print('Default C_R = 10^%f, C_M0 = %f, C_M1 = %f' % (np.log10(cr_def), cm0_def, cm1_def))
    print('New C_R = 10^%f, C_M0 = %f, C_M1 = %f' % (np.log10(theta_min['x'][0]), theta_min['x'][1], theta_min['x'][2]))
    print('takes %f mins' % (time.time() - _t0)/60.) 
    return None 


def BOSS_cont(w): 
    ''' Get surface brightness continuum at w (micron) 
    '''
    fboss = ''.join([UT.dat_dir(), 'sky/', 'Bright_BOSS_Sky_blue', str(int(w)), '.p'])
    if os.path.isfile(fboss): 
        boss_skycont = pickle.load(open(fboss, 'rb'))
        return boss_skycont 
    else:
        boss = aTable.read(''.join([UT.dat_dir(), 'sky/', 'Bright_BOSS_Sky_blue.fits']))
        boss_wlim = ((boss[0]['WAVE'] > w - 5.) & (boss[0]['WAVE'] < w + 5.))
        boss_elmask = airglow_emline_mask(boss[0]['WAVE']*10.)

        boss_skycont = np.zeros(len(boss))
        for i in range(len(boss)):
            boss_i_cont = gaussian_filter(medfilt(boss[i]['SKY'], 51), 100)
            boss_skycont[i] = np.median(boss_i_cont[boss_wlim & boss_elmask])
        boss_skycont /= np.pi 
        # pickle dump 
        pickle.dump(boss_skycont, open(fboss, 'wb'))
        return boss_skycont 


def airglow_emline_mask(wave, dwave=5.): 
    ''' Get airglow emission line mask 
    '''
    uves_ws, uves_emfluxes = UVES_lines() 
    keep = (uves_emfluxes > 0.5) & (uves_ws >= 3600.) & (uves_ws <= 10400.)

    lines_mask = np.ones(len(wave)).astype(bool)
    for w in uves_ws[keep]: 
        nearline = ((wave > w * n_edlen(w) - dwave) & (wave < w * n_edlen(w) + dwave))
        lines_mask = lines_mask & ~nearline

    lamp = np.array([4047, 4048, 4165, 4168, 4358, 4420, 4423, 4665, 4669, 4827, 
                     4832, 4983, 5461, 5683, 5688, 5770, 5791, 5893, 6154, 6161]) 
    for w in lamp: 
        nearline = ((wave > w - dwave) & (wave < w + dwave))
        lines_mask = lines_mask & ~nearline
    return lines_mask 


def n_edlen(ll):
    return 1. + 10**-8 * (8432.13 + 2406030./(130.-(1/ll)**2) + 15977/(38.9 - (1/ll)**2))


def UVES_lines(): 
    ''' Read airglow emission lines in UVES
    '''
    wls, emfluxes = [], []
    for n in ['346', '437', '580L', '580U', '800U', '860L', '860U']:
        f = ''.join([UT.code_dir(), 'dat/sky/UVES_ident/', 'gident_', n, '.dat'])
        wl, emfwhm, emflux = np.loadtxt(open(f, 'rt').readlines()[:-1], skiprows=3, unpack=True, usecols=[1, 3, 4])
        wls.append(wl)
        emfluxes.append(emflux)
    wls = np.concatenate(wls)
    emfluxes = np.concatenate(emfluxes)
    return wls, emfluxes


if __name__=="__main__": 
    refit_KSsky()
