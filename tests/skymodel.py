#!/bin/python
''' 

Tests for the sky model 


'''
import os 
import pickle
import numpy as np 
import astropy.units as u
# -- other -- 
from datetime import datetime
from astroplan import Observer
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, GCRS, get_sun, get_moon, BaseRADecFrame
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

    
# get moon at some night  
utc_time = Time(datetime(2019, 3, 25, 9, 0, 0)) 
moon = get_moon(utc_time)

# kpno
kpno = EarthLocation.of_site('kitt peak')
kpno_altaz = AltAz(obstime=utc_time, location=kpno)


def MoonSeparation(): 
    ''' Astropy coordinate system is confusing so I need to 
    run some sanity checks on whether the separation is working 
    '''
    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    # altitude and azimuth bins 
    az_bins = np.linspace(0., 360., 33)
    alt_bins = np.linspace(0., 90., 19)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
    
    phi_grid, r_grid = np.meshgrid(az_bins/180.*np.pi, 90.-alt_bins)
    
    # calculate moon separation 
    moon_sep = np.zeros(az_grid.shape)     
    for i in range(az_grid.shape[0]): 
        for j in range(az_grid.shape[1]): 
            pt_aa = AltAz(az=az_grid[i,j]*u.deg, alt=alt_grid[i,j]*u.deg, obstime=utc_time, location=kpno)
            pt = SkyCoord(pt_aa)
            moon_sep[i,j] = moon.separation(pt).deg
    
    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, moon_sep, cmap='bone_r', vmin=0., vmax=180.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title("Moon Separation", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.moon_separation.png']), bbox_inches='tight') 
    return None


def Airmass(): 
    ''' Astropy coordinate system is confusing so I need to 
    run some sanity checks on whether the separation is working 
    '''
    # get moon at some night  
    #utc_time = Time(datetime(2019, 3, 25, 9, 0, 0)) 
    #moon = get_moon(utc_time)
    
    # kpno
    kpno = EarthLocation.of_site('kitt peak')
    kpno_altaz = AltAz(obstime=utc_time, location=kpno)
    #site = Observer(kpno, timezone='UTC')

    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    # altitude and azimuth bins 
    az_bins = np.linspace(0., 360., 33)
    alt_bins = np.linspace(0., 90., 19)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
    
    phi_grid, r_grid = np.meshgrid(az_bins/180.*np.pi, 90.-alt_bins)
    
    # calculate moon separation 
    airmass = np.zeros(az_grid.shape)     
    for i in range(az_grid.shape[0]): 
        for j in range(az_grid.shape[1]): 
            pt_aa = AltAz(az=az_grid[i,j]*u.deg, alt=alt_grid[i,j]*u.deg, obstime=utc_time, location=kpno)
            airmass[i,j] = pt_aa.secz #moon.separation(pt).deg
    
    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, airmass, cmap='bone', vmin=0., vmax=2.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title("Airmass", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.airmass.png']), bbox_inches='tight') 
    return None


def MoonCoeffs(): 
    ''' plot the different moon coefficients so I can get an idea 
    of their sign and wavelength dependence. 
    '''
    # initial SkySpec (doesn't matter, what the values are since 
    # we only want to get their coefficients) 
    #utc_time = Time(datetime(2019, 3, 25, 9, 0, 0)) 
    sky = Sky.skySpec(215, 0., utc_time) 
    
    fig = plt.figure(figsize=(15,10))
    sub = fig.add_subplot(211)
    for k in ['m2', 'm3']:  
        sub.plot(10.* sky.coeffs['wl'], sky.coeffs[k], label=k) 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlim([3500., 1.e4])

    sub = fig.add_subplot(212)
    for k in ['m0', 'm1', 'm4', 'm5', 'm6']:  
        sub.plot(10.* sky.coeffs['wl'], sky.coeffs[k], label=k) 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'Wavelength [$\AA$]', fontsize=25)
    sub.set_xlim([3500., 1.e4])
    fig.savefig(''.join([UT.fig_dir(), 'test.moon_coeffs.png']), bbox_inches='tight') 
    return None 


def flux_onthesky(): 
    ''' Astropy coordinate system is confusing so I need to 
    run some sanity checks on whether the separation is working 
    '''
    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    phi_grid, r_grid, totsky = Isky_onthesky(n_az=32, n_alt=18, key='I_cont')

    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, totsky, cmap='bone', vmin=0., vmax=60.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title("Sky Flux", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.flux_onthesky.png']), bbox_inches='tight') 
    return None


def Imoon_onthesky(): 
    ''' Astropy coordinate system is confusing so I need to 
    run some sanity checks on whether the separation is working 
    '''
    # kpno
    kpno = EarthLocation.of_site('kitt peak')
    kpno_altaz = AltAz(obstime=utc_time, location=kpno)
    #site = Observer(kpno, timezone='UTC')

    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    # sky brightness in  bins 
    phi_grid, r_grid, totsky = Isky_onthesky(n_az=8, n_alt=3, key='Icont')

    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, totsky, cmap='bone', vmin=0., vmax=60.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title("Sky Flux", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.Imoon_onthesky.png']), bbox_inches='tight') 
    return None

    
def Isky_onthesky(n_az=8, n_alt=3, key='Icont', overwrite=False):
    # altitude and azimuth bins 
    az_bins = np.linspace(0., 360., n_az+1)
    alt_bins = np.linspace(0., 90., n_alt+1)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
    # binning for the plot  
    phi_grid, r_grid = np.meshgrid(az_bins/180.*np.pi, 90.-alt_bins)
    
    f = ''.join([UT.fig_dir(), 'Isky_onthesky.az', str(n_az), '.alt', str(n_alt), '.p']) 
    if os.path.isfile(f) and not overwrite: 
        phi_grid, r_grid, totsky = pickle.load(open(f, 'rb'))
    else: 
        # calculate sky brightness
        keys = ['I_cont', 'I_airmass', 'I_zodiacal', 'I_isl', 'I_solar_flux', 'I_seashour', 'dT', 'I_twilight', 
                'I_moon', 'I_moon_noexp', 'I_add']
        totsky = {} 
        for k in keys: 
            totsky[k] = np.zeros(az_grid.shape)
        for i in range(az_grid.shape[0]): 
            for j in range(az_grid.shape[1]): 
                Isky = skyflux_onthesky(alt_grid[i,j], az_grid[i,j], band='blue')
                for k in keys: 
                    totsky[k][i,j] = Isky[k]
        pickle.dump([phi_grid, r_grid, totsky], open(f, 'wb'))

    return phi_grid, r_grid, totsky[key] 


def skyflux_onthesky(alt, az, band='blue'): 
    ''' return the sky flux on a given point (alt, az) on the sky  
    '''
    if band == 'blue': 
        wmin, wmax = 4000., 4500. 
    
    sky_aa = AltAz(az=az*u.deg, alt=alt*u.deg, obstime=utc_time, location=kpno)
    sky = SkyCoord(sky_aa)
    pt = Sky.skySpec(sky.icrs.ra.deg, sky.icrs.dec.deg, utc_time)

    w, Icont = pt.get_Icontinuum()
    wlim = ((w > wmin) & (w < wmax))
    out = {} 
    out['I_cont'] = np.average(Icont[wlim]) 
    out['I_airmass'] = np.average(pt._Iairmass[wlim]) 
    out['I_zodiacal'] = np.average(pt._Izodiacal[wlim]) 
    out['I_isl'] = np.average(pt._Iisl[wlim]) 
    out['I_solar_flux'] = np.average(pt._Isolar_flux[wlim]) 
    out['I_seashour'] = np.average(pt._Iseasonal[wlim] + pt._Ihourly[wlim]) 
    out['dT'] = np.average(pt._dT[wlim]) 
    out['I_twilight'] = np.average(pt._Itwilight[wlim]) 
    out['I_moon'] = np.average(pt._Imoon[wlim]) 
    out['I_moon_noexp'] = np.average((pt._Imoon * np.exp(pt.coeffs['m6'] * pt.X))[wlim]) 
    out['I_add'] = np.average(pt._Iadd_continuum[wlim]) 
    return out  


if __name__=='__main__': 
    #MoonSeparation()
    #Airmass()
    flux_onthesky()
