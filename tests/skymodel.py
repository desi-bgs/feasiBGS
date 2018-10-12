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
from feasibgs import forwardmodel as FM 
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
    
    for i in range(7): 
        print 'm'+str(i), np.average(sky.coeffs['m'+str(i)][(sky.coeffs['wl'] > 400.) & (sky.coeffs['wl'] < 450.)])
    
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


def MoonAlbedo(): 
    ''' plot the different moon coefficients so I can get an idea 
    of their sign and wavelength dependence. 
    '''
    # initial SkySpec (doesn't matter, what the values are since 
    # we only want to get their coefficients) 
    #utc_time = Time(datetime(2019, 3, 25, 9, 0, 0)) 
    sky = Sky.skySpec(215, 0., utc_time) 

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    for g in np.linspace(0., np.pi, 10):
        sub.plot(10.* sky.coeffs['wl'], sky._albedo(g), label='phase = '+str(round(g,2))) 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'Wavelength [$\AA$]', fontsize=25)
    sub.set_xlim([3500., 1.e4])
    fig.savefig(''.join([UT.fig_dir(), 'test.albedo.png']), bbox_inches='tight') 
    return None 


def Isky_onthesky(): 
    ''' Astropy coordinate system is confusing so I need to 
    run some sanity checks on whether the separation is working 
    '''
    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    phi_grid, r_grid, totsky = _Isky_onthesky(n_az=32, n_alt=18, key='I_cont')

    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, totsky / np.pi, cmap='bone', vmin=5., vmax=20.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title(r"Sky Surf. Bright. [$10^{-17} erg/cm^{2}/s/\AA/arcsec^2$]", fontsize=25) 
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
    phi_grid, r_grid, totsky = _Isky_onthesky(n_az=32, n_alt=18, key='I_moon')
    _, _, dT = _Isky_onthesky(n_az=32, n_alt=18, key='dT')

    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, dT * totsky / np.pi, cmap='bone', vmin=0., vmax=20.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title(r"Moon contribution [$10^{-17} erg/cm^{2}/s/\AA/arcsec^2$]", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.Imoon_onthesky.png']), bbox_inches='tight') 
    return None


def cImoon_onthesky(): 
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
    
    # calculate moon phase 
    sun = get_sun(utc_time) 
    elongation = sun.separation(moon)
    phase = np.arctan2(sun.distance * np.sin(elongation), moon.distance - sun.distance*np.cos(elongation)).value 
    moon_ill = (1. + np.cos(phase))/2.
    
    #altitude and azimuth bins 
    n_az = 32
    n_alt = 8 
    az_bins = np.linspace(0., 360., n_az+1)
    alt_bins = np.linspace(0., 90., n_alt+1)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
   
    sky_obj = Sky.skySpec(215, 0., utc_time) 

    cImoons = np.zeros(az_grid.shape)
    for i in range(az_grid.shape[0]): 
        for j in range(az_grid.shape[1]): 
            sky_aa = AltAz(az=az_grid[i,j]*u.deg, alt=alt_grid[i,j]*u.deg, obstime=utc_time, location=kpno)
            sky = SkyCoord(sky_aa)
            sep = moon.separation(sky).deg 
            
            cImoon = (0.09088725092834156 * moon_alt**2 + 
                    5.578562780174388 * moon_alt -
                    375.5916279673738 * moon_ill**2 + 
                    493.894534837979 * moon_ill + 
                    0.009287738000050467 * sep**2 -
                    1.9141591334780124 * sep) * np.average(sky_obj._albedo(phase)[(sky_obj.coeffs['wl'] > 400.) & (sky_obj.coeffs['wl'] < 450.)])
            cImoons[i,j] = cImoon#np.average(Bmoon[wlim]) 
    
    # binning for the plot  
    phi_grid, r_grid = np.meshgrid(az_bins/180.*np.pi, 90.-alt_bins)
    
    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, cImoons, cmap='bone', vmin=0., vmax=20.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title("c Imoon", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.cImoon_onthesky.png']), bbox_inches='tight') 
    return None


def Imoon_noexp_onthesky(): 
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
    phi_grid, r_grid, totsky = _Isky_onthesky(n_az=32, n_alt=18, key='I_moon_noexp')
    _, _, dT = _Isky_onthesky(n_az=32, n_alt=18, key='dT')
    print totsky

    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, totsky/np.pi, cmap='bone', vmin=0., vmax=20.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title("Sky Flux (moon contribution airmass indep.)", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.Imoon_noexp_onthesky.png']), bbox_inches='tight') 
    return None


def KrisciunasSchaefer_onthesky(): 
    ''' Astropy coordinate system is confusing so I need to 
    run some sanity checks on whether the separation is working 
    '''
    import desimodel.io
    import desisim.simexp
    params = desimodel.io.load_desiparams() 
    wavemin = params['ccd']['b']['wavemin']
    wavemax = params['ccd']['z']['wavemax']
    print('%f < lambda < %f' % (wavemin, wavemax))

    waves = np.arange(wavemin, wavemax, 0.2) * u.angstrom
    wlim = ((waves > 4000.*u.angstrom) & (waves < 4500.*u.angstrom)) 
    config = desisim.simexp._specsim_config_for_wave((waves).to('Angstrom').value, specsim_config_file='desi')
    desi = FM.SimulatorHacked(config, num_fibers=1, camera_output=True)

    extinction_coefficient = config.load_table(config.atmosphere.extinction, 'extinction_coefficient')
    #surface_brightness_dict = config.load_table(config.atmosphere.sky, 'surface_brightness', as_dict=True)

    # kpno
    kpno = EarthLocation.of_site('kitt peak')
    kpno_altaz = AltAz(obstime=utc_time, location=kpno)
    #site = Observer(kpno, timezone='UTC')

    # moon at KPNO 
    moon_altaz = moon.transform_to(kpno_altaz)
    moon_az = moon_altaz.az.deg 
    moon_alt = moon_altaz.alt.deg
    
    sun = get_sun(utc_time) 
            
    elongation = sun.separation(moon)
    phase = np.arctan2(sun.distance * np.sin(elongation),
            moon.distance - sun.distance*np.cos(elongation))
    desi.atmosphere.moon.moon_phase = phase.value/np.pi #moon_phase/np.pi #np.arccos(2*moonfrac-1)/np.pi
    desi.atmosphere.moon.moon_zenith = (90. - moon_alt) * u.deg
    
    #altitude and azimuth bins 
    n_az = 32
    n_alt = 8 
    az_bins = np.linspace(0., 360., n_az+1)
    alt_bins = np.linspace(0., 90., n_alt+1)
    az_grid, alt_grid = np.meshgrid(0.5*(az_bins[1:]+az_bins[:-1]), 0.5*(alt_bins[1:]+alt_bins[:-1])) 
   
    Bmoons = np.zeros(az_grid.shape)
    for i in range(az_grid.shape[0]): 
        for j in range(az_grid.shape[1]): 
            sky_aa = AltAz(az=az_grid[i,j]*u.deg, alt=alt_grid[i,j]*u.deg, obstime=utc_time, location=kpno)
            sky = SkyCoord(sky_aa)
            sep = moon.separation(sky).deg 
            
            desi.atmosphere.airmass = sky_aa.secz 
            desi.atmosphere.moon.separation_angle = sep * u.deg
            Bmoon = desi.atmosphere.moon.surface_brightness.value * 1e17
            Bmoons[i,j] = np.average(Bmoon[wlim]) 
    
    # binning for the plot  
    phi_grid, r_grid = np.meshgrid(az_bins/180.*np.pi, 90.-alt_bins)

    # plot the separations 
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, polar=True)
    sub.set_theta_zero_location('N')
    sub.set_theta_direction(-1)
    c = sub.pcolormesh(phi_grid, r_grid, Bmoons, cmap='bone', vmin=0., vmax=20.)
    sub.scatter([moon_altaz.az.deg/180.*np.pi], [90.-moon_altaz.alt.deg], c='C1', s=200, label='Moon')
    plt.colorbar(c)
    sub.legend(fontsize=25, markerscale=3, handletextpad=0)
    sub.set_yticks(range(0, 90+10, 10))
    sub.set_ylim([0., 90.])
    _ = sub.set_yticklabels([90, '', '', 60, '', '', 30, '', '', 0])
    sub.grid(True, which='major')
    sub.set_title(r"Krisciunas Schaefer [$10^{-17} erg/cm^{2}/s/\AA/arcsec^2$]", fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), 'test.Krisciunas_Schaefer_onthesky.png']), bbox_inches='tight') 
    return None

    
def _Isky_onthesky(n_az=8, n_alt=3, key='Icont', overwrite=False):
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
    MoonCoeffs()
    #MoonSeparation()
    #Airmass()
    #Isky_onthesky()
    #Imoon_onthesky()
    #MoonAlbedo()
    #cImoon_onthesky()
    #Imoon_noexp_onthesky()
    #KrisciunasSchaefer_onthesky()
