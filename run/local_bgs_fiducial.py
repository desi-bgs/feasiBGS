'''

Tests of fiducial spectra with David. We will both generate 
spectra with some fiducial parameters and then run them through
our respective redshift fitters and determine if we agree or not.  

'''
#!/bin/usr/python 
import os
import time 
import h5py
import pickle
import subprocess 
import numpy as np 
# -- astropy -- 
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
# -- desi -- 
from desispec.io import write_spectra
# -- feasibgs -- 
from feasibgs import util as UT
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


def mockexp_gleg_simSpec_fiducial(r_ap, iexp=562, seed=1): 
    ''' simulate a single DESI BGS spectrum with 
    r_apflux : between r_ap-0.05 and r_ap+0.05
    sky brightness : new sky model(survey sim mock exposure observing condition) that 
        is ~2.5x dark sky (iexp = 562). 
    halpha : no halpha emission 
    redshift : ~0.2
    exposure time : 600 seconds 
    '''
    from feasibgs import catalogs as Cat
    from feasibgs import forwardmodel as FM 
    np.random.seed(seed) # random seed

    # GAMA-Legacy catalog
    cata = Cat.GamaLegacy()
    gleg = cata.Read('g15', dr_gama=3, dr_legacy=7)

    redshift = gleg['gama-spec']['z']
    absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
    r_mag_apflux = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1])
    r_mag_gama = gleg['gama-photo']['r_model'] # r-band magnitude from GAMA (SDSS) photometry
    ha_gama = gleg['gama-spec']['ha_flux'] # halpha line flux 

    ngal = len(redshift) # number of galaxies
    vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    match = bgs3._GamaLegacy(gleg)
    hasmatch = (match != -999)
    nohalpha = (ha_gama <= 0.) 
    median_z = ((redshift > 0.195) & (redshift < 0.205)) 
    rmag_cut = ((r_mag_apflux < r_ap + 0.05) & (r_mag_apflux > r_ap - 0.05)) 
    assert np.sum(hasmatch & nohalpha & median_z & rmag_cut) > 0 

    i_pick = np.random.choice(np.arange(ngal)[hasmatch & nohalpha & median_z & rmag_cut], 1) 

    # get source spectra  
    dir_spec = os.path.join(UT.dat_dir(), 'spectra', 'gamadr3_legacydr7') 
    fsource = os.path.join(dir_spec, 'g15.source_spectra.r_ap%.2f.v2.fits' % r_ap)
    if os.path.isfile(fsource): 
        f = fits.open(fsource)
        fdata = f[1].data 
        wave = fdata['wave'] 
        flux_eml = np.array([fdata['flux']])
    else: 
        s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=2e4)
        emline_flux = s_bgs.EmissionLineFlux(gleg, index=np.arange(ngal)[i_pick], dr_gama=3, silent=True)
        flux_eml, wave, _, magnorm_flag = s_bgs.Spectra(r_mag_apflux[i_pick], redshift[i_pick],
                                                        vdisp[i_pick], seed=1, templateid=match[i_pick],
                                                        emflux=emline_flux, mag_em=r_mag_gama[i_pick], 
                                                        silent=False)
        t = Table([wave, flux_eml[0]], names=('wave', 'flux')) 
        t.write(fsource, format='fits') 
        assert magnorm_flag[0]

    # read in sky surface brightness
    w_sky, exps_skybright = pickle.load(open(''.join([UT.dat_dir(), 
        'newKSsky_twi_brightness.bgs_survey_exposures.withsun.p']), 'rb'))
    skybright = exps_skybright[iexp,:]
    u_surface_brightness = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second
    # output sky surface brightness
    f_sky = os.path.join(dir_spec, 'sky_brightness.iexp%i.fits' % iexp)
    if not os.path.isfile(f_sky): 
        t = Table([w_sky, skybright], names=('wave', 'surface_brightness')) 
        t.write(f_sky, format='fits') 

    # read exposures from file
    fexp = h5py.File(''.join([UT.dat_dir(), 'bgs_survey_exposures.withsun.hdf5']), 'r')
    exps = {} 
    for k in fexp.keys(): 
        exps[k] = fexp[k].value 
    fexp.close()

    # simulate the exposures 
    fdesi = FM.fakeDESIspec()
    f_simspec = os.path.join(dir_spec, 'g15.sim_spectra.r_ap%.2f.texp600.iexp%i.newsky.v2.fits' % (r_ap, iexp))
    print('-- constructing %s --' % f_simspec) 
    bgs_spectra = fdesi.simExposure(wave, flux_eml, 
            exptime=600., 
            airmass=exps['AIRMASS'][iexp],
            skycondition={'name': 'input',
                'sky': np.clip(skybright, 0, None) * u_surface_brightness,
                'wave': w_sky},
            filename=f_simspec)

    # bgs noise spectra with source spectra signal
    f_simspec0 = os.path.join(dir_spec, 'g15.sim_spectra.source_flux.r_ap%.2f.texp600.iexp%i.newsky.v2.fits' % (r_ap, iexp))
    print('-- constructing %s --' % f_simspec0)
    for band in ['b', 'r', 'z']: 
        w_band = bgs_spectra.wave[band]
        # interpolate to spectograph resolution 
        bgs_spectra.flux[band] = np.atleast_2d(np.interp(w_band, wave, flux_eml[0]))
    write_spectra(f_simspec0, bgs_spectra)
    return None 


def Redrock_gleg_simSpec_fiducial(r_ap, flux='regular', ncpu=4): 
    ''' Run redrock 
    '''
    import redrock
    import redrock.templates
    import redrock.archetypes
    import redrock.plotspec
    from redrock.external import desi
    from scipy.signal import medfilt
    
    dir_spec = os.path.join(UT.dat_dir(), 'spectra', 'gamadr3_legacydr7') 
    if flux == 'regular': 
        f_simspec = os.path.join(dir_spec, 'g15.sim_spectra.r_ap%.2f.texp600.iexp%i.newsky.v2.fits' % (r_ap, iexp))
    elif flux == 'source': 
        f_simspec = os.path.join(dir_spec, 'g15.sim_spectra.source_flux.r_ap%.2f.texp600.iexp%i.newsky.v2.fits' % (r_ap, iexp))
    f_rrbest = ''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.fits']) 
    f_rrout = ''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.h5']) 
    
    if not os.path.isfile(f_rrout): 
        cmd = ''.join(['rrdesi --mp ', str(ncpu), ' --zbest ', f_rrbest, ' --output ', f_rrout, ' ', f_simspec])  
        subprocess.check_output(cmd.split(' ') )

    # output best-fit redrock template
    targetids = None
    #- Templates
    templates_path = redrock.templates.find_templates(None)
    templates = {}
    for el in templates_path:
        t = redrock.templates.Template(filename=el)
        templates[t.full_type] = t
    targets = desi.DistTargetsDESI(f_simspec, targetids=None, coadd=True)._my_data
    zscan, zfit = redrock.results.read_zscan(f_rrout) 
    
    targetid = 0 
    target = targets[targetid]
    zfit = zfit[zfit['targetid'] == targetid]

    zz = zfit[zfit['znum'] == 0][0] # best-fit 
    coeff = zz['coeff'] 
    fulltype = zz['spectype']
    tp = templates[fulltype]
    
    ww = np.linspace(3000., 10000., 7001)
    bestfit_template = tp.eval(coeff[0:tp.nbasis], ww, zz['z']) * (1+zz['z'])
    
    f_rrbest_tp = ''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.best_tp.fits']) 
    if not os.path.isfile(f_rrbest_tp): 
        t = Table([ww, bestfit_template], names=('wave', 'flux')) 
        t.write(f_rrbest_tp, format='fits') 

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(211) 
    for spectype, fmt in [('STAR', 'k-'), ('GALAXY', 'b-'), ('QSO', 'g-')]:
        if spectype in zscan[target.id]:
            zx = zscan[target.id][spectype]
            sub.plot(zx['redshifts'], zx['zchi2'], fmt, alpha=0.2)
            sub.plot(zx['redshifts'], zx['zchi2']+zx['penalty'], fmt, label=spectype)
    sub.plot(zfit['z'], zfit['chi2'], 'r.')
    sub.set_xlabel('redshift', fontsize=20)
    sub.set_xlim([0., 0.5]) 
    sub.set_ylabel(r'$\chi^2$', fontsize=20)
    sub.set_ylim([6000., 8000.]) #0.5*zfit['chi2'].min(), 1.5*zfit['chi2'].max()])  
    sub.legend(loc='upper left', fontsize=20)

    sub = fig.add_subplot(212) 
    specs_to_read = target.spectra
    for i, spec in enumerate(specs_to_read):
        mx = tp.eval(coeff[0:tp.nbasis], spec.wave, zz['z']) * (1+zz['z'])
        model = spec.R.dot(mx)
        flux = spec.flux.copy()
        isbad = (spec.ivar == 0)
        ## model[isbad] = mx[isbad]
        flux[isbad] = np.NaN
        sub.plot(spec.wave, medfilt(flux, 1), alpha=0.5, label=r'$r_{ap}= '+str(r_ap)+'$ BGS spectra')
        sub.plot(spec.wave, medfilt(mx, 1), 'k:', alpha=0.8)
        model[isbad] = np.NaN
        sub.plot(spec.wave, medfilt(model, 1), 'k-', alpha=0.8, label='Best-fit')
        if i == 0: sub.legend(loc='upper right', fontsize=20)
    sub.set_xlabel('wavelength [A]', fontsize=20)
    sub.set_xlim([3600., 9800.]) 
    sub.set_ylabel('flux', fontsize=20)
    fig.savefig(''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.png']), bbox_inches='tight') 
    return None 


def Redrock_mockexp_gleg_simSpec_fiducial(r_ap, iexp=562, ncpu=4, flux='noisy'): 
    ''' Run redrock 
    '''
    import redrock
    import redrock.templates
    import redrock.archetypes
    import redrock.plotspec
    from redrock.external import desi
    from scipy.signal import medfilt

    dir_spec = os.path.join(UT.dat_dir(), 'spectra', 'gamadr3_legacydr7') 
    if flux == 'noisy': 
        f_simspec = os.path.join(dir_spec, 'g15.sim_spectra.r_ap%.2f.texp600.iexp%i.newsky.v2.fits' % (r_ap, iexp))
    elif flux == 'source': 
        f_simspec = os.path.join(dir_spec, 'g15.sim_spectra.source_flux.r_ap%.2f.texp600.iexp%i.newsky.v2.fits' % (r_ap, iexp))
    f_rrbest = ''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.fits']) 
    f_rrout = ''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.h5']) 
    
    if not os.path.isfile(f_rrout): 
        cmd = ''.join(['rrdesi --mp ', str(ncpu), ' --zbest ', f_rrbest, ' --output ', f_rrout, ' ', f_simspec])  
        subprocess.check_output(cmd.split(' ') )

    # output best-fit redrock template
    targetids = None
    #- Templates
    templates_path = redrock.templates.find_templates(None)
    templates = {}
    for el in templates_path:
        t = redrock.templates.Template(filename=el)
        templates[t.full_type] = t
    targets = desi.DistTargetsDESI(f_simspec, targetids=None, coadd=True)._my_data
    zscan, zfit = redrock.results.read_zscan(f_rrout) 
    
    targetid = 0 
    target = targets[targetid]
    zfit = zfit[zfit['targetid'] == targetid]

    zz = zfit[zfit['znum'] == 0][0] # best-fit 
    coeff = zz['coeff'] 
    fulltype = zz['spectype']
    tp = templates[fulltype]
    
    ww = np.linspace(3000., 10000., 7001)
    bestfit_template = tp.eval(coeff[0:tp.nbasis], ww, zz['z']) * (1+zz['z'])
    
    f_rrbest_tp = ''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.best_tp.fits']) 
    if not os.path.isfile(f_rrbest_tp): 
        t = Table([ww, bestfit_template], names=('wave', 'flux')) 
        t.write(f_rrbest_tp, format='fits') 

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(211) 
    for spectype, fmt in [('STAR', 'k-'), ('GALAXY', 'b-'), ('QSO', 'g-')]:
        if spectype in zscan[target.id]:
            zx = zscan[target.id][spectype]
            sub.plot(zx['redshifts'], zx['zchi2'], fmt, alpha=0.2)
            sub.plot(zx['redshifts'], zx['zchi2']+zx['penalty'], fmt, label=spectype)
    sub.plot(zfit['z'], zfit['chi2'], 'r.')
    sub.set_xlabel('redshift', fontsize=20)
    sub.set_xlim([0., 0.5]) 
    sub.set_ylabel(r'$\chi^2$', fontsize=20)
    #sub.set_ylim([6000., 8000.]) #0.5*zfit['chi2'].min(), 1.5*zfit['chi2'].max()])  
    #sub.set_ylim(zscan[target.id]['GALAXY']['zchi2'].min(), zscan[target.id]['GALAXY']['zchi2'].max())
    sub.set_ylim(zfit['chi2'].min(), zfit['chi2'].max())  
    sub.legend(loc='upper left', markerscale=4, frameon=True, fontsize=20)

    sub = fig.add_subplot(212) 
    specs_to_read = target.spectra
    for i, spec in enumerate(specs_to_read):
        mx = tp.eval(coeff[0:tp.nbasis], spec.wave, zz['z']) * (1+zz['z'])
        model = spec.R.dot(mx)
        flux = spec.flux.copy()
        isbad = (spec.ivar == 0)
        ## model[isbad] = mx[isbad]
        flux[isbad] = np.NaN
        #sub.plot(spec.wave, medfilt(flux, 1), alpha=0.5, label=r'$r_{ap}= '+str(r_ap)+'$ BGS spectra')
        sub.errorbar(spec.wave, medfilt(flux, 1), yerr=spec.ivar**-0.5, fmt=('.C%i' % i), label=r'$r_{ap}= '+str(r_ap)+'$ BGS spectra', zorder=0)
        sub.plot(spec.wave, medfilt(mx, 1), 'k:', lw=1, alpha=0.8)
        model[isbad] = np.NaN
        sub.plot(spec.wave, medfilt(model, 1), 'k-', alpha=0.8, lw=1, label='Best-fit')
        if i == 0: sub.legend(loc='upper right', fontsize=20)
    sub.set_xlabel('wavelength [A]', fontsize=20)
    sub.set_xlim(3600., 9800.) 
    sub.set_ylabel('flux', fontsize=20)
    sub.set_ylim(-5, 10.)
    fig.savefig(''.join([f_simspec.rsplit('.fits', 1)[0], '.rr.png']), bbox_inches='tight') 
    return None 


if __name__=='__main__': 
    mockexp_gleg_simSpec_fiducial(20.55, iexp=562, seed=1)
    Redrock_mockexp_gleg_simSpec_fiducial(20.55, iexp=562, ncpu=4, flux='noisy')
    Redrock_mockexp_gleg_simSpec_fiducial(20.55, iexp=562, ncpu=4, flux='source')
    #Redrock_gleg_simSpec_fiducial(20.55, ncpu=4)
