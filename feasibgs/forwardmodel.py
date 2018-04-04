'''

forward model a spectra from meta data. 
most of the functions here serve as wrappers
for `desisim` code 

References: 
    - https://github.com/desihub/desisim/blob/master/doc/nb/bgs-reference-spectra.ipynb
    - https://github.com/desihub/desisim/blob/master/doc/nb/bgs-redshift-efficiency.ipynb 
    - https://github.com/desihub/desisim/blob/master/py/desisim/scripts/quickspectra.py
'''
import os
import time 
import numpy as np 
from scipy.spatial import cKDTree as KDTree
# -- astropy 
from astropy.table import vstack
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
# -- local -- 
from speclite import filters
import specsim.simulator
import specsim.config
from desimodel.io import load_throughput
from desisim.io import empty_metatable
from desisim.io import read_basis_templates
from desisim.obs import get_night
from desisim.templates import BGS
import desisim.simexp
import desisim.specsim
import desimodel.io
import desispec
import desitarget
from desispec.spectra import Spectra
from desispec.resolution import Resolution
from desitarget.cuts import isBGS_bright, isBGS_faint

os.environ['DESI_BASIS_TEMPLATES']='/Volumes/chang_eHDD/projects/desi/spectro/templates/basis_templates/v2.3'


class BGStree(object):
    '''class to deal with KDTree from BGS basis template metadata.
    Read in meta data of BGS basis templates, construct a KDTree
    and then use the KDTree to identify closest templates. 
    '''
    def __init__(self):
        # read in the meta data of the BGS basis templates
        self.meta = read_basis_templates(objtype='BGS', onlymeta=True)
        # construct KDTree from it 
        self.tree = KDTree(self.extractMeta())

    def extractMeta(self):
        ''' Extract quantities used to construct KDTree from the basis
        template meta data: redshift (z), M_0.1r, and 0.1(g-r).
        '''
        # read in necessary meta data
        zobj = self.meta['Z'].data  # redshift 
        mabs = self.meta['SDSS_UGRIZ_ABSMAG_Z01'].data # absolute magnitudes in SDSS ugriz bands kcorrected to z=0.1
        rmabs = mabs[:,2] # r-band absolute magnitude
        gr = mabs[:,1] - mabs[:,2] # g-r color 

        return np.vstack((zobj, rmabs, gr)).T
    
    def Query(self, matrix):
        '''Return the nearest template number based on the KD Tree.

        Parameters
        ----------
          matrix (numpy.ndarray): 
            M x N array (M=number of properties, N=number of objects) 
            in the same format as the corresponding function for each 
            object type (e.g., self.bgs).

        Returns
        -------
          - indx: index of nearest template (main item of interest) 
          - dist: distance to nearest template
        '''
        dist, indx = self.tree.query(matrix)
        return indx, dist


class BGStemplates(object):
    '''Generate spectra for BGS templates.  
    '''
    def __init__(self, wavemin=None, wavemax=None, dw=0.2):
        ''' initiate BGS template spectra. Mainly for initializing `desisim.templates.BGS`
        '''
        # default (buffered) wavelength vector
        if wavemin is None: self.wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None: self.wavemax = load_throughput('z').wavemax + 10.0
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        # initialize the templates once:
        self.bgs_templates = BGS(wave=self.wave, normfilter='decam2014-r') 
        #normfilter='sdss2010-r') # Need to generalize this!
        self.bgs_templates.normline = None # no emission lines!

    def Spectra(self, r_mag, zred, vdisp, seed=None, templateid=None, silent=True):
        ''' Given data in the output format of `feasibgs.catalog.Read`
        generate spectra given the `templateid`.
        '''
        np.random.seed(seed) # set random seed

        # meta data of 'mag', 'redshift', 'vdisp'
        input_meta = empty_metatable(nmodel=len(r_mag), objtype='BGS')
        input_meta['SEED'] = np.random.randint(2**32, size=len(r_mag)) 
        input_meta['MAG'] = r_mag # r band apparent magnitude
        input_meta['REDSHIFT'] = zred # redshift
        input_meta['VDISP'] = vdisp 
        input_meta['TEMPLATEID'] = templateid

        flux, _, meta = self.bgs_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, 
                                                          novdisp=False,
                                                          verbose=(not silent))
        return flux, self.wave, meta

    def simExposure(self, wave, flux, airmass=1.0, exptime=1000, moonalt=-60, moonsep=180, moonfrac=0.0, seeing=1.1, 
            seed=1, skyerr=0.0): 
        ''' insert description here 
        '''
        nspec, _ = flux.shape # number of spectra 

        # observation conditions
        obvs_dict = dict(
                AIRMASS=airmass, 
                EXPTIME=exptime, # s 
                MOONALT=moonalt, # deg
                MOONFRAC=moonfrac,
                MOONSEP=moonsep, # deg
                SEEING=seeing)   # arc sec

        tileid  = 0
        dateobs = time.gmtime()
        night   = get_night(utc=dateobs)
        
        frame_fibermap = desispec.io.fibermap.empty_fibermap(nspec) # empty fibermap ndarray
        frame_fibermap.meta["FLAVOR"] = "custom"
        frame_fibermap.meta["NIGHT"] = night
        frame_fibermap.meta["EXPID"] = 0 
        # add DESI_TARGET
        tm = desitarget.targetmask.desi_mask
        frame_fibermap['DESI_TARGET'] = tm.BGS_ANY
        frame_fibermap['TARGETID'] =  np.arange(nspec).astype(int)
    
        # spectra fibermap has two extra fields : night and expid
        # This would be cleaner if desispec would provide the spectra equivalent
        # of desispec.io.empty_fibermap()
        spectra_fibermap = desispec.io.empty_fibermap(nspec)
        spectra_fibermap = desispec.io.util.add_columns(spectra_fibermap,
                           ['NIGHT', 'EXPID', 'TILEID'],
                           [np.int32(night), np.int32(0), np.int32(tileid)],
                           )
        for s in range(nspec):
            for tp in frame_fibermap.dtype.fields:
                spectra_fibermap[s][tp] = frame_fibermap[s][tp]

        # ccd wavelength limit 
        params = desimodel.io.load_desiparams()
        wavemin = params['ccd']['b']['wavemin']
        wavemax = params['ccd']['z']['wavemax']

        if wave[0] > wavemin or wave[-1] < wavemax:
            raise ValueError

        wlim = (wavemin <= wave) & (wave <= wavemax) # wavelength limit 
        wave = wave[wlim]*u.Angstrom

        flux_unit = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm ** 2 )
        flux = flux[:,wlim]*flux_unit

        sim = self._simulate_spectra(wave, flux, fibermap=frame_fibermap,
            obsconditions=obvs_dict, redshift=None, seed=seed, psfconvolve=True)
        
        # put in random noise 
        random_state = np.random.RandomState(seed)
        sim.generate_random_noise(random_state)

        scale=1e17
        specdata = None

        resolution={}
        for camera in sim.instrument.cameras:
            R = Resolution(camera.get_output_resolution_matrix())
            resolution[camera.name] = np.tile(R.to_fits_array(), [nspec, 0, 1])

        skyscale = skyerr * random_state.normal(size=sim.num_fibers)

        for table in sim.camera_output :
            
            wave = table['wavelength'].astype(float)
            flux = (table['observed_flux']+table['random_noise_electrons']*table['flux_calibration']).T.astype(float)
            if np.any(skyscale):
                flux += ((table['num_sky_electrons']*skyscale)*table['flux_calibration']).T.astype(float)

            ivar = table['flux_inverse_variance'].T.astype(float)
            
            band  = table.meta['name'].strip()[0]
            
            flux = flux * scale
            ivar = ivar / scale**2
            mask  = np.zeros(flux.shape).astype(int)
            
            spec = Spectra([band], {band : wave}, {band : flux}, {band : ivar}, 
                           resolution_data={band : resolution[band]}, 
                           mask={band : mask}, 
                           fibermap=spectra_fibermap, 
                           meta=None,
                           single=True)
            if specdata is None :
                specdata = spec
            else:
                specdata.update(spec)
        return specdata  

    def _simulate_spectra(self, wave, flux, fibermap=None, obsconditions=None, redshift=None,
                     dwave_out=None, seed=None, psfconvolve=True,
                     specsim_config_file = "desi"):
        '''
        A more streamlined BGS version of the method `desisim.simexp.simulate_spectra`, which 
        simulates an exposure 

        Args:
            wave (array): 1D wavelengths in Angstroms
            flux (array): 2D[nspec,nwave] flux in 1e-17 erg/s/cm2/Angstrom
                or astropy Quantity with flux units

        Optional:
            fibermap: table from fiberassign or fibermap; uses X/YFOCAL_DESIGN, TARGETID, DESI_TARGET
            obsconditions: (dict-like) observation metadata including
                SEEING (arcsec), EXPTIME (sec), AIRMASS,
                MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)
            redshift : list/array with each index being the redshifts for that target
            seed: (int) random seed
            psfconvolve: (bool) passed to simspec.simulator.Simulator camera_output.
                if True, convolve with PSF and include per-camera outputs
            specsim_config_file: (str) path to DESI instrument config file.
                default is desi config in specsim package.
        TODO: galsim support

        Returns a specsim.simulator.Simulator object
        '''
        # Input cosmology to calculate the angular diameter distance of the galaxy's redshift
        LCDM = FlatLambdaCDM(H0=70, Om0=0.3)
        ang_diam_dist = LCDM.angular_diameter_distance
        
        random_state = np.random.RandomState(seed)

        nspec, nwave = flux.shape

        #- Convert to unit-ful quantities for specsim
        #if not isinstance(flux, u.Quantity):
        #    fluxunits = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm**2)
        #    flux = flux * fluxunits

        #if not isinstance(wave, u.Quantity):
        #    wave = wave * u.Angstrom

        # Generate specsim config object for a given wavelength grid
        config = desisim.simexp._specsim_config_for_wave(wave.to('Angstrom').value, 
                dwave_out=dwave_out, specsim_config_file=specsim_config_file)

        #- Create simulator
        desi = specsim.simulator.Simulator(config, nspec,
            camera_output=psfconvolve)
        #desisim.specsim.get_simulator(config, num_fibers=nspec,
        #    camera_output=psfconvolve)

        if obsconditions is None: raise ValueError

        desi.atmosphere.seeing_fwhm_ref = obsconditions['SEEING'] * u.arcsec
        desi.observation.exposure_time = obsconditions['EXPTIME'] * u.s
        desi.atmosphere.airmass = obsconditions['AIRMASS']
        desi.atmosphere.moon.moon_phase = np.arccos(2*obsconditions['MOONFRAC']-1)/np.pi
        desi.atmosphere.moon.moon_zenith = (90 - obsconditions['MOONALT']) * u.deg
        desi.atmosphere.moon.separation_angle = obsconditions['MOONSEP'] * u.deg

        #- Set fiber locations from meta Table or default fiberpos
        fiberpos = desimodel.io.load_fiberpos()
        if len(fiberpos) != len(fibermap):
            ii = np.in1d(fiberpos['FIBER'], fibermap['FIBER'])
            fiberpos = fiberpos[ii]

        #- Extract fiber locations from meta Table -> xy[nspec,2]
        assert np.all(fibermap['FIBER'] == fiberpos['FIBER'][0:nspec])
        if 'XFOCAL_DESIGN' in fibermap.dtype.names:
            xy = np.vstack([fibermap['XFOCAL_DESIGN'], fibermap['YFOCAL_DESIGN']]).T * u.mm
        elif 'X' in fibermap.dtype.names:
            xy = np.vstack([fibermap['X'], fibermap['Y']]).T * u.mm
        else:
            xy = np.vstack([fiberpos['X'], fiberpos['Y']]).T * u.mm

        if 'TARGETID' in fibermap.dtype.names:
            unassigned = (fibermap['TARGETID'] == -1)
            if np.any(unassigned):
                #- see https://github.com/astropy/astropy/issues/5961
                #- for the units -> array -> units trick
                xy[unassigned,0] = np.asarray(fiberpos['X'][unassigned], dtype=xy.dtype) * u.mm
                xy[unassigned,1] = np.asarray(fiberpos['Y'][unassigned], dtype=xy.dtype) * u.mm
            
        #- Determine source types
        source_types = desisim.simexp.get_source_types(fibermap)
        if np.any(source_types != "bgs"): raise ValueError("source types are not all BGS!") 

        desi.instrument.fiberloss_method = 'fastsim'

        source_fraction=None
        source_half_light_radius=None

        # BGS parameters based on SDSS main sample, in g-band
        # see analysis from J. Moustakas in
        # https://github.com/desihub/desitarget/blob/master/doc/nb/bgs-morphology-properties.ipynb 
        # B/T (bulge-to-total ratio): 0.48 (0.36 - 0.59).
        # Bulge Sersic n: 2.27 (1.12 - 3.60).
        # log10 (Bulge Half-light radius): 0.11 (-0.077 - 0.307) arcsec
        # log10 (Disk Half-light radius): 0.67 (0.54 - 0.82) arcsec
        # This gives
        # bulge_fraction = 0.48
        # disk_fraction  = 0.52
        # bulge_half_light_radius = 1.3 arcsec
        # disk_half_light_radius  = 4.7 arcsec
        # note we use De Vaucouleurs' law , which correspond to a Sersic index n=4
        
        # source_fraction[:,0] is DISK profile (exponential) fraction
        # source_fraction[:,1] is BULGE profile (devaucouleurs) fraction
        # 1 - np.sum(source_fraction,axis=1) is POINT source profile fraction
        # see specsim.GalsimFiberlossCalculator.create_source routine
        source_fraction=np.zeros((nspec,2)) 
        source_fraction[:,0]=0.52 # disk comp in BGS
        source_fraction[:,1]=0.48 # bulge comp in BGS       

        # source_half_light_radius[:,0] is the half light radius in arcsec for the DISK profile
        # source_half_light_radius[:,1] is the half light radius in arcsec for the BULGE profile        
        # see specsim.GalsimFiberlossCalculator.create_source routine
        source_half_light_radius=np.zeros((nspec,2))
        
        # 4.7 is angular size of z=0.1 disk, and 1.3 is angular size of z=0.1 bulge
        bgs_disk_z01 = 4.7  # in arcsec
        bgs_bulge_z01 = 1.3 # in arcsec
        
        # Convert to angular size of the objects in this sample with given redshifts
        if redshift is None:
            angscales = np.ones(len(source_types))
        else:
            # Avoid infinities
            if np.any(redshift <= 0.):
                bgs_redshifts[redshift <= 0.] = 0.0001
            angscales = ( ang_diam_dist(0.1) / ang_diam_dist(redshift) ).value
        source_half_light_radius[:,0]= bgs_disk_z01 * angscales # disk comp in BGS, arcsec
        source_half_light_radius[:,1]= bgs_bulge_z01 * angscales  # bulge comp in BGS, arcsec
            
        #- Work around randomness in specsim quickfiberloss calculations
        #- while not impacting global random state.
        #- See https://github.com/desihub/specsim/issues/83
        randstate = np.random.get_state()
        np.random.seed(seed)
        desi.simulate(source_fluxes=flux, focal_positions=xy, source_types=source_types,
                      source_fraction=source_fraction,
                      source_half_light_radius=source_half_light_radius,
                      source_minor_major_axis_ratio=None,
                      source_position_angle=None)
        np.random.set_state(randstate)
        return desi
        
