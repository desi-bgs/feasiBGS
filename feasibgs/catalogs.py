'''

submodules to handle Catalogs used in the project


'''
import os 
import numpy as np
import h5py
from astropy.io import fits 
from astropy.table import Table as aTable
from astropy.cosmology import FlatLambdaCDM
from pydl.pydlutils.spheregroup import spherematch
# -- local --
from . import util as UT


class Catalog(object): 
    ''' parent object for the objects in this module. Currently
    has no functionality
    '''
    def __init__(self): 
        self.catalog = None 
    
    def _h5py_create_dataset(self, grp, key, data): 
        ''' the arrays from the fits files do not play well with the new h5py 
        and python3
        '''
        if isinstance(data, np.chararray) or isinstance(data[0], np.str_): 
            _chararray = np.array(data, dtype=h5py.special_dtype(vlen=str))
            grp.create_dataset(key.lower(), data=_chararray) 
        elif isinstance(data[0], np.bool_): 
            _bool = np.zeros(len(data)).astype(bool) 
            _bool[data] = True
            grp.create_dataset(key.lower(), data=_bool)
        else: 
            grp.create_dataset(key.lower(), data=data)
        return None 

    def flux_to_mag(self, flux): 
        return 22.5 - 2.5*np.log10(flux)


class GAMA(Catalog):
    '''  class to build/read in photometric and spectroscopic overlap 
    of the GAMA DR2/DR3 data. 
    
    The GAMA DR2 data contains photometry and
    spectroscopy from GAMA I, which covers three regions of 48 deg^2 
    area for a total of 144 deg^2. 
    
    The GAMA DR3 data contains photometry and spectroscopy from GAMA II, 
    which covers the 14x6.5 GAMA regions in NGP (G02 region is EXCLUDED).
    '''
    def __init__(self): 
        pass 

    def Read(self, field, data_release=3, silent=True):
        ''' Read in spherematched photometric and spectroscopic 
        data from GAMA DR2 (constructed using _Build). 
        '''
        _file = self._File(field, data_release=data_release)
        if not os.path.isfile(_file): # if file is not constructed
            if not silent: print('Building %s' % _file) 
            if field == 'all': self._Build(data_release=data_release, silent=silent)
            else: self._fieldSplit(data_release=data_release, silent=silent)
    
        # read in data and compile onto a dictionary
        f = h5py.File(_file, 'r') 
        grp_p = f['photo'] # photo data
        grp_s = f['spec'] # spec data
        grp_k0 = f['kcorr_z0.0']
        grp_k1 = f['kcorr_z0.1']

        if not silent: 
            print('colums in GAMA photometry') 
            print(sorted(grp_p.keys()))
            print('========================')
            print('colums in GAMA spectroscopy')
            print(sorted(grp_s.keys()))
            print('========================')
            print('colums in GAMA kcorrects')
            print(sorted(grp_k0.keys()))
            print('========================')
            print('%i objects' % len(grp_p['ra'][...])) 
            print('========================')

        data = {} 
        for dkey, grp in zip(['photo', 'spec', 'kcorr_z0.0', 'kcorr_z0.1'], [grp_p, grp_s, grp_k0, grp_k1]): 
            data[dkey] = {} 
            for key in grp.keys():
                data[dkey][key] = grp[key][...]
        return data 
    
    def _File(self, field, data_release=3): 
        ''' hdf5 file name of spherematched photometric and spectroscopic 
        data from GAMA DR3. 

        notes
        -----
        * v2 flag was added when photometry catalog was changed from InputCatA.fits
        to TilingCat.fits 
        '''
        if field == 'all': 
            return ''.join([UT.dat_dir(), 'GAMA_photo_spec.DR', str(data_release), '.v2.hdf5']) # output file 
        elif field == 'g09': 
            return ''.join([UT.dat_dir(), 'GAMA_photo_spec.DR', str(data_release), '.G09.v2.hdf5']) # output file 
        elif field == 'g12': 
            return ''.join([UT.dat_dir(), 'GAMA_photo_spec.DR', str(data_release), '.G12.v2.hdf5']) # output file 
        elif field == 'g15': 
            return ''.join([UT.dat_dir(), 'GAMA_photo_spec.DR', str(data_release), '.G15.v2.hdf5']) # output file 

    def _Build(self, data_release=3, silent=True): 
        ''' Read in the photometric data and the spectroscopic data,
        spherematch them and write the intersecting data to hdf5 file. 
        '''
        if data_release == 3: 
            # this includes *three* of the four gama fields G02 field has its own data
            # read in photometry (GAMA`s tiling catalog; http://www.gama-survey.org/dr3/schema/table.php?id=3) 
            gama_p = fits.open(UT.dat_dir()+'gama/dr3/TilingCat.fits')[1].data
            # read in emission line measurements (http://www.gama-survey.org/dr3/schema/table.php?id=40) 
            gama_s = fits.open(UT.dat_dir()+'gama/dr3/GaussFitSimple.fits')[1].data
            # read in kcorrect z = 0.0 (http://www.gama-survey.org/dr2/schema/table.php?id=177)
            gama_k0 = self._readKcorrect(UT.dat_dir()+'gama/dr3/kcorr_model_z00.fits')
            # read in kcorrect z = 0.1 (http://www.gama-survey.org/dr2/schema/table.php?id=178)
            gama_k1 = self._readKcorrect(UT.dat_dir()+'gama/dr3/kcorr_model_z01.fits')

        elif data_release == 2: # Data Release 2 (what I had before) 
            # read in photometry (GAMA`s master input catalogue; http://www.gama-survey.org/dr2/schema/table.php?id=156)
            gama_p = fits.open(UT.dat_dir()+'gama/InputCatA.fits')[1].data
            # read in spectroscopy (http://www.gama-survey.org/dr2/schema/table.php?id=197)
            gama_s = fits.open(UT.dat_dir()+'gama/SpecLines.fits')[1].data
            # read in kcorrect z = 0.0 (http://www.gama-survey.org/dr2/schema/table.php?id=177)
            gama_k0 = self._readKcorrect(UT.dat_dir()+'gama/kcorr_z00.fits')
            # read in kcorrect z = 0.1 (http://www.gama-survey.org/dr2/schema/table.php?id=178)
            gama_k1 = self._readKcorrect(UT.dat_dir()+'gama/kcorr_z01.fits')
        if not silent: 
            #print('colums in GAMA photometry') 
            #print(sorted(gama_p.__dict__.keys()))
            print('%i GAMA photometry objects' % len(gama_p['ra']))
            print('========================')
            #print('colums in GAMA spectroscopy')
            #print(sorted(gama_s.__dict__.keys()))
            print('%i GAMA spectroscopy (emission line) objects' % len(gama_s['ra'])) 
            print('========================')
            #print('colums in GAMA k-correct')
            #print(sorted(gama_k0.__dict__.keys()))
            print('%i GAMA k-correct objects' % len(gama_k0['mass'])) 
            print('========================')
         
        # impose some common sense cuts to make sure there's SDSS photometry 
        # these magnitudes are extinction corrected!
        has_sdss_photo = (
                (gama_p['u_model'] > -9999.) & 
                (gama_p['g_model'] > -9999.) & 
                (gama_p['r_model'] > -9999.) & 
                (gama_p['i_model'] > -9999.) & 
                (gama_p['z_model'] > -9999.)) 
        # impose science catalog cuts 
        # sc >= 4: r < 19.8, GAMA II main survey
        # sc >= 5: r < 19.8 and satisfies r-band star-galaxy separation
        # sc = 6:  r < 19.4 and satisfies r-band star-galaxy separation
        # r = r_petro
        sciencecut = (gama_p['survey_class'] > 3) 
        # match cataid with spectroscopic data 
        has_spec = np.in1d(gama_p['cataid'], gama_s['cataid']) 
        # match cataid with k-correct data 
        assert np.array_equal(gama_k0['cataid'], gama_k1['cataid']) 
        has_kcorr = np.in1d(gama_p['cataid'], gama_k0['cataid'])
        # combined sample cut 
        sample_cut = (has_spec & sciencecut & has_kcorr & has_sdss_photo)

        if not silent: 
            print('of %i GAMA photometry objects' % len(gama_p['cataid']))
            print('========================')
            print('%i have SDSS photometry data' % np.sum(has_sdss_photo))
            print('========================')
            print('%i have spectroscopic data' % np.sum(has_spec))
            print('========================')
            print('%i have k-correct data' % np.sum(has_kcorr))
            print('========================')
            print('%i have all of the above' % np.sum(sample_cut))
            print('========================')
        # match up with spectroscopic data 
        s_match = np.searchsorted(gama_s['cataid'], gama_p['cataid'][sample_cut]) 
        assert np.array_equal(gama_s['cataid'][s_match], gama_p['cataid'][sample_cut]) 
        # match up with k-correct data 
        k_match = np.searchsorted(gama_k0['cataid'], gama_p['cataid'][sample_cut])
        assert np.array_equal(gama_k0['cataid'][k_match], gama_p['cataid'][sample_cut])
        
        # write everything into a hdf5 file 
        f = h5py.File(self._File('all', data_release=data_release), 'w') 
        # store photometry data in photometry group 
        grp_p = f.create_group('photo') 
        for key in gama_p.names:
            self._h5py_create_dataset(grp_p, key, gama_p[key][sample_cut])

        # store spectroscopic data in spectroscopic group 
        grp_s = f.create_group('spec') 
        for key in gama_s.names:
            self._h5py_create_dataset(grp_s, key, gama_s[key][s_match])

        # store kcorrect data in kcorrect groups
        grp_k0 = f.create_group('kcorr_z0.0') 
        for key in gama_k0.names: 
            self._h5py_create_dataset(grp_k0, key, gama_k0[key][k_match])

        grp_k1 = f.create_group('kcorr_z0.1') 
        for key in gama_k1.names:
            self._h5py_create_dataset(grp_k1, key, gama_k1[key][k_match])
        f.close() 
        return None 

    def _fieldSplit(self, data_release=3, silent=True): 
        ''' Split the GAMA photo-spectroscopic data into the differnt
        GAMA regions. Different regions have different r-mag limits and
        etc so treating them separately is the most sensible!
        '''
        all_gama = self.Read('all', data_release=data_release, silent=True)
        
        fields = ['g09', 'g12', 'g15']
        ra_min = [129.0, 174.0, 211.5]
        ra_max = [141.0, 186.0, 223.5]

        for i_f, field in enumerate(fields): 
            in_ra = ((all_gama['photo']['ra'] >= ra_min[i_f]) & (all_gama['photo']['ra'] <= ra_max[i_f]))
            if not silent: print('%i objects in %s field' % (np.sum(in_ra), field.upper()))
        
            # write each field into hdf5 files
            f = h5py.File(self._File(field, data_release=data_release), 'w') 

            for k_grp in all_gama.keys(): # photo, spec, kcorr_z0.0, kcorr_z0.1
                grp = f.create_group(k_grp) 
                for key in all_gama[k_grp].keys():
                    grp.create_dataset(key, data=all_gama[k_grp][key][in_ra]) 

            f.close() 
        return None

    def _readKcorrect(self, fitsfile): 
        ''' GAMA Kcorrect raises VerifyError if read in the usual fashion.
        '''
        f = fits.open(fitsfile)
        f.verify('fix') 
        return f[1].data


class GamaLegacy(Catalog): 
    ''' class to append imaging data from the Legacy survey DR7 for the objects 
    in the GAMA DR3 photo+spec data (.GAMA object). The objects in the final 
    catalog has GAMA photometry, GAMA spectroscopy, and Legacy-survey photometry
    '''
    def AbsMag(self, data, kcorr=0.1, H0=70, Om0=0.3, galext=False):  
        ''' Calculate absolute magnitude in SDSS u, g, r, i, z bands with kcorrect 
        at z=`kcorr` given the data dictionary from the `GamaLegacy.Read` method. 
        H0 and Om0 specifies the cosmology for the distance modulus. 
        '''
        # check data's structure 
        for k in ['gama-photo', 'gama-spec','gama-kcorr-z0.0', 'gama-kcorr-z0.1']: 
            if k not in data.keys(): 
                raise ValueError('input data does not have the approprite keys') 
        # check kcorr 
        if kcorr not in [0.0, 0.1]: raise ValueError('kcorr = 0.0, 0.1 only') 
    
        bands_sdss = ['u','g','r','i','z']
        # apparent magnitude from GAMA photometry
        if not galext: 
            mag_ugriz = np.array([data['gama-photo'][b+'_model'] for b in bands_sdss]) 
        else: 
            mag_ugriz =np.array([data['gama-kcorr-z0.1'][b+'_model'] for b in bands_sdss]) 

        redshift = data['gama-spec']['z']  # redshift
        # distance modulus 
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        D_L = cosmo.luminosity_distance(redshift).value # Mpc 
        DM = 5. * np.log10(1e5*D_L)
        # k-correct 
        if kcorr == 0.0: 
            kcorr = np.array([data['gama-kcorr-z0.0']['kcorr_'+b] for b in bands_sdss]) 
        elif kcorr == 0.1: 
            kcorr = np.array([data['gama-kcorr-z0.1']['kcorr_'+b] for b in bands_sdss])
        
        absmag_ugriz = mag_ugriz - DM - kcorr
        return absmag_ugriz
    
    def Read(self, field, dr_gama=3, dr_legacy=7, silent=True):
        ''' Read in objects from legacy survey DR 5 that overlap with the 
        GAMA photo+spectra objects
        '''
        fgleg = self._File(field, dr_gama=dr_gama, dr_legacy=dr_legacy)
        if not os.path.isfile(fgleg): # if file is not constructed
            if not silent: print('Building %s' % fgleg)
            self._Build(field, dr_gama=dr_gama, dr_legacy=dr_legacy, silent=silent)
    
        # read in data and compile onto a dictionary
        f = h5py.File(fgleg, 'r') 
        grp_gp = f['gama-photo'] 
        grp_gs = f['gama-spec']
        grp_k0 = f['gama-kcorr-z0.0']
        grp_k1 = f['gama-kcorr-z0.1']
        grp_lp = f['legacy-photo'] 

        if not silent: 
            print('colums in GAMA Photo Data:') 
            print(sorted(grp_gp.keys()))
            print('colums in GAMA Spec Data:') 
            print(sorted(grp_gs.keys()))
            print('colums in Legacy Data:') 
            print(sorted(grp_lp.keys()))
            print('========================')
            print('%i objects' % len(grp_gp['ra'][...])) 

        data = {} 
        for dk, grp in zip(['gama-photo', 'gama-spec', 'gama-kcorr-z0.0', 'gama-kcorr-z0.1', 'legacy-photo'], 
                [grp_gp, grp_gs, grp_k0, grp_k1, grp_lp]):
            data[dk] = {} 
            for key in grp.keys():
                data[dk][key] = grp[key][...]
        self.catalog = data.copy() 
        return data 

    def select(self, index=None): 
        ''' select objects in the catalog by their index
        '''
        if index is not None: 
            if isinstance(index, list): 
                index = np.array(index)
            elif isinstance(index, np.ndarray): 
                pass
            else: 
                raise ValueError("index can only be a list of array") 
            
            select_data = {} 
            for grp in self.catalog.keys(): 
                select_data[grp] = {} 
                for key in self.catalog[grp].keys(): 
                    select_data[grp][key] = self.catalog[grp][key][index]
        return select_data 

    def write(self, catalog, fname):  
        ''' Given dictionary with same structure as self.catalog 
        write to hdf5 file 
        '''
        f = h5py.File(fname, 'w') 
        for g in catalog.keys(): 
            grp = f.create_group(g) 
            for k in catalog[g].keys(): 
                grp.create_dataset(k, data=catalog[g][k]) 
        f.close() 
        return None 

    def _File(self, field, dr_gama=3, dr_legacy=7): 
        return ''.join([UT.dat_dir(), 'GAMAdr', str(dr_gama), '.', field, '.LEGACYdr', str(dr_legacy), '.v2.hdf5'])

    def _Build(self, field, dr_gama=3, dr_legacy=7, sweep_dir=None, silent=True): 
        ''' Get Legacy Survey photometry for objects in the GAMA DR`dr_gama`
        photo+spec objects from the sweep files. This is meant to run on nersc
        but you can also manually download the sweep files and specify the dir
        where the sweep files are located in. 
        '''
        if dr_legacy == 5: 
            if sweep_dir is None: sweep_dir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr5/sweep/5.0/'
            tractor_dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr5/tractor/'
        elif dr_legacy == 7: 
            if sweep_dir is None: sweep_dir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr7/sweep/7.1/'
            tractor_dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr7/tractor/'

        # read in the names of the sweep files 
        fsweep = ''.join([UT.dat_dir(), 'legacy/', field, '.sweep_list.dat'])
        if not os.path.isfile(fsweep): _ = self._getSweeps(field, silent=silent)
        sweep_files = np.loadtxt(fsweep, unpack=True, usecols=[0], dtype='S') 
        if not silent: print("there are %i sweep files in the %s GAMA region" % (len(sweep_files), field)) 

        # read in GAMA objects
        gama = GAMA() 
        gama_data = gama.Read(field, data_release=dr_gama, silent=silent)
    
        sweep_dict = {} 
        gama_photo_dict, gama_spec_dict, gama_kcorr0_dict, gama_kcorr1_dict = {}, {}, {}, {} 
        # loop through the files and only keep ones that spherematch with GAMA objects
        for i_f, f in enumerate(sweep_files): 
            # read in sweep object 
            sweep = fits.open(os.path.join(sweep_dir, f.decode('unicode_escape')))[1].data
            if not silent: print('matching %s' % os.path.join(sweep_dir, f.decode('unicode_escape'))) 
        
            # spherematch the sweep objects with GAMA objects 
            if len(sweep['ra']) > len(gama_data['photo']['ra']):
                match = spherematch(sweep['ra'], sweep['dec'], 
                        gama_data['photo']['ra'], gama_data['photo']['dec'], 0.000277778)
            else: 
                match_inv = spherematch(gama_data['photo']['ra'], gama_data['photo']['dec'], 
                        sweep['ra'], sweep['dec'], 0.000277778)
                match = [match_inv[1], match_inv[0], match_inv[2]] 

            if not silent: 
                print('%i matches from the %s sweep file' % (len(match[0]), f))
            
            # save sweep photometry to `sweep_dict`
            for key in sweep.names: 
                if i_f == 0: 
                    sweep_dict[key.lower()] = sweep[key][match[0]] 
                else: 
                    sweep_dict[key.lower()] = np.concatenate([sweep_dict[key.lower()], sweep[key][match[0]]]) 

            # save matching GAMA data ('photo', 'spec', and kcorrects) 
            for gkey, gdict in zip(['photo', 'spec', 'kcorr_z0.0', 'kcorr_z0.1'],
                    [gama_photo_dict, gama_spec_dict, gama_kcorr0_dict, gama_kcorr1_dict]): 
                for key in gama_data[gkey].keys(): 
                    if i_f == 0: 
                        gdict[key] = gama_data[gkey][key][match[1]]
                    else: 
                        gdict[key] = np.concatenate([gdict[key], gama_data[gkey][key][match[1]]])

            del sweep  # free memory? (apparently not really) 

        if not silent: 
            print('========================')
            print('%i objects out of %i GAMA objects mached' % (len(sweep_dict['ra']), len(gama_data['photo']['dec'])) )

        assert len(sweep_dict['ra']) == len(gama_photo_dict['ra']) 
        assert len(sweep_dict['ra']) == len(gama_spec_dict['ra']) 
        assert len(sweep_dict['ra']) == len(gama_kcorr0_dict['mass']) 
        assert len(sweep_dict['ra']) == len(gama_kcorr1_dict['mass']) 

        # writeout all the GAMA objects without sweep objects
        if not silent: 
            nosweep = ~np.in1d(gama_data['photo']['objid'], gama_photo_dict['objid']) 
            f_nosweep = ''.join([UT.dat_dir(), 
                'GAMAdr', str(dr_gama), '.', field, '.LEGACYdr', str(dr_legacy), '.nosweep_match.fits'])
            print('========================')
            print('Writing out RA, Dec of %i GAMA objects without Legacy sweep objects to %s' % 
                    (np.sum(nosweep), f_nosweep))
            tb = aTable([gama_data['photo']['ra'][nosweep], gama_data['photo']['dec'][nosweep]], 
                    names=('ra', 'dec'))
            tb.meta['COMMENTS'] = 'RA, Dec of GAMA objects without matches in Legacy DR5 sweep' 
            tb.write(f_nosweep, format='fits', overwrite=True) 
            #np.savetxt(f_nosweep, np.array([gama_data['photo']['ra'], gama_data['photo']['dec']]).T, header='RA, Dec')

        # read apfluxes from tractor catalogs 
        apflux_dict = self._getTractorApflux(sweep_dict['brickname'], sweep_dict['objid'], tractor_dir=tractor_dir) 
        assert apflux_dict['apflux_g'].shape[0] == len(sweep_dict['brickname']) 

        # save data to hdf5 file
        if not silent: print('writing to %s' % self._File(field, dr_gama=dr_gama, dr_legacy=dr_legacy))
        f = h5py.File(self._File(field, dr_gama=dr_gama, dr_legacy=dr_legacy), 'w') 
        grp_gp = f.create_group('gama-photo') 
        grp_gs = f.create_group('gama-spec') 
        grp_k0 = f.create_group('gama-kcorr-z0.0') 
        grp_k1 = f.create_group('gama-kcorr-z0.1') 
        grp_lp = f.create_group('legacy-photo') 
    
        for key in sweep_dict.keys():
            self._h5py_create_dataset(grp_lp, key, sweep_dict[key]) 
        for key in apflux_dict.keys(): # additional apflux data. 
            self._h5py_create_dataset(grp_lp, key, apflux_dict[key]) 
        for key in gama_photo_dict.keys(): 
            grp_gp.create_dataset(key, data=gama_photo_dict[key]) 
        for key in gama_spec_dict.keys(): 
            grp_gs.create_dataset(key, data=gama_spec_dict[key]) 
        for key in gama_kcorr0_dict.keys(): 
            grp_k0.create_dataset(key, data=gama_kcorr0_dict[key]) 
        for key in gama_kcorr1_dict.keys(): 
            grp_k1.create_dataset(key, data=gama_kcorr1_dict[key]) 
        f.close() 
        return None 

    def _getSweeps(self, field, silent=True): 
        ''' Construct list of sweep files given GAMA object.
        '''
        # read in GAMA objects in field 
        gama = GAMA() 
        if field == 'all': raise ValueError("only select specific GAMA fields; not the entire data release") 
        gama_data = gama.Read(field, silent=silent)
    
        # get brickmin and brickmax of sweep files 
        ra_mins = 10.*np.arange(gama_data['photo']['ra'].min() // 10., (gama_data['photo']['ra'].max() // 10.) + 1) 
        ra_maxs = ra_mins + 10.
        dec_mins = 5.*np.arange(gama_data['photo']['dec'].min() // 5., (gama_data['photo']['dec'].max() // 5.) + 1)
        dec_maxs = dec_mins + 5. 
        
        legacy_gama_sweep = []
        for i in range(len(ra_mins)): 
            for j in range(len(dec_mins)): 
                if dec_mins[j] < 0: pm_sign = 'm'
                else: pm_sign = 'p'
                brickmin = ''.join([str(int(ra_mins[i])).zfill(3), pm_sign, 
                    str(int(np.abs(dec_mins[j]))).zfill(3)])

                if dec_maxs[j] < 0: pm_sign = 'm'
                else: pm_sign = 'p'
                brickmax = ''.join([str(int(ra_maxs[i])).zfill(3), pm_sign, 
                    str(int(np.abs(dec_maxs[j]))).zfill(3)])
                
                f_sweep = ''.join(['sweep-', brickmin, '-', brickmax, '.fits'])
                legacy_gama_sweep.append(f_sweep)
                if not silent: print('... %s' % f_sweep)
        np.savetxt(''.join([UT.dat_dir(), 'legacy/', field, '.sweep_list.dat']), 
                legacy_gama_sweep, fmt='%s')
        return ra_mins, dec_mins 
    
    def _getTractorApflux(self, brickname, objids, 
            tractor_dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr7/tractor/', silent=True): 
        ''' The catalog is constructed from the sweep catalog and the 
        GAMA DR3 photo+spec data. The sweep catalog does not include 
        all the photometric data from the legacy survey. This methods 
        appends 'apflux_g', 'apflux_r', 'apflux_z' and relevant columsn 
        from the tractor files. 
        
        This can (and probably should) be extended to other columns 
        '''
        bricks_uniq = np.unique(brickname)  # unique bricks
        AAAs = np.array([brick[:3] for brick in bricks_uniq]) 
        
        # apfluxes in 'g', 'r', and 'z' bands 
        bands = ['g', 'r', 'z']
        apfluxes = np.zeros((3, len(brickname), 8)) 
        apflux_ivars = np.zeros((3, len(brickname), 8)) 
        apflux_resids = np.zeros((3, len(brickname), 8)) 

        n_brick = 0 
        for ii, AAA, brick in zip(range(len(AAAs)), AAAs, bricks_uniq): 
            name = ''.join([tractor_dir, AAA, '/tractor-', brick, '.fits'])
            if not silent: print('%i of %i unique bricks -- %s' % (ii, len(AAAs), brick)) 
            if not os.path.isfile(name): raise ValueError('%s tractor file not available' % name)
            f_tractor = fits.open(name) 
            tractor = f_tractor[1].data

            inbrick = (brickname == brick) 
            for i_k, key in enumerate(bands):
                apfluxes[i_k, inbrick, :] = tractor.field('apflux_'+key)[objids[inbrick]]
                apflux_ivars[i_k, inbrick, :] = tractor.field('apflux_ivar_'+key)[objids[inbrick]]
                apflux_resids[i_k, inbrick, :] = tractor.field('apflux_resid_'+key)[objids[inbrick]]
            n_brick += np.sum(inbrick)

        assert n_brick == len(brickname) 

        # return dictionary with appropriate keys 
        apflux_dict = {} 
        for i_k, key in enumerate(bands):
            apflux_dict['apflux_'+key] = apfluxes[i_k,:,:]
            apflux_dict['apflux_ivar_'+key] = apflux_ivars[i_k,:,:]
            apflux_dict['apflux_resid_'+key] = apflux_resids[i_k,:,:]

        return apflux_dict


class Legacy(Catalog): 
    '''
    '''

    def _1400deg2_test(self, dr=8, rlimit=None): 
        '''
        '''
        # hardcoded patch of sky 
        ra_min, ra_max      = 160., 230.
        dec_min, dec_max    = -2., 18.

        area = (np.radians(ra_max) - np.radians(ra_min))*(np.sin(np.radians(dec_max)) - np.sin(np.radians(dec_min))) 
        area *= (180/np.pi)**2
        print('%.f deg^2 test region' % area) 
        
        # read legacy sweeps data in 1400 deg^2 region 
        if rlimit is None:  
            _fsweep = os.path.join(UT.dat_dir(), 'survey_validation', 'legacy_sweeps.1400deg2.hdf5')
        else: 
            _fsweep = os.path.join(UT.dat_dir(), 'survey_validation', 'legacy_sweeps.1400deg2.rlim%.1f.hdf5' % rlimit)
        fsweep = h5py.File(_fsweep, 'r') 
        sweep = {} 
        for k in fsweep.keys(): sweep[k] = fsweep[k][...]
        print('%i sweep objects' % len(sweep['flux_r']))

        # spatial masking 
        _spatial_mask = self.spatial_mask(sweep['maskbits'], [sweep['nobs_g'], sweep['nobs_r'], sweep['nobs_z']])
        print('%i spatial mask' % np.sum(_spatial_mask))
        
        # star-galaxy separation 
        _star_galaxy = self.star_galaxy(sweep['gaia_phot_g_mean_mag'], sweep['flux_r'])
        print('%i star-galaxy separation' % np.sum(_star_galaxy))

        # quality cut 
        gmag = self.flux_to_mag(sweep['flux_g']/sweep['mw_transmission_g']) 
        rmag = self.flux_to_mag(sweep['flux_r']/sweep['mw_transmission_r'])
        zmag = self.flux_to_mag(sweep['flux_z']/sweep['mw_transmission_z'])
        _quality_cut = self.quality_cut(
                np.array([sweep['fracflux_g'], sweep['fracflux_r'], sweep['fracflux_z']]), 
                np.array([sweep['fracmasked_g'], sweep['fracmasked_r'], sweep['fracmasked_z']]),
                np.array([sweep['fracin_g'], sweep['fracin_r'], sweep['fracin_z']]), 
                gmag - rmag, 
                rmag - zmag) 
        print('%i quality cut' % np.sum(_quality_cut))

        sample_select = (_spatial_mask & _star_galaxy & _quality_cut) 

        print('%i (spatial mask) & (star-galaxy sep.) & (quality cut)' % (np.sum(sample_select)))
        fout = os.path.join(UT.dat_dir(), 'survey_validation', 'bgs.1400deg2.rlim%.1f.hdf5' % rlimit)
        f = h5py.File(fout, 'w') 
        for k in sweep.keys(): 
            self._h5py_create_dataset(f, k, sweep[k][sample_select])
        f.close() 
        return None 
        return None 


    def _1400deg2_test_random(self, dr=8, rlimit=None): 
        '''
        '''
        # hardcoded patch of sky 
        ra_min, ra_max      = 160., 230.
        dec_min, dec_max    = -2., 18.

        # read random data in the 14000 deg^2 region made by Omar
        _frand = os.path.join(UT.dat_dir(), 'survey_validation', 'dr8-south_random_160_230_-2_18_N_3.npy')
        rand = np.load(_frand)
        sweep = {} 
        for k in fsweep.keys(): sweep[k] = fsweep[k][...]
        print('%i sweep objects' % len(sweep['flux_r']))

        # spatial masking 
        _spatial_mask = self.spatial_mask(sweep['maskbits'], [sweep['nobs_g'], sweep['nobs_r'], sweep['nobs_z']])
        print('%i spatial mask' % np.sum(_spatial_mask))
        
        # star-galaxy separation 
        _star_galaxy = self.star_galaxy(sweep['gaia_phot_g_mean_mag'], sweep['flux_r'])
        print('%i star-galaxy separation' % np.sum(_star_galaxy))

        # quality cut 
        gmag = self.flux_to_mag(sweep['flux_g']/sweep['mw_transmission_g']) 
        rmag = self.flux_to_mag(sweep['flux_r']/sweep['mw_transmission_r'])
        zmag = self.flux_to_mag(sweep['flux_z']/sweep['mw_transmission_z'])
        _quality_cut = self.quality_cut(
                np.array([sweep['fracflux_g'], sweep['fracflux_r'], sweep['fracflux_z']]), 
                np.array([sweep['fracmasked_g'], sweep['fracmasked_r'], sweep['fracmasked_z']]),
                np.array([sweep['fracin_g'], sweep['fracin_r'], sweep['fracin_z']]), 
                gmag - rmag, 
                rmag - zmag) 
        print('%i quality cut' % np.sum(_quality_cut))

        sample_select = (_spatial_mask & _star_galaxy & _quality_cut) 

        print('%i (spatial mask) & (star-galaxy sep.) & (quality cut)' % (np.sum(sample_select)))
        fout = os.path.join(UT.dat_dir(), 'survey_validation', 'bgs.1400deg2.rlim%.1f.hdf5' % rlimit)
        f = h5py.File(fout, 'w') 
        for k in sweep.keys(): 
            self._h5py_create_dataset(f, k, sweep[k][sample_select])
        f.close() 
        return None 
        return None 

    def quality_cut(self, frac_flux, fracmasked, fracin, g_r, r_z):
        ''' apply baseline quality cut  

        * frac_flux_[g,r,z]<5 Not overwhelmed by neighbouring source (any band)
        * fracmasked_[g,r,z]<0.4 Model not dominated by masked pixels in any band
        * fracin_[g,r,z]>0.3 Most of the model flux not outside the region of the data used to fit the model
        * -1< g-r < 4 Not an absolutely bizarre colour
        * -1< r-z < 4 Not an absolutely bizarre colour
        '''
        assert frac_flux.shape[0] == 3
        assert fracmasked.shape[0] == 3
        assert fracin.shape[0] == 3
    
        # Not overwhelmed by neighbouring source (any band)
        _frac_flux = ((frac_flux[0] < 5.) & (frac_flux[1] < 5.) & (frac_flux[2] < 5.)) 
        # Model not dominated by masked pixels in any band
        _fracmasked = ((fracmasked[0] < 0.4) & (fracmasked[1] < 0.4) & (fracmasked[2] < 0.4)) 
        # Most of the model flux not outside the region of the data used to fit the model
        _fracin = ((fracin[0] > 0.3) & (fracin[1] > 0.3) & (fracin[2] > 0.3)) 
        # color cut
        _colorcut = ((g_r > -1.) & (g_r < 4.) & (r_z > -1.) & (r_z < 4.)) 

        cut = (_frac_flux & _fracmasked & _fracin & _colorcut) 
        return cut 

    def star_galaxy(self, gaia_G, r_flux): 
        ''' star-galaxy separation using GAIA and tractor photometry 
        (gaia G mag) - (raw r mag) > 0.6 or (gaia G mag) == 0 
        '''
        G_rr = gaia_G - self.flux_to_mag(r_flux)
        isgalaxy = (G_rr > 0.6) | (gaia_G == 0) 
        return isgalaxy 

    def spatial_mask(self, maskbits, nobs): 
        ''' spatial masking around 

        * bright stars
        * medium bright stars 
        * clusters 
        * large galaxies 
        '''
        nobs_g, nobs_r, nobs_z = nobs 
        BS = (np.uint64(maskbits) & np.uint64(2**1))!=0    # bright stars
        MS = (np.uint64(maskbits) & np.uint64(2**11))!=0   # medium bright stars 
        GC = (np.uint64(maskbits) & np.uint64(2**13))!=0   # clusters 
        LG = (np.uint64(maskbits) & np.uint64(2**12))!=0   # large galaxies 
        allmask = ((maskbits & 2**6) != 0) | ((maskbits & 2**5) != 0) | ((maskbits & 2**7) != 0)
        nobs = ((nobs_g < 1) | (nobs_r < 1) | (nobs_z < 1)) 
        mask = ~(BS | MS | GC | LG | allmask | nobs) 
        return mask

    def _collect_1400deg2_test(self, dr=8, rlimit=None): 
        ''' collect sweeps data within the same 1400 deg2 test region that Omar used for dr7 
        and save to file. 
        '''
        import glob 
        if dr != 8: raise NotImplementedError
        if os.environ['NERSC_HOST'] != 'cori': raise ValueError('this script is meant to run on cori only') 
    
        # hardcoded patch of sky 
        ra_min, ra_max      = 160., 230.
        dec_min, dec_max    = -2., 18.

        dir_legacy = '/project/projectdirs/cosmo/data/legacysurvey/'
        dir_north = os.path.join(dir_legacy, 'dr8/north/sweep/8.0')
        dir_south = os.path.join(dir_legacy, 'dr8/south/sweep/8.0')
        
        fsweeps_N = glob.glob('%s/*.fits' % dir_north) 
        print('%i North sweep files' % len(fsweeps_N))
        fsweeps_S = glob.glob('%s/*.fits' % dir_south) 
        print('%i South sweep files' % len(fsweeps_S))
        fsweeps = sorted([os.path.join(dir_north, _fs) for _fs in fsweeps_N] + [os.path.join(dir_south, _fs) for _fs in fsweeps_S])
    
        sweeps = {} 
        for _fsweep in fsweeps: 
            # get sweep RA and Dec range 
            sweep_ra_min, sweep_ra_max, sweep_dec_min, sweep_dec_max = self._parse_brickname(_fsweep) 
    
            # check whether it's in the region or not 
            not_in_region = (
                    (sweep_ra_max < ra_min) | 
                    (sweep_ra_min > ra_max) | 
                    (sweep_dec_max < dec_min) | 
                    (sweep_dec_min > dec_max)
                    ) 
            if not_in_region: continue 
    
            # read sweep file 
            sweep = fits.open(_fsweep)[1].data
            
            # area that's within the test region 
            mask_region = (
                (sweep['RA'] >= ra_min) & 
                (sweep['RA'] <= ra_max) & 
                (sweep['DEC'] >= dec_min) & 
                (sweep['DEC'] <= dec_max))  
            if np.sum(mask_region) == 0: continue 

            if rlimit is None: 
                rcut = np.ones(sweep['RA']).astype(bool) 
            else: 
                rflux = sweep['FLUX_R'] / sweep['MW_TRANSMISSION_R'] 
                rcut = (rflux > 10**((22.5-rlimit)/2.5)) 
            
            print('%i obj in %s' % (np.sum(mask_region), os.path.basename(_fsweep)))

            if len(sweeps.keys()) == 0: 
                for k in sweep.names: 
                    sweeps[k] = sweep[k][mask_region & rcut]
            else: 
                for k in sweep.names: 
                    sweeps[k] = np.concatenate([sweeps[k], sweep[k][mask_region & rcut]], axis=0) 

        if rlimit is None:  
            fout = os.path.join(UT.dat_dir(), 'survey_validation', 'legacy_sweeps.1400deg2.hdf5')
        else: 
            fout = os.path.join(UT.dat_dir(), 'survey_validation', 'legacy_sweeps.1400deg2.rlim%.1f.hdf5' % rlimit)
        f = h5py.File(fout, 'w') 
        for k in sweeps.keys(): 
            self._h5py_create_dataset(f, k, sweeps[k])
        f.close() 
        return None 
       
    def _parse_brickname(self, brickname): 
        ''' parse ra and dec range from brick name 
        '''
        name = os.path.basename(brickname).replace('.fits', '') # get rid of directory and ext 
        radec1 = name.split('-')[1]
        radec2 = name.split('-')[2] 
        
        if 'p' in radec1: _c = 'p'
        elif 'm' in radec1: _c = 'm' 
        ra_min = float(radec1.split(_c)[0]) 
        dec_min = float(radec1.split(_c)[1]) 

        if 'p' in radec2: _c = 'p'
        elif 'm' in radec2: _c = 'm' 
        ra_max = float(radec2.split(_c)[0]) 
        dec_max = float(radec2.split(_c)[1]) 

        return ra_min, ra_max, dec_min, dec_max
    
    def _Tycho(self, ra_lim=None, dec_lim=None): 
        ''' read in tycho2 catalog within RA and Dec range 
        '''
        _tycho = fits.open(os.path.join(UT.dat_dir(), 'survey_validation', 'tycho2.fits'))[1].data
        
        mask_region = np.ones(len(_tycho['RA'])).astype(bool) 
        if ra_lim is not None: 
            mask_region = mask_region & (_tycho['RA'] >= ra_lim[0]) & (_tycho['RA'] <= ra_lim[1])
        if dec_lim is not None: 
            mask_region = mask_region & (_tycho['DEC'] >= dec_lim[0]) & (_tycho['DEC'] <= dec_lim[1])

        tycho = {} 
        for k in _tycho.names: 
            tycho[k] = _tycho[k][mask_region] 
        return tycho

    def _LSLGA(self, ra_lim=None, dec_lim=None):
        ''' read in Legacy Survey Large Galaxy Atlas 
        '''
        _lslga = fits.open(os.path.join(UT.dat_dir(), 'survey_validation', 'LSLGA-v2.0.fits'))[1].data

        mask_region = np.ones(len(_lslga['RA'])).astype(bool) 
        if ra_lim is not None: 
            mask_region = mask_region & (_lslga['RA'] >= ra_lim[0]) & (_lslga['RA'] <= ra_lim[1])
        if dec_lim is not None: 
            mask_region = mask_region & (_lslga['DEC'] >= dec_lim[0]) & (_lslga['DEC'] <= dec_lim[1])

        lslga = {} 
        for k in _lslga.names: 
            lslga[k] = _lslga[k][mask_region] 
        return lslga


def _GamaLegacy_TractorAPFLUX(): 
    ''' Retroactively add apflux columns from the tractor catalogs 
    to the GamaLegacy catalog constructed and saved to file. This is a 
    hack.
    '''
    gleg = GamaLegacy() 

    # open saved gama-legacy catalog for appending
    f_gleg = h5py.File(gleg._File(), 'r+') 
    # legacy photometry group 
    grp_lp = f_gleg['legacy-photo'] 

    if 'apflux_g' in grp_lp.keys(): 
        # check that the columsn dont' already exist 
        f_gleg.close() 
        raise ValueError('apfluxes already in the catalog') 

    # read apfluxes from tractor catalogs 
    apflux_dict = gleg._getTractorApflux(grp_lp['brickname'].value, grp_lp['objid'].value, 
            dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr5/tractor/') 
    assert apflux_dict['apflux_g'].shape[0] == len(grp_lp['brickname'].value) 
    
    # save fluxes to the dataset 
    for key in apflux_dict.keys(): 
        grp_lp.create_dataset(key, data=apflux_dict[key]) 

    f_gleg.close()  
    return None 
