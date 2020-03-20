#!/bin/python
import os 
import glob
import numpy as np 


dir_redux = "/global/cfs/cdirs/desi/spectro/redux/daily" 
dir_output = "/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd"


def get_dates(tileid): 
    dates = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), "*"))
    dates = [os.path.basename(date) for date in dates]
    return dates 


def get_exposures(tileid, date): 
    cframes = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), date,
        'cframe-b*.fits')) 
    exps = [cframe.split('-')[-1].split('.fits')[0] for cframe in cframes]
    return np.unique(exps)


def get_spectograph(tileid, date, exp): 
    cframes = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), date,
        'cframe-b*-%s.fits' % exp.zfill(8)))
    spectographs = [os.path.basename(cframe).split('-')[1].split('b')[-1] 
            for cframe in cframes]
    return spectographs


def coadd(tileid, date, exp, spec): 
    
    cframe = os.path.join(dir_redux, 'tiles', str(tileid), date,
            'cframe-[brz]%s-%s.fits' % (spec, exp.zfill(8)))
    fcoadd = os.path.join(dir_output, 
            'coadd-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))
    cmd = 'desi_coadd_spectra --coadd-cameras -i %s -o %s' % (cframe, fcoadd)
    os.system(cmd) 
    return None 


def rr_coadd(tileid, date, exp, spec): 
    ''' run redrock on specified coadd  
    '''
    fcoadd  = os.path.join(dir_output, 
            'coadd-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))
    frr     = os.path.join(dir_output, 
            'redrock-%s-%s-%s-%s.h5' % (str(tileid), date, spec, exp.zfill(8)))
    fzbest  = os.path.join(dir_output, 
            'zbest-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -N 1", 
        "#SBATCH -C haswell", 
        "#SBATCH -q regular", 
        '#SBATCH -J rr_%s_%s' % (exp, spec),
        '#SBATCH -o _rr_%s_%s.o' % (exp, spec),
        "#SBATCH -t 00:10:00", 
        "", 
        "export OMP_NUM_THREADS=1", 
        "export OMP_PLACES=threads", 
        "export OMP_PROC_BIND=spread", 
        "", 
        "", 
        "conda activate desi", 
        "", 
        "srun -n 32 -c 2 --cpu-bind=cores rrdesi_mpi -o %s -z %s %s" % (frr, fzbest, fcoadd), 
        ""]) 
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()

    os.system('sbatch script.slurm') 
    os.system('rm script.slurm') 
    return None 


if __name__=="__main__": 
    bgs_minisv_tiles    = [70500, 70502, 70510]
    bgs_sv0_tiles       = [66000, 66014, 66003]
    bgs_tiles = bgs_minisv_tiles + bgs_sv0_tiles 
    
    for tile in bgs_tiles: 
        dates = get_dates(tile)
        for date in dates: 
            exps = get_exposures(tile, date)
            for exp in exps: 
                spectographs = get_spectograph(tile, date, exp)
                for spec in spectographs: 
                    #coadd(tile, date, exp, spec)
                    rr_coadd(tile, date, exp, spec)
