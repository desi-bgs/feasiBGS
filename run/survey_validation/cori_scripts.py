import os 
import numpy as np 
import subprocess


def redrock_scripts(i, flag='v0p5'): 
    ''' generate scripts for running redrock 
    '''
    # job name 
    fjob = os.path.join('cori_redrock%i.slurm' % i)

    queue = 'regular'
    constraint = 'knl'
    
    jb = '\n'.join([ 
        '#!/bin/bash',
        '#SBATCH --qos=%s' % queue, 
        '#SBATCH --time=01:30:00', 
        '#SBATCH --constraint=%s' % constraint,
        '#SBATCH -N 1',
        '#SBATCH -J sv_redrock%i' % i, 
        '#SBATCH -o _sv_redrock%i.o' % i, 
        '#SBATCH -L SCRATCH,project', 
        '', 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        '', 
        'source /project/projectdirs/desi/software/desi_environment.sh 19.2',
        '',
        'export OMP_NUM_THREADS=1', 
        '', 
        'dir_spec=$CSCRATCH"/feasibgs/survey_validation/"', 
        '',
        'f_str="GALeg.g15.bgsSpec.3000.%s.sample%i.seed0"' % (flag, i), 
        'f_spec=$dir_spec$f_str".fits"', 
        'f_redr=$dir_spec$f_str".rr.fits"', 
        'f_zout=$dir_spec$f_str".rr.h5"', 
        'rrdesi --mp 68 --zbest $f_redr --output $f_zout $f_spec', 
        '', 
        'now=$(date +"%T")', 
        'echo "end time ... $now"']) 

    job = open(fjob, 'w') 
    job.write(jb) 
    job.close() 
    return fjob 


def redrock_scripts_TSreview(iexp, texp): 
    ''' generate scripts for running redrock on TS review spectra
    '''
    # job name 
    fjob = os.path.join('cori_redrock.TSreview.exp%i.texp_%.f.slurm' % (iexp, texp))

    queue = 'regular'
    constraint = 'knl'
    
    jb = '\n'.join([ 
        '#!/bin/bash',
        '#SBATCH --qos=%s' % queue, 
        '#SBATCH --time=03:00:00', 
        '#SBATCH --constraint=%s' % constraint,
        '#SBATCH -N 1',
        '#SBATCH -J RR_TSreview_%i_%.f' % (iexp, texp),
        '#SBATCH -o _RR_TSreview_%i_%.f.o' % (iexp, texp),  
        '#SBATCH -L SCRATCH,project', 
        '', 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        '', 
        'source /project/projectdirs/desi/software/desi_environment.sh 19.2',
        '',
        'export OMP_NUM_THREADS=1', 
        '', 
        'dir_spec=$CSCRATCH"/feasibgs/survey_validation/"', 
        '',
        'f_str="GALeg.g15.bgsSpec.5000.TSreview.exp%i.texp_%.f"' % (iexp, texp), 
        'f_spec=$dir_spec$f_str".fits"', 
        'f_redr=$dir_spec$f_str".rr.fits"', 
        'f_zout=$dir_spec$f_str".rr.h5"', 
        'rrdesi --mp 68 --zbest $f_redr --output $f_zout $f_spec', 
        '', 
        'now=$(date +"%T")', 
        'echo "end time ... $now"']) 

    job = open(fjob, 'w') 
    job.write(jb) 
    job.close() 
    return fjob 


def submit_job(fjob):
    ''' run sbatch jobname.slurm 
    '''
    if not os.path.isfile(fjob): raise ValueError
    subprocess.check_output(['sbatch', fjob])
    return None 


if __name__=="__main__": 
    #for i in range(8): 
    #    fjob = redrock_scripts(i, flag='v0p5') 
    #    submit_job(fjob)
    texps = 60. * np.array([3, 5, 8, 12, 15]) # 3 to 15 min 
 
    for iexp in range(3): 
        for texp in texps: 
            fjob = redrock_scripts_TSreview(iexp, texp)
            submit_job(fjob)
