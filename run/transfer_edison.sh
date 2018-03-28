#!bin/bash/
# script to transfer file from edison to local 
local_dir="/Volumes/chang_eHDD/projects/desi/mocks/bgs/MXXL/desi_footprint/v0.0.4"
edison_dir="/project/projectdirs/desi/mocks/bgs/MXXL/desi_footprint/v0.0.4"
if [ ! -d "$local_dir" ]; then 
    # make directory tree
    mkdir -p $local_dir
fi
echo "password"
read -s pwd
# transfer all files in the directory
sshpass -p $pwd scp edison:$edison_dir/* $local_dir
