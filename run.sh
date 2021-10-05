#!/usr/bin/zsh
set -e

# active environment
source /home/maximw/tfe_env/bin/activate

nohup python -m src.start_benchmark files/G1_6_top/profile.hdf5 > files/G1_6_top/logs.txt &
nohup python -m src.start_benchmark files/G1_7_top/profile.hdf5 > files/G1_7_top/logs.txt &
