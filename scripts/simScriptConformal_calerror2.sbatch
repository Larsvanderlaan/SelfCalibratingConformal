#!/bin/bash

#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=Rout/par-%J.out
#SBATCH --error=Rout/par-%J.err
#SBATCH --time=14-00:00:00
echo "LOADING R"
module load R
module load Python/3.8.2-GCCcore-9.3.0
echo "R LOADED"
Rscript -e '
set.seed(12345)
dir_path = "~";
source(paste0(dir_path, "/conformal/scripts/R_setup.R"));
source(paste0(dir_path, "/conformal/scripts/run_sims.R"));
source(paste0(dir_path, "/conformal/scripts/sim_calerror2.R"));
'

