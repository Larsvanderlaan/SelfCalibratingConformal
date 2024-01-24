#!/bin/bash

#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=Rout/par-%J.out
#SBATCH --error=Rout/par-%J.err
#SBATCH --time=14-00:00:00
echo "LOADING R"
module load R
echo "R LOADED"
Rscript -e 'source("~/DRinference/scripts/R_setup.R"); source("~/DRinference/scripts/DR2.R"); n = as.numeric(Sys.getenv("n")); pos_const = as.numeric(Sys.getenv("const")); do_sims(n, 5000)'
