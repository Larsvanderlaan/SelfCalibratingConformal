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
dir_path = "~";
source(paste0(dir_path, "/conformal/scripts/R_setup.R"));
source(paste0(dir_path, "/conformal/scripts/run_sims.R"));
lrnr_name <- Sys.getenv("lrnr_name")
d <- Sys.getenv("d")
shape <- Sys.getenv("shape")
b <- Sys.getenv("b")
name <- Sys.getenv("name")

lrnr_name <- as.character(lrnr_name)
d <- as.numeric(d)
shape <- as.numeric(shape)
b <- as.numeric(b)
name <- as.character(name)

print(paste(lrnr_name, d, shape))

if(lrnr_name == "gam") {
  lrnr <- lrnr_gam
} else if(lrnr_name == "xg") {
  lrnr <- lrnr_xg
} else if(lrnr_name == "rf") {
  lrnr <- Lrnr_ranger$new()
}

n_train <- 1000
alpha <- 0.1
out <- run_sim_once(n_train = n_train, lrnr = lrnr, d = d, alpha = alpha, shape = shape,   b = b, n_test = 5000)
fwrite(out, file = paste0("~/conformal/results/sims_", name, "_", n_train, "_", lrnr_name, "_", d, "_", alpha, "_", shape, "_", b, ".csv"))
'
