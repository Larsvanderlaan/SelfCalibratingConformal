#!/bin/usr/env bash
nsims=2500
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for d in 1 2 5 10
do
  for lrnr_name in "gam" "xg" "rf"
      do
    for shape in 1 2 3 4
    do
    sbatch  --export=d=$d,lrnr_name=$lrnr_name,shape=$shape ~/conformal/scripts/simScriptConformal.sbatch
    done
  done
done
