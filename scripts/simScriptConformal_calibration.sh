#!/bin/usr/env bash
nsims=2500
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for d in 1 2 3 5 20
do
  for lrnr_name in "xg" "rf"
    do
    for shape in 1 2 3 4
      do
      for b in 0.3 0.5 0.6
        do
        sbatch  --export=d=$d,lrnr_name=$lrnr_name,shape=$shape,b=$b,name="calibration" ~/conformal/scripts/simScriptConformal.sbatch
        done
    done
  done
done
