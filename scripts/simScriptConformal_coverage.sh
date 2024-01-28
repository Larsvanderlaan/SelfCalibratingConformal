#!/bin/usr/env bash
nsims=2500
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for d in 1 2 3 5 20
do
  for lrnr_name in "gam"
    do
    for shape in 1
      do
      for b in 0.01 0.2 0.4 0.6 0.8 1
        do
        sbatch  --export=d=$d,lrnr_name=$lrnr_name,shape=$shape,b=$b,name="coverage" ~/conformal/scripts/simScriptConformal.sbatch
        done
    done
  done
done
