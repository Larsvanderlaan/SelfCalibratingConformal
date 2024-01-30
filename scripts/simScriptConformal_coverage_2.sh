#!/bin/usr/env bash
nsims=2500
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
b=0
for d in 3 5 20
do
  for lrnr_name in "gam"
    do
    for shape in 1
      do
      for avar in 0.6
        do
        sbatch  --export=d=$d,lrnr_name=$lrnr_name,shape=$shape,b=$b,avar=$avar,name="coverage" ~/conformal/scripts/simScriptConformal.sbatch
        done
    done
  done
done
