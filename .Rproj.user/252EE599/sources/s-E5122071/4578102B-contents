#!/bin/usr/env bash
nsims=2500
export R_LIBS=~/Rlibs2
export R_LIBS_USER=~/Rlibs2
for n in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
  for const in 2
      do
    for misp in 1
    do
    sbatch  --export=n=$n,const=$const ~/DRinference/scripts/simScriptDR.sbatch
    done
  done
done
