#!/bin/bash
#SBATCH -J julia
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o out    

export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1

julia -p 32 ./HSolver.jl --config config.json
