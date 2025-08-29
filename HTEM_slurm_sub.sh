#!/bin/bash
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -x cc0105,cc0106

mpirun vasp_std
