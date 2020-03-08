#!/bin/bash
for i in {1..1000}
do
	echo "python3 /ddn/home/sklv77/HamiltonDir/param_search.py ${i}" >> runbase.slurm
	sbatch -p par7.q -n 1 -o /ddn/home/sklv77/HamiltonDir/data/d${i}.out -e /ddn/home/sklv77/HamiltonDir/err/e${i}.err --exclusive runbase.slurm
	sed -i '$d' runbase.slurm
done
