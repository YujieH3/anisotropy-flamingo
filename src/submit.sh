#!/bin/bash -l
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --time=7-00:00:00 # max 7 days
#SBATCH --output=/home/hey4/anisotropy-flamingo/logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --nodes=1
#SBATCH --ntasks=6	# max 24 per node for gpu_strw
#SBATCH --cpus-per-task=1
##SBATCH --mem=61G	# max 61G per node for gpu_strw

#SBATCH --job-name=fit3
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type=TIME_LIMIT_50,TIME_LIMIT_90,ALL

echo "Date: $(date)"
echo "Batch script submitted:"
cat "$0"
echo ""
echo "----"
echo ""

module purge
module load ALICE/default Miniconda3/24.7.1-0

conda activate cluster

relation=3

srun /home/hey4/.conda/envs/cluster/bin/python /home/hey4/anisotropy-flamingo/src/h0mc.py --mpi --num-relation $relation --corner-plot --sample-dir /home/hey4/anisotropy-flamingo/data/processed

# for i in {1..1000}
# do
#     srun /home/hey4/.conda/envs/cluster/bin/python /home/hey4/anisotropy-flamingo/src/h0mc.py --mpi --num-relation $relation --shuffle
# done