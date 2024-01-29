#!/bin/bash
# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="MTRSAP - ICRA 2024"
#SBATCH --error="my_job.err"
#SBATCH --output="my_job.output"
#SBATCH --partition="gpu"
#SBATCH --nodelist="jaguar02"

echo "$HOSTNAME"
conda init zsh &&
source /u/cjh9fw/.bashrc &&
conda activate icra24 &&
python -u train_recognition.py --model transformer --dataloader v2 --modality 16 &&
echo "Done" &&
exit
