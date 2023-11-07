#!/bin/bash
#$ -l mem=5G
#$ -l h_rt=42:00:00
#$ -l gpu=1
#$ -pe smp 16
#$ -N o2omed_dro_vit
#$ -R y
# request an A100 node only
#$ -ac allow=L
#$ -S /bin/bash
#$ -wd /home/zcahaly/Scratch/DomainBed
#$ -j y

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate domainbed

date
nvidia-smi

# Run the Python command
python -m domainbed.scripts.loop_arch_resnet --dataset SpawriousO2O_medium --algorithm GroupDRO --architecture vit-b --da_bool False --n_iter 3