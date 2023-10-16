#$ -l mem=5G
#$ -l h_rt=42:00:00
#$ -l gpu=1
#$ -pe smp 16
#$ -N resnet50-nopretraining da false
#$ -R y
#$ -ac allow=E,F
#$ -S /bin/bash
#$ -wd /home/zcahaly/Scratch/DomainBed
#$ -j y

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate domainbed

date
nvidia-smi

## resnet18

# Group A, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet18 --da_bool False
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet18 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet18 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet18 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet18 --da_bool True --da_strategy LISA-CutMix

# # Group B, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet18 --da_bool False
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet18 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet18 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet18 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet18 --da_bool True --da_strategy LISA-CutMix

# # Group C, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet18 --da_bool False
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet18 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet18 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet18 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet18 --da_bool True --da_strategy LISA-CutMix

# # Group A, GroupDRO algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm GroupDRO --architecture resnet18 --da_bool False

# # Group B, GroupDRO algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm GroupDRO --architecture resnet18 --da_bool False

# # Group C, GroupDRO algorithm 
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm GroupDRO --architecture resnet18 --da_bool False

# # Group A, MMD-AE algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm MMD-AE --architecture resnet18 --da_bool False

# # Group B, MMD-AE algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm MMD-AE --architecture resnet18 --da_bool False

# # Group C, MMD-AE algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm MMD-AE --architecture resnet18 --da_bool False


# ## resnet50-nopretraining

# # Group A, ERM algorithm
python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix

# # Group B, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix

# # Group C, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix

# # Group A, GroupDRO algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm GroupDRO --architecture resnet50-nopretraining --da_bool False

# # Group A, MMD-AE algorithm
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm MMD-AE --architecture resnet50-nopretraining --da_bool False

# ## Additional combinations with resnet50 and all da_strategy options

# # Group A, ERM algorithm with resnet50 and various da_strategy options
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix

# # Group B, ERM algorithm with resnet50 and various da_strategy options
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix

# # Group C, ERM algorithm with resnet50 and various da_strategy options
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet.py --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix