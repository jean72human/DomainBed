#$ -l mem=5G
#$ -l h_rt=42:00:00
#$ -l gpu=1
#$ -pe smp 16
#$ -N tester_for_script
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


# ## resnet50-nopretraining

# # Group A, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix

# # Group B, ERM algorithm
python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix

# # Group C, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix

# # GroupDRO algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm GroupDRO --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm GroupDRO --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm GroupDRO --architecture resnet50-nopretraining --da_bool False

# # MMD-AE algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm MMD-AE --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm MMD-AE --architecture resnet50-nopretraining --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm MMD-AE --architecture resnet50-nopretraining --da_bool False

### vit-b

# Group A, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-CutMix

# Group B, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-CutMix

# Group C, ERM algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-CutMix

## GroupDRO algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm GroupDRO --architecture vit-b --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm GroupDRO --architecture vit-b --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm GroupDRO --architecture vit-b --da_bool False

## MMD-AE algorithm
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm MMD-AE --architecture vit-b --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm MMD-AE --architecture vit-b --da_bool False
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm MMD-AE --architecture vit-b --da_bool False


# ## Additional combinations with resnet50 and all da_strategy options

# # Group A, ERM algorithm with resnet50 and various da_strategy options
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix

# # Group B, ERM algorithm with resnet50 and various da_strategy options
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix

# # Group C, ERM algorithm with resnet50 and various da_strategy options
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup
# python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix