#!/bin/bash

# Outer Bash script

# List of Python commands
declare -a PYTHON_COMMANDS=(
    # no pretraining
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy LISA-CutMix"
    # GroupDRO algorithm with resnet50-nopretraining
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm GroupDRO --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm GroupDRO --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm GroupDRO --architecture resnet50-nopretraining --da_bool False"
    # MMD-AE algorithm with resnet50-nopretraining
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm MMD-AE --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm MMD-AE --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm MMD-AE --architecture resnet50-nopretraining --da_bool False"
   
    # vit

    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture vit-b --da_bool True --da_strategy LISA-CutMix"
    
    ## GroupDRO algorithm
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm GroupDRO --architecture vit-b --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm GroupDRO --architecture vit-b --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm GroupDRO --architecture vit-b --da_bool False"

    ## MMD-AE algorithm
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm MMD-AE --architecture vit-b --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm MMD-AE --architecture vit-b --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm MMD-AE --architecture vit-b --da_bool False"

    # resnet50

    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group B --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy random_shuffle-CutMix"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-Mixup"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group C --algorithm ERM --architecture resnet50 --da_bool True --da_strategy LISA-CutMix"
)

# Initialize a counter for the loop
counter=0

# Iterate through the list of Python commands
for CMD in "${PYTHON_COMMANDS[@]}"; do

    ((counter++))

    # Check if 10 iterations have passed
    if [ $((counter % 10)) -eq 0 ]; then
        # Pause for 1 hour
        sleep 1h
    fi

    DATASET_GROUP=$(echo $CMD | awk -F '--dataset_group ' '{split($2, a, " "); print a[1]}')
    ALGORITHM=$(echo $CMD | awk -F '--algorithm ' '{split($2, a, " "); print a[1]}')
    ARCHITECTURE=$(echo $CMD | awk -F '--architecture ' '{split($2, a, " "); print a[1]}')
    DA_BOOL=$(echo $CMD | awk -F '--da_bool ' '{split($2, a, " "); print a[1]}')
    DA_STRATEGY=$(echo $CMD | awk -F '--da_strategy ' '{split($2, a, " "); print a[1]}')

    # DATASET_GROUP=$(echo $CMD | sed -n 's/.*--dataset_group \([a-zA-Z0-9_]*\).*/\1/p')
    # ALGORITHM=$(echo $CMD | sed -n 's/.*--algorithm \([a-zA-Z0-9_]*\).*/\1/p')
    # ARCHITECTURE=$(echo $CMD | sed -n 's/.*--architecture \([a-zA-Z0-9_]*\).*/\1/p')
    # DA_BOOL=$(echo $CMD | sed -n 's/.*--da_bool \([a-zA-Z0-9_]*\).*/\1/p'd)
    # DA_STRATEGY=$(echo $CMD | sed -n 's/.*--da_strategy \([a-zA-Z0-9_]*\).*/\1/p')



    JOB_NAME="${DATASET_GROUP}_${ALGORITHM}_${ARCHITECTURE}_${DA_BOOL}"
    if [ ! -z "$DA_STRATEGY" ]; then
        JOB_NAME="${JOB_NAME}_${DA_STRATEGY}"
    fi

    # Create the inner Bash script
    INNER_SCRIPT="inner_script_${JOB_NAME}.sh"
    cat > $INNER_SCRIPT << EOL
#!/bin/bash
#$ -l mem=5G
#$ -l h_rt=42:00:00
#$ -l gpu=1
#$ -pe smp 16
#$ -N $JOB_NAME
#$ -R y
#$ -ac allow=E,F
#$ -S /bin/bash
#$ -wd /home/zcahaly/Scratch/DomainBed
#$ -j y

module load python/miniconda3/4.10.3
source \$UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate domainbed

date
nvidia-smi

# Run the Python command
$CMD
EOL

    # Make the inner script executable
    chmod +x $INNER_SCRIPT

    # Submit the inner script using qsub
    qsub $INNER_SCRIPT

    # Optional: Remove the inner script file after submission (uncomment if you want to enable this)
    rm $INNER_SCRIPT

done

