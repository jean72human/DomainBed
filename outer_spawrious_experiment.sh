#!/bin/bash

# Outer Bash script

# List of Python commands
declare -a PYTHON_COMMANDS=(
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool False"
    "python -m domainbed.scripts.loop_arch_resnet --dataset_group A --algorithm ERM --architecture resnet50-nopretraining --da_bool True --da_strategy random_shuffle-Mixup"
    # Add more commands here
)

# Iterate through the list of Python commands
for CMD in "${PYTHON_COMMANDS[@]}"; do

    # Parse the command to generate an appropriate job name for -N
    DATASET_GROUP=$(echo $CMD | grep -oP '--dataset_group \K\w+')
    ALGORITHM=$(echo $CMD | grep -oP '--algorithm \K\w+')
    ARCHITECTURE=$(echo $CMD | grep -oP '--architecture \K\w+')
    DA_BOOL=$(echo $CMD | grep -oP '--da_bool \K\w+')
    DA_STRATEGY=$(echo $CMD | grep -oP '--da_strategy \K\w+')

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
    # rm $INNER_SCRIPT

done
