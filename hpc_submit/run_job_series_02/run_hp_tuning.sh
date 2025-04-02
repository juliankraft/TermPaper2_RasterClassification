#!/usr/bin/env bash

# Input file
INPUT_FILE="config.txt"
FOLDER="jobs"
RUN_NAME="hp_tuning_01_label:sealing_simple"
OUTPUT_FILE=$FOLDER/"submitted_jobs.txt"

JOB_LIST=()

# Ensure the file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: File $INPUT_FILE not found."
    exit 1
fi

# Check if the folder exists and delete it
if [[ -d "$FOLDER" ]]; then
    echo Folder exists
    echo "Removing folder: $FOLDER"
    rm -rf "$FOLDER"
else
    echo "Folder $FOLDER does not exist."
fi
    echo "Creating folder: $FOLDER"
    mkdir -p "$FOLDER"

# Initialize counter
counter=0

# Loop through each line in the file
while IFS= read -r line || [[ -n "$line" ]]; do

    counter=$((counter + 1))

    formatted_counter=$(printf "%03d" "$counter")
    FILE_PATH="$FOLDER/job_nr_$formatted_counter.sh"
    JOB_NAME="jk_$formatted_counter"
    MODEL_NAME="Model_$formatted_counter"

    cat > "$FILE_PATH" <<EOF
#!/usr/bin/env bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --mail-type=fail,end
#SBATCH --time=01-12:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=42G
#SBATCH --gres=gpu:l40s:1

# shellcheck disable=SC1091

# ## strict bash mode
set -eEuo pipefail

# module reset
module purge
module load DefaultModules

# ## init micromamba
export MAMBA_ROOT_PREFIX="/cfs/earth/scratch/\${USER}/.conda/"
eval "\$("/cfs/earth/scratch/\${USER}/bin/micromamba" shell hook -s posix)"

# ## Loading Arguments
PYTHON_ARGS=(
    --device=gpu
    --batch_size=256
    --num_workers=12
    --cutout_size=51
    --output_patch_size=5
    --patience=10
    --use_class_weights
    --disable_progress_bar
    --learning_rate=0.001
    --weight_decay=0.01
EOF

    # Add each argument from $line dynamically
    for arg in $line; do
        echo "    $arg" >> "$FILE_PATH"
    done

    # Append the rest of the script

    cat >> "$FILE_PATH" <<EOF
)
echo '#########################################################################################'
echo '### Model info: #########################################################################'
echo 
echo 'Hp tuning run name: $RUN_NAME'
echo 'Model name: $MODEL_NAME'
echo
echo '### Arguments:'
echo
for arg in "\${PYTHON_ARGS[@]}"; do
    echo "\$arg"
done
echo
echo '#########################################################################################'
echo '### Host info: ##########################################################################'
echo
echo 'Running on host:'
hostname
echo
nvidia-smi
echo
echo 'Working directory:'
cd /cfs/earth/scratch/kraftjul/sa2/code
pwd
echo
echo '#########################################################################################'
echo
echo '#########################################################################################'
echo '### Running skript ######################################################################'
echo '#########################################################################################'
echo

micromamba run -n sa2 python run_model.py "\${PYTHON_ARGS[@]}"

echo
echo '#########################################################################################'
echo '### Completed skript ####################################################################'
echo '#########################################################################################'

EOF

    

    echo "file created: $FILE_PATH"
    echo
    echo "Submitting job: $FILE_PATH"
    OUTPUT=$(sbatch "$FILE_PATH")

    # Check if the submission was successful
    if [[ "$OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"  # Extract the job ID from the response
        echo "Job submitted successfully."
        JOB_LIST+=("$JOB_NAME - ID: $JOB_ID")
    else
        echo "Error: Job submission failed for $FILE_PATH"
        exit 1
    fi
    
done < "$INPUT_FILE"

echo
echo "All jobs submitted successfully."
echo

# Display the list of job names and IDs at the end
if [[ ${#JOB_LIST[@]} -gt 0 ]]; then
    echo
    echo "List of submitted jobs:"
    for job in "${JOB_LIST[@]}"; do
        echo "  $job" | tee -a "$OUTPUT_FILE"
    
    done
else
    echo "No jobs were submitted."
fi