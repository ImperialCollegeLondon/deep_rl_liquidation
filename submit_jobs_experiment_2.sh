#!/bin/sh

INDIR="/Users/melodiemonod/git/deep_rl_liquidation"
OUTDIR="/Users/melodiemonod/projects/2024/deep_rl_liquidation"

# Jobname of the training (experiment 1) 
JOBNAME_TRAINING="experiment_1-exponential_decay_kernel"
CWD_TRAINING="$OUTDIR/${JOBNAME_TRAINING}"

# Check if the training directory exists
if [ ! -d "$CWD_TRAINING" ]; then
    echo "Error: Training directory $CWD_TRAINING does not exist. Please run experiment 1 first"
    exit 1
fi

# Read JSON data from config.json
json_data=$(cat $INDIR/config_experiment_2.json)

# Parse JSON and process each entry
echo "$json_data" | sed -n 's/.*\({[^}]*}\).*/\1/p' | while read -r entry; do

    # Extract job_name using grep (carefully crafted pattern)
    JOBNAME=$(echo "$entry" | sed -n 's/.*"job_name":"\([^"]*\)".*/\1/p')

    # Directory to folder
    CWD="$OUTDIR/${JOBNAME}"
    
    # Create folder
    mkdir -p "$CWD"

    # Add path to memory and trained parameters to JSON entry
    new_key_value="\"file_pre_trained_critic_weights\": \"$CWD_TRAINING/weights_critic.pth\""
    escaped_new_key_value=$(echo "$new_key_value" | sed 's/\//\\\//g')
    updated_entry=$(echo "$entry" | sed "s/}$/,$escaped_new_key_value}/")

    new_key_value="\"file_pre_trained_actor_weights\": \"$CWD_TRAINING/weights_actor.pth\""
    escaped_new_key_value=$(echo "$new_key_value" | sed 's/\//\\\//g')
    updated_entry=$(echo "$updated_entry" | sed "s/}$/,$escaped_new_key_value}/")

    new_key_value="\"file_pre_trained_memory\": \"$CWD_TRAINING/outputs_memory.pkl\""
    escaped_new_key_value=$(echo "$new_key_value" | sed 's/\//\\\//g')
    updated_entry=$(echo "$updated_entry" | sed "s/}$/,$escaped_new_key_value}/")

    # Create a JSON file with the entire entry (avoid parsing within loop)
    echo "$updated_entry" > "$CWD/${JOBNAME}.json"
    
    # Create pbs
    cat > $CWD/${JOBNAME}.sh <<EOF

#!/bin/sh

source ~/.bash_profile # source the conda configuration file

conda activate deep_rl_liquidation

INDIR=$INDIR
CWD=$CWD
JOBNAME=$JOBNAME

python \$INDIR/run_experiment.py --config=\$CWD/\$JOBNAME.json --fpath=\$CWD

EOF

done

echo "Scripts to run experiment 2 generated in $OUTDIR"