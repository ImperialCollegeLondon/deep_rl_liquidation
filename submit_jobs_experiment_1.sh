#!/bin/sh

INDIR="/Users/melodiemonod/git/deep_rl_liquidation"
OUTDIR="/Users/melodiemonod/projects/2024/deep_rl_liquidation"

# Check if the environment deep_rl_liquidation exists
if conda env list | grep -q "^deep_rl_liquidation"; then
    echo "Conda environment deep_rl_liquidation found."
else
    echo "Conda environment deep_rl_liquidation not found. Please create it first."
    exit 1
fi

# Read JSON data from config.json
json_data=$(cat $INDIR/config_experiment_1.json)

# Parse JSON and process each entry
echo "$json_data" | sed -n 's/.*\({[^}]*}\).*/\1/p' | while read -r entry; do

    # Extract job_name using grep (carefully crafted pattern)
    JOBNAME=$(echo "$entry" | sed -n 's/.*"job_name":"\([^"]*\)".*/\1/p')

    # Directory to folder
    CWD="$OUTDIR/${JOBNAME}"

    # Create folder
    mkdir -p "$CWD"

    # Create a JSON file with the entire entry (avoid parsing within loop)
    echo "$entry" > "$CWD/${JOBNAME}.json"

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

echo "Scripts to run experiment 1 generated in $OUTDIR"