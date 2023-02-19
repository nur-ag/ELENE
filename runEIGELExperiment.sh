GNN_TYPE=${1:-GINEConv}

# PROBLEM should be out of "zinc" "pattern" "cifar10" "molhiv" "molpcba" "graph_property" "counting"
PROBLEM=${2:-zinc}

# MAX_MEMORY is the memory threshold after which this script sleeps before submitting new jobs
MAX_MEMORY=${3:-30000}

# MAX_DISTANCES is a list of encoding distances that we will check through
MAX_DISTANCES=${4:-1 2}

# Define the number of 'classic' GNN layers used in the GNN-AK paper
# We will run each experiment with the original configuration, and with half.
declare -A PROBLEM_LAYERS
PROBLEM_LAYERS["exp"]="4"
PROBLEM_LAYERS["zinc"]="6"
PROBLEM_LAYERS["pattern"]="6"
PROBLEM_LAYERS["cifar10"]="4"
PROBLEM_LAYERS["molhiv"]="2"
PROBLEM_LAYERS["molpcba"]="5"
PROBLEM_LAYERS["graph_property"]="6"
PROBLEM_LAYERS["counting"]="6"
PROBLEM_LAYERS["tu_datasets"]="4"

# Define the max. degree per problem
declare -A PROBLEM_DEGREE
PROBLEM_LAYERS["exp"]="12"
PROBLEM_DEGREE["zinc"]="8"
PROBLEM_DEGREE["pattern"]="220"
PROBLEM_DEGREE["cifar10"]="8"
PROBLEM_DEGREE["molhiv"]="18"
PROBLEM_DEGREE["molpcba"]="10"
PROBLEM_DEGREE["graph_property"]="46"
PROBLEM_DEGREE["counting"]="12"
PROBLEM_DEGREE["tu_datasets"]="50" # Using highest -- Enzymes: 18; MUTAG/PTC_MR: 8; PROTEINS: 50

# Get the number of layers and degrees
PROBLEM_KEY=$(echo $PROBLEM | cut -d" " -f1)
MAX_DEGREE=${PROBLEM_DEGREE["$PROBLEM_KEY"]}
GNN_AK_LAYERS=${PROBLEM_LAYERS["$PROBLEM_KEY"]}
HALF_LAYERS=$(($GNN_AK_LAYERS / 2))

# Get the tuples for EIGEL on the first, half or all layes
EIGEL_FIRST="(0,)"
EIGEL_HALF=`seq 0 $(( $HALF_LAYERS - 1 )) | tr '\n' ',' | sed 's/\(.*\),$/(\1)/g'`;
EIGEL_FULL=`seq 0 $(( $GNN_AK_LAYERS - 1 )) | tr '\n' ',' | sed 's/\(.*\),$/(\1)/g'`;

# Run all the jobs as required
for MODEL_TYPE in "joint" "disjoint"
do
  for MAX_DISTANCE in $MAX_DISTANCES
  do
    for LAYER_LAYOUT in "$EIGEL_FIRST" "$EIGEL_HALF" "$EIGEL_FULL"
    do
      for MINI_LAYERS in -1 0
      do
        MINI_LAYER_CFG="model.mini_layers $MINI_LAYERS"
        if [ $MINI_LAYERS -lt 0 ]; then
          MINI_LAYER_CFG=""
        fi

        # Only use GNN _when actually needed, otherwise it's a no-op
        for USE_GNN in "False" "True" 
        do
          LEAVE_GNN_BRANCH="True"
          GNN_MINI_LAYER_CFG="$MINI_LAYER_CFG"
          if [ "$USE_GNN" == "False" ] && [ "$LAYER_LAYOUT" == "$EIGEL_FULL" ]; then
            GNN_MINI_LAYER_CFG="use_gnn False $MINI_LAYER_CFG"
            LEAVE_GNN_BRANCH="False"
          fi

          # Check memory and wait until ready
          CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
          TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
          JOB_COMMAND="python -m train.${PROBLEM} eigel.model_type ${MODEL_TYPE} eigel.max_distance ${MAX_DISTANCE} eigel.max_degree ${MAX_DEGREE} eigel.layer_indices ${LAYER_LAYOUT} ${GNN_MINI_LAYER_CFG}"
          echo "[$(date '+%Y-%M-%d %H:%m')] Found ${CURR_MEMORY} MB out of ${TOTAL_MEMORY} MB."
          while (( $CURR_MEMORY > $MAX_MEMORY )); do
            echo "[$(date '+%Y-%M-%d %H:%m')] Sleeping as ${CURR_MEMORY} MB is greater than ${MAX_MEMORY} MB."
            sleep 15
            CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
            TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
          done
          echo "[$(date '+%Y-%m-%d %H:%M')] Executing: ${JOB_COMMAND}"
          ${JOB_COMMAND} &

          # Sleep after submitting the job to wait until memory gets allocated
          echo "[$(date '+%Y-%m-%d %H:%M')] Sleeping for 300 seconds after submission."
          sleep 300
          # Only run the no-GNN once if we are not in the full EIGEL layout
          if [ "$LEAVE_GNN_BRANCH" == "True" ]; then
            break
          fi
        done
      done
    done
  done
done
