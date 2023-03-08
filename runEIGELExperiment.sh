GNN_TYPE=${1:-GINEConv}

# PROBLEM should be out of "zinc" "pattern" "cifar10" "molhiv" "molpcba" "graph_property" "counting"
PROBLEM=${2:-zinc}

# MAX_MEMORY is the memory threshold after which this script sleeps before submitting new jobs
MAX_MEMORY=${3:-30000}

# MAX_DISTANCES is a list of encoding distances that we will check through
MAX_DISTANCES=${4:-1 2}

# MAX_DISTANCES is a list of encoding distances that we will check through
MODEL_TYPES=${5:-joint disjoint}

# EXTRA_PARAMS is a list of extra parameters to append to all jobs
EXTRA_PARAMS=${6:-eigel.embedding_dim 32}

# DELAY_BETWEEN_JOB_RUNS is the time in seconds to wait until a successful submission (where the job appears in nvidia-smi)
DELAY_BETWEEN_JOB_RUNS=${7:-60}

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
PROBLEM_LAYERS["counting"]="3"
PROBLEM_LAYERS["tu_datasets"]="4"
PROBLEM_LAYERS["proximity"]="3"

# Define the max. degree per problem
declare -A PROBLEM_DEGREE
PROBLEM_DEGREE["exp"]="12"
PROBLEM_DEGREE["zinc"]="8"
PROBLEM_DEGREE["pattern"]="220"
PROBLEM_DEGREE["cifar10"]="8"
PROBLEM_DEGREE["molhiv"]="18"
PROBLEM_DEGREE["molpcba"]="10"
PROBLEM_DEGREE["graph_property"]="46"
PROBLEM_DEGREE["counting"]="12"
PROBLEM_DEGREE["tu_datasets"]="50" # Using highest -- Enzymes: 18; MUTAG/PTC_MR: 8; PROTEINS: 50
PROBLEM_DEGREE["proximity"]="0" # Ignore degree information

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
for MODEL_TYPE in $MODEL_TYPES
do
  for MAX_DISTANCE in $MAX_DISTANCES
  do
    # Only use half-layers if there is actually a half!
    LAYER_LAYOUTS="$EIGEL_FIRST $EIGEL_HALF $EIGEL_FULL"
    if [ $GNN_AK_LAYERS -le 3 ]; then
        LAYER_LAYOUTS="$EIGEL_FIRST $EIGEL_FULL"
    fi

    for LAYER_LAYOUT in $LAYER_LAYOUTS
    do
      for MINI_LAYERS in -1 0
      do
        MINI_LAYER_CFG="model.mini_layers $MINI_LAYERS"
        if [ $MINI_LAYERS -lt 0 ]; then
          MINI_LAYER_CFG=""
        fi

        EIGEL_USE_GNN="True"
        if [ $MINI_LAYERS -lt 0 ]; then
          EIGEL_USE_GNN="True False"
        fi

        # Only use GNN _when actually needed, otherwise it's a no-op
        for USE_GNN in $EIGEL_USE_GNN
        do
          LEAVE_GNN_BRANCH="True"
          GNN_MINI_LAYER_CFG="$MINI_LAYER_CFG"
          if [ "$USE_GNN" == "False" ] && [ "$LAYER_LAYOUT" == "$EIGEL_FULL" ]; then
            GNN_MINI_LAYER_CFG="use_gnn False $MINI_LAYER_CFG"
            LEAVE_GNN_BRANCH="False"
          fi

          JOB_COMMAND="python -m train.${PROBLEM} model.gnn_type ${GNN_TYPE} eigel.model_type ${MODEL_TYPE} eigel.max_distance ${MAX_DISTANCE} eigel.max_degree ${MAX_DEGREE} eigel.layer_indices ${LAYER_LAYOUT} ${GNN_MINI_LAYER_CFG}"
          ./runCommandOnGPUMemThreshold.sh "${JOB_COMMAND} ${EXTRA_PARAMS}" ${MAX_MEMORY}

          # Sleep after submitting the job to wait until memory gets allocated
          echo "[$(date '+%Y-%m-%d %H:%M')] Sleeping for ${DELAY_BETWEEN_JOB_RUNS} seconds after submission."
          sleep ${DELAY_BETWEEN_JOB_RUNS}
          # Only run the no-GNN once if we are not in the full EIGEL layout
          if [ "$LEAVE_GNN_BRANCH" == "True" ]; then
            break
          fi
        done
      done
    done
  done
done
