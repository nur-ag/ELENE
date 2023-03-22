GNN_TYPE=${1:-GINEConv}

# PROBLEM should be out of "zinc" "pattern" "cifar10" "molhiv" "molpcba" "graph_property" "counting"
PROBLEM=${2:-zinc}

# MAX_MEMORY is the memory threshold after which this script sleeps before submitting new jobs
MAX_MEMORY=${3:-30000}

# MAX_DISTANCES is a list of encoding distances that we will check through
MAX_DISTANCES=${4:-1 2}

# MODEL_TYPES is the set of EIGEL model types that we check
MODEL_TYPES=${5:-joint disjoint}

# MINI_SETUPS are the mini-layer configurations for GNN-AK, where -1 just uses the default
MINI_SETUPS=${6:-0 -1}

# EXTRA_PARAMS is a list of extra parameters to append to all jobs
EXTRA_PARAMS=${7:-eigel.embedding_dim 32}

# DELAY_BETWEEN_JOB_RUNS is the time in seconds to wait until a successful submission (where the job appears in nvidia-smi)
DELAY_BETWEEN_JOB_RUNS=${8:-60}

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
PROBLEM_LAYERS["tu_datasets_gin_split"]="4"
PROBLEM_LAYERS["proximity"]="3"
PROBLEM_LAYERS["sr25"]="2"

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
PROBLEM_DEGREE["tu_datasets_gin_split"]="50" # Using highest -- Enzymes: 18; MUTAG/PTC_MR: 8; PROTEINS: 50
PROBLEM_DEGREE["proximity"]="0" # Ignore degree information
PROBLEM_DEGREE["sr25"]="25" # Upper bound degree by total number of nodes

# Get the number of layers and degrees
PROBLEM_KEY=$(echo $PROBLEM | cut -d" " -f1)
MAX_DEGREE=${PROBLEM_DEGREE["$PROBLEM_KEY"]}
GNN_AK_LAYERS=${PROBLEM_LAYERS["$PROBLEM_KEY"]}

# Get the tuples for EIGEL on the first, half or all layes
EIGEL_FIRST="(0,)"
EIGEL_FULL=`seq 0 $(( $GNN_AK_LAYERS - 1 )) | tr '\n' ',' | sed 's/\(.*\),$/(\1)/g'`;

# Run all the jobs as required
for MODEL_TYPE in $MODEL_TYPES
do
  for MAX_DISTANCE in $MAX_DISTANCES
  do
    LAYER_LAYOUTS="$EIGEL_FIRST $EIGEL_FULL"

    for LAYER_LAYOUT in $LAYER_LAYOUTS
    do
      for MINI_LAYERS in $MINI_SETUPS
      do
        MINI_LAYER_CFG="model.mini_layers $MINI_LAYERS"
        if [ $MINI_LAYERS -lt 0 ]; then
          MINI_LAYER_CFG=""
        fi

        DEGREE_ARGS="eigel.max_degree ${MAX_DEGREE}"
        if [ $(echo "$EXTRA_ARGS" | grep "eigel.max_degree" | wc -l) -gt 0 ]; then
          DEGREE_ARGS=""
        fi
        JOB_COMMAND="python -m train.${PROBLEM} model.gnn_type ${GNN_TYPE} eigel.model_type ${MODEL_TYPE} eigel.max_distance ${MAX_DISTANCE} ${DEGREE_ARGS} eigel.layer_indices ${LAYER_LAYOUT} ${MINI_LAYER_CFG}"
        ./runCommandOnGPUMemThreshold.sh "${JOB_COMMAND} ${EXTRA_PARAMS}" ${MAX_MEMORY}

        # Sleep after submitting the job to wait until memory gets allocated
        echo "[$(date '+%Y-%m-%d %H:%M')] Sleeping for ${DELAY_BETWEEN_JOB_RUNS} seconds after submission."
        sleep ${DELAY_BETWEEN_JOB_RUNS}
      done
    done
  done
done
