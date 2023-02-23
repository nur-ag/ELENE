GNN_TYPE=${1:-GINEConv}

# PROBLEM should be out of "exp" "zinc" "pattern" "cifar10" "molhiv" "molpcba" "graph_property" "counting"
PROBLEM=${2:-zinc}

# MAX_PARALLEL is the maximum number of parallel jobs fitting in the GPU
MAX_MEMORY=${3:-30000}

# IGEL_DISTANCES is a list of encoding distances that we will check through
IGEL_DISTANCES=${4:-0 1 2}

# EXTRA_PARAMS is a list of extra parameters to append to all jobs
EXTRA_PARAMS=${5:-igel.use_edge_encodings True}

# MAX_MEMORY is the memory threshold after which this script sleeps before submitting new jobs
DELAY_BETWEEN_JOB_RUNS=120

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

# We run a version of the problem with half of the layers
PROBLEM_KEY=$(echo $PROBLEM | cut -d" " -f1)
GNN_AK_LAYERS=${PROBLEM_LAYERS["$PROBLEM_KEY"]}

for IGEL_DISTANCE in $IGEL_DISTANCES
do
  IGEL_REL_DEGREES="False"
  if [ $IGEL_DISTANCE -gt 0 ]; then
    IGEL_REL_DEGREES="False True"
  fi

  for REL_DEGREE in $IGEL_REL_DEGREES
  do
    for MINI_LAYERS in -1 0
    do
      MINI_LAYER_CFG="model.mini_layers $MINI_LAYERS"
      if [ $MINI_LAYERS -lt 0 ]; then
        MINI_LAYER_CFG=""
      fi

      # Check memory and wait until ready
      CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
      TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
      JOB_COMMAND="python -m train.${PROBLEM} model.num_layers ${GNN_AK_LAYERS} model.gnn_type ${GNN_TYPE} igel.distance $IGEL_DISTANCE igel.use_relative_degrees $REL_DEGREE $MINI_LAYER_CFG"
      echo "[$(date '+%Y-%m-%d %H:%M')] Found ${CURR_MEMORY} MB out of ${TOTAL_MEMORY} MB."
      while (( $CURR_MEMORY > $MAX_MEMORY )); do
        echo "[$(date '+%Y-%m-%d %H:%M')] Sleeping as ${CURR_MEMORY} MB is greater than ${MAX_MEMORY} MB."
        sleep 15
        CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
        TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
      done
      echo "[$(date '+%Y-%m-%d %H:%M')] Executing: ${JOB_COMMAND} ${EXTRA_PARAMS}"
      ${JOB_COMMAND} ${EXTRA_PARAMS} &

      # Sleep after submitting the job to wait until memory gets allocated
      echo "[$(date '+%Y-%m-%d %H:%M')] Sleeping for ${DELAY_BETWEEN_JOB_RUNS} seconds after submission."
      sleep ${DELAY_BETWEEN_JOB_RUNS}
    done
  done
done
