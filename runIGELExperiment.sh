GNN_TYPE=${1:-GINEConv}

# PROBLEM should be out of "zinc" "pattern" "cifar10" "molhiv" "molpcba" "graph_property" "counting"
PROBLEM=${2:-zinc}

# LAYER_SCALE is the fraction for the number of layers, defaults to 1
LAYER_SCALE=${3:-1}

# MAX_PARALLEL is the maximum number of parallel jobs fitting in the GPU
MAX_PARALLEL=${4:-5}

# IGEL_DISTANCES is a list of encoding distances that we will check through
IGEL_DISTANCES=${5:-0 1 2}

# Define the number of 'classic' GNN layers used in the GNN-AK paper
# We will run each experiment with the original configuration, and with half.
declare -A PROBLEM_LAYERS
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
EXP_LAYERS=$(($GNN_AK_LAYERS / $LAYER_SCALE))
NUM_PARALLEL=0
for USE_EDGE_ENCODING in "True"
do
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

        python -m train.${PROBLEM} igel.use_edge_encodings ${USE_EDGE_ENCODING} model.num_layers ${EXP_LAYERS} model.gnn_type ${GNN_TYPE} igel.distance $IGEL_DISTANCE igel.use_relative_degrees $REL_DEGREE $MINI_LAYER_CFG &
        NUM_PARALLEL=$(($NUM_PARALLEL + 1))
        if [ $NUM_PARALLEL -ge $MAX_PARALLEL ]; then
          wait
          NUM_PARALLEL=0
        fi
      done
    done
  done
done
