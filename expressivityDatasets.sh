# Execute all experiments on expressivity datasets
# Run all EXP experiments in 50 epochs
echo "[$(date '+%Y-%m-%d %H:%M')] Running EXP."
./runSparseELENEExperiment.sh GINEConv exp 32000 "0 1 2" "0 -1" "igel.use_edge_encodings True train.epochs 50" "15"
./runELENELExperiment.sh GINEConv exp 32000 "1 2" "joint disjoint" "-1 0" "elene.embedding_dim 8 train.epochs 50" "15"
./runELENELExperiment.sh GINEConv exp 32000 "1 2" "joint-nodeonly disjoint-nodeonly" "-1 0" "elene.embedding_dim 8 train.epochs 50" "15"

# Run on SR25 -- we know this will not improve with node-level information alone, as the graphs are SRGs
echo "[$(date '+%Y-%m-%d %H:%M')] Running SR25."
./runSparseELENEExperiment.sh GINEConv sr25 32000 "0 1 2" "0 -1" "igel.use_edge_encodings True" "15"
./runELENELExperiment.sh GINEConv sr25 35000 "1 2" "joint disjoint joint-nodeonly disjoint-nodeonly" "-1 0" "elene.embedding_dim 32 subgraph.hops 2" "15"

# Test SR25 with more layers and deeper MLPs -- Disjoint ELENE-L should be able to match Joint with a 3-layer MLP
./runELENELExperimentSmall.sh GINEConv sr25 35000 "1 2" "joint disjoint joint-nodeonly disjoint-nodeonly" "0" "elene.embedding_dim 32 subgraph.hops 2 model.num_layers 3" "15"
./runELENELExperimentSmall.sh GINEConv sr25 35000 "1 2" "joint disjoint joint-nodeonly disjoint-nodeonly" "0" "elene.embedding_dim 32 subgraph.hops 2 model.num_layers 3 model.mlp_layers 3" "15"

# Run Graphlet Counting experiments
for TASK_ID in `seq 0 4`; do
  echo "[$(date '+%Y-%m-%d %H:%M')] Running Counting on task ${TASK_ID}."
  ./runSparseELENEExperiment.sh GINEConv counting 28000 "0 1 2" "0 -1" "igel.use_edge_encodings True task $TASK_ID" "30"
  ./runELENELExperiment.sh GINEConv counting 28000 "1 2" "joint disjoint" "-1 0" "elene.embedding_dim 16 task $TASK_ID" "30"
  ./runELENELExperiment.sh GINEConv counting 34000 "1 2" "joint-nodeonly disjoint-nodeonly" "-1 0" "elene.embedding_dim 16 task $TASK_ID" "30"
done

# Run Property prediction experiments
for TASK_ID in `seq 0 2`; do
   echo "[$(date '+%Y-%m-%d %H:%M')] Running Property on task ${TASK_ID}."
  ./runSparseELENEExperiment.sh GINEConv graph_property 17000 "0 1 2" "0 -1" "igel.use_edge_encodings True task $TASK_ID" "30"
  ./runELENELExperimentSmall.sh GINEConv graph_property 17000 "1 2" "joint disjoint" "-1 0" "elene.embedding_dim 16 task $TASK_ID" "60"
  ./runELENELExperimentSmall.sh GINEConv graph_property 25000 "1 2" "joint-nodeonly disjoint-nodeonly" "-1 0" "elene.embedding_dim 16 task $TASK_ID" "60"
  ./runELENELExperimentSmall.sh GINEConv graph_property 25000 "3" "joint-nodeonly" "0" "elene.embedding_dim 16 task $TASK_ID" "60"
done

