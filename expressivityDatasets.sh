# Execute all experiments on expressivity datasets
# Run all EXP experiments in 20 epochs
./runEIGELExperiment.sh GINEConv exp 32000 "1 2" "joint disjoint" "eigel.embedding_dim 8 train.epochs 20"
./runIGELExperiment.sh GINEConv exp 32000 "0 1 2" "igel.use_edge_encodings True train.epochs 20"

# Run Graphlet Counting experiments
for TASK_ID in `seq 0 4`; do
  ./runEIGELExperiment.sh GINEConv counting 22000 "1 2" "joint disjoint" "eigel.embedding_dim 32 task $TASK_ID"
  ./runIGELExperiment.sh GINEConv counting 22000 "0 1 2" "task $TASK_ID"
done

# Run Property prediction experiments
for TASK_ID in `seq 0 2`; do
  ./runEIGELExperiment.sh GINEConv graph_property 22000 "1 2" "joint disjoint" "eigel.embedding_dim 32 task $TASK_ID"
  ./runIGELExperiment.sh GINEConv graph_property 22000 "0 1 2" "task $TASK_ID"
done

