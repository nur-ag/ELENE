# Execute all experiments on the K-Proximity datasets
for TASK_ID in 1 3 5 8 10; do
  echo "[$(date '+%Y-%m-%d %H:%M')] Running ${TASK_ID}-Proximity."
  ./runIGELExperiment.sh GINEConv proximity 18000 "0 2" "igel.use_edge_encodings True task $TASK_ID"
  for DISTANCE in 1 3 5; do
    ./runEIGELExperimentSmall.sh GINEConv proximity 18000 "$DISTANCE" "joint-nodeonly" "subgraph.hops $DISTANCE eigel.embedding_dim 32 task $TASK_ID"
  done
done
