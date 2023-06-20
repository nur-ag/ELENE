# Execute all experiments on the K-Proximity datasets (from hardest to easiest)
for TASK_ID in 10 8 5 3; do
  echo "[$(date '+%Y-%m-%d %H:%M')] Running ${TASK_ID}-Proximity on ELENE."
  for DISTANCE in 1 3 5; do
    echo "Processing Dist=$DISTANCE ELENE-L."
    ./runELENEExperimentSmall.sh GINEConv proximity 27000 "$DISTANCE" "joint-nodeonly" "0" "subgraph.hops $DISTANCE elene.embedding_dim 32 task $TASK_ID"
  done
done

