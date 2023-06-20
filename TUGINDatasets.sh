# These datasets are small and noisy --- we do not report results on the paper
# as for actual reporting we would need to tune parameters to avoid overfitting.
#
# Evaluating on a (non-standard) averaged best performance across folds shows:
# - ELENE-L performance increases with the size of the ego-network, k
# - Introducing degree and distance signals on each layer is slightly more expressive

# Execute all experiments on TU datasets in the splits from GIN
# Smaller datasets have lower memory bounds
for DATASET in MUTAG PTC PROTEINS NCI1; do
  echo "[$(date '+%Y-%m-%d %H:%M')] Running $DATASET with ELENE."
  ./runELENEExperimentSmall.sh GINEConv tu_datasets_gin_split 32000 "1 2 3" "joint-nodeonly" "0" "dataset $DATASET elene.embedding_dim 32 subgraph.hops 3"
done

# Find baselines for all datasets
for DATASET in MUTAG PTC PROTEINS NCI1; do
  echo "[$(date '+%Y-%m-%d %H:%M')] Running $DATASET with IGEL."
  ./runIGELExperiment.sh GINEConv tu_datasets_gin_split 20000 "0 1 2" "0" "dataset $DATASET"
done
