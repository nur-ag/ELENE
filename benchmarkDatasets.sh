# Execute all experiments on benchmark datasets
# Run ZINC experiments with 32-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running ZINC."
./runIGELExperiment.sh GINEConv zinc 32000 "0 1 2"
./runEIGELExperiment.sh GINEConv zinc 32000 "1 2" "joint disjoint" "eigel.embedding_dim 32"

# Run MolHIV experiments with 32-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running MolHIV."
./runIGELExperiment.sh GINEConv molhiv 32000 "0 1 2"
./runEIGELExperiment.sh GINEConv molhiv 32000 "1 2" "joint disjoint" "eigel.embedding_dim 32"

# Run PATTERN experiments with 16-dim embeddings (no hops, using random walks)
# We can only run one job at a time since PATTERN uses up to 40GB of memory
echo "[$(date '+%Y-%m-%d %H:%M')] Running PATTERN."
./runIGELExperiment.sh GINEConv pattern 600 "0 1 2"
./runEIGELExperiment.sh GINEConv pattern 600 "1 2" "joint disjoint" "eigel.embedding_dim 16"

# Run MolPCBA experiments with 32-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running MolPCBA."
./runIGELExperiment.sh GINEConv molpcba 18000 "0 1 2"
./runEIGELExperimentSmall.sh GINEConv molpcba 18000 "1 2" "joint disjoint" "eigel.embedding_dim 32"

# Run CIFAR experiments with 16-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running CIFAR."
./runIGELExperiment.sh GINEConv cifar10 15000 "0 1 2"
./runEIGELExperimentSmall.sh GINEConv cifar10 15000 "1 2" "joint disjoint" "eigel.embedding_dim 16"

