# Execute all experiments on benchmark datasets
# Run ZINC experiments with 32-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running ZINC."
./runSparseELENEExperiment.sh GINEConv zinc 32000 "0 1 2" "0 -1"
./runELENELExperiment.sh GINEConv zinc 32000 "1 2 3" "joint disjoint" "-1 0" "elene.embedding_dim 32"
./runELENELExperimentSmall.sh GINEConv zinc 32000 "1 2 3" "joint-nodeonly disjoint-nodeonly" "-1 0" "elene.embedding_dim 32"

# Run MolHIV experiments with 32-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running MolHIV."
./runSparseELENEExperiment.sh GINEConv molhiv 32000 "0 1 2" "0 -1"
./runELENELExperiment.sh GINEConv molhiv 32000 "1 2" "joint disjoint" "-1 0" "elene.embedding_dim 32"
./runELENELExperimentSmall.sh GINEConv molhiv 32000 "1 2 3" "joint-nodeonly disjoint-nodeonly" "-1 0" "elene.embedding_dim 32"

# Run PATTERN experiments with 128-dim embeddings (no hops, using random walks)
# We can only run one job at a time since PATTERN uses up to 40GB of memory
echo "[$(date '+%Y-%m-%d %H:%M')] Running PATTERN."
./runSparseELENEExperiment.sh GINEConv pattern 400 "0 1 2" "0"
./runELENELExperimentSmall.sh GINEConv pattern 20000 "1 2 3" "joint-nodeonly" "0" "elene.embedding_dim 64 elene.max_degree 0 elene.reuse_embeddings all"
./runELENELExperimentSmall.sh GINEConv pattern 20000 "1 2 3" "joint-nodeonly" "0" "elene.embedding_dim 64"

# Run CIFAR experiments with 64-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running CIFAR."
./runSparseELENEExperiment.sh GINEConv cifar10 15000 "0 2" "0"
./runELENELExperimentSmall.sh GINEConv cifar10 15000 "1 2" "joint-nodeonly" "0" "elene.embedding_dim 64 elene.max_degree 0"

# Run MolPCBA experiments with 64-dim embeddings
echo "[$(date '+%Y-%m-%d %H:%M')] Running MolPCBA."
# We started from the largest value of k=3 due to computational constraints
./runELENELExperimentSmall.sh GINEConv molpcba 28000 "3 2 1" "joint-nodeonly" "0" "elene.embedding_dim 64"
