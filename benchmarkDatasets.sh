# Execute all experiments on benchmark datasets
# Run ZINC experiments with 32-dim embeddings
./runEIGELExperiment.sh GINEConv zinc 32000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv zinc 32000 "0 1 2"

# Run MolHIV experiments with 32-dim embeddings
./runEIGELExperiment.sh GINEConv molhiv 32000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv molhiv 32000 "0 1 2"

# Run MolPCBA experiments with 32-dim embeddings
./runEIGELExperiment.sh GINEConv molpcba 18000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv molpcba 18000 "0 1 2"

# Run PATTERN experiments with 32-dim embeddings (only 2 hops, no walks)
./runEIGELExperiment.sh GINEConv pattern 16000 "1 2" "joint disjoint" "eigel.embedding_dim 32 subgraph.hops 2 subgraph.walk_length 0"
./runIGELExperiment.sh GINEConv pattern 16000 "0 1 2"

# Run CIFAR experiments with 16-dim embeddings
./runEIGELExperiment.sh GINEConv cifar10 15000 "1 2" "joint disjoint" "eigel.embedding_dim 16"
./runIGELExperiment.sh GINEConv cifar10 15000 "0 1 2"

