# Execute all experiments on benchmark datasets
# Run ZINC experiments with 32-dim embeddings
./runEIGELExperiment.sh GINEConv zinc 32000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv zinc 32000 "0 1 2"

# Run MolHIV experiments with 32-dim embeddings
./runEIGELExperiment.sh GINEConv molhiv 35000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv molhiv 35000 "0 1 2"

# Run MolPCBA experiments with 32-dim embeddings
./runEIGELExperiment.sh GINEConv molpcba 25000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv molpcba 25000 "0 1 2"

# Run PATTERN experiments with 32-dim embeddings
./runEIGELExperiment.sh GINEConv pattern 16000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv pattern 16000 "0 1 2"

# Run CIFAR experiments with 16-dim embeddings
./runEIGELExperiment.sh GINEConv cifar10 15000 "1 2" "joint disjoint" "eigel.embedding_dim 32"
./runIGELExperiment.sh GINEConv cifar10 15000 "0 1 2"

