# ELENE: Exploring MP-GNN Expressivity via Edge-Level Ego-Network Encodings

Based on the Official code for [**GNN-As-Kernel**](https://github.com/LingxiaoShawn/GNNAsKernel) for evaluation consistency. 

We introduce ELENE (edge-level ego-network encodings) with the two variants presented in the paper --- ELENE and ELENE-L. 

This code is sufficient to reproduce all the results reported in our paper. Using our configurations 
and scripts, you may also reproduce results from the original GNN-AK.

We provide an overview of our effort to keep [this research reproducible in REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

For the reproducibility checklist, consult [our annotated PDF](./ELENE-ReproducibilityChecklist-v2.0.pdf)

## Setup

This section follows the installation guide for GNN-As-Kernel, as we do not introduce new dependencies for consistency.

```
# environment name
export ENV=elene

# create env 
conda create --name $ENV python=3.10 -y
conda activate $ENV

# install cuda according to your system --- or ignore for CPU-only
conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit

# install pytorch 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -y

# install pyg
conda install pyg -c pyg -y

# install ogb 
pip install ogb

# install rdkit
conda install -c conda-forge rdkit -y

# update yacs and tensorboard
pip install yacs==0.1.8 --force  # PyG currently use 0.1.6 which doesn't support None argument. 
pip install tensorboard
pip install matplotlib

# install jupyter and ipython 
conda install -c conda-forge nb_conda -y

# clone the IGEL dependency
git clone git@github.com:nur-ag/IGEL.git ../IGEL

# install igraph for IGEL and missing networkx
conda install -c conda-forge python-igraph -y
pip install networkx dill

# install torch-scatter, torch-cluster and torch-sparse
# note: this is hard to automate, and requires finding the appropriate version for your system
# using conda makes it easier, but your mileage may vary
conda install pytorch-scatter -c pyg -y
conda install pytorch-sparse -c pyg -y
conda install pytorch-cluster -c pyg -y

```

Note that we provide a `setup.sh` for convenience. However, we recommend that this script be executed manually line by line to ensure that each installation step executes correctly.

## Code structure

We introduce ELENE-L and ELENE in the GNN-AK codebase. See:
* `core/elene.py` — contains the implementation of the ELENE-L representation.
* `core/igel_utils.py` — contains the implementation of IGEL and ELENE as the relative degree encoding extension.
  * For IGEL, the IGEL repository is required and pulled by the setup script.

The necessary sub-graph information required for ELENE-L is introduced in `SubgraphsTransform` under `core/transform.py`.

## Hyperparameters

See ``core/config.py`` for all the extended ELENE / ELENE-L options.

## Reproducibility

For reproducibility details, see [REPRODUCIBILITY.md](./REPRODUCIBILITY.md). We provide several bash scripts to reproduce the results of each benchmark. The results reported in our paper are computed using them.
See: `expressivityDatasets.sh`, `benchmarkDatasets.sh` and `proximityResults.sh`.

After installation, one quick check is to reproduce the results on the pair of Shrikhande and 4x4 Rook graphs, which can be validated using:

```bash
# GINE without ELENE --- Should _not_ be able to distinguish both graphs (accuracy: 0.5 for all epochs)
python3 -m train.pair3wl model.gnn_type GINEConv model.mini_layers 0 igel.distance 0 elene.max_distance 0 elene.model_type joint elene.max_degree 0 elene.embedding_dim 32 elene.layer_indices \(0,\) model.num_layers 2 model.hidden_size 32

# GINE with ELENE (k = 1, rho = 6 (max. degree)) --- Should be able to distinguish both graphs (best acc: 0 or 1 in some epoch, meaning we identify 2 classes)
python3 -m train.pair3wl model.gnn_type GINEConv model.mini_layers 0 igel.distance 0 elene.max_distance 1 elene.model_type joint elene.max_degree 6 elene.embedding_dim 32 elene.layer_indices \(0,\) model.num_layers 2 model.hidden_size 32
```

You can collect results as reported in the paper with the `process_results.py`, which parses Tensorboard logs.

To manage resources in our research cluster, we wrapped our execution scripts to detect the available GPU memory in our system.
See `runELENELExperiment.sh`, `runELENELExperimentSmall.sh` and `runSparseELENEExperiment.sh` for our hyper-parameter tuning approach.

Note that in some of our scripts, ELENE parameters may refer to ELENE-L for brevity — that is, they refer to the learnable variant of ELENE.
