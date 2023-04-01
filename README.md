# Simple yet Powerful MP-GNNs from Explicit Ego-Network Attributes

Based on the Official code for [**GNN-As-Kernel**](https://github.com/LingxiaoShawn/GNNAsKernel). 

We introduce three explicit ego-network attribute encodings --- IGEL, EIGEL and EIGEL-L. This code is sufficient to reproduce all the results reported in our paper. Using our configurations and scripts, you may also reproduce results from the original GNN-AK.

## Setup 

This section follows the installation guide for GNN-As-Kernel, as we do not introduce new dependencies for consistency.

```
# params
ENV=eigel

# create env 
conda create --name $ENV python=3.10 -y
conda activate $ENV

# install pytorch 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install pyg
conda install pyg -c pyg

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

```

## Code structure

We introduce IGEL, EIGEL and EIGEL-L in the GNN-AK codebase. See:
* `core/eigel.py` — contains the implementation of the EIGEL-L representation.
* `core/igel_utils.py` — contains the implementation of IGEL and EIGEL as a the relative degree encoding extension.
  * For IGEL, the IGEL repository is required.

The necessary sub-graph information required for EIGEL-L is introduced in `SubgraphsTransform` under `core/transform.py`.

## Hyperparameters 

See ``core/config.py`` for all the extended IGEL / EIGEL / EIGEL-L options.

## Reproducibility

We provide several bash scripts to reproduce the results of each benchmark. The results reported in our paper are computed using them.
See: `expressivityDatasets.sh`, `benchmarkDatasets.sh` and `proximityResults.sh`.

To manage resources in our research cluster, we wrapped our execution scripts to detect the available GPU memory in our system.
See `runEIGELExperiment.sh`, `runEIGELExperimentSmall.sh` and `runIGELExperiment.sh` for our hyper-parameter tuning approach.

Note that in our scripts, EIGEL refers to EIGEL-L — that is, the learnable variant of explicit ego-network attributes.
