Reproducible Research --- ELENE: Exploring MP-GNN Expressivity via Edge-Level Ego-Network Encodings
===================================================================================================

This document describes how to reproduce the results for our paper "ELENE: Exploring MP-GNN 
Expressivity via Edge-Level Ego-Network Encodings". We provide the following supplementary
materials including code and instructions.

## Reproducibility

All reproducibility work is performed under `./ELENE/`. To reproduce our results:

1. Set up the Conda environment. Follow the installation script `./ELENE/setup.sh`.
  * The script must be executed **line by line**.
  * You must ensure that the Conda environment is created and activated.
  * Upon installation, you should have Pytorch Geometric and all required dependencies installed.
2. Execute the reproducibility scripts. 
  * The Expressivity benchmark is captured by `expressivityDatasets.sh` (Corresponds to Table 1).
  * The Real World Graphs benchmark is captured by `benchmarkDatasets.sh` (Corr. to Tables 2 and 3).
  * The h-Proximity benchmark is captured by `proximityResults.sh` (Corr. to Table 4.).
  * We also include a `TUGINDatasets.sh` to collect results on TU datasets with splits from the GIN paper.
  	* We assume our scripts are run in an environment with NVIDIA GPUs with the `nvidia-smi` command.
  	* If this is not the case, you may modify the `runCommandOnGPUMemThreshold.sh` to manage resources according to your compute / processing capabilities.
3. Hyper-parameters and configuration are all defined in the benchmarking scripts or under ./train/configs/*.yaml
  * Except for the h-Proximity and 3WL Pair (4x4 Rook and Shrikhande) datasets, which were not included in the GNN-AK benchmark, all configurations match the original.
4. Data analysis can be performed ad-hoc using Tensorboard.
  * Execute: `tensorboard serve --logdir results/ --port 7000` to get Tensorboard up and running.
  * To collect the results that we used in the paper, we provide `process_results.py` to collect and aggregate Tensorboard logs.
    * Results will be written to `./tables/` as a series of .csv Pandas dataframes for ease analysis and reporting.
