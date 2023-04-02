# environment name
export ENV=eigel

# create env 
conda create --name $ENV python=3.10 -y
conda activate $ENV

# install pytorch 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

# install pyg
conda install pyg -c pyg -y

# install ogb 
pip install ogb -y

# install rdkit
conda install -c conda-forge rdkit -y

# update yacs and tensorboard
pip install yacs==0.1.8 --force  # PyG currently use 0.1.6 which doesn't support None argument. 
pip install tensorboard
pip install matplotlib

# install jupyter and ipython 
conda install -c conda-forge nb_conda -y

# install igraph for IGEL and missing networkx
conda install -c conda-forge python-igraph -y
pip install networkx dill

# install torch-scatter, torch-cluster and torch-sparse
# note: this is hard to automate, and requires finding the appropriate version for your system
# using conda makes it easier, but your mileage may vary
conda install pytorch-scatter -c pyg -y
conda install pytorch-sparse -c pyg -y
conda install pytorch-cluster -c pyg -y
