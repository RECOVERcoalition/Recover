conda create --name recover
conda activate recover

# Install pip
conda install -c anaconda pip

# Install requirements
pip install -r requirements.txt

# Install recover
pip install -e .


# Install rdkit
conda install -c rdkit rdkit


# Install pytorch geometric
TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)")
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric


# Get Reservoir and install it in the environment (Git LFS NEEDED)
git clone git@github.com:RECOVERcoalition/Reservoir.git
cd Reservoir/
python setup.py develop
cd ../

# Deactivate environment
conda deactivate