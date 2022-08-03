conda create --name recover python=3.6
conda activate recover

# Install pip
conda install -c anaconda pip

# Install requirements
pip install -r requirements.txt

# Install ray
pip install -U "ray[tune]"

# Install recover
pip install -e .


# Install rdkit
conda install -c rdkit rdkit

# Get Recover-Data-Lake and install it in the environment (Git LFS NEEDED)
git clone git@github.com:RECOVERcoalition/Recover-Data-Lake.git
cd Recover-Data-Lake/
python setup.py develop
cd ../

# Deactivate environment
conda deactivate
