conda create --name recover python=3.6
conda activate recover

# Install pip
conda install -c anaconda pip

# Install requirements
pip install -r requirements.txt

# Install pytorch with conda
conda install -c pytorch pytorch==1.10.2

# Install ray
pip install -U "ray[tune]"

# Install recover
pip install -e .

# Install rdkit
conda install -c rdkit rdkit

# Get Reservoir and install it in the environment (Git LFS NEEDED)
git clone git@github.com:RECOVERcoalition/Reservoir.git
cd Reservoir/
python setup.py develop
cd ../

# Deactivate environment
conda deactivate
