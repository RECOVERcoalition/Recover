conda create --name recover python=3.9
conda activate recover

# Install pip
conda install -c anaconda pip

# Install requirements
pip install -r requirements.txt

# Install pytorch to support GPU on mac m1
# Expect the M1-GPU support to be included in the next stable release. For the time being, it only is found in the Nightly release.
conda install pytorch -c pytorch-nightly

# Install ray
# Firts make sure grpcio is only installed with conda
pip uninstall grpcio; conda install grpcio=1.43.0
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
