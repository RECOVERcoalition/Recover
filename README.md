# RECOVER: sequential model optimization platform for combination drug repurposing identifies novel synergistic compounds *in vitro*

RECOVER is a platform that can guide wet lab experiments to quickly discover synergistic drug combinations active 
against a cancer cell line, requiring substantially less screening than an exhaustive evaluation 
([preprint](https://arxiv.org/abs/2202.04202)).


![Overview](docs/images/overview.png "Overview")

## Environment setup

**Requirements**: Anaconda (https://www.anaconda.com/) and Git LFS (https://git-lfs.github.com/). Please make sure 
both are installed on the system prior to running installation.

**Installation**: enter the command `source install.sh` and follow the instructions. This will create a conda 
environment named **recover** and install all the required package including the 
[reservoir](https://github.com/RECOVERcoalition/Reservoir) package that stores the primary data acquisition scripts.

## Running the pipeline

Configuration files for our experiments are provided in the following directory: `Recover/recover/config`

To run the pipeline with a custom configuration:
- Create your configuration file and move it to `Recover/recover/config/`
- Run `python train.py --config <my_configuration_file>`

Note that `<my_configuration_file>` should not include *.py*. For example, to run the pipeline with configuration from 
the file `model_evaluation.py`, run `python train.py --config model_evaluation`


## Note

This Recover repository is based on research funded by (or in part by) the Bill & Melinda Gates Foundation. The 
findings and conclusions contained within are those of the authors and do not necessarily reflect positions or policies 
of the Bill & Melinda Gates Foundation.