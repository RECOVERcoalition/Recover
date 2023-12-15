# Machine Learning Driven Candidate Compound Generation for Drug Repurposing
Based on RECOVER: sequential model optimization platform for combination drug repurposing identifies novel synergistic compounds *in vitro*
[![DOI](https://zenodo.org/badge/320327566.svg)](https://zenodo.org/badge/latestdoi/320327566)

This repository is an implementation of RECOVER, a platform that can guide wet lab experiments to quickly discover synergistic drug combinations,
([preprint](https://arxiv.org/abs/2202.04202)), howerver instead of using an ensemble model to get Synergy predictions with uncertainty, we used multiple realization of a Bayesian Neural Network model. 
Since the weights are drawn from a distribution, they differ for every run of a trained model and hence give different results. The goal was to get a more precise uncertainty and achieve i quicker since the model doesn't have to be trained multiple times. 

## Bayesian Before and After Merge
This branch is refering to a model using Bayesian modeling in both single drug MLP and Combination MLP. The predictors with all Bayesian layers are added in `Recover/recover/models/predictors.py`. The `train.py` was updated with a train_epoch_bayesian function that trains the model using KL loss and test_epoch for testing of the model. In Bayesian Basic Trainer test_epoch is used to test the trained model and easily get the mean and the standard deviation of synergy predictions.   
In Bayesian Active Trainer, realizations of the trained model are used instead of the Ensemble Model to get the acqusition function scores and rank the drug combinations. Probability of Improvement and Expected Improvement acquistion functions were added to `Recover/recover/acquisition/acquisition.py` since we are now working with Bayesian Optimization.  
Config files are also updated to be use BNNs. In this branch there are separate bayesian config files, while in the **master** use of bayesian layers feature was added to existing config files.

## Environment setup

**Requirements and Installation**: 
For all the requirements and installation steps check th orginal RECOVER repository (https://github.com/RECOVERcoalition/Recover.git). 

## Running the pipeline

Configuration files for our experiments are provided in the following directory: `Recover/recover/config`

To run the pipeline with a custom configuration:
- Create your configuration file and move it to `Recover/recover/config/`
- Run `python train.py --config <my_configuration_file>`

For example, to run the pipeline with configuration from 
the file `model_evaluation.py`, run `python train.py --config model_evaluation`.

Log files will automatically be created to save the results of the experiments.
