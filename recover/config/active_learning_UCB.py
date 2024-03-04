from recover.datasets.drugcomb_matrix_data import DrugCombMatrix
from recover.models.models import Baseline, EnsembleModel
from recover.models.predictors import BilinearFilmMLPPredictor, \
    BilinearMLPPredictor, BilinearFilmWithFeatMLPPredictor
from recover.utils.utils import get_project_root
from recover.acquisition.acquisition import RandomAcquisition, GreedyAcquisition, UCB, ProbabilityOfImprovementAcquisition, ExpectedImprovementAcquisition
from recover.train import train_epoch, eval_epoch, BasicTrainer, ActiveTrainer
import os
from ray import tune

########################################################################################################################
# Configuration
########################################################################################################################


pipeline_config = {
    "use_tune": True,
    "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    "seed": tune.grid_search([1]),
    # Optimizer config
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "batch_size": 128,
    # Train epoch and eval_epoch to use
    "train_epoch": train_epoch,
    "eval_epoch": eval_epoch,
}

predictor_config = {
    "predictor": BilinearMLPPredictor,
    "bayesian_predictor": False,
    "bayesian_before_merge": False, # For bayesian predictor implementation - Layers after merge are bayesian by default
    "sigmoid": False,
    "num_realizations": 0, # For bayesian uncertainty
    "predictor_layers":
        [
            2048,
            128,
            64,
            1,
        ],
    "merge_n_layers_before_the_end": 2,  # Computation on the sum of the two drug embeddings for the last n layers
    "allow_neg_eigval": True,
}

model_config = {
    "model": EnsembleModel,
    # Loading pretrained model
    "load_model_weights": False,  # tune.grid_search([True, False]),
    "model_weights_file": "",
}

"""
List of cell line names:

['786-0', 'A498', 'A549', 'ACHN', 'BT-549', 'CAKI-1', 'EKVX', 'HCT-15', 'HCT116', 'HOP-62', 'HOP-92', 'HS 578T', 'HT29',
 'IGROV1', 'K-562', 'KM12', 'LOX IMVI', 'MALME-3M', 'MCF7', 'MDA-MB-231', 'MDA-MB-468', 'NCI-H226', 'NCI-H460', 
 'NCI-H522', 'NCIH23', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'OVCAR3', 'PC-3', 'RPMI-8226', 'SF-268', 'SF-295', 'SF-539', 
 'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'SK-OV-3', 'SNB-75', 'SR', 'SW-620', 'T-47D', 'U251', 'UACC-257', 'UACC62', 
 'UO-31']
"""

dataset_config = {
    "dataset": DrugCombMatrix,
    "study_name": 'ALMANAC',
    "in_house_data": 'without',
    "rounds_to_include": [],
    "cell_line": 'MCF7',  # Restrict to a specific cell line
    "val_set_prop": 0.1,
    "test_set_prop": 0.,
    "test_on_unseen_cell_line": False,
    "split_valid_train": "pair_level",  # either "cell_line_level" or "pair_level"
    "cell_lines_in_test": None,  # ['MCF7', 'PC-3'],
    "target": "bliss_max",
    "fp_bits": 1024,
    "fp_radius": 2
}

active_learning_config = {
    "ensemble_size": 5,
    "acquisition": tune.grid_search([GreedyAcquisition, UCB, RandomAcquisition]),
    "patience_max": 4,
    "kappa": 1,
    "kappa_decrease_factor": 1,
    "n_epoch_between_queries": 500,
    "acquire_n_at_a_time": 30,
    "n_initial": 30,
}

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "trainer": ActiveTrainer,  # PUT NUM GPU BACK TO 1
    "trainer_config": {
        **pipeline_config,
        **predictor_config,
        **model_config,
        **dataset_config,
        **active_learning_config
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "stop": {"training_iteration": 1000, 'all_space_explored': 1},
    "checkpoint_score_attr": 'eval/comb_r_squared',
    "keep_checkpoints_num": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 1,
    "resources_per_trial": {"cpu": 16, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
}