from recover.datasets.drugcomb_matrix_data import DrugCombMatrix
from recover.models.models import Baseline
from recover.models.predictors import BilinearFilmMLPPredictor, BilinearMLPPredictor, MLPPredictor
from recover.utils.utils import get_project_root
from recover.train import train_epoch, bayesian_train_epoch, eval_epoch, bayesian_eval_epoch, BasicTrainer, BayesianBasicTrainer
import os
from ray import tune
from importlib import import_module

########################################################################################################################
# Configuration
########################################################################################################################


pipeline_config = {
    "use_tune": True,
    "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    "seed": tune.grid_search([1, 2, 3, 42]),
    # Optimizer config
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "batch_size": 128,
    # Train epoch and eval_epoch to use
    "train_epoch": bayesian_train_epoch,
    "eval_epoch": bayesian_eval_epoch,
}

predictor_config = {
    "predictor": BilinearMLPPredictor,
    "bayesian_predictor": True,
    "bayesian_before_merge": False, # For bayesian predictor implementation - Layers after merge are bayesian by default
    "sigmoid": False,
    "num_realizations": 10, # For bayesian uncertainty
    "predictor_layers":
        [
            2048,
            128,
            64,
            1,
        ],
    "merge_n_layers_before_the_end": 2,  # Computation on the sum of the two drug embeddings for the last n layers
    "allow_neg_eigval": True,
    "stop": {"training_iteration": 1000, 'patience': 10}
}

model_config = {
    "model": Baseline,
    "load_model_weights": False,
}

dataset_config = {
    "dataset": DrugCombMatrix,
    "study_name": 'ALMANAC',
    "in_house_data": 'without',
    "rounds_to_include": [],
    "val_set_prop": 0.2,
    "test_set_prop": 0.1,
    "test_on_unseen_cell_line": False,
    "split_valid_train": "pair_level",
    "cell_line": 'MCF7',  # 'PC-3',
    "target": "bliss_max",  # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
    "fp_bits": 1024,
    "fp_radius": 2,
    "add_noise": True,
    "noise_type": 'salt_pepper', # 'gaussian', 'salt_pepper', 'random'
    "noise_prop": 0.1,
}

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "trainer": BayesianBasicTrainer,  # PUT NUM GPU BACK TO 1
    "trainer_config": {
        **pipeline_config,
        **predictor_config,
        **model_config,
        **dataset_config,
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "stop": {"training_iteration": 1000, 'patience': 10},
    "checkpoint_score_attr": 'eval/comb_r_squared',
    "keep_checkpoints_num": 1,
    "checkpoint_at_end": True,
    "checkpoint_freq": 0,
    "resources_per_trial": {"cpu": 16, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
}