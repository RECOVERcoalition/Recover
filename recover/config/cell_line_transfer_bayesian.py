from recover.datasets.drugcomb_matrix_data import DrugCombMatrix
from recover.models.models import Baseline
from recover.models.predictors import BayesianBilinearFilmMLPPredictor, BayesianBilinearLinFilmWithFeatMLPPredictor #adding Bayesian predictor
from recover.utils.utils import get_project_root
from recover.train import train_epoch_bayesian, eval_epoch, test_epoch, BayesianBasicTrainer #adding Bayesian training and trainer and including a tesing epoch
import os
from ray import tune
from importlib import import_module

########################################################################################################################
# Configuration
########################################################################################################################


pipeline_config = {
    "use_tune": True,
    "num_epoch_without_tune": 500,  # Used only if "use_tune" == False
    "seed": tune.grid_search([2, 3, 4]),
    # Optimizer config
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "batch_size": 128,
    # Train epoch and eval_epoch to use
    "train_epoch": train_epoch_bayesian, #updated train function, where KL divergence is used
    "eval_epoch": eval_epoch,
    "test_epoch": test_epoch, #added a Bayesian test epoch, used for differebt realizations
}

predictor_config = {
    "predictor": BayesianBilinearLinFilmWithFeatMLPPredictor,
    "predictor_layers":
        [
            2048,
            128,
            64,
            1,
        ],
    "merge_n_layers_before_the_end": 2,  # Computation on the sum of the two drug embeddings for the last n layers
    "allow_neg_eigval": True,
    "stop": {"training_iteration": 1000, 'patience': 10}, #in order to check when the training in over, we parse these arguments
    "realizations": 10 #define the number of realizations
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
    "test_on_unseen_cell_line": True,
    "cell_lines_in_test": ['MCF7'],
    "split_valid_train": "cell_line_level",
    "cell_line": None,  # 'PC-3',
    "target": "bliss_max",  # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
    "fp_bits": 1024,
    "fp_radius": 2
}

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "trainer": BayesianBasicTrainer,  #Adding Bayesian trainer
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
    "checkpoint_at_end": False,
    "checkpoint_freq": 1,
    "resources_per_trial": {"cpu": 8, "gpu": 0},
    "scheduler": None,
    "search_alg": None,
}
