from recover.loggers import RecoverTBXLogger, AcquisitionMatrixWriter
from recover.datasets.drugcomb_matrix_data import DrugCombMatrix, DrugCombMatrixNoPPI
from recover.models.models import (
    GiantGraphGCN,
    Baseline,
    RawResponseBaseline,
    RawResponseGiantGraphGCN,
)
from recover.models.message_conv_layers import ThreeMessageConvLayer
from hyperopt import hp
from recover.models.predictors import (
    BasicMLPPredictor,
    FilmMLPPredictor,
    SharedLayersMLPPredictor,
    BilinearBasicMLPPredictor,
    BilinearFilmMLPPredictor,
    BilinearSharedLayersMLPPredictor,
)
from recover.utils import get_project_root
from recover.models.acquisition import (
    RandomAcquisition,
    ExpectedImprovement,
    GreedyAcquisition,
    GPUCB,
    Thompson,
)
from recover.train import train_epoch, eval_epoch, ActiveTrainer, BasicTrainer
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from recover.datasets.utils import take_first_k_vals
import os

"""
Configuration file where you can define the parameters for the pipeline. 

If you wish to perform grid search on some parameter, set its value to:
    tune.grid_search([value_1, ..., value_k])
"""

########################################################################################################################
# Pipeline Configuration
########################################################################################################################


pipeline_config = {
    "use_tune": False,
    "num_epoch_without_tune": 100,  # Used only if "use_tune" == False
    "seed": 1,
    # Optimizer config
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 64,  # tune.grid_search([1024, 2048]),
    # Train epoch and eval_epoch to use
    "train_epoch": train_epoch,
    "eval_epoch": eval_epoch,
}

########################################################################################################################
# Model Configuration
########################################################################################################################

trainer = ActiveTrainer  # Choose among [ActiveTrainer, BasicTrainer]

model_config = {
    "model": Baseline,  # Choose among [GiantGraphGCN, Baseline, RawResponseGiantGraphGCN, RawResponseBaseline]
    ##################################
    # Useful only if model uses raw response
    ##################################
    # Dimension of the hidden layers of the inhibition model (predicted by the hypernetwork)
    "inhibition_model_hid_lay_dim": 8,
    "inhibition_model_weight_decay": 1,  # Weight decay on the parameters of the inhibition model
    "mono_loss_scale_factor": 1,  # Set to zero to not take into account the prediction of monotherapy inhibitions
    "combo_loss_scale_factor": 1,  # Set to zero to not take into account the prediction of combination inhibitions
    # If false, the prediction of combinations will be the expected response as defined by the BLISS score
    "model_synergy": True,
    ##################################
    # Used if model is GiantGraphGCN
    ##################################
    "conv_layer": ThreeMessageConvLayer,
    # Booleans which control whether to pass messages along the different types of edges
    "pass_d2p_msg": True,
    "pass_p2d_msg": True,  # always set to True
    "pass_p2p_msg": True,
    "drug_self_loop": True,
    "prot_self_loop": True,
    # Residual layers
    "residual_layers_dim": 32,
    "num_res_layers": 1,
    # Periodic backprop
    "backprop_period": 4,
    "do_periodic_backprop": True,
    # Which protein features to use
    "use_prot_emb": True,
    "use_prot_feats": True,
    "prot_emb_dim": 16,  # Dimension of the learnable protein embeddings
    # Attention modules
    "drug_attention": {
        "attention": True,
        "attention_rank": 64,
        "dropout_proba": 0.4,
    },
    "prot_attention": {
        "attention": True,
        "attention_rank": 64,
        "dropout_proba": 0.4,
    },
}

########################################################################################################################
# Predictor Configuration
########################################################################################################################


predictor_config = {
    "predictor": BasicMLPPredictor,
    "predictor_layers": [
        1024,
        64,
        32,
        1,  # If using raw response, the dimension of the output will be overwritten
    ],
    "merge_n_layers_before_the_end": 1,  # Computation on the sum of the two drug embeddings for the last n layers
    # Whether the predictor will be given extra features as input (if using Baseline, with_fp and with_expr are ignored)
    "with_fp": True,
    "with_expr": True,
    "with_prot": False,
}

########################################################################################################################
# Dataset Configuration
########################################################################################################################


dataset_config = {
    "dataset": DrugCombMatrix,
    "transform": None,  # take_first_k_vals(500) if you wish to restrict the dataset to 500 randomly chosen combinations
    "pre_transform": None,
    "split_level": "drug",  # Choose among ["drug", "pair"]. Split at the "drug" level or at the drug "pair" level
    "val_set_prop": 0.3,
    "test_set_prop": 0.0,
    "cell_line": None,  # Restrict youself to a specific cell line: 'K-562'
    "target": "css",  # Choose among ["css", "bliss", "zip", "loewe", "hsa"], ignored if using raw response pipeline
    "fp_bits": 1024,  # Dimension of the drug fingerprints
    "fp_radius": 3,  # Radius of drug fingerprints
    # ppi_graph must be one of the following:
    # 'snap.csv', 'huri.csv', 'covid_liang_ppi.csv', 'biogrid_mv.csv', 'biogrid.csv', 'string_high_confidence.csv',
    # 'covid_krogan_ppi.csv'
    "ppi_graph": "huri.csv",
    # dti_graph must be one of the following:
    # 'chembl_dm.csv', 'chembl_dtis.csv', 'drug_repurposing_hub.csv'
    "dti_graph": "chembl_dtis.csv",
    "use_l1000": True,
    "restrict_to_l1000_covered_drugs": True,
}

########################################################################################################################
# Active learning Configuration
########################################################################################################################


active_learning_config = {
    # Choose among [ExpectedImprovement, RandomAcquisition, GreedyAcquisition, GPUCB, Thompson]
    "acquisition": ExpectedImprovement,
    "kappa": 0.5,  # Kappa parameter for GPUCB
    "kappa_decrease_factor": 1,  # Set lower than 1 for exponential decay
    "max_examp_per_epoch": 128,
    "n_epoch_between_queries": 3,
    "acquire_n_at_a_time": 32,  # Number of combinations to query at the same time
    "n_initial": 128,  # Initial size of the training set
}

logger_config = {
    "logger_classes": [],  # [RecoverTBXLogger],
    "tbx_writers": [AcquisitionMatrixWriter],
    "acquisition_mtx_cell_line_name": "K-562",
    "acquisition_matrix_similarity_metric": "jaccard",
}

bayesian_learning_config = {
    # It is possible to combine ensemble with MC dropout
    "ensemble": False,
    # For ensemble
    "ensemble_size": 5,
    # For MC-dropout
    "dropout_proba": 0.0,
    "num_dropout_lyrs": 1,  # Number of layers (starting from the end of the network) on which dropout is applied
    "n_train_forward_passes": 1,
    "n_scoring_forward_passes": 1,
}

########################################################################################################################
# Scheduler
########################################################################################################################
"""
To use schedulers, you need to change the "scheduler" and "search_alg" entries in the configuration dictionary 
at the end of this file
"""

asha_scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="valid_mse",
    mode="min",
    max_t=1000,
    grace_period=10,
    reduction_factor=3,
    brackets=1,
)

search_space = {
    "lr": hp.loguniform("lr", -16.118095651, -5.52146091786),
    "batch_size": hp.choice("batch_size", [128, 256, 512, 1024]),
}

current_best_params = [
    {
        "lr": 1e-4,
        "batch_size": 1024,
    }
]

search_alg = HyperOptSearch(
    search_space, metric="valid_mse", mode="min", points_to_evaluate=current_best_params
)

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "trainer": trainer,
    "trainer_config": {
        **pipeline_config,
        **predictor_config,
        **model_config,
        **dataset_config,
        **active_learning_config,
        **bayesian_learning_config,
        **logger_config,
    },
    "summaries_dir": os.path.join(get_project_root(), "RayLogs"),
    "memory": 1800,
    "checkpoint_freq": 0,
    "stop": {"training_iteration": 500, "all_space_explored": 1},
    "checkpoint_at_end": False,
    "resources_per_trial": {"cpu": 4, "gpu": 1},
    "scheduler": None,
    "search_alg": None,
}
