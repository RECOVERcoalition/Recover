import torch
import torchbnn as bnn
import os
from torch.utils.data import DataLoader
from recover.utils.utils import get_tensor_dataset, get_tensor_dataset_swapped_combination, trial_dirname_creator
from recover.utils.utils import add_gaussian_noise, add_random_noise, add_salt_and_pepper_noise
from torch.utils.data import random_split
from ray import tune
import ray
import time
import argparse
import numpy as np
import pandas as pd
import importlib
from scipy import stats
from scipy.stats import spearmanr


########################################################################################################################
# Epoch loops
########################################################################################################################


def train_epoch(data, loader, model, optim):

    model.train()

    epoch_loss = 0
    num_batches = len(loader)

    all_mean_preds = []
    all_targets = []

    for _, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch)

        # Save all predictions and targets
        all_mean_preds.extend(out.mean(dim=1).tolist())
        all_targets.extend(drug_drug_batch[2].tolist())

        loss = model.loss(out, drug_drug_batch)

        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    epoch_comb_r_squared = stats.linregress(all_mean_preds, all_targets).rvalue**2

    summary_dict = {
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared
    }

    print("Training", summary_dict)

    return summary_dict

"""
The cost function is updated to consider both MSE loss and KL loss
KL weight is set according to batchsize 
Default model loss is unchanged - MSE (For comparision with Rec
"""
def bayesian_train_epoch(data, loader, model, optim):

    model.train()

    epoch_loss = 0
    num_batches = len(loader)
    batch = 1

    all_mean_preds = []
    all_targets = []
    
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    for _, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch)

        # Save all predictions and targets
        all_mean_preds.extend(out.mean(dim=1).tolist())
        all_targets.extend(drug_drug_batch[2].tolist())

        loss = model.loss(out, drug_drug_batch) #MSE
        kl = kl_loss(model)

        kl_weight = pow(2, num_batches-batch)/(pow(2, num_batches)-1) #kl weight based on the paper "Weight Uncertainty in Neural Networks"

        cost = loss + kl_weight*kl

        cost.backward()
        optim.step()

        epoch_loss += loss.item()
        batch += 1

    epoch_comb_r_squared = stats.linregress(all_mean_preds, all_targets).rvalue**2

    summary_dict = {
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared
    }

    print("Training", summary_dict)

    return summary_dict


def eval_epoch(data, loader, model):
    model.eval()

    epoch_loss = 0
    num_batches = len(loader)

    all_out = []
    all_mean_preds = []
    all_targets = []

    with torch.no_grad():
        for _, drug_drug_batch in enumerate(loader):
            out = model.forward(data, drug_drug_batch)

            # Save all predictions and targets
            all_out.append(out)
            all_mean_preds.extend(out.mean(dim=1).tolist())
            all_targets.extend(drug_drug_batch[2].tolist())

            loss = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()

        epoch_comb_r_squared = stats.linregress(all_mean_preds, all_targets).rvalue**2
        epoch_spear = spearmanr(all_targets, all_mean_preds).correlation

    summary_dict = {
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared,
        "spearman": epoch_spear
    }

    print("Testing", summary_dict, '\n')

    all_out = torch.cat(all_out)

    return summary_dict, all_out


"""
Eval epoch is the same as the non-bayesian method, 
minor changes were done to return drug combinations for ease of analysis
"""
def bayesian_eval_epoch(data, loader, model):
    model.eval()

    epoch_loss = 0
    num_batches = len(loader)

    all_out = []
    all_mean_preds = []
    all_targets = []
    all_combs = []

    with torch.no_grad():
        for _, drug_drug_batch in enumerate(loader):
            out = model.forward(data, drug_drug_batch)

            # Save all predictions and targets - rec_id_to_idx_dict
            all_out.append(out)
            all_mean_preds.extend(out.mean(dim=1).tolist())
            all_targets.extend(drug_drug_batch[2].tolist())
            all_combs.extend(drug_drug_batch[0].tolist())

            loss = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()
        epoch_comb_r_squared = stats.linregress(all_mean_preds, all_targets).rvalue**2
        epoch_spear = spearmanr(all_targets, all_mean_preds).correlation

    summary_dict = {
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared,
        "spearman": epoch_spear
    }
    
    print("Testing", summary_dict, '\n')

    all_out = torch.cat(all_out)

    return summary_dict, all_out, all_combs


"""
Custom Aggregration function for obtaining results when training with multi cell-lines 

Functionality: group data by the combination (permute invarient) and get maximum mean synergy score and the corresponding std
"""
def custom_agg(group):
    max_row = group.loc[group['mean'].idxmax()]
    return pd.Series({'combination': max_row['combination'],
                  'mean': max_row['mean'],
                  'std': max_row['std']})


########################################################################################################################
# Basic trainer
########################################################################################################################


class BasicTrainer(tune.Trainable):
    def setup(self, config):
        print("Initializing regular training pipeline")

        self.batch_size = config["batch_size"]
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.training_it = 0

        # Initialize dataset
        dataset = config["dataset"](
            fp_bits=config["fp_bits"],
            fp_radius=config["fp_radius"],
            cell_line=config["cell_line"],
            study_name=config["study_name"],
            in_house_data=config["in_house_data"],
            rounds_to_include=config["rounds_to_include"],
        )

        self.data = dataset.data.to(self.device)

        # If a score is the target, we store it in the ddi_edge_response attribute of the data object
        if "target" in config.keys():
            possible_target_dicts = {
                "bliss_max": self.data.ddi_edge_bliss_max,
                "bliss_av": self.data.ddi_edge_bliss_av,
                "css_av": self.data.ddi_edge_css_av,
            }
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

        # Perform train/valid/test split. Test split is fixed regardless of the user defined seed
        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(config)

        # Train loader
        train_ddi_dataset = get_tensor_dataset(self.data, self.train_idxs)

        self.train_loader = DataLoader(
            train_ddi_dataset,
            batch_size=config["batch_size"]
        )

        # Valid loader
        valid_ddi_dataset = get_tensor_dataset(self.data, self.val_idxs)

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"]
        )

        # Initialize model
        self.model = config["model"](self.data, config)

        # Initialize model with weights from file
        load_model_weights = config.get("load_model_weights", False)
        if load_model_weights:
            model_weights_file = config.get("model_weights_file")
            model_weights = torch.load(model_weights_file, map_location="cpu")
            self.model.load_state_dict(model_weights)
            print("pretrained weights loaded")
        else:
            print("model initialized randomly")

        self.model = self.model.to(self.device)
        print(self.model)

        # Initialize optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

        self.patience = 0
        self.max_eval_r_squared = -1

    def step(self):

        train_metrics = self.train_epoch(
            self.data,
            self.train_loader,
            self.model,
            self.optim,
        )

        eval_metrics, _ = self.eval_epoch(self.data, self.valid_loader, self.model)

        train_metrics = [("train/" + k, v) for k, v in train_metrics.items()]
        eval_metrics = [("eval/" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(train_metrics + eval_metrics)

        metrics["training_iteration"] = self.training_it
        self.training_it += 1

        # Compute patience
        if metrics['eval/comb_r_squared'] > self.max_eval_r_squared:
            self.patience = 0
            self.max_eval_r_squared = metrics['eval/comb_r_squared']
        else:
            self.patience += 1

        metrics['patience'] = self.patience
        metrics['all_space_explored'] = 0

        return metrics

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


########################################################################################################################
# Bayesian Basic trainer
########################################################################################################################


class BayesianBasicTrainer(tune.Trainable):
    def setup(self, config):
        print("Initializing regular training pipeline")
        
        self.test_invariance = True 

        self.batch_size = config["batch_size"]
        self.num_realizations = config["num_realizations"]
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.training_it = 0
        self.add_noise = False
        
        try:
            self.add_noise = config["add_noise"]
            self.noise_type = config["noise_type"]
            self.noise_prop = config["noise_prop"]
        except Exception as e:
            self.noise_type = None
            self.noise_prop = 0.1
        

        # Initialize dataset
        dataset = config["dataset"](
            fp_bits=config["fp_bits"],
            fp_radius=config["fp_radius"],
            cell_line=config["cell_line"],
            study_name=config["study_name"],
            in_house_data=config["in_house_data"],
            rounds_to_include=config["rounds_to_include"],
        )
        
        self.patience_stop = config["stop"]["patience"]
        self.max_iter = config["stop"]["training_iteration"]

        self.data = dataset.data.to(self.device)

        # If a score is the target, we store it in the ddi_edge_response attribute of the data object
        if "target" in config.keys():
            possible_target_dicts = {
                "bliss_max": self.data.ddi_edge_bliss_max,
                "bliss_av": self.data.ddi_edge_bliss_av,
                "css_av": self.data.ddi_edge_css_av,
            }
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

        # Perform train/valid/test split. Test split is fixed regardless of the user defined seed
        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(config)

        # Train loader
        train_ddi_dataset = get_tensor_dataset(self.data, self.train_idxs)
        
        if self.add_noise:
            if self.noise_type == 'gaussian':
                train_ddi_dataset = add_gaussian_noise(self.data, self.train_idxs, self.noise_prop)
            elif self.noise_type == 'salt_pepper':
                train_ddi_dataset = add_salt_and_pepper_noise(self.data, self.train_idxs, self.noise_prop)
            else :
                train_ddi_dataset = add_random_noise(self.data, self.train_idxs, self.noise_prop)
            

        self.train_loader = DataLoader(
            train_ddi_dataset,
            batch_size=config["batch_size"]
        )

        # Valid loader
        valid_ddi_dataset = get_tensor_dataset(self.data, self.val_idxs)

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"]
        )
        
        # Test loader
        test_ddi_dataset = get_tensor_dataset(self.data, self.test_idxs)

        self.test_loader = DataLoader(
            test_ddi_dataset,
            batch_size=config["batch_size"]
        )


        # Initialize model
        self.model = config["model"](self.data, config)

        # Initialize model with weights from file
        load_model_weights = config.get("load_model_weights", False)
        if load_model_weights:
            model_weights_file = config.get("model_weights_file")
            model_weights = torch.load(model_weights_file, map_location="cpu")
            self.model.load_state_dict(model_weights)
            print("pretrained weights loaded")
        else:
            print("model initialized randomly")

        self.model = self.model.to(self.device)
        print(self.model)

        # Initialize optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

        self.patience = 0
        self.max_eval_r_squared = -1

    def step(self):
    
        num_realizations = self.num_realizations

        train_metrics = self.train_epoch(
            self.data,
            self.train_loader,
            self.model,
            self.optim,
        )

        eval_metrics, _, _ = self.eval_epoch(self.data, self.valid_loader, self.model)

        train_metrics = [("train/" + k, v) for k, v in train_metrics.items()]
        eval_metrics = [("eval/" + k, v) for k, v in eval_metrics.items()]

        metrics = dict(train_metrics + eval_metrics)
        
        metrics["training_iteration"] = self.training_it
        self.training_it += 1
                
        # Compute patience
        if metrics['eval/comb_r_squared'] > self.max_eval_r_squared:
            self.patience = 0
            self.max_eval_r_squared = metrics['eval/comb_r_squared']
        else:
            self.patience += 1
            
        metrics['patience'] = self.patience
        metrics['all_space_explored'] = 0                        


        """
        Start the test on unseen dataset once the training is Done
        P.S. The combinations are unseen, this is not related to the drugs
        """
        if ((self.patience >= self.patience_stop) | (self.training_it > self.max_iter)):
            print("Test performance")
            test_result = {}
            
            realization_results, result_synergy, drug_combs = self.eval_epoch(self.data, self.test_loader, self.model)
            drug_combinations_synergy = {'data': self.test_idxs}
            test_metrics = dict([("test/" + k, [v]) for k, v in realization_results.items()])
            
            for i in range(num_realizations-1):
                realization_results, new_synergy, _ = self.eval_epoch(self.data, self.test_loader, self.model)
                result_synergy = torch.cat((result_synergy, new_synergy), dim=1)
                for k, v in realization_results.items():
                    test_metrics["test/" + k].append(v)
                    
            for key in test_metrics:
                test_result[str(key)+ "/mean"] = np.mean(test_metrics[key])
                test_result[str(key)+ "/std"] = np.std(test_metrics[key])
                
            
            synergy_mean = list(torch.mean(result_synergy, dim=1).numpy())
            synergy_std = list(torch.std(result_synergy, dim=1).numpy())
            
            """
            # Uncomment this to get the synergy prediction performance when working with multi-cell lines 
            # Aggregate result for duplicate sets regardless of cell-line
            result_tuples = list(map(lambda inner_list: tuple(sorted(inner_list)), drug_combs))
            dataset = pd.DataFrame({'combination': result_tuples, 'mean': synergy_mean, 'std': synergy_std }, columns=['combination', 'mean', 'std'])
            result_df = dataset.groupby('combination').apply(custom_agg) # Do custom aggregration  - Not needed when working on one cell-line but no difference in results
            result_df.reset_index(drop=True, inplace=True)
            """
            
            metrics.update(dict(test_result))
            
            metrics['synergy_combs'] = list(drug_combs)
            metrics['synergy_mean'] = list(synergy_mean)
            metrics['synergy_std'] = list(synergy_std)
            
            """
            # Start the test on permuted unseen dataset - To confirm the invariance
            # Change the test_invariance boolean in Trainer setup to enable/disable this part of the code
            """
            if self.test_invariance:
            
                print("Test on swapped combinations")
                
                test_ddi_dataset_swapped = get_tensor_dataset_swapped_combination(self.data, self.test_idxs)
                permuted_test_loader = DataLoader(
                    test_ddi_dataset_swapped,
                    batch_size=self.batch_size
                )
                
                swapped_test_result = {}
                swapped_test_metrics = {}
                
                realization_results, result_synergy, drug_combs = self.eval_epoch(self.data, permuted_test_loader, self.model)
                drug_combinations_synergy = {'data': self.test_idxs}
                swapped_test_metrics = dict([("test_swapped/" + k, [v]) for k, v in realization_results.items()])
                
                for i in range(num_realizations-1):
                    realization_results, new_synergy, _ = self.eval_epoch(self.data, self.test_loader, self.model)
                    result_synergy = torch.cat((result_synergy, new_synergy), dim=1)
                    for k, v in realization_results.items():
                        swapped_test_metrics["test_swapped/" + k].append(v)
                        
                for key in swapped_test_metrics:
                    swapped_test_result[str(key)+ "/mean"] = np.mean(swapped_test_metrics[key])
                    swapped_test_result[str(key)+ "/std"] = np.std(swapped_test_metrics[key])
                    
                
                synergy_mean = list(torch.mean(result_synergy, dim=1).numpy())
                synergy_std = list(torch.std(result_synergy, dim=1).numpy())
                
                metrics.update(dict(swapped_test_result))
                
                metrics['synergy_combs_permuted'] = list(drug_combs)
                metrics['synergy_mean_permuted'] = list(synergy_mean)
                metrics['synergy_std_permuted'] = list(synergy_std)
            
        return metrics
    
    
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))



########################################################################################################################
# Active learning Trainer
########################################################################################################################


class ActiveTrainer(BasicTrainer):
    """
    Trainer class to perform active learning. Retrains models from scratch after each query. Uses early stopping
    """

    def setup(self, config):
        print("Initializing active training pipeline")
        super(ActiveTrainer, self).setup(config)

        self.acquire_n_at_a_time = config["acquire_n_at_a_time"]
        self.acquisition = config["acquisition"](config)
        self.n_epoch_between_queries = config["n_epoch_between_queries"]

        # randomly acquire data at the beginning
        self.seen_idxs = self.train_idxs[:config["n_initial"]]
        self.unseen_idxs = self.train_idxs[config["n_initial"]:]
        self.immediate_regrets = torch.empty(0)

        # Initialize variable that saves the last query
        self.last_query_idxs = self.seen_idxs

        # Initialize dataloaders
        self.seen_loader, self.unseen_loader = self.update_loaders(self.seen_idxs, self.unseen_idxs)

        # Get the set of top 1% most synergistic combinations
        one_perc = int(0.01 * len(self.unseen_idxs))
        scores = self.data.ddi_edge_response[self.unseen_idxs]
        self.best_score = scores.max()
        self.top_one_perc = set(self.unseen_idxs[torch.argsort(scores, descending=True)[:one_perc]].numpy())
        self.count = 0

    def step(self):
        # Check whether we have explored everything
        if len(self.unseen_loader) == 0:
            print("All space has been explored")
            return {"all_space_explored": 1, "training_iteration": self.training_it}

        # Train on seen examples
        if len(self.seen_idxs) > 0:
            seen_metrics = self.train_between_queries()
        else:
            seen_metrics = {}

        # Evaluate on valid set
        eval_metrics, _ = self.eval_epoch(self.data, self.valid_loader, self.model)

        # Score unseen examples
        unseen_metrics, unseen_preds = self.eval_epoch(self.data, self.unseen_loader, self.model)

        active_scores = self.acquisition.get_scores(unseen_preds)

        # Build summary
        seen_metrics = [("seen/" + k, v) for k, v in seen_metrics.items()]
        unseen_metrics = [("unseen/" + k, v) for k, v in unseen_metrics.items()]
        eval_metrics = [("eval/" + k, v) for k, v in eval_metrics.items()]

        metrics = dict(
            seen_metrics
            + unseen_metrics
            + eval_metrics
        )

        # Acquire new data
        print("query data")
        query = self.unseen_idxs[torch.argsort(active_scores, descending=True)[:self.acquire_n_at_a_time]]

        # Get the best score among unseen examples
        self.best_score = self.data.ddi_edge_response[self.unseen_idxs].max().detach().cpu()
        # remove the query from the unseen examples
        self.unseen_idxs = self.unseen_idxs[torch.argsort(active_scores, descending=True)[self.acquire_n_at_a_time:]]

        # Add the query to the seen examples
        self.seen_idxs = torch.cat((self.seen_idxs, query))
        metrics["seen_idxs"] = self.data.ddi_edge_idx[:, self.seen_idxs].detach().cpu().tolist()
        metrics["seen_idxs_in_dataset"] = self.seen_idxs.detach().cpu().tolist()

        # Compute proportion of top 1% synergistic drugs which have been discovered
        query_set = set(query.detach().numpy())
        self.count += len(query_set & self.top_one_perc)
        metrics["top"] = self.count / len(self.top_one_perc)

        query_ground_truth = self.data.ddi_edge_response[query].detach().cpu()

        query_pred_syn = unseen_preds[torch.argsort(active_scores, descending=True)[:self.acquire_n_at_a_time]]
        query_pred_syn = query_pred_syn.detach().cpu()

        metrics["query_pred_syn_mean"] = query_pred_syn.mean().item()
        metrics["query_true_syn_mean"] = query_ground_truth.mean().item()

        # Diversity metric
        metrics["n_unique_drugs_in_query"] = len(self.data.ddi_edge_idx[:, query].unique())

        # Get the quantiles of the distribution of true synergy in the query
        for q in np.arange(0, 1.1, 0.1):
            metrics["query_pred_syn_quantile_" + str(q)] = np.quantile(query_pred_syn, q)
            metrics["query_true_syn_quantile_" + str(q)] = np.quantile(query_ground_truth, q)

        query_immediate_regret = torch.abs(self.best_score - query_ground_truth)
        self.immediate_regrets = torch.cat((self.immediate_regrets, query_immediate_regret))

        metrics["med_immediate_regret"] = self.immediate_regrets.median().item()
        metrics["log10_med_immediate_regret"] = np.log10(metrics["med_immediate_regret"])
        metrics["min_immediate_regret"] = self.immediate_regrets.min().item()
        metrics["log10_min_immediate_regret"] = np.log10(metrics["min_immediate_regret"])

        # Update the dataloaders
        self.seen_loader, self.unseen_loader = self.update_loaders(self.seen_idxs, self.unseen_idxs)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0
        self.training_it += 1

        return metrics

    def train_between_queries(self):
        # Create the train and early_stop loaders for this iteration
        iter_dataset = self.seen_loader.dataset
        train_length = int(0.8 * len(iter_dataset))
        early_stop_length = len(iter_dataset) - train_length

        train_dataset, early_stop_dataset = random_split(iter_dataset, [train_length, early_stop_length])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(train_dataset) > 0,
        )

        early_stop_loader = DataLoader(
            early_stop_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(early_stop_dataset) > 0,
        )

        # Reinitialize model before training
        self.model = self.config["model"](self.data, self.config).to(self.device)

        # Initialize model with weights from file
        load_model_weights = self.config.get("load_model_weights", False)
        if load_model_weights:
            model_weights_file = self.config.get("model_weights_file")
            model_weights = torch.load(model_weights_file, map_location="cpu")
            self.model.load_state_dict(model_weights)
            print("pretrained weights loaded")
        else:
            print("model initialized randomly")

        # Reinitialize optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"],
                                      weight_decay=self.config["weight_decay"])

        best_eval_r2 = float("-inf")
        patience_max = self.config["patience_max"]
        patience = 0

        for _ in range(self.n_epoch_between_queries):
            # Perform several training epochs. Save only metrics from the last epoch
            train_metrics = self.train_epoch(self.data, train_loader, self.model, self.optim)

            early_stop_metrics, _ = self.eval_epoch(self.data, early_stop_loader, self.model)

            if early_stop_metrics["comb_r_squared"] > best_eval_r2:
                best_eval_r2 = early_stop_metrics["comb_r_squared"]
                print("best early stop r2", best_eval_r2)
                patience = 0
            else:
                patience += 1

            if patience > patience_max:
                break

        return train_metrics

    def update_loaders(self, seen_idxs, unseen_idxs):
        # Seen loader
        seen_ddi_dataset = get_tensor_dataset(self.data, seen_idxs)

        seen_loader = DataLoader(
            seen_ddi_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(seen_idxs) > 0,
        )

        # Unseen loader
        unseen_ddi_dataset = get_tensor_dataset(self.data, unseen_idxs)

        unseen_loader = DataLoader(
            unseen_ddi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cpu"),
        )

        return seen_loader, unseen_loader


########################################################################################################################
# Bayesian Active learning Trainer 
########################################################################################################################


class ActiveTrainerBayesian(BasicTrainer):
    """
    Trainer class to perform active learning with Bayesian Model. Retrains models from scratch after each query. Uses early stopping
    """

    def setup(self, config):
        print("Initializing active training pipeline")
        super(ActiveTrainerBayesian, self).setup(config)

        self.acquire_n_at_a_time = config["acquire_n_at_a_time"]
        self.acquisition = config["acquisition"](config)
        self.n_epoch_between_queries = config["n_epoch_between_queries"]
        self.num_realizations = config["num_realizations"]

        # randomly acquire data at the beginning
        self.seen_idxs = self.train_idxs[:config["n_initial"]]
        self.unseen_idxs = self.train_idxs[config["n_initial"]:]
        self.immediate_regrets = torch.empty(0)

        # Initialize variable that saves the last query
        self.last_query_idxs = self.seen_idxs

        # Initialize dataloaders
        self.seen_loader, self.unseen_loader = self.update_loaders(self.seen_idxs, self.unseen_idxs)

        # Get the set of top 1% most synergistic combinations
        one_perc = int(0.01 * len(self.unseen_idxs))
        scores = self.data.ddi_edge_response[self.unseen_idxs]
        self.best_score = scores.max()
        self.top_one_perc = set(self.unseen_idxs[torch.argsort(scores, descending=True)[:one_perc]].numpy())
        self.count = 0

    def step(self):
        # Check whether we have explored everything
        if len(self.unseen_loader) == 0:
            print("All space has been explored")
            return {"all_space_explored": 1, "training_iteration": self.training_it}

        # Train on seen examples
        if len(self.seen_idxs) > 0:
            seen_metrics = self.train_between_queries()
        else:
            seen_metrics = {}

        # Evaluate on valid set
        eval_metrics, _, _ = self.eval_epoch(self.data, self.valid_loader, self.model)
        
        # Test on test set for given number of realizations
        unseen_metrics = {}
        realization_results, unseen_preds, drug_combs = self.eval_epoch(self.data, self.unseen_loader, self.model)
        unseen_result = dict([("unseen/" + k, [v]) for k, v in realization_results.items()])
        
        for i in range(self.num_realizations - 1):
            realization_results, result_tensor, _ = self.eval_epoch(self.data, self.unseen_loader, self.model)
            unseen_preds = torch.cat((unseen_preds, result_tensor), dim=1)
            for k, v in realization_results.items():
                unseen_result["unseen/" + k].append(v)
                
        for key in unseen_result:
            unseen_metrics[str(key)+ "/mean"] = np.mean(unseen_result[key])
            unseen_metrics[str(key)+ "/std"] = np.var(unseen_result[key])
            

        active_scores = self.acquisition.get_scores(unseen_preds)

        # Build summary
        seen_metrics = [("seen/" + k, v) for k, v in seen_metrics.items()]
        eval_metrics = [("eval/" + k, v) for k, v in eval_metrics.items()]

        metrics = dict(
            seen_metrics
            + eval_metrics
        )
        
        metrics.update(dict(unseen_metrics))
        
        

        # Acquire new data
        print("query data")
        query = self.unseen_idxs[torch.argsort(active_scores, descending=True)[:self.acquire_n_at_a_time]]

        # Get the best score among unseen examples
        self.best_score = self.data.ddi_edge_response[self.unseen_idxs].max().detach().cpu()
        # remove the query from the unseen examples
        self.unseen_idxs = self.unseen_idxs[torch.argsort(active_scores, descending=True)[self.acquire_n_at_a_time:]]

        # Add the query to the seen examples
        self.seen_idxs = torch.cat((self.seen_idxs, query))
        metrics["seen_idxs"] = self.data.ddi_edge_idx[:, self.seen_idxs].detach().cpu().tolist()
        metrics["seen_idxs_in_dataset"] = self.seen_idxs.detach().cpu().tolist()

        # Compute proportion of top 1% synergistic drugs which have been discovered
        query_set = set(query.detach().numpy())
        self.count += len(query_set & self.top_one_perc)
        metrics["top"] = self.count / len(self.top_one_perc)

        query_ground_truth = self.data.ddi_edge_response[query].detach().cpu()

        query_pred_syn = unseen_preds[torch.argsort(active_scores, descending=True)[:self.acquire_n_at_a_time]]
        query_pred_syn = query_pred_syn.detach().cpu()

        metrics["query_pred_syn_mean"] = query_pred_syn.mean().item()
        metrics["query_true_syn_mean"] = query_ground_truth.mean().item()

        # Diversity metric
        metrics["n_unique_drugs_in_query"] = len(self.data.ddi_edge_idx[:, query].unique())

        # Get the quantiles of the distribution of true synergy in the query
        for q in np.arange(0, 1.1, 0.1):
            metrics["query_pred_syn_quantile_" + str(q)] = np.quantile(query_pred_syn, q)
            metrics["query_true_syn_quantile_" + str(q)] = np.quantile(query_ground_truth, q)

        query_immediate_regret = torch.abs(self.best_score - query_ground_truth)
        self.immediate_regrets = torch.cat((self.immediate_regrets, query_immediate_regret))

        metrics["med_immediate_regret"] = self.immediate_regrets.median().item()
        metrics["log10_med_immediate_regret"] = np.log10(metrics["med_immediate_regret"])
        metrics["min_immediate_regret"] = self.immediate_regrets.min().item()
        metrics["log10_min_immediate_regret"] = np.log10(metrics["min_immediate_regret"])

        # Update the dataloaders
        self.seen_loader, self.unseen_loader = self.update_loaders(self.seen_idxs, self.unseen_idxs)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0
        self.training_it += 1

        return metrics

    def train_between_queries(self):
        # Create the train and early_stop loaders for this iteration
        iter_dataset = self.seen_loader.dataset
        train_length = int(0.8 * len(iter_dataset))
        early_stop_length = len(iter_dataset) - train_length

        train_dataset, early_stop_dataset = random_split(iter_dataset, [train_length, early_stop_length])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(train_dataset) > 0,
        )

        early_stop_loader = DataLoader(
            early_stop_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(early_stop_dataset) > 0,
        )

        # Reinitialize model before training
        self.model = self.config["model"](self.data, self.config).to(self.device)

        # Initialize model with weights from file
        load_model_weights = self.config.get("load_model_weights", False)
        if load_model_weights:
            model_weights_file = self.config.get("model_weights_file")
            model_weights = torch.load(model_weights_file, map_location="cpu")
            self.model.load_state_dict(model_weights)
            print("pretrained weights loaded")
        else:
            print("model initialized randomly")

        # Reinitialize optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"],
                                      weight_decay=self.config["weight_decay"])

        best_eval_r2 = float("-inf")
        patience_max = self.config["patience_max"]
        patience = 0

        for _ in range(self.n_epoch_between_queries):
            # Perform several training epochs. Save only metrics from the last epoch
            train_metrics = self.train_epoch(self.data, train_loader, self.model, self.optim)

            early_stop_metrics, _results, _ = self.eval_epoch(self.data, early_stop_loader, self.model)

            if early_stop_metrics["comb_r_squared"] > best_eval_r2:
                best_eval_r2 = early_stop_metrics["comb_r_squared"]
                print("best early stop r2", best_eval_r2)
                patience = 0
            else:
                patience += 1

            if patience > patience_max:
                break

        return train_metrics

    def update_loaders(self, seen_idxs, unseen_idxs):
        # Seen loader
        seen_ddi_dataset = get_tensor_dataset(self.data, seen_idxs)

        seen_loader = DataLoader(
            seen_ddi_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
            shuffle=len(seen_idxs) > 0,
        )

        # Unseen loader
        unseen_ddi_dataset = get_tensor_dataset(self.data, unseen_idxs)

        unseen_loader = DataLoader(
            unseen_ddi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cpu"),
        )

        return seen_loader, unseen_loader


########################################################################################################################
# Main train method
########################################################################################################################


def train(configuration):
    if configuration["trainer_config"]["use_tune"]:
        ###########################################
        # Use tune
        ###########################################

        ray.init(num_cpus=configuration["resources_per_trial"]["cpu"],
                 num_gpus=configuration["resources_per_trial"]["gpu"])

        time_to_sleep = 5
        print("Sleeping for %d seconds" % time_to_sleep)
        time.sleep(time_to_sleep)
        print("Woke up.. Scheduling")

        tune.run(
            configuration["trainer"],
            name=configuration["name"],
            config=configuration["trainer_config"],
            stop=configuration["stop"],
            resources_per_trial=configuration["resources_per_trial"],
            local_dir=configuration["summaries_dir"],
            checkpoint_freq=configuration.get("checkpoint_freq"),
            checkpoint_at_end=configuration.get("checkpoint_at_end"),
            checkpoint_score_attr=configuration.get("checkpoint_score_attr"),
            keep_checkpoints_num=configuration.get("keep_checkpoints_num"),
            trial_dirname_creator=trial_dirname_creator
        )

    else:
        ###########################################
        # Do not use tune
        ###########################################

        trainer = configuration["trainer"](configuration["trainer_config"])
        for i in range(configuration["trainer_config"]["num_epoch_without_tune"]):
            trainer.train()


if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help='Name of the configuration file without ".py" at the end',
    )
    args = parser.parse_args()

    # Retrieve configuration
    my_config = importlib.import_module("recover.config." + args.config)
    print("Running with configuration from", "recover.config." + args.config)

    # Set the name of the log directory after the name of the config file
    my_config.configuration["name"] = args.config

    # Train
    train(my_config.configuration)
