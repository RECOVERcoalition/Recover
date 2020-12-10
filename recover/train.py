import torch
from recover.models.models import GiantGraphGCN, EnsembleModel, Baseline
import os
from torch.utils.data import DataLoader
from recover.datasets.utils import get_tensor_dataset
from ray import tune
import ray
import time
import argparse
import importlib


########################################################################################################################
# Epoch loops
########################################################################################################################


def train_epoch(
    data, loader, model, optim, max_examp_per_epoch=None, n_forward_passes=1
):
    """
    Trains a model for one epoch

    :param data: `torch_geometric.data.Data` object
    :param loader: dataloader to iterate over
    :param model:
    :param optim: optimizer used
    :param max_examp_per_epoch: Maximum size of the epoch
    :param n_forward_passes: Number of forward passes to perform for each batch (for MC dropout)
    :return: dictionary of losses
    """
    model.train()
    model.enable_periodic_backprop()

    epoch_loss = 0
    num_batches = len(loader)

    examples_seen = 0

    for _, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch, n_forward_passes=n_forward_passes)
        loss, comb_r_squared, mono_r_squared = model.loss(out, drug_drug_batch)

        loss.backward()
        optim.step()

        epoch_loss += loss.item()

        # If we have seen enough examples in this epoch, break
        examples_seen += drug_drug_batch[0].shape[0]
        if max_examp_per_epoch is not None:
            if examples_seen >= max_examp_per_epoch:
                break

    print("Mean train loss: {:.4f}".format(epoch_loss / num_batches))

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches}


def eval_epoch(
    data, loader, model, acquisition=None, n_forward_passes=1, message="Mean valid loss"
):
    """
    Evaluates a model

    :param data: `torch_geometric.data.Data` object
    :param loader: dataloader to iterate over
    :param model:
    :param acquisition: acquisition function to compute scores with. If None, no scores will be computed
    :param n_forward_passes: Number of forward passes to perform for each batch (for MC dropout)
    :param message: message to plot in front of the loss
    :return: dictionary of metrics and scores (if an acquisition function is provided)
    """
    model.eval()
    model.disable_periodic_backprop()

    # If using several forward passes at eval time, we need to enable dropout
    if n_forward_passes > 1:
        model.enable_dropout()

    epoch_loss = 0
    epoch_comb_r_squared = 0
    epoch_mono_r_squared = 0
    num_batches = len(loader)
    active_scores = torch.empty(0)

    with torch.no_grad():
        for _, drug_drug_batch in enumerate(loader):
            out = model.forward(
                data, drug_drug_batch, n_forward_passes=n_forward_passes
            )

            if acquisition is not None:
                active_scores = torch.cat((active_scores, acquisition.get_scores(out)))

            loss, comb_r_squared, mono_r_squared = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()
            epoch_comb_r_squared += comb_r_squared
            epoch_mono_r_squared += mono_r_squared

    print(message + ": {:.4f}".format(epoch_loss / num_batches))

    summary_dict = {
        "loss_sum": epoch_loss,
        "loss_mean": epoch_loss / num_batches,
        "comb_r_squared": epoch_comb_r_squared / num_batches,
        "mono_r_squared": epoch_mono_r_squared / num_batches,
    }

    if acquisition is not None:
        return summary_dict, active_scores
    return summary_dict


########################################################################################################################
# Abstract trainer
########################################################################################################################


class AbstractTrainer(tune.Trainable):
    """
    Abstract trainer class
    """

    def setup(self, config):
        self.batch_size = config["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_train_forward_passes = config["n_train_forward_passes"]
        self.training_it = 0
        config["device"] = self.device

        # Initialize dataset
        dataset = config["dataset"](
            transform=config["transform"],
            pre_transform=config["pre_transform"],
            fp_bits=config["fp_bits"],
            fp_radius=config["fp_radius"],
            ppi_graph=config["ppi_graph"],
            dti_graph=config["dti_graph"],
            cell_line=config["cell_line"],
            use_l1000=config["use_l1000"],
            restrict_to_l1000_covered_drugs=config["restrict_to_l1000_covered_drugs"],
        )

        self.data = dataset[0].to(self.device)

        # If a score is the target, we store it in the ddi_edge_response attribute of the data object
        if "target" in config.keys():
            possible_target_dicts = {
                "css": self.data.ddi_edge_css,
                "bliss": self.data.ddi_edge_bliss,
                "zip": self.data.ddi_edge_zip,
                "hsa": self.data.ddi_edge_hsa,
                "loewe": self.data.ddi_edge_loewe,
            }
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])

        # Perform train/valid/test split. Test split is fixed regardless of the user defined seed
        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(
            config["test_set_prop"], config["val_set_prop"], config["split_level"]
        )

        # Valid loader
        valid_ddi_dataset = get_tensor_dataset(self.data, self.val_idxs)

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"],
            pin_memory=(self.device == "cpu"),
        )

        # Initialize model
        if config["ensemble"] is True:
            self.model = EnsembleModel(self.data, config).to(self.device)
        else:
            self.model = config["model"](self.data, config).to(self.device)
        print(self.model)

        # Initialize optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

        self.loggers = self._build_loggers(config)

    def log_result(self, result):
        super().log_result(result)

        for logger in self.loggers:
            logger.log_result(result)

    def _build_loggers(self, config):
        logger_classes = config.get("logger_classes", [])
        return [
            logger_class(config, self.logdir, self.data)
            for logger_class in logger_classes
        ]

    def cleanup(self):
        for logger in self.loggers:
            logger.flush()
            logger.close()

    def reset_config(self, new_config):
        for logger in self.loggers:
            logger.flush()
            logger.close()

        self.loggers = self._build_loggers(new_config)

    def step(self):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


########################################################################################################################
# Basic trainer
########################################################################################################################


class BasicTrainer(AbstractTrainer):
    """
    Trainer to use if one does not wish to use active learning
    """

    def setup(self, config):
        print("Initializing regular training pipeline")
        super(BasicTrainer, self).setup(config)
        # Train loader
        train_ddi_dataset = get_tensor_dataset(self.data, self.train_idxs)

        self.train_loader = DataLoader(
            train_ddi_dataset,
            batch_size=config["batch_size"],
            pin_memory=(self.device == "cpu"),
        )

    def step(self):
        train_metrics = self.train_epoch(
            self.data,
            self.train_loader,
            self.model,
            self.optim,
            n_forward_passes=self.n_train_forward_passes,
        )
        eval_metrics = self.eval_epoch(self.data, self.valid_loader, self.model)

        train_metrics = [("train_" + k, v) for k, v in train_metrics.items()]
        eval_metrics = [("eval_" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(train_metrics + eval_metrics)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0  # For compatibility with active trainer
        self.training_it += 1

        return metrics


########################################################################################################################
# Active Trainer
########################################################################################################################


class ActiveTrainer(AbstractTrainer):
    """
    Trainer class to perform active learning
    """

    def setup(self, config):
        print("Initializing active training pipeline")
        super(ActiveTrainer, self).setup(config)

        self.acquire_n_at_a_time = config["acquire_n_at_a_time"]
        self.max_examp_per_epoch = config["max_examp_per_epoch"]
        self.acquisition = config["acquisition"](config)
        self.n_scoring_forward_passes = config["n_scoring_forward_passes"]
        self.n_epoch_between_queries = config["n_epoch_between_queries"]

        # randomly acquire data at the beginning
        self.seen_idxs = self.train_idxs[: config["n_initial"]]
        self.unseen_idxs = self.train_idxs[config["n_initial"] :]

        # Initialize variable that saves the last query
        self.last_query_idxs = self.seen_idxs

        # Initialize dataloaders
        (
            self.seen_loader,
            self.unseen_loader,
            self.last_query_loader,
        ) = self.update_loaders(self.seen_idxs, self.unseen_idxs, self.last_query_idxs)

        # Get the set of top 1% most synergistic combinations
        one_perc = int(0.01 * len(self.unseen_idxs))
        scores = self.data.ddi_edge_response[self.unseen_idxs]
        self.top_one_perc = set(
            self.unseen_idxs[torch.argsort(scores, descending=True)[:one_perc]].numpy()
        )
        self.count = 0

    def step(self):
        # Check whether we have explored everything
        if len(self.unseen_loader) == 0:
            print("All space has been explored")
            return {"all_space_explored": 1, "training_iteration": self.training_it}

        # Evaluate on last query before training
        last_query_before_metric = self.eval_epoch(
            self.data,
            self.last_query_loader,
            self.model,
            message="Last query before training loss",
        )

        # Train on seen examples
        for _ in range(self.n_epoch_between_queries):
            # Perform several training epochs. Save only metrics from the last epoch
            seen_metrics = self.train_epoch(
                self.data,
                self.seen_loader,
                self.model,
                self.optim,
                self.max_examp_per_epoch,
                n_forward_passes=self.n_train_forward_passes,
            )

        # Evaluate on last query after training
        last_query_after_metric = self.eval_epoch(
            self.data,
            self.last_query_loader,
            self.model,
            message="Last query after training loss",
        )

        # Evaluate on valid set
        eval_metrics = self.eval_epoch(
            self.data, self.valid_loader, self.model, message="Eval loss"
        )

        # Score unseen examples
        unseen_metrics, active_scores = self.eval_epoch(
            self.data,
            self.unseen_loader,
            self.model,
            self.acquisition,
            self.n_scoring_forward_passes,
            message="Unseen loss",
        )

        # Build summary
        seen_metrics = [("seen_" + k, v) for k, v in seen_metrics.items()]
        last_query_before_metric = [
            ("last_query_before_tr_" + k, v)
            for k, v in last_query_before_metric.items()
        ]
        last_query_after_metric = [
            ("last_query_after_tr_" + k, v) for k, v in last_query_after_metric.items()
        ]
        unseen_metrics = [("unseen_" + k, v) for k, v in unseen_metrics.items()]
        eval_metrics = [("eval_" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(
            seen_metrics
            + unseen_metrics
            + eval_metrics
            + last_query_before_metric
            + last_query_after_metric
        )

        # Acquire new data
        print("query data")
        query = self.unseen_idxs[
            torch.argsort(active_scores, descending=True)[: self.acquire_n_at_a_time]
        ]
        # remove the query from the unseen examples
        self.unseen_idxs = self.unseen_idxs[
            torch.argsort(active_scores, descending=True)[self.acquire_n_at_a_time :]
        ]

        # Add the query to the seen examples
        self.seen_idxs = torch.cat((self.seen_idxs, query))
        metrics["seen_idxs"] = self.data.ddi_edge_idx[:, self.seen_idxs]

        # Compute proportion of top 1% synergistic drugs which have been discovered
        query_set = set(query.detach().numpy())
        self.count += len(query_set & self.top_one_perc)
        metrics["top"] = self.count / len(self.top_one_perc)

        self.last_query_idxs = query

        # Update the dataloaders
        (
            self.seen_loader,
            self.unseen_loader,
            self.last_query_loader,
        ) = self.update_loaders(self.seen_idxs, self.unseen_idxs, self.last_query_idxs)

        metrics["training_iteration"] = self.training_it
        metrics["all_space_explored"] = 0
        self.training_it += 1

        return metrics

    def update_loaders(self, seen_idxs, unseen_idxs, last_query_idxs):
        # Seen loader
        seen_ddi_dataset = get_tensor_dataset(self.data, seen_idxs)

        seen_loader = DataLoader(
            seen_ddi_dataset,
            batch_size=self.batch_size,
            pin_memory=(self.device == "cpu"),
        )

        # Unseen loader
        unseen_ddi_dataset = get_tensor_dataset(self.data, unseen_idxs)

        unseen_loader = DataLoader(
            unseen_ddi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cpu"),
        )

        # Last query loader
        last_query_ddi_dataset = get_tensor_dataset(self.data, last_query_idxs)

        last_query_loader = DataLoader(
            last_query_ddi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cpu"),
        )

        # Update acquisition function
        self.acquisition.update_with_seen(self.data.ddi_edge_response[seen_idxs])

        return seen_loader, unseen_loader, last_query_loader


def train(configuration):
    """
    Train with or without tune depending on configuration
    """
    if configuration["trainer_config"]["use_tune"]:
        ###########################################
        # Use tune
        ###########################################

        ray.init(num_cpus=20)

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
            num_samples=1,
            checkpoint_at_end=configuration["checkpoint_at_end"],
            local_dir=configuration["summaries_dir"],
            checkpoint_freq=configuration["checkpoint_freq"],
            scheduler=configuration["scheduler"],
            search_alg=configuration["search_alg"],
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
