import math
import torch
from torch.nn.functional import relu
from recover.models.utils import get_predicted_weights, predict_inhibition
import numpy as np


########################################################################################################################
# Custom objective
########################################################################################################################


def custom_objective(comb, h_1, h_2, hl_dim, model_synergy):
    """
    Custom objective that we can design in case we use raw response data
    Example: evaluate HSA on a logscale between 0.1 uM and 10 uM on a 7x7 grid

    :param comb: Embedding of the combination of drugs (output of the predictor)
        shape (batch_size, hl_dim * (hl_dim + 3) + 1, num_forward_passes)
    :param h_1: Embedding of drug 1 (output of the predictor)
        shape (batch_size, 4 * hl_dim + 1, num_forward_passes)
    :param h_2: Embedding of drug 2 (output of the predictor)
        shape (batch_size, 4 * hl_dim + 1, num_forward_passes)
    :param hl_dim: dimension of the hidden layers in the predicted network
    :return: Tensor containing the synergy scores
        shape (batch_size, num_forward_passes)
    """

    with torch.no_grad():
        # Get the weights for the inhibtion network
        predicted_weights = get_predicted_weights(comb, h_1, h_2, hl_dim)

        # Build Tensor of concentration pairs on which the inhibition network will be evaluated
        conc_pairs = torch.Tensor(
            [
                [c_1, c_2]
                for c_1 in np.linspace(np.log(0.1), np.log(10), 7)
                for c_2 in np.linspace(np.log(0.1), np.log(10), 7)
            ]
        )

        conc_pairs = torch.cat([conc_pairs[None, None, :, :]] * comb.shape[0], dim=0)

        # Build Tensor of row concentrations on which the inhibition network will be evaluated
        conc_r = torch.Tensor(
            [
                [np.log(1e-6), c_2]
                for c_1 in np.linspace(np.log(0.1), np.log(10), 7)
                for c_2 in np.linspace(np.log(0.1), np.log(10), 7)
            ]
        )

        conc_r = torch.cat([conc_r[None, None, :, :]] * comb.shape[0], dim=0)

        # Build Tensor of column concentrations on which the inhibition network will be evaluated
        conc_c = torch.Tensor(
            [
                [c_1, np.log(1e-6)]
                for c_1 in np.linspace(np.log(0.1), np.log(10), 7)
                for c_2 in np.linspace(np.log(0.1), np.log(10), 7)
            ]
        )

        conc_c = torch.cat([conc_c[None, None, :, :]] * comb.shape[0], dim=0)

        # Forward in the inhition network
        comb_inhib = predict_inhibition(
            conc_pairs, *predicted_weights, model_synergy
        ).numpy()
        r_inhib = predict_inhibition(conc_r, *predicted_weights, model_synergy).numpy()
        c_inhib = predict_inhibition(conc_c, *predicted_weights, model_synergy).numpy()

        # Get average HSA
        HSA = (comb_inhib - np.maximum(r_inhib, c_inhib)) / (7 * 7)

    return torch.Tensor(HSA.sum(axis=2))


########################################################################################################################
# Abstract Acquisition
########################################################################################################################


class AbstractAcquisition:
    """
    Acquisition functions are used to assign a score to unseen examples, in order to decide which examples will be
    acquired next
    """

    def __init__(self, config):
        # The following two attributes are seful only if using raw response pipeline
        self.hl_dim = config["inhibition_model_hid_lay_dim"]
        self.model_synergy = config["model_synergy"]

    def get_scores(self, output):
        raise NotImplementedError

    def update_with_seen(self, seen_labels):
        raise NotImplementedError

    def get_objective_to_maximize(self, output):
        """
        Return the value of the objective that we want to maximize for each of the examples in the batch

        :param output: output of the predictor
        :return: Tensor containing the value of the objective (e.g. synergy score) for each example in the batch
            shape (batch_size, num_forward_passes)
        """

        comb, h_1, h_2 = output

        if comb.shape[1] == 1:  # We are using the synergy pipeline
            # The output is the synergy score to maximize already
            return comb[:, 0, :]
        else:  # We are using the raw response pipeline
            # The output is the set of parameters of the inhibition network, we need to compute the objective
            return custom_objective(comb, h_1, h_2, self.hl_dim, self.model_synergy)


########################################################################################################################
# Acquisition functions
########################################################################################################################


class ExpectedImprovement(AbstractAcquisition):
    """
    Expected Improvement
    """

    def __init__(self, config):
        super().__init__(config)
        self.s_max = 0.0

    def get_scores(self, output):
        obj = self.get_objective_to_maximize(output)
        scores = relu(obj - self.s_max)
        return scores.sum(dim=1).to("cpu")

    def update_with_seen(self, seen_labels):
        # Retrieve the best score seen so far
        self.s_max = max(self.s_max, torch.max(seen_labels).item())


class RandomAcquisition(AbstractAcquisition):
    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        return torch.randn(output.shape[0])

    def update_with_seen(self, seen_labels):
        pass


class GPUCB(AbstractAcquisition):
    """
    Gaussian Process Upper Confidence Bound. Exponentially decreasing kappa
    """

    def __init__(self, config):
        super().__init__(config)
        self.kappa = config["kappa"]
        self.decrease_factor = config["kappa_decrease_factor"]

        assert 0 < self.decrease_factor <= 1

    def get_scores(self, output):
        obj = self.get_objective_to_maximize(output)
        mean = obj.mean(dim=1)
        std = obj.std(dim=1)

        scores = mean + self.kappa * std

        self.kappa *= self.decrease_factor

        return scores.to("cpu")

    def update_with_seen(self, seen_labels):
        pass


class GreedyAcquisition(AbstractAcquisition):
    """
    Greedy, pure exploitation:
    The score is equal to the (expected) value of the objective for each example in the batch
    """

    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        obj = self.get_objective_to_maximize(output)
        scores = obj.mean(dim=1).to("cpu")

        return scores

    def update_with_seen(self, seen_labels):
        pass


class Thompson(AbstractAcquisition):
    """
    Thompson sampling. For each example in the batch, we sample at random one forward pass,
    and the score is equal to the value of the objective (e.g. synergy) predicted by this forward pass
    When using this acquisition function, it is better to use:
    - small batch size
    - high n_scoring_forward_passes
    as the scores within a batch are correlated (same dropout configurations were used)
    """

    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        obj = self.get_objective_to_maximize(output).to("cpu")
        # Select one sample at random for each example in the batch
        idx = torch.randint(high=obj.shape[1], size=(obj.shape[0],))[:, None]
        scores = obj.gather(1, idx)[:, 0]

        return scores

    def update_with_seen(self, seen_labels):
        pass
