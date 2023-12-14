import torch
import numpy as np
from scipy.special import erf

########################################################################################################################
# Abstract Acquisition
########################################################################################################################


class AbstractAcquisition:
    """
    Acquisition functions are used to assign a score to unseen examples, in order to decide which examples will be
    acquired next
    """

    def __init__(self, config):
        pass

    def get_scores(self, output):
        raise NotImplementedError

    def get_mean_and_std(self, output):
        output = torch.tensor(output)
        mean = output.mean(dim=1)
        std = output.std(dim=1)
        return mean, std

    def get_best(self, output):
        best, _ = output.max(dim=1)
        
        return best


########################################################################################################################
# Acquisition functions
########################################################################################################################

class ExpectedImprovementAcquisition(AbstractAcquisition):
    def __init__(self, config):
        super().__init__(config)
        
    def get_scores(self, output):
        mean, std = self.get_mean_and_std(output)
        best = self.get_best(output)
        epsilon = 1e-6
        
        z = (mean-best-epsilon)/(std+epsilon)
        phi = np.exp(-0.5*(z**2))/np.sqrt(2*np.pi)
        Phi = 0.5*(1+erf(z/np.sqrt(2)))
        scores = (mean-best)*Phi+std*phi

        return scores.to("cpu")


class RandomAcquisition(AbstractAcquisition):
    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        return torch.randn(output.shape[0])


class UCB(AbstractAcquisition):
    """
    Upper Confidence Bound. Exponentially decreasing kappa
    """

    def __init__(self, config):
        super().__init__(config)
        self.kappa = config["kappa"]
        self.decrease_factor = config["kappa_decrease_factor"]

        assert 0 < self.decrease_factor <= 1

    def get_scores(self, output):
        mean, std = self.get_mean_and_std(output)

        scores = mean + self.kappa * std

        self.kappa *= self.decrease_factor

        return scores.to("cpu")


class GreedyAcquisition(AbstractAcquisition):
    """
    Greedy, pure exploitation:
    The score is equal to the (expected) value of the objective for each example in the batch
    """

    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        mean, std = self.get_mean_and_std(output)

        scores = mean

        return scores.to("cpu")
