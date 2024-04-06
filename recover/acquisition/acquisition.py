import torch
from scipy.stats import norm
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
        mean = output.mean(dim=1)
        std = output.std(dim=1)

        return mean, std

    def get_current_best(self, output):
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
        best = self.get_current_best(output)
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

class ProbabilityOfImprovementAcquisition(AbstractAcquisition):
    """
    Probability of Improvement Aquisition Function
    
    """
    def __init__(self, config):
        super().__init__(config)
        
    def get_scores(self, output):
        mean, std = self.get_mean_and_std(output)
        current_best = self.get_current_best(output)
        
        z = (mean - current_best) / std
        prob_of_improvement_scores = norm.cdf(z)

        return torch.tensor(prob_of_improvement_scores).to("cpu")

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

# class PI(AbstractAcquisition):
#     """
#     Probability of Improvement. Exponentially decreasing threshold
#     """

#     def __init__(self, config):
#         super().__init__(config)
#         self.threshold = config["threshold"]
#         self.decrease_factor = config["threshold_decrease_factor"]

#         assert 0 < self.decrease_factor <= 1

#     def get_scores(self, output):
#         mean, std = self.get_mean_and_std(output)

#         z = (mean - self.threshold) / std
#         scores = norm.cdf(z)

#         self.threshold *= self.decrease_factor

#         return scores.to("cpu")
