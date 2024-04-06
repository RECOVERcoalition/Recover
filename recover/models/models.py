import torch


########################################################################################################################
# Baselines with no GCN
########################################################################################################################


class Baseline(torch.nn.Module):
    def __init__(self, data, config):

        super(Baseline, self).__init__()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.criterion = torch.nn.MSELoss()

        # Compute dimension of input for predictor
        predictor_layers = [data.x_drugs.shape[1]] + config["predictor_layers"]

        assert predictor_layers[-1] == 1

        self.predictor = self.get_predictor(data, config, predictor_layers)
    
    def kl_loss(self):
        return self.predictor.kl_loss()

    def forward(self, data, drug_drug_batch):
        return self.predictor(data, drug_drug_batch)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](data, config, predictor_layers)

    def loss(self, output, drug_drug_batch):
        """
        Loss function for the synergy prediction pipeline
        :param output: output of the predictor
        :param drug_drug_batch: batch of drug-drug combination examples
        :return:
        """
        comb = output
        ground_truth_scores = drug_drug_batch[2][:, None]
        loss = self.criterion(comb, ground_truth_scores)

        return loss


########################################################################################################################
# Model wrappers: Ensemble and Predictive Uncertainty
########################################################################################################################


class EnsembleModel(torch.nn.Module):
    """
    Wrapper class that can handle an ensemble of models
    """

    def __init__(self, data, config):
        super(EnsembleModel, self).__init__()
        models = []
        self.ensemble_size = config["ensemble_size"]

        for _ in range(self.ensemble_size):
            models.append(Baseline(data, config))

        self.models = torch.nn.ModuleList(models)

    def forward(self, data, drug_drug_batch):
        comb_list = []
        for model_i in self.models:
            comb = model_i(data, drug_drug_batch)
            comb_list.append(comb)
        return torch.cat(comb_list, dim=1)

    def loss(self, output, drug_drug_batch):
        loss = 0
        for i in range(self.ensemble_size):
            output_i = output[:, i][:, None]
            loss_i = self.models[i].loss(output_i, drug_drug_batch)
            loss += loss_i

        loss /= self.ensemble_size
        return loss


class PredictiveUncertaintyModel(torch.nn.Module):
    """
    Wrapper class that can handle Predictive Uncertainty models
    """

    def __init__(self, data, config):
        super(PredictiveUncertaintyModel, self).__init__()

        self.mu_predictor = Baseline(data, config)
        self.std_predictor = Baseline(data, config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = data

    def forward(self, data, drug_drug_batch):
        """
        Used to get samples from the predictive distribution. The number of samples is set to 100.
        """
        mean_prediction = self.mu_predictor(data, drug_drug_batch)
        std_prediction = self.std_predictor(data, drug_drug_batch)

        return mean_prediction + \
               torch.randn((len(mean_prediction), 100)).to(self.device) * torch.exp(1/2*std_prediction)

    def loss(self, output, drug_drug_batch):
        """
        the mu_predictor model is trained using MSE while the std_predictor is trained using the adaptive NLL criterion
        """
        predicted_mean = self.mu_predictor(self.data, drug_drug_batch)
        ground_truth_scores = drug_drug_batch[2][:, None]
        predicted_log_sigma2 = self.std_predictor(self.data, drug_drug_batch)

        # Loss for the mu_predictor
        MSE = (predicted_mean - ground_truth_scores) ** 2

        # Loss for the std_predictor. We cap the exponent at 80 for stability
        denom = (2 * torch.exp(torch.min(predicted_log_sigma2, torch.tensor(80, dtype=torch.float32).to(self.device))))

        return torch.mean(MSE) + torch.mean(predicted_log_sigma2 / 2 + MSE.detach() / denom)
