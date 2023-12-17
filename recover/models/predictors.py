import torch
import numpy as np
import torch.nn as nn
import torchbnn as bnn
from torch.nn import Parameter
import math
import torch.nn.functional as F
########################################################################################################################
# Modules
########################################################################################################################


class LinearModule(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearModule, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class ReLUModule(nn.ReLU):
    def __init__(self):
        super(ReLUModule, self).__init__()

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]
        
class ScaledSigmoid(nn.Module):
    def __init__(self, scale_factor=100):
        super(ScaledSigmoid, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if isinstance(x, list):
            # Apply sigmoid to each tensor in the list
            return [self.scale_factor * F.sigmoid(tensor) for tensor in x]
        else:
            # Apply sigmoid to a single tensor
            return self.scale_factor * F.sigmoid(x)



class ScaleMixtureGaussian(object): #scale mixture Gaussian
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        if self.pi == 1:
            return torch.log(prob1).sum()
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
        


PI =  0.25
SIGMA_1 = torch.FloatTensor([math.exp(-0)]) #torch.FloatTensor([0.005]) #
SIGMA_2 = torch.FloatTensor([math.exp(-5)])

class BayesianLinearModule(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features) #BayesianLinearModule, self
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        #self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
     
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def sample_bias(self):
        epsilon = torch.distributions.Normal(0,1).sample(self.bias_rho.size())
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return self.bias_mu + sigma * epsilon

    def sample_weight(self):
        epsilon = torch.distributions.Normal(0,1).sample(self.weight_rho.size())
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return self.weight_mu + sigma * epsilon
    
    def log_prob_bias(self, input):
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.bias_mu) ** 2) / (2 * sigma ** 2)).sum()

    def log_prob_weight(self, input):
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.weight_mu) ** 2) / (2 * sigma ** 2)).sum() 

    
    def forward(self, input, sample=True):
        x, cell_line = input[0], input[1] #BAYESIAN ADD ON
        
        #new lines
        weight_mu = self.weight_mu
        weight_rho = self.weight_rho
        bias_mu = self.bias_mu
        bias_rho = self.bias_rho
        
        weight = self.sample_weight()
        bias = self.sample_bias()


        if self.training and sample:
            
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.log_prob_weight(weight) + self.log_prob_bias(bias)
          
        else:
            self.log_prior, self.log_variational_posterior = torch.FloatTensor([0]), torch.FloatTensor([0])

        return [F.linear(x, weight, bias), cell_line]
        
    
    def kl_loss(self):
        return self.log_variational_posterior - self.log_prior

# class BayesianLineardropoutModule(nn.Linear): 
#     def __init__(self, in_features, out_features, input_dropout=0.8, hidden_dropout=0.5):
#         super().__init__(in_features, out_features) #BayesianLinearModule, self
#         self.in_features = in_features
#         self.out_features = out_features
#         self.input_dropout = input_dropout
#         self.hidden_dropout = hidden_dropout
#         # Weight parameters
#         self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
#         self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
#         #self.weight = Gaussian(self.weight_mu, self.weight_rho)
#         # Bias parameters
#         self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
#         self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
     
#         # Prior distributions
#         self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
#         self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
#         self.log_prior = 0
#         self.log_variational_posterior = 0

#         # Dropout layers
#         self.input_dropout_layer = nn.Dropout(p=self.input_dropout)
#         self.hidden_dropout_layer = nn.Dropout(p=self.hidden_dropout)

#     def sample_bias(self):
#         epsilon = torch.distributions.Normal(0,1).sample(self.bias_rho.size())
#         sigma = torch.log1p(torch.exp(self.bias_rho))
#         return self.bias_mu + sigma * epsilon

#     def sample_weight(self):
#         epsilon = torch.distributions.Normal(0,1).sample(self.weight_rho.size())
#         sigma = torch.log1p(torch.exp(self.weight_rho))
#         return self.weight_mu + sigma * epsilon
    
#     def log_prob_bias(self, input):
#         sigma = torch.log1p(torch.exp(self.bias_rho))
#         return (-math.log(math.sqrt(2 * math.pi))
#                 - torch.log(sigma)
#                 - ((input - self.bias_mu) ** 2) / (2 * sigma ** 2)).sum()

#     def log_prob_weight(self, input):
#         sigma = torch.log1p(torch.exp(self.weight_rho))
#         return (-math.log(math.sqrt(2 * math.pi))
#                 - torch.log(sigma)
#                 - ((input - self.weight_mu) ** 2) / (2 * sigma ** 2)).sum() 

    
#     def forward(self, input, sample=True):
#         x, cell_line = input[0], input[1] #BAYESIAN ADD ON
#         x = self.input_dropout_layer(x)
        
#         #new lines
#         weight_mu = self.weight_mu
#         weight_rho = self.weight_rho
#         bias_mu = self.bias_mu
#         bias_rho = self.bias_rho
        
#         weight = self.sample_weight()
#         bias = self.sample_bias()


#         if self.training and sample:
            
#             self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
#             self.log_variational_posterior = self.log_prob_weight(weight) + self.log_prob_bias(bias)
          
#         else:
#             self.log_prior, self.log_variational_posterior = torch.FloatTensor([0]), torch.FloatTensor([0])
        
#         # Apply hidden layer dropout
#         x = self.hidden_dropout_layer(x)

#         return [F.linear(x, weight, bias), cell_line]
        
    
#     def kl_loss(self):
#         return self.log_variational_posterior - self.log_prior

#Sparse Variational Dropout(SVDO) on Bayesian Nueral Network
class BayesianLinearDropoutModule(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features) #BayesianLinearModule, self
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
     
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def sample_bias(self):
        epsilon = torch.distributions.Normal(0,1).sample(self.bias_rho.size())
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return self.bias_mu + sigma * epsilon

    def sample_weight(self):
        epsilon = torch.distributions.Normal(0,1).sample(self.weight_rho.size())
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return self.weight_mu + sigma * epsilon
    
    def log_prob_bias(self, input):
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.bias_mu) ** 2) / (2 * sigma ** 2)).sum()

    def log_prob_weight(self, input):
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.weight_mu) ** 2) / (2 * sigma ** 2)).sum() 

    
    def forward(self, input, sample=True):
        x, cell_line = input[0], input[1] #BAYESIAN ADD ON
        
        #new lines
        weight_mu = self.weight_mu
        weight_rho = self.weight_rho
        bias_mu = self.bias_mu
        bias_rho = self.bias_rho
        
        weight = self.sample_weight()
        bias = self.sample_bias()


        if self.training and sample:
            # Apply SVDO during training
            # Sample weights using the reparameterization trick
            epsilon_weight = torch.distributions.Normal(0, 1).sample(self.weight_rho.size())
            sigma_weight = torch.log1p(torch.exp(self.weight_rho))
            weight = weight_mu + sigma_weight * epsilon_weight

            epsilon_bias = torch.distributions.Normal(0, 1).sample(self.bias_rho.size())
            sigma_bias = torch.log1p(torch.exp(self.bias_rho))
            bias = bias_mu + sigma_bias * epsilon_bias

            # Compute log probabilities for KL loss
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = (
                self.log_prob_weight(weight) + self.log_prob_bias(bias)
            )
        else:
            # During evaluation or when sample=False, use expected values
            weight = weight_mu
            bias = bias_mu
            self.log_prior, self.log_variational_posterior = (
                torch.FloatTensor([0]),
                torch.FloatTensor([0]),
            )
        return [F.linear(x, weight, bias), cell_line]
        
    
    def kl_loss(self):
        return self.log_variational_posterior - self.log_prior
########################################################################################################################
# Advanced Bayesian MLP
########################################################################################################################

class AdvancedBayesianBilinearMLPPredictor(nn.Module): #BAYESIAN ADD ON

    def __init__(self, data, config, predictor_layers):
        super().__init__()
        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.layer_dims = predictor_layers
        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]
        
        # self.layers = BayesianLinearModule(in_features, out_features)

        
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)
        layers_before_merge = []
        layers_after_merge = []
        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            # layers_before_merge = self.add_bayesian_layer(
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )
        # Build last layers (after addition of the two embeddings)
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_bayesian_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )
        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        # self.before_merge_mlp = layers_before_merge
        # self.after_merge_mlp = layers_after_merge

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))
        self.allow_neg_eigval = config["allow_neg_eigval"]
        if self.allow_neg_eigval:
            self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)

    def forward(self, data, drug_drug_batch, sample=True):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)
        # Apply before merge MLP

        # Instead of
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]
        # # We do
        # input_1 = [h_drug_1, cell_lines]
        # for layer in self.before_merge_mlp:
        #     # if layer.__class__.__name__ == 'BayesianLinearModule':
        #     #     input_1 = layer(input_1, sample=False)
        #     # else:
        #     #     input_1 = layer(input_1)
        #     input_1 = layer(input_1)

        # input_2 = [h_drug_2, cell_lines]
        # for layer in self.before_merge_mlp:
        #     # if layer.__class__.__name__ == 'BayesianLinearModule':
        #     #     input_2 = layer(input_2, sample=False)
        #     # else:
        #     #     input_2 = layer(input_2)
        #     input_2 = layer(input_2)

        # h_1 = input_1[0]
        # h_2 = input_2[0]

        # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T
        if self.allow_neg_eigval:
            # Multiply by diagonal matrix to allow for negative eigenvalues
            h_2 *= self.bilinear_diag
        # "Transpose" h_1
        h_1 = h_1.permute(0, 2, 1)
        # Multiplication
        h_1_scal_h_2 = (h_1 * h_2).sum(1)
        # Add offset
        h_1_scal_h_2 += self.bilinear_offsets

        # Instead of
        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]
        # We do
        # input_3 = [h_1_scal_h_2, cell_lines]
        # for layer in self.after_merge_mlp:
        #     input_3 = layer(input_3)#, sample=True)
         
        # comb = input_3[0]

        return comb
    
    def get_batch(self, data, drug_drug_batch):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]
        return h_drug_1, h_drug_2, cell_lines
    
    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
        return layers

    def add_bayesian_layer(self, layers, i, dim_i, dim_i_plus_1):
        # layers.extend(self.bayesian_linear_layer(i, mu, sigma, dim_i, dim_i_plus_1))
        # bayesian_linear_layer = [BayesianLinearModule(dim_i, dim_i_plus_1)]

        #layers.extend(self.bayesian_linear_layer(i, dim_i, dim_i_plus_1))
        layers += self.bayesian_linear_layer(i, dim_i, dim_i_plus_1)
        # layers.append(BayesianLinearModule(dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
        else:
            # layers.append(nn.Sigmoid() * 100)
            layers.append(ScaledSigmoid(scale_factor=100))
        return layers

    def bayesian_linear_layer(self, i, dim_i, dim_i_plus_1):
        return [BayesianLinearDropoutModule(dim_i, dim_i_plus_1)]
        # return [BayesianLinearModule(dim_i, dim_i_plus_1)]

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]

    def kl_loss(self): 
        kl = 0
        for layer in self.before_merge_mlp:
            if hasattr(layer, "kl_loss"):
                kl += layer.kl_loss()
        for layer in self.after_merge_mlp:
            if hasattr(layer, "kl_loss"):
                kl += layer.kl_loss()
        return kl
    
