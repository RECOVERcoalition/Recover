import torch
from torch.nn import functional as F
import numpy as np
from torch.nn import Parameter
import torch.nn as nn


########################################################################################################################
# Modules which can handle cell line for predictors
########################################################################################################################

"""
These modules forward a list [activation, cell_line] instead of only activations. This allows the 
list of cell line indices of the batch to be used by any of the modules of the predictor network.
"""


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


class DropoutModule(nn.Dropout):
    def __init__(self, p):
        super(DropoutModule, self).__init__(p)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class FilmModule(torch.nn.Module):
    def __init__(self, num_cell_lines, out_dim):
        super(FilmModule, self).__init__()
        film_init = 1 / 100 * torch.randn(num_cell_lines, 2 * out_dim)
        film_init = film_init + torch.Tensor([([1] * out_dim) + ([0] * out_dim)])

        self.film = Parameter(film_init)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [
            self.film[cell_line][:, : x.shape[1]] * x
            + self.film[cell_line][:, x.shape[1] :],
            cell_line,
        ]


class CellLineSpecificLinearModule(nn.Module):
    def __init__(self, in_features, out_features, num_cell_lines):
        super(CellLineSpecificLinearModule, self).__init__()

        self.cell_line_matrices = Parameter(
            1 / 100 * torch.randn((num_cell_lines, out_features, in_features))
        )
        self.cell_line_offsets = Parameter(
            1 / 100 * torch.randn((num_cell_lines, out_features, 1))
        )

    def forward(self, input):
        x, cell_line = input[0], input[1]

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1)

        x = (
            self.cell_line_matrices[cell_line].matmul(x)
            + self.cell_line_offsets[cell_line]
        )
        return [x[:, :, 0], cell_line]


########################################################################################################################
# Modules for GCN
########################################################################################################################


class ResidualModule(torch.nn.Module):
    """
    Module composed of two graph convolution layers with a residual connection.
    """

    def __init__(
        self,
        ConvLayer,
        drug_channels,
        prot_channels,
        pass_d2p_msg,
        pass_p2d_msg,
        pass_p2p_msg,
        drug_self_loop,
        prot_self_loop,
        data,
    ):

        super(ResidualModule, self).__init__()
        self.conv1 = ConvLayer(
            drug_channels,
            prot_channels,
            drug_channels,
            prot_channels,
            pass_d2p_msg,
            pass_p2d_msg,
            pass_p2p_msg,
            drug_self_loop,
            prot_self_loop,
            data,
        )

        self.conv2 = ConvLayer(
            drug_channels,
            prot_channels,
            drug_channels,
            prot_channels,
            pass_d2p_msg,
            pass_p2d_msg,
            pass_p2p_msg,
            drug_self_loop,
            prot_self_loop,
            data,
        )

    def forward(self, h_drug, h_prot, data):
        out_drug, out_prot = self.conv1(h_drug, h_prot, data)
        out_drug = F.relu(out_drug)
        out_prot = F.relu(out_prot)
        out_drug, out_prot = self.conv2(out_drug, out_prot, data)

        return F.relu(h_drug + out_drug), F.relu(h_prot + out_prot)


class LowRankAttention(torch.nn.Module):
    """
    Low Rank Global Attention. see https://arxiv.org/pdf/2006.07846.pdf
    """

    def __init__(self, k, d, dropout):
        """
        :param k: rank of the attention matrix
        :param d: dimension of the embeddings on which attention is performed
        :param dropout: probability of dropout
        """
        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 4 * k), torch.nn.ReLU())
        self.activation = torch.nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        U = tmp[:, : self.k]
        V = tmp[:, self.k : 2 * self.k]
        Z = tmp[:, 2 * self.k : 3 * self.k]
        T = tmp[:, 3 * self.k :]
        V_T = torch.t(V)
        # normalization
        D = joint_normalize2(U, V_T)
        res = torch.mm(U, torch.mm(V_T, Z))
        res = torch.cat((res * D, T), dim=1)
        return self.dropout(res)


def joint_normalize2(U, V_T):
    # U and V_T are in block diagonal form
    if torch.cuda.is_available():
        tmp_ones = torch.ones((V_T.shape[1], 1)).to("cuda")
    else:
        tmp_ones = torch.ones((V_T.shape[1], 1))
    norm_factor = torch.mm(U, torch.mm(V_T, tmp_ones))
    norm_factor = (torch.sum(norm_factor) / U.shape[0]) + 1e-6
    return 1 / norm_factor


def weight_init(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight.data)
        # nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
    return


########################################################################################################################
# Functions for Predictors
########################################################################################################################


def get_batch(
    data,
    drug_drug_batch,
    h_drug,
    drug2target_dict,
    with_fp=False,
    with_expr=False,
    with_prot=False,
):
    """
    Function that generates the batch to be fed to the predictor.
    If required, additional drug features are concatenated with the drug embeddings computed by the GCN.

    if using the baseline, h_drug is None, the fingerprints (and gene expressions if available) are taken as embeddings
    of the drugs, and the parameters with_fp and with_expr are ignored.

    The protein information, if included, is a one hot encoding of drug targets, with dimension <number of proteins>.
    """

    batch_size = drug_drug_batch[0].shape[0]

    drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
    drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
    cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

    #####################################################
    # Get drug embeddings
    #####################################################

    if h_drug is not None:
        batch_data_1 = h_drug[drug_1s]  # Embeddings of tail drugs in the batch
        batch_data_2 = h_drug[drug_2s]  # Embeddings of head drugs in the batch
    else:  # The embedding of the drug is the fingerprint + eventually gene expression
        batch_data_1 = data.x_drugs[drug_1s]
        batch_data_2 = data.x_drugs[drug_2s]

    #####################################################
    # Include fingerprints
    #####################################################

    if (
        with_fp and h_drug is not None
    ):  # Concatenate the embeddings with fingerprints and IC50s
        x_drug_1s = data.x_drugs[drug_1s, : data.fp_bits]
        x_drug_2s = data.x_drugs[drug_2s, : data.fp_bits]
        batch_data_1 = torch.cat((batch_data_1, x_drug_1s), dim=1)
        batch_data_2 = torch.cat((batch_data_2, x_drug_2s), dim=1)

    #####################################################
    # Get expression data
    #####################################################

    if with_expr and h_drug is not None:  # Include gene expression data
        expr_drug_1s = data.x_drugs[drug_1s, data.fp_bits :]
        expr_drug_2s = data.x_drugs[drug_2s, data.fp_bits :]
        batch_data_1 = torch.cat((batch_data_1, expr_drug_1s), dim=1)
        batch_data_2 = torch.cat((batch_data_2, expr_drug_2s), dim=1)

    #####################################################
    # Add protein target information
    #####################################################

    if with_prot:  # Include protein target information
        prot_1 = torch.zeros((batch_size, data.x_prots.shape[0]))
        prot_2 = torch.zeros((batch_size, data.x_prots.shape[0]))

        if torch.cuda.is_available():
            prot_1 = prot_1.to("cuda")
            prot_2 = prot_2.to("cuda")

        for i in range(batch_size):
            prot_1[i, drug2target_dict[int(drug_1s[i])]] = 1
            prot_2[i, drug2target_dict[int(drug_2s[i])]] = 1

        batch_data_1 = torch.cat((batch_data_1, prot_1), dim=1)
        batch_data_2 = torch.cat((batch_data_2, prot_2), dim=1)

    return batch_data_1, batch_data_2, cell_lines


def get_layer_dims(
    predictor_layers,
    fp_dim,
    expr_dim,
    prot_numb,
    with_fp=False,
    with_expr=False,
    with_prot=False,
):
    """
    Updates the dimension of the input for the predictor, depending on which features are to be fed
    """
    if with_expr:
        predictor_layers[0] += expr_dim
    if with_fp:
        predictor_layers[0] += fp_dim
    if with_prot:
        predictor_layers[0] += prot_numb
    return predictor_layers


########################################################################################################################
# Functions for Raw Response models
########################################################################################################################


def raw_response_loss(
    output,
    drug_drug_batch,
    hl_dim,
    criterion,
    mono_scale_factor,
    combo_scale_factor,
    weight_decay,
    model_synergy,
):
    """
    Computes the loss of the model for the raw response pipeline (when predicting inhibitions)

    For each example in the batch, an inhibtion model is instanciated with the weights predited by the hypernetwork,
    and this model is used to predict monotherapy and combination predictions from concentration pairs.

    :param output: parameters of the inhibition network, as predicted by the predictor hypernetwork
    :param drug_drug_batch:
    :param hl_dim: dimension of the hidden layers of the inhibition network
    :param criterion: Mean square error
    :param mono_scale_factor: set to zero to ignore the loss related to monotherapy prediction
    :param combo_scale_factor: set to zero to ignore the loss related to combination inhibition prediction
    :param weight_decay: multiplication factor for the L2 regularization applied on the weights of inhibition model
    :param model_synergy: Boolean. If False, predictions for the combinations will be equal to the expected response as
    defined by the BLISS score

    :return: value of the loss
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comb, h_1, h_2 = output

    n_samples = comb.shape[-1]

    # Get the weights of the inhibtion model (one network per example in the batch)
    predicted_weights = get_predicted_weights(comb, h_1, h_2, hl_dim)

    ###########################################
    # Loss on combo
    ###########################################

    batch_ddi_edge_conc_pair = drug_drug_batch[3]
    batch_ddi_edge_inhibitions = drug_drug_batch[4]

    # Make predictions using the predicted network
    predicted_inhibitions = predict_inhibition(
        batch_ddi_edge_conc_pair[:, None, :, :], *predicted_weights, model_synergy
    )

    ground_truth_scores = batch_ddi_edge_inhibitions[:, None, :]
    ground_truth_scores = torch.cat([ground_truth_scores] * n_samples, dim=1)

    # Prediction loss for combination inhibitions
    combo_loss = criterion(predicted_inhibitions, ground_truth_scores)

    # Explained variance (only on drug combinations)
    comb_var = batch_ddi_edge_inhibitions.var().item()
    comb_r_squared = (comb_var - combo_loss.item()) / comb_var

    ###########################################
    # Loss on monotherapies
    ###########################################

    # Build tensor of concentrations that will be fed to the inhibition model for monotherapy inhibition prediction
    batch_ddi_edge_conc_r = torch.cat(
        (
            drug_drug_batch[5][:, None, :, None],
            np.log(1e-6) * torch.ones((comb.shape[0], 1, 4, 1)).to(device),
        ),
        dim=3,
    )
    batch_ddi_edge_conc_c = torch.cat(
        (
            np.log(1e-6) * torch.ones((comb.shape[0], 1, 4, 1)).to(device),
            drug_drug_batch[6][:, None, :, None],
        ),
        dim=3,
    )
    batch_ddi_edge_inhibition_r = drug_drug_batch[7]
    batch_ddi_edge_inhibition_c = drug_drug_batch[8]

    # Make predictions using the predicted network
    predicted_inhibition_r = predict_inhibition(
        batch_ddi_edge_conc_r, *predicted_weights, model_synergy
    )[:, 0, :]

    predicted_inhibition_c = predict_inhibition(
        batch_ddi_edge_conc_c, *predicted_weights, model_synergy
    )[:, 0, :]

    # Prediction loss for monotherapy inhibitions
    mono_loss = criterion(
        predicted_inhibition_r, batch_ddi_edge_inhibition_r
    ) + criterion(predicted_inhibition_c, batch_ddi_edge_inhibition_c)

    # Explained variance (only on monotherapies)
    mono_var = (
        batch_ddi_edge_inhibition_r.var().item()
        + batch_ddi_edge_inhibition_r.var().item()
    )
    mono_r_squared = (mono_var - mono_loss.item()) / mono_var

    return (
        combo_scale_factor * combo_loss
        + mono_scale_factor * mono_loss
        + weight_decay * torch.norm(comb) ** 2,
        comb_r_squared,
        mono_r_squared,
    )


def get_predicted_weights(comb, h_1, h_2, hl_dim):
    """
    Rearranges the output of the predictor (hypernetwork) for compatibility with the inhibition model

    :param comb: parameters which depend on both drugs
    :param h_1: parameters which depend on drug 1
    :param h_2: parameters which depend on drug 2
    :param hl_dim: dimension of the hidden layers

    :return: h_1_params, h_2_params, combo_params: parameters (in the form of 3 lists of weight matrices and biases)
        for the three components of the inhibition model
    """

    batch_size = comb.shape[0]
    n_samples = comb.shape[-1]

    def get_monotherapy_params(h, hl_dim):
        """
        :return: weight matrices and biases for the 2 layers of the monotherapy predictor part of the inhibition model
        """
        W_h_1 = (
            h[:, :hl_dim, :]
            .reshape((batch_size, 1, hl_dim, n_samples))
            .permute(0, 3, 1, 2)
        )

        b_h_1 = (
            h[:, hl_dim : 2 * hl_dim, :]
            .reshape((batch_size, 1, hl_dim, n_samples))
            .permute(0, 3, 1, 2)
        )

        W_h_2 = (
            h[:, 2 * hl_dim : 3 * hl_dim, :]
            .reshape((batch_size, 1, hl_dim, n_samples))
            .permute(0, 3, 2, 1)
        )
        b_h_2 = (
            h[:, 3 * hl_dim, :]
            .reshape((batch_size, 1, 1, n_samples))
            .permute(0, 3, 1, 2)
        )

        return W_h_1, b_h_1, W_h_2, b_h_2

    ###########################################
    # For monotherapy prediction of drugs
    ###########################################

    h_1_params = get_monotherapy_params(h_1, hl_dim)
    h_2_params = get_monotherapy_params(h_2, hl_dim)

    ###########################################
    # For combo prediction
    ###########################################

    W_1 = (
        torch.cat(
            (
                h_1[:, 3 * hl_dim + 1 : 4 * hl_dim + 1, :],
                h_2[:, 3 * hl_dim + 1 : 4 * hl_dim + 1, :],
            ),
            dim=1,
        )
        .reshape((batch_size, 2, hl_dim, n_samples))
        .permute(0, 3, 1, 2)
    )

    b_1 = (
        comb[:, :hl_dim, :]
        .reshape((batch_size, 1, hl_dim, n_samples))
        .permute(0, 3, 1, 2)
    )

    W_2 = (
        comb[:, hl_dim : hl_dim * (hl_dim + 1), :]
        .reshape((batch_size, hl_dim, hl_dim, n_samples))
        .permute(0, 3, 1, 2)
    )
    b_2 = (
        comb[:, hl_dim * (hl_dim + 1) : hl_dim * (hl_dim + 2), :]
        .reshape((batch_size, 1, hl_dim, n_samples))
        .permute(0, 3, 1, 2)
    )
    W_3 = (
        comb[:, hl_dim * (hl_dim + 2) : hl_dim * (hl_dim + 3), :]
        .reshape((batch_size, 1, hl_dim, n_samples))
        .permute(0, 3, 2, 1)
    )
    b_3 = (
        comb[:, hl_dim * (hl_dim + 3), :]
        .reshape((batch_size, 1, 1, n_samples))
        .permute(0, 3, 1, 2)
    )

    combo_params = (W_1, b_1, W_2, b_2, W_3, b_3)

    return h_1_params, h_2_params, combo_params


def predict_inhibition(x, h_1_params, h_2_params, combo_params, model_synergy):
    """
    Inhibition model that predicts an inhibition given a pair of concentrations

    The predictions are computed as follows:

    normalization factors: Norm(c) = (sigmoid(c, IC50=exp(-11)) - 1/2)
    The normalization factor ensures that the inhibition is zero when the concentration is zero (corresponds to the way
    inhibitions have been normalized)

    # Single drug responses:
    R_1 = Norm(c_1) * Single_NN((c_1, 0), h_1_params)
    R_2 = Norm(c_2) * Single_NN((0, c_2), h_2_params)

    where Single_NN is a 1 hidden layer neural network predicting inhibition from concentrations

    Synergy = Norm(c_1)*Norm(c_2) * Comb_NN(c_1, c_2, h_1_params, h_2_params)

    where Comb_NN is a 2 hidden layer neural network predicting an inhibition from a pair of concentrations. It has the
    following invariance property:
        Comb_NN(c_1, c_2, h_1_params, h_2_params) = Comb_NN(c_2, c_1, h_2_params, h_1_params)


    :return: if model_synergy: R_1 + R_2 - R_1*R_2 + Synergy

    """

    def sigmoid(c, ic50, slope, y_max):
        # y_min fixed to zero because data is normalized
        y = y_max / (1 + torch.exp(-slope * (c - ic50)))
        return y

    def predict_mono(c, W_h_1, b_h_1, W_h_2, b_h_2):
        # Layer 1
        h = c.matmul(W_h_1)
        h += b_h_1
        h = F.relu(h)

        # Layer 2
        h = h.matmul(W_h_2)
        h += b_h_2

        return h[:, :, :, 0]

    R_1 = sigmoid(x[:, :, :, 0], -11, 1, 1) * predict_mono(x[:, :, :, :1], *h_1_params)
    R_2 = sigmoid(x[:, :, :, 1], -11, 1, 1) * predict_mono(x[:, :, :, 1:], *h_2_params)

    if model_synergy:
        # Synergy
        W_1, b_1, W_2, b_2, W_3, b_3 = combo_params

        # Layer 1
        h = x.matmul(W_1)
        h += b_1
        h = F.relu(h)

        # Layer 2
        h = h.matmul(W_2)
        h += b_2

        # Layer 2
        h = h.matmul(W_3)
        h += b_3

        synergy = (
            sigmoid(x[:, :, :, 0], -11, 1, 1)
            * sigmoid(x[:, :, :, 1], -11, 1, 1)
            * h[:, :, :, 0]
        )

        return R_1 + R_2 - (R_1 * R_2) / 100 + synergy
    else:
        return R_1 + R_2 - (R_1 * R_2) / 100
