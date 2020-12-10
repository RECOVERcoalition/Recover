import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
from recover.models.utils import (
    get_batch,
    get_layer_dims,
    ReLUModule,
    DropoutModule,
    FilmModule,
    LinearModule,
    CellLineSpecificLinearModule,
)


########################################################################################################################
# Abstract Predictor
########################################################################################################################


class AbstractPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers, uses_raw_response):
        """
        Abstract class for the predictor models. Predictors take as input the embeddings of two drugs and compute the
        synergy or inhibitions of the combination.

        If predicting inhibitions (raw response pipeline), this Predictor is a hypernetwork that predicts the weights
        of a smaller network (the "inhibition model") which maps a pair of concentrations to an inhibition.

        :param data: `torch_geometric.data.Data` object
        :param config: configuration dictionary
        :param predictor_layers: list of dimensions defining the architecture of the predictor
        :param uses_raw_response: Boolean, False if using the synergy prediction pipeline, True otherwise
        """
        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
        self.device = config["device"]
        self.with_fp = config["with_fp"]
        self.with_expr = config["with_expr"]
        self.with_prot = config["with_prot"]

        self.uses_raw_response = uses_raw_response
        if self.uses_raw_response:
            # Get the dimension of the hidden layers of the inhibition model
            self.hl_dim = config["inhibition_model_hid_lay_dim"]

        self.layer_dims = predictor_layers

        # Compute the dimension of the input and the output
        (
            self.layer_dims,
            self.output_dim_comb,
            self.output_dim_mono,
        ) = self.get_layer_dims(
            self.layer_dims,
            fp_dim=int(data.fp_bits),
            expr_dim=data.x_drugs.shape[1] - int(data.fp_bits),
            prot_numb=data.x_prots.shape[0],
        )

        # Create dictionary linking drug to targets
        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(
                int(edge[1]) - data.x_drugs.shape[0]
            )

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        super(AbstractPredictor, self).__init__()

    def forward(self, data, drug_drug_batch, h_drug, n_forward_passes=1):
        """
        :param data: `torch_geometric.data.Data` object
        :param drug_drug_batch: batch of drug-drug combination examples
        :param h_drug: embeddings of all drugs
        :param n_forward_passes: Number of forward passes (with different dropout configurations) to perform

        :return: comb, h_1 and h_2. If using synergy pipeline, h_1 and h_2 are not used. If using the inhibition
        pipeline, h_<x> contains the parameters of the inhibition model that only depend on drug <x>, and comb contains
        the parameters that depend on both drugs
        """
        h_drug_1s, h_drug_2s, cell_lines = self.get_batch(data, drug_drug_batch, h_drug)

        # Initialize tensors
        comb = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_comb, 0]).to(
            self.device
        )
        h_1 = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_mono, 0]).to(
            self.device
        )
        h_2 = torch.empty([drug_drug_batch[0].shape[0], self.output_dim_mono, 0]).to(
            self.device
        )

        # Perform several forward passes for MC dropout
        for i in range(n_forward_passes):
            comb_i, h_1_i, h_2_i = self.single_forward_pass(
                h_drug_1s, h_drug_2s, cell_lines
            )

            comb = torch.cat((comb, comb_i[:, :, None]), dim=2)
            if h_1_i is not None:
                h_1 = torch.cat((h_1, h_1_i[:, :, None]), dim=2)
                h_2 = torch.cat((h_2, h_2_i[:, :, None]), dim=2)

        return comb, h_1, h_2

    def get_layer_dims(self, predictor_layers, fp_dim, expr_dim, prot_numb):
        """
        Compute the dimension of the input (in case we concatenate drug embeddings computed by the GCN with other drug
        features) and dimension of the output (output dim = 1 if using synergy pipeline, number of
        parameters in the inhibition network otherwise)
        :param predictor_layers: list of dimensions that define the architecture of the predictor
        :param fp_dim: dimension of the fingerprints
        :param expr_dim: dimension of the differential gene expression features (for the drugs)
        :param prot_numb: Number of proteins

        :return: a list of length 3:
            1 - list of layer dimensions
            2 - dimension of the output for the combination part of the inhibtion model
            3 - dimension of the output for the monotherapy part of the inhibtion model
        """
        if not self.uses_raw_response:
            return (
                get_layer_dims(
                    predictor_layers,
                    fp_dim,
                    expr_dim,
                    prot_numb,
                    with_fp=self.with_fp,
                    with_expr=self.with_expr,
                    with_prot=self.with_prot,
                ),
                1,
                1,
            )
        else:
            return (
                get_layer_dims(
                    predictor_layers,
                    fp_dim,
                    expr_dim,
                    prot_numb,
                    with_fp=self.with_fp,
                    with_expr=self.with_expr,
                    with_prot=self.with_prot,
                ),
                self.hl_dim * (self.hl_dim + 3) + 1,
                4 * self.hl_dim + 1,
            )

    def get_batch(self, data, drug_drug_batch, h_drug):
        return get_batch(
            data,
            drug_drug_batch,
            h_drug,
            self.drug2target_dict,
            with_fp=self.with_fp,
            with_expr=self.with_expr,
            with_prot=self.with_prot,
        )

    def single_forward_pass(self, h_drug_1s, h_drug_2s, cell_lines):
        raise NotImplementedError


########################################################################################################################
# MLP Abstract Predictors
########################################################################################################################


class MLPAbstractPredictor(AbstractPredictor):
    """
    Abstract MLP predictor, invariant per permutation of the two drugs. Drug embeddings are computed independently with
    before_merge_mlp, then the embeddings are summed and fed into after_merge_mlp.

    Different types of conditioning (on cell line) are available (no conditioning, Film, Cell line specific layers)
    Does not take into account cell line.
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(MLPAbstractPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.layer_dims[
            -1
        ] = (
            self.output_dim_comb
        )  # Change dimension of the output for the combination MLP

        # Build last layers (after addition of the two embeddings)
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1],
                config,
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        if self.uses_raw_response:
            # If using raw response data, we initialize single drug predictors as well
            layers_single_drug = []
            self.layer_dims[
                -1
            ] = (
                self.output_dim_mono
            )  # Change dimension of the output for the monotherapy MLP

            # Build single drug layers
            for i in range(len(self.layer_dims) - 1):
                layers_single_drug = self.add_layer(
                    layers_single_drug,
                    i,
                    self.layer_dims[i],
                    self.layer_dims[i + 1],
                    config,
                )

            self.single_drug_mlp = nn.Sequential(*layers_single_drug)
        else:
            # If using the synergy pipeline, we do not need single drug MLP
            self.single_drug_mlp = None

    def add_layer(self, layers, i, dim_i, dim_i_plus_1, config):
        """
        Adds a linear layer (as defined by the self.linear_layer method) to the list of modules, and optionally dropout
        depending on the configuration

        :param layers: list of torch Modules
        :param i: index of the layer to be added
        :param dim_i: dimension before the layer
        :param dim_i_plus_1: dimension after
        :param config: configuration file
        :return: extended list of torch Modules
        """
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

            should_do_dropout = (
                "num_dropout_lyrs" not in config
                or len(self.layer_dims) - 2 - config["num_dropout_lyrs"] <= i
            )

            if should_do_dropout:
                layers.append(DropoutModule(p=config["dropout_proba"]))

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        """
        Defines the type of linear layers the predictor will be composed of (Shared layers, FilmComditioned layers, ...)
        """
        raise NotImplementedError

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):
        comb = self.after_merge_mlp(
            [
                self.before_merge_mlp([h_drug_1, cell_lines])[0]
                + self.before_merge_mlp([h_drug_2, cell_lines])[0],
                cell_lines,
            ]
        )[0]
        return (
            comb,
            self.transform_single_drug(h_drug_1, cell_lines),
            self.transform_single_drug(h_drug_2, cell_lines),
        )

    def transform_single_drug(self, h, cell_lines):
        """
        Sets the embbedding of the single drug to the right dimension
        """
        if self.single_drug_mlp is None:
            return None
        else:
            return self.single_drug_mlp([h, cell_lines])[0]


class BilinearMLPAbstractPredictor(MLPAbstractPredictor):
    """
    Similar to the MLP Abstract Predictor but applies a bilinear transform instead of addition.
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(BilinearMLPAbstractPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

    def single_forward_pass(self, h_drug_1, h_drug_2, cell_lines):

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        # "Transpose" h_1
        h_1 = h_1.permute(0, 2, 1)

        # Multiplication
        h_1_scal_h_2 = (h_1 * h_2).sum(1)

        # Add offset
        h_1_scal_h_2 += self.bilinear_offsets

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return (
            comb,
            self.transform_single_drug(h_drug_1, cell_lines),
            self.transform_single_drug(h_drug_2, cell_lines),
        )


########################################################################################################################
# Addition Predictors
########################################################################################################################


class BasicMLPPredictor(MLPAbstractPredictor):
    """
    Addition MLP. Does not take into account cell line.
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(BasicMLPPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


class FilmMLPPredictor(MLPAbstractPredictor):
    """
    Addition MLP. Takes into account cell line via film conditioning
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(FilmMLPPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i != len(self.layer_dims) - 2:
            return [
                LinearModule(dim_i, dim_i_plus_1),
                FilmModule(self.num_cell_lines, dim_i_plus_1),
            ]
        else:
            return [LinearModule(self.layer_dims[i], self.layer_dims[i + 1])]


class SharedLayersMLPPredictor(MLPAbstractPredictor):
    """
    Addition MLP. Takes into account cell line via cell line specific layer (for the last two layers).
    Shared (between cell lines) layers are also used
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(SharedLayersMLPPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i < len(self.layer_dims) - 3:
            return [LinearModule(dim_i, dim_i_plus_1)]
        else:
            return [
                CellLineSpecificLinearModule(dim_i, dim_i_plus_1, self.num_cell_lines)
            ]


########################################################################################################################
# Bilinear Predictors
########################################################################################################################


class BilinearBasicMLPPredictor(BilinearMLPAbstractPredictor):
    """
    Bilinear MLP. Does not take into account cell line.
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(BilinearBasicMLPPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


class BilinearFilmMLPPredictor(BilinearMLPAbstractPredictor):
    """
    Bilinear MLP. Takes into account cell line via film conditioning
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(BilinearFilmMLPPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i != len(self.layer_dims) - 2:
            return [
                LinearModule(dim_i, dim_i_plus_1),
                FilmModule(self.num_cell_lines, dim_i_plus_1),
            ]
        else:
            return [LinearModule(dim_i, dim_i_plus_1)]


class BilinearSharedLayersMLPPredictor(BilinearMLPAbstractPredictor):
    """
    Bilinear MLP. Takes into account cell line via cell line specific layer (for the last two layers).
    Shared (between cell lines) layers are also used
    """

    def __init__(self, data, config, predictor_layers, uses_raw_response):
        super(BilinearSharedLayersMLPPredictor, self).__init__(
            data, config, predictor_layers, uses_raw_response
        )

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        if i < len(self.layer_dims) - 3:
            return [LinearModule(dim_i, dim_i_plus_1)]
        else:
            return [
                CellLineSpecificLinearModule(dim_i, dim_i_plus_1, self.num_cell_lines)
            ]
