import torch
import numpy as np
from torch.nn import functional as F
from torch.nn import Parameter
from recover.models.utils import ResidualModule, LowRankAttention, raw_response_loss
from recover.utils import get_dropout_modules_recursive

########################################################################################################################
# Abstract Model
########################################################################################################################


class AbstractModel(torch.nn.Module):
    def __init__(self, data, config):
        """
        Abstract class for Models. Models take as input the features of all drugs and all proteins, and
        compute embeddings for the drugs.
        The embeddings of drugs are then fed two at a time to a predictor model
        :param data: `torch_geometric.data.Data` object
        :param config: configuration dictionary
        """
        self.device = config["device"]
        super(AbstractModel, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        raise NotImplementedError

    def enable_dropout(self):
        """
        Manually enable dropout modules of the Model. This is useful as we need to activate dropout at evaluation time
        """
        for m in get_dropout_modules_recursive(self):
            m.train()

    def loss(self, output, drug_drug_batch):
        """
        Loss function for the synergy prediction pipeline
        :param output: output of the predictor
        :param drug_drug_batch: batch of drug-drug combination examples
        :return:
        """

        comb = output[
            0
        ]  # We do not take into account the representation of single drugs

        ground_truth_scores = drug_drug_batch[2][:, None, None]
        ground_truth_scores = torch.cat([ground_truth_scores] * comb.shape[2], dim=2)

        loss = self.criterion(comb, ground_truth_scores)

        # Explained variance
        var = drug_drug_batch[2].var().item()
        r_squared = (var - loss.item()) / var

        return loss, r_squared, 0

    def enable_periodic_backprop(self):
        """
        Virtual methods to be overridden by implementations of this class if need be
        """
        pass

    def disable_periodic_backprop(self):
        pass


class DummyRawResponse(AbstractModel):
    def __init__(self, data, config):
        """
        Dummy model for the raw inhibition prediction pipeline:
        - The hypernetwork does not take into account any feature
        - The inhibition model that maps concentrations to inhibitions is learnt (regardless of drug ID and cell line)
        """
        super(DummyRawResponse, self).__init__(data, config)
        self.hl_dim = config["inhibition_model_hid_lay_dim"]
        self.inhibition_model_weight_decay = config["inhibition_model_weight_decay"]
        self.mono_loss_scale_factor = config["mono_loss_scale_factor"]
        self.combo_scale_factor = config["combo_loss_scale_factor"]
        self.model_synergy = config["model_synergy"]

        self.dummy_pred_net_comb_params = Parameter(
            1 / 100 * torch.randn((1, self.hl_dim * (self.hl_dim + 3) + 1, 1))
        )
        self.dummy_pred_net_mono_params = Parameter(
            1 / 100 * torch.randn((1, 4 * self.hl_dim + 1, 1))
        )

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        batch_size = drug_drug_batch[0].shape[0]

        comb = torch.cat([self.dummy_pred_net_comb_params] * batch_size, dim=0)
        comb = torch.cat([comb] * n_forward_passes, dim=2)

        mono = torch.cat([self.dummy_pred_net_mono_params] * batch_size, dim=0)
        mono = torch.cat([mono] * n_forward_passes, dim=2)

        return comb, mono, mono

    def loss(self, output, drug_drug_batch):
        return raw_response_loss(
            output,
            drug_drug_batch,
            self.hl_dim,
            self.criterion,
            self.mono_loss_scale_factor,
            self.combo_scale_factor,
            self.inhibition_model_weight_decay,
            self.model_synergy,
        )


########################################################################################################################
# Baselines with no GCN
########################################################################################################################


class Baseline(AbstractModel):
    def __init__(self, data, config):
        """
        Baseline Model (without GCN) for the synergy prediction pipeline.
        A predictor is defined separately (c.f. predictors.py) and is used to map drug embeddings to synergies
        """
        super(Baseline, self).__init__(data, config)
        config[
            "with_fp"
        ] = False  # Forcefully set to False as fps are used by default in this case
        config[
            "with_expr"
        ] = False  # Forcefully set to False as gene expr are used by default if available

        # Compute dimension of input for predictor
        predictor_layers = [data.x_drugs.shape[1]] + config["predictor_layers"]

        self.predictor = self.get_predictor(data, config, predictor_layers)

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        return self.predictor(
            data, drug_drug_batch, h_drug=None, n_forward_passes=n_forward_passes
        )

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers, uses_raw_response=False
        )


class RawResponseBaseline(Baseline):
    def __init__(self, data, config):
        """
        Baseline Model (without GCN) for the raw inhibition prediction pipeline.
        A predictor is defined separately (c.f. predictors.py) and is used to map drug embeddings to inhibitions
        """
        self.hl_dim = config["inhibition_model_hid_lay_dim"]
        self.inhibition_model_weight_decay = config["inhibition_model_weight_decay"]
        self.mono_loss_scale_factor = config["mono_loss_scale_factor"]
        self.combo_scale_factor = config["combo_loss_scale_factor"]
        self.model_synergy = config["model_synergy"]
        super(RawResponseBaseline, self).__init__(data, config)

    def loss(self, output, drug_drug_batch):
        """
        We overwrite the loss function for the raw inhibition pipeline.
        """
        return raw_response_loss(
            output,
            drug_drug_batch,
            self.hl_dim,
            self.criterion,
            self.mono_loss_scale_factor,
            self.combo_scale_factor,
            self.inhibition_model_weight_decay,
            self.model_synergy,
        )

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers, uses_raw_response=True
        )


########################################################################################################################
# Base Periodic Backprop Model
########################################################################################################################


class BasePeriodicBackpropModel(AbstractModel):
    def __init__(self, data, config):
        """
        Abstract model which can handle periodic backprop. It dynamically sets the `requires_grad` status of the model
        parameters, in order to control how often backprop is performed
        """
        super(BasePeriodicBackpropModel, self).__init__(data, config)

        self.do_periodic_backprop = (
            config["do_periodic_backprop"]
            if "do_periodic_backprop" in config
            else False
        )
        self.backprop_period = (
            config["backprop_period"] if "backprop_period" in config else None
        )
        self.periodic_backprop_enabled = (
            False  # True during training, False during Evaluation
        )
        self.curr_backprop_status = (
            False  # Whether backprop through all variables is enabled or not
        )
        self.backprop_iter = 0

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        if self.periodic_backprop_enabled:
            # Figure out whether we should backprop through all variables are not
            should_enable = self.backprop_iter % self.backprop_period == 0
            # Change the 'requires_grad' status of variables if necessary
            self.set_backprop_enabled_status(should_enable)
            self.backprop_iter += 1

        return self._forward(data, drug_drug_batch, n_forward_passes)

    def _forward(self, data, drug_drug_batch, n_forward_passes=1):
        raise NotImplementedError

    def set_backprop_enabled_status(self, status):
        """
        Changes the 'requires_grad' status of variables if necessary
        :param status: Boolean, whether we should backprop through periodic_backprop_vars
        """
        if status != self.curr_backprop_status:
            for var in self.get_periodic_backprop_vars():
                var.requires_grad = status

            self.curr_backprop_status = status

    def get_periodic_backprop_vars(self):
        """
        Iterator over the parameters that are subject to periodic backprop. Other parameters require grad all the time
        """
        raise NotImplementedError

    def enable_periodic_backprop(self):
        """
        Will be called when the model is set to training mode
        """
        assert self.backprop_period is not None
        self.periodic_backprop_enabled = self.do_periodic_backprop

    def disable_periodic_backprop(self):
        """
        Will be called when the model is set to eval mode
        """
        assert self.backprop_period is not None
        self.periodic_backprop_enabled = False


########################################################################################################################
# Giant Graph GCN
########################################################################################################################


class GiantGraphGCN(BasePeriodicBackpropModel):
    def __init__(self, data, config):
        """
        Graph Convolutional model for the synergy prediction pipeline. The graph is composed of drug nodes and protein
        nodes, and different types of messages are used for the different types of connections (drug->prot, prot->drug,
        prot<->prot)

        The GCN can include low rank global attention over drugs and/or proteins, self loops and residual connections.

        A predictor is defined separately (c.f. predictors.py) and is used to map drug embeddings (as computed by the
        GCN) to synergies.
        """
        super(GiantGraphGCN, self).__init__(data, config)

        self.use_prot_emb = (
            config["use_prot_emb"] if "use_prot_emb" in config else False
        )
        self.use_prot_feats = (
            config["use_prot_feats"] if "use_prot_feats" in config else False
        )

        # Default to using both embeddings and features if both options are false / missing
        if (not self.use_prot_emb) and (not self.use_prot_feats):
            print(
                f"NOTE: 'use_prot_emb' and 'use_prot_feats' are missing, using both embeddings and features"
            )
            self.use_prot_emb = True
            self.use_prot_feats = True

        self.prot_emb_dim = (
            self.use_prot_emb * config["prot_emb_dim"]
            + self.use_prot_feats * data.x_prots.shape[1]
        )

        # Learnable protein embeddings
        if self.use_prot_emb:
            self.prot_emb = Parameter(
                1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"]))
            )

        # First Graph convolution layer
        self.conv1 = config["conv_layer"](
            data.x_drugs.shape[1],
            self.prot_emb_dim,
            config["residual_layers_dim"],
            config["residual_layers_dim"],
            config["pass_d2p_msg"],
            config["pass_p2d_msg"],
            config["pass_p2p_msg"],
            config["drug_self_loop"],
            config["prot_self_loop"],
            data,
        )

        # First Low rank attention layer for drugs
        if "drug_attention" in config:
            self.has_drug_attention = config["drug_attention"]["attention"]
            self.drug_attention_conf = config["drug_attention"]
        else:
            self.has_drug_attention = False

        if self.has_drug_attention:
            self.low_rank_drug_attention = []
            self.low_rank_drug_attention.append(
                LowRankAttention(
                    k=self.drug_attention_conf["attention_rank"],
                    d=data.x_drugs.shape[1],
                    dropout=self.drug_attention_conf["dropout_proba"],
                )
            )

        # First Low rank attention layer for proteins
        if "prot_attention" in config:
            self.has_prot_attention = config["prot_attention"]["attention"]
            self.prot_attention_conf = config["prot_attention"]
        else:
            self.has_prot_attention = False

        if self.has_prot_attention:
            self.low_rank_prot_attention = []
            self.low_rank_prot_attention.append(
                LowRankAttention(
                    k=self.prot_attention_conf["attention_rank"],
                    d=self.prot_emb_dim,
                    dropout=self.prot_attention_conf["dropout_proba"],
                )
            )

        # Residual layers
        self.residual_layers = []
        drug_channels, prot_channels = (
            config["residual_layers_dim"],
            config["residual_layers_dim"],
        )

        for i in range(config["num_res_layers"]):
            if self.has_drug_attention:
                # If attention is used, we must increase the number of drug channels
                drug_channels += 2 * self.drug_attention_conf["attention_rank"]
                self.low_rank_drug_attention.append(
                    LowRankAttention(
                        k=self.drug_attention_conf["attention_rank"],
                        d=drug_channels,
                        dropout=self.drug_attention_conf["dropout_proba"],
                    )
                )

            if self.has_prot_attention:
                # If attention is used, we must increase the number of protein channels
                prot_channels += 2 * self.prot_attention_conf["attention_rank"]
                self.low_rank_prot_attention.append(
                    LowRankAttention(
                        k=self.prot_attention_conf["attention_rank"],
                        d=prot_channels,
                        dropout=self.prot_attention_conf["dropout_proba"],
                    )
                )

            self.residual_layers.append(
                ResidualModule(
                    ConvLayer=config["conv_layer"],
                    drug_channels=drug_channels,
                    prot_channels=prot_channels,
                    pass_d2p_msg=config["pass_d2p_msg"],
                    pass_p2d_msg=config["pass_p2d_msg"],
                    pass_p2p_msg=config["pass_p2p_msg"],
                    drug_self_loop=config["drug_self_loop"],
                    prot_self_loop=config["prot_self_loop"],
                    data=data,
                )
            )

        # Convert to ModuleList
        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_drug_attention:
            self.low_rank_drug_attention = torch.nn.ModuleList(
                self.low_rank_drug_attention
            )

        if self.has_prot_attention:
            self.low_rank_prot_attention = torch.nn.ModuleList(
                self.low_rank_prot_attention
            )

        # Compute dimension of input for predictor
        if self.has_drug_attention:
            predictor_layers = [
                drug_channels + self.drug_attention_conf["attention_rank"] * 2
            ]
        else:
            predictor_layers = [drug_channels]

        predictor_layers += config["predictor_layers"]

        # Predictor on top of GCN
        self.predictor = self.get_predictor(data, config, predictor_layers)

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers, uses_raw_response=False
        )

    def get_periodic_backprop_vars(self):
        """
        Iterator returning all the parameters of the GCN, but not the parameters of the predictor. We always backprop
        through the predictor, but backprop through the GCN only once every k forward passes
        """
        if self.use_prot_emb:
            yield self.prot_emb

        yield from self.conv1.parameters()
        yield from self.residual_layers.parameters()
        if self.has_drug_attention:
            yield from self.low_rank_drug_attention.parameters()
        if self.has_prot_attention:
            yield from self.low_rank_prot_attention.parameters()

    def _forward(self, data, drug_drug_batch, n_forward_passes=1):
        h_drug = data.x_drugs

        # Get protein features
        if self.use_prot_emb and self.use_prot_feats:
            h_prot = torch.cat((self.prot_emb, data.x_prots), dim=1)
        elif self.use_prot_emb:
            h_prot = self.prot_emb
        elif self.use_prot_feats:
            h_prot = data.x_prots

        ##########################################
        # GNN forward pass
        ##########################################

        # First layer
        h_drug_next, h_prot_next = self.conv1(h_drug, h_prot, data)
        if self.has_drug_attention:
            att = self.low_rank_drug_attention[0](h_drug)
            h_drug_next = torch.cat((h_drug_next, att), dim=1)

        if self.has_prot_attention:
            att = self.low_rank_prot_attention[0](h_prot)
            h_prot_next = torch.cat((h_prot_next, att), dim=1)

        h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # Residual layers
        for i in range(len(self.residual_layers)):
            h_drug_next, h_prot_next = self.residual_layers[i](h_drug, h_prot, data)
            if self.has_drug_attention:
                att = self.low_rank_drug_attention[i + 1](h_drug)
                h_drug_next = torch.cat((h_drug_next, att), dim=1)

            if self.has_prot_attention:
                att = self.low_rank_prot_attention[i + 1](h_prot)
                h_prot_next = torch.cat((h_prot_next, att), dim=1)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # Predict synergies with predictor
        return self.predictor(
            data, drug_drug_batch, h_drug, n_forward_passes=n_forward_passes
        )


class RawResponseGiantGraphGCN(GiantGraphGCN):
    def __init__(self, data, config):
        """
        GCN model for the raw inhibition prediction pipeline.
        A predictor is defined separately (c.f. predictors.py) and is used to map drug embeddings to inhibitions
        """
        self.hl_dim = config["inhibition_model_hid_lay_dim"]
        self.inhibition_model_weight_decay = config["inhibition_model_weight_decay"]
        self.mono_loss_scale_factor = config["mono_loss_scale_factor"]
        self.combo_scale_factor = config["combo_loss_scale_factor"]
        self.model_synergy = config["model_synergy"]
        super(RawResponseGiantGraphGCN, self).__init__(data, config)

    def loss(self, output, drug_drug_batch):
        """
        We overwrite the loss function for the raw inhibition pipeline.
        """
        return raw_response_loss(
            output,
            drug_drug_batch,
            self.hl_dim,
            self.criterion,
            self.mono_loss_scale_factor,
            self.combo_scale_factor,
            self.inhibition_model_weight_decay,
            self.model_synergy,
        )

    def get_predictor(self, data, config, predictor_layers):
        return config["predictor"](
            data, config, predictor_layers, uses_raw_response=True
        )


########################################################################################################################
# Ensemble
########################################################################################################################


class EnsembleModel(AbstractModel):
    """
    Wrapper class that can handle an ensemble of models
    """

    def __init__(self, data, config):
        super(EnsembleModel, self).__init__(data, config)

        models = []
        for _ in range(config["ensemble_size"]):
            models.append(config["model"](data, config))

        self.models = torch.nn.ModuleList(models)

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        out = []
        for m in self.models:
            out.append(m(data, drug_drug_batch, n_forward_passes))
        return torch.cat(out, dim=1)
