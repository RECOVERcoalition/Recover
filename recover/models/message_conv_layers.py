from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch
from torch_sparse import SparseTensor, matmul

########################################################################################################################
# Convolution layers
########################################################################################################################


class ThreeMessageConvLayer(MessagePassing):
    def __init__(
        self,
        in_drug_channels,
        in_prot_channels,
        out_drug_channels,
        out_prot_channels,
        pass_d2p_msg,
        pass_p2d_msg,
        pass_p2p_msg,
        drug_self_loop,
        prot_self_loop,
        data,
    ):
        """
        Graph Convolution layer with 3 types of messages (drug-prot, prot-drug and prot-prot)

        :param in_drug_channels: dimension of the drug embeddings (input)
        :param in_prot_channels: dimension of the protein embeddings (input)
        :param out_drug_channels: dimension of the drug embeddings (output)
        :param out_prot_channels: dimension of the protein embeddings (output)
        :param pass_d2p_msg: Boolean, if True, messages drug->prot will be passed
        :param pass_p2d_msg: Boolean, if True, messages prot->drug will be passed
        :param pass_p2p_msg: Boolean, if True, messages prot->prot will be passed
        :param drug_self_loop: Boolean, if True, self loop will be computed for the drugs
        :param prot_self_loop: Boolean, if True, self loop will be computed for the proteins
        :param data: `torch_geometric.data.Data` object
        """

        super(ThreeMessageConvLayer, self).__init__(aggr="add")  # "Add" aggregation.

        # Define message passing matrices
        self.drug_to_prot_mat = torch.nn.Linear(in_drug_channels, out_prot_channels)
        self.prot_to_drug_mat = torch.nn.Linear(in_prot_channels, out_drug_channels)
        self.prot_to_prot_mat = torch.nn.Linear(in_prot_channels, out_prot_channels)

        # Define self loops matrices
        self.self_prot_loop = torch.nn.Linear(in_prot_channels, out_prot_channels)
        self.self_drug_loop = torch.nn.Linear(in_drug_channels, out_drug_channels)

        # Turn on/off self loops on proteins and drug nodes
        self.use_drug_self_loop = drug_self_loop
        self.use_prot_self_loop = prot_self_loop

        # Use these as a switch to turn on/off message passing between various
        # node types in the graph.
        self.pass_d2p_msg = int(pass_d2p_msg)
        self.pass_p2d_msg = int(pass_p2d_msg)
        self.pass_p2p_msg = int(pass_p2p_msg)

        self.num_drugs = data.x_drugs.shape[0]
        self.num_prots = data.x_prots.shape[0]

        # Compute normalization.
        row, col = torch.cat(
            (data.ppi_edge_idx, data.dpi_edge_idx, data.dpi_edge_idx[[1, 0], :]), dim=1
        )

        deg = degree(
            col, data.x_drugs.shape[0] + data.x_prots.shape[0], dtype=data.x_drugs.dtype
        )
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Norms for each edge type
        self.ppi_norm = norm[: data.ppi_edge_idx.shape[1]]
        self.dpi_norm = norm[
            data.ppi_edge_idx.shape[1] : data.ppi_edge_idx.shape[1]
            + data.dpi_edge_idx.shape[1]
        ]
        self.pdi_norm = norm[-data.dpi_edge_idx.shape[1] :]

        # Build sparse tensors (adjacency matrices) that will be used for sparse matrix multiplication
        self.sparse_pdi_idx = SparseTensor(
            row=data.dpi_edge_idx[0, :],
            col=data.dpi_edge_idx[1, :] - self.num_drugs,
            value=self.pdi_norm,
            sparse_sizes=(self.num_drugs, self.num_prots),
        )
        self.sparse_dpi_idx = SparseTensor(
            row=data.dpi_edge_idx[1, :] - self.num_drugs,
            col=data.dpi_edge_idx[0, :],
            value=self.dpi_norm,
            sparse_sizes=(self.num_prots, self.num_drugs),
        )
        self.sparse_ppi_idx = SparseTensor(
            row=data.ppi_edge_idx[0, :] - self.num_drugs,
            col=data.ppi_edge_idx[1, :] - self.num_drugs,
            value=self.ppi_norm,
            sparse_sizes=(self.num_prots, self.num_prots),
        )

    def forward(self, h_drug, h_prot, data):
        """

        :param h_drug: Embeddings of all drugs from the previous layer (num_drugs, dim_drug_embedding)
        :param h_prot: Embeddings of all proteins from the previous layer (num_drugs, dim_drug_embedding)
        :param data: `torch_geometric.data.Data` object
        :return: two tensors containing the embeddings of all drugs and all proteins in the next layer
        """

        # Compute messages
        d2p_msg = self.drug_to_prot_mat(h_drug) * self.pass_d2p_msg
        p2d_msg = self.prot_to_drug_mat(h_prot) * self.pass_p2d_msg
        p2p_msg = self.prot_to_prot_mat(h_prot) * self.pass_p2p_msg

        # Compute self loops
        self_drug_loop_msg = self.self_drug_loop(h_drug) * self.use_drug_self_loop
        self_prot_loop_msg = self.self_prot_loop(h_prot) * self.use_prot_self_loop

        # Propagating messages.
        drug_output = (
            self.propagate(self.sparse_pdi_idx, x=p2d_msg) + self_drug_loop_msg
        )

        if self.sparse_ppi_idx.numel() == 0:  # Check if there exists any PPI
            protein_output = (
                self.propagate(self.sparse_dpi_idx, x=d2p_msg) + self_prot_loop_msg
            )
        else:
            protein_output = (
                self.propagate(self.sparse_dpi_idx, x=d2p_msg)
                + self.propagate(self.sparse_ppi_idx, x=p2p_msg)
                + self_prot_loop_msg
            )

        return drug_output, protein_output

    def message_and_aggregate(self, adj_t, x):
        """
        Sparse matrix multiplication
        """
        return matmul(adj_t, x, reduce=self.aggr)
