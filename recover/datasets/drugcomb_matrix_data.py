from torch_geometric.data import Data, InMemoryDataset, download_url
from recover.datasets.l1000_data import L1000
import numpy as np
import pandas as pd
import torch
import os
import random
import reservoir as rsv
from recover.utils import get_project_root, get_fingerprint
from recover.datasets.utils import (
    process_l1000,
    process_data,
    get_ddi_edges,
    get_dpi_edges,
    get_ppi_edges,
    get_drug_nodes,
    get_protein_nodes,
    fit_sigmoid_on_monotherapies,
    drug_level_random_split,
    pair_level_random_split,
)
import zipfile


class DrugCombMatrix(InMemoryDataset):
    def __init__(
        self,
        transform=None,
        pre_transform=None,
        fp_bits=1024,
        fp_radius=4,
        ppi_graph="huri.csv",
        dti_graph="chembl_dtis.csv",
        cell_line=None,
        use_l1000=False,
        restrict_to_l1000_covered_drugs=False,
    ):
        """
        Dataset object for the pipeline. It creates a `torch_geometric.data.Data` object containing various attributes,
        including:
            - x_drugs: tensor of drug features (fingerprints, differential expression profiles)
                Shape (n_drugs, n_drug_features)
            - x_prots: tensor of protein features (derived from protein sequence)
            - ppi_edge_idx: list of protein-protein interaction edges.
                Shape (2, num_ppi_edges)
            - dpi_edge_idx: list of drug-protein interaction edges.
                Shape (2, num_dpi_edges)
            - data.ddi_edge_idx: list of pairwise drug combinations available in the DrugComb dataset.
                Shape (2, num_ddi_edges)
            - ddi_edge_<score>: tensor containing synergy score for pairs of drugs
            - ddi_edge_inhibitions: list of raw inhibition matrices for pairwise drug combinations. The associated pairs
                of concentrations are stored in ddi_edge_conc_pair.
            - ddi_edge_inhibition_<r, c>: list of monotherapy inihitions for the row/column drug of a given pairwise
                combination. The associated concentrations are stored in ddi_edge_conc_<r, c>


        :param transform: A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        :param pre_transform: A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        :param fp_bits: number of bits for fingerprints
        :param fp_radius: radius for fingerprints
        :param ppi_graph: Name of the database to use for protein-protein interactions
        :param dti_graph: Name of the database to use for drug-target interactions
        :param cell_line: Name of the cell line to include (e.g. 'K-562'). If None, all cell lines are included
        :param use_l1000: If True, features from the L1000 dataset will used. Each drug will be associated with two
            differencial gene expression profiles acquired on cell lines PC3 and MCF7
        :param restrict_to_l1000_covered_drugs: If True, only the drugs covered in both L1000 and DrugComb will be
            .included
        """

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.ppi_graph = ppi_graph
        self.dti_graph = dti_graph
        self.use_l1000 = use_l1000
        self.restrict_to_l1000_covered_drugs = restrict_to_l1000_covered_drugs

        assert self.dti_graph in rsv.available_dtis()
        assert self.ppi_graph in rsv.available_ppis()

        super().__init__(
            os.path.join(get_project_root(), "DrugComb/"), transform, pre_transform
        )

        # Load processed dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

        # If the parameters do not correspond to the processed dataset, reprocess dataset
        if (
            self.data.fp_bits != self.fp_bits
            or self.data.fp_radius != self.fp_radius
            or self.data.ppi_graph[0] != self.ppi_graph
            or self.data.dti_graph[0] != self.dti_graph
            or self.data.use_l1000 != self.use_l1000
            or self.data.restrict_to_l1000_covered_drugs
            != self.restrict_to_l1000_covered_drugs
        ):
            self.data, self.slices = 0, 0
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

        # Select a specific cell line is cell_line is not None
        if cell_line is not None:
            assert cell_line in self.data.cell_line_to_idx_dict[0].keys()
            ixs = (
                self.data.ddi_edge_classes
                == self.data.cell_line_to_idx_dict[0][cell_line]
            )
            for attr in dir(self.data):
                if attr.startswith("ddi_edge_idx"):
                    self.data[attr] = self.data[attr][:, ixs]
                elif attr.startswith("ddi_edge_"):
                    self.data[attr] = self.data[attr][ixs]

        if self.transform is not None:
            self.data = self.transform(self.data)

        ##################################################################
        # Update cell line index in case some cell lines have been removed
        ##################################################################

        new_unique_cell_lines = self.data.ddi_edge_classes.unique()
        old_idx_to_new_idx = {
            int(new_unique_cell_lines[i]): i for i in range(len(new_unique_cell_lines))
        }
        # Update cell_line_to_idxs dictionary
        self.data.cell_line_to_idx_dict[0] = {
            k: old_idx_to_new_idx[self.data.cell_line_to_idx_dict[0][k]]
            for k in self.data.cell_line_to_idx_dict[0].keys()
            if self.data.cell_line_to_idx_dict[0][k] in old_idx_to_new_idx.keys()
        }
        # Update ddi_edge_classes
        self.data.ddi_edge_classes = self.data.ddi_edge_classes.apply_(
            lambda x: old_idx_to_new_idx[x]
        )

        ##################################################################
        # Log transform concentrations
        ##################################################################

        self.data.ddi_edge_conc_pair = torch.log(self.data.ddi_edge_conc_pair)
        self.data.ddi_edge_conc_r = torch.log(self.data.ddi_edge_conc_r + 1e-6)
        self.data.ddi_edge_conc_c = torch.log(self.data.ddi_edge_conc_c + 1e-6)

        print("Dataset loaded.")
        print(
            "\t",
            self.data.ddi_edge_idx.shape[1],
            "drug comb experiments among",
            self.data.x_drugs.shape[0],
            "drugs",
        )
        print("\t fingeprints with radius", self.fp_radius, "and nbits", self.fp_bits)
        print("\t", self.data.dpi_edge_idx.shape[1], "drug target interactions")
        print("\t", int(self.data.ppi_edge_idx.shape[1] / 2), "prot-prot interactions")
        print("\t", "drug features dimension", self.data.x_drugs.shape[1])
        print("\t", "protein features dimension", self.data.x_prots.shape[1])
        print("\t", len(set(self.data.ddi_edge_classes.numpy())), "cell-lines")
        if self.use_l1000:
            print("\t", "using differential gene expression")
        else:
            print("\t", "not using differential gene expression")

    @property
    def raw_file_names(self):
        return ["uniref50_v2/options.json", "uniref50_v2/weights.hdf5"]

    def download(self):
        download_url("https://rostlab.org/~deepppi/seqvec.zip", self.raw_dir)

        with zipfile.ZipFile(
            os.path.join(get_project_root(), "DrugComb/raw/seqvec.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(self.raw_dir)

    @property
    def processed_file_names(self):
        return ["drugcomb_matrix_data.pt"]

    def process(self):
        """
        Processs the raw files to compute and save processed files

        We restrict ourselves to pairwise combinations for which we have access to a 3x3 inhibition matrix,
        along with 4 monotherapy inhibtions for each of the drugs. (only Almanac data)

        This leaves us with 255049 examples corresponding to 4374 unique combos.

        Note: only 4 cell lines in common between L1000 and
            blocks with combo_measurements=9, mono_row_measurements=4, mono_col_measurements=4
            Restricting ourselves to these 4 cell lines would leave us with only 17k examples in drugcomb
            (instead of 255k)
        """

        ##################################################################
        # Load DrugComb dataframe
        ##################################################################

        blocks = rsv.get_specific_drug_combo_blocks(
            combo_measurements=9, mono_row_measurements=4, mono_col_measurements=4
        )

        mono_data = rsv.get_drug_combo_data_monos(block_ids=blocks["block_id"])
        combo_data = rsv.get_drug_combo_data_combos(block_ids=blocks["block_id"])

        data_df = pd.concat(
            (
                combo_data,
                mono_data[["conc_r", "inhibition_r", "conc_c", "inhibition_c"]],
            ),
            axis=1,
        )

        ##################################################################
        # Load L1000 dataframe
        ##################################################################

        if os.path.isfile(os.path.join(self.raw_dir, "processed_l1000.csv")):
            # If the L1000 dataset has already been processed, load it
            l1000_df = pd.read_csv(
                os.path.join(self.raw_dir, "processed_l1000.csv"), index_col=0
            ).astype(np.float32)
        else:
            # Load L1000
            l1000 = L1000()

            # Load metadata
            sig_info, landmark_gene_list = l1000.load_metadata()
            expr_data = pd.concat(
                [l1000.load_expr_data("phase1"), l1000.load_expr_data("phase2")],
                sort=False,
            )

            # Concat with metadata
            l1000_df = pd.concat((expr_data, sig_info), axis=1)
            # Process L1000 dataset
            l1000_df = process_l1000(l1000_df)

            l1000_df.to_csv(os.path.join(self.raw_dir, "processed_l1000.csv"))

        ##################################################################
        # Get node features and edges
        ##################################################################

        # Get nodes features
        drug_nodes, protein_nodes, rec_id_to_idx_dict, is_drug = self._get_nodes(
            data_df, l1000_df
        )

        # Process data_df
        data_df, cell_lines = process_data(data_df, rec_id_to_idx_dict)

        # Build edges
        # PPI edges are both ways
        ppi_edge_idx = self._get_ppi_edges(rec_id_to_idx_dict)
        # DPI and DDI edges are one way only
        dpi_edge_idx = self._get_dpi_edges(rec_id_to_idx_dict)

        # DDI edges
        (
            ddi_edge_idx,
            ddi_edge_classes,
            ddi_edge_css,
            ddi_edge_zip,
            ddi_edge_bliss,
            ddi_edge_loewe,
            ddi_edge_hsa,
            ddi_edge_conc_pair,
            ddi_edge_inhibitions,
            ddi_edge_conc_r,
            ddi_edge_conc_c,
            ddi_edge_inhibition_r,
            ddi_edge_inhibition_c,
            cell_line_to_idx_dict,
        ) = self._get_ddi_edges(data_df)

        # Fit sigmoids on monotherapy data and retieve fitted parameters
        ddi_edge_sig_params_r = fit_sigmoid_on_monotherapies(
            ddi_edge_conc_r,
            ddi_edge_inhibition_r,
            save_file="mono_sigm_params_r_"
            + str(self.restrict_to_l1000_covered_drugs)
            + ".npy",
        )
        ddi_edge_sig_params_c = fit_sigmoid_on_monotherapies(
            ddi_edge_conc_c,
            ddi_edge_inhibition_c,
            save_file="mono_sigm_params_c_"
            + str(self.restrict_to_l1000_covered_drugs)
            + ".npy",
        )

        ##################################################################
        # Create data object
        ##################################################################

        data = Data(
            x_drugs=torch.tensor(drug_nodes.to_numpy(), dtype=torch.float),
            x_prots=torch.tensor(protein_nodes.to_numpy(), dtype=torch.float),
        )

        # Add ppi attributes to data
        data.ppi_edge_idx = torch.tensor(ppi_edge_idx, dtype=torch.long)
        # Add dpi attributes to data
        data.dpi_edge_idx = torch.tensor(dpi_edge_idx, dtype=torch.long)
        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.long)
        data.cell_line_to_idx_dict = cell_line_to_idx_dict

        # Scores
        data.ddi_edge_css = torch.tensor(ddi_edge_css, dtype=torch.float)
        data.ddi_edge_zip = torch.tensor(ddi_edge_zip, dtype=torch.float)
        data.ddi_edge_bliss = torch.tensor(ddi_edge_bliss, dtype=torch.float)
        data.ddi_edge_loewe = torch.tensor(ddi_edge_loewe, dtype=torch.float)
        data.ddi_edge_hsa = torch.tensor(ddi_edge_hsa, dtype=torch.float)

        # Raw data (combo)
        data.ddi_edge_conc_pair = torch.tensor(ddi_edge_conc_pair, dtype=torch.float)
        data.ddi_edge_inhibitions = torch.tensor(
            ddi_edge_inhibitions, dtype=torch.float
        )

        # Raw data (monotherapy)
        data.ddi_edge_conc_r = torch.tensor(ddi_edge_conc_r, dtype=torch.float)
        data.ddi_edge_conc_c = torch.tensor(ddi_edge_conc_c, dtype=torch.float)
        data.ddi_edge_inhibition_r = torch.tensor(
            ddi_edge_inhibition_r, dtype=torch.float
        )
        data.ddi_edge_inhibition_c = torch.tensor(
            ddi_edge_inhibition_c, dtype=torch.float
        )

        # Fitted params of sigmoids on monotherapy data
        data.ddi_edge_sig_params_r = torch.tensor(
            ddi_edge_sig_params_r, dtype=torch.float
        )
        data.ddi_edge_sig_params_c = torch.tensor(
            ddi_edge_sig_params_c, dtype=torch.float
        )

        # Add attributes to data
        data.fp_radius = self.fp_radius
        data.fp_bits = self.fp_bits
        data.ppi_graph = self.ppi_graph
        data.dti_graph = self.dti_graph
        data.use_l1000 = self.use_l1000
        data.restrict_to_l1000_covered_drugs = self.restrict_to_l1000_covered_drugs

        data.is_drug = torch.tensor(is_drug, dtype=torch.long)

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_nodes(self, data_df, l1000_df):
        """
        :param data_df: dataframe for the DrugComb dataset
        :param l1000_df: dataframe for the (processed) L1000 dataset
        :return:
            - dataframe of drug features
            - dataframe of protein features
            - dictionary mapping RECOVER_ID to index of the node in the graph
            - 1D numpy binary numpy array indicating which nodes are drugs (and which nodes are proteins)
        """
        # Drug nodes are duplicated (e.g. if several names of the same drug appear in drugcomb)
        drug_nodes = get_drug_nodes(
            data_df,
            l1000_df,
            self.dti_graph,
            self.use_l1000,
            self.restrict_to_l1000_covered_drugs,
        )
        protein_nodes = get_protein_nodes(self.raw_dir)

        ################################################################################################################
        # Combine drug nodes and protein nodes together
        ################################################################################################################

        all_nodes = pd.concat((drug_nodes, protein_nodes), sort=False)
        all_nodes["recover_id"] = all_nodes.index
        all_nodes.reset_index(drop=True, inplace=True)

        # Build dictionary recover ID -> node index
        rec_id_to_idx_dict = {all_nodes.at[i, "recover_id"]: i for i in all_nodes.index}
        is_drug = all_nodes[["is_drug"]].to_numpy()

        # Computing fingerprints
        idx_to_fp_dict = {
            idx: get_fingerprint(
                all_nodes.at[idx, "smiles"], self.fp_radius, self.fp_bits
            )
            for idx in all_nodes[all_nodes["is_drug"] == 1].index
        }

        drug_rec_ids = drug_nodes.index
        drug_fps = pd.DataFrame.from_dict(idx_to_fp_dict, orient="index").set_index(
            drug_rec_ids
        )
        drug_fps.columns = ["fp_" + str(i) for i in range(self.fp_bits)]

        # Add fingerprints
        drug_nodes.drop(["smiles", "is_drug"], axis=1, inplace=True)
        drug_nodes = pd.concat((drug_fps, drug_nodes), axis=1)

        # Keep only relevant features
        protein_nodes.drop("is_drug", axis=1, inplace=True)

        return drug_nodes, protein_nodes, rec_id_to_idx_dict, is_drug

    def _get_ppi_edges(self, rec_id_to_idx_dict):
        return get_ppi_edges(rec_id_to_idx_dict, self.ppi_graph)

    def _get_dpi_edges(self, rec_id_to_idx_dict):
        return get_dpi_edges(rec_id_to_idx_dict, self.dti_graph)

    def _get_ddi_edges(self, data_df):
        return get_ddi_edges(data_df)

    def random_split(self, test_prob, valid_prob, level="drug"):
        """
        :param level: Two options, "drug" and "pair". Specify whether the split is performed at the drug level (i.e.
            drugs are assigned to a split, and we only consider combinations of drugs which are in the same split), or
            at the drug pair level (i.e. all examples corresponding to a given combination of drugs are assigned to the
            same split, regardless of cell line)
        :return: 3 tensors containing indices of train, valid and test drug-drug edges
        """
        assert level in ["drug", "pair"]

        if level == "drug":
            return drug_level_random_split(test_prob, valid_prob, self.data)
        else:
            return pair_level_random_split(test_prob, valid_prob, self.data)


class DrugCombMatrixNoPPI(DrugCombMatrix):
    def __init__(
        self,
        transform=None,
        pre_transform=None,
        fp_bits=1024,
        fp_radius=4,
        ppi_graph="huri.csv",
        dti_graph="chembl_dtis.csv",
        cell_line=None,
        use_l1000=False,
        restrict_to_l1000_covered_drugs=False,
    ):
        """
        Version of the dataset which do not include Protein-Protein information
        """
        super().__init__(
            transform,
            pre_transform,
            fp_bits,
            fp_radius,
            ppi_graph,
            dti_graph,
            cell_line,
            use_l1000,
            restrict_to_l1000_covered_drugs,
        )

    def process(self):
        super().process()

    def download(self):
        super().download()

    @property
    def processed_file_names(self):
        return ["drugcomb_matrix_data_no_ppi.pt"]

    def _get_ppi_edges(self, rec_id_to_idx_dict):
        return np.empty([2, 0])


if __name__ == "__main__":
    dataset = DrugCombMatrix(
        fp_bits=1024, fp_radius=4, use_l1000=True, restrict_to_l1000_covered_drugs=True
    )
