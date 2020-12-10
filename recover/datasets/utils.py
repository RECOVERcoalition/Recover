import torch
import pandas as pd
import numpy as np
import reservoir as rsv
import os
from recover.utils import get_project_root
from tqdm import tqdm
from torch.utils.data import TensorDataset
import warnings
import random


########################################################################################################################
# Get TensorDataset
########################################################################################################################


def get_tensor_dataset(data, idxs):
    """
    Given a  `torch_geometric.data.Data` object and a list of indices, build a pytorch dataset object
        of drug combination examples
    :param data:
    :param idxs:
    :return: Pytorch TensorDataset object
    """
    return TensorDataset(
        data.ddi_edge_idx[:, idxs].T,
        data.ddi_edge_classes[idxs],
        data.ddi_edge_response[idxs],
        data.ddi_edge_conc_pair[idxs],
        data.ddi_edge_inhibitions[idxs],
        data.ddi_edge_conc_r[idxs],
        data.ddi_edge_conc_c[idxs],
        data.ddi_edge_inhibition_r[idxs],
        data.ddi_edge_inhibition_c[idxs],
        data.ddi_edge_sig_params_r[idxs],
        data.ddi_edge_sig_params_c[idxs],
    )


########################################################################################################################
# Transform methods
########################################################################################################################


def take_first_k_vals(k):
    """
    Used as a transform function for the dataset, in order to select a subset of the total dataset
    """

    def _take_first_k_vals(data):
        idxs = torch.randperm(data.ddi_edge_idx.shape[1])
        idxs = idxs[:k]

        data.ddi_edge_idx = data.ddi_edge_idx[:, idxs]
        data.ddi_edge_classes = data.ddi_edge_classes[idxs]

        data.ddi_edge_css = data.ddi_edge_css[idxs]
        data.ddi_edge_zip = data.ddi_edge_zip[idxs]
        data.ddi_edge_bliss = data.ddi_edge_bliss[idxs]
        data.ddi_edge_loewe = data.ddi_edge_loewe[idxs]
        data.ddi_edge_hsa = data.ddi_edge_hsa[idxs]

        data.ddi_edge_conc_pair = data.ddi_edge_conc_pair[idxs]
        data.ddi_edge_inhibitions = data.ddi_edge_inhibitions[idxs]

        data.ddi_edge_conc_r = data.ddi_edge_conc_r[idxs]
        data.ddi_edge_conc_c = data.ddi_edge_conc_c[idxs]
        data.ddi_edge_inhibition_r = data.ddi_edge_inhibition_r[idxs]
        data.ddi_edge_inhibition_c = data.ddi_edge_inhibition_c[idxs]
        data.ddi_edge_sig_params_r = data.ddi_edge_sig_params_r[idxs]
        data.ddi_edge_sig_params_c = data.ddi_edge_sig_params_c[idxs]

        return data

    return _take_first_k_vals


########################################################################################################################
# Edge getters
########################################################################################################################


def get_dpi_edges(rec_id_to_idx_dict, dti_graph):
    """

    :param rec_id_to_idx_dict: dictionary mapping recover ID of a drug to its node index in the graph
    :param dti_graph: Name of the graph to use (available options: rsv.available_dtis())
    :return: dpi_edge_idx, a (2, num_drug_prot_interactions) array.  dpi_edge_idx[:, i] contains the indices of the
        drug and protein of the ith interaction.
    """
    print("Processing drug protein edges..")

    dpi_edges = rsv.get_dtis(dti_graph)

    # Add idx columns
    dpi_edges["drug_idx"] = dpi_edges["recover_id"].apply(
        lambda x: rec_id_to_idx_dict[x] if x in rec_id_to_idx_dict.keys() else -1
    )
    dpi_edges["prot_idx"] = dpi_edges["gene_hgnc_id"].apply(
        lambda x: rec_id_to_idx_dict[x]
    )

    # Remove edges for which the drug is not in drugcomb
    dpi_edges = dpi_edges[dpi_edges["drug_idx"] != -1]

    # Remove duplicated edges
    dpi_edges = dpi_edges.loc[
        dpi_edges[["drug_idx", "prot_idx"]].drop_duplicates().index
    ]

    dpi_edge_idx = dpi_edges[["drug_idx", "prot_idx"]].to_numpy().T

    return dpi_edge_idx


def get_ppi_edges(rec_id_to_idx_dict, ppi_graph):
    """

    :param rec_id_to_idx_dict: dictionary mapping recover ID of a drug to its node index in the graph
    :param ppi_graph: Name of the graph to use (available options: rsv.available_ppis())
    :return: ppi_edge_idx, a (2, 2*num_prot_prot_interactions) array.  ppi_edge_idx[:, i] contains the indices of the
        proteins of the ith interaction.
        Note that each (undirected) edge in the ppi_graph corresponds to two (directed)
        edges in the ppi_edge_idx array
    """
    print("Processing protein protein edges..")

    ppi_edges = rsv.get_ppis(ppi_graph)

    # Remove self loops
    ppi_edges = ppi_edges[ppi_edges["gene_1_hgnc_id"] != ppi_edges["gene_2_hgnc_id"]]

    # Add idx columns
    ppi_edges["gene_1_idx"] = ppi_edges["gene_1_hgnc_id"].apply(
        lambda x: rec_id_to_idx_dict[x]
    )
    ppi_edges["gene_2_idx"] = ppi_edges["gene_2_hgnc_id"].apply(
        lambda x: rec_id_to_idx_dict[x]
    )

    ppi_edge_idx = ppi_edges[["gene_1_idx", "gene_2_idx"]].to_numpy().T

    # Edges are directed, we need to feed them both ways
    ppi_edge_idx = np.concatenate((ppi_edge_idx, ppi_edge_idx[::-1, :]), axis=1)

    return ppi_edge_idx


def get_ddi_edges(data_df):
    """

    :param data_df: DrugComb dataframe. Each line corresponds to a block (a drug combination experiment) and contains
        various features:
        - node index of the drugs
        - cell line on which the experiment has been performed
        - monotherapy inhibitions and corresponding concentrations
        - combination inhibtions and corresponding pairs of concentrations
        - synergy scores (css, zip, bliss, loewe, HSA)
    :return:
        - ddi_edge_idx (2, num_drug_comb_experiments) numpy array containing the indices of the drugs
        - several arrays of shape (num_drug_comb_experiments, *) containing informations about each experiment

    """
    print("Processing drug drug interaction edges..")

    ddi_edge_idx = data_df[["drug_row_idx", "drug_col_idx"]].to_numpy().T

    # Cell lines
    ddi_edge_classes = data_df["cell_line_cat"].to_numpy()
    cell_line_to_idx_dict = pd.Series(
        data_df["cell_line_cat"].values, index=data_df["cell_line_name"].values
    ).to_dict()

    # Scores
    ddi_edge_css = data_df["css_ri"].to_numpy()
    ddi_edge_zip = data_df["synergy_zip"].to_numpy()
    ddi_edge_bliss = data_df["synergy_bliss"].to_numpy()
    ddi_edge_loewe = data_df["synergy_loewe"].to_numpy()
    ddi_edge_hsa = data_df["synergy_hsa"].to_numpy()

    # Raw data
    ddi_edge_conc_pair = np.stack(data_df["concentration_pairs"])
    ddi_edge_inhibitions = np.stack(data_df["inhibitions"])
    ddi_edge_conc_r = np.stack(data_df["conc_r"])
    ddi_edge_conc_c = np.stack(data_df["conc_c"])
    ddi_edge_inhibition_r = np.stack(data_df["inhibition_r"])
    ddi_edge_inhibition_c = np.stack(data_df["inhibition_c"])

    return (
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
    )


########################################################################################################################
# Data processing methods
########################################################################################################################


def process_data(data, rec_id_to_idx_dict):
    """
    Add some information to the drugcomb dataframe

    :param data: DrugComb dataframe. Each line corresponds to a block (a drug combination experiment)
    :param rec_id_to_idx_dict: dictionary mapping recover IDs to node indices in the graph
    :return:
        - data: processed drugcomb dataframe (node indices of the drugs added, removed examples with missing
        information, cell line categorical encoding)
        - cell_lines: dictionary mapping cell line names to an index
    """

    # Add index of drugs
    data["drug_row_idx"] = data["drug_row_recover_id"].apply(
        lambda rec_id: rec_id_to_idx_dict[rec_id]
        if rec_id in rec_id_to_idx_dict.keys()
        else -1
    )

    data["drug_col_idx"] = data["drug_col_recover_id"].apply(
        lambda rec_id: rec_id_to_idx_dict[rec_id]
        if rec_id in rec_id_to_idx_dict.keys()
        else -1
    )

    # Remove rows for which one of the drugs is not covered
    data = data[data["drug_row_idx"] != -1]
    data = data[data["drug_col_idx"] != -1]

    # Remove rows for which the two drugs in the combo are the same
    data = data[data["drug_row_idx"] != data["drug_col_idx"]]

    # Add categorical encoding of cell lines
    data["cell_line_cat"] = data["cell_line_name"].astype("category").cat.codes

    # Get cell line dictionary
    cell_lines = data["cell_line_name"].astype("category").cat.categories
    cell_lines = dict((v, k) for k, v in enumerate(cell_lines))

    return data, cell_lines


def process_l1000(l1000_df):
    """
    Processed the L1000 dataframe for the needs of the integration with DrugComb
    :param l1000_df: dataframe containing differential gene expression profiles along with metadata
    :return: dataframe containing differential gene expression profiles. We restricted ourselves to examples with
        incubation time = 24h and dosage = 10uM.
        Each line in the dataframe corresponds to a drug, and contains two gene expression profiles from two
        cell lines (PC3 and MCF7)
    """
    l1000_df = l1000_df.loc[l1000_df["cid"].dropna().index]
    l1000_df["cid"] = l1000_df["cid"].astype(int).astype(str)

    # Retrieve recover ID
    cids = l1000_df[["cid"]].drop_duplicates()
    cids.columns = ["pubchem_cid"]
    mapped_drugs = rsv.map_drugs(pd.DataFrame(cids))
    mapped_drugs = mapped_drugs.dropna()
    mapped_drugs.set_index("pubchem_cid")
    cid_to_rec_id_dict = mapped_drugs.set_index("pubchem_cid").to_dict()["recover_id"]

    l1000_df["recover_id"] = l1000_df["cid"].apply(
        lambda cid: cid_to_rec_id_dict[cid] if cid in cid_to_rec_id_dict.keys() else -1
    )

    # Drop examples without recover ID
    l1000_df = l1000_df[l1000_df["recover_id"] != -1]

    # Restrict ourselves to 24h incubation time
    l1000_df = l1000_df[l1000_df["pert_itime_value"] == 24]

    # Restrict ourselves to 10 uM concentrations
    l1000_df = l1000_df[l1000_df["pert_idose_value"] == 10.0]

    # Select two cell lines which cover most drugs in drugcomb: 'PC3' and 'MCF7'
    l1000_df_pc3 = l1000_df[l1000_df["cell_id"] == "PC3"]
    l1000_df_mcf7 = l1000_df[l1000_df["cell_id"] == "MCF7"]

    # Drop duplicates
    l1000_df_mcf7 = l1000_df_mcf7.loc[
        l1000_df_mcf7["recover_id"].drop_duplicates().index
    ]
    l1000_df_pc3 = l1000_df_pc3.loc[l1000_df_pc3["recover_id"].drop_duplicates().index]

    l1000_df_mcf7.set_index("recover_id", drop=True, inplace=True)
    l1000_df_pc3.set_index("recover_id", drop=True, inplace=True)

    # Drop metadata columns
    l1000_df_pc3.drop(
        [
            "cid",
            "pert_idose_value",
            "pert_itime_value",
            "pert_itime",
            "pert_idose",
            "cell_id",
            "pert_id",
        ],
        axis=1,
        inplace=True,
    )
    l1000_df_mcf7.drop(
        [
            "cid",
            "pert_idose_value",
            "pert_itime_value",
            "pert_itime",
            "pert_idose",
            "cell_id",
            "pert_id",
        ],
        axis=1,
        inplace=True,
    )

    l1000_df = pd.concat((l1000_df_pc3, l1000_df_mcf7), axis=1)
    l1000_df.columns = ["pc3_" + str(i) for i in range(l1000_df_pc3.shape[1])] + [
        "mcf7_" + str(i) for i in range(l1000_df_mcf7.shape[1])
    ]

    return l1000_df


########################################################################################################################
# Node getters
########################################################################################################################


def get_protein_nodes(raw_dir):
    """
    Retrieves features for the proteins. Embeddings are computed from protein sequences using the  Elmo model from
    https://github.com/rostlab/SeqVec

    :param raw_dir: Path to the directory containing preprocessed protein embeddings
    :return: protein_nodes dataframe. Each line corresponds to a gene and contains an embedding of the gene based on
        the sequence of the corresponding protein
    """
    print("Processing protein nodes..")

    if os.path.isfile(os.path.join(raw_dir, "protein_embeddings.csv")):
        protein_nodes = pd.read_csv(
            os.path.join(raw_dir, "protein_embeddings.csv"), index_col="gene_hgnc_id"
        )
    else:
        print("Computing protein embeddings, only happens the first time")
        from allennlp.commands.elmo import ElmoEmbedder

        # Embedder of protein sequences
        embedder = ElmoEmbedder(
            os.path.join(get_project_root(), "DrugComb/raw/uniref50_v2/options.json"),
            os.path.join(get_project_root(), "DrugComb/raw/uniref50_v2/weights.hdf5"),
            cuda_device=-1,
        )

        unique_gene_id = (
            rsv.get_proteins()["gene_hgnc_id"].drop_duplicates(keep="first").index
        )

        # Get dataframe containing protein sequences
        protein_nodes = (
            rsv.get_proteins()[["gene_hgnc_id", "protein_sequence"]]
            .loc[unique_gene_id]
            .set_index("gene_hgnc_id")
        )

        # Embed the protein sequences
        seqs = list(protein_nodes["protein_sequence"])
        seqs = [list(seq) for seq in seqs]
        seqs.sort(key=len)
        embedding = []
        for i in tqdm(range(len(seqs) // 10 + 1)):
            batch_embd = list(embedder.embed_sentences(seqs[10 * i : 10 * (i + 1)]))
            batch_embd = [embd.sum(axis=0).mean(axis=0) for embd in batch_embd]

            embedding.extend(batch_embd)

        protein_nodes = pd.DataFrame(
            embedding,
            index=protein_nodes.index,
            columns=["emb_" + str(i) for i in range(len(embedding[0]))],
        )
        protein_nodes["is_drug"] = 0

        # Save the embeddings
        protein_nodes.to_csv(os.path.join(raw_dir, "protein_embeddings.csv"))

    return protein_nodes


def get_drug_nodes(
    data_df, l1000_df, dti_graph, use_l1000, restrict_to_l1000_covered_drugs
):
    """
    Retrieves features for the drugs (smiles + gene expression profiles)
    Note: we concatenate with l1000_df even if we do not wish to use l1000 in order to be able to drop the
    nodes not covered by l1000 (if required).

    :param data_df: dataframe of the DrugComb dataset
    :param l1000_df: processed dataframe of the l1000 dataset
    :param dti_graph: Name of the graph to use (available options: rsv.available_dtis())
    :param use_l1000: Boolean, whether to include gene expression features or not
    :param restrict_to_l1000_covered_drugs: Boolean, whether to drop drugs which are not covered by L1000.
        If False and use_l1000=True, the unknown gene expression profiles are set to zero
    :return: drug_nodes dataframe of node features
    """

    print("Processing drug nodes..")

    # Get dataframe containing the smiles of all the drugs
    row_smiles = data_df[["drug_row_recover_id", "drug_row_smiles"]]
    col_smiles = data_df[["drug_col_recover_id", "drug_col_smiles"]]
    row_smiles.columns = ["drug_recover_id", "smiles"]
    col_smiles.columns = ["drug_recover_id", "smiles"]

    drug_nodes = pd.concat((row_smiles, col_smiles))
    drug_nodes.reset_index(inplace=True, drop=True)

    # Drop duplicates
    drug_nodes = drug_nodes.loc[drug_nodes["drug_recover_id"].drop_duplicates().index]
    drug_nodes.set_index("drug_recover_id", inplace=True)

    # Restrict ourselves to drugs with targets in the dti_graph
    dpi_edges = rsv.get_dtis(dti_graph)
    drug_nodes = drug_nodes.loc[
        set(dpi_edges["recover_id"].unique()).intersection(drug_nodes.index)
    ]

    drug_nodes["is_drug"] = 1

    # Keep only gene expression profiles that correspond to drugs covered in the drugcomb dataset
    l1000_df["recover_id"] = l1000_df.index
    l1000_df = l1000_df[
        l1000_df["recover_id"].apply(lambda idx: idx in drug_nodes.index)
    ]

    drug_nodes = pd.concat((drug_nodes, l1000_df), axis=1, sort=False)
    drug_nodes.drop("recover_id", axis=1, inplace=True)

    # Drop the nodes not covered by l1000
    if restrict_to_l1000_covered_drugs:
        drug_nodes = drug_nodes.dropna()
    else:
        drug_nodes = drug_nodes.fillna(0)

    # Drop the l1000 features if we do not want to use them (after possibly having dropped the uncovered drugs)
    if not use_l1000:
        drug_nodes = drug_nodes[["smiles", "is_drug"]]

    return drug_nodes


def fit_sigmoid_on_monotherapies(
    ddi_edge_conc, ddi_edge_inhibition, save_file="mono_sigm_params.npy"
):
    """
    Fit sigmoid on each of the monotherapy inhibition profiles, and returns the parameters of the sigmoids

    :param ddi_edge_conc: numpy array (num_monotherapies, num_concentrations_per_monotherapy) containing the
        monotherapy concentrations
    :param ddi_edge_inhibition: numpy array (num_monotherapies, num_concentrations_per_monotherapy) containing the
        monotherapy inhibitions
    :param save_file:
    :return: numpy array (num_monotherapies, 3) containing the ic50, slope and y_max for each monotherapy profile
    """

    if os.path.isfile(os.path.join(get_project_root(), "DrugComb/raw/", save_file)):
        return np.load(os.path.join(get_project_root(), "DrugComb/raw/", save_file))
    else:
        print("Fitting sigmoids on monotherapy responses, only happens the first time")

        from scipy.optimize import curve_fit

        def sigmoid(x, ic50, slope, y_max):
            # y_min fixed to zero because data is normalized
            y = y_max / (1 + np.exp(-slope * (x - ic50)))
            return y

        xdata = np.log(ddi_edge_conc + 1e-6)  # Log transform concentrations
        ydata = ddi_edge_inhibition

        all_popt = []

        for i in tqdm(range(len(xdata))):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Initialize gradient descent with reasonable parameters
                p0 = [(xdata[i, 1] + xdata[i, 2]) / 2, 1, ydata[i, 3]]
                popt, pcov = curve_fit(
                    sigmoid,
                    xdata[i],
                    ydata[i],
                    p0=p0,
                    method="lm",
                    maxfev=1000000,
                )

            all_popt.append(popt.tolist())

        all_popt = np.array(all_popt)
        np.save(os.path.join(get_project_root(), "DrugComb/raw/", save_file), all_popt)

    return all_popt


########################################################################################################################
# Random split methods
########################################################################################################################


def drug_level_random_split(test_prob, valid_prob, data):
    """
    Splits the drug combination examples into train/valid/test sets. The split happens at the drug level,
    that is to say, each drug is assigned to a split. Then, the combinations of drugs from the same split are assigned
    to this split. Combinations of drugs which lie in different splits are dropped.

    :param test_prob: proportion of examples in the test set
    :param valid_prob: proportion of examples in the valid set
    :param data: `torch_geometric.data.Data` object
    :return: 3 1D tensors containing the indices of ddi_edges for each of the three splits
    """

    # Get all drugs
    unique_drugs = data.ddi_edge_idx.unique()
    num_drugs = len(unique_drugs)

    shuflled_idx = np.arange(num_drugs)
    random.Random(0).shuffle(shuflled_idx)

    # Get shuffled list of unique drug indices (always shuffle with seed zero)
    unique_drugs = unique_drugs[shuflled_idx]

    # Assign each drug to a split
    train_valid_drugs = unique_drugs[int(num_drugs * test_prob) :]
    test_drugs = unique_drugs[
        : int(num_drugs * test_prob)
    ]  # Does not depend on user defined seed

    # Shuffle train and valid according to user defined seed
    train_valid_drugs = train_valid_drugs[torch.randperm(len(train_valid_drugs))]

    valid_drugs = set(train_valid_drugs[: int(num_drugs * valid_prob)].numpy())
    train_drugs = set(train_valid_drugs[int(num_drugs * valid_prob) :].numpy())
    test_drugs = set(test_drugs.numpy())

    # Get indices of drug-drug edges for each split. A combination is selected for a split of both drugs in the
    # combination belong to this split. Other combinations are dropped
    train_idx = np.where(
        np.apply_along_axis(
            lambda x: x[0] in train_drugs and x[1] in train_drugs, 0, data.ddi_edge_idx
        )
    )[0]
    valid_idx = np.where(
        np.apply_along_axis(
            lambda x: x[0] in valid_drugs and x[1] in valid_drugs, 0, data.ddi_edge_idx
        )
    )[0]
    test_idx = np.where(
        np.apply_along_axis(
            lambda x: x[0] in test_drugs and x[1] in test_drugs, 0, data.ddi_edge_idx
        )
    )[0]

    # Shuffle the order within each split
    np.random.shuffle(train_idx)
    np.random.shuffle(valid_idx)
    np.random.shuffle(test_idx)

    return torch.tensor(train_idx), torch.tensor(valid_idx), torch.tensor(test_idx)


def pair_level_random_split(test_prob, valid_prob, data):
    """
    Splits the drug combination examples into train/valid/test sets. The split happens at the pair level,
    that is to say, all the examples corresponding to the same pair of drugs will be assigned to the same split

    Note: several example can correspond to the same pair of drugs, but different cell lines. There can also be
    duplicates (same experience was performed several times)

    :param test_prob: proportion of examples in the test set
    :param valid_prob: proportion of examples in the valid set
    :param data: `torch_geometric.data.Data` object
    :return: 3 1D tensors containing the indices of ddi_edges for each of the three splits
    """

    # Get unique edges (one for each drug pair, regardless of cell line)
    unique_ddi_edge_idx = data.ddi_edge_idx.unique(dim=1)
    num_unique_examples = unique_ddi_edge_idx.shape[1]

    # Number of unique edges for each of the splits
    nvalid = int(num_unique_examples * valid_prob)
    ntest = int(num_unique_examples * test_prob)

    # Shuffle with seed 0
    shuflled_idx = np.arange(num_unique_examples)
    random.Random(0).shuffle(shuflled_idx)

    unique_test_idx = shuflled_idx[
        :ntest
    ]  # Fixed test set (does not depend on the seed)
    unique_train_valid_idx = shuflled_idx[ntest:]  # Train and valid sets
    # Shuffle train and valid with current seed
    unique_train_valid_idx = unique_train_valid_idx[
        torch.randperm(len(unique_train_valid_idx))
    ]
    # Get train and valid
    unique_valid_idx = unique_train_valid_idx[:nvalid]
    unique_train_idx = unique_train_valid_idx[nvalid:]

    # Dictionary that associate each unique edge with a split (train: 0, valid: 1, test: 2)
    edge_to_split_dict = {
        **{tuple(unique_ddi_edge_idx.T[i].tolist()): 0 for i in unique_train_idx},
        **{tuple(unique_ddi_edge_idx.T[i].tolist()): 1 for i in unique_valid_idx},
        **{tuple(unique_ddi_edge_idx.T[i].tolist()): 2 for i in unique_test_idx},
    }

    # Associate each (non unique) edges with a split
    all_edges_split = np.array(
        [edge_to_split_dict[tuple(edge.tolist())] for edge in data.ddi_edge_idx.T]
    )

    # Get train/valid/test indices for all (non unique) edges
    train_idx = np.where(all_edges_split == 0)[0]
    val_idx = np.where(all_edges_split == 1)[0]
    test_idx = np.where(all_edges_split == 2)[0]

    # Shuffle the order within each split
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)
