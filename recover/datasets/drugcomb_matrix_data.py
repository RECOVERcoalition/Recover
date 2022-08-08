import numpy as np
import pandas as pd
import torch
import os
import random
import copy
import reservoir as rsv
from sklearn.decomposition import PCA
from pandas.api.types import is_string_dtype, is_numeric_dtype
from pathlib import Path
from recover.utils.utils import get_project_root, get_fingerprint

########################################################################################################################
# Underlying data object
########################################################################################################################


class Data(object):
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            self[key] = item

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def to(self, device, *keys, **kwargs):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            yield key, self[key]

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys


########################################################################################################################
# Main Dataset object
########################################################################################################################


class DrugCombMatrix:
    def __init__(
            self,
            fp_bits=1024,
            fp_radius=4,
            cell_line=None,
            study_name="ALMANAC",
            in_house_data="without",
            rounds_to_include=(),
    ):
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.cell_line = cell_line
        self.study_name = study_name
        self.rounds_to_include = list(rounds_to_include)

        assert in_house_data in ["with", "without", "in_house_only"]
        self.in_house_data = in_house_data

        if in_house_data != "without":
            assert len(self.rounds_to_include) > 0

        # Load processed dataset
        saved_file = os.path.join(self.processed_paths, "processed", self.processed_file_name)
        if os.path.isfile(saved_file):
            self.data = torch.load(saved_file)
        else:
            Path(os.path.join(self.processed_paths, "processed")).mkdir(parents=True, exist_ok=True)
            self.data = self.process()

        # Select a specific cell line is cell_line is not None
        if cell_line is not None:
            assert cell_line in self.data.cell_line_to_idx_dict.keys()
            ixs = (
                    self.data.ddi_edge_classes
                    == self.data.cell_line_to_idx_dict[cell_line]
            )
            for attr in dir(self.data):
                if attr.startswith("ddi_edge_idx"):
                    self.data[attr] = self.data[attr][:, ixs]
                elif attr.startswith("ddi_edge_"):
                    self.data[attr] = self.data[attr][ixs]

            # # Update cell line index in case some cell lines have been removed
            # new_unique_cell_lines = self.data.ddi_edge_classes.unique()
            # old_idx_to_new_idx = {
            #     int(new_unique_cell_lines[i]): i for i in range(len(new_unique_cell_lines))
            # }
            # # Update cell_line_to_idxs dictionary
            # self.data.cell_line_to_idx_dict = {
            #     k: old_idx_to_new_idx[self.data.cell_line_to_idx_dict[k]]
            #     for k in self.data.cell_line_to_idx_dict.keys()
            #     if self.data.cell_line_to_idx_dict[k] in old_idx_to_new_idx.keys()
            # }
            # # Update ddi_edge_classes
            # self.data.ddi_edge_classes = self.data.ddi_edge_classes.apply_(
            #     lambda x: old_idx_to_new_idx[x]
            # )

        # Restrict to in house data or study data only if in_house_data is not "with"
        if (self.in_house_data == "without") or (self.in_house_data == "in_house_only"):
            if self.in_house_data == "without":
                ixs = self.data.ddi_edge_in_house == 0
            else:
                ixs = self.data.ddi_edge_in_house == 1
            for attr in dir(self.data):
                if attr.startswith("ddi_edge_idx"):
                    self.data[attr] = self.data[attr][:, ixs]
                elif attr.startswith("ddi_edge_"):
                    self.data[attr] = self.data[attr][ixs]

        print("Dataset loaded.")
        print(
            self.data.ddi_edge_idx.shape[1],
            "drug comb experiments among",
            self.data.x_drugs.shape[0],
            "drugs",
        )
        print("\t fingeprints with radius", self.fp_radius, "and nbits", self.fp_bits)
        print("\t", "drug features dimension", self.data.x_drugs.shape[1])
        print("\t", len(set(self.data.ddi_edge_classes.numpy())), "cell-lines")

    @property
    def processed_paths(self):
        return os.path.join(get_project_root(), "Drugcomb/")

    @property
    def processed_file_name(self):

        proc_file_name = "drugcomb_matrix_data_" + \
                         str(self.fp_bits) + "_" + \
                         str(self.fp_radius) + "_" + \
                         str(self.rounds_to_include)

        return proc_file_name

    def get_blocks(self):
        blocks = rsv.get_specific_drug_combo_blocks(
            data_set='DrugComb',
            # version=1.5,
            study_name=self.study_name,
        )["block_id"]

        return blocks

    def process(self):

        print("Processing the dataset, only happens the first time.")

        ##################################################################
        # Load DrugComb dataframe
        ##################################################################

        blocks = self.get_blocks()

        combo_data = rsv.get_drug_combo_data_combos(
            block_ids=blocks,
            # data_set='DrugComb',
            # version=1.5
        )

        if len(self.rounds_to_include) == 0:
            # If no rounds are included, we can add specific scores which are not provided in in house data
            combo_data = combo_data[['cell_line_name', 'drug_row_recover_id', 'drug_row_smiles',
                                     'drug_col_recover_id', 'drug_col_smiles', 'synergy_bliss',
                                     'css_ri']]
        else:
            combo_data = combo_data[['cell_line_name', 'drug_row_recover_id', 'drug_row_smiles',
                                     'drug_col_recover_id', 'drug_col_smiles']]

        combo_data['is_in_house'] = 0

        for round_id in self.rounds_to_include:
            in_house_data = rsv.get_inhouse_data(project='oncology', experiment_round=round_id)
            in_house_data = in_house_data[['cell_line_name', 'drug_row_relation_id', 'drug_row_smiles',
                                           'drug_col_relation_id', 'drug_col_smiles']]
            in_house_data['is_in_house'] = 1

            # Add scores that are not included in in_house_data
            in_house_data['css_ri'] = 0
            in_house_data['synergy_bliss'] = 0

            combo_data = pd.concat((combo_data, in_house_data))

        ##################################################################
        # Get node features and edges
        ##################################################################

        # Get nodes features
        drug_nodes, rec_id_to_idx_dict = self._get_nodes(combo_data)

        # Get combo experiments
        ddi_edge_idx, ddi_edge_classes, ddi_edge_bliss_av, ddi_edge_css_av, \
        cell_line_to_idx_dict, \
        ddi_is_in_house = self._get_ddi_edges(combo_data, rec_id_to_idx_dict)

        ##################################################################
        # Create data object
        ##################################################################

        data = Data(
            x_drugs=torch.tensor(drug_nodes.to_numpy(), dtype=torch.float)
        )

        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.long)
        data.ddi_edge_in_house = torch.tensor(ddi_is_in_house, dtype=torch.long)
        data.cell_line_to_idx_dict = cell_line_to_idx_dict
        data.rec_id_to_idx_dict = rec_id_to_idx_dict
        # data.cell_line_features = torch.tensor(cell_line_features, dtype=torch.float)

        # Scores
        # data.ddi_edge_bliss_max = torch.tensor(ddi_edge_bliss_max, dtype=torch.float)
        data.ddi_edge_bliss_av = torch.tensor(ddi_edge_bliss_av, dtype=torch.float)
        data.ddi_edge_css_av = torch.tensor(ddi_edge_css_av, dtype=torch.float)

        # Add attributes to data
        data.fp_radius = self.fp_radius
        data.fp_bits = self.fp_bits
        data.study_name = self.study_name

        torch.save(data, os.path.join(self.processed_paths, "processed", self.processed_file_name))

        return data

    def _get_nodes(self, combo_df):

        # Retrieve drugs that are not in the initial training set but that we want to add to the knowledge graph
        additional_drugs_df = pd.read_csv(os.path.join(
            rsv.RESERVOIR_DATA_FOLDER,
            'parsed/drug_combos/drug_combos_test_experiments_Almanac54xDrugcomb54_withReplacements.csv'
        ))

        additional_drugs_df.rename(columns={'recover_id_drug1': "drug_row_recover_id",
                                            'recover_id_drug2': "drug_col_recover_id",
                                            'smiles_drug1': "drug_row_smiles",
                                            'smiles_drug2': "drug_col_smiles"}, inplace=True)

        all_nodes_df = pd.concat((additional_drugs_df, combo_df))

        # Get dataframe containing the smiles of all the drugs
        row_smiles = all_nodes_df[["drug_row_recover_id", "drug_row_smiles"]]
        col_smiles = all_nodes_df[["drug_col_recover_id", "drug_col_smiles"]]
        row_smiles.columns = ["drug_recover_id", "smiles"]
        col_smiles.columns = ["drug_recover_id", "smiles"]

        drug_nodes = pd.concat((row_smiles, col_smiles))
        drug_nodes = drug_nodes.drop_duplicates()

        # Associate each drug with an index number
        drug_nodes.reset_index(inplace=True, drop=True)
        rec_id_to_idx_dict = {drug_nodes.at[i, "drug_recover_id"]: i for i in drug_nodes.index}

        drug_nodes.set_index("drug_recover_id", inplace=True)

        # Compute fingerprints
        idx_to_fp_dict = {
            idx: get_fingerprint(
                drug_nodes.at[idx, "smiles"], self.fp_radius, self.fp_bits
            )
            for idx in drug_nodes.index
        }

        drug_fps = pd.DataFrame.from_dict(idx_to_fp_dict, orient="index").set_index(drug_nodes.index)
        drug_fps.columns = ["fp_" + str(i) for i in range(self.fp_bits)]

        # Add a one hot encoding of drugs to the drug features
        one_hot = pd.DataFrame(np.eye(len(drug_fps)), columns=["hot_" + str(i)
                                                               for i in range(len(drug_fps))],
                               dtype=int).set_index(drug_fps.index)

        drug_feat = pd.concat((drug_fps, one_hot), axis=1)

        return drug_feat, rec_id_to_idx_dict

    def _do_pca(self, df):
        n_components = df.shape[0] - 1
        pca = PCA(n_components=n_components)

        transformed = pca.fit_transform(df.to_numpy())

        # Normalize each feature between -1 and 1
        mins = transformed.min(axis=0)
        maxs = transformed.max(axis=0)

        transformed -= mins
        transformed /= ((maxs - mins) / 2)
        transformed -= 1

        return transformed

    def _transform_cell_line_metadata_df(self, df):
        to_concat = []
        df.drop('Unnamed: 0', axis=1, inplace=True)
        for col in df.columns:
            this_res = None
            srs = df[col]

            if is_numeric_dtype(srs):
                srs = srs.fillna(srs.mean())
                this_res = srs
            elif is_string_dtype(srs):
                this_res = pd.get_dummies(srs, dummy_na=True)

            to_concat.append(this_res)

        return pd.concat(to_concat, axis=1).to_numpy()

    def _get_ddi_edges(self, data_df, rec_id_to_idx_dict):

        # Add drug index information to the df
        data_df["drug_row_idx"] = data_df['drug_row_recover_id'].apply(lambda id: rec_id_to_idx_dict[id])
        data_df["drug_col_idx"] = data_df['drug_col_recover_id'].apply(lambda id: rec_id_to_idx_dict[id])

        # Get list of cell lines
        cell_line_list = list(data_df["cell_line_name"].unique())
        cell_line_list.sort()

        # Retrieve cell line features
        # gene_expr, gene_mutation, gene_cn, metadata = rsv.get_cell_line_features(cell_line_list)

        # to_concat = [self._do_pca(df) for df in (gene_expr, gene_mutation, gene_cn)]
        # to_concat.append(self._transform_cell_line_metadata_df(metadata))

        # cell_line_features = np.concatenate(to_concat, axis=1)

        # Restrict to cell lines with features
        # cell_line_list = list(gene_expr.index)
        # data_df = data_df[data_df['cell_line_name'].apply(lambda cl: cl in cell_line_list)]

        # Get cell line dictionary
        cell_line_to_idx_dict = dict((v, k) for k, v in enumerate(cell_line_list))

        # Add categorical encoding of cell lines
        data_df = copy.deepcopy(data_df)  # To avoid pandas issue
        data_df["cell_line_cat"] = data_df["cell_line_name"].apply(lambda c: cell_line_to_idx_dict[c])

        # Drug indices of combos
        ddi_edge_idx = data_df[["drug_row_idx", "drug_col_idx"]].to_numpy().T
        # Cell lines
        ddi_edge_classes = data_df["cell_line_cat"].to_numpy()
        # Scores
        # ddi_edge_bliss_max = data_df['synergy_bliss_max'].to_numpy()
        ddi_edge_bliss_av = data_df['synergy_bliss'].to_numpy()
        ddi_edge_css_av = data_df['css_ri'].to_numpy()

        # Is in house
        ddi_is_in_house = data_df['is_in_house'].to_numpy()

        return ddi_edge_idx, ddi_edge_classes, ddi_edge_bliss_av, ddi_edge_css_av, \
               cell_line_to_idx_dict, ddi_is_in_house

    def random_split(self, config):

        test_on_unseen_cell_line = config["test_on_unseen_cell_line"]
        split_valid_train_level = config["split_valid_train"]

        assert split_valid_train_level in ["cell_line_level", "pair_level"]

        valid_prob, test_prob = config["val_set_prop"], config["test_set_prop"]

        if test_on_unseen_cell_line:

            ##################################################################
            # Split s.t. test is a set of cell lines which do not appear in train and valid
            ##################################################################

            assert config["cell_line"] is None
            assert len(config["cell_lines_in_test"]) > 0

            test_cell_line_idxs = [self.data.cell_line_to_idx_dict[cl_name] for cl_name in config["cell_lines_in_test"]]

            test_idx = []
            for cl_idx in test_cell_line_idxs:
                cl_idxs = np.where(self.data.ddi_edge_classes.to('cpu') == cl_idx)[0].tolist()
                test_idx.extend(cl_idxs)

            valid_train_cell_line_idxs = list(set(self.data.cell_line_to_idx_dict.values()) -
                                              set(test_cell_line_idxs))

            test_idx = np.array(test_idx)

            if split_valid_train_level == "cell_line_level":

                # Get number of cell lines in valid
                nvalid = int(len(valid_train_cell_line_idxs) * valid_prob)

                # Shuffle train_valid cell line indices
                valid_train_cell_line_idxs = \
                    torch.tensor(valid_train_cell_line_idxs)[torch.randperm(len(valid_train_cell_line_idxs))]

                # Assign cell lines to either valid or train
                train_cell_line_idxs = valid_train_cell_line_idxs[nvalid:]
                valid_cell_line_idxs = valid_train_cell_line_idxs[:nvalid]

                train_idx = []
                for cl_idx in train_cell_line_idxs:
                    cl_idxs = np.where(self.data.ddi_edge_classes.to('cpu') == cl_idx)[0].tolist()
                    train_idx.extend(cl_idxs)
                valid_idx = []
                for cl_idx in valid_cell_line_idxs:
                    cl_idxs = np.where(self.data.ddi_edge_classes.to('cpu') == cl_idx)[0].tolist()
                    valid_idx.extend(cl_idxs)

                train_idx = np.array(train_idx)
                valid_idx = np.array(valid_idx)

            else:

                ##################################################################
                # Split train valid at the pair level
                ##################################################################

                # Get all valid train indices
                valid_train_idx = []
                for cl_idx in valid_train_cell_line_idxs:
                    cl_idxs = np.where(self.data.ddi_edge_classes.to('cpu') == cl_idx)[0].tolist()
                    valid_train_idx.extend(cl_idxs)

                # Get unique edges (one for each drug pair, regardless of cell line) which appear in valid_train combos
                unique_train_valid_ddi_edge_idx = self.data.ddi_edge_idx[:, valid_train_idx].unique(dim=1)
                num_unique_examples = unique_train_valid_ddi_edge_idx.shape[1]

                # Number of unique edges in valid
                nvalid = int(num_unique_examples * valid_prob)

                # Shuffle train and valid unique indices with current seed
                unique_train_valid_idx = torch.randperm(unique_train_valid_ddi_edge_idx.shape[1])

                # Get train and valid
                unique_valid_idx = unique_train_valid_idx[:nvalid]
                unique_train_idx = unique_train_valid_idx[nvalid:]

                # Dictionary that associate each unique edge with a split (train: 0, valid: 1)
                edge_to_split_dict = {
                    **{tuple(unique_train_valid_ddi_edge_idx.T[i].tolist()): 0 for i in unique_train_idx},
                    **{tuple(unique_train_valid_ddi_edge_idx.T[i].tolist()): 1 for i in unique_valid_idx},
                }

                # Associate each (non unique) edges with a split
                all_edges_split = []
                for edge in self.data.ddi_edge_idx.T:
                    try:
                        all_edges_split.append(edge_to_split_dict[tuple(edge.tolist())])
                    except KeyError:
                        all_edges_split.append(-1)
                all_edges_split = np.array(all_edges_split)

                # Discard edges that are already in the test
                all_edges_split[test_idx] = -1

                # Get train/valid indices for all (non unique) edges
                train_idx = np.where(all_edges_split == 0)[0]
                valid_idx = np.where(all_edges_split == 1)[0]

        else:

            ##################################################################
            # Split everything at the pair level
            ##################################################################

            assert config["split_valid_train"] == "pair_level", "if not testing on separate cell lines, " \
                                                                "split level must be on the pair level"

            # Get unique edges (one for each drug pair, regardless of cell line)
            unique_ddi_edge_idx = self.data.ddi_edge_idx.unique(dim=1)
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
                [edge_to_split_dict[tuple(edge.tolist())] for edge in self.data.ddi_edge_idx.T]
            )

            # Get train/valid/test indices for all (non unique) edges
            train_idx = np.where(all_edges_split == 0)[0]
            valid_idx = np.where(all_edges_split == 1)[0]
            test_idx = np.where(all_edges_split == 2)[0]

        # Shuffle the order within each split
        np.random.shuffle(train_idx)
        np.random.shuffle(valid_idx)
        np.random.shuffle(test_idx)

        return torch.tensor(train_idx, dtype=torch.long), \
               torch.tensor(valid_idx, dtype=torch.long), \
               torch.tensor(test_idx, dtype=torch.long)


########################################################################################################################
# Dataset objects where some drug features are removed
########################################################################################################################


class DrugCombMatrixNoFP(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)
        self.data.x_drugs[:, :self.fp_bits] = 0


class DrugCombMatrixNoOneHot(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)
        self.data.x_drugs[:, self.fp_bits:] = 0


########################################################################################################################
# Dataset objects for transfer study
########################################################################################################################


class DrugCombMatrixTrainOneil(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ONEIL",
                 in_house_data="without",
                 rounds_to_include=()):
        assert study_name == "ONEIL"
        assert in_house_data == "without"
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)
        print("keep only fingerprint features")
        self.data.x_drugs = self.data.x_drugs[:, :self.fp_bits]

    @property
    def processed_file_name(self):

        proc_file_name = "drugcomb_matrix_data_trainOneil_" + \
                         str(self.fp_bits) + "_" + \
                         str(self.fp_radius) + "_" + \
                         str(self.rounds_to_include)

        return proc_file_name

    def get_blocks(self):

        all_splits = pd.read_pickle(os.path.join(
            rsv.RESERVOIR_DATA_FOLDER, 'parsed/drug_combos/transfer_splits_1_5.pkl'
        )
        )

        all_splits = all_splits[all_splits["overlap"] == "divided"]
        all_splits = all_splits[all_splits["type"] == "train"]
        all_splits = all_splits[all_splits["cell_line"] == "all_overlapping"]
        all_splits = all_splits[all_splits["study_train"] == "ONEIL"]
        all_splits = all_splits[all_splits["study_predict"] == "ALMANAC"]

        all_splits = all_splits[all_splits["trim"] == "not_trimmed"]
        assert len(all_splits) == 1
        block_ids = all_splits["block_ids"].item()

        blocks = pd.Series(block_ids)

        return blocks


class DrugCombMatrixTestAlmanac(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        assert study_name == "ALMANAC"
        assert in_house_data == "without"
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)
        print("keep only fingerprint features")
        self.data.x_drugs = self.data.x_drugs[:, :self.fp_bits]

    @property
    def processed_file_name(self):

        proc_file_name = "drugcomb_matrix_data_testAlmanac_" + \
                         str(self.fp_bits) + "_" + \
                         str(self.fp_radius) + "_" + \
                         str(self.rounds_to_include)

        return proc_file_name

    def get_blocks(self):
        all_splits = pd.read_pickle(os.path.join(
            rsv.RESERVOIR_DATA_FOLDER, 'parsed/drug_combos/transfer_splits_1_5.pkl'
        )
        )

        all_splits = all_splits[all_splits["overlap"] == "divided"]
        all_splits = all_splits[all_splits["type"] == "test"]
        all_splits = all_splits[all_splits["cell_line"] == "all_overlapping"]
        all_splits = all_splits[all_splits["study_train"] == "ONEIL"]
        all_splits = all_splits[all_splits["study_predict"] == "ALMANAC"]

        assert len(all_splits) == 1
        block_ids = all_splits["block_ids"].item()

        blocks = pd.Series(block_ids)

        return blocks


class DrugCombMatrixTrainAlmanac(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        assert study_name == "ALMANAC"
        assert in_house_data == "without"
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)
        print("keep only fingerprint features")
        self.data.x_drugs = self.data.x_drugs[:, :self.fp_bits]

    @property
    def processed_file_name(self):

        proc_file_name = "drugcomb_matrix_data_trainAlmanac_" + \
                         str(self.fp_bits) + "_" + \
                         str(self.fp_radius) + "_" + \
                         str(self.rounds_to_include)

        return proc_file_name

    def get_blocks(self):
        all_splits = pd.read_pickle(os.path.join(
            rsv.RESERVOIR_DATA_FOLDER, 'parsed/drug_combos/transfer_splits_1_5.pkl'
        )
        )

        all_splits = all_splits[all_splits["overlap"] == "divided"]
        all_splits = all_splits[all_splits["type"] == "train"]
        all_splits = all_splits[all_splits["cell_line"] == "all_overlapping"]
        all_splits = all_splits[all_splits["study_train"] == "ALMANAC"]
        all_splits = all_splits[all_splits["study_predict"] == "ONEIL"]

        all_splits = all_splits[all_splits["trim"] == "trimmed"]

        assert len(all_splits) == 1
        block_ids = all_splits["block_ids"].item()

        blocks = pd.Series(block_ids)

        return blocks


########################################################################################################################
# Dataset objects with various numbers of unique drugs
########################################################################################################################


class DrugCombMatrixRemoveHalfDrugs(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)

        to_be_removed_drug_prop = 0.5

        # Choose at random a proportion drug_prop of the drugs
        all_drug_idxs = list(self.data.rec_id_to_idx_dict.values())
        random.Random(0).shuffle(all_drug_idxs)
        to_be_removed_drug_idxs = all_drug_idxs[:int(to_be_removed_drug_prop*len(all_drug_idxs))]

        to_be_removed_ixs = []
        for i in range(len(to_be_removed_drug_idxs)):
            to_be_removed_ixs += list(np.where(self.data.ddi_edge_idx == to_be_removed_drug_idxs[i])[1])

        # Indices to keep
        ixs = list(set(range(self.data.ddi_edge_idx.shape[1])) - set(np.unique(to_be_removed_ixs)))

        for attr in dir(self.data):
            if attr.startswith("ddi_edge_idx"):
                self.data[attr] = self.data[attr][:, ixs]
            elif attr.startswith("ddi_edge_"):
                self.data[attr] = self.data[attr][ixs]

        print(
            self.data.ddi_edge_idx.shape[1],
            "drug comb experiments among",
            self.data.x_drugs.shape[0],
            "drugs",
        )


class DrugCombMatrixRemoveOneFourthfDrugs(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)

        to_be_removed_drug_prop = 0.25

        # Choose at random a proportion drug_prop of the drugs
        all_drug_idxs = list(self.data.rec_id_to_idx_dict.values())
        random.Random(0).shuffle(all_drug_idxs)
        to_be_removed_drug_idxs = all_drug_idxs[:int(to_be_removed_drug_prop*len(all_drug_idxs))]

        to_be_removed_ixs = []
        for i in range(len(to_be_removed_drug_idxs)):
            to_be_removed_ixs += list(np.where(self.data.ddi_edge_idx == to_be_removed_drug_idxs[i])[1])

        # Indices to keep
        ixs = list(set(range(self.data.ddi_edge_idx.shape[1])) - set(np.unique(to_be_removed_ixs)))

        for attr in dir(self.data):
            if attr.startswith("ddi_edge_idx"):
                self.data[attr] = self.data[attr][:, ixs]
            elif attr.startswith("ddi_edge_"):
                self.data[attr] = self.data[attr][ixs]

        print(
            self.data.ddi_edge_idx.shape[1],
            "drug comb experiments among",
            self.data.x_drugs.shape[0],
            "drugs",
        )


########################################################################################################################
# Dataset objects for drug level split
########################################################################################################################


class DrugCombMatrixDrugLevelSplitTrain(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)
        self.data.x_drugs[:, self.fp_bits:] = 0

        propr_drug_in_test = 0.3

        # Choose at random a proportion drug_prop of the drugs
        all_drug_idxs = list(self.data.rec_id_to_idx_dict.values())
        random.Random(0).shuffle(all_drug_idxs)
        test_drug_idxs = all_drug_idxs[:int(propr_drug_in_test*len(all_drug_idxs))]

        test_ixs = []
        for i in range(len(test_drug_idxs)):
            test_ixs += list(np.where(self.data.ddi_edge_idx == test_drug_idxs[i])[1])

        # Indices to keep
        ixs = list(set(range(self.data.ddi_edge_idx.shape[1])) - set(np.unique(test_ixs)))

        for attr in dir(self.data):
            if attr.startswith("ddi_edge_idx"):
                self.data[attr] = self.data[attr][:, ixs]
            elif attr.startswith("ddi_edge_"):
                self.data[attr] = self.data[attr][ixs]

        print(
            self.data.ddi_edge_idx.shape[1],
            "drug comb experiments among",
            self.data.x_drugs.shape[0],
            "drugs",
        )


class DrugCombMatrixDrugLevelSplitTest(DrugCombMatrix):
    def __init__(self,
                 fp_bits=1024,
                 fp_radius=4,
                 cell_line=None,
                 study_name="ALMANAC",
                 in_house_data="without",
                 rounds_to_include=()):
        super().__init__(fp_bits, fp_radius, cell_line, study_name, in_house_data, rounds_to_include)
        self.data.x_drugs[:, self.fp_bits:] = 0

        # "To be removed" drugs are kept for the test
        propr_drug_in_test = 0.3

        # Choose at random a proportion drug_prop of the drugs
        all_drug_idxs = list(self.data.rec_id_to_idx_dict.values())
        random.Random(0).shuffle(all_drug_idxs)
        trainval_drug_idxs = all_drug_idxs[int(propr_drug_in_test*len(all_drug_idxs)):]

        trainval_ixs = []
        for i in range(len(trainval_drug_idxs)):
            trainval_ixs += list(np.where(self.data.ddi_edge_idx == trainval_drug_idxs[i])[1])

        # Indices to keep
        ixs = list(set(range(self.data.ddi_edge_idx.shape[1])) - set(np.unique(trainval_ixs)))

        for attr in dir(self.data):
            if attr.startswith("ddi_edge_idx"):
                self.data[attr] = self.data[attr][:, ixs]
            elif attr.startswith("ddi_edge_"):
                self.data[attr] = self.data[attr][ixs]

        print(
            self.data.ddi_edge_idx.shape[1],
            "drug comb experiments among",
            self.data.x_drugs.shape[0],
            "drugs",
        )


if __name__ == "__main__":

    dataset = DrugCombMatrixTrainOneil(
        in_house_data='without',
        rounds_to_include=[],
        cell_line=None,
        fp_bits=1024,
        fp_radius=2
    )

    print(dataset.data.cell_line_to_idx_dict)

    dataset = DrugCombMatrixTrainAlmanac(
        in_house_data='without',
        rounds_to_include=[],
        cell_line=None,
        fp_bits=1024,
        fp_radius=2
    )

    print(dataset.data.cell_line_to_idx_dict)
