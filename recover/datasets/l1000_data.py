from torch_geometric.data import Data, InMemoryDataset, download_url
from recover.utils import get_project_root
import pickle
import pandas as pd
import torch
import os
import gzip
import shutil
from tqdm import tqdm


class L1000(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None):
        """
        Dataset object for the LINCS L1000 dataset.
        """

        super().__init__(
            os.path.join(get_project_root(), "LINCS/"), transform, pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

        print("L1000 dataset loaded.")

    @property
    def raw_file_names(self):
        """ The first 5 files correspond to phase 2, the last 3 to phase 1"""
        return [
            "Level5_COMPZ_n118050x12328_2017-03-06.gctx",
            "sig_info_2017-03-06.txt",
            "gene_info_2017-03-06.txt",
            "pert_info_2017-03-06.txt",
            "cell_info_2017-04-28.txt",
            "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
            "GSE92742_Broad_LINCS_sig_info.txt",
            "GSE92742_Broad_LINCS_pert_info.txt",
        ]

    @property
    def processed_file_names(self):
        return ["l1000_data.pt"]

    def download(self):
        urls = [
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5FLevel5"
            "%5FCOMPZ%5Fn118050x12328%5F2017%2D03%2D06%2Egctx%2Egz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fsig%5F"
            "info%5F2017%2D03%2D06%2Etxt%2Egz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fgene%5"
            "Finfo%5F2017%2D03%2D06%2Etxt%2Egz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fpert%5"
            "Finfo%5F2017%2D03%2D06%2Etxt%2Egz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fcell%5"
            "Finfo%5F2017%2D04%2D28%2Etxt%2Egz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5FLevel5"
            "%5FCOMPZ%2EMODZ%5Fn473647x12328%2Egctx%2Egz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5Fsig%5F"
            "info%2Etxt%2Egz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5Fpert%5"
            "Finfo%2Etxt%2Egz",
        ]

        for i in range(len(urls)):
            download_url(urls[i], self.raw_dir)
            with gzip.open(
                os.path.join(self.raw_dir, urls[i].split("/")[-1]), "rb"
            ) as f_in:
                with open(
                    os.path.join(self.raw_dir, self.raw_file_names[i]), "wb"
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def process(self):
        # Load metadata
        self.sig_info, self.landmark_gene_list = self.load_metadata()
        self.expr_data = pd.concat(
            [self.load_expr_data("phase1"), self.load_expr_data("phase2")], sort=False
        )

        # Concat with metadata
        data_df = pd.concat((self.expr_data, self.sig_info), axis=1)

        # Extract numpy arrays
        gene_expr = data_df[data_df.columns[: len(self.landmark_gene_list)]].to_numpy()
        cid = data_df["cid"].to_numpy()
        cell_id = data_df["cell_id"].to_numpy()
        dose = data_df["pert_idose_value"].to_numpy()
        incub_time = data_df["pert_itime_value"].to_numpy()

        # Create data object
        data = Data(
            gene_expr=torch.tensor(gene_expr, dtype=torch.float),
            cid=torch.tensor(cid, dtype=torch.long),
            cell_id=cell_id,
            dose=torch.tensor(dose, dtype=torch.float),
            incub_time=torch.tensor(incub_time, dtype=torch.float),
        )

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def load_metadata(self):
        """
        :return:
            - sig_info: dataframe containing metadata information about the examples ("signatures") of the L1000 dataset
            - landmark_gene_list: List of landmark genes (i.e. genes which have actually been measured and not infered)
        """
        # Get list of landmark genes. Gene info and cell info are the same for both phases
        gene_info = pd.read_csv(
            os.path.join(self.raw_dir, self.raw_file_names[2]), sep="\t"
        )
        landmark_gene_list = gene_info[gene_info["pr_is_lm"] == 1]["pr_gene_id"].astype(
            str
        )

        # Load pert_info
        pert_info_with_cid_path = os.path.join(self.raw_dir, "pert_info_with_cid.txt")

        ##################################################################
        # Retrieve CID of the drugs from PubChem
        ##################################################################

        if os.path.isfile(pert_info_with_cid_path):
            pert_info = pd.read_csv(pert_info_with_cid_path, index_col="pert_id")
        else:
            print("Retrieving cids from PubChem, only happens the first time...")
            import pubchempy as pcp

            # Load both phases
            pert_info_1 = pd.read_csv(
                os.path.join(self.raw_dir, self.raw_file_names[3]),
                sep="\t",
                index_col="pert_id",
                usecols=["pert_id", "canonical_smiles"],
            )
            pert_info_2 = pd.read_csv(
                os.path.join(self.raw_dir, self.raw_file_names[7]),
                sep="\t",
                index_col="pert_id",
                usecols=["pert_id", "canonical_smiles"],
            )
            pert_info = pd.concat([pert_info_1, pert_info_2])
            # Remove duplicate indices
            pert_info = pert_info.loc[~pert_info.index.duplicated(keep="first")]

            # Remove examples for which smiles are missing
            pert_info = pert_info[pert_info["canonical_smiles"] != "-666"]
            pert_info["cid"] = -1
            for i in tqdm(pert_info.index):
                try:
                    pert_info.at[i, "cid"] = pcp.get_compounds(
                        pert_info.at[i, "canonical_smiles"], "smiles"
                    )[0].cid
                except:
                    pass
            pert_info.to_csv(pert_info_with_cid_path)

        ##################################################################
        # Load sig_info for both phases
        ##################################################################

        sig_info_1 = pd.read_csv(
            os.path.join(self.raw_dir, self.raw_file_names[1]),
            sep="\t",
            index_col="sig_id",
            usecols=["sig_id", "pert_id", "cell_id", "pert_idose", "pert_itime"],
        )
        sig_info_2 = pd.read_csv(
            os.path.join(self.raw_dir, self.raw_file_names[6]),
            sep="\t",
            index_col="sig_id",
            usecols=["sig_id", "pert_id", "cell_id", "pert_idose", "pert_itime"],
        )
        sig_info = pd.concat([sig_info_1, sig_info_2])

        # Convert time to float and add to sig_info
        sig_info["pert_itime_value"] = sig_info["pert_itime"].apply(self.get_time)
        # Convert concentrations to float and add to sig_info
        sig_info["pert_idose_value"] = sig_info["pert_idose"].apply(
            self.get_concentration
        )

        # Add cid to sig_info
        pert_to_cid_dict = pert_info["cid"].to_dict()
        sig_info["cid"] = sig_info["pert_id"].apply(
            lambda s: pert_to_cid_dict[s] if s in pert_to_cid_dict.keys() else -1
        )
        sig_info = sig_info[sig_info["cid"] != -1]

        return sig_info, landmark_gene_list

    def load_expr_data(self, phase):
        """
        Load differential gene expression profiles (in a dataframe) from one of the two phases of the L1000 dataset
        """
        assert phase in ["phase1", "phase2"]
        if phase == "phase1":
            df_path = os.path.join(self.raw_dir, "dataframe_phase1.pkl")
            file_name = self.raw_file_names[5]
        else:
            df_path = os.path.join(self.raw_dir, "dataframe_phase2.pkl")
            file_name = self.raw_file_names[0]
        if os.path.isfile(df_path):
            pickle_in = open(df_path, "rb")
            expr_data = pickle.load(pickle_in)
        else:  # If the data has not been saved yet, parse the original file and save dataframe
            print("Parsing original data, only happens the first time...")
            from cmapPy.pandasGEXpress.parse import parse

            expr_data = parse(
                os.path.join(self.raw_dir, file_name), rid=self.landmark_gene_list
            ).data_df.T
            # Ensure that the order of columns corresponds to landmark_gene_list
            expr_data = expr_data[self.landmark_gene_list]
            # Remove rows that are not in sig_info
            expr_data = expr_data[expr_data.index.isin(self.sig_info.index)]

            # Save data
            pickle_out = open(df_path, "wb")
            pickle.dump(expr_data, pickle_out, protocol=2)
            pickle_out.close()
        return expr_data

    def get_concentration(self, s):
        if s.endswith("µM") or s.endswith("um"):
            return float(s[:-3])
        if s.endswith("nM"):
            return 0.001 * float(s[:-3])
        return -1

    def get_time(self, s):
        return float(s[:-2])


if __name__ == "__main__":
    dataset = L1000()
