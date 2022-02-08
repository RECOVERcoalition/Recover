from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from torch.utils.data import TensorDataset


def get_project_root():
    return Path(__file__).parent.parent.parent


def get_fingerprint(smile, radius, n_bits):
    if smile == "none":
        return np.array([-1] * n_bits)
    try:
        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smile), radius, n_bits
            )
        )
    except Exception as ex:
        return np.array([-1] * n_bits)


########################################################################################################################
# Get TensorDataset
########################################################################################################################


def get_tensor_dataset(data, idxs):

    return TensorDataset(
        data.ddi_edge_idx[:, idxs].T,
        data.ddi_edge_classes[idxs],
        data.ddi_edge_response[idxs],
    )

########################################################################################################################
# Ray Tune
########################################################################################################################


def trial_dirname_creator(trial):
    return str(trial)
