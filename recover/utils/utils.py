import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from torch.utils.data import TensorDataset

np.random.seed(42)

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


"""
Function to get the dataset with the permutation of the combination
"""
def get_tensor_dataset_swapped_combination(data, idxs):

    combinations = data.ddi_edge_idx[:, idxs].T
    reversed_tensor = torch.tensor(np.array([list(reversed(inner_list)) for inner_list in combinations]))

    return TensorDataset(
        reversed_tensor,
        data.ddi_edge_classes[idxs],
        data.ddi_edge_response[idxs],
    )

"""    
Add Gaussian noise to a given percentage of data
Inputs: data, idxs, Noice percentage and std for noice
"""
def add_gaussian_noise(data, idxs, percentage, std_dev=0.1):

    print("Added gaussian noise to ", percentage*100, " percent of training data with ", std_dev, " std.")
    
    target = data.ddi_edge_response[idxs]
    num_samples = len(target)
    num_noisy_samples = int(percentage * num_samples)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    noise = np.random.normal(0, std_dev, size=num_noisy_samples).astype(np.float32)
    target[noisy_indices] += noise
    
    return TensorDataset(
        data.ddi_edge_idx[:, idxs].T,
        data.ddi_edge_classes[idxs],
        torch.tensor(target),
    )

"""
Add Salt and Pepper noise to a given percentage of data
Inputs: data, idxs, Noice percentage and probabilities
"""
def add_salt_and_pepper_noise(data, idxs, percentage, salt_prob=0.01, pepper_prob=0.1):

    print("Added salt pepper noise to ", percentage*100, " percent of training data.")
    
    target = data.ddi_edge_response[idxs]
    num_samples = len(target)
    num_noisy_samples = int(percentage * num_samples)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    salt_indices = np.random.choice(noisy_indices, int(salt_prob * num_noisy_samples), replace=False)
    pepper_indices = np.random.choice(noisy_indices, int(pepper_prob * num_noisy_samples), replace=False)

    target[salt_indices] = 100.0
    target[pepper_indices] = 0.0

    return TensorDataset(
        data.ddi_edge_idx[:, idxs].T,
        data.ddi_edge_classes[idxs],
        torch.tensor(target),
    )

"""
Add random noise to a given percentage of data
Inputs: data, idxs, Noice percentage and Factor at which noice should be introduced
"""
def add_random_noise(data, idxs, percentage, noise_factor=0.1):

    print("Added random noise to ", percentage*100, " percent of training data.")

    target = data.ddi_edge_response[idxs]
    num_samples = len(target)
    num_noisy_samples = int(percentage * num_samples)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    noise = np.random.uniform(-noise_factor, noise_factor, size=num_noisy_samples).astype(np.float32)
    target[noisy_indices] += noise
    
    return TensorDataset(
        data.ddi_edge_idx[:, idxs].T,
        data.ddi_edge_classes[idxs],
        torch.tensor(target),
    )

########################################################################################################################
# Ray Tune
########################################################################################################################


def trial_dirname_creator(trial):
    return str(trial)
