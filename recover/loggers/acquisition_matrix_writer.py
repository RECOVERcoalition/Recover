from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import pairwise_distances
from scipy.cluster import hierarchy
from recover.utils import to_zero_base
from recover.loggers.tbx_logger import TBXWriter
from tensorboardX.utils import figure_to_image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from argparse import ArgumentError
import torch
from scipy.spatial.distance import squareform

TRAINING_ITERATION = "training_iteration"


def get_acquisition_matrix_figure(acq_mtx, fig, ax):
    """Add the acquisition matrix to the figure"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    im = ax.imshow(acq_mtx, cmap="GnBu")
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("Acquisition Status")


def get_synergy_heatmap_figure(syn_mtx, fig, ax):
    """Add the synergy heatmap to the figure"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    im = ax.imshow(syn_mtx, cmap="inferno")
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("Synergy")


def _get_all_drug_idx_to_new_idx_tensor(num_drugs, matching_line_ddi):
    """
    Returns a tensor of shape (num_drugs,) wherein index i of the tensor
    indicates the index of the zero-based ddi for the cell line
    this logger is using.  If a drug at index j is not in the cell
    line specific ddi (matching_line_ddi), then its index in this vector
    has the value -1.

    """
    result = torch.full((num_drugs,), -1, dtype=torch.long)

    uniq_matching_drugs = torch.unique(matching_line_ddi, sorted=True)
    result[uniq_matching_drugs] = torch.arange(uniq_matching_drugs.shape[0])

    return result


def _get_synergy_matrix(synergy_scores, matching_ddi, match_idx, xind=None, yind=None):
    """
    Returns a tensor of shape (num_drugs, num_drugs) wherein entry (i, j) is
    the synergy of the experiment for drugs d_i and d_j.  If xind and yind are
    not None (they would both be indexing tensors of shape (num_drugs,)), then
    we will re-order the matrix according to xind and yind.
    """
    num_match_drugs = torch.unique(matching_ddi).shape[0]
    resps = synergy_scores[match_idx]

    result = torch.full(
        (num_match_drugs, num_match_drugs), 0.0, device=matching_ddi.device
    )
    result[matching_ddi[0], matching_ddi[1]] = resps

    if xind is not None and yind is not None:
        return _reindex_mtx(result, xind, yind)
    else:
        return result


def _reindex_mtx(mtx, xind, yind):
    if isinstance(mtx, torch.Tensor):
        df = pd.DataFrame(mtx.cpu().numpy())
    else:
        df = pd.DataFrame(mtx)

    return df.iloc[yind, xind].to_numpy()


def _get_pairwise_distance_matrix(x_drugs, matching_ddi, metric):
    """
    Returns a square matrix where entry (i, j) is the distance
    between drugs d_i and d_j according to distance metric `metric`.
    """
    uniq_matches = torch.unique(matching_ddi, sorted=True)
    matching_x = x_drugs[uniq_matches]

    return pairwise_distances(matching_x.cpu().numpy(), metric=metric)


def _get_distance_matrix(metric, data, matching_ddi, data_ddi_match_idx):
    """
    Returns a square distance matrix based on the desired metric.  If
    the metric is "synergy", the values of the matrix will just be synergy
    scores.  Otherwise they will be pairwise distances based on the morgan
    fingerprint and desired metric.
    """
    metric = metric.lower()
    valid_metrics = {"synergy", "tanimoto", "jaccard", "l2"}
    if metric not in valid_metrics:
        raise ArgumentError(
            "Metric '%s' is invalid.  Must be in %s" % (metric, valid_metrics)
        )

    if metric == "synergy":
        return _get_synergy_matrix(
            data.ddi_edge_response, matching_ddi, data_ddi_match_idx
        )
    elif metric == "tanimoto" or metric == "jaccard":
        # Only use fingerprint to compute the distance
        return _get_pairwise_distance_matrix(
            data.x_drugs[:, : data.fp_bits].type(torch.bool), matching_ddi, "jaccard"
        )
    else:
        return _get_pairwise_distance_matrix(
            data.x_drugs[:, : data.fp_bits], matching_ddi, "l2"
        )


def _get_clustered_ordering(distance_mtx, mode):
    """
    Returns an ordering of indices according to hierarchical clustering.
    The clustering is done according to the passed in distance_mtx.
    """
    valid_modes = {"row", "col"}
    if mode not in valid_modes:
        raise ArgumentError(
            "Mode '%s' is invalid.  Must be in %s" % (mode, valid_modes)
        )

    if isinstance(distance_mtx, torch.Tensor):
        distance_mtx = distance_mtx.cpu().numpy()

    if mode == "col":
        distance_mtx = distance_mtx.T

    linkage = hierarchy.linkage(squareform(distance_mtx))
    dendrogram = hierarchy.dendrogram(linkage, no_plot=True)

    return dendrogram["leaves"]


class AcquisitionMatrixWriter(TBXWriter):
    """
    Writes a matrix of the drugs which have been acquired, as well as a
    heatmap of the drug pairs' synergies.  The writer does this after the
    conclusion of each call to step() on the trainable.
    """

    def _init(self):
        """
        Gets the ddi_edge_idx for the specific cell line, then
        reindes that index to a zero-base (e.g., if the tensor was
        [3, 9, 12], the zero-base tensor would be [0, 1, 2]).

        Finally, the method sets up the synergy matrix and clustered
        row/col orders for writing.
        """
        cell_line_idx = self.data.cell_line_to_idx_dict[
            self.config["acquisition_mtx_cell_line_name"]
        ]
        idxs = self.data.ddi_edge_classes == cell_line_idx
        matching_line_ddi = self.data.ddi_edge_idx[:, idxs]
        zero_base_ddi = to_zero_base(matching_line_ddi)

        # A tensor of shape (num_drugs,) wherein index i of the tensor
        # indicates the index of the zero-based ddi for the cell line
        # this logger is using.  If a drug at index j is not in the cell
        # line specific ddi, then its index in this vector has the value -1.
        self.all_drug_idx_to_new_idx = _get_all_drug_idx_to_new_idx_tensor(
            num_drugs=self.data.ddi_edge_idx.max() + 1,
            matching_line_ddi=matching_line_ddi,
        )

        self.have_warned = False

        metric = self.config.get("acquisition_matrix_similarity_metric", "synergy")
        distance_mtx = _get_distance_matrix(metric, self.data, zero_base_ddi, idxs)

        self._row_idx_order = _get_clustered_ordering(distance_mtx, mode="row")
        self._col_idx_order = _get_clustered_ordering(distance_mtx, mode="col")
        self._synergy_matrix = _get_synergy_matrix(
            self.data.ddi_edge_response,
            zero_base_ddi,
            idxs,
            self._row_idx_order,
            self._col_idx_order,
        )

    def write_result(self, result):
        """
        Gets the acquisition matrix, puts the synergy heatmap and acquisition
        matrix into a figure, writes the figure as an image, then writes the
        image to tensorboard.
        """
        if self._file_writer is None:
            return

        seen_idxs = result.get("seen_idxs")
        if seen_idxs is None:
            # Don't want a ton of noisy logs, so just write on the first epoch
            if not self.have_warned:
                self.have_warned = True
                print(
                    "Did not have seen_idxs in result dict. "
                    "Will not log acquisition matrix this experiment."
                )

            return

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 11))

        synergy_heatmap_fig = get_synergy_heatmap_figure(self._synergy_matrix, fig, ax1)

        acquisition_mtx = self._get_acquisition_mtx(seen_idxs)
        acquisition_mtx_fig = get_acquisition_matrix_figure(acquisition_mtx, fig, ax2)

        mtx_img = figure_to_image(fig)

        step = result.get(TRAINING_ITERATION) or result[TRAINING_ITERATION]
        self._file_writer.add_image(
            "Acquisition Matrix and Synergy Map", mtx_img, global_step=step
        )

    def _get_acquisition_mtx(self, seen_idxs):
        """
        Sets the drug pairs which have been acquired to 1 in
        an 0-1 acquisition matrix and returns it.
        """
        seen_idxs = self._to_cell_line_spec_seen_idxs(seen_idxs)

        num_drugs = self._synergy_matrix.shape[0]
        acq_mtx = torch.zeros((num_drugs, num_drugs))
        acq_mtx[seen_idxs[0], seen_idxs[1]] = 1

        return _reindex_mtx(acq_mtx, self._row_idx_order, self._col_idx_order)

    def _to_cell_line_spec_seen_idxs(self, seen_idxs):
        """
        Gets the drug pairs which were acquired on the cell line we care about.
        """
        re_index = self.all_drug_idx_to_new_idx[seen_idxs]

        # A drug pair is only valid if both drugs in it exist for the cell line
        is_valid = (re_index != -1).all(dim=0)
        re_index = re_index[:, is_valid]

        # Make edges go both directions for matrix vis
        return torch.cat((re_index, re_index.flip((0,))), dim=1)
