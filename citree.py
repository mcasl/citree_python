# For this algorithm we will use this nomenclature:
#     - 'B_s' is short for node 's'
#     - 'B_t' is short for node 't'
#     - 'B_s_t' is short for node 's_t' that represents the combination of B_s and B_t
#
# These are the names of several variables involved in the calculations:
#
# - 'x_s' is short for the sample mean vector of B_s
# - 'V_s' is short for the variance-covariance matrix of B_s
# - 'inv_V_s' is short for the inverse of 'V_s'
# - 'x_t' is short for the sample mean vector of B_t
# - 'V_t' is short for the variance-covariance matrix of B_t
# - 'inv_V_t' is short for the inverse of 'V_t'
# - 'x_s_t' is short for the mean vector of B_s_t
# - 'V_s_t' is short for the variance-covariance matrix of B_s_t
# - 'h_s_t' is short for the height of B_s_t
# - 'h_s' is short for the height of B_s
# - 'h_t' is short for the height of B_t
# - 'd_s_t' is the distance between B_s and B_t


# --------------- Import libraries --------------
from __future__ import annotations
from sortedcontainers import SortedList
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass, field
from scipy.cluster.hierarchy import fcluster
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from math import log
import citree_rust as ct

# define distribution as a new type consisting of the three possible distributions
# the quotes are needed to avoid circular imports (porque se definen más abajo)
Bin = Union["NormalBin", "PoissonBin", "MultinomialBin"]


# ------------------------ Define a node for the tree ------------------------
@dataclass
class Node:
    def __init__(
        self,
        id: float,  # id is a float to allow decimal notation
        height: float,  # height of the node
        count: int,
        left_child: Optional[Node],  # left child
        right_child: Optional[Node],  # right child
        bin: Optional[Bin] = None,
    ) -> None:  # bin associated with the node
        self.id = id
        self.height = height
        self.count = count
        self.left = left_child
        self.right = right_child
        self.bin = bin


@dataclass
class NormalBin:
    def __init__(self, node: Node, x: ndarray, V: ndarray, inv_V: ndarray) -> None:
        self.node = node
        self.x = x
        self.V = V
        self.inv_V = inv_V
        self.node.bin = self

    def __eq__(self, other):
        return (
            False
            if type(self) != type(other)
            else (
                np.array_equal(self.x, other.x)
                and np.array_equal(self.V, other.V)
                and np.array_equal(self.inv_V, other.inv_V)
            )
        )


@dataclass
class PoissonBin:
    def __init__(self, node: Node, n: int, N: int) -> None:
        self.node = node
        self.n = n
        self.N = N
        self.ln_n_over_N = log(n / N)
        self.node.bin = self


@dataclass
class MultinomialBin:
    def __init__(self, node: Node, n: ndarray) -> None:
        self.node = node
        self.n = n
        self.N = self.n.sum()
        self.ln_n_over_N = np.log(self.n / self.N)
        self.node.bin = self

    def __eq__(self, other):
        return False if type(self) != type(other) else np.array_equal(self.n, other.n)


# ------------------ Create the bins from the data using k-means -------------


def stage_1_fast_clustering_kmeans(
    data: ndarray,
    number_of_bins: int,
    minimum_cardinality: int,
) -> Dict:
    kmeans = KMeans(n_clusters=number_of_bins, n_init="auto").fit(data)
    clusters = kmeans.labels_
    unique_clusters, cluster_sizes = np.unique(clusters, return_counts=True)
    nodes = []
    for cluster, size in zip(unique_clusters, cluster_sizes):
        if size < minimum_cardinality:
            continue
        cluster_data = data[clusters == cluster, :]
        nodes.append(
            NormalBin(
                node=Node(
                    id=cluster,
                    height=0.0,
                    count=cluster_data.shape[0],
                    left_child=None,
                    right_child=None,
                ),
                x=np.mean(cluster_data, axis=0),
                V=(V := np.cov(cluster_data.T)),
                inv_V=np.linalg.inv(V + 1e-6 * np.eye(V.shape[0])),
            )
        )
    return {"leaf_bins": nodes, "clusters": clusters}


# --------- Define the dictionary of strategies for stage #1 ---------------
# The will be used to create the bins from the data
stage_1_fast_clustering_strategies = {
    "kmeans_vanilla": stage_1_fast_clustering_kmeans,
}


# ------------------------ Define a node pair ------------------------
@dataclass(order=True)
class NormalPair:
    distance: float  # distance between the nodes s and t
    bin_s: NormalBin = field(compare=False)
    bin_t: NormalBin = field(compare=False)


@dataclass(order=True)
class PoissonPair:
    distance: float  # distance between the nodes s and t
    bin_s: PoissonBin = field(compare=False)
    bin_t: PoissonBin = field(compare=False)


@dataclass(order=True)
class MultinomialPair:
    distance: float  # distance between the nodes s and t
    bin_s: MultinomialBin = field(compare=False)
    bin_t: MultinomialBin = field(compare=False)


bin_pair_classes = {
    "normal": NormalPair,
    "poisson": PoissonPair,
    "multinomial": MultinomialPair,
}
# ------------------------ Get the observation id's ------------------------


def get_tree_ids_from(bin: Optional[Bin]) -> List[Union[int, float]]:
    # Get the left nodes
    if bin is None:
        return []

    left_ids = [] if bin.node.left is None else get_tree_ids_from(bin.node.left.bin)

    # Get the right nodes
    right_ids = [] if bin.node.right is None else get_tree_ids_from(bin.node.right.bin)

    # Get the ids
    ids = left_ids + right_ids

    # Return the ids if there are any, else return the node id
    return ids if len(ids) > 0 else [bin.node.id]


# ------------------------ Calculate the distance between two nodes --------------------


def calculate_Normal_distance(Bs: NormalBin, Bt: NormalBin) -> float:
    # Calculate the difference between the nodes
    diff = Bs.x - Bt.x

    # Calculate the combined variance
    combined_V = np.linalg.inv(Bs.V + Bt.V)

    # Return the distance
    return diff.T @ combined_V @ diff  # type: ignore


def calculate_Poisson_distance(Bs: PoissonBin, Bt: PoissonBin) -> float:
    return (
        Bs.n * Bs.ln_n_over_N
        + Bt.n * Bt.ln_n_over_N
        - (Bs.n + Bt.n) * log((Bs.n + Bt.n) / (Bs.N + Bt.N))
    )


def calculate_Multinomial_distance(Bs: MultinomialBin, Bt: MultinomialBin) -> float:
    # type: ignore
    return (
        Bs.n @ Bs.ln_n_over_N
        + Bt.n @ Bt.ln_n_over_N
        - (Bs.n + Bt.n) @ np.log((Bs.n + Bt.n) / (Bs.N + Bt.N))
    )


# ------------------------ Fuse two nodes ------------------------#


def calculate_normal_fusion_representative(
    node_pair: NormalPair,
    id: int,
) -> NormalBin:
    Bs = node_pair.bin_s
    Bt = node_pair.bin_t

    # Calculate the inverse of the variance
    inv_V_s_t = Bs.inv_V + Bt.inv_V

    # Calculate the variance
    V_s_t = np.linalg.inv(inv_V_s_t)

    # Calculate the x
    x_s_t = V_s_t @ (Bs.inv_V @ Bs.x + Bt.inv_V @ Bt.x)

    # Calculate the height
    height = Bs.node.height + Bt.node.height + node_pair.distance

    # Return the node
    return NormalBin(
        node=Node(
            id=id,
            height=height,
            count=Bs.node.count + Bt.node.count,
            left_child=Bs.node,
            right_child=Bt.node,
        ),
        x=x_s_t,
        V=V_s_t,
        inv_V=inv_V_s_t,
    )


def calculate_poisson_fusion_representative(
    node_pair: PoissonPair,
    id: int,
) -> PoissonBin:
    """
    Calculate the representative of a node pair.

    Args:
        node_pair: The node pair
        id: The id of the representative

    Returns:
        The representative of the node pair
    """
    return PoissonBin(
        node=Node(
            id=id,
            height=(
                node_pair.bin_s.node.height
                + node_pair.bin_t.node.height
                + node_pair.distance
            ),
            count=node_pair.bin_s.node.count + node_pair.bin_t.node.count,
            left_child=node_pair.bin_s.node,
            right_child=node_pair.bin_t.node,
        ),
        n=node_pair.bin_s.n + node_pair.bin_t.n,
        N=node_pair.bin_s.N + node_pair.bin_t.N,
    )


def calculate_multinomial_fusion_representative(
    node_pair: MultinomialPair,
    id: int,
) -> MultinomialBin:
    return MultinomialBin(
        node=Node(
            id=id,
            height=(
                node_pair.bin_s.node.height
                + node_pair.bin_t.node.height
                + node_pair.distance
            ),
            count=node_pair.bin_s.node.count + node_pair.bin_t.node.count,
            left_child=node_pair.bin_s.node,
            right_child=node_pair.bin_t.node,
        ),
        n=node_pair.bin_s.n + node_pair.bin_t.n,
    )


# ----- Define helper dictionaries for factories -----

calculate_distance_functions = {
    "normal": calculate_Normal_distance,
    "poisson": calculate_Poisson_distance,
    "multinomial": calculate_Multinomial_distance,
}


calculate_node_fusion_representative = {
    "normal": calculate_normal_fusion_representative,
    "poisson": calculate_poisson_fusion_representative,
    "multinomial": calculate_multinomial_fusion_representative,
}


stage_2_strategies = {
    strategy: {
        "calculate_distance": calculate_distance_functions[strategy],
        "calculate_node_combination": calculate_node_fusion_representative[strategy],
        "bin_pair_class": bin_pair_classes[strategy],
    }
    for strategy in ["normal", "poisson", "multinomial"]
}


# ------------------------ Hierarchical clustering ------------------------


def hierarchical_clustering(
    bins: List[Bin], stage_2_strategies: Dict
) -> Tuple[List[Bin], ndarray]:
    calculate_distance = stage_2_strategies["calculate_distance"]
    calculate_node_combination = stage_2_strategies["calculate_node_combination"]
    bin_pair_class = stage_2_strategies["bin_pair_class"]

    node_pairs = SortedList()
    next_id = len(bins)

    for i, bin_i in enumerate(bins):
        for bin_j in bins[i + 1 :]:
            distance = calculate_distance(bin_i, bin_j)
            node_pairs.add(bin_pair_class(distance, bin_i, bin_j))

    initial_length_of_nodes = len(bins)
    linkage_matrix = np.zeros((initial_length_of_nodes - 1, 4))
    cluster_idx = initial_length_of_nodes

    while len(bins) > 1:
        closest_pair = node_pairs.pop(0)
        node_s_id, node_t_id = closest_pair.bin_s.node.id, closest_pair.bin_t.node.id

        new_node = calculate_node_combination(closest_pair, next_id)
        next_id += 1

        bins = [bin for bin in bins if bin.node.id not in [node_s_id, node_t_id]]
        bins.append(new_node)

        # Remove node_pairs containing node ids to be combined
        ids_to_remove = {node_s_id, node_t_id}
        node_pairs = SortedList(
            [
                pair
                for pair in node_pairs
                if not {pair.bin_s.node.id, pair.bin_t.node.id} & ids_to_remove
            ]
        )

        for node in bins[:-1]:
            distance = calculate_distance(node, new_node)
            node_pairs.add(bin_pair_class(distance, node, new_node))

        linkage_matrix[cluster_idx - initial_length_of_nodes] = [
            node_s_id,
            node_t_id,
            closest_pair.distance,
            len(get_tree_ids_from(new_node)),
        ]
        cluster_idx += 1

    return bins, linkage_matrix


def hierarchical_clustering_rust(
    bins: List[Bin],
) -> Tuple[List[Bin], ndarray]:
    return ct.cluster_normal_bins(bins)


# ------------------------ Plot dendrogram ------------------------


def plot_dendrogram(linkage_matrix: ndarray, labels: List[str]):
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    plt.xlabel("Index")
    plt.ylabel("Distance")

    dendrogram(linkage_matrix, labels=labels, leaf_font_size=8)

    plt.show()


# ------------------------ Cut the dendrogram ------------------------


def cut_tree(
    linkage_matrix: ndarray, num_clusters: int, stage_1_cluster_labels: ndarray
) -> dict:
    node_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")

    row_labels = [
        node_labels[stage_1_label] for stage_1_label in stage_1_cluster_labels
    ]

    return {"node_labels": node_labels, "row_labels": row_labels}


# --------------------------------------------------------------------------------------
