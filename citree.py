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
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import namedtuple
from typing import List, Tuple, Dict, Set, Optional
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
import pandas as pd
from sortedcollections import SortedList
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


# ------------------------ Define a node for the tree ------------------------


@dataclass
class Node:
    id: float  # node id
    height: float  # height of the node in the tree
    x: ndarray  # mean of the Gaussian
    V: ndarray  # covariance of the Gaussian
    inv_V: ndarray  # inverse of the covariance of the Gaussian
    left: Optional[Node]  # left child
    right: Optional[Node]  # right child


# ------------------------ Create the bins from the data using k-means ------------------------


def stage_1_fast_clustering_kmeans(data: ndarray,
                                   number_of_bins: int,
                                   minimum_cardinality: int,
                                   ) -> Dict:
    kmeans = KMeans(n_clusters=number_of_bins, n_init='auto').fit(data)
    clusters = kmeans.labels_
    unique_clusters, cluster_sizes = np.unique(clusters, return_counts=True)
    nodes = []
    for cluster, size in zip(unique_clusters, cluster_sizes):
        if size < minimum_cardinality:
            continue
        cluster_data = data[clusters == cluster, :]
        nodes.append(
            Node(
                id=cluster,
                height=0.0,
                x=np.mean(cluster_data, axis=0),
                V=(V := np.cov(cluster_data.T)),
                inv_V=np.linalg.inv(V + 1e-6 * np.eye(V.shape[0])),
                left=None,
                right=None,
            )
        )
    return {'leaf_nodes': nodes, 'clusters': clusters}


# --------- Define the dictionary of strategies for stage #1 ---------------
# The will be used to create the bins from the data
stage_1_fast_clustering_strategies = {
    "kmeans_vanilla": stage_1_fast_clustering_kmeans,
}


# ------------------------ Define a node pair ------------------------
@dataclass
class NodePairCitree:
    distance: float  # distance between the nodes s and t
    node_s: Node
    node_t: Node

    def __lt__(self, other):
        return self.distance < other.distance


# ---- Define the dictionary of functions that will be used to create the node pairs ------------------------

node_pair_classes = {
    "citree": NodePairCitree,
}

# ------------------------ Get the observation id's ------------------------


def get_tree_ids_from(node: Node) -> List[int | float]:
    # Get the left nodes
    left_ids = [] if node.left is None else get_tree_ids_from(node.left)

    # Get the right nodes
    right_ids = [] if node.right is None else get_tree_ids_from(node.right)

    # Get the ids
    ids = left_ids + right_ids

    # Return the ids if there are any, else return the node id
    return ids if len(ids) > 0 else [node.id]


# ------------------------ Calculate the distance between two nodes ------------------------


def calculate_citree_distance(Bs: Node, Bt: Node) -> float:
    # Calculate the difference between the nodes
    diff = Bs.x - Bt.x

    # Calculate the combined variance
    combined_V = np.linalg.inv(Bs.V + Bt.V)

    # Return the distance
    return diff.T @ combined_V @ diff # type: ignore


# ------------------------ Fuse two nodes ------------------------#


def calculate_node_fusion_representative_citree(node_pair: NodePairCitree,
                                                id: int,
                                                ) -> Node:
    Bs = node_pair.node_s
    Bt = node_pair.node_t

    # Calculate the inverse of the variance
    inv_V_s_t = Bs.inv_V + Bt.inv_V

    # Calculate the variance
    V_s_t = np.linalg.inv(inv_V_s_t)

    # Calculate the x
    x_s_t = V_s_t @ (Bs.inv_V @ Bs.x + Bt.inv_V @ Bt.x)

    # Calculate the height
    height = Bs.height + Bt.height + node_pair.distance

    # Return the node
    return Node(id=id,
                height=height,
                x=x_s_t,
                V=V_s_t,
                inv_V=inv_V_s_t,
                left=Bs,
                right=Bt)


# ----- Define helper dictionaries for factories -----

calculate_distance_functions = {
    "citree": calculate_citree_distance,
}


calculate_node_fusion_representative = {
    "citree": calculate_node_fusion_representative_citree,
}


stage_2_strategies = {
    strategy: {'calculate_distance': calculate_distance_functions[strategy],
               'calculate_node_combination': calculate_node_fusion_representative[strategy],
               'Node_Pair_Class': node_pair_classes[strategy],
               } for strategy in ['citree']
}


# ------------------------ Hierarchical clustering ------------------------


def hierarchical_clustering(nodes: List[Node],
                            stage_2_strategies: Dict) -> Tuple[List[Node], ndarray]:

    calculate_distance = stage_2_strategies['calculate_distance']
    calculate_node_combination = stage_2_strategies['calculate_node_combination']
    Node_Pair_Class = stage_2_strategies['Node_Pair_Class']

    node_pairs = SortedList()
    next_id = len(nodes)

    for i, node_i in enumerate(nodes):
        for node_j in nodes[i + 1:]:
            distance = calculate_distance(node_i, node_j)
            node_pairs.add(Node_Pair_Class(distance, node_i, node_j))

    initial_length_of_nodes = len(nodes)
    linkage_matrix = np.zeros((initial_length_of_nodes - 1, 4))
    cluster_idx = initial_length_of_nodes
    
    while len(nodes) > 1:
        closest_pair = node_pairs.pop(0)
        node_s_id, node_t_id = closest_pair.node_s.id, closest_pair.node_t.id

        new_node = calculate_node_combination(closest_pair, next_id)
        next_id += 1

        nodes = [node for node in nodes if node.id not in [node_s_id, node_t_id]]
        nodes.append(new_node)
        
        # Remove node_pairs containing node ids to be combined
        ids_to_remove = {node_s_id, node_t_id}
        node_pairs = SortedList(
            [pair for pair in node_pairs
             if not {pair.node_s.id, pair.node_t.id} & ids_to_remove]
        )

        for node in nodes[:-1]:
            distance = calculate_distance(node, new_node)
            node_pairs.add(Node_Pair_Class(distance, node, new_node))
        
        linkage_matrix[cluster_idx - initial_length_of_nodes] = [
            node_s_id,
            node_t_id,
            closest_pair.distance,
            len(get_tree_ids_from(new_node)),
        ]
        cluster_idx += 1

    return nodes, linkage_matrix


# ------------------------ Plot dendrogram ------------------------


def plot_dendrogram(linkage_matrix: ndarray, labels: List[str]):
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    plt.xlabel("Index")
    plt.ylabel("Distance")

    dendrogram(linkage_matrix, labels=labels, leaf_font_size=8)

    plt.show()

# ------------------------ Cut the dendrogram ------------------------


def cut_tree(linkage_matrix: ndarray, num_clusters: int, stage_1_cluster_labels: ndarray) -> dict:
    node_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    
    row_labels = [node_labels[stage_1_label] for stage_1_label in stage_1_cluster_labels]
    
    return {'node_labels': node_labels, 'row_labels': row_labels }
    
    


# --------------------------------------------------------------------------------------

