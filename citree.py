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

from dataclasses import dataclass, field
from functools import singledispatch
from math import log
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.cluster import KMeans
from sortedcontainers import SortedList

# define distribution as a new type consisting of the three possible distributions
# the quotes are needed to avoid circular imports (as they are defined later below)
Node = Union["NormalNode", "PoissonNode", "MultinomialNode"]
NodePair = Union["NormalPair", "PoissonPair", "MultinomialPair"]


@dataclass(frozen=True)
class NormalNode:
	id: float
	height: float
	count: int
	left: Optional[NormalNode]
	right: Optional[NormalNode]
	x: ndarray
	V: ndarray
	inv_V: ndarray
	
	def __post_init__(self):
		if self.height < 0:
			raise ValueError("Height must be non-negative")
		if self.count < 1:
			raise ValueError("Count must be positive")
		if self.V.shape[0] != self.V.shape[1]:
			raise ValueError("Variance-covariance matrix must be square")
		if self.x.shape[0] != self.V.shape[0]:
			raise ValueError("Mean vector and variance-covariance matrix must have the same number of elements")
		if self.inv_V.shape[0] != self.inv_V.shape[1]:
			raise ValueError("Inverse of variance-covariance matrix must be square")
		if self.inv_V.shape[0] != self.V.shape[0]:
			raise ValueError(
					"Inverse of variance-covariance matrix and variance-covariance matrix must have the same number of elements")
	
	def __eq__(self, other):
		if not isinstance(other, self.__class__):
			return False
		is_x_equal = np.array_equal(self.x, other.x)
		if not is_x_equal:
			return False
		is_V_equal = np.array_equal(self.V, other.V)
		if not is_V_equal:
			return False
		is_inv_V_equal = np.array_equal(self.inv_V, other.inv_V)
		if not is_inv_V_equal:
			return False
		return True


@dataclass(frozen=True)
class PoissonNode:
	id: float
	height: float
	count: int
	left: Optional[PoissonNode]
	right: Optional[PoissonNode]
	n: float
	N: float
	
	def __post_init__(self):
		if self.height < 0:
			raise ValueError("Height must be non-negative")
		if self.count < 1:
			raise ValueError("Count must be positive")
		object.__setattr__(self, "ln_n_over_N", log(self.n / self.N))
	
	def __eq__(self, other):
		if not isinstance(other, self.__class__):
			return False
		return (self.n == other.n) and (self.N == other.N)


@dataclass(frozen=True)
class MultinomialNode:
	id: float
	height: float
	count: int
	left: Optional[MultinomialNode]
	right: Optional[MultinomialNode]
	n: ndarray
	
	def __post_init__(self):
		if self.height < 0:
			raise ValueError("Height must be non-negative")
		if self.count < 1:
			raise ValueError("Count must be positive")
		object.__setattr__(self,
		                   "N",
		                   self.n.sum())
		object.__setattr__(self,
		                   "ln_n_over_N",
		                   np.log(self.n / self.N))
	
	def __eq__(self, other):
		if not isinstance(other, self.__class__):
			return False
		
		is_n_equal = np.array_equal(self.n, other.n)
		return is_n_equal


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
		nodes.append(NormalNode(
				id=cluster,
				height=0.0,
				count=cluster_data.shape[0],
				left=None,
				right=None,
				x=np.mean(cluster_data, axis=0),
				                                     V=(V := np.cov(cluster_data.T)),
				                                     inv_V=np.linalg.inv(V + 1e-6 * np.eye(V.shape[0])), )
		)
	return {"leaf_bins": nodes, "clusters": clusters}


# --------- Define the dictionary of strategies for stage #1 ---------------
# The will be used to create the bins from the data
stage_1_fast_clustering_strategies = {
		"kmeans_vanilla": stage_1_fast_clustering_kmeans,
}


@dataclass(order=True)
class NormalPair:
	distance: float  # distance between the nodes s and t
	node_s: NormalNode = field(compare=False)
	node_t: NormalNode = field(compare=False)

@dataclass(order=True)
class PoissonPair:
	distance: float  # distance between the nodes s and t
	node_s: PoissonNode = field(compare=False)
	node_t: PoissonNode = field(compare=False)

@dataclass(order=True)
class MultinomialPair:
	distance: float  # distance between the nodes s and t
	node_s: MultinomialNode = field(compare=False)
	node_t: MultinomialNode = field(compare=False)


def get_tree_ids_from(node: Node) -> List[Union[int, float]]:
	# Get the left nodes
	if node is None:
		return []
	
	left_ids = [] if node.left is None else get_tree_ids_from(node.left)
	
	# Get the right nodes
	right_ids = [] if node.right is None else get_tree_ids_from(node.right)
	
	# Get the ids
	ids = left_ids + right_ids
	
	# Return the ids if there are any, else return the node id
	return ids if len(ids) > 0 else [node.id]


@singledispatch
def calculate_distance(Bs: Node, Bt: Node) -> float:
	raise NotImplementedError("Unsupported type for distance calculation")


@calculate_distance.register
def _(Bs: NormalNode, Bt: NormalNode) -> float:
	# Calculate the difference between the nodes
	diff = Bs.x - Bt.x
	# Calculate the combined variance
	combined_V = np.linalg.inv(Bs.inv_V + Bt.inv_V)
	# Return the distance
	return diff.T @ combined_V @ diff  # type: ignore


@calculate_distance.register
def _(Bs: PoissonNode, Bt: PoissonNode) -> float:
	return (Bs.n * Bs.ln_n_over_N
	        + Bt.n * Bt.ln_n_over_N
	        - (Bs.n + Bt.n) * log((Bs.n + Bt.n) / (Bs.N + Bt.N)))


@calculate_distance.register
def _(Bs: MultinomialNode, Bt: MultinomialNode) -> float:
	# type: ignore
	return (Bs.n @ Bs.ln_n_over_N
	        + Bt.n @ Bt.ln_n_over_N
	        - (Bs.n + Bt.n) @ np.log((Bs.n + Bt.n) / (Bs.N + Bt.N)))


@singledispatch
def calculate_fusion_representative(node_pair: NodePair, id_number: int) -> Node:
	raise NotImplementedError("Unsupported type for fusion representative")


@calculate_fusion_representative.register
def _(node_pair: NormalPair, id_number: int) -> NormalNode:
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
	return NormalNode(id=id_number,
	                  height=height,
	                  count=Bs.count + Bt.count,
	                  left=Bs,
	                  right=Bt,
	                  x=x_s_t,
	                  V=V_s_t,
	                  inv_V=inv_V_s_t,
	                  )


@calculate_fusion_representative.register
def _(node_pair: PoissonPair, id_number: int) -> PoissonNode:
	Bs = node_pair.node_s
	Bt = node_pair.node_t
	"""
	Calculate the representative of a node pair.

	Args:
		node_pair: The node pair
		id: The id of the representative

	Returns:
		The representative of the node pair
	"""
	return PoissonNode(id=id_number,
	                   height=(
			                   Bs.height
			                   + Bt.height
			                   + node_pair.distance),
	                   count=Bs.count + Bt.count,
	                   left=Bs,
	                   right=Bt,
	                   n=Bs.n + Bt.n,
	                   N=Bs.N + Bt.N, )


@calculate_fusion_representative.register
def _(node_pair: MultinomialPair, id_number: int) -> MultinomialNode:
	Bs = node_pair.node_s
	Bt = node_pair.node_t
	return MultinomialNode(id=id_number,
	                       height=(Bs.height
	                               + Bt.height
	                               + node_pair.distance),
	                       count=Bs.count + Bt.count,
	                       left=Bs,
	                       right=Bt,
	                       n=Bs.n + Bt.n, )

@singledispatch
def make_node_pair(distance: float, node_s: Node, node_t: Node):
	raise NotImplementedError("Unsupported type for nodes")

@make_node_pair.register
def _(node_s: NormalNode, node_t: NormalNode, distance: float) -> NormalPair:
	return NormalPair(distance, node_s, node_t)

@make_node_pair.register
def _(node_s: MultinomialNode, node_t: MultinomialNode, distance: float) -> MultinomialPair:
	return MultinomialPair(distance, node_s, node_t)

@make_node_pair.register
def _(node_s: PoissonNode, node_t: PoissonNode, distance: float, ) -> PoissonPair:
	return PoissonPair(distance, node_s, node_t)

def hierarchical_clustering(bins: List[Node]) -> Tuple[List[Node], ndarray]:
	node_pairs = SortedList()
	next_id = len(bins)
	
	for i, bin_i in enumerate(bins):
		for bin_j in bins[i + 1:]:
			distance = calculate_distance(bin_i, bin_j)
			node_pairs.add(make_node_pair(bin_i, bin_j, distance))
	
	initial_length_of_nodes = len(bins)
	linkage_matrix = np.zeros((initial_length_of_nodes - 1, 4))
	cluster_idx = initial_length_of_nodes
	
	while len(bins) > 1:
		closest_pair = node_pairs.pop(0)
		node_s_id, node_t_id = closest_pair.node_s.id, closest_pair.node_t.id
		
		new_node = calculate_fusion_representative(closest_pair, next_id)
		next_id += 1
		
		bins = [bin for bin in bins if bin.id not in [node_s_id, node_t_id]]
		bins.append(new_node)
		
		# Remove node_pairs containing node ids to be combined
		ids_to_remove = {node_s_id, node_t_id}
		node_pairs = SortedList(
				[pair for pair in node_pairs if not {pair.node_s.id, pair.node_t.id} & ids_to_remove]
		)
		
		for node in bins[:-1]:
			distance = calculate_distance(node, new_node)
			node_pairs.add(make_node_pair(node, new_node, distance))
		
		linkage_matrix[cluster_idx - initial_length_of_nodes] = [
				node_s_id,
				node_t_id,
				closest_pair.distance,
				len(get_tree_ids_from(new_node)),
		]
		cluster_idx += 1
	
	return bins, linkage_matrix


def plot_dendrogram(linkage_matrix: ndarray, labels: Sequence[str]):
	plt.figure(figsize=(10, 7))
	plt.title("Dendrogram")
	plt.xlabel("Index")
	plt.ylabel("Distance")
	
	dendrogram(Z=linkage_matrix, labels=labels, leaf_font_size=8)
	
	plt.show()


def cut_tree(linkage_matrix: ndarray, num_clusters: int, stage_1_cluster_labels: ndarray) -> dict:
	node_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
	row_labels = [node_labels[stage_1_label] for stage_1_label in stage_1_cluster_labels]
	return {"node_labels": node_labels, "row_labels": row_labels}
