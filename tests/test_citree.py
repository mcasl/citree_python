# from sklearn.cluster import KMeans
import os
import sys
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

sys.path.append(os.path.abspath(os.path.join('..')))

from citree import (Node, NormalPair, PoissonPair, MultinomialPair,
                    NormalBin, PoissonBin, MultinomialBin,
                    stage_1_fast_clustering_strategies,
                    stage_2_strategies,
                    stage_1_fast_clustering_kmeans,
                    get_tree_ids_from,
                    calculate_normal_fusion_representative,
                    calculate_poisson_fusion_representative,
                    calculate_multinomial_fusion_representative,
                    calculate_Poisson_distance,
                    calculate_Multinomial_distance,
                    hierarchical_clustering,
	# plot_dendrogram,
	                cut_tree,
                    )


# -------------- Node related tests -------------- #
class TestNode(unittest.TestCase):
	def test_attributes(self):
		# print("\nTesting Node attributes")
		attributes = vars(Node(id=0, height=0, count=1, left=None, right=None, ))
		self.assertEqual(attributes["id"], 0)
		self.assertEqual(attributes["height"], 0)
		self.assertEqual(attributes["count"], 1)
		self.assertIsNone(attributes["left"])
		self.assertIsNone(attributes["right"])
	
	def test_height_gt_0(self):
		# print("\nTesting Node height > 0")
		with self.assertRaises(ValueError):
			Node(id=0, height=-1, count=1, left=None, right=None)
	
	def test_count_gt_0(self):
		# print("\nTesting Node count > 0")
		with self.assertRaises(ValueError):
			Node(id=0, height=0, count=0, left=None, right=None)


class Stage1FastClusteringKMeansTests(TestCase):
	def test_stage_1_fast_clustering_kmeans_correctness_v1(self):
		# print("\nTesting stage_1_fast_clustering_kmeans correctness v1")
		# Read data from CSV file
		X = pd.read_csv('tests/media/datos.csv', index_col='indice')
		# print(f"X.columns : {X.columns}")
		# Get the class labels and bin labels
		# y_bin = X['Bin']
		# y_class = X['Class']
		
		# Remove the class and bin labels from the data
		X = X.drop(['Bin', 'Class'], axis=1)
		# print(f"X.columns : {X.columns}")
		# print(f"X.shape : {X.shape}")
		expected_output_ids = list(range(20))
		actual_output = stage_1_fast_clustering_kmeans(data=X.values,
		                                               number_of_bins=20,
		                                               minimum_cardinality=10)
		actual_output_ids = [bin.node.id for bin in actual_output['leaf_bins']]
		# print("Expected output:", expected_output)
		# print("Actual output:", actual_output)
		self.assertEqual(expected_output_ids, actual_output_ids)
		
		self.assertIsInstance(actual_output['leaf_bins'], list)
		for item in actual_output['leaf_bins']:
			self.assertIsInstance(item.node, Node)
			self.assertIsNone(item.node.left)
			self.assertIsNone(item.node.right)
			self.assertEqual(item.inv_V.shape, (3, 3))
	
	def test_stage_1_fast_clustering_kmeans_correctness_2(self):
		# print("\nTesting stage_1_fast_clustering_kmeans correctness v2")
		# Generate some sample data
		k = 20
		minimum_cardinality = 10
		X = np.random.rand(1000, 3)
		# Test that the number of bins is correct
		result = stage_1_fast_clustering_kmeans(data=X,
		                                        number_of_bins=k,
		                                        minimum_cardinality=minimum_cardinality)
		bins = result['leaf_bins']
		self.assertEqual(len(bins), k)
		clusters = result['clusters']
		# print(f'clusters: {clusters}')
		# count how many different items are in clusters
		unique_clusters, element_counts = np.unique(clusters, return_counts=True)
		self.assertEqual(
				all(item > minimum_cardinality for item in element_counts), True
		)
		# print(f'bins: {bins}')
		
		# Test that the IDs of the bins are correct
		bin_ids = [bin.node.id for bin in bins]
		# print(f'bin_ids: {bin_ids}')
		self.assertEqual(set(bin_ids), set(range(k)))
		
		# Test that the size of each bin is greater than or equal to the threshold
		bin_sizes = [get_tree_ids_from(bin) for bin in bins]
		# print(f"bin_sizes: {bin_sizes}")
		self.assertTrue(all(bin_sizes))


# -------------- Citree related tests -------------- #


class TestNormalCitree(unittest.TestCase):
	
	def test_calculate_node_fusion_representative_Normal(self):
		# print("\nTesting calculate_node_fusion_representative_citree")
		V1 = np.array([[2, 2], [2, 4]])
		V2 = np.array([[3, 2], [2, 5]])
		inv_V1 = np.linalg.inv(V1)
		inv_V2 = np.linalg.inv(V2)
		
		bin1 = NormalBin(node=Node(id=1, height=1, count=2, left=None, right=None),
		                 x=np.array([1, 2]),
		                 V=V1,
		                 inv_V=inv_V1,
		                 )
		
		bin2 = NormalBin(node=Node(id=2, height=1, count=3, left=None, right=None),
		                 x=np.array([3, 4]),
		                 V=V2,
		                 inv_V=inv_V2,
		                 )
		
		expected_new_bin = NormalBin(node=Node(id=3,
		                                       height=4.827586206896552,
		                                       count=5,
		                                       left=bin1.node,
		                                       right=bin2.node),
		                             x=np.array([1.82758621, 2.96551724]),
		                             V=np.array([[1.17241379, 1.03448276],
		                                         [1.03448276, 2.20689655]]),
		                             inv_V=np.array([[1.45454545, -0.68181818],
		                                             [-0.68181818, 0.77272727]]),
		                             )
		
		new_bin_pair = NormalPair(distance=2.8275862068965516, bin_s=bin1, bin_t=bin2)
		new_bin = calculate_normal_fusion_representative(new_bin_pair, id=3)
		# print(f'new_node.x: {new_node.x}')
		# print(f'expected_new_node.x: {expected_new_node.x}')
		# print(f'new_node.V: {new_node.V}')
		# print(f'expected_new_node.V: {expected_new_node.V}')
		# print(f'new_node.inv_V: {new_node.inv_V}')
		# print(f'expected_new_node.inv_V: {expected_new_node.inv_V}')
		self.assertTrue(np.allclose(expected_new_bin.x, new_bin.x))
		self.assertTrue(np.allclose(expected_new_bin.V, new_bin.V))
		self.assertTrue(np.allclose(expected_new_bin.inv_V,
		                            new_bin.inv_V))
		self.assertEqual(expected_new_bin.node.height, new_bin.node.height)
		self.assertEqual(expected_new_bin.node.left, new_bin.node.left)
		self.assertEqual(expected_new_bin.node.right, new_bin.node.right)


class TestHierarchicalClustering(unittest.TestCase):
	def setUp(self):
		# Read data from CSV file
		X = pd.read_csv('tests/media/datos.csv', index_col='indice')
		
		# Get the class labels and bin labels
		# y_bin = X['Bin']
		# y_class = X['Class']
		
		# Remove the class and bin labels from the data
		X = X.drop(['Bin', 'Class'], axis=1)
		self.X = X.values
		self.k = 20
		first_stage = stage_1_fast_clustering_strategies[
			'kmeans_vanilla'](data=self.X,
		                      number_of_bins=self.k,
		                      minimum_cardinality=10)
		
		self.nodes = first_stage['leaf_bins']
		self.clusters = first_stage['clusters']
		self.stage_2_strategies = stage_2_strategies['normal']
	
	def test_hierarchical_clustering(self):
		node_list, linkage_matrix = hierarchical_clustering(self.nodes,
		                                                    self.stage_2_strategies)
		
		self.assertEqual(len(node_list), 1)  # at the end there should be one node
		self.assertEqual(len(linkage_matrix), self.k - 1)
		# number of pairwise comparisons equal to number of clusters - 1
		
		# Get the class labels of the leaf nodes in the dendrogram
		dendrogram_classes = get_tree_ids_from(node_list[0])
		# print(f'dendrogram_classes: {dendrogram_classes}')
		# Compare the bin labels and the dendrogram classes
		self.assertEqual(set(dendrogram_classes), set(self.clusters))
		
		# Compare the class labels and the observations in the bins
		
		# plot the dendrogram using the linkage matrix and scipy
		# plot_dendrogram(linkage_matrix, labels = [str(i) for i in range(self.k)] )
		
		# Cut the dendrogram to obtain a specific number of clusters
		num_clusters = 3
		cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
		# print(f'cluster_labels: {cluster_labels}')
		
		# Compare the class labels and the observations in the bins
		
		citree = cut_tree(linkage_matrix,
		                  num_clusters=num_clusters,
		                  stage_1_cluster_labels=self.clusters)


# print(f'citree: {citree}')

# plot the dataset X colored by the cluster labels
# plt.scatter(self.X[:,0], self.X[:,1], c=citree['row_labels'], cmap='rainbow')
# plt.show()


class TestPoisson(unittest.TestCase):
	def test_calculate_poisson_distance(self):
		node_1 = PoissonBin(node=Node(id=1,
		                              height=0,
		                              count=1,
		                              left=None,
		                              right=None),
		                    n=3,
		                    N=5)
		node_2 = PoissonBin(node=Node(id=2,
		                              height=0,
		                              count=1,
		                              left=None,
		                              right=None),
		                    n=2,
		                    N=10)
		expected_result = 0.741709
		result = calculate_Poisson_distance(node_1, node_2)
		self.assertEqual(round(result, 6), expected_result)
	
	def test_calculate_Poisson_node_fusion_representative(self):
		node_s = PoissonBin(node=Node(id=1,
		                              height=1.5,
		                              count=30,
		                              left=None,
		                              right=None),
		                    n=2,
		                    N=3)
		node_t = PoissonBin(node=Node(id=2,
		                              height=4.0,
		                              count=43,
		                              left=None,
		                              right=None),
		                    n=4,
		                    N=9)
		node_pair = PoissonPair(bin_s=node_s,
		                        bin_t=node_t,
		                        distance=6.0)
		result_bin = calculate_poisson_fusion_representative(node_pair=node_pair,
		                                                     id=3)
		self.assertIsNotNone(result_bin)
		self.assertIsNotNone(result_bin.node)
		self.assertIsNotNone(result_bin.node.left)
		self.assertIsNotNone(result_bin.node.right)
		self.assertEqual(result_bin.node.id, 3)
		self.assertEqual(result_bin.node.height, 11.5)
		self.assertEqual(result_bin.node.count, 73)
		if result_bin.node.left is not None:
			self.assertEqual(result_bin.node.left.bin, node_pair.bin_s)
		if result_bin.node.right is not None:
			self.assertEqual(result_bin.node.right.bin, node_pair.bin_t)
		self.assertEqual(result_bin.n, 6)
		self.assertEqual(result_bin.N, 12)


class TestMultinomial(unittest.TestCase):
	def test_calculate_Multinomial_distance_1(self):
		Bs = MultinomialBin(node=Node(id=1, height=0, count=1, left=None, right=None),
		                    n=np.array([2, 1])
		                    )
		Bt = MultinomialBin(node=Node(id=2, height=0, count=1, left=None, right=None),
		                    n=np.array([1, 2])
		                    )
		expected = 0.33979807359079395
		actual = calculate_Multinomial_distance(Bs=Bs, Bt=Bt)
		self.assertAlmostEqual(actual, expected, places=7)
	
	def test_calculate_Multinomial_distance_2(self):
		Bs = MultinomialBin(node=Node(id=1, height=0, count=1, left=None, right=None),
		                    n=np.array([5, 10])
		                    )
		
		Bt = MultinomialBin(node=Node(id=2, height=0, count=1, left=None, right=None),
		                    n=np.array([15, 5])
		                    )
		
		expected = 3.1073682477181492
		actual = calculate_Multinomial_distance(Bs, Bt)
		self.assertAlmostEqual(actual, expected, places=7)
	
	def test_calculate_Multinomial_node_fusion_representative(self):
		# Test case 1:
		bin_s = MultinomialBin(node=Node(id=1,
		                                 height=10,
		                                 count=25,
		                                 left=None, right=None),
		                       n=np.array([2, 3, 1])
		                       )
		
		node_t = MultinomialBin(node=Node(id=2,
		                                  height=15,
		                                  count=52,
		                                  left=None,
		                                  right=None),
		                        n=np.array([1, 2, 4])
		                        )
		
		node_pair = MultinomialPair(bin_s=bin_s,
		                            bin_t=node_t,
		                            distance=7)
		id_ = 3
		expected_bin = MultinomialBin(node=Node(id=3,
		                                        height=32,
		                                        count=77,
		                                        left=bin_s.node,
		                                        right=node_t.node),
		                              n=np.array([3, 5, 5])
		                              )
		
		actual_node = calculate_multinomial_fusion_representative(node_pair, id_)
		self.assertEqual(actual_node, expected_bin)
		
		# Test case 2:
		bin_s = MultinomialBin(node=Node(id=4, height=14, count=1, left=None, right=None),
		                       n=np.array([3, 6, 4])
		                       )
		bin_t = MultinomialBin(node=Node(id=5, height=18, count=1, left=None, right=None),
		                       n=np.array([2, 5, 4])
		                       )
		node_pair = MultinomialPair(bin_s=bin_s,
		                            bin_t=bin_t,
		                            distance=9)
		id_ = 7
		expected_bin = MultinomialBin(node=Node(id=7,
		                                        height=41,
		                                        count=2,
		                                        left=bin_s.node,
		                                        right=bin_t.node),
		                              n=np.array([5, 11, 8])
		                              )
		
		actual_node = calculate_multinomial_fusion_representative(node_pair, id_)
		self.assertEqual(actual_node, expected_bin)


class TestRustModule(unittest.TestCase):
	pass


# def test_sum_int_1(self):
#     x = np.array([1, 2, 3])
#     result = sum_int_1(input_array=x)
#     assert_array_equal(result, np.array([2, 3, 4]))


if __name__ == '__main__':
	unittest.main()
