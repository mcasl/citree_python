
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
from numpy import ndarray
from citree import (NormalNode, PoissonNode, MultinomialNode,
                    NormalNodePair, PoissonNodePair, MultinomialNodePair,
                    stage_1_fast_clustering_strategies,
                    stage_2_strategies,
                    stage_1_fast_clustering_kmeans,
                    get_tree_ids_from,
                    calculate_Normal_node_fusion_representative,
                    calculate_Poisson_node_fusion_representative,
                    calculate_Multinomial_node_fusion_representative,
                    calculate_Normal_distance,
                    calculate_Poisson_distance,
                    calculate_Multinomial_distance,
                    hierarchical_clustering,
                    plot_dendrogram,
                    node_pair_classes,
                    cut_tree,
                    )
import sklearn as sk
from sklearn.cluster import KMeans
import unittest
from unittest import TestCase
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# -------------- Node related tests -------------- #

class TestNode(unittest.TestCase):
    def test_attributes(self):
        print("\nTesting Node attributes")
        attributes = vars(NormalNode(0, 0, np.array([1, 2]), np.array([[3, 4], [5, 3]]),
                                     np.array([[7, 8], [9, 10]]), None, None))
        self.assertEqual(attributes["id"], 0)
        self.assertEqual(attributes["height"], 0)
        self.assertTrue(np.all(attributes["x"] == np.array([1, 2])))
        self.assertTrue(np.all(attributes["V"] == np.array([[3, 4], [5, 3]])))
        self.assertTrue(np.all(attributes["inv_V"] == np.array([[7, 8], [9, 10]])))
        self.assertIsNone(attributes["left"])
        self.assertIsNone(attributes["right"])

# -------------- NodePairCitree related tests -------------- #
class TestNormalNodePair(unittest.TestCase):
    def setUp(self):
        self.distance = 0.45
        self.node_r = NormalNode(id=12,
                           height=3.5,
                           x=np.array([0, 0]),
                           V=np.array([[1, 0], [0, 1]]),
                           inv_V=np.array([[1, 0], [0, 1]]),
                           left=None,
                           right=None)
        
        self.node_s = NormalNode(id=17,
                           height=8.8,
                           x=np.array([10, 10]),
                           V=np.array([[1, 0], [0, 1]]),
                           inv_V=np.array([[1, 0], [0, 1]]),
                           left=None,
                           right=None)
        
        self.node_t = NormalNode(id=24,
                           height=7.2,
                           x=np.array([11, 10]),
                           V=np.array([[1, 0], [0, 1]]),
                           inv_V=np.array([[1, 0], [0, 1]]),
                           left=None,
                           right=None)
        

    def test_node_pair_citree(self):
        print("\nTesting NodePairCitree attributes")
        result = NormalNodePair(self.distance, self.node_s, self.node_t        )
        self.assertEqual(result.distance, self.distance)
        self.assertEqual(result.node_s, self.node_s)
        self.assertEqual(result.node_t, self.node_t)

# -------------- Stage 1 Fast Clustering KMeans related tests -------------- #
class Stage1FastClusteringKMeansTests(TestCase):
    def test_stage_1_fast_clustering_kmeans_correctness_v1(self):
        print("\nTesting stage_1_fast_clustering_kmeans correctness v1")
        # Read data from CSV file
        X = pd.read_csv('tests/media/datos.csv', index_col='indice')
        #print(f"X.columns : {X.columns}")
        # Get the class labels and bin labels
        # y_bin = X['Bin']
        # y_class = X['Class']

        # Remove the class and bin labels from the data
        X = X.drop(['Bin', 'Class'], axis=1)
        #print(f"X.columns : {X.columns}")
        #print(f"X.shape : {X.shape}")
        expected_output_ids = list(range(20))
        actual_output = stage_1_fast_clustering_kmeans(data=X.values, number_of_bins=20, minimum_cardinality=10)
        actual_output_ids = [ node.id for node in actual_output['leaf_nodes'] ]
        #print("Expected output:", expected_output)
        #print("Actual output:", actual_output)
        self.assertEqual(expected_output_ids, actual_output_ids)
        
        self.assertIsInstance(actual_output['leaf_nodes'], list)
        for item in actual_output['leaf_nodes']:
            self.assertIsInstance(item, NormalNode)
            self.assertIsNone(item.left)
            self.assertIsNone(item.right)
            self.assertEqual(item.inv_V.shape, (3,3) )

    def test_stage_1_fast_clustering_kmeans_correctness_2(self):
        print("\nTesting stage_1_fast_clustering_kmeans correctness v2")
        # Generate some sample data
        k=20
        minimum_cardinality=10
        X = np.random.rand(1000, 3)
        # Test that the number of bins is correct
        result = stage_1_fast_clustering_kmeans(data=X, number_of_bins=k, minimum_cardinality=minimum_cardinality)
        bins = result['leaf_nodes']
        self.assertEqual(len(bins), k)
        clusters = result['clusters']
        #print(f'clusters: {clusters}')
        # count how many different items are in clusters
        unique_clusters, element_counts = np.unique(clusters, return_counts=True)
        self.assertEqual(
            all(item > minimum_cardinality for item in element_counts), True
        )
        #print(f'bins: {bins}')

        # Test that the IDs of the bins are correct
        bin_ids = [bin.id for bin in bins]
        #print(f'bin_ids: {bin_ids}')
        self.assertEqual(set(bin_ids), set(range(k)))

        # Test that the size of each bin is greater than or equal to the threshold
        bin_sizes = [get_tree_ids_from(bin) for bin in bins]
        #print(f"bin_sizes: {bin_sizes}")
        self.assertTrue(all(bin_sizes))

        


# -------------- Citree related tests -------------- #


class TestNormalCitree(unittest.TestCase):
    
    def test_calculate_node_fusion_representative_Normal(self):
        print("\nTesting calculate_node_fusion_representative_citree")
        V1 = np.array([[2, 2], [2, 4]])
        V2 = np.array([[3, 2], [2, 5]])
        inv_V1 = np.linalg.inv(V1)
        inv_V2 = np.linalg.inv(V2)
        
        node1 = NormalNode(id=1, height=1, x=np.array([1, 2]), V=V1, inv_V=inv_V1, left=None, right=None)
            
        node2 = NormalNode(id=2, height=1, x=np.array([3, 4]), V=V2, inv_V=inv_V2, left=None, right=None)

        expected_new_node = NormalNode(
            id=3,
            height=4.827586206896552, 
            x=np.array([1.82758621, 2.96551724]), 
            V=np.array([[1.17241379, 1.03448276],
                        [1.03448276, 2.20689655]]),
            inv_V=np.array([[ 1.45454545, -0.68181818],
                            [-0.68181818,  0.77272727]]),
            left=node1, right=node2
        )

        new_node_pair = NormalNodePair(distance=2.8275862068965516, node_s=node1, node_t=node2)
        new_node = calculate_Normal_node_fusion_representative(new_node_pair, id=3)
        #print(f'new_node.x: {new_node.x}') 
        #print(f'expected_new_node.x: {expected_new_node.x}')
        #print(f'new_node.V: {new_node.V}')
        #print(f'expected_new_node.V: {expected_new_node.V}')
        #print(f'new_node.inv_V: {new_node.inv_V}')
        #print(f'expected_new_node.inv_V: {expected_new_node.inv_V}')
        self.assertTrue(np.allclose(expected_new_node.x, new_node.x))
        self.assertTrue(np.allclose(expected_new_node.V, new_node.V))
        self.assertTrue(np.allclose(expected_new_node.inv_V, new_node.inv_V))
        self.assertEqual(expected_new_node.height, new_node.height)        
        self.assertEqual(expected_new_node.left, new_node.left)
        self.assertEqual(expected_new_node.right, new_node.right)

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
            'kmeans_vanilla'](data=self.X, number_of_bins=self.k, minimum_cardinality=10)
        
        self.nodes = first_stage['leaf_nodes']
        self.clusters = first_stage['clusters']
        self.stage_2_strategies = stage_2_strategies['normal']
    
    def test_hierarchical_clustering(self):
        node_list, linkage_matrix = hierarchical_clustering(self.nodes, self.stage_2_strategies)
        
        self.assertEqual(len(node_list), 1) # at the end there should be one node
        self.assertEqual(len(linkage_matrix), self.k - 1) # number of pairwise comparisons equal to number of clusters - 1

        # Get the class labels of the leaf nodes in the dendrogram
        dendrogram_classes = get_tree_ids_from(node_list[0])
        print(f'dendrogram_classes: {dendrogram_classes}')
        # Compare the bin labels and the dendrogram classes
        self.assertEqual(set(dendrogram_classes), set(self.clusters))
        
        # Compare the class labels and the observations in the bins

        # plot the dendrogram using the linkage matrix and scipy
        plot_dendrogram(linkage_matrix, labels = [str(i) for i in range(self.k)] )
         
        # Cut the dendrogram to obtain a specific number of clusters
        num_clusters = 3
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        print(f'cluster_labels: {cluster_labels}')
        
        # Compare the class labels and the observations in the bins
        
        citree = cut_tree(linkage_matrix,
                          num_clusters=num_clusters,
                          stage_1_cluster_labels=self.clusters)
        print(f'citree: {citree}')
        
        # plot the dataset X colored by the cluster labels
        # plt.scatter(self.X[:,0], self.X[:,1], c=citree['row_labels'], cmap='rainbow')
        # plt.show()
        
        
class TestPoisson(unittest.TestCase):
    def test_calculate_poisson_distance(self):
        node_1 = PoissonNode(id=1, n=3, N=5,  height=0, left=None, right=None)
        node_2 = PoissonNode(id=2, n=2, N=10, height=0, left=None, right=None)
        expected_result =  0.741709
        result = calculate_Poisson_distance(node_1, node_2)
        self.assertEqual(round(result, 6), expected_result)

    def test_calculate_Poisson_node_fusion_representative(self):
        node_s = PoissonNode(id=1, height=1.5, left=None, right=None, n=2, N=3)
        node_t = PoissonNode(id=2, height=4.0, left=None, right=None, n=4, N=9)
        node_pair = PoissonNodePair(node_s=node_s, node_t=node_t, distance=6.0)
        result_node = calculate_Poisson_node_fusion_representative(node_pair=node_pair,
                                                                   id=3)

        self.assertEqual(result_node.id, 3)
        self.assertEqual(result_node.height, 11.5)
        self.assertEqual(result_node.left, node_pair.node_s)
        self.assertEqual(result_node.right, node_pair.node_t)
        self.assertEqual(result_node.n, 6)
        self.assertEqual(result_node.N, 12)

class TestMultinomial(unittest.TestCase):
    def test_calculate_Multinomial_distance_1(self):
        Bs = MultinomialNode(id=1, height=0, n=np.array([2, 1]), left=None, right=None)
        Bt = MultinomialNode(id=2, height=0, n=np.array([1, 2]), left=None, right=None)
        expected = 0.33979807359079395
        actual = calculate_Multinomial_distance(Bs, Bt)
        self.assertAlmostEqual(actual, expected, places=7)
        
    def test_calculate_Multinomial_distance_2(self):
        Bs = MultinomialNode(id=1, height=0, n=np.array([5, 10]), left=None, right=None)
        Bt = MultinomialNode(id=2, height=0, n=np.array([15, 5]), left=None, right=None)
        expected = 3.1073682477181492 
        actual = calculate_Multinomial_distance(Bs, Bt)
        self.assertAlmostEqual(actual, expected, places=7)
        
    def test_calculate_Multinomial_node_fusion_representative(self):
        # Test case 1:
        node_s = MultinomialNode(id=1,
                                 height=10,
                                 left=None, right=None,
                                 n=np.array([2, 3, 1]))

        node_t = MultinomialNode(id=2,
                                 height=15,
                                 left=None, right=None,
                                 n=np.array([1, 2, 4]))
        
        node_pair = MultinomialNodePair(node_s   = node_s,
                                        node_t   = node_t,
                                        distance = 7)
        id = 3
        expected_node = MultinomialNode(id=3,
                                        height=32,
                                        left=node_s, 
                                        right=node_t, 
                                        n=np.array([3, 5, 5]))
        
        actual_node = calculate_Multinomial_node_fusion_representative(node_pair, id)
        self.assertEqual(actual_node, expected_node)
        
        # Test case 2: 
        node_s=MultinomialNode(id=4, height=14, left=None, right=None,
                               n=np.array([3,6,4]))
        node_t=MultinomialNode(id=5, height=18, left=None, right=None,
                               n=np.array([2,5,4]))
        node_pair = MultinomialNodePair(node_s=node_s,
                                        node_t=node_t,
                                        distance=9)
        id = 7
        expected_node = MultinomialNode(id=7, height=41, left=node_s, right=node_t,
                                        n=np.array([5, 11, 8]))
                                       
        actual_node = calculate_Multinomial_node_fusion_representative(node_pair, id)
        self.assertEqual(actual_node, expected_node)
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    unittest.main()