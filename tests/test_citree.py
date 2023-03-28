
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
from citree import (Node,
                    NodePairCitree,
                    stage_1_fast_clustering_strategies,
                    stage_2_strategies,
                    stage_1_fast_clustering_kmeans,
                    get_tree_ids_from,
                    calculate_node_fusion_representative_citree,
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
        attributes = vars(Node(0, 0, np.array([1, 2]), np.array([[3, 4], [5, 3]]), np.array([[7, 8], [9, 10]]), None, None))
        self.assertEqual(attributes["id"], 0)
        self.assertEqual(attributes["height"], 0)
        self.assertTrue(np.all(attributes["x"] == np.array([1, 2])))
        self.assertTrue(np.all(attributes["V"] == np.array([[3, 4], [5, 3]])))
        self.assertTrue(np.all(attributes["inv_V"] == np.array([[7, 8], [9, 10]])))
        self.assertIsNone(attributes["left"])
        self.assertIsNone(attributes["right"])

# -------------- NodePairCitree related tests -------------- #
class TestNodePairCitree(unittest.TestCase):
    def setUp(self):
        self.distance = 0.45
        self.node_r = Node(id=12,
                           height=3.5,
                           x=np.array([0, 0]),
                           V=np.array([[1, 0], [0, 1]]),
                           inv_V=np.array([[1, 0], [0, 1]]),
                           left=None,
                           right=None)
        
        self.node_s = Node(id=17,
                           height=8.8,
                           x=np.array([10, 10]),
                           V=np.array([[1, 0], [0, 1]]),
                           inv_V=np.array([[1, 0], [0, 1]]),
                           left=None,
                           right=None)
        
        self.node_t = Node(id=24,
                           height=7.2,
                           x=np.array([11, 10]),
                           V=np.array([[1, 0], [0, 1]]),
                           inv_V=np.array([[1, 0], [0, 1]]),
                           left=None,
                           right=None)
        

    def test_node_pair_citree(self):
        print("\nTesting NodePairCitree attributes")
        result = NodePairCitree(self.distance, self.node_s, self.node_t        )
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
        y_bin = X['Bin']
        y_class = X['Class']

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
            self.assertIsInstance(item, Node)
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


class TestCitree(unittest.TestCase):
    
    def test_calculate_node_fusion_representative_citree(self):
        print("\nTesting calculate_node_fusion_representative_citree")
        V1 = np.array([[2, 2], [2, 4]])
        V2 = np.array([[3, 2], [2, 5]])
        inv_V1 = np.linalg.inv(V1)
        inv_V2 = np.linalg.inv(V2)
        
        node1 = Node(id=1, height=1, x=np.array([1, 2]), V=V1, inv_V=inv_V1, left=None, right=None)
            
        node2 = Node(id=2, height=1, x=np.array([3, 4]), V=V2, inv_V=inv_V2, left=None, right=None)

        expected_new_node = Node(
            id=3,
            height=4.827586206896552, 
            x=np.array([1.82758621, 2.96551724]), 
            V=np.array([[1.17241379, 1.03448276],
                        [1.03448276, 2.20689655]]),
            inv_V=np.array([[ 1.45454545, -0.68181818],
                            [-0.68181818,  0.77272727]]),
            left=node1, right=node2
        )

        new_node_pair = NodePairCitree(distance=2.8275862068965516, node_s=node1, node_t=node2)
        new_node = calculate_node_fusion_representative_citree(new_node_pair, id=3)
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
        y_bin = X['Bin']
        y_class = X['Class']

        # Remove the class and bin labels from the data
        X = X.drop(['Bin', 'Class'], axis=1)
        self.X = X.values
        self.k = 20
        first_stage = stage_1_fast_clustering_strategies[
            'kmeans_vanilla'](data=self.X, number_of_bins=self.k, minimum_cardinality=10)
        
        self.nodes = first_stage['leaf_nodes']
        self.clusters = first_stage['clusters']
        self.stage_2_strategies = stage_2_strategies['citree']
    
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
        plt.scatter(self.X[:,0], self.X[:,1], c=citree['row_labels'], cmap='rainbow')
        plt.show()
        
if __name__ == '__main__':
    unittest.main()