// ------------------------------------------------------------------------------------------- //
//                                            UNIT TESTS                                       //
// ------------------------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {

    use ndarray::prelude::*;
    use ndarray::{arr1, arr2};
    use std::borrow::Borrow;
    use std::cmp::Ordering;

    #[cfg(test)]
    use pretty_assertions::assert_eq;

    use crate::{
        cluster_bins,
        //cluster_multinomial_nodes,
        //cluster_normal_nodes,
        //cluster_poisson_nodes,
        create_nodepairs_sorted_list,
        multinomial::MultinomialNode,
        normal::NormalNode,
        poisson::PoissonNode,
        pynodes::Array1Wrapper,
        pynodes::Array2Wrapper,
        pynodes::PyNormalNode,
        Node,
        NodeCount,
        NodeId,
        NodePair,
    };
    use pyo3::prelude::*;

    // ------------------ PyNormalNode migration tests ------------------ //

    #[test]
    fn test_convert_py_normal_node_to_box_py_normal_node() -> PyResult<()> {
        // create python code to create a PyNormalNode
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_normal_node = PyNormalNode::new(
                1,
                0.0,
                1,
                None,
                None,
                Array1Wrapper(arr1(&[1.0, 1.0])),
                Array2Wrapper(arr2(&[[2.0, 0.0], [0.0, 2.0]])),
                Array2Wrapper(arr2(&[[0.5, 0.0], [0.0, 0.5]])),
            );

            let py_normal_node = py_normal_node.into_py(py);

            //compare with assert eq py normal node id attribute with 1_usize
            assert_eq!(
                py_normal_node.getattr(py, "id")?.extract::<usize>(py)?,
                1_usize
            );

            println!("py_normal_node: {:?}", py_normal_node);

            Ok(())
        })
    }

    // ------------------ Node trait tests ------------------ //
    #[test]
    fn get_tree_ids_with_no_children() {
        // create a node with no children and verify the resulting vec contains just it's id
        let node = PoissonNode::new(NodeId(7), 0.0, NodeCount(1), None, None, 6.0, 6.0, 0.0);
        let tree_ids = node.get_tree_ids();
        assert_eq!(vec![NodeId(7)], tree_ids);
    }

    #[test]
    fn get_tree_ids_with_left_child_only() {
        // create a node with a left child and verify the resulting vec contains the ids of both nodes
        let node_5 = PoissonNode::new(NodeId(5), 0.0, NodeCount(1), None, None, 6.0, 6.0, 0.0);
        let node_7 = PoissonNode::new(
            NodeId(7),
            0.0,
            NodeCount(1),
            Some(Box::new(node_5)),
            None,
            6.0,
            6.0,
            0.0,
        );
        let tree_ids = node_7.get_tree_ids();
        assert_eq!(vec![NodeId(7), NodeId(5)], tree_ids);
    }

    #[test]
    fn get_tree_ids_with_right_child_only() {
        // create a node with a right child and verify the resulting vec contains the ids of both nodes
        let node_6 = PoissonNode::new(NodeId(6), 0.0, NodeCount(1), None, None, 6.0, 6.0, 0.0);
        let node_7 = PoissonNode::new(
            NodeId(7),
            0.0,
            NodeCount(1),
            None,
            Some(Box::new(node_6)),
            6.0,
            6.0,
            0.0,
        );
        let tree_ids = node_7.get_tree_ids();
        assert_eq!(vec![NodeId(7), NodeId(6)], tree_ids);
    }

    #[test]
    fn get_tree_ids_with_both_children() {
        // create a node with both children and verify the resulting vec contains the ids of all three nodes
        let node_5 = PoissonNode::new(NodeId(5), 0.0, NodeCount(1), None, None, 6.0, 6.0, 0.0);
        let node_6 = PoissonNode::new(NodeId(6), 0.0, NodeCount(1), None, None, 6.0, 6.0, 0.0);
        let node_7 = PoissonNode::new(
            NodeId(7),
            0.0,
            NodeCount(1),
            Some(Box::new(node_5)),
            Some(Box::new(node_6)),
            3.0,
            6.0,
            0.0,
        );

        let tree_ids = node_7.get_tree_ids();
        assert_eq!(vec![NodeId(7), NodeId(5), NodeId(6)], tree_ids);
    }

    // ------------------ NormalNode struct tests ------------------ //

    #[test]
    fn test_normal_node_new() {
        let x = arr1(&[1.0, 1.0]);
        let v = arr2(&[[2.0, 0.0], [0.0, 2.0]]);
        let inv_v = arr2(&[[0.5, 0.0], [0.0, 0.5]]);

        let node = NormalNode::new(
            NodeId(1),
            0.0,
            NodeCount(1),
            None,
            None,
            x.clone(),
            v.clone(),
            inv_v.clone(),
        );

        assert_eq!(node.get_id(), NodeId(1));
        assert_eq!(node.get_height(), 0.0);
        assert_eq!(node.get_x(), x);
        assert_eq!(node.get_v(), v);
        assert_eq!(node.get_inv_v(), inv_v);
    }

    #[test]
    fn test_normal_node_repr() {
        let normal_node_3 = NormalNode::new(
            NodeId(3),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            NodeId(4),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node = NormalNode::new(
            NodeId(1),
            2.0,
            NodeCount(1),
            Some(Box::new(normal_node_3)),
            Some(Box::new(normal_node_4)),
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        assert_eq!(normal_node.__repr__(), "Normal node { id: 1, height: 2, count: 1, left_child_id: Some(NodeId(3)), right_child_id: Some(NodeId(4)), x: [1.0, 1.0], shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1, v: [[2.0, 0.0],\n [0.0, 2.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), const ndim=2, inv_v: [[0.5, 0.0],\n [0.0, 0.5]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), const ndim=2 }");
    }

    #[test]
    fn test_normal_node_distance() {
        let normal_node1 = NormalNode::new(
            NodeId(1),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node2 = NormalNode::new(
            NodeId(2),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[3.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );
        let dist = normal_node1.distance(&normal_node2).unwrap();
        assert_eq!(dist, 4.0);
    }

    //  ------------------ PoissonNode struct tests ------------------ //

    #[test]
    fn test_poisson_node_new() {
        let node = PoissonNode::new(NodeId(1), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        assert_eq!(node.get_id(), NodeId(1));
        assert_eq!(node.get_height(), 0.9);
        assert!(node.get_left_child().is_none());
        assert!(node.get_right_child().is_none());
        assert_eq!(node.get_n(), 0.5);
        assert_eq!(node.get_total(), 1.0);
        assert_eq!(node.get_ln_n_over_total(), 0.01);
        assert_eq!(node.get_count(), NodeCount(1));
    }

    #[test]
    fn test_poisson_node_repr() {
        let node_2 = PoissonNode::new(NodeId(2), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        let node = PoissonNode::new(
            NodeId(1),
            0.9,
            NodeCount(1),
            Some(Box::new(node_2)),
            None,
            0.5,
            1.0,
            0.01,
        );
        let expected = "Poisson node { id: 1, height: 0.9, count: 1, left_child: Some(NodeId(2)), right_child: None, n: 0.5, total: 1, ln_n_over_total: 0.01 }";
        assert_eq!(node.__repr__(), expected);
    }

    #[test]
    fn test_poisson_node() {
        let poisson_node1 = PoissonNode::new(
            NodeId(1),
            0.9,
            NodeCount(1),
            None,
            None,
            2.0,
            4.0,
            ((2.0 / 4.0) as f64).ln(),
        );

        let poisson_node2 = PoissonNode::new(
            NodeId(2),
            0.9,
            NodeCount(1),
            None,
            None,
            3.0,
            9.0,
            ((3.0 / 9.0) as f64).ln(),
        );
        let dist = poisson_node1.distance(&poisson_node2).unwrap();
        assert_eq!(dist, 0.0954259980129617);
    }

    // ------------------ MultinomialNode struct tests ------------------ //

    #[test]
    fn test_multinomial_node_new() {
        let n = arr1(&[3.0, 7.0]);
        let total = 10.0;
        let ln_n_over_total = arr1(&[-1.2039728043259361, -0.35667494393873267]);
        let node = MultinomialNode::new(
            NodeId(1),
            0.0,
            NodeCount(1),
            None,
            None,
            n.clone(),
            total,
            ln_n_over_total.clone(),
        );

        assert_eq!(node.get_id(), NodeId(1));
        assert_eq!(node.get_height(), 0.0);
        assert_eq!(node.get_n(), n);
        assert_eq!(node.get_total(), total);
        assert_eq!(
            node.get_ln_n_over_total(),
            n.mapv(|x: f64| x.ln() - total.ln())
        );
    }

    #[test]
    fn test_repr() {
        let mnode_12 = MultinomialNode::new(
            NodeId(12),
            2.0,
            NodeCount(1),
            None,
            None,
            Array::from_vec(vec![3.0, 7.0, 1.0]),
            11.0,
            Array::from_vec(vec![
                -1.2459866438544598,
                -0.6689809884656745,
                -2.3025850929940455,
            ]),
        );

        let mnode_123 = MultinomialNode::new(
            NodeId(123),
            2.0,
            NodeCount(1),
            None,
            None,
            Array::from_vec(vec![3.0, 7.0, 1.0]),
            11.0,
            Array::from_vec(vec![
                -1.2459866438544598,
                -0.6689809884656745,
                -2.3025850929940455,
            ]),
        );

        let mnode = MultinomialNode::new(
            NodeId(42),
            2.0,
            NodeCount(1),
            Some(Box::new(mnode_12)),
            Some(Box::new(mnode_123)),
            Array::from_vec(vec![3.0, 7.0, 1.0]),
            11.0,
            Array::from_vec(vec![
                -1.2459866438544598,
                -0.6689809884656745,
                -2.3025850929940455,
            ]),
        );

        assert_eq!(mnode.__repr__(), "Multinomial node { id: 42, height: 2, count: 1, left_child_id: Some(NodeId(12)), right_child_id: Some(NodeId(123)), n: [3.0, 7.0, 1.0], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1, total: 11, ln_n_over_total: [-1.2459866438544598, -0.6689809884656746, -2.3025850929940455], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1 }");
    }

    #[test]
    fn test_multinomial_node() {
        let multinomial_node1 = MultinomialNode::new(
            NodeId(34),
            2.4,
            NodeCount(1),
            None,
            None,
            Array::from_vec(vec![3.0, 2.0, 4.0]),
            9.0,
            Array::from_vec(vec![-1.103304, -1.6094379, -1.2039728]),
        );
        let multinomial_node2 = MultinomialNode::new(
            NodeId(43),
            4.2,
            NodeCount(1),
            None,
            None,
            Array::from_vec(vec![4.0, 2.0, 1.0]),
            7.0,
            Array::from_vec(vec![-1.386294, -1.6094379, -2.3025851]),
        );
        let dist = multinomial_node1.distance(&multinomial_node2).unwrap();
        assert_eq!(dist, -5.263634394200757);
    }

    // ------------------ NodePair struct tests ------------------ //

    #[test]
    fn test_nodepair_new() {
        let node_s = NormalNode::new(
            NodeId(1),
            0.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );
        let node_t = NormalNode::new(
            NodeId(2),
            0.4,
            NodeCount(1),
            None,
            None,
            arr1(&[-2.0, 2.0]),
            arr2(&[[4.0, 0.0], [0.0, 4.0]]),
            arr2(&[[0.25, 0.0], [0.0, 0.25]]),
        );

        let nodepair = NodePair::new(&node_s, &node_t);
        match nodepair {
            Ok(nodepair) => {
                assert_eq!(nodepair.distance, 13.333333333333334);
                assert_eq!(nodepair.node_s.get_id(), NodeId(1));
                assert_eq!(nodepair.node_t.get_id(), NodeId(2));
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    #[test]
    fn node_pair_partialeq_works() {
        let normal_node_3 = NormalNode::new(
            NodeId(3),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            NodeId(4),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let pair1 = NodePair::<NormalNode> {
            distance: 10.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone(),
        };

        let pair2 = NodePair::<NormalNode> {
            distance: 10.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone(),
        };

        assert_eq!(pair1, pair2);
    }

    #[test]
    fn node_pair_eq_works() {
        let normal_node_3 = NormalNode::new(
            NodeId(3),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            NodeId(4),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let pair1 = NodePair::<NormalNode> {
            distance: 1.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone(),
        };

        let pair2 = NodePair::<NormalNode> {
            distance: 1.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone(),
        };
        assert_eq!((pair1 == pair2), true);
    }

    #[test]
    fn node_pair_partialord_works() {
        let normal_node_1 = NormalNode::new(
            NodeId(1),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_2 = NormalNode::new(
            NodeId(2),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );
        let normal_node_3 = NormalNode::new(
            NodeId(3),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            NodeId(4),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let pair2 = NodePair::<NormalNode> {
            distance: 15.0,
            node_s: normal_node_1.clone(),
            node_t: normal_node_2.clone(),
        };

        let pair1 = NodePair::<NormalNode> {
            distance: 10.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone(),
        };
        assert_eq!(pair2.partial_cmp(&pair1), Some(Ordering::Less));
    }

    #[test]
    fn node_pair_ord_works() {
        let normal_node_1 = NormalNode::new(
            NodeId(1),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_2 = NormalNode::new(
            NodeId(2),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_3 = NormalNode::new(
            NodeId(3),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            NodeId(4),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let pair2 = NodePair::<NormalNode> {
            distance: 15.0,
            node_s: normal_node_1.clone(),
            node_t: normal_node_2.clone(),
        };

        let pair1 = NodePair::<NormalNode> {
            distance: 10.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone(),
        };
        assert_eq!(pair2.cmp(&pair1), Ordering::Less);
    }

    #[test]
    fn test_get_nodes() {
        let normal_node_1 = NormalNode::new(
            NodeId(1),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_2 = NormalNode::new(
            NodeId(2),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let pair1 = NodePair::<NormalNode> {
            distance: 15.0,
            node_s: normal_node_1,
            node_t: normal_node_2,
        };
        assert_eq!(pair1.get_node_ids(), (NodeId(1), NodeId(2)));
    }

    #[test]
    fn test_create_nodepairs_sorted_list() {
        let normal_node_1 = NormalNode::new(
            NodeId(1),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_2 = NormalNode::new(
            NodeId(2),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_3 = NormalNode::new(
            NodeId(3),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[5.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            NodeId(4),
            2.0,
            NodeCount(1),
            None,
            None,
            arr1(&[6.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let nodes = vec![
            normal_node_1.clone(),
            normal_node_2.clone(),
            normal_node_3.clone(),
            normal_node_4.clone(),
        ];

        let node_pairs = create_nodepairs_sorted_list(nodes.borrow()).unwrap();

        assert_eq!(node_pairs.len(), 6);
        assert_eq!(
            node_pairs.peek().unwrap(),
            &NodePair::new(&normal_node_1, &normal_node_2).unwrap()
        );
    }

    #[test]
    fn get_tree_ids_from_test_basic() {
        let node_1 = PoissonNode::new(NodeId(1), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        let node_2 = PoissonNode::new(NodeId(2), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        let node_3 = PoissonNode::new(NodeId(3), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        let node_4 = PoissonNode::new(NodeId(4), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);

        let mut nodes = Vec::new();
        nodes.insert(
            0,
            PoissonNode::new(
                NodeId(0),
                0.0,
                NodeCount(1),
                Some(Box::new(node_1)),
                Some(Box::new(node_2)),
                0.0,
                10.0,
                10.0,
            ),
        );
        nodes.insert(
            1,
            PoissonNode::new(
                NodeId(1),
                0.0,
                NodeCount(1),
                None,
                Some(Box::new(node_3)),
                10.0,
                10.0,
                10.0,
            ),
        );
        nodes.insert(
            2,
            PoissonNode::new(
                NodeId(2),
                0.0,
                NodeCount(1),
                Some(Box::new(node_4)),
                None,
                10.0,
                10.0,
                10.0,
            ),
        );
        nodes.insert(
            3,
            PoissonNode::new(NodeId(3), 0.0, NodeCount(1), None, None, 10.0, 10.0, 10.0),
        );
        nodes.insert(
            4,
            PoissonNode::new(NodeId(4), 0.0, NodeCount(1), None, None, 10.0, 10.0, 10.0),
        );

        let ids = vec![
            nodes[0].get_id(),
            nodes[1].get_id(),
            nodes[2].get_id(),
            nodes[3].get_id(),
            nodes[4].get_id(),
        ];
        assert_eq!(
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
            ids
        );
    }

    // --------------------- get_tree_ids_from ---------------------
    #[test]
    fn get_tree_leaves_from_test_nonexistent_node() {
        let node_1 = PoissonNode::new(NodeId(1), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        let node_2 = PoissonNode::new(NodeId(2), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        let node_3 = PoissonNode::new(NodeId(3), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);
        let node_4 = PoissonNode::new(NodeId(4), 0.9, NodeCount(1), None, None, 0.5, 1.0, 0.01);

        let mut nodes = Vec::new();

        nodes.insert(
            0,
            PoissonNode::new(
                NodeId(5),
                0.0,
                NodeCount(1),
                Some(Box::new(node_1)),
                Some(Box::new(node_2)),
                10.0,
                10.0,
                10.0,
            ),
        );

        nodes.insert(
            1,
            PoissonNode::new(
                NodeId(6),
                0.0,
                NodeCount(1),
                None,
                Some(Box::new(node_3)),
                10.0,
                10.0,
                10.0,
            ),
        );

        nodes.insert(
            2,
            PoissonNode::new(
                NodeId(7),
                0.0,
                NodeCount(1),
                Some(Box::new(node_4)),
                None,
                10.0,
                10.0,
                10.0,
            ),
        );

        nodes.insert(
            3,
            PoissonNode::new(NodeId(8), 0.0, NodeCount(1), None, None, 10.0, 10.0, 10.0),
        );

        nodes.insert(
            4,
            PoissonNode::new(NodeId(9), 0.0, NodeCount(1), None, None, 10.0, 10.0, 10.0),
        );

        let ids = vec![
            nodes[0].get_id(),
            nodes[1].get_id(),
            nodes[2].get_id(),
            nodes[3].get_id(),
            nodes[4].get_id(),
        ];

        assert_eq!(
            vec![NodeId(5), NodeId(6), NodeId(7), NodeId(8), NodeId(9)],
            ids
        );
    }

    #[test]
    fn test_cluster_normal_bins() {
        let node_0 = NormalNode::new(
            NodeId(0),
            0.0,
            NodeCount(1),
            None,
            None,
            array![5.0, 6.0, -1.0],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
        );

        let node_1 = NormalNode::new(
            NodeId(1),
            0.0,
            NodeCount(1),
            None,
            None,
            array![4.0, -1.0, 6.0],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
        );

        let node_2 = NormalNode::new(
            NodeId(2),
            0.0,
            NodeCount(1),
            None,
            None,
            array![1.0, 1.0, 1.0],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
        );

        let node_3 = NormalNode::new(
            NodeId(3),
            0.0,
            NodeCount(1),
            None,
            None,
            array![1.0, 6.0, 1.0],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
        );

        let bins = vec![node_0, node_1, node_2, node_3];

        let linkage_matrix = cluster_bins(&bins);

        match linkage_matrix {
            Ok(linkage_matrix) => {
                let expected_linkage_matrix = array![
                    [0.0, 3.0, 10.0, 2.0],
                    [2.0, 4.0, 20.0, 3.0],
                    [1.0, 5.0, 35.833333333333336, 4.0],
                ];

                assert_eq!(linkage_matrix, expected_linkage_matrix);
                //assert!(linkage_matrix.iter().eq(expected_linkage_matrix.iter()));
            }
            Err(_) => panic!("Error clustering bins"),
        };
    }

    #[test]
    fn test_cluster_poisson_bins() {
        let node_0 = PoissonNode::new(
            NodeId(0),
            0.0,
            NodeCount(1),
            None,
            None,
            1.0,
            1.0,
            1.0_f64.ln(),
        );
        let node_1 = PoissonNode::new(
            NodeId(1),
            0.0,
            NodeCount(1),
            None,
            None,
            1.0,
            10.0,
            0.1_f64.ln(),
        );
        let node_2 = PoissonNode::new(
            NodeId(2),
            0.0,
            NodeCount(1),
            None,
            None,
            1.0,
            100.0,
            0.01_f64.ln(),
        );
        let node_3 = PoissonNode::new(
            NodeId(3),
            0.0,
            NodeCount(1),
            None,
            None,
            1.0,
            1000.0,
            0.001_f64.ln(),
        );

        let bins = vec![node_0, node_1, node_2, node_3];

        let linkage_matrix = cluster_bins(&bins);

        match linkage_matrix {
            Ok(linkage_matrix) => {
                let expected_linkage_matrix = array![
                    [0.0, 1.0, 1.106911091482805, 2.0],
                    [2.0, 3.0, 1.1069110914828055, 2.0],
                    [4.0, 5.0, 8.691375156114688, 4.0],
                ];

                assert_eq!(linkage_matrix, expected_linkage_matrix);
                //assert!(linkage_matrix.iter().eq(expected_linkage_matrix.iter()));
            }
            Err(_) => panic!("Error clustering bins"),
        };
    }

    #[test]
    fn test_cluster_multinomial_bins() {
        let epsilon = 1e-6;
        let n0 = array![epsilon, epsilon, 2.0];
        let n1 = array![epsilon, 1.0, 2.0];
        let n2 = array![2.0, epsilon, 2.0];
        let n3 = array![2.0, 3.0, epsilon];

        let total0 = n0.sum();
        let total1 = n1.sum();
        let total2 = n2.sum();
        let total3 = n3.sum();

        let ln0 = (&n0 / total0).mapv(f64::ln);
        let ln1 = (&n1 / total1).mapv(f64::ln);
        let ln2 = (&n2 / total2).mapv(f64::ln);
        let ln3 = (&n3 / total3).mapv(f64::ln);

        let node_0 =
            MultinomialNode::new(NodeId(0), 0.0, NodeCount(1), None, None, n0, total0, ln0);
        let node_1 =
            MultinomialNode::new(NodeId(1), 0.0, NodeCount(1), None, None, n1, total1, ln1);
        let node_2 =
            MultinomialNode::new(NodeId(2), 0.0, NodeCount(1), None, None, n2, total2, ln2);
        let node_3 =
            MultinomialNode::new(NodeId(3), 0.0, NodeCount(1), None, None, n3, total3, ln3);

        let bins = vec![node_0, node_1, node_2, node_3];

        let linkage_matrix = cluster_bins(&bins);

        match linkage_matrix {
            Ok(linkage_matrix) => {
                let expected_linkage_matrix = array![
                    [0.0, 1.0, 0.5924557544079021, 2.0],
                    [2.0, 4.0, 2.955983061770836, 3.0],
                    [3.0, 5.0, 7.058627648272106, 4.0],
                ];

                assert_eq!(linkage_matrix, expected_linkage_matrix);
                //assert!(linkage_matrix.iter().eq(expected_linkage_matrix.iter()));
            }
            Err(_) => panic!("Error clustering bins"),
        };
    }
}
