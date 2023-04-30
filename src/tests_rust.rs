// ------------------------------------------------------------------------------------------- //
//                                            UNIT TESTS                                       //
// ------------------------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {

    use ndarray::prelude::*;
    use ndarray::{arr1, arr2};
    use std::borrow::Borrow;
    use std::cmp::Ordering;
    use pyo3::prelude::*;

    #[cfg(test)]
    use pretty_assertions::assert_eq;

    use crate::pynodes::OptionalBoxedPyNormalNode;
    use crate::{
                //cluster_multinomial_nodes,
                //cluster_normal_nodes,
                //cluster_poisson_nodes,
                create_nodepairs_sorted_list,
                Node,
                NodePair,
                normal::NormalNode, 
                poisson::PoissonNode,
                multinomial::MultinomialNode, 
                pynodes::PyNormalNode,
                pynodes::Array1Wrapper,
                pynodes::Array2Wrapper,
    };



// ------------------ PyNormalNode migration tests ------------------ //
    
    #[test]
    fn test_convert_PyNormalNode_to_BoxPyNormalNode() -> PyResult<()> {
        // create python code to create a PyNormalNode
        let gil = Python::acquire_gil();
        let py = gil.python();
        let py_normal_node = PyNormalNode::new(1,
                                                            0.0, 
                                                            OptionalBoxedPyNormalNode::from(None),
                                                            OptionalBoxedPyNormalNode::from(None),
                                                            Array1Wrapper(arr1(&[1.0, 1.0])),
                                                            Array2Wrapper(arr2(&[[2.0, 0.0], [0.0, 2.0]])),
                                                            Array2Wrapper(arr2(&[[0.5, 0.0], [0.0, 0.5]])));
        let py_normal_node = py_normal_node.into_py(py);


        println!("py_normal_node: {:?}", py_normal_node);
        Ok(())
    }

    // ------------------ NormalNode struct tests ------------------ //

    #[test]
    fn test_normal_node_new() {
        let x = arr1(&[1.0, 1.0]);
        let v = arr2(&[[2.0, 0.0], [0.0, 2.0]]);
        let inv_v = arr2(&[[0.5, 0.0], [0.0, 0.5]]);

        let node = NormalNode::new(1, 0.0, None, None, x.clone(), v.clone(), inv_v.clone());

        assert_eq!(node.get_id(), 1);
        assert_eq!(node.get_height(), 0.0);
        assert_eq!(node.get_x(), x);
        assert_eq!(node.get_v(), v);
        assert_eq!(node.get_inv_v(), inv_v);
    }

    #[test]
    fn test_normal_node_repr() {
        let normal_node_3 = NormalNode::new(
            3,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            4,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node = NormalNode::new(
            1,
            2.0,
            Some(Box::new(normal_node_3)),
            Some(Box::new(normal_node_4)),
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        assert_eq!(normal_node.__repr__(), "Normal node { id: 1, height: 2, left_child_id: Some(3), right_child_id: Some(4), x: [1.0, 1.0], shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1, v: [[2.0, 0.0],\n [0.0, 2.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), const ndim=2, inv_v: [[0.5, 0.0],\n [0.0, 0.5]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), const ndim=2 }");
    }

    #[test]
    fn test_normal_node_distance() {
        let normal_node1 = NormalNode::new(
            1,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node2 = NormalNode::new(
            1,
            2.0,
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
        let node = PoissonNode::new(1, 0.9, None, None, 0.5, 1.0, 0.01);
        assert_eq!(node.get_id(), 1);
        assert_eq!(node.get_height(), 0.9);
        assert!(node.get_left_child().is_none());
        assert!(node.get_right_child().is_none());
        assert_eq!(node.get_n(), 0.5);
        assert_eq!(node.get_total(), 1.0);
        assert_eq!(node.get_ln_n_over_total(), 0.01);
    }

    #[test]
    fn test_poisson_node_repr() {
        let node_2 = PoissonNode::new(2, 0.9, None, None, 0.5, 1.0, 0.01);
        let node = PoissonNode::new(1, 0.9, Some(Box::new(node_2)), None, 0.5, 1.0, 0.01);
        let expected = "Poisson node { id: 1, height: 0.9, left_child: Some(2), right_child: None, n: 0.5, total: 1, ln_n_over_total: 0.01 }";
        assert_eq!(node.__repr__(), expected);
    }

    #[test]
    fn test_poisson_node() {
        let poisson_node1 =
            PoissonNode::new(1, 0.9, None, None, 2.0, 4.0, ((2.0 / 4.0) as f64).ln());

        let poisson_node2 =
            PoissonNode::new(2, 0.9, None, None, 3.0, 9.0, ((3.0 / 9.0) as f64).ln());
        let dist = poisson_node1.distance(&poisson_node2).unwrap();
        assert_eq!(dist, -3.7266197820967837);
    }

    // ------------------ MultinomialNode struct tests ------------------ //

    #[test]
    fn test_multinomial_node_new() {
        let n = arr1(&[3.0, 7.0]);
        let total = 10.0;
        let ln_n_over_total = arr1(&[-1.2039728043259361, -0.35667494393873267]);
        let node = MultinomialNode::new(
            1,
            0.0,
            None,
            None,
            n.clone(),
            total,
            ln_n_over_total.clone(),
        );

        assert_eq!(node.get_id(), 1);
        assert_eq!(node.get_height(), 0.0);
        assert_eq!(node.get_n(), n);
        assert_eq!(node.get_total(), total);
        assert_eq!(node.get_ln_n_over_total(), n.mapv(|x: f64| x.ln() - total.ln()));
    }

    #[test]
    fn test_repr() {

        let mnode_12 = MultinomialNode::new(
            12,
            2.0,
            None,
            None,
            Array::from_vec(vec![3.0, 7.0, 1.0]),
            11.0,
            Array::from_vec(vec![-1.2459866438544598, -0.6689809884656745, -2.3025850929940455]
            ),
        );

        let mnode_123 = MultinomialNode::new(
            123,
            2.0,
            None,
            None,
            Array::from_vec(vec![3.0, 7.0, 1.0]),
            11.0,
            Array::from_vec(vec![-1.2459866438544598, -0.6689809884656745, -2.3025850929940455]
            ),
        );


        let mnode = MultinomialNode::new(
            42,
            2.0,
            Some(Box::new(mnode_12)),
            Some(Box::new(mnode_123)),
            Array::from_vec(vec![3.0, 7.0, 1.0]),
            11.0,
            Array::from_vec(vec![-1.2459866438544598, -0.6689809884656745, -2.3025850929940455]
            ),
        );

        assert_eq!(mnode.__repr__(), "Multinomial node { id: 42, height: 2, left_child_id: Some(12), right_child_id: Some(123), n: [3.0, 7.0, 1.0], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1, total: 11, ln_n_over_total: [-1.2459866438544598, -0.6689809884656746, -2.3025850929940455], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1 }");
    }

    #[test]
    fn test_multinomial_node() {
        let multinomial_node1 = MultinomialNode::new(
            34,
            2.4,
            None,
            None,
            Array::from_vec(vec![3.0, 2.0, 4.0]),
            9.0,
            Array::from_vec(vec![-1.103304, -1.6094379, -1.2039728]),
        );
        let multinomial_node2 = MultinomialNode::new(
            43,
            4.2,
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
            1,
            0.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );
        let node_t = NormalNode::new(
            2,
            0.4,
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
                assert_eq!(nodepair.node_s.get_id(), 1);
                assert_eq!(nodepair.node_t.get_id(), 2);
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    #[test]
    fn node_pair_partialeq_works() {
        let normal_node_3 = NormalNode::new(
            3,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
            );

        let normal_node_4 = NormalNode::new(
            4,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let pair1 = NodePair::<NormalNode> {
            distance: 10.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone()
        };

        let pair2 = NodePair::<NormalNode> {
            distance: 10.0,
            node_s: normal_node_3.clone(),
            node_t: normal_node_4.clone()
        };

        assert_eq!(pair1, pair2);
    }

    #[test]
    fn node_pair_eq_works() {
        let normal_node_3 = NormalNode::new(
            3,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
            );

        let normal_node_4 = NormalNode::new(
            4,
            2.0,
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
            1,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
            );

        let normal_node_2 = NormalNode::new(
            2,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
            );
        let normal_node_3 = NormalNode::new(
            3,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
            );

        let normal_node_4 = NormalNode::new(
            4,
            2.0,
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
        assert_eq!(pair1.partial_cmp(&pair2), Some(Ordering::Less));
    }

    #[test]
    fn node_pair_ord_works() {

        let normal_node_1 = NormalNode::new(
            1,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
            );

        let normal_node_2 = NormalNode::new(
            2,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
            );

        let normal_node_3 = NormalNode::new(
            3,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            4,
            2.0,
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
        assert_eq!(pair1.cmp(&pair2), Ordering::Less);

    }

    #[test]
    fn test_get_nodes() {

        let normal_node_1 = NormalNode::new(
            1,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_2 = NormalNode::new(
            2,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let pair1 = NodePair::<NormalNode> {
            distance: 15.0,
            node_s: normal_node_1,
            node_t: normal_node_2
        };
        assert_eq!(pair1.get_node_ids(), (1, 2));
    }


    #[test]
    fn test_create_nodepairs_sorted_list() {

        let normal_node_1 = NormalNode::new(
            1,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_2 = NormalNode::new(
            2,
            2.0,
            None,
            None,
            arr1(&[1.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_3 = NormalNode::new(
            3,
            2.0,
            None,
            None,
            arr1(&[5.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let normal_node_4 = NormalNode::new(
            4,
            2.0,
            None,
            None,
            arr1(&[6.0, 1.0]),
            arr2(&[[2.0, 0.0], [0.0, 2.0]]),
            arr2(&[[0.5, 0.0], [0.0, 0.5]]),
        );

        let mut nodes = vec![normal_node_1.clone(), normal_node_2, normal_node_3, normal_node_4.clone()];
        
        let node_pairs = create_nodepairs_sorted_list(nodes.borrow()).unwrap();

        assert_eq!(node_pairs.len(), 6);
        assert_eq!(
            node_pairs.peek().unwrap(),
            &NodePair::new(&normal_node_4, &normal_node_1).unwrap()
        );
    }

    #[test]
    fn get_tree_ids_from_test_basic() {

        let node_1 = PoissonNode::new(1, 0.9, None, None, 0.5, 1.0, 0.01);
        let node_2 = PoissonNode::new(2, 0.9, None, None, 0.5, 1.0, 0.01);
        let node_3 = PoissonNode::new(3, 0.9, None, None, 0.5, 1.0, 0.01);
        let node_4 = PoissonNode::new(4, 0.9, None, None, 0.5, 1.0, 0.01);

        let mut nodes = Vec::new();
        nodes.insert(0, PoissonNode::new(0, 0.0, Some(Box::new(node_1)),  Some(Box::new(node_2)),  0.0, 10.0, 10.0));
        nodes.insert(1, PoissonNode::new(1, 0.0, None,                    Some(Box::new(node_3)), 10.0, 10.0, 10.0));
        nodes.insert(2, PoissonNode::new(2, 0.0, Some(Box::new(node_4)),  None,                   10.0, 10.0, 10.0));
        nodes.insert(3, PoissonNode::new(3, 0.0, None,                    None,                   10.0, 10.0, 10.0));
        nodes.insert(4, PoissonNode::new(4, 0.0, None,                    None,                   10.0, 10.0, 10.0));
        
        let ids = vec![ nodes[0].get_id(),
                                  nodes[1].get_id(),
                                  nodes[2].get_id(),
                                  nodes[3].get_id(),
                                  nodes[4].get_id()];
        assert_eq!(vec![0, 1, 2, 3, 4], ids);
    }
    
    

    // --------------------- get_tree_ids_from ---------------------
    #[test]
    fn get_tree_leaves_from_test_nonexistent_node() {
        let node_1 = PoissonNode::new(1, 0.9, None, None, 0.5, 1.0, 0.01);
        let node_2 = PoissonNode::new(2, 0.9, None, None, 0.5, 1.0, 0.01);
        let node_3 = PoissonNode::new(3, 0.9, None, None, 0.5, 1.0, 0.01);
        let node_4 = PoissonNode::new(4, 0.9, None, None, 0.5, 1.0, 0.01);


        let mut nodes = Vec::new();

        nodes.insert(0, PoissonNode::new(5,
                                                        0.0,
                                                        Some(Box::new(node_1)),
                                                        Some(Box::new(node_2)),   
                                                        10.0,
                                                        10.0,
                                                        10.0));

        nodes.insert(1, PoissonNode::new(6,
                                                        0.0,
                                                        None,
                                                        Some(Box::new(node_3)),
                                                        10.0,
                                                        10.0,
                                                        10.0));

        nodes.insert(2, PoissonNode::new(7,
                                                        0.0,
                                                        Some(Box::new(node_4)),
                                                        None,
                                                        10.0,
                                                        10.0,
                                                        10.0));

        nodes.insert(3, PoissonNode::new(8,
                                                        0.0,
                                                        None,
                                                        None,
                                                        10.0,
                                                        10.0,
                                                        10.0));

        nodes.insert(4, PoissonNode::new(9, 
                                                        0.0,
                                                        None,
                                                        None,
                                                        10.0,
                                                        10.0,
                                                        10.0));
        
        let ids = vec![ nodes[0].get_id(),
                                    nodes[1].get_id(),
                                    nodes[2].get_id(),
                                    nodes[3].get_id(),
                                    nodes[4].get_id()];

        assert_eq!(vec![5, 6, 7, 8, 9], ids);
    }
    
    
    
    



    /*
        fn test_numpy_to_ndarray_according_to_pyo3_docs() {
            pyo3::Python::with_gil(|py| {
                let py_array = array![[1i64, 2], [3, 4]].to_pyarray(py);

                assert_eq!(py_array.readonly().as_array(), array![[1i64, 2], [3, 4]]);
            });
        }
    */


}
