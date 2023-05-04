use ndarray;
use ndarray::Array2;

use anyhow::Result;
use std::fmt;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::{Add, Sub};

use std::fmt::Display;

mod multinomial;
mod normal;
mod poisson;
mod pynodes;
mod tests_rust;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(usize);

impl From<usize> for NodeId {
    fn from(id: usize) -> Self {
        NodeId(id)
    }
}

impl Sub<usize> for NodeId {
    type Output = NodeId;

    fn sub(self, other: usize) -> NodeId {
        NodeId(self.0 - other)
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeCount(usize);

impl Add for NodeCount {
    type Output = NodeCount;

    fn add(self, other: NodeCount) -> NodeCount {
        NodeCount(self.0 + other.0)
    }
}

impl Display for NodeCount {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ------------------------------------- Node Trait-------------------------------------

pub trait Node {
    fn get_id(&self) -> NodeId;

    fn get_height(&self) -> f64;

    fn get_count(&self) -> NodeCount;

    fn get_left_child(&self) -> &Option<Box<Self>>;

    fn get_left_child_id(&self) -> Option<NodeId> {
        self.get_left_child()
            .as_ref()
            .map(|boxed_node| boxed_node.get_id())
    }

    fn get_right_child(&self) -> &Option<Box<Self>>;

    fn get_right_child_id(&self) -> Option<NodeId> {
        self.get_right_child()
            .as_ref()
            .map(|boxed_node| boxed_node.get_id())
    }

    fn fuse(&self, other: &Self, id: NodeId, distance: Option<f64>) -> Result<Self>
    where
        Self: Sized;

    fn distance(&self, other: &Self) -> Result<f64>;

    fn get_tree_ids(&self) -> Vec<NodeId> {
        let mut stack = Vec::new();
        let mut tree_ids = Vec::new();

        stack.push(self);

        while let Some(node) = stack.pop() {
            tree_ids.push(node.get_id());

            if let Some(right_child) = node.get_right_child() {
                stack.push(&*right_child);
            }

            if let Some(left_child) = node.get_left_child() {
                stack.push(&*left_child);
            }
        }

        tree_ids
    }

    fn get_tree_leaves_ids(&self) -> Vec<NodeId> {
        let mut stack = vec![self];
        let mut leaves = Vec::new();

        while let Some(node) = stack.pop() {
            match (node.get_left_child(), node.get_right_child()) {
                (None, None) => {
                    leaves.push(node.get_id());
                }
                (Some(left), None) => {
                    stack.push(&left);
                }
                (None, Some(right)) => {
                    stack.push(&right);
                }
                (Some(left), Some(right)) => {
                    stack.push(&left);
                    stack.push(&right);
                }
            }
        }

        leaves
    }

    fn __repr__(&self) -> String;
}

// ------------------------------------- NodePair Struct -------------------------------------

#[derive(Debug, Clone)]
pub struct NodePair<T: Node + Clone> {
    distance: f64,
    node_s: T,
    node_t: T,
}

// ------------------------------------- NodePair Struct Methods -------------------------------------
impl<T: Node + Clone> NodePair<T> {
    fn new(node_s: &T, node_t: &T) -> Result<Self> {
        let distance = node_s.distance(node_t)?;
        Ok(NodePair::<T> {
            distance,
            node_s: node_s.clone(),
            node_t: node_t.clone(),
        })
    }

    fn get_node_ids(&self) -> (NodeId, NodeId) {
        (self.node_s.get_id(), self.node_t.get_id())
    }

    fn fuse(&self, id: NodeId) -> Result<T> {
        let node_s_t = self.node_s.fuse(&self.node_t, id, Some(self.distance))?;
        Ok(node_s_t)
    }
}

impl<T: Node + Clone> PartialEq for NodePair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: Node + Clone> Eq for NodePair<T> {}
// implemented in reverse order to get a min heap
impl<T: Node + Clone> PartialOrd for NodePair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl<T: Node + Clone> Ord for NodePair<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap()
    }
}

impl<T: Node + Clone> fmt::Display for NodePair<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Distance: {:.2}, Node S: {}, Node T: {}",
            self.distance,
            self.node_s.get_id(),
            self.node_t.get_id(),
        )
    }
}
// ------------------------------------- Other functions -------------------------------------

// This function creates a list of node pairs, sorted by distance, without repetition.
fn create_nodepairs_sorted_list<'a, N: Node + Clone>(
    nodes: &[N],
) -> Result<BinaryHeap<NodePair<N>>> {
    let mut nodepairs_sorted_list: BinaryHeap<NodePair<N>> = BinaryHeap::new();
    for (i, node_s) in nodes.iter().enumerate() {
        for node_t in nodes.iter().skip(i + 1) {
            let nodepair = NodePair::new(node_s, node_t)?;
            nodepairs_sorted_list.push(nodepair);
        }
    }
    Ok(nodepairs_sorted_list)
}

trait RemoveNode<N: Node> {
    fn remove_node(&mut self, node_id: NodeId) -> Option<N>;
}

impl<N: Node> RemoveNode<N> for Vec<N> {
    fn remove_node(&mut self, node_id: NodeId) -> Option<N> {
        let node_index = self.iter().position(|node| node.get_id() == node_id)?;
        Some(self.swap_remove(node_index))
    }
}

pub fn cluster_bins<N: Node + Clone>(bins: &Vec<N>) -> Result<Array2<f64>> {
    let number_of_leaves = bins.len();
    let expected_number_of_nodes = 2 * number_of_leaves - 1;

    let mut linkage_matrix = Array2::<f64>::zeros((number_of_leaves - 1, 4));
    let mut sorted_nodepairs: BinaryHeap<NodePair<N>> = create_nodepairs_sorted_list(&bins)?;
    let mut bins_copy = Vec::<N>::with_capacity(expected_number_of_nodes);
    bins_copy.extend(bins.iter().map(|node| node.clone()));

    let mut closest_pair: NodePair<N>;
    let mut fusion_node: N;
    let mut node_s_id: NodeId;
    let mut node_t_id: NodeId;
    let mut fusion_new_nodepairs: Vec<NodePair<N>>;

    for current_id in number_of_leaves..expected_number_of_nodes {
        closest_pair = sorted_nodepairs.pop().unwrap();
        fusion_node = closest_pair.fuse(NodeId(current_id))?;
        (node_s_id, node_t_id) = closest_pair.get_node_ids();

        bins_copy.remove_node(node_s_id);
        bins_copy.remove_node(node_t_id);
        bins_copy.push(fusion_node.clone()); // add fusion_node to the end

        // retain in sorted_nodepairs those nodes not in (node_s_id, node_t_id) and add fusion_node to nodes_copy
        // but retain method is yet experimental as of 2023-05-04
        sorted_nodepairs = sorted_nodepairs
            .into_iter()
            .filter(|nodepair| -> bool {
                let (id_s, id_t) = nodepair.get_node_ids();
                (id_s != node_s_id && id_s != node_t_id) && (id_t != node_s_id && id_t != node_t_id)
            })
            .collect();

        // calculate new nodepairs for all nodes in nodes_copy (but fusion_node) with fusion_node. nodes_copy is already filtered
        fusion_new_nodepairs = bins_copy
            .iter()
            .take(bins_copy.len() - 1)
            .map(|node| NodePair::new(node, &fusion_node))
            .collect::<Result<Vec<_>, _>>()?;

        sorted_nodepairs.extend(fusion_new_nodepairs);

        linkage_matrix[[current_id - number_of_leaves as usize, 0]] = node_s_id.0 as f64;
        linkage_matrix[[current_id - number_of_leaves as usize, 1]] = node_t_id.0 as f64;
        linkage_matrix[[current_id - number_of_leaves as usize, 2]] = fusion_node.get_height();
        linkage_matrix[[current_id - number_of_leaves as usize, 3]] =
            fusion_node.get_tree_leaves_ids().len() as f64;
    }

    Ok(linkage_matrix)
}
