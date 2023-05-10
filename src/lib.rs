use ndarray;
use ndarray::Array2;

use anyhow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::fmt;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use thiserror::Error;
mod multinomial;
mod normal;
mod poisson;
mod pynodes;
mod tests_rust;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeId(usize);

impl From<usize> for NodeId {
    fn from(id: usize) -> Self {
        NodeId(id)
    }
}

impl Deref for NodeId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<usize> for NodeId {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]

pub struct NodeCount(usize);

impl From<usize> for NodeCount {
    fn from(id: usize) -> Self {
        NodeCount(id)
    }
}

impl Deref for NodeCount {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<usize> for NodeCount {
    fn as_ref(&self) -> &usize {
        &self.0
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

    fn get_left_child(&self) -> Option<&Self>;

    fn get_left_child_id(&self) -> Option<NodeId> {
        self.get_left_child().map(|boxed_node| boxed_node.get_id())
    }

    fn get_right_child(&self) -> Option<&Self>;

    fn get_right_child_id(&self) -> Option<NodeId> {
        self.get_right_child().map(|boxed_node| boxed_node.get_id())
    }

    fn merge(&self, other: &Self, id: NodeId, distance: Option<f64>) -> Result<Self, anyhow::Error>
    where
        Self: Sized;

    fn distance(&self, other: &Self) -> Result<f64, anyhow::Error>;

    fn get_leaves_ids(&self) -> Result<Vec<NodeId>, Error> {
        let mut stack = vec![self];
        let mut leaves = Vec::new();

        while let Some(node) = stack.pop() {
            match (node.get_left_child(), node.get_right_child()) {
                (None, None) => {
                    leaves.push(node.get_id());
                }
                (Some(_), None) => {
                    Error::SingleChildNode(node.get_id());
                }
                (None, Some(_)) => {
                    Error::SingleChildNode(node.get_id());
                }

                (Some(left), Some(right)) => {
                    stack.push(&left);
                    stack.push(&right);
                }
            }
        }

        Ok(leaves)
    }

    fn __repr__(&self) -> String;
}

// ------------------------------------- NodePair Struct -------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct NodePair<T: Node + Clone> {
    distance: f64,
    node_s: T,
    node_t: T,
}

// ------------------------------------- NodePair Struct Methods -------------------------------------
impl<T: Node + Clone> NodePair<T> {
    fn new(node_s: &T, node_t: &T) -> Result<Self, anyhow::Error> {
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

    fn merge(&self, id: NodeId) -> Result<T, anyhow::Error> {
        let node_s_t = self.node_s.merge(&self.node_t, id, Some(self.distance))?;
        Ok(node_s_t)
    }
}

impl<T: Node + Clone> Hash for NodePair<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.node_s.get_id(), self.node_t.get_id()).hash(state);
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

#[derive(Error, Debug)]
pub enum Error {
    #[error("Empty node list")]
    EmptyNodeList,
    #[error("Non unique node id: {0}")]
    NonUniqueNodeId(NodeId),
    #[error("Non consecutive node ids")]
    NonConsecutiveNodeIds,
    #[error("Node pair with same id: {0}")]
    NodePairWithSameId(NodeId),
    #[error("Node with single child: {0}")]
    SingleChildNode(NodeId),
}

fn check_node_list<N: Node>(nodes: &[N]) -> Result<(), Error> {
    if nodes.is_empty() {
        return Err(Error::EmptyNodeList);
    }

    // check all nodes are unique
    let mut node_ids: HashSet<NodeId> = HashSet::new();
    for node in nodes {
        if node_ids.contains(&node.get_id()) {
            return Err(Error::NonUniqueNodeId(node.get_id()));
        }
        node_ids.insert(node.get_id());
    }

    // check all ids go from 0 to n-1
    let mut node_ids: Vec<NodeId> = node_ids.into_iter().collect();
    node_ids.sort();
    for (i, node_id) in node_ids.iter().enumerate() {
        if *node_id != NodeId(i) {
            return Err(Error::NonConsecutiveNodeIds);
        }
    }

    Ok(())
}
// This function creates a list of node pairs, sorted by distance, without repetition.
fn create_nodepairs_sorted_list<'a, N: Node + Clone>(
    nodes: &[N],
) -> Result<BinaryHeap<NodePair<N>>, anyhow::Error> {
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

pub fn cluster_bins<N: Node + Clone>(bins: &Vec<N>) -> Result<(Array2<f64>, N), anyhow::Error> {
    check_node_list(bins)?;
    let number_of_leaves = bins.len();
    let expected_number_of_nodes = 2 * number_of_leaves - 1;

    let mut linkage_matrix = Array2::<f64>::zeros((number_of_leaves - 1, 4));
    let mut sorted_nodepairs: BinaryHeap<NodePair<N>> = create_nodepairs_sorted_list(&bins)?;
    let mut bins_copy = Vec::<N>::with_capacity(expected_number_of_nodes);
    bins_copy.extend(bins.iter().map(|node| node.clone()));

    let mut closest_pair: NodePair<N>;
    let mut fusion_node_option: Option<N> = None;
    let mut node_s_id: NodeId;
    let mut node_t_id: NodeId;
    let mut fusion_new_nodepairs: Vec<NodePair<N>>;
    let mut fusion_node_inner: N;

    for current_id in number_of_leaves..expected_number_of_nodes {
        closest_pair = sorted_nodepairs.pop().unwrap();
        fusion_node_inner = closest_pair.merge(NodeId(current_id))?;

        fusion_node_option = Some(fusion_node_inner.clone());

        (node_s_id, node_t_id) = closest_pair.get_node_ids();

        bins_copy.remove_node(node_s_id);
        bins_copy.remove_node(node_t_id);

        bins_copy.push(fusion_node_inner.clone()); // add fusion_node to the end

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
            .map(|node| NodePair::new(node, &fusion_node_inner))
            .collect::<Result<Vec<_>, _>>()?;

        sorted_nodepairs.extend(fusion_new_nodepairs);

        linkage_matrix[[current_id - number_of_leaves, 0]] = node_s_id.0 as f64;
        linkage_matrix[[current_id - number_of_leaves, 1]] = node_t_id.0 as f64;
        linkage_matrix[[current_id - number_of_leaves, 2]] = (&fusion_node_inner).get_height();
        linkage_matrix[[current_id - number_of_leaves, 3]] =
            (&fusion_node_inner).get_count().0 as f64;
    }

    Ok((
        linkage_matrix,
        fusion_node_option.expect("Error: returned fusion node Option is None"),
    ))
}

fn get_whole_tree_ids(node: impl Node) -> Vec<NodeId> {
    let mut stack = Vec::new();
    let mut tree_ids = Vec::new();

    stack.push(&node);

    while let Some(node) = stack.pop() {
        tree_ids.push(node.get_id());

        if let Some(right_child) = node.get_right_child() {
            stack.push(right_child);
        }

        if let Some(left_child) = node.get_left_child() {
            stack.push(left_child);
        }
    }

    tree_ids
}
