
use ndarray;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use ndarray_linalg::krylov::R;
use std::marker::PhantomData;

use std::cmp::Ordering;
use crate::normal::NormalNode;


mod pynodes;
mod normal;
mod poisson;
mod multinomial;
mod tests_rust;
pub use pynodes::{
    PyNormalNode as PyNormalNode,
    PyPoissonNode as PyPoissonNode,
    PyMultinomialNode as PyMultinomialNode,
    cluster_normal_nodes_py,
     
    
};

// ------------------------------------- Node Trait-------------------------------------

pub trait Node {
    fn get_id(&self) -> usize;

    fn get_height(&self) -> f64;

    fn get_left_child(&self) -> &Option<Box<Self>>;

    fn get_left_child_id(&self) -> Option<usize> {
        self.get_left_child().as_ref().map(|boxed_node| {boxed_node.get_id()})
    }

    fn get_right_child(&self) -> &Option<Box<Self>>;
    
    fn get_right_child_id(&self) -> Option<usize> {
        self.get_right_child().as_ref().map(|boxed_node| {boxed_node.get_id()})
    }
    
    fn combine(&self, other: &Self, id: usize, distance: Option<f64>) -> Result<Self, String>
    where
        Self: Sized;

    fn distance(&self, other: &Self) -> Result<f64, String>;

  
    fn get_tree_ids(&self) -> Vec<usize> {
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
    

    
    fn get_tree_leaves_ids(&self) -> Vec<usize> {
        let mut stack = vec![self];
        let mut leaves = Vec::new();
        
        while let Some(node) = stack.pop() {
            match (node.get_left_child(), node.get_right_child()) {
                (None, None) => {
                    leaves.push(node.get_id());
                },
                (Some(left), None) => {
                    stack.push(&left);
                },
                (None, Some(right)) => {
                    stack.push(&right);
                },
                (Some(left), Some(right)) => {
                    stack.push(&left);
                    stack.push(&right);
                },
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
    fn new(
        node_s: &T,
        node_t: &T,
    ) -> Result<Self, String> {
        let distance = node_s.distance(node_t)?;
        Ok(NodePair::<T> {
            distance,
            node_s: node_s.clone(),
            node_t: node_t.clone(),
        })
    }


    fn get_node_ids(&self) -> (usize, usize) {
        (self.node_s.get_id(), self.node_t.get_id())
    }


    fn calculate_fusion_node_representative(&self, id: usize) -> Result<T, String> {
        let node_s_t = self.node_s.combine(&self.node_t, id, Some(self.distance))?;
        Ok(node_s_t)
    }
}



impl<T: Node + Clone> PartialEq for NodePair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: Node + Clone> Eq for NodePair<T> {}

impl<T: Node + Clone> PartialOrd for NodePair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<T: Node + Clone> Ord for NodePair<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// ------------------------------------- Other functions -------------------------------------





// This function creates a list of node pairs, sorted by distance, without repetition.
fn create_nodepairs_sorted_list<'a, T: Node + Clone>(nodes: &Vec<T>) -> Result<std::collections::BinaryHeap<NodePair<T>>, String> {
    let mut nodepairs_sorted_list = std::collections::BinaryHeap::<NodePair<T>>::new();
    for (i, node_s) in nodes.iter().enumerate() {
        for node_t in nodes.iter().skip(i + 1) {
            let nodepair = NodePair::new(node_s, node_t)?;
            nodepairs_sorted_list.push(nodepair);
        }
    }
    Ok(nodepairs_sorted_list)
}




/* 
pub fn cluster_normal_nodes_0(nodes_dict: &mut HashMap<usize, NormalNode>) -> Result<(Array2<f64>, HashMap<usize, NormalNode>), String> {
    let mut nodes_list = nodes_dict.values().collect::<Vec<_>>();
    let expected_number_of_nodes = nodes_list.len() * 2 - 1;
    //let mut nodes_dict = HashMap::<usize, NormalNode>::with_capacity(expected_number_of_nodes);
    //nodes_dict.extend(nodes_dict.iter().map(|(&k, v)| (k, v.clone())));
    


    let mut internodes_sorted_list = create_nodepairs_sorted_list(&nodes_dict)?;
    
    let mut linkage_matrix = Array2::<f64>::zeros((nodes_list.len() - 1, 4));
    let mut number_of_leaves = nodes_list.len();

    for current_id in number_of_leaves..expected_number_of_nodes {
        let closest_pair = internodes_sorted_list.pop().unwrap();
        let node_s_id = closest_pair.node_s_id;
        let node_t_id = closest_pair.node_t_id;
        let new_node = closest_pair.calculate_fusion_node_representative(&nodes_dict, current_id)?;
        nodes_dict.insert(current_id, new_node);
let new_node_ref = nodes_dict.get(&current_id).unwrap();

        nodes_list = nodes_list.into_iter()
            .filter(|bin| bin.id != node_s_id && bin.id != node_t_id)
            .chain(std::iter::once(new_node_ref))
            .collect();

        let new_pairs = nodes_list.iter()
            .take(nodes_list.len() - 1)
            .filter(|node| node.id != node_s_id && node.id != node_t_id)
            .map(|node| {
                NodePair::<NormalNode>::new(node.id, current_id, &mut nodes_dict)
            })
            .collect::<Result<Vec<_>, _>>()?;
        internodes_sorted_list.extend(new_pairs);

        linkage_matrix[[number_of_leaves as usize - nodes_list.len(), 0]] = node_s_id as f64;
        linkage_matrix[[number_of_leaves as usize - nodes_list.len(), 1]] = node_t_id as f64;
        linkage_matrix[[number_of_leaves as usize - nodes_list.len(), 2]] = closest_pair.distance; // TODO: distance or height?
        linkage_matrix[[number_of_leaves as usize - nodes_list.len(), 3]] = get_tree_ids_from(current_id, &nodes_dict).len() as f64;
        number_of_leaves += 1;
    }

    Ok((linkage_matrix, nodes_dict))
}

*/


pub fn cluster_normal_nodes(nodes: &Vec<NormalNode>) -> Result<Array2<f64>, String> {
    
    let number_of_leaves = nodes.len();
    let expected_number_of_nodes = 2 * number_of_leaves - 1;

    let mut nodes_copy = Vec::<NormalNode>::with_capacity(expected_number_of_nodes);
    nodes_copy.extend(nodes.iter().map(|node| node.clone()));

    
    let mut linkage_matrix = Array2::<f64>::zeros((number_of_leaves - 1, 4));
    let mut sorted_nodepairs: std::collections::BinaryHeap<NodePair<NormalNode>> = create_nodepairs_sorted_list(&nodes)?;
    

    for current_id in number_of_leaves..expected_number_of_nodes {
        let closest_pair = sorted_nodepairs.pop().unwrap();
        let fusion_node = closest_pair.calculate_fusion_node_representative(current_id)?;
        let node_s_id = closest_pair.node_s.get_id();
        let node_t_id = closest_pair.node_t.get_id();

        // retain in nodes_copy those nodes not in (node_s_id, node_t_id) and add fusion_node to nodes_copy
        nodes_copy = nodes_copy.into_iter()
            .filter(|node| node.get_id() != node_s_id && node.get_id() != node_t_id)
            .chain(std::iter::once(fusion_node.clone()))
            .collect();
        
        // retain in sorted_nodepairs those nodes not in (node_s_id, node_t_id) and add fusion_node to nodes_copy
        sorted_nodepairs = sorted_nodepairs.into_iter()
            .filter(|nodepair| {
                nodepair.node_s.get_id() != node_s_id && nodepair.node_s.get_id() != node_t_id &&
                nodepair.node_t.get_id() != node_s_id && nodepair.node_t.get_id() != node_t_id
            })
            .collect();

        // calculate new nodepairs for all nodes in nodes_copy (but fusion_node) with fusion_node. nodes_copy is already filtered
        let new_pairs = nodes_copy.iter()
            .take(nodes_copy.len() - 1)
            .map(|node| {
                NodePair::<NormalNode>::new(node, &fusion_node)
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        sorted_nodepairs.extend(new_pairs);
        

        linkage_matrix[[current_id - number_of_leaves as usize, 0]] = closest_pair.node_s.get_id() as f64;
        linkage_matrix[[current_id - number_of_leaves as usize, 1]] = closest_pair.node_t.get_id() as f64;
        linkage_matrix[[current_id - number_of_leaves as usize, 2]] = closest_pair.distance; // TODO: distance or height?
        linkage_matrix[[current_id - number_of_leaves as usize, 3]] = fusion_node.get_tree_leaves_ids().len() as f64;
    }

    Ok(linkage_matrix)
    
}





 
/* 

pub fn cluster_poisson_nodes(nodes_dict: &PoissonNodeRustDict) -> Result<(), String> {
    todo!()
}

pub fn cluster_multinomial_nodes(nodes_dict: &MultinomialNodeRusDict) -> Result<(), String> {
    todo!()
}


*/