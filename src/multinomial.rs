use ndarray;
use ndarray::Array1;

use crate::{Node, NodeCount, NodeId};
use anyhow::Result;

// -------------------------- MultinomialNode --------------------------

#[derive(Debug, Clone)]
pub struct MultinomialNode {
    id: NodeId,
    height: f64,
    count: NodeCount,
    left_child: Option<Box<MultinomialNode>>,
    right_child: Option<Box<MultinomialNode>>,
    n: Array1<f64>,
    total: f64,
    ln_n_over_total: Array1<f64>,
}

// Multinomial Node Methods

impl MultinomialNode {
    pub fn new(
        id: NodeId,
        height: f64,
        count: NodeCount,
        left_child: Option<Box<MultinomialNode>>,
        right_child: Option<Box<MultinomialNode>>,
        n: Array1<f64>,
        total: f64,
        ln_n_over_total: Array1<f64>,
    ) -> Self {
        MultinomialNode {
            id,
            height,
            count,
            left_child,
            right_child,
            n,
            total,
            ln_n_over_total,
        }
    }

    pub fn get_n(&self) -> &Array1<f64> {
        &self.n
    }

    pub fn get_total(&self) -> f64 {
        self.total
    }

    pub fn get_ln_n_over_total(&self) -> &Array1<f64> {
        &self.ln_n_over_total
    }
}

impl Node for MultinomialNode {
    fn get_id(&self) -> NodeId {
        self.id
    }

    fn get_height(&self) -> f64 {
        self.height
    }

    fn get_count(&self) -> NodeCount {
        self.count
    }

    fn get_left_child(&self) -> &Option<Box<Self>> {
        &self.left_child
    }

    fn get_right_child(&self) -> &Option<Box<Self>> {
        &self.right_child
    }

    fn distance(&self, other: &Self) -> Result<f64> {
        let n1 = &self.n;
        let ln1 = &self.ln_n_over_total;

        let n2 = &other.n;
        let ln2 = &other.ln_n_over_total;

        let n_s_t = n1 + n2;
        let total_s_t = self.total + other.total;
        let ln_s_t = (&n_s_t / total_s_t).mapv(f64::ln); // ln(n_s_t / total_s_t)
        let distance = n1.dot(ln1) + n2.dot(ln2) - n_s_t.dot(&ln_s_t);

        Ok(distance)
    }

    fn fuse(&self, other: &Self, id: NodeId, distance: Option<f64>) -> Result<Self> {
        let n1 = &self.n;
        let n2 = &other.n;
        let n_s_t = n1 + n2;
        let total_s_t = self.total + other.total;

        let ln_s_t = (&n_s_t / total_s_t).mapv(f64::ln); // ln(n_s_t / total_s_t)

        let distance = match distance {
            Some(d) => d,
            None => self.distance(&other)?,
        };

        Ok(MultinomialNode::new(
            id,
            self.height + other.height + distance,
            self.count + other.count,
            Some(Box::new(self.clone())),
            Some(Box::new(other.clone())),
            n_s_t,
            total_s_t,
            ln_s_t,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Multinomial node {{ id: {}, height: {}, count: {}, left_child_id: {:?}, right_child_id: {:?}, n: {:?}, total: {}, ln_n_over_total: {:?} }}",
            self.id, self.height, self.count, self.get_left_child_id(), self.get_right_child_id(), self.n, self.total, self.ln_n_over_total
        )
    }
}
