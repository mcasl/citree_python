use crate::{Node, NodeCount, NodeId};
use anyhow::Result;

// -------------------------- PoissonNode --------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct PoissonNode {
    id: NodeId,
    height: f64,
    count: NodeCount,
    left_child: Option<Box<PoissonNode>>,
    right_child: Option<Box<PoissonNode>>,
    n: f64,
    total: f64,
    ln_n_over_total: f64,
}

// Poisson Node Methods
impl PoissonNode {
    pub fn new(
        id: NodeId,
        height: f64,
        count: NodeCount,
        left_child: Option<Box<PoissonNode>>,
        right_child: Option<Box<PoissonNode>>,
        n: f64,
        total: f64,
        ln_n_over_total: f64,
    ) -> Self {
        PoissonNode {
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

    pub fn get_n(&self) -> f64 {
        self.n
    }

    pub fn get_total(&self) -> f64 {
        self.total
    }

    pub fn get_ln_n_over_total(&self) -> f64 {
        self.ln_n_over_total
    }
}

impl Node for PoissonNode {
    fn get_id(&self) -> NodeId {
        self.id
    }

    fn get_height(&self) -> f64 {
        self.height
    }

    fn get_count(&self) -> NodeCount {
        self.count
    }

    fn get_left_child(&self) -> Option<&Self> {
        match &self.left_child {
            Some(child) => Some(child),
            None => None,
        }
    }

    fn get_right_child(&self) -> Option<&Self> {
        match &self.right_child {
            Some(child) => Some(child),
            None => None,
        }
    }

    fn distance(&self, other: &Self) -> Result<f64> {
        let n1 = self.n;
        let n2 = other.n;

        let total1 = self.total;
        let total2 = other.total;

        let ln1 = self.ln_n_over_total;
        let ln2 = other.ln_n_over_total;

        let n_s_t = n1 + n2;
        let total_s_t = total1 + total2;
        let ln_s_t = f64::ln(n_s_t / total_s_t);

        let distance = n1 * ln1 + n2 * ln2 - n_s_t * ln_s_t;
        Ok(distance)
    }

    fn merge(&self, other: &Self, id: NodeId, distance: Option<f64>) -> Result<Self> {
        let n_s_t = self.n + other.n;
        let total_s_t = self.total + other.total;
        let ln_n_over_total_s_t = f64::ln(n_s_t / total_s_t);

        let distance = match distance {
            Some(d) => d,
            None => self.distance(&other)?,
        };

        Ok(PoissonNode::new(
            id,
            self.height + other.height + distance,
            NodeCount(*self.count + *other.count),
            Some(Box::new(self.clone())),
            Some(Box::new(other.clone())),
            n_s_t,
            total_s_t,
            ln_n_over_total_s_t,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Poisson node {{ id: {}, height: {}, count: {}, left_child: {:?}, right_child: {:?}, n: {}, total: {}, ln_n_over_total: {} }}",
            self.id, self.height, self.count, self.get_left_child_id(), self.get_right_child_id(), self.n, self.total, self.ln_n_over_total
        )
    }
}
