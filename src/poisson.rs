
use crate::Node;

// -------------------------- PoissonNode --------------------------

#[derive(Debug, Clone)]
pub struct PoissonNode {
    id: usize,
    height: f64,
    left_child:  Option<Box<PoissonNode>>,
    right_child: Option<Box<PoissonNode>>,
    n: f64,
    total: f64,
    ln_n_over_total: f64,
}

// Poisson Node Methods
impl PoissonNode {
    pub fn new(
        id: usize,
        height: f64,
        left_child: Option<Box<PoissonNode>>,
        right_child: Option<Box<PoissonNode>>,
        n: f64,
        total: f64,
        ln_n_over_total: f64,
    ) -> Self {
        PoissonNode {
            id,
            height,
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
    fn get_id(&self) -> usize {
        self.id
    }

    fn get_height(&self) -> f64 {
        self.height
    }

    fn get_left_child(&self) -> &Option<Box<Self>> {
        &self.left_child
    }

    fn get_right_child(&self) -> &Option<Box<Self>> {
        &self.right_child
    }




    fn distance(&self, other: &Self) -> Result<f64, String> {
        let n1 = self.n;
        let n2 = other.n;
        let total1 = self.total;
        let total2 = other.total;
        let ln1 = self.ln_n_over_total;
        let ln2 = other.ln_n_over_total;

        let distance = n1 * ln1 + n2 * ln2 - (n1 + n2).ln() + (total1 + total2).ln();
        Ok(distance)
    }

    fn combine(&self, other: &Self, id: usize, distance: Option<f64>) -> Result<Self, String> {
        let n_s_t = self.n + other.n;
        let total_s_t = self.total + other.total;
        let ln_n_over_total_s_t = (n_s_t / total_s_t).ln();

        let distance = match distance {
            Some(d) => d,
            None => self.distance(&other)?,
        };

        Ok(PoissonNode::new(
            id,
            self.height + other.height + distance,
            Some(Box::new(self.clone())),
            Some(Box::new(other.clone())),
            n_s_t,
            total_s_t,
            ln_n_over_total_s_t,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Poisson node {{ id: {}, height: {}, left_child: {:?}, right_child: {:?}, n: {}, total: {}, ln_n_over_total: {} }}",
            self.id, self.height, self.get_left_child_id(), self.get_right_child_id(), self.n, self.total, self.ln_n_over_total
        )
    }
}

