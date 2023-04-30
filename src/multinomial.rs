use ndarray;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use ndarray_linalg::krylov::R;
use std::marker::PhantomData;

use std::cmp::Ordering;

use crate::Node;

// -------------------------- MultinomialNode --------------------------

#[derive(Debug, Clone)]
pub struct MultinomialNode {
    id: usize,
    height: f64,
    left_child: Option<Box<MultinomialNode>>,
    right_child: Option<Box<MultinomialNode>>,
    n: Array1<f64>,
    total: f64,
    ln_n_over_total: Array1<f64>,
}

// Multinomial Node Methods

impl MultinomialNode {
    pub fn new(
        id: usize,
        height: f64,
        left_child: Option<Box<MultinomialNode>>,
        right_child: Option<Box<MultinomialNode>>,
        n: Array1<f64>,
        total: f64,
        ln_n_over_total: Array1<f64>,
    ) -> Self {
        MultinomialNode {
            id,
            height,
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
        let sum_n = &self.n + &other.n;
        let distance = &self.n.dot(&self.ln_n_over_total.clone())
            + &other.n.dot(&other.ln_n_over_total.clone())
            - &sum_n.dot(
                &(sum_n
                    .mapv(|x| x.ln() - (self.total + other.total).ln())
                    .clone()),
            );
        Ok(distance)
    }

    fn combine(&self, other: &Self, id: usize, distance: Option<f64>) -> Result<Self, String> {
        let n_s_t = &self.n + &other.n;
        let total_s_t = self.total + other.total;
        let ln_n_over_total_s_t = n_s_t.mapv(|x| (x / total_s_t).ln());

        let distance = match distance {
            Some(d) => d,
            None => self.distance(&other)?,
        };

        Ok(MultinomialNode::new(
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
            "Multinomial node {{ id: {}, height: {}, left_child_id: {:?}, right_child_id: {:?}, n: {:?}, total: {}, ln_n_over_total: {:?} }}",
            self.id, self.height, self.get_left_child_id(), self.get_right_child_id(), self.n, self.total, self.ln_n_over_total
        )
    }
}

