use ndarray;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use ndarray_linalg::krylov::R;
use std::marker::PhantomData;

use std::cmp::Ordering;

use crate::Node;

// -------------------------- NormalNode --------------------------

#[derive(Debug, Clone)]
pub struct NormalNode {
    id: usize,
    height: f64,
    left_child:  Option<Box<NormalNode>>,
    right_child: Option<Box<NormalNode>>,
    x: Array1<f64>,
    v: Array2<f64>,
    inv_v: Array2<f64>,
}

// Normal Node Methods
impl NormalNode {
    pub fn new(
        id: usize,
        height: f64,
        left_child:   Option<Box<NormalNode>>,
        right_child:  Option<Box<NormalNode>>,
        x: Array1<f64>,
        v: Array2<f64>,
        inv_v: Array2<f64>,
    ) -> Self {
        NormalNode {
            id,
            height,
            left_child:  left_child,
            right_child: right_child,
            x,
            v,
            inv_v,
        }
    }

    pub fn get_x(&self) -> &Array1<f64> {
        &self.x
    }

    pub fn get_v(&self) -> &Array2<f64> {
        &self.v
    }

    pub fn get_inv_v(&self) -> &Array2<f64> {
        &self.inv_v
    }

}

impl Node for NormalNode {
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

    fn combine(&self, other: &Self, id: usize, distance: Option<f64>) -> Result<Self, String> {
        let inv_v_s_t = &self.inv_v + &other.inv_v;
        let v_s_t = inv_v_s_t.inv().map_err(|e| e.to_string())?;

        let distance = match distance {
            Some(d) => d,
            None => self.distance(&other)?,
        };

        Ok(NormalNode::new(
            id,
            self.height + other.height + distance,
            Some(Box::new(self.clone())),
            Some(Box::new(other.clone())),
            v_s_t.dot(&(self.inv_v.dot(&self.x) + other.inv_v.dot(&other.x))),
            v_s_t,
            inv_v_s_t,
            )
        )
    }

    fn distance(&self, other: &Self) -> Result<f64, String> {
        let diff = &self.x - &other.x;
        let combined_v = (&self.inv_v + &other.inv_v)
            .inv()
            .map_err(|e| e.to_string())?;
        let distance = diff.t().dot(&combined_v).dot(&diff);
        Ok(distance)
    }

    fn __repr__(&self) -> String {
        format!(
            "Normal node {{ id: {}, height: {}, left_child_id: {:?}, right_child_id: {:?}, x: {:?}, v: {:?}, inv_v: {:?} }}",
            self.id, self.height, self.get_left_child_id(), self.get_right_child_id(), self.x, self.v, self.inv_v
        )
    }
}

