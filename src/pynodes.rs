use crate::{
    //cluster_multinomial_nodes,
    //cluster_normal_nodes,
    //cluster_poisson_nodes,
    Node,
    normal::NormalNode  as NormalNodeRust,
    poisson::PoissonNode as PoissonNodeRust,
    multinomial::MultinomialNode as MultinomialNodeRust,
};

use std::ops::Deref;
use pyo3;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use pyo3::Py;
use pyo3::PyResult;
//use pyo3::prelude::*;
use numpy::convert::IntoPyArray;
use ndarray;
use ndarray::prelude::*;
use ndarray::Array2;

use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use numpy;
use numpy::pyo3::Python;
use numpy::{PyArray1, PyArray2};
//use std::collections::HashMap;


use std::sync::{Arc, Weak};
// -------------------------- Classes for Python ------------------------------------------
// -------------------------- Array1Wrapper --------------------------


#[derive(Debug, Clone)]
pub struct Array1Wrapper(pub Array1<f64>);


impl<'source> FromPyObject<'source> for Array1Wrapper {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Python::with_gil(|py| {
            let py_array = ob.extract::<Py<PyArray1<f64>>>()?;
            let array = py_array.as_ref(py).readonly().as_array().to_owned();
            Ok(Array1Wrapper(array))
        })
    }
}


impl From<Array1Wrapper> for Array1<f64> {
    fn from(wrapper: Array1Wrapper) -> Self {
        wrapper.0
    }
}


impl IntoPy<PyObject> for Array1Wrapper {
    fn into_py(self, py: Python<'_>) -> PyObject {
        PyArray1::from_vec(py, self.0.to_vec()).into()
    }
}


// -------------------------- Array2Wrapper


#[derive(Debug, Clone)]
pub struct Array2Wrapper(pub Array2<f64>);

impl<'source> FromPyObject<'source> for Array2Wrapper {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Python::with_gil(|py| {
            let py_array = ob.extract::<Py<PyArray2<f64>>>()?;
            let array = py_array.as_ref(py).readonly().as_array().to_owned();
            Ok(Array2Wrapper(array))
        })
    }
}

impl From<Array2Wrapper> for Array2<f64> {
    fn from(wrapper: Array2Wrapper) -> Self {
        wrapper.0
    }
}


impl IntoPy<PyObject> for Array2Wrapper {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        Python::with_gil(|py| {
             self.0.into_pyarray(py).to_object(py)
        })
}
}


// -------------------------- NormalNodePy
#[pyclass(name = "PyNormalNode")]
#[derive(Debug, Clone)]
pub struct PyNormalNode {
    #[pyo3(get)]
    id: usize,

    #[pyo3(get)]
    height: f64,

    #[pyo3(get)]
    left_child: OptionalBoxedPyNormalNode,

    #[pyo3(get)]
    right_child: OptionalBoxedPyNormalNode,

    #[pyo3(get)]
    x: Array1Wrapper,

    #[pyo3(get)]
    v: Array2Wrapper,

    #[pyo3(get)]
    inv_v: Array2Wrapper,
}

// -------------------------- NormalNodePy Methods
#[pymethods]
impl PyNormalNode {
    #[new]
    pub fn new(
        id:         usize,
        height:     f64,
        #[pyo3(from_py_with = "OptionalBoxedPyNormalNode::extract")]
        left_child: OptionalBoxedPyNormalNode,
        #[pyo3(from_py_with = "OptionalBoxedPyNormalNode::extract")]
        right_child:OptionalBoxedPyNormalNode,
        x:          Array1Wrapper,
        v:          Array2Wrapper,
        inv_v:      Array2Wrapper,
    ) -> Self {
        PyNormalNode {
            id,
            height,
            left_child,
            right_child,
            x,
            v,
            inv_v,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Normal Node {{ id: {}, height: {}, left_child_id: {:?}, right_child_id: {:?}, x: {:#?}, v: {:#?}, inv_v: {:#?} }}",
            self.id, self.height, self.left_child.get_id(), self.right_child.get_id(), self.x.0.dim(), self.v.0.dim(), self.inv_v.0.dim()
        )
    }
}



/*
impl From<PyNormalNode> for NormalNodeRust {
    fn from(py_node: PyNormalNode) -> Self {
        let rust_left_child  = match py_node.left_child {

            Some(boxed_python_node) => Some(Box::new(NormalNodeRust::from(*boxed_python_node.inner))),
            None => None,
        };
        
        let rust_right_child  = match py_node.right_child {

            Some(boxed_python_node) => Some(Box::new(NormalNodeRust::from(*boxed_python_node.inner))),
            None => None,
        };
        
        
        NormalNodeRust::new(
            py_node.id,
            py_node.height,
            rust_left_child,
            rust_right_child,
            ArrayBase::from(py_node.x),
            ArrayBase::from(py_node.v),
            ArrayBase::from(py_node.inv_v),
        )
    }
}

*/

// -------------------------- BoxPyNormalNode --------------------------

#[pyclass]
#[derive(Debug, Clone)]
pub struct OptionalBoxedPyNormalNode {
    #[pyo3(get)]
    inner: Option<Box<PyNormalNode>>,
}

impl OptionalBoxedPyNormalNode {
    // Constructor
    pub fn new(inner: PyNormalNode) -> Self {
        OptionalBoxedPyNormalNode {
             inner: Some(Box::new(inner)),
        }
    }

    // Converter:
    // I implemented this method here instead of in the FromPyObject trait
    // to avoid trait conflicts with the implementation 
    fn extract(ob: &PyAny) -> PyResult<Self> {
        Python::with_gil(|_py| {
           // println!("DEBUG: Entrando en extract 220");
          // case 1: None
            if ob.is_none() {
                Ok(OptionalBoxedPyNormalNode { inner: None })
            } else {
                // case 2: Some
                let py_node = ob.extract::<PyNormalNode>()?;
                Ok(OptionalBoxedPyNormalNode { inner: Some(Box::new(py_node)) })
            }
        })
    }

    // Getter 
    pub fn get_id(&self) -> Option<usize> {
        self.inner.as_ref().map(|node| node.id)
    }

}   



impl IntoPy<PyObject> for Box<PyNormalNode> {
    fn into_py(self, py: Python) -> PyObject {
        print!("Entrando en into_py 70");
        let py_node = self.deref().clone();

        py_node.into_py(py)
    }
}

impl From<PyNormalNode> for OptionalBoxedPyNormalNode{
    fn from(item: PyNormalNode) -> Self {
        OptionalBoxedPyNormalNode {
            inner: Some(Box::new(item)),
        }
    }
}

impl From<Option<Box<PyNormalNode>>> for OptionalBoxedPyNormalNode
{
    fn from(item: Option<Box<PyNormalNode>>) -> Self {
        let inner = match item {
            None => None,
            other_value => other_value,
        };
        OptionalBoxedPyNormalNode { inner }
    }
}
 

/* 
impl From<Box<NormalNodeRust>> for BoxPyNormalNode {
    fn from(node: Box<NormalNodeRust>) -> Self {
        println!("Entrando en from Box<NormalNode> 90");
        let py_node = PyNormalNode {
            id: node.get_id(),
            height: node.get_height(),
            left_child: node.get_left_child().as_ref().map(|boxed| BoxPyNormalNode::from(*boxed.clone())),
            right_child: node.get_right_child().as_ref().map(|boxed| BoxPyNormalNode::from(*boxed.clone())),
            x: Array1Wrapper(node.get_x().clone()),
            v: Array2Wrapper(node.get_v().clone()),
            inv_v: Array2Wrapper(node.get_inv_v().clone()),
        };

        BoxPyNormalNode {
            inner: Box::new(py_node),
        }
    }
}

impl From<NormalNodeRust> for BoxPyNormalNode {
    fn from(node: NormalNodeRust) -> Self {
        println!("Entrando en from NormalNodeRust 90");
        let py_node = PyNormalNode {
            id: node.get_id(),
            height: node.get_height(),
            left_child: node.get_left_child().as_ref().map(|boxed| BoxPyNormalNode::from(*boxed.clone())),
            right_child: node.get_right_child().as_ref().map(|boxed| BoxPyNormalNode::from(*boxed.clone())),
            x: Array1Wrapper(node.get_x().clone()),
            v: Array2Wrapper(node.get_v().clone()),
            inv_v: Array2Wrapper(node.get_inv_v().clone()),
        };

        BoxPyNormalNode {
            inner: Box::new(py_node),
        }
    }
}

*/




// -------------------------- PoissonNodePy
 
#[pyclass(name = "PyPoissonNode")]
#[derive(Debug, Clone)]
pub struct PyPoissonNode {
    #[pyo3(get)]
    pub id: usize,

    #[pyo3(get)]
    pub height: f64,

    #[pyo3(get)]
    pub left_child: Option<BoxPyPoissonNode>,

    #[pyo3(get)]
    pub right_child: Option<BoxPyPoissonNode>,

    #[pyo3(get)]
    pub n: f64,

    #[pyo3(get)]
    pub total: f64,

    #[pyo3(get)]
    pub ln_n_over_total: f64,
}

// ------- PoissonNodePy methods
#[pymethods]
impl PyPoissonNode {
    #[new]
    #[pyo3(signature = (id, height, n, total, ln_n_over_total, left_child=None, right_child=None))]
    pub fn new(
        id: usize,
        height: f64,
        n: f64,
        total: f64,
        ln_n_over_total: f64,
        left_child: Option<BoxPyPoissonNode>,
        right_child: Option<BoxPyPoissonNode>,
    ) -> Self {
        PyPoissonNode {
            id,
            height,
            left_child,
            right_child,
            n,
            total,
            ln_n_over_total,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Poisson Node {{ id: {}, height: {}, left_child: {:?}, right_child: {:?}, n: {}, total: {}, ln_n_over_total: {} }}",
            self.id, self.height, self.left_child, self.right_child, self.n, self.total, self.ln_n_over_total
        )
    }
}

// -------------------------- BoxPyPoissonNode ----------------------------



impl From<PyPoissonNode> for PoissonNodeRust {
    fn from(py_node: PyPoissonNode) -> Self {
        let rust_left_child  = match py_node.left_child {

            Some(boxed_python_node) => Some(Box::new(PoissonNodeRust::from(*boxed_python_node.inner))),
            None => None,
        };
        
        let rust_right_child  = match py_node.right_child {

            Some(boxed_python_node) => Some(Box::new(PoissonNodeRust::from(*boxed_python_node.inner))),
            None => None,
        };
        
        PoissonNodeRust::new(
            py_node.id,
            py_node.height,
            rust_left_child,
            rust_right_child,
            py_node.n,
            py_node.total,
            py_node.ln_n_over_total,
        )
    }
}


#[pyclass]
#[derive(Debug, Clone)]
pub struct BoxPyPoissonNode {
    inner: Box<PyPoissonNode>,
}


impl IntoPy<PyObject> for Box<PyPoissonNode> {
    fn into_py(self, py: Python) -> PyObject {
        let py_node = self.deref().clone();

        py_node.into_py(py)
    }
}



// -------------------------- MultinomialNodePy

#[pyclass(name = "MultinomialNode")]
#[derive(Debug, Clone)]
pub struct PyMultinomialNode {
    #[pyo3(get)]
    id: usize,

    #[pyo3(get)]
    height: f64,

    #[pyo3(get)]
    left_child: Option<BoxPyMultinomialNode>,

    #[pyo3(get)]
    right_child: Option<BoxPyMultinomialNode>,

    #[pyo3(get)]
    n: Array1Wrapper,

    #[pyo3(get)]
    total: f64,

    #[pyo3(get)]
    ln_n_over_total: Array1Wrapper,
}

// -------------------------- MultinomialNodePy methods
#[pymethods]
impl PyMultinomialNode {
    #[new]
    #[pyo3(signature = (id, height, n, total, ln_n_over_total, left_child=None, right_child=None))]
    pub fn new(
        id: usize,
        height: f64,
        n: Array1Wrapper,
        total: f64,
        ln_n_over_total: Array1Wrapper,
        left_child: Option<BoxPyMultinomialNode>,
        right_child: Option<BoxPyMultinomialNode>,
    ) -> PyMultinomialNode {
        PyMultinomialNode {
            id,
            height,
            left_child,
            right_child,
            n,
            total,
            ln_n_over_total,
        }
    }


    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Multinomial Node: {{ id: {}, height: {}, left child: {:?}, right child: {:?}, n: {:?}, total: {} }}",
            self.id,
            self.height,
            self.left_child.as_ref().map_or(None, |node| Some(node.inner.id)),
            self.right_child.as_ref().map_or(None, |node| Some(node.inner.id)),
            self.n,
            self.total
        ))
    }
}

// -------------------------- BoxPyMultinomialNode ----------------------------

#[pyclass]
#[derive(Debug, Clone)]
pub struct BoxPyMultinomialNode {
    inner: Box<PyMultinomialNode>,
}


impl IntoPy<PyObject> for Box<PyMultinomialNode> {
    fn into_py(self, py: Python) -> PyObject {
        let py_node = self.deref().clone();

        py_node.into_py(py)
    }
}


impl From<PyMultinomialNode> for MultinomialNodeRust {
    fn from(py_node: PyMultinomialNode) -> Self {
        let rust_left_child  = match py_node.left_child {

            Some(boxed_python_node) => Some(Box::new(MultinomialNodeRust::from(*boxed_python_node.inner))),
            None => None,
        };
        
        let rust_right_child  = match py_node.right_child {

            Some(boxed_python_node) => Some(Box::new(MultinomialNodeRust::from(*boxed_python_node.inner))),
            None => None,
        };
        
        MultinomialNodeRust::new(
            py_node.id,
            py_node.height,
            rust_left_child,
            rust_right_child,
            ArrayBase::from(py_node.n),
            py_node.total,
            ArrayBase::from(py_node.ln_n_over_total),
        )
    }
}


// -------------------------- Helper functions ----------------------------

fn make_normal_vec_from(py: Python, bins: Vec<PyNormalNode>) -> PyResult<Vec<NormalNodeRust>> {
    let mut result: Vec<NormalNodeRust> = Vec::new();
    for python_node in bins {
     //   let rust_node = NormalNodeRust::from(python_node);
            //x: item.x.extract::<Array1Wrapper>(py)?.into(),
            //v: item.v.extract::<Array2Wrapper>(py)?.into(),
            //inv_v: item.inv_v.extract::<Array2Wrapper>(py)?.into(),
       // result.push(rust_node);
    }
    Ok(result)
}









 
#[pyfunction]
#[pyo3(name = "cluster_normal_nodes")]
pub fn cluster_normal_nodes_py(py: Python, bins: Vec<PyNormalNode>) -> PyResult<Vec<PyNormalNode>> {
    

    println!("Hello from Rust!");
    println!("Python Bins: {:?}", bins);

    let _rustbins = make_normal_vec_from(py, bins)?;

    //let dendrogram = cluster_normal_nodes(bins);
    //println!("Dendrogram: {:?}", dendrogram);

    //println!("Result: {:?}", bins);
    todo!()
}






// -------------------------- PYTHON MODULE --------------------------
#[pymodule]
fn citree_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PyNormalNode>()?;
    m.add_class::<PyPoissonNode>()?;
    m.add_class::<PyMultinomialNode>()?;

    // Functions
    //m.add_function(wrap_pyfunction!(cluster_normal_nodes_py, m)?)?;
    Ok(())
}
