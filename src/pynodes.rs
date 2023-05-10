use crate::{
    //cluster_multinomial_nodes,
    cluster_bins,
    multinomial::MultinomialNode as MultinomialNodeRust,
    normal::NormalNode as NormalNodeRust,
    poisson::PoissonNode as PoissonNodeRust,
    //cluster_poisson_nodes,
    Node,
    NodeCount,
    NodeId,
};

use pyo3;
use pyo3::types::PyModule;
use std::ops::Deref;

use pyo3::Py;
use pyo3::PyResult;
//use pyo3::prelude::*;
use ndarray;
use ndarray::prelude::*;
use ndarray::Array2;
use numpy::convert::IntoPyArray;

use ndarray::Array1;

use pyo3::conversion::FromPyObject;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use numpy;
use numpy::pyo3::Python;
use numpy::{PyArray1, PyArray2};

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

impl IntoPy<PyObject> for Array1Wrapper {
    fn into_py(self, py: Python<'_>) -> PyObject {
        PyArray1::from_vec(py, self.0.to_vec()).into()
    }
}

impl From<Array1Wrapper> for Array1<f64> {
    fn from(wrapper: Array1Wrapper) -> Self {
        wrapper.0
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

impl IntoPy<PyObject> for Array2Wrapper {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        Python::with_gil(|py| self.0.into_pyarray(py).to_object(py))
    }
}

impl From<Array2Wrapper> for Array2<f64> {
    fn from(wrapper: Array2Wrapper) -> Self {
        wrapper.0
    }
}

// -------------------------- Box<PyNormalNode> --------------------------

// A trait may be implemented for Box<T> in the same crate as T,
// which the orphan rules prevent for other generic types.

impl IntoPy<PyObject> for Box<PyNormalNode> {
    fn into_py(self, py: Python) -> PyObject {
        let py_node = self.deref().clone();
        py_node.into_py(py)
    }
}

impl FromPyObject<'_> for Box<PyNormalNode> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        Python::with_gil(|_py| {
            let py_node = ob.extract::<PyNormalNode>()?;
            Ok(Box::new(py_node))
        })
    }
}

// -------------------------- Box<PyPoissonNode> --------------------------

impl IntoPy<PyObject> for Box<PyPoissonNode> {
    fn into_py(self, py: Python) -> PyObject {
        let py_node = self.deref().clone();
        py_node.into_py(py)
    }
}

impl FromPyObject<'_> for Box<PyPoissonNode> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        Python::with_gil(|_py| {
            let py_node = ob.extract::<PyPoissonNode>()?;
            Ok(Box::new(py_node))
        })
    }
}

// -------------------------- Box<PyMultinomialNode> --------------------------

impl IntoPy<PyObject> for Box<PyMultinomialNode> {
    fn into_py(self, py: Python) -> PyObject {
        let py_node = self.deref().clone();
        py_node.into_py(py)
    }
}

impl FromPyObject<'_> for Box<PyMultinomialNode> {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        Python::with_gil(|_py| {
            let py_node = ob.extract::<PyMultinomialNode>()?;
            Ok(Box::new(py_node))
        })
    }
}

// -------------------------- PyNormalNode --------------------------

#[pyclass(name = "NormalNode")]
#[derive(Debug, Clone)]
pub struct PyNormalNode {
    #[pyo3(get)]
    id: usize,

    #[pyo3(get)]
    height: f64,

    #[pyo3(get)]
    count: usize,

    #[pyo3(get)]
    left_child: Option<Box<PyNormalNode>>,

    #[pyo3(get)]
    right_child: Option<Box<PyNormalNode>>,

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
    #[pyo3(
        signature = (id, height, count, x, v, inv_v, left_child=None, right_child=None)
    )]
    pub fn new(
        id: usize,
        height: f64,
        count: usize,
        x: Array1Wrapper,
        v: Array2Wrapper,
        inv_v: Array2Wrapper,
        left_child: Option<Box<PyNormalNode>>,
        right_child: Option<Box<PyNormalNode>>,
    ) -> Self {
        PyNormalNode {
            id,
            height,
            count,
            left_child,
            right_child,
            x,
            v,
            inv_v,
        }
    }

    pub fn __repr__(&self) -> String {
        let left_id = match &self.left_child {
            Some(boxed_node) => boxed_node.id.to_string(),
            None => "None".to_string(),
        };

        let right_id = match &self.right_child {
            Some(boxed_node) => boxed_node.id.to_string(),
            None => "None".to_string(),
        };

        format!(
            "Normal Node {{ id: {}, height: {}, count: {}, left_child_id: {:?}, right_child_id: {:?} }}",
            self.id, self.height, self.count, left_id, right_id
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl From<NormalNodeRust> for PyNormalNode {
    fn from(node: NormalNodeRust) -> Self {
        let left_child = match node.get_left_child() {
            Some(boxed_python_node) => Some(Box::new(PyNormalNode::from(
                boxed_python_node.deref().clone(),
            ))),
            None => None,
        };

        let right_child = match node.get_right_child() {
            Some(boxed_python_node) => Some(Box::new(PyNormalNode::from(
                boxed_python_node.deref().clone(),
            ))),
            None => None,
        };

        PyNormalNode::new(
            node.get_id().0,
            node.get_height(),
            node.get_count().0,
            Array1Wrapper(node.get_x().clone()),
            Array2Wrapper(node.get_v().clone()),
            Array2Wrapper(node.get_inv_v().clone()),
            left_child,
            right_child,
        )
    }
}

impl From<PyNormalNode> for NormalNodeRust {
    fn from(py_node: PyNormalNode) -> Self {
        let rust_left_child = match py_node.left_child {
            Some(boxed_python_node) => Some(Box::new(NormalNodeRust::from(*boxed_python_node))),
            None => None,
        };

        let rust_right_child = match py_node.right_child {
            Some(boxed_python_node) => Some(Box::new(NormalNodeRust::from(*boxed_python_node))),
            None => None,
        };

        NormalNodeRust::new(
            NodeId(py_node.id),
            py_node.height,
            NodeCount(py_node.count),
            rust_left_child,
            rust_right_child,
            ArrayBase::from(py_node.x),
            ArrayBase::from(py_node.v),
            ArrayBase::from(py_node.inv_v),
        )
    }
}

// -------------------------- PoissonNodePy

#[pyclass(name = "PyPoissonNode")]
#[derive(Debug, Clone)]
pub struct PyPoissonNode {
    #[pyo3(get)]
    pub id: usize,

    #[pyo3(get)]
    pub height: f64,

    #[pyo3(get)]
    pub count: usize,

    #[pyo3(get)]
    pub left_child: Option<Box<PyPoissonNode>>,

    #[pyo3(get)]
    pub right_child: Option<Box<PyPoissonNode>>,

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
    #[pyo3(signature = (id, height, count, n, total, ln_n_over_total, left_child=None, right_child=None))]
    pub fn new(
        id: usize,
        height: f64,
        count: usize,
        n: f64,
        total: f64,
        ln_n_over_total: f64,
        left_child: Option<Box<PyPoissonNode>>,
        right_child: Option<Box<PyPoissonNode>>,
    ) -> Self {
        PyPoissonNode {
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

    pub fn __repr__(&self) -> String {
        format!(
            "Poisson Node {{ id: {}, height: {}, count: {}, left_child: {:?}, right_child: {:?}, n: {}, total: {}, ln_n_over_total: {} }}",
            self.id, self.height, self.count, self.left_child, self.right_child, self.n, self.total, self.ln_n_over_total
        )
    }
}

impl From<PoissonNodeRust> for PyPoissonNode {
    fn from(node: PoissonNodeRust) -> Self {
        let left_child = match node.get_left_child() {
            Some(boxed_python_node) => Some(Box::new(PyPoissonNode::from(
                boxed_python_node.deref().clone(),
            ))),
            None => None,
        };

        let right_child = match node.get_right_child() {
            Some(boxed_python_node) => Some(Box::new(PyPoissonNode::from(
                boxed_python_node.deref().clone(),
            ))),
            None => None,
        };

        PyPoissonNode::new(
            node.get_id().0,
            node.get_height(),
            node.get_count().0,
            node.get_n(),
            node.get_total(),
            node.get_ln_n_over_total(),
            left_child,
            right_child,
        )
    }
}

impl From<PyPoissonNode> for PoissonNodeRust {
    fn from(py_node: PyPoissonNode) -> Self {
        let rust_left_child = match py_node.left_child {
            Some(boxed_python_node) => Some(Box::new(PoissonNodeRust::from(*boxed_python_node))),
            None => None,
        };

        let rust_right_child = match py_node.right_child {
            Some(boxed_python_node) => Some(Box::new(PoissonNodeRust::from(*boxed_python_node))),
            None => None,
        };

        PoissonNodeRust::new(
            NodeId(py_node.id),
            py_node.height,
            NodeCount(py_node.count),
            rust_left_child,
            rust_right_child,
            py_node.n,
            py_node.total,
            py_node.ln_n_over_total,
        )
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
    count: usize,

    #[pyo3(get)]
    left_child: Option<Box<PyMultinomialNode>>,

    #[pyo3(get)]
    right_child: Option<Box<PyMultinomialNode>>,

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
    #[pyo3(signature = (id, height, count, n, total, ln_n_over_total, left_child=None, right_child=None))]
    pub fn new(
        id: usize,
        height: f64,
        count: usize,
        n: Array1Wrapper,
        total: f64,
        ln_n_over_total: Array1Wrapper,
        left_child: Option<Box<PyMultinomialNode>>,
        right_child: Option<Box<PyMultinomialNode>>,
    ) -> PyMultinomialNode {
        PyMultinomialNode {
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

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Multinomial Node: {{ id: {}, height: {}, count: {}, left child: {:?}, right child: {:?}, n: {:?}, total: {} }}",
            self.id,
            self.height,
            self.count,
            self.left_child.as_ref().map_or(None, |node| Some(node.id)),
            self.right_child.as_ref().map_or(None, |node| Some(node.id)),
            self.n,
            self.total
        ))
    }
}

impl From<MultinomialNodeRust> for PyMultinomialNode {
    fn from(node: MultinomialNodeRust) -> Self {
        let left_child = match node.get_left_child() {
            Some(boxed_python_node) => Some(Box::new(PyMultinomialNode::from(
                boxed_python_node.deref().clone(),
            ))),
            None => None,
        };

        let right_child = match node.get_right_child() {
            Some(boxed_python_node) => Some(Box::new(PyMultinomialNode::from(
                boxed_python_node.deref().clone(),
            ))),
            None => None,
        };

        PyMultinomialNode::new(
            node.get_id().0,
            node.get_height(),
            node.get_count().0,
            Array1Wrapper(node.get_n().clone()),
            node.get_total(),
            Array1Wrapper(node.get_ln_n_over_total().clone()),
            left_child,
            right_child,
        )
    }
}

impl From<PyMultinomialNode> for MultinomialNodeRust {
    fn from(py_node: PyMultinomialNode) -> Self {
        let rust_left_child = match py_node.left_child {
            Some(boxed_python_node) => {
                Some(Box::new(MultinomialNodeRust::from(*boxed_python_node)))
            }
            None => None,
        };

        let rust_right_child = match py_node.right_child {
            Some(boxed_python_node) => {
                Some(Box::new(MultinomialNodeRust::from(*boxed_python_node)))
            }
            None => None,
        };

        MultinomialNodeRust::new(
            NodeId(py_node.id),
            py_node.height,
            NodeCount(py_node.count),
            rust_left_child,
            rust_right_child,
            ArrayBase::from(py_node.n),
            py_node.total,
            ArrayBase::from(py_node.ln_n_over_total),
        )
    }
}

// -------------------------- Python functions ----------------------------

#[pyfunction]
#[pyo3(name = "cluster_normal_bins")]
pub fn cluster_normal_bins_py(
    py: Python,
    bins: Vec<PyNormalNode>,
) -> PyResult<(Array2Wrapper, PyNormalNode)> {
    let rustbins = bins.into_iter().map(NormalNodeRust::from).collect();
    let (linkage_matrix, dendrogram) = cluster_bins::<NormalNodeRust>(&rustbins)
        .map_err(|e| PyRuntimeError::new_err(format!("Error clustering normal nodes: {:?}", e)))?;
    Ok((
        Array2Wrapper(linkage_matrix),
        PyNormalNode::from(dendrogram),
    ))
}

#[pyfunction]
#[pyo3(name = "cluster_poisson_bins")]
pub fn cluster_poisson_bins_py(
    py: Python,
    bins: Vec<PyPoissonNode>,
) -> PyResult<(Array2Wrapper, PyPoissonNode)> {
    let rustbins = bins.into_iter().map(PoissonNodeRust::from).collect();
    let (linkage, dendrogram) = cluster_bins::<PoissonNodeRust>(&rustbins)
        .map_err(|e| PyRuntimeError::new_err(format!("Error clustering normal nodes: {:?}", e)))?;
    Ok((Array2Wrapper(linkage), PyPoissonNode::from(dendrogram)))
}

#[pyfunction]
#[pyo3(name = "cluster_multinomial_bins")]
pub fn cluster_multinomial_bins_py(
    py: Python,
    bins: Vec<PyMultinomialNode>,
) -> PyResult<(Array2Wrapper, PyMultinomialNode)> {
    let rustbins = bins.into_iter().map(MultinomialNodeRust::from).collect();
    let (linkage, dendrogram) = cluster_bins::<MultinomialNodeRust>(&rustbins)
        .map_err(|e| PyRuntimeError::new_err(format!("Error clustering normal nodes: {:?}", e)))?;
    Ok((Array2Wrapper(linkage), PyMultinomialNode::from(dendrogram)))
}

// -------------------------- PYTHON MODULE --------------------------
#[pymodule]
fn citree_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PyNormalNode>()?;
    m.add_class::<PyPoissonNode>()?;
    m.add_class::<PyMultinomialNode>()?;

    // Functions
    m.add_function(wrap_pyfunction!(cluster_normal_bins_py, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_poisson_bins_py, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_multinomial_bins_py, m)?)?;
    Ok(())
}
