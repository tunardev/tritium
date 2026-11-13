use std::cell::{Cell, Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

pub type Gradients = Vec<Option<Vec<f32>>>;
pub type BackwardFunction = Box<dyn Fn(&[f32]) -> Gradients>;

thread_local! {
    static IS_GRADIENT_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_gradient_enabled() -> bool {
    IS_GRADIENT_ENABLED.with(Cell::get)
}

pub fn set_gradient_enabled(is_enabled: bool) -> bool {
    IS_GRADIENT_ENABLED.with(|flag| flag.replace(is_enabled))
}

struct Node {
    data: RefCell<Vec<f32>>,
    shape: Vec<usize>,
    gradient: RefCell<Option<Vec<f32>>>,
    requires_gradient: bool,
    parents: Vec<Tensor>,
    backward: Option<BackwardFunction>,
}

#[derive(Clone)]
pub struct Tensor(Rc<Node>);

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("shape", &self.0.shape)
            .field("requires_gradient", &self.0.requires_gradient)
            .finish()
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Tensor {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "data does not match {shape:?}");
        Tensor(Rc::new(Node {
            data: RefCell::new(data),
            shape: shape.to_vec(),
            gradient: RefCell::new(None),
            requires_gradient: false,
            parents: Vec::new(),
            backward: None,
        }))
    }

    pub fn parameter(data: Vec<f32>, shape: &[usize]) -> Tensor {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "data does not match {shape:?}");
        Tensor(Rc::new(Node {
            data: RefCell::new(data),
            shape: shape.to_vec(),
            gradient: RefCell::new(None),
            requires_gradient: true,
            parents: Vec::new(),
            backward: None,
        }))
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::new(vec![0.0; shape.iter().product()], shape)
    }

    pub fn from_operation(
        data: Vec<f32>,
        shape: &[usize],
        parents: &[&Tensor],
        backward: BackwardFunction,
    ) -> Tensor {
        let requires_gradient =
            is_gradient_enabled() && parents.iter().any(|parent| parent.requires_gradient());
        if !requires_gradient {
            return Tensor::new(data, shape);
        }
        Tensor(Rc::new(Node {
            data: RefCell::new(data),
            shape: shape.to_vec(),
            gradient: RefCell::new(None),
            requires_gradient: true,
            parents: parents.iter().map(|&parent| parent.clone()).collect(),
            backward: Some(backward),
        }))
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn len(&self) -> usize {
        self.0.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data(&self) -> Ref<'_, Vec<f32>> {
        self.0.data.borrow()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data().clone()
    }

    pub fn requires_gradient(&self) -> bool {
        self.0.requires_gradient
    }

    pub fn gradient(&self) -> Option<Vec<f32>> {
        self.0.gradient.borrow().clone()
    }

    pub fn zero_gradient(&self) {
        *self.0.gradient.borrow_mut() = None;
    }

    pub fn detach(&self) -> Tensor {
        Tensor::new(self.to_vec(), self.shape())
    }

    pub fn update(&self, function: impl FnOnce(&mut [f32], &[f32])) {
        let gradient = self.0.gradient.borrow();
        let empty = Vec::new();
        function(&mut self.0.data.borrow_mut(), gradient.as_ref().unwrap_or(&empty));
    }

    pub fn assign(&self, data: &[f32]) {
        assert_eq!(data.len(), self.len(), "assigned data does not match {:?}", self.shape());
        self.0.data.borrow_mut().copy_from_slice(data);
    }

    fn identity(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn backward(&self) {
        assert_eq!(
            self.len(),
            1,
            "backward needs a scalar; this tensor has shape {:?}",
            self.shape()
        );
        if !self.requires_gradient() {
            return;
        }
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![(self.clone(), false)];
        while let Some((tensor, is_expanded)) = stack.pop() {
            if is_expanded {
                order.push(tensor);
                continue;
            }
            if !visited.insert(tensor.identity()) {
                continue;
            }
            stack.push((tensor.clone(), true));
            for parent in &tensor.0.parents {
                if parent.requires_gradient() && !visited.contains(&parent.identity()) {
                    stack.push((parent.clone(), false));
                }
            }
        }
        let mut gradients: HashMap<usize, Vec<f32>> = HashMap::new();
        gradients.insert(self.identity(), vec![1.0]);
        for tensor in order.iter().rev() {
            let Some(gradient) = gradients.remove(&tensor.identity()) else { continue };
            match &tensor.0.backward {
                Some(backward) => {
                    for (parent, parent_gradient) in
                        tensor.0.parents.iter().zip(backward(&gradient))
                    {
                        let Some(parent_gradient) = parent_gradient else { continue };
                        if !parent.requires_gradient() {
                            continue;
                        }
                        match gradients.get_mut(&parent.identity()) {
                            Some(total) => add_into(total, &parent_gradient),
                            None => {
                                gradients.insert(parent.identity(), parent_gradient);
                            }
                        }
                    }
                }
                None => {
                    let mut stored = tensor.0.gradient.borrow_mut();
                    match stored.as_mut() {
                        Some(total) => add_into(total, &gradient),
                        None => *stored = Some(gradient),
                    }
                }
            }
        }
    }
}

pub fn add_into(total: &mut [f32], values: &[f32]) {
    for (total, value) in total.iter_mut().zip(values) {
        *total += value;
    }
}
