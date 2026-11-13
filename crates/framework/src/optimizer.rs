use crate::tensor::Tensor;

pub struct AdamW {
    parameters: Vec<Tensor>,
    first_moments: Vec<Vec<f32>>,
    second_moments: Vec<Vec<f32>>,
    pub learning_rate: f32,
    pub betas: (f32, f32),
    pub epsilon: f32,
    pub weight_decay: f32,
    step: i32,
}

impl AdamW {
    pub fn new(parameters: Vec<Tensor>, learning_rate: f32, weight_decay: f32) -> AdamW {
        let first_moments = parameters.iter().map(|parameter| vec![0.0; parameter.len()]).collect();
        let second_moments =
            parameters.iter().map(|parameter| vec![0.0; parameter.len()]).collect();
        AdamW {
            parameters,
            first_moments,
            second_moments,
            learning_rate,
            betas: (0.9, 0.95),
            epsilon: 1e-8,
            weight_decay,
            step: 0,
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
        let (first_beta, second_beta) = self.betas;
        let first_correction = 1.0 - first_beta.powi(self.step);
        let second_correction = 1.0 - second_beta.powi(self.step);
        for ((parameter, first), second) in
            self.parameters.iter().zip(&mut self.first_moments).zip(&mut self.second_moments)
        {
            if parameter.gradient().is_none() {
                continue;
            }
            let decay = if parameter.shape().len() >= 2 { self.weight_decay } else { 0.0 };
            let (learning_rate, epsilon) = (self.learning_rate, self.epsilon);
            parameter.update(|values, gradients| {
                for (index, value) in values.iter_mut().enumerate() {
                    let gradient = gradients[index];
                    first[index] = first_beta * first[index] + (1.0 - first_beta) * gradient;
                    second[index] =
                        second_beta * second[index] + (1.0 - second_beta) * gradient * gradient;
                    let corrected = first[index] / first_correction;
                    let variance = second[index] / second_correction;
                    *value -=
                        learning_rate * (corrected / (variance.sqrt() + epsilon) + decay * *value);
                }
            });
        }
    }

    pub fn zero_gradients(&self) {
        self.parameters.iter().for_each(Tensor::zero_gradient);
    }
}
