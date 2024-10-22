#![allow(unused)]
use ndarray::prelude::*;
use polars::prelude::*;
use rand::distributions::{weighted, Uniform};
use rand::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use num_integer::Roots;

#[derive(Debug, Clone)]
pub struct LinearCache {
    pub a: Array2<f32>, //激活矩阵
    pub w: Array2<f32>, //权重矩阵
    pub b: Array2<f32>, //偏置矩阵
}

#[derive(Debug, Clone)]
pub struct ActivationCache {
    pub z: Array2<f32>, //前向传播中的logit矩阵
}

pub fn sigmod(z: &f32) -> f32 {
    return 1.0 / (1.0 + E.powf(-z));
}

pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

pub fn sigmod_activation(z: Array2<f64>) -> (Array2<f64>, ActivationCache) {
    (z.mapv(|x| sigmod(&x)), ActivationCache{z})
}

pub fn relu_activation(z: Array2<f64>) -> (Array2<f64>, ActivationCache) {
    (z.mapv(|x| relu(&x)), ActivationCache{z})
}

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {

    let data = CsvReader::from_path(file_path)?.has_header(true).finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;

    return Ok((training_dataset, training_labels));

}

pub fn array_from_dataframe(df: DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>().unwrap().reversed_axes()
}

struct DeepNeuralNetwork {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
}

impl DeepNeuralNetwork {
    /// Initializes the parameters of the neural network.
    ///
    /// ### Returns
    /// a Hashmap dictionary of randomly initialized weights and biases.
    pub fn initialize_parameters(&self) -> HashMap<String, Array2<f32>> {
        let between = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let number_of_layers = self.layers.len();
        
        let mut paramsters: HashMap<String, Array2<f32>> = HashMap::new();

        for l in 1..number_of_layers {
            let weight_array: Vec<f32> = (0..self.layers[1]*self.layers[l-1])
                                        .map(|_| between.sample(&mut rng))
                                        .collect(); //create a flattened weights array of (N * M) values
        
            let bias_array: Vec<f32> = (0..self.layers[l]).map(|_| 0.0).collect();

            let weight_matrix = Array::from_shape_vec((self.layers[l], self.layers[l-1]), weight_array).unwrap();
            let bias_matrix = Array::from_shape_vec((self.layers[l], 1), bias_array).unwrap();

            let weight_string = ["W", &l.to_string()].concat();
            let bias_string = ["b", &l.to_string()].concat();

            paramsters.insert(weight_string, weight_matrix);
            paramsters.insert(bias_string, bias_matrix);
        }

        return paramsters;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
