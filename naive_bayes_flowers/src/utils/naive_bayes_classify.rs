#![allow(unused)]

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Message {
    pub feature: Vec<f64>,
    pub classification: i32,
}

#[derive(Debug)]
pub struct MessageUnknow {
    pub feature_style: Vec<usize>,
    pub feature: Vec<f64>,
}

pub struct NaiveBayesClassifier {
    tag_name: Vec<String>,
    alpha: f64,
    features_list: Vec<Vec<f64>>,
    features_counts: Vec<Vec<i32>>,
    classification_num: Vec<Vec<Vec<i32>>>,
    proportion_of_classification: Vec<f64>,

}

pub fn new_classifier(tag_name: Vec<String>, alpha: f64) -> NaiveBayesClassifier {
    return NaiveBayesClassifier {
        tag_name,
        alpha,
        features_list: Vec::new(),
        features_counts: Vec::new(),
        classification_num: Vec::new(),
        proportion_of_classification: Vec::new(),
    };
}

impl NaiveBayesClassifier {
    pub fn train( &mut self, data_set: Vec<Message>, feature_num: u32 ) {
        self.features_list.resize_with(feature_num as usize, || Vec::new());
        self.features_counts.resize_with(feature_num as usize, || Vec::new());
        self.classification_num.resize_with(feature_num as usize, || vec![Vec::new();self.tag_name.len()]);
        self.proportion_of_classification = vec![0_f64;self.tag_name.len()];

        for data_set in data_set.iter() {
            self.increment_features_count(data_set);
            self.increment_classification_count(data_set);
        }
        
        //println!("{:?}", self.features_list);
        //println!("{:?}", self.features_counts);
        //println!("{:?}", self.classification_num);
    }

    fn increment_features_count( &mut self, data_set: &Message ) {
        for (index, item) in data_set.feature.iter().enumerate() {
            if !self.features_list[index].contains(item) {
                self.features_list[index].push(*item);
                self.features_counts[index].push(1);
                for (_tag_index, _) in self.tag_name.iter().enumerate() {
                    if data_set.classification == _tag_index as i32 {
                        self.classification_num[index][_tag_index].push(1);
                    } else {
                        self.classification_num[index][_tag_index].push(0);
                    }
                }
            } else {
                if let Some(_index) = self.features_list[index].iter().position(|&x| x == *item) {
                    self.features_counts[index][_index] += 1;
                    for (_tag_index, _) in self.tag_name.iter().enumerate() {
                        if data_set.classification == _tag_index as i32 {
                            self.classification_num[index][_tag_index][_index] += 1;
                        }
                    }
                } else {
                    panic!("error in increment_features_count");
                }
            }
        }
    }

    fn increment_classification_count( &mut self, data_set: &Message ) {
        for (_tag_index, _) in self.tag_name.iter().enumerate() {
            if data_set.classification == _tag_index as i32 {
                self.proportion_of_classification[_tag_index] += 1_f64;
                break;
            }
        }
    }

    pub fn predict( &self, data_set: Vec<MessageUnknow> ) -> Vec<i32> {
        let mut tag_predict_index: Vec<i32> = Vec::new();
        
        let tag_num_all:f64 = self.proportion_of_classification.iter().sum();
        let tag_probability: Vec<f64> = self.proportion_of_classification.iter().map(|&x| (x+self.alpha) / (tag_num_all+2_f64*self.alpha)).collect();

        let predict_tag: Vec<Message> = Vec::new();
        for data_set in data_set.iter() {
            let data_feature_num = data_set.feature_style.len();
            if ( data_feature_num != data_set.feature.len() ) {
                panic!("The features of the test data are not the same as the number of feature numbers");
            }

            let mut feature_probability_in_tag = vec![0_f64;self.tag_name.len()];
            for (fea_i, &item) in data_set.feature_style.iter().enumerate() { //迭代测试样本包含的特征
                for (j, _) in self.tag_name.iter().enumerate() { //得到某特征在不同分类下的概率
                    let tag_num_sum: i32 = self.classification_num[item][j].iter().sum();
                    if let Some(index) = self.features_list[item].iter().position(|&ddd| ddd==data_set.feature[fea_i]) {
                        feature_probability_in_tag[j] += ((self.classification_num[item][j][index] as f64 + self.alpha) / (tag_num_sum as f64 + 2_f64 * self.alpha)).ln();
                        //println!("{:?}", (self.alpha / (tag_num_sum as f64 + 2_f64 * self.alpha)).ln());
                    } else {
                        feature_probability_in_tag[j] += (self.alpha / (tag_num_sum as f64 + 2_f64 * self.alpha)).ln();  //不同特征的条件概率相乘
                    }
                }
            }

            let ddd: Vec<f64> = feature_probability_in_tag.iter().zip(tag_probability.iter()).map(|(&a, &b)| a + b.ln()).collect();
            //let sss: Vec<f64> = ddd.iter().map(|&xx| xx.exp()).collect();
            
            let mut max_value = f64::NEG_INFINITY;
            let mut max_index = 0;
            for (index, &value) in ddd.iter().enumerate() {
                if value > max_value {
                    max_value = value;
                    max_index = index;
                }
            }
            tag_predict_index.push(max_index as i32);
        }

        return tag_predict_index;
    }

    //判断错误率：目前测试集与训练集相同
    pub fn get_errorrate( &self, test_data_set: Vec<Message>, predict_tag: Vec<i32> ) {
        let mut error: i32 = 0;

        for (index, item) in test_data_set.iter().enumerate() {
            if (item.classification != predict_tag[index]) {
                error += 1;
            }
        }

        println!("error rate: {:.2}%", (error*100) as f64 / test_data_set.len() as f64)
    }

}