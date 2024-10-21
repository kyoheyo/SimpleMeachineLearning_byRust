use super::read_data_set::DataOne;
use super::naive_bayes_classify::Message;
use super::naive_bayes_classify::MessageUnknow;

use rand::thread_rng;
use rand::seq::SliceRandom;

pub fn split_data_set( data_arrays: Vec<DataOne>, percent: f64, feature_selected: Vec<usize>, rand_or_not: bool ) -> (Vec<Message>, Vec<Message>, Vec<MessageUnknow>) {
    let mut rng = thread_rng();
    let selected_num = (data_arrays.len() as f64 * percent) as usize;

    let mut selected_data: Vec<DataOne> = Vec::new();
    let mut remaining_data: Vec<DataOne> = Vec::new();
    if (rand_or_not) {
        selected_data = data_arrays.choose_multiple(&mut rng, selected_num).cloned().collect();
        remaining_data = data_arrays.into_iter().filter(|x| !selected_data.contains(x)).collect();
    } else {
        for (index, item) in data_arrays.into_iter().enumerate() {
            if (index < selected_num) {
                selected_data.push(item);
            } else {
                remaining_data.push(item);
            }
        }
    }
    
    let train_data_set: Vec<Message> = selected_data.into_iter()
                                                    .map(|xx| 
                                                        Message {
                                                            feature: xx.feature.clone(),
                                                            classification: xx.tag,
                                                        }
                                                    ).collect();

    let test_data_set: Vec<Message> = remaining_data.iter()
                                                    .map(|xx| 
                                                        Message {
                                                            feature: xx.feature.clone(),
                                                            classification: xx.tag,
                                                        }
                                                    ).collect();

    let test_data: Vec<MessageUnknow> = remaining_data.into_iter()
                                                    .map(|xx| 
                                                        MessageUnknow {
                                                            feature_style: feature_selected.clone(),
                                                            feature: feature_selected.iter().map(|&index| xx.feature[index]).collect(),
                                                        }
                                                    ).collect();

    return (train_data_set, test_data_set, test_data);
}