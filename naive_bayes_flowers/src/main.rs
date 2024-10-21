#![allow(unused)]

mod utils;
use utils::RecordTuple;
use utils::DataOne;
use utils::Message;
use utils::MessageUnknow;
use utils::split_data_set;

#[cfg(test)]
#[test]
fn naive_bayes() {
    // 读取数据集
    let data_set: RecordTuple = utils::read_data_set_from_file("./bin/iris_training.csv");

    // 解构数据集
    let RecordTuple(data_num, 
                feature_num, 
                tag_name,
                data_arrays) = data_set;

    // 截取训练集、测试集以及测试数据
    let percent = 0.8;
    let feature_selected: Vec<usize> = vec![0,1,2,3];
    let rand_use_ornot = false;
    let (train_data_set, test_data_set, test_data) = split_data_set(data_arrays, percent, feature_selected, rand_use_ornot);
    
    // 新建分类器
    let alpha = 1_f64;
    let mut model = utils::new_classifier(tag_name, alpha);

    // 训练分类器
    model.train(train_data_set, feature_num);

    // 预测并给出误差结果
    let predict_tag = model.predict(test_data);
    model.get_errorrate(test_data_set, predict_tag);

}

fn main() {
    let data_set: RecordTuple = utils::read_data_set_from_file(r"E:\Active Files\varible_code\rust\machine_learning\iris_training.csv");

    let RecordTuple(data_num, 
                    feature_num, 
                    tag_name,
                    data_arrays) = data_set;

    for i in data_arrays.iter() {
        println!("{}", i.tag);
    }

    println!("{:?}", tag_name);
    println!("Hello, bayes classifier!");
}
