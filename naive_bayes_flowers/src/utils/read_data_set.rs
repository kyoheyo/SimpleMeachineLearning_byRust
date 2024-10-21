//#![allow(unused)]

use std::fs::File;
use std::io::{BufReader, BufRead, Error};

#[derive(Debug,Clone,PartialEq)]
pub struct DataOne {
    pub feature: Vec<f64>,
    pub tag: i32,
}

#[derive(Debug)]
struct Record {
    data_num: u32,
    feature_num: u32,
    tag_name: Vec<String>,
    data_arrays: Vec<DataOne>,
}

#[derive(Debug)]
pub struct RecordTuple(
            pub u32,
            pub u32,
            pub Vec<String>,
            pub Vec<DataOne>
        );

pub fn read_data_set_from_file(file_path: &str) -> RecordTuple{
    let file_input = File::open(file_path).unwrap();
    let buffered = BufReader::new(file_input);
    
    let mut contains:Vec<String> = Vec::new();
    for result in buffered.lines() {
        let record = result.unwrap();
        contains.push(record);
    }

    let data_set_tuple = string_seg_tuple(&contains);

    return data_set_tuple;
    //println!("{:?}", data_set_tuple);

    //Ok(())
}

fn string_seg(contains: &Vec<String>) -> Record{
    let data_sum = contains[0].split(',').collect::<Vec<&str>>();
    
    let mut data_array: Vec<DataOne> = Vec::new();
    for item in contains.iter().skip(1) {
        let data_temp = item.split(',').collect::<Vec<&str>>();
        let onedata_temp = DataOne{
            feature: data_temp.iter().take(data_temp.len()-1).map(|s| 
                                    s.parse::<f64>().unwrap())
                                    .collect::<Vec<f64>>()
                                    .try_into().unwrap(),
            tag: data_temp[data_temp.len()-1].parse::<i32>().unwrap(),
        };
        data_array.push(onedata_temp);
    }

    let record = Record {
                    data_num: data_sum[0].parse::<u32>().unwrap(),
                    feature_num: data_sum[1].parse::<u32>().unwrap(),
                    tag_name: data_sum.iter().skip(2).map(|s| 
                                                        s.to_string())
                                                        .collect::<Vec<String>>()
                                                        .try_into().unwrap(),
                    data_arrays: data_array,
    };

    return record;
}

fn string_seg_tuple(contains: &Vec<String>) -> RecordTuple{
    let data_sum = contains[0].split(',').collect::<Vec<&str>>();
    
    let mut data_array: Vec<DataOne> = Vec::new();
    for item in contains.iter().skip(1) {
        let data_temp = item.split(',').collect::<Vec<&str>>();
        let onedata_temp = DataOne{
            feature: data_temp.iter().take(data_temp.len()-1).map(|s| 
                                    s.parse::<f64>().unwrap())
                                    .collect::<Vec<f64>>()
                                    .try_into().unwrap(),
            tag: data_temp[data_temp.len()-1].parse::<i32>().unwrap(),
        };
        data_array.push(onedata_temp);
    }

    let record = RecordTuple(
                                    data_sum[0].parse::<u32>().unwrap(),
                                    data_sum[1].parse::<u32>().unwrap(),
                                    data_sum.iter().skip(2).map(|s| 
                                        s.to_string())
                                        .collect::<Vec<String>>()
                                        .try_into().unwrap(),
                                    data_array
    );

    return record;
}