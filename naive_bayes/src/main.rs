#![allow(unused)]

use std::collections::{HashSet,HashMap};

mod utils;

#[cfg(test)]
#[test]
fn naive_bayes() {
    let train_messages = [
        utils::Message {
            text: "Free Bitcoin viagra XXX christmas deals 😻😻😻",
            is_spam: true,
        },
        utils::Message {
            text: "My dear Granddaughter, please explain Bitcoin over Christmas dinner",
            is_spam: false,
        },
        utils::Message {
            text: "Here in my garage...",
            is_spam: true,
        },
    ];

    let alpha = 1_f64;
    let num_spam_messages = 2;
    let num_ham_messages = 1;

    let mut model = utils::new_classifier(alpha);
    model.train(&train_messages);

    let mut expected_tokens: HashSet<String> = HashSet::new();
    for message in train_messages.iter() {
        for token in utils::tokenize(&message.text.to_lowercase()) {
            expected_tokens.insert(token.to_string());
        }
    }

    let input_text = "Bitcoin crypto academy Christmas deals";

    println!("{}",model.predict(input_text));

}

fn main() {
    println!("Hello, naive bayes!");
}
