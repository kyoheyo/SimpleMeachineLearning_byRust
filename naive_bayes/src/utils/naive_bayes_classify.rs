//#![allow(unused)]

use std::collections::{HashSet,HashMap};

extern crate regex;
use regex::Regex;

// 将语句转化为token（词语）
pub fn tokenize(lower_case_text: &str) -> HashSet<&str> {
    Regex::new(r"[a-z0-9']+")
        .unwrap()
        .find_iter(lower_case_text)
        .map(|mat| mat.as_str())
        .collect()
}

// 单个邮件样本
pub struct Message<'a> {
    pub text: &'a str,
    pub is_spam: bool,
}

// 分类器结构体
pub struct NaiveBayesClassifier {
    pub alpha: f64,
    pub tokens: HashSet<String>,
    pub token_ham_counts: HashMap<String, i32>,
    pub token_spam_counts: HashMap<String, i32>,
    pub spam_messages_count: i32,
    pub ham_messages_count: i32,
}

pub fn new_classifier(alpha: f64) -> NaiveBayesClassifier {
    return NaiveBayesClassifier {
        alpha,
        tokens: HashSet::new(),
        token_ham_counts: HashMap::new(),
        token_spam_counts: HashMap::new(),
        spam_messages_count: 0,
        ham_messages_count: 0,
    };
}

impl NaiveBayesClassifier {
    pub fn train( &mut self, message: &[Message]) {
        for message in message.iter() {
            self.increment_message_classifications_count(message);
            for token in tokenize(&message.text.to_lowercase()) {
                self.tokens.insert(token.to_string());
                self.increment_token_count(token, message.is_spam);
            }
        }
    }

    fn increment_message_classifications_count(&mut self, message: &Message) {
        if message.is_spam {
            self.spam_messages_count += 1;
        } else {
            self.ham_messages_count += 1;
        }
    }

    fn increment_token_count( &mut self, token: &str, is_spam: bool) {
        if !self.token_spam_counts.contains_key(token) {
            self.token_spam_counts.insert(token.to_string(), 0);
        }

        if !self.token_ham_counts.contains_key(token) {
            self.token_ham_counts.insert(token.to_string(), 0);
        }

        if is_spam {
            self.increment_spam_count(token);
        } else {
            self.increment_ham_count(token);
        }
    }
    
    fn increment_spam_count( &mut self, token: &str ) {
        *self.token_spam_counts.get_mut(token).unwrap() += 1;
    }

    fn increment_ham_count( &mut self, token: &str ) {
        *self.token_ham_counts.get_mut(token).unwrap() += 1;
    }

    pub fn predict( &self, text: &str ) -> f64 {
        let lower_case_text = text.to_lowercase();
        let message_tokens = tokenize(&lower_case_text);
        let (prob_if_spam, prob_if_ham) = self.probabilities_of_message(message_tokens);

        return prob_if_spam / (prob_if_spam + prob_if_ham);
    }

    fn probabilities_of_message( &self, message_tokens: HashSet<&str> ) -> (f64, f64) {
        let mut log_prob_if_spam = 0_f64;
        let mut log_prob_if_ham = 0_f64;

        for token in self.tokens.iter() {
            let (prob_if_spam, prob_if_ham) = self.probabilities_of_token(&token);
            
            if message_tokens.contains(token.as_str()) {
                log_prob_if_spam += prob_if_spam.ln();
                log_prob_if_ham += prob_if_ham.ln();
            } else {
                log_prob_if_spam += (1. - prob_if_spam).ln();
                log_prob_if_ham += (1. - prob_if_ham).ln();
            }
        }

        let prob_if_spam = log_prob_if_spam.exp();
        let prob_if_ham = log_prob_if_ham.exp();

        return (prob_if_spam, prob_if_ham);
    }

    fn probabilities_of_token( &self, token: &str ) -> (f64, f64) {
        let prob_of_token_spam = (self.token_spam_counts[token] as f64 + self.alpha)
            / (self.spam_messages_count as f64 + 2.*self.alpha);

        let prob_of_token_ham = (self.token_ham_counts[token] as f64 + self.alpha)
            / (self.ham_messages_count as f64 + 2.*self.alpha);

        return (prob_of_token_spam, prob_of_token_ham);
    }
}