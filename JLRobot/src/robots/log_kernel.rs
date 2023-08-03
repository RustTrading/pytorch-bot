use rand::thread_rng;
use rand::seq::SliceRandom;
use anyhow::{Result};

fn make_training_set() -> (Vec<(f64, f64)>, Vec<i64>) {
  let train_vec = vec![
    (0.2, 0.3),
    (0.1, 0.5),
    (0.2, 0.7),
    (0.3, 0.2),
    (0.3, 0.8),
    (0.4, 0.2),
    (0.4, 0.8),
    (0.5, 0.2),
    (0.5, 0.8),
    (0.6, 0.3),
    (0.7, 0.5),
    (0.6, 0.7),
    (0.3, 0.4),
    (0.3, 0.5),
    (0.3, 0.6),
    (0.4, 0.4),
    (0.4, 0.5),
    (0.4, 0.6),
    (0.5, 0.4),
    (0.5, 0.5),
    (0.5, 0.6),
  ];
  let label_vec = vec![
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ];

  (train_vec, label_vec)    
}

fn kernel(val1: &(f64, f64), val2: &(f64, f64), sigma: f64) -> f64 {
  ((-(val1.0 - val2.0) * (val1.0 - val2.0) -  (val1.1 - val2.1) * (val1.1 - val2.1))/2.0 / sigma/ sigma).exp()
}

fn log_sig(x: f64) -> f64 {
  if x < -10.0 {
    return 0.0;
  }
  else if x > 10.0 {
    return 1.0;
  }
  
  1.0 / (1.0 + (-x).exp())
}

fn assign_prob(feature : &(f64, f64), alphas: &Vec<f64>, sigma: f64, train_x: &Vec<(f64, f64)>) -> f64 {
 // bias is last cell of alphas[], 
  let n = train_x.len();  // == number of training/reference items
  let mut sum = 0.0;
  for i in 0..n {
    sum += alphas[i] * kernel(feature, &train_x[i], sigma);
  }
  sum += alphas[n];  // add the bias
  return log_sig(sum);  // result is [0.0, 1.0]
}

fn train(train_x: &Vec<(f64, f64)>, train_y: &Vec<i64>, lr: f64, max_iter: i64, sigma: f64) -> Vec<f64> {
  // train == compute the alphas
  let n = train_x.len();  // number train items
  let mut alphas = vec![0.0; n + 1];  // 1 per data item. extra cell for bias
  // 1. compute all item-item kernel values
  let mut kernels = vec![vec![1.0; n]; n]; // item-item similarity
  for i in 0..n {
    for j in 0..i {
      kernels[i][j] = kernel(&train_x[i], &train_x[j], sigma);
      kernels[j][i] = kernels[i][j]       
    }
  }
  let last_idx = alphas.len() - 1;
  for iter in 0..max_iter {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut thread_rng());
    for i in indices.into_iter() {
      let p = assign_prob(&train_x[i], &alphas, sigma, train_x);  // computed output
      let y = train_y[i] as f64;  // target 0 or 1 output
      for j in 0..alphas.len() - 1 {  // update each alpha
        alphas[j] += lr * (y - p) * kernels[i][j];
      }
      //alphas[j] += lr * (y - p) * p * (1-p) * kernels[i][j];
      alphas[last_idx] += lr * (y - p);  // update bias (dummy input)
    } // each item

    if iter % (max_iter/5) == 0 {
      let err = get_error(&train_x, &train_y, &train_x, &alphas, sigma);
      println!(" iter: {:?}, error: {:?}", iter, err);
      }
    } // main iteration loop

    return alphas;
  } 

#[test]
fn fit_log_kernel_model() {
    assert!(matches!(run_log_kernel(), Ok(())));
}

fn run_log_kernel() -> Result<()> {
  println!("Begin kernel logistic regression using demo");
  println!("not lin. sep. logistic regression gives 12/21 = .57 accuracy");

  // load training data - not lin. sep. logistic regression gives 12/21 = .57 accuracy
  let (train_x, train_y) = make_training_set();
  let max_iter = 1000;
  let lr = 0.001;
  let sigma = 0.2;  // small sigma for small values

  println!("Using RBF kernel() with sigma = {:?}", sigma);
  println!("Using SGD with lr = {:?} and max_iter = {:?}", lr, max_iter);
  println!("Starting training");
  let alphas = train(&train_x, &train_y, lr, max_iter, sigma);
  println!("training complete");
  show_some_alphas(&alphas, 3, 2);  // first 3, last 2 (last is bias)
  let acc_train = accuracy(&train_x, &train_y, &train_x, &alphas, sigma);
  println!("Model accuracy on train data: {:?} ", acc_train);
  println!("Predicting class for (0.15, 0.45)");
  let unclassified = (0.15, 0.45);
  let p = assign_prob(&unclassified, &alphas, sigma, &train_x);
   if p < 0.5 {
     println!("Computed p: {:?}, predicted class 0 ", p);
   }
   else {
     println!("Computed p: {:?}, predicted class 1", p);
   }
  Ok(())   
} 

fn get_error(
  data_x: &Vec<(f64,f64)>, 
  data_y: &Vec<i64>, 
  train_x: &Vec<(f64,f64)>,
  alphas: &Vec<f64>, sigma: f64) -> f64 {
  let n = data_x.len();
  let mut sum = 0.0;  // sum of squarede error
  for i in 0..n {
    let p = assign_prob(&data_x[i], &alphas, sigma, &train_x);  // [0.0, 1.0]
    let y = data_y[i] as f64;  // target 0 or 1
    sum += (p - y) * (p - y);
  }
  return sum / n as f64;
}

fn accuracy(
  data_x: &Vec<(f64, f64)>, 
  data_y: &Vec<i64>, 
  train_x: &Vec<(f64, f64)>, 
  alphas: &Vec<f64>, sigma: f64) -> f64 {
  let mut correct = 0; 
  let mut wrong = 0;
  let n = data_x.len();
  for i in 0..n {
    let p = assign_prob(&data_x[i], &alphas, sigma, &train_x);  // [0.0, 1.0]
    let y = data_y[i];  // target 0 or 1
    if p < 0.5 && y == 0 || p >= 0.5 && y == 1 {
      correct += 1;
    }
    else {
      wrong += 1;
    }
  }
  correct as f64/ (correct + wrong) as f64
}

pub fn show_some_alphas(alphas: &Vec<f64>, first: usize, last: usize) {
  println!("fitted alphas: from {:?}, to {:?} {:?}", first, last, &alphas[first..last + 1]);      
} 