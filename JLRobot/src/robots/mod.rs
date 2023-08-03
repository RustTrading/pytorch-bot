use serde::{Serialize, Serializer, Deserialize, ser::SerializeSeq};
use chrono::{Utc, DateTime, NaiveDateTime};
use std::{fs::File, fmt::Debug};
use tch::{nn, nn::Module, Device, Tensor, Kind};
use std::{io::{Write, Read}, marker::PhantomData};
#[cfg(test)]
use std::io::{BufReader};
#[cfg(test)]
use std::fs;
#[cfg(test)]
use anyhow::{Result};
#[cfg(test)]
use std::slice;

pub mod data_collection;
pub mod config;
pub mod log_kernel;

use data_collection::{
  FeatureVolume, 
  FeatureTrade, 
  TradingContext, 
  convert_1d_tensor_to_vec, 
  convert_2d_tensor_to_vec,
};

use config::{
  DateTimeOwner,
  InstrumentValue,
};

pub struct Model {
  input_output_dim: (i64, i64),
  pub m: Box<nn::Sequential>,
  pub vs: nn::VarStore,
}

#[derive(Clone, Serialize, Debug)]
enum OrderType {
  BuyLimit,
  SellLimit,
  BuyMarket,
  SellMarket,
}

#[derive(Clone, Serialize, Debug)]
pub enum Instrument {
  BTC,
  BNB,
}

#[derive(Serialize, Debug)]
pub struct Order {
  life_time: Vec<DateTimeOwner>,
  order_type: OrderType,
  instrument: Instrument,
  price: f64,
  size: f64,
}

impl Model {
  fn new(input_output_dim: (i64, i64)) -> Self {
    let vs = nn::VarStore::new(Device::Cpu);
    let m = Box::new(nn::seq()
    .add(nn::linear(&vs.root() / "layer1", input_output_dim.0, input_output_dim.0 / 2, Default::default()))
    .add_fn(|xs| xs.relu())
    .add(nn::linear(&vs.root() / "layer2", input_output_dim.0 / 2, input_output_dim.1, Default::default())));
    Model {
      input_output_dim,
      m,
      vs,
    }
  }

  pub fn train_and_test(&self, epoch_num: i64, fset: &FeatureSet) -> anyhow::Result<()> {
    let force_sparse = false;
    let mut opt = sparse_adam::SparseAdam::new(&self.vs, 5e-3, 0.9, 0.999, 1e-8, force_sparse);
    for epoch in 0..epoch_num {
      let loss =  self.m.forward(&fset.train_features.tensor).log_softmax(1, Kind::Float).nll_loss(&fset.train_labels.tensor);
      //let loss = self.m.forward(&fset.train_features).cross_entropy_for_logits(&fset.train_labels);
      opt.zero_grad();
      loss.backward();
      opt.step();  
      let test_diff = self.m.forward(&fset.test_features.tensor).softmax(1, Kind::Float).argmax(1, false) - &fset.test_labels.tensor;
      let train_size = fset.test_labels.tensor.numel() as f64;
      let test_accuracy = (train_size - f64::from(test_diff.count_nonzero(0))) / train_size;
      println!(
        "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
        epoch,
        f64::from(&loss),
        100. * test_accuracy,
      );
  }
  Ok(())
  } 
}

#[derive(Serialize, Deserialize, Clone)]
pub enum RobotType {
  SimpleLogistic,
}

use std::fmt;
impl fmt::Display for RobotType {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    f.write_str(match self {
      RobotType::SimpleLogistic => "SimpleLogistic",
    })
  }
}

pub struct Robot {
  pub robot_config: RobotConfig,
  jump_ranges: Vec<(f64, f64, i64)>,
  pub model: Model,
  pub open_positions: Vec<Order>,
  pub closed_position: Vec<Order>,
}

pub struct RobotConfig {
  pub robot_type: RobotType,
  pub training_rate: f64,
  pub max_ob_level: i64,
  pub quantile_ratios: Vec<f64>,
  pub quantile_margins: Vec<f64>,
  pub jump_horizon: i64,
  pub trade_history_length: i64,
  pub balance: Vec<InstrumentValue>,
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
  bias: Vec<f32>,
  weights: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelExport {
  robot_type: RobotType,
  label_ranges: Vec<(f64, f64, i64)>,
  layers: Vec<Layer>,
}

impl Robot {
  pub fn new(robot_config: RobotConfig) -> Self {
    let  feature_size = robot_config.max_ob_level * 6;
    let model = create_model((feature_size, robot_config.quantile_ratios.len() as i64 + 2));
    Self {
      robot_config,
      jump_ranges: Vec::new(),
      model,
      open_positions: Vec::new(),
      closed_position: Vec::new(),
    }
  }

  pub fn execute_orders(self) {

  }

  fn get_range_price(&self, probabilities: Vec<Vec<f32>>, current_price: f64) -> f64 {
    let maxprob = probabilities[0].iter().max_by(|x,y| x.partial_cmp(&y).unwrap()).unwrap();
    let idx = probabilities[0].iter().position(|r| r == maxprob).unwrap();
    let price_range = if idx == 0 {
      self.jump_ranges[idx].1
    } else if idx == probabilities[0].len() - 1 {
      self.jump_ranges[idx].0
    } else {
      (self.jump_ranges[idx].0 + self.jump_ranges[idx].1) / 2.0
    };
    (1.0 + price_range) * current_price
  }
  
  pub fn create_orders(&self, features: (&FeatureVolume, &FeatureVolume, &FeatureTrade), price_tick_size: f64) -> Option<Vec<Order>> {
    let features_labeled = data_collection::extract_price_levels(
      &self.jump_ranges,
      features,
      self.robot_config.max_ob_level,
      price_tick_size
    ).unwrap();
    let test_features  = data_collection::convert_features_to_tensor(vec![features_labeled.0]);
    let test_diff = self.model.m.forward(&test_features.tensor).softmax(1, Kind::Float);
    test_diff.print();
    let probabilities = convert_2d_tensor_to_vec::<f32>(&test_diff);
    let price_predicted = self.get_range_price(probabilities, features.0.current_price);
    println!("price prediction: current {}, predicited {}, actual {}", 
      features.0.current_price, price_predicted, (features.0.rel_price + 1.0) * features.0.current_price);
    let mut orders = Vec::<Order>::new();
    let size = f64::min(features.0.ask_volume[0].1, features.0.bid_volume[0].1) / 100.0; 
    let dt_start = DateTime::<Utc>::from_utc(NaiveDateTime::from_timestamp(features.0.ts.0 / 1000000, 0), Utc);
    let dt_end = DateTime::<Utc>::from_utc(NaiveDateTime::from_timestamp(features.0.ts.1 / 1000000, 0), Utc);
    if price_predicted > features.0.current_price {
      let order1 = Order {
        life_time: vec![DateTimeOwner{dt: dt_start}, DateTimeOwner{dt: dt_end}], 
        order_type: OrderType::BuyLimit, 
        instrument: features.0.instrument.clone(), 
        price: features.0.bid_volume[0].0, 
        size 
      };
      let order2 = Order { 
        life_time: vec![DateTimeOwner{dt: dt_start}, DateTimeOwner{dt: dt_end}],
        order_type: OrderType::SellLimit,
        instrument: features.0.instrument.clone(), 
        price: price_predicted, 
        size 
      };
      orders.push(order1);
      orders.push(order2);
    } else {
      let order1 = Order { 
        life_time: vec![DateTimeOwner{dt: dt_start}, DateTimeOwner{dt: dt_end}],
        order_type: OrderType::SellLimit, 
        instrument: features.0.instrument.clone(), 
        price: features.0.ask_volume[0].0, 
        size 
      };
      let order2 = Order { 
        life_time: vec![DateTimeOwner{dt: dt_start}, DateTimeOwner{dt: dt_end}],
        order_type: OrderType::BuyLimit, 
        instrument: features.0.instrument.clone(), 
        price: price_predicted, 
        size 
      }; 
      orders.push(order1);
      orders.push(order2);
    }
    Some(orders)
  }

  pub fn train_and_test(
    &mut self, 
    epochs: i64, 
    tcontext: &TradingContext,
    price_tick_size: f64, 
    export_training_set: bool
  ) -> Option<(Vec<i64>, Vec<i64>)> {
    let jump_ranges = data_collection::create_jump_range_labels(
      &tcontext, self.robot_config.quantile_ratios.clone(), 
      self.robot_config.quantile_margins.clone()
    );
    let mut all_features = data_collection::make_nn_features(
      &jump_ranges, &tcontext, 
      self.robot_config.max_ob_level,
      price_tick_size
    ).unwrap();
      let fset = data_collection::make_training_and_test_features(
      &mut all_features, 
      self.robot_config.training_rate
    );
    if export_training_set {
      let mut f = File::create("training.json").unwrap();
      let export_fset_str: String = serde_json::to_string_pretty(&fset).unwrap();
      write!(f, "{}", &export_fset_str).unwrap();
    }
    self.model.train_and_test(epochs, &fset).ok()?;
    self.jump_ranges = jump_ranges;
    Some((fset.train_features.tensor.size(), fset.test_features.tensor.size()))
  }

  pub fn test_accuracy(&self, tcontext: &TradingContext, price_tick_size: f64) -> Option<f64> {
    let mut features_joined = data_collection::make_nn_features(&self.jump_ranges, tcontext, self.robot_config.max_ob_level, price_tick_size)?;
    let fset = data_collection::make_training_and_test_features(&mut features_joined, 0.0);
    let test_diff = self.model.m.forward(&fset.test_features.tensor).softmax(1, Kind::Float).argmax(1, false) - &fset.test_labels.tensor;
    let train_size = fset.test_labels.tensor.numel() as f64;
    Some((train_size - f64::from(test_diff.count_nonzero(0))) / train_size)
  }

  pub fn export_model(&self, path: &str) -> anyhow::Result<()> {
    println!("exporting model to {}", path);
    let vars = self.model.vs.trainable_variables();
    let bias = convert_1d_tensor_to_vec(&vars[0]);
    let weights = convert_2d_tensor_to_vec(&vars[1]);
    let export_model = ModelExport { 
      robot_type: self.robot_config.robot_type.clone(), 
      label_ranges: self.jump_ranges.clone(),
      layers: vec![Layer { bias, weights }], };
    let export_model_str: String = serde_json::to_string_pretty(&export_model)?;
    let mut f = File::create(format!("{}.json", path))?;
    write!(f, "{}", &export_model_str)?;
    self.model.vs.save(format!("{}.pt", path))?;
    Ok(())
  }

  pub fn load_model(&mut self, path: &str) -> anyhow::Result<()> {
    println!("loading model from {}.pt", path);
    self.model.vs.load(format!("{}.pt", path))?;
    let mut f = File::open(format!("{}.json", path))?;
    let mut buf = String::new();
    f.read_to_string(&mut buf)?;
    let res: ModelExport= serde_json::from_str(&buf)?;
    self.jump_ranges = res.label_ranges;
    Ok(())
  }
}

fn create_model(input_output_dim : (i64, i64)) -> Model {
  Model::new(input_output_dim)
}
#[cfg(test)]
const FEATURE_DIM: i64 = 4; 
#[cfg(test)]
const LABELS: i64 = 3;
pub mod sparse_adam;
#[cfg(test)]
fn net(vs: &nn::Path) -> impl Module {
  nn::seq()
  .add(nn::linear(vs / "layer1", FEATURE_DIM, LABELS, Default::default()))
  //.add_fn(|xs| xs.softmax(1, Kind::Float))
  //.add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

#[derive(Debug, Serialize)]
pub struct FeatureSet {
  pub train_features: TensorOwner<f32>,
  pub train_labels: TensorOwner<i64>,
  pub test_features: TensorOwner<f32>,
  pub test_labels: TensorOwner<i64>,
}

#[derive(Debug)]
pub struct TensorOwner<T> {
  tensor: Tensor,
  _marker: PhantomData<T>,
}

impl<T: Clone + Serialize> Serialize for TensorOwner<T> {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
      S: Serializer,
  {
    let s = self.tensor.size();
    if s.len()  == 1 {
      let vec1d = convert_1d_tensor_to_vec::<T>(&self.tensor);
      let mut seq = serializer.serialize_seq(Some(vec1d.len()))?;
      for element in vec1d.iter() {
        seq.serialize_element(element)?;
      }
      return seq.end();
    }; 
    let vec2d = convert_2d_tensor_to_vec::<T>(&self.tensor);
    let mut seq = serializer.serialize_seq(Some(vec2d.len()))?;
    for element in vec2d.iter() {
       seq.serialize_element(element)?;
    }
    seq.end()
  }
}

#[cfg(test)]
fn read_f64<T: Read>(reader: &mut T) -> Result<f64> {
  let mut b = [0u8; 8];
  reader.read_exact(&mut b)?;
  Ok(f64::from_ne_bytes(b))
}
#[cfg(test)]
fn read_i64<T: Read>(reader: &mut T) -> Result<i64> {
  let mut b = [0u8; 8];
  reader.read_exact(&mut b)?;
  Ok(i64::from_ne_bytes(b))
}
#[cfg(test)]
fn read_features(filename: &std::path::Path) -> Result<(Tensor, i64)> {
  let mut buf_reader = BufReader::new(fs::File::open(filename)?);
  let sample_size = read_f64(&mut buf_reader)?;
  let feature_size = read_f64(&mut buf_reader)?;
  let mut data = vec![0f64; sample_size as usize];
  let ptr = data.as_mut_ptr() as *mut u8;
  let slice = unsafe { slice::from_raw_parts_mut(ptr, 8 * sample_size as usize) };
  buf_reader.read_exact(slice)?;
  
  let tensor = Tensor::of_slice::<f64>(&data)
    .view(((sample_size / feature_size) as i64, feature_size as i64))
    .to_kind(Kind::Float);

  Ok((tensor, feature_size as i64))      
}

#[cfg(test)]
fn read_labels(filename: &std::path::Path) -> Result<Tensor> {
  let mut buf_reader = BufReader::new(fs::File::open(filename)?);
  let sample_size = read_i64(&mut buf_reader)?;
  let mut data = vec![0i64; sample_size as usize];
  let ptr = data.as_mut_ptr() as *mut u8;
  let slice = unsafe { slice::from_raw_parts_mut(ptr, 8 * sample_size as usize) };
  buf_reader.read_exact(slice)?;
  
  let tensor = Tensor::of_slice(&data).to_kind(Kind::Int64);
       
  Ok(tensor)
}

#[cfg(test)]
fn load_feature<T: AsRef<std::path::Path>>(dir: T) -> Result<FeatureSet> {
  let dir = dir.as_ref();
  let train_features = read_features(&dir.join("train-features"))?;
  let train_labels = read_labels(&dir.join("train-labels"))?;
  let test_features = read_features(&dir.join("test-features"))?;
  let test_labels = read_labels(&dir.join("test-labels"))?;
  
  Ok(FeatureSet { 
    train_features: TensorOwner::<f32>{ tensor: train_features.0, _marker: PhantomData }, 
    train_labels: TensorOwner::<i64>{ tensor: train_labels, _marker: PhantomData}, 
    test_features: TensorOwner::<f32>{ tensor: test_features.0, _marker: PhantomData} , 
    test_labels: TensorOwner::<i64>{ tensor: test_labels, _marker: PhantomData}, 
  })
}

#[cfg(test)]
mod model {
  use super::*;
  #[test]
  fn run_pytorch() {
    assert!(matches!(run(), Ok(())));
  }
}

#[cfg(test)]
pub fn run() -> Result<()> {
  let vs = nn::VarStore::new(Device::Cpu);
  let net = net(&vs.root());
  let m = load_feature("data")?;
  let force_sparse = false;
  let mut opt = sparse_adam::SparseAdam::new(&vs, 5e-3, 0.9, 0.999, 1e-8, force_sparse);
  for epoch in 1..350 {
    let loss =  net.forward(&m.train_features.tensor).log_softmax(1, Kind::Float).nll_loss(&m.train_labels.tensor);
    //cross_entropy_for_logits(&m.train_labels);
    //let loss = net.forward(&m.train_features).log_softmax(-1, Kind::Float).nll_loss(&m.train_labels);
    opt.zero_grad();
    loss.backward();
    opt.step();  
    let test_diff = net.forward(&m.test_features.tensor).softmax(1, Kind::Float).argmax(1, false) - &m.test_labels.tensor;
    let train_size = m.test_labels.tensor.numel() as f64;
    let test_accuracy = (train_size - f64::from(test_diff.count_nonzero(0))) / train_size;
    println!(
        "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
        epoch,
        f64::from(&loss),
        100. * test_accuracy,
    );
}
  Ok(())
}
