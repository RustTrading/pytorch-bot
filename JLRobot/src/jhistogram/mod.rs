#[cfg(test)]
use rand::{rngs::StdRng, SeedableRng};
use std::collections::{BTreeMap};
#[cfg(test)]
use rand_distr::{ Normal, Distribution };
#[cfg(test)]
use mix_distribution::Mix;
use serde::{Serialize};
use std::cmp::{min_by, max_by};
use plotters::element::{ Rectangle };
use plotters::style::{ShapeStyle, GREEN, Color};

pub mod dataview;

#[derive(Serialize)]
pub struct Histogram {
  idx: usize,
  #[serde(skip_serializing)]
  style: fn(&f64, &u64) -> ShapeStyle,
  bin_number: i64,
  time_spread: i64,
  points_num: i64,
  start_tick_time: i64,
  end_tick_time: i64,
  limits: (f64, f64),
  raw_ticks: Vec<(i64, f64)>,
  binticks: Vec<(i64, BTreeMap<i64, f64>)>,
  bincounts: (Vec<(f64, f64)>, Vec<i64>),
  description: String,
}

impl Clone for Histogram {
  fn clone(&self) -> Self {
    Self {
      idx: self.idx,
      style: self.style,
      bin_number: self.bin_number,
      time_spread: self.time_spread,
      points_num: self.points_num,
      start_tick_time: self.start_tick_time,
      end_tick_time: self.end_tick_time,
      limits: self.limits,
      raw_ticks: self.raw_ticks.clone(),
      binticks: self.binticks.clone(),
      bincounts: self.bincounts.clone(),
      description: self.description.clone(),
    }
  }
}

impl Iterator for Histogram {
    type Item = Rectangle<(f64, f64)>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((idx, y)) = self.binticks.get(self.idx) {
          let binstep = (self.limits.1 - self.limits.0) / self.bin_number as f64; 
          let v1 = binstep * (*idx as f64) + self.limits.0;
          let v2 = v1 + binstep;
          let style = (self.style)(&v1, &(y.len() as u64));
          let mut rect = Rectangle::<(f64,f64)>::new([(v1, 0f64), (v2, y.len() as f64)], style);
          rect.set_margin(0, 0, 1, 1);
          self.idx += 1;
          return Some(rect);  
        }
        None
    }
} 

impl Histogram {
  pub fn new(bin_number: i64, time_spread: i64, points: &Vec<(i64, f64)>, description: &str) -> Self {
    let (mintime, maxtime, minjump, maxjump) = points.iter().
    fold((i64::MAX,i64::MIN,f64::MAX, f64::MIN), |prev, current| (
      min_by(prev.0, current.0, |a, b| a.partial_cmp(&b).unwrap()),
      max_by(prev.1, current.0, |a, b| a.partial_cmp(&b).unwrap()),
      min_by(prev.2, current.1, |a, b| a.partial_cmp(&b).unwrap()),
      max_by(prev.3, current.1, |a, b| a.partial_cmp(&b).unwrap()),
    ));
    let mut histo = Self {
      idx: 0,
      style: |_, _| GREEN.filled(),
      bin_number: bin_number,
      time_spread: time_spread,
      points_num: points.len() as i64,
      start_tick_time: mintime,
      end_tick_time: maxtime,
      limits: (minjump, maxjump),
      raw_ticks: points.clone(),
      binticks: (0..bin_number).into_iter().zip(vec!(BTreeMap::new(); bin_number as usize)).collect::<Vec::<(i64, BTreeMap::<i64, f64>)>>(),
      bincounts: (Vec::new(), Vec::new()),
      description: String::from(description),
    };
    histo.bincounts = histo.get_histogram();
    histo 
  } 
  
  pub fn update_bin_content(&mut self) {
    self.idx = 0;
    for i in 0..self.bin_number as usize {
      self.binticks[i].1.clear();
    }
    let (mintime, maxtime) = (self.raw_ticks).iter().
    fold((i64::MAX,i64::MIN), |prev, current| (
      min_by(prev.0, current.0, |a, b| a.partial_cmp(&b).unwrap()),
      max_by(prev.1, current.0, |a, b| a.partial_cmp(&b).unwrap()),
    ));
    if maxtime - mintime > self.time_spread {
      self.raw_ticks = (self.raw_ticks).iter().filter(|tick| tick.0 >= maxtime - self.time_spread).map(|val| *val).collect();
    }
    self.limits = (self.raw_ticks).iter().
    fold((f64::MAX, f64::MIN), |prev, current| (
      min_by(prev.0, current.1, |a, b| a.partial_cmp(&b).unwrap()),
      max_by(prev.1, current.1, |a, b| a.partial_cmp(&b).unwrap()),
    ));
    let old_ticks = self.raw_ticks.clone();
    self.raw_ticks.clear();
    old_ticks.iter().for_each(|tick| self.add_tick(*tick));
  }

  fn add_tick(&mut self, tick: (i64, f64)) {
    let idx = self.get_bin_index(tick);
    self.raw_ticks.push(tick);
    if tick.1 >= self.limits.0 && tick.1 <= self.limits.1 {
        self.binticks[idx].1.insert(tick.0, tick.1);
    } else {
      self.update_bin_content();
    }
  }

  fn get_bin_index(&self, tick: (i64, f64)) -> usize {
    if tick.1 >= self.limits.0 {
      let binstep = (self.limits.1 - self.limits.0) / self.bin_number as f64;
      let idx =  ((tick.1 - self.limits.0) / binstep).floor();
      return if idx < self.bin_number as f64 { idx as usize } else { self.bin_number as usize - 1 };
    }
    0
  }
  
  pub fn get_histogram(&mut self) -> (Vec<(f64, f64)>, Vec<i64>) {
    self.update_bin_content();
    let mut binends: Vec<(f64, f64)> = Vec::new();
    let mut values: Vec<i64> = Vec::new();
    let binstep = (self.limits.1 - self.limits.0) / self.bin_number as f64; 
    for i in 0..self.bin_number as usize {
      binends.push((binstep * i as f64 + self.limits.0, binstep * (i as f64  + 1.0) + self.limits.0));
      values.push(self.binticks[i].1.len() as i64); 
    }
    (binends, values)
  }
  #[cfg(test)]
  pub fn dump(&self) {
    println!("Histogram {:?}: {:?}", self.description, self.binticks);
  }
}

#[cfg(test)]
pub fn generate_samples(means : &Vec<f64>, 
  stdevs : &Vec<f64>, weights : &Vec<f64>, 
  sample_size : usize, number_of_samples : usize) -> Vec<Vec<f64>> {
  let seed: [u8; 32] = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let mut rng: StdRng = SeedableRng::from_seed(seed);
  let mut samples : Vec<Vec<f64>> = vec![Vec::<f64>::new(); 0];
  for _ in 0..number_of_samples {
    samples.push(sample_from_mixture(means, stdevs, weights, sample_size, &mut rng));
  }  
  samples 
}
#[cfg(test)]
pub fn sample_from_mixture(means : &Vec<f64>, 
  stdevs : &Vec<f64>, weights : &Vec<f64>, 
  sample_size : usize, rng : &mut StdRng) -> Vec<f64> {
  let mix = {
    let dists  = (0..weights.len()).map(|idx|{Normal::new(means[idx], stdevs[idx]).unwrap()});
    Mix::new(dists, weights).unwrap()
  };
  (0..sample_size).map(|idx| (&mix).sample(rng)).collect()
}

pub fn data_range(data : &Vec<f64>) -> (f64, f64) {
  let minval = data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
  let maxval = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
  (*minval, *maxval)
}

pub fn gaussian_kernel_smoother_many(data_samples : &Vec<Vec<f64>>, bandwidth : f64) -> Vec<Vec<(f64, f64)>> {
  let mut data_smoothed_many : Vec<Vec<(f64,f64)>>= vec![Vec::new(); 0];
  for data in data_samples {
    let mut data_smoothed = Vec::<(f64, f64)>::new();
    let (minval, maxval) = data_range(data);
    let binstep = (maxval - minval)/2.0/( data.len() as f64);
    for i in 0..2 * data.len()+1 {
      let mut smoothed = 0.0;
      let val1 = (i as f64) * binstep + minval; 
      for val2 in data {
        smoothed += (-(val1 - val2) * (val1 - val2)/2.0/bandwidth/bandwidth).exp()/(2.0 * std::f64::consts::PI).sqrt();
      }
      data_smoothed.push((val1, smoothed));
    }
    data_smoothed_many.push(data_smoothed);
  }
  data_smoothed_many    
}

pub fn epanechnikov_kernel_smoother_many(data_samples : &Vec<Vec<f64>>, bandwidth : f64) -> Vec<Vec<(f64, f64)>> {
  let mut data_smoothed_many : Vec<Vec<(f64,f64)>>= vec![Vec::new(); 0];
  for data in data_samples {
    let mut data_smoothed = Vec::<(f64, f64)>::new();
    let (minval, maxval) = data_range(data);
    let binstep = (maxval - minval)/2.0/( data.len() as f64);
    for i in 0..2 * data.len()+1 {
      let mut smoothed = 0.0;
      let val1 = (i as f64) * binstep + minval; 
      for val2 in data {
        let val = 1.0 -(val1 - val2) * (val1 - val2)/bandwidth/bandwidth;
        if val > 0.0 {
          smoothed += val;
        }
      }
      data_smoothed.push((val1, smoothed));
    }
    data_smoothed_many.push(data_smoothed);
  }
  data_smoothed_many    
}
#[cfg(test)]
pub fn tophat_kernel_smoother_many(data_samples : &Vec<Vec<f64>>, bandwidth : f64) -> Vec<Vec<(f64, f64)>> {
  let mut data_smoothed_many : Vec<Vec<(f64,f64)>>= vec![Vec::new(); 0];
  for data in data_samples {
    let mut data_smoothed = Vec::<(f64, f64)>::new();
    let (minval, maxval) = data_range(data);
    let binstep = (maxval - minval)/2.0/( data.len() as f64);
    for i in 0..2 * data.len()+1 {
      let mut smoothed = 0.0;
      let val1 = (i as f64) * binstep + minval; 
      for val2 in data {
        let val = { 
          if (val1 - val2).abs() <= bandwidth { 
            1.0 
          } else { 
            0.0 
          } 
        };
        if val > 0.0 {
          smoothed += val;
        }
      }
      data_smoothed.push((val1, smoothed));
    }
    data_smoothed_many.push(data_smoothed);
  }
  data_smoothed_many    
}
#[cfg(test)]
pub fn gaussian_kernel_smoother(data : &Vec<f64>, bandwidth : f64) -> Vec<(f64, f64)> {
  let mut data_smoothed = Vec::<(f64, f64)>::new();
  let (minval, maxval) = data_range(data);
  let binstep = (maxval - minval)/2.0/( data.len() as f64);
  for i in 0..2 * data.len()+1 {
    let mut smoothed = 0.0;
    let val1 = (i as f64) * binstep + minval; 
    for val2 in data {
      smoothed += (-(val1 - val2) * (val1 - val2)/2.0/bandwidth/bandwidth).exp()/(2.0 * std::f64::consts::PI).sqrt();
    }
  data_smoothed.push((val1, smoothed));
  }
  data_smoothed    
}
#[cfg(test)]
pub fn make_moment_series(data : &Vec<(f64, f64)>, order : f64) -> Vec<(f64, f64)> {
  data.iter().map(|val| { (val.0, val.0.powf(order) * val.1) }).collect::<Vec<(f64, f64)>>()
}

#[cfg(test)]
mod ticks {
  use super::*;
  #[test]
  fn expired_ticks() {
    let data: Vec<(i64, f64)> = vec![
      (0, 10.0),
      (1, 1000.1),
      (2, 4.0),
      (3, 1000.5),
      (4, 1000.75), 
      (5, 4.1),
      (6, 12.1),
      (12, 15.0)
    ];
    let mut histo = Histogram::new(
      4,
      10, 
      &data, 
      &"test1"
    );
    let _ = histo.get_histogram();
    assert_eq!(histo.raw_ticks.len(), data.len() - 2);
  }
  #[test]
  fn boundary_tick() {
    let data: Vec<(i64, f64)> = vec![
      (0, 1.0),
      (1, 2.0),
      (2, 3.0),
      (3, 4.0),
      (4, 5.0), 
      (5, 6.0),
      (6, 7.0),
      (7, 8.0),
      (8, 9.0),
      (9, 10.0),
      (10, 11.0),
      (11, 12.0)
    ];
    let mut histo = Histogram::new(
      4,
      11, 
      &data, 
      &"test2"
    );
    let (_,bins) = histo.get_histogram();
    /*
    println!("histogram: {:?}", (&edges, &bins));
    println!("all ticks: {:?}", data);  
    println!("binned ticks: {:?}", histo.binned_ticks);
    */
    assert_eq!(&bins[..], [3, 3, 3, 3]);
  }
}


