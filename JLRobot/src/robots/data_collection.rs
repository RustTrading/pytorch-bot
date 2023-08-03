use std::collections::{HashMap, HashSet, BTreeMap};
use xz2::stream::{Action, Stream};
use std::io::{Read, prelude::*};
use std::ops::Bound::{Included, Excluded};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, TimeZone};
use anyhow::{anyhow, Result};
use tch::{Tensor, Kind};
use serde_json::{Value};
use std::str::FromStr;
use std::num::ParseIntError;
use rand::thread_rng;
use rand::seq::SliceRandom;
use regex::Regex;
use std::{fs, io, fmt::Debug, marker::PhantomData, slice::Iter};
use super::{ TensorOwner, config::TimeValue, config::TimeUnits };
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub fn convert_time(value: TimeValue, to_units: TimeUnits) -> TimeValue {
  match value.units {
    TimeUnits::Seconds => {
      match to_units {
        TimeUnits::Microseconds => {
          return TimeValue {
            value: value.value * 1000000,
            units: TimeUnits::Microseconds,
          }
        },
        _ => {}
      }
    },
    _ => {}
 }
  value
}

struct DateTimeSquizzed {
  year: i32,
  month: u32,
  day: u32,
  hours: u32,
  minutes: u32,
  seconds: u32,
  micro: u32,
}

impl FromStr for DateTimeSquizzed {
  type Err = ParseIntError;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
      let year = String::from(&s[0..4]);
      let month = String::from(&s[4..6]);
      let day = String::from(&s[6..8]);
      let hours = String::from(&s[8..10]);
      let minutes = String::from(&s[10..12]);
      let seconds = String::from(&s[12..14]);
      let micro = String::from(&s[14..]);
      Ok(DateTimeSquizzed { 
        year: year.parse::<i32>()?, 
        month: month.parse::<u32>()?,
        day: day.parse::<u32>()?,
        hours: hours.parse::<u32>()?,
        minutes: minutes.parse::<u32>()?,
        seconds: seconds.parse::<u32>()?,
        micro: if micro.len() == 0 {0} else { micro.parse::<u32>()? }
      })
  }
}

pub fn make_datetime(date: &str) -> Result<DateTime<Utc>> {
  let parsed = DateTimeSquizzed::from_str(date)?;
  Ok(Utc.ymd(parsed.year, parsed.month, parsed.day)
    .and_hms_micro(parsed.hours, parsed.minutes, parsed.seconds, parsed.micro))
}

pub fn get_instrument_list(path: &str) -> Result<Vec<String>> {
  let mut inst_set = HashSet::<String>::new();
  const BSIZE: usize = 1000000;
  let mut buffer = vec![0u8; BSIZE];
  let mut buf: Vec<u8> = Vec::with_capacity(10 * BSIZE);
  let mut file = fs::File::open(path)?;
  let mut decoder = Stream::new_stream_decoder(u64::MAX, 0).unwrap();
  let form =  r#""gts":\d+.*"i":"(.+/.+)","#;
  let re = Regex::new(&form).unwrap();
  let mut bcounter = 0;
  let mut offset: usize = 0;
  loop {
    match file.read(&mut buffer[offset..]) {
      Ok(r) => {
        bcounter += r;
        print!(".");
        io::stdout().flush()?;
      }
      Err(_) => { break; }
    }
    buf.clear();
    let res = decoder.process_vec(&buffer, &mut buf, Action::Run);
    offset = bcounter - decoder.total_in() as usize;
    if offset > 0 {
      buffer = (&buffer[BSIZE - offset..]).to_vec(); 
      buffer.extend(vec![0u8; BSIZE - offset]);
    } 
    match res {
      Ok(_) => {},
      Err(_) => { 
        continue; 
      },
    }
    let data = std::str::from_utf8(&buf).unwrap();
    let lines: Vec<&str> = data.lines().collect();
    for line in lines {
      for cap in re.captures_iter(line) {
        inst_set.insert(String::from(&cap[1]));
      }
    }
    if bcounter > 0 {
      break;
    }
  } 
  println!(".");
  let mut inst_vec : Vec<String> = inst_set.into_iter().collect();
  inst_vec.sort();
  Ok(inst_vec)
} 

pub fn get_instrument_records(path: &str, instrument: &str, tick_bounds : (i64, i64)) -> Result<Vec<String>> {
  const BSIZE: usize = 1000000;
  let mut buffer = vec![0u8; BSIZE];
  let mut buf: Vec<u8> = Vec::with_capacity(10 * BSIZE);
  let mut file = fs::File::open(path)?;
  let mut decoder = Stream::new_stream_decoder(u64::MAX, 0).unwrap();
  let form =  format!(r#""gts":(\d+).*"i":"{}""#, instrument);
  let re = Regex::new(&form).unwrap();
  let mut bcounter = 0;
  let mut offset: usize = 0;
  let mut bounded_records: Vec<String> = Vec::new();
  loop {
    match file.read(&mut buffer[offset..]) {
      Ok(r) => {
        bcounter += r;
        print!(".");
        io::stdout().flush()?;
      }
      Err(_) => { break; }
    }
    buf.clear();
    let res = decoder.process_vec(&buffer, &mut buf, Action::Run);
    offset = bcounter - decoder.total_in() as usize;
    if offset > 0 {
      buffer = (&buffer[BSIZE - offset..]).to_vec(); 
      buffer.extend(vec![0u8; BSIZE - offset]);
    } 
    match res {
      Ok(_) => {},
      Err(_) => { 
        continue; 
      },
    }
    let data = std::str::from_utf8(&buf).unwrap();
    let lines: Vec<&str> = data.lines().collect();
    let size = bounded_records.len();
    let end_token = String::from("},");
    let prev_record = if size > 0 { &bounded_records[size - 1] } else { &end_token };
    let last_symbs = String::from(&prev_record[prev_record.len() - 2..]);
    for line in lines {
      if let Some(start) = line.chars().nth(0) {
        if start != '{' && size > 0 && last_symbs != end_token {
          bounded_records[size - 1].push_str(line);
        continue;
      }
      }else {
        println!("bad line: {}", line);
      }
      for cap in re.captures_iter(line) {
        let ts = cap[1].parse::<i64>()?;
        if ts > tick_bounds.1 {
          return Ok(bounded_records);
        }
        if ts >= tick_bounds.0 {
          bounded_records.push(String::from(line));
        }
      }
    }
  } 
  println!(".");
  Ok(bounded_records)
} 

fn assign_label(feature: f64, ranges: &Vec<(f64, f64, i64)>) -> i64 {
  for val in ranges.iter() {
    if feature > val.0 && feature <= val.1 {
      return val.2;
    }
  }
  return -1;
}

pub fn convert_1d_tensor_to_vec<T>(tensor: &Tensor) -> Vec<T>
where T: Clone 
{
  let ptr = tensor.data_ptr() as *const T;
  let shape = tensor.size();
  unsafe {
    std::slice::from_raw_parts(ptr, shape[0] as usize).to_vec()
  }
}

pub fn convert_2d_tensor_to_vec<T>(tensor: &Tensor) -> Vec<Vec<T>>
where T: Clone 
{
  let ptr = tensor.data_ptr() as *const T;
  let shape = tensor.size();
  let mut tvec: Vec<Vec<T>> = Vec::new();
  for i in 0..shape[0] {
    unsafe {
      let vec = std::slice::from_raw_parts(ptr.add((i * shape[1]) as usize), shape[1] as usize).to_vec();
      tvec.push(vec);
    }
  }
  tvec
}

pub fn convert_features_to_tensor(features: Vec<Vec<f64>>)-> TensorOwner<f32> {
  if features.len() == 0 {
    return TensorOwner::<f32>{ tensor: Tensor::of_slice::<f64>(&[]), _marker: PhantomData }
  }
  let len1 = features.len() as i64;
  let len2 = features[0].len() as i64;
  TensorOwner::<f32> {
    tensor: Tensor::of_slice(&features.into_iter().flatten().collect::<Vec<f64>>()[..])
    .view((len1, len2))
    .to_kind(Kind::Float),
    _marker: PhantomData,
  }
}

pub fn convert_labels_to_tensor(labels: Vec<i64>)-> TensorOwner<i64> {
  TensorOwner::<i64> { tensor: Tensor::of_slice(&labels).to_kind(Kind::Int64), _marker: PhantomData }
}

fn get_quantile(
  q: f64, 
  data: &Vec<f64>
) -> f64 {
  let offset = if data.len() % 2 == 0 { 1 } else { 0 }; 
  data[(data.len() as f64 * q).floor() as usize - offset]
}

// Splits jumps into ranges 
pub fn create_jump_range_labels(
  tcontext: &TradingContext,
  quantile_ratios: Vec<f64>,
  mut quantile_margins: Vec<f64>,
) -> Vec<(f64, f64, i64)> {
  let mut jumps: Vec<f64> = tcontext.bid_ask_volume_map.iter()
    .filter(|val| val.1.rel_price != 0.0)
    .map(| val | val.1.rel_price ).collect();
  jumps.sort_by(|a,b| a.partial_cmp(&b).unwrap());
  let mut quantiles: Vec<f64> = quantile_ratios.into_iter().map(|q| get_quantile(q, &jumps)).collect();
  let mut margins: Vec<f64> = Vec::new();
  assert_eq!(quantile_margins.len(), quantiles.len());
  if quantiles.len() == 1 {
    quantiles = vec![jumps[0], quantiles[0], jumps[jumps.len() -1]];
    quantile_margins.push(quantile_margins[0]);
  }

  for i in 0..quantiles.len() - 1 {
    margins.push(quantile_margins[i] * (quantiles[i + 1] - quantiles[i]));
  }

  let margin = margins.iter().min_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap() / 2.0;
  let mut counter = 0;
  let mut jump_ranges: Vec<(f64, f64, i64)> = Vec::with_capacity(quantiles.len() + 2);
  jump_ranges.push((f64::MIN, quantiles[0] - margin, 0));
  quantiles.iter().for_each(|val| { counter += 1; jump_ranges.push((val - margin, val + margin, counter)); });
  jump_ranges.push((quantiles[quantiles.len() - 1] + margin, f64::MAX, counter + 1));
  jump_ranges
}

pub fn make_training_and_test_features(
  feature_vec: &mut Vec<(Vec<f64>, i64)>, 
  rate: f64) -> super::FeatureSet {
  feature_vec.shuffle(&mut thread_rng());
  let train_len = (feature_vec.len() as f64 * rate).round() as usize; 
  let train_features  = convert_features_to_tensor(
    (&feature_vec[0 .. train_len]).iter().map(|val| val.clone().0).collect());
  let train_labels = convert_labels_to_tensor(Vec::from(&feature_vec[0 .. train_len]).iter().map(|val| val.1).collect());
  let test_features  = convert_features_to_tensor(Vec::from(&feature_vec[train_len..]).iter().map(|val| val.clone().0).collect());
  let test_labels = convert_labels_to_tensor(Vec::from(&feature_vec[train_len..]).iter().map(|val| val.1).collect());
  super::FeatureSet {
    train_features,
    train_labels,
    test_features,
    test_labels
  }
}

pub fn scale_vec(data: &mut Vec<(Vec<f64>, i64)>) {
  let all_features: Vec<f64> = data.iter().map(|val| val.0.clone()).flatten().collect();
  let mean: f64 = all_features.iter().sum::<f64>() / all_features.len() as f64;
  let div = all_features.iter().map(|v| (v - mean) * (v - mean) ).sum::<f64>().sqrt() / (all_features.len() as f64 - 1.0);
  for x in data.iter_mut() {
    for y in x.0.iter_mut() {
      *y = (*y - mean)/div;
    } 
  }
} 

pub fn merge_features(
  it1: Iter<(f64, f64)>, 
  it2: Iter<(f64, f64)>, 
  price: f64, 
  price_direction: f64, 
  impact_degree: f64) -> Vec<(f64, f64)> {
  it1.rev().map(|val| (val.0 - price, price_direction * val.1.powf(impact_degree))).chain(
    it2.map(|val| (val.0 - price, -price_direction * val.1.powf(impact_degree)))).collect()
}

#[derive(Debug, EnumIter)]
pub enum FeatureType {
  AskVolumeCurrent = 0,
  BidVolumeCurrent,
  AskVolumePrev,
  BidVolumePrev,
  SellTrade,
  BuyTrade,
}

pub fn populate_nn_feature_vec(
  feature_type: FeatureType,
  feature: &[(f64, f64)],
  nn_feature_vec: &Vec<f64>,
  price_tick_size: f64,
  best_prices: (f64, f64),
  max_ob_level: i64
) {
  let feature_offset = feature_type as i64 * max_ob_level;
  println!("extracting features");
}

pub fn extract_price_levels(
  ranges: &Vec<(f64, f64, i64)>, 
  features: (&FeatureVolume, &FeatureVolume, &FeatureTrade),
  max_ob_level: i64,
  price_tick_size: f64,
) -> Option<(Vec<f64>, i64)> {
  let label = assign_label(features.1.rel_price, ranges);
  if label == -1 {
    return None;
  }
  let pb_ask = features.1.ask_volume[0].0;
  let pb_bid = features.1.bid_volume[0].0;
  let feature_merged = vec![0.0; 3 * 2 * max_ob_level as usize];
  FeatureType::iter().
  for_each(|ftype| { 
    let sub_features : &[(f64, f64)] = 
    match ftype {
      FeatureType::BidVolumePrev => &features.0.bid_volume[0..max_ob_level as usize],
      FeatureType::AskVolumePrev => &features.0.ask_volume[0..max_ob_level as usize],
      FeatureType::BidVolumeCurrent => &features.1.bid_volume[0..max_ob_level as usize],
      FeatureType::AskVolumeCurrent => &features.1.ask_volume[0..max_ob_level as usize],
      FeatureType::SellTrade => &features.2.sell,
      FeatureType::BuyTrade => &features.2.buy
    };
    populate_nn_feature_vec(ftype, sub_features, &feature_merged, price_tick_size, (pb_bid, pb_ask), max_ob_level);
  });
  Some((feature_merged, label))
}

pub fn make_nn_features(
  ranges: &Vec<(f64, f64, i64)>, 
  tcontext: &TradingContext, 
  max_ob_level: i64,
  price_tick_size: f64
) -> Option<Vec<(Vec<f64>, i64)>> {
  let mut features: Vec<(Vec<f64>, i64)> = Vec::new(); 
  for ((t1, t2), trades) in tcontext.books_and_trades.iter() {
    let (s1, s2, trades) = (tcontext.bid_ask_volume_map.get(&t1), tcontext.bid_ask_volume_map.get(&t2), trades);
    if let Some(feature) = extract_price_levels(ranges, (s1.unwrap(), s2.unwrap(), &trades), max_ob_level, price_tick_size) {
      features.push(feature);
    }
  }
  Some(features)
}

#[derive(Serialize, Deserialize, Debug)]
struct Book {
  r#as: Vec<(f64, f64)>,
  bs: Vec<(f64, f64)>,  
}

#[derive(Serialize, Deserialize, Debug)]
struct Bbp {
  bbp: f64,
  bbs: f64,
  bap: f64,
  bas: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Trade {
  d: String,
  p: f64,
  s: f64, 
}

#[derive(Serialize, Deserialize, Debug)]
struct Record {
  t: String,
  gts: i64,
  e: i64,
  i: String,
  #[serde(flatten)]
  extra: HashMap<String, Value>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Obss { 
  ask : Vec<(f64, f64)>,
  bid : Vec<(f64, f64)> 
}

#[derive(Serialize, Debug, Deserialize)]
struct PriceJump {
  obss: Obss,
  obssdelay: i64,
  tsdiff: i64,
  jump: f64,
}

fn smooth_bbp_map(
  bbp_ts_map: &BTreeMap<i64, Bbp>, 
  window: i64, 
  weight_func : fn(f64, f64) -> f64
) -> Box<HashMap<i64, f64>> {
  let mut bbp_smoothed = Box::new(HashMap::<i64, f64>::new());
  for ts in bbp_ts_map.keys() {
    let range_iter = bbp_ts_map.range((Excluded(*ts), Included(*ts + window)));
    let ilengh = range_iter.clone().count();
    if ilengh == 0 {
      continue;
    }
    let last_ts = range_iter.clone().last().unwrap();
    let smoothed: f64 = range_iter.map(|val| weight_func((val.1.bbp + val.1.bap)/ 2.0, 1.0/ilengh as f64)).sum();
    bbp_smoothed.insert(*last_ts.0, smoothed);
  }
  bbp_smoothed
}

fn find_nearest_tick(
  tick: i64, 
  tcontext: &TradingContext, 
  offset_error: i64
) -> Result<&Bbp> {
  if let Some(ts1) = (&tcontext.bbp_ts_map).range((Included(tick), 
    Included(tick + offset_error))).map(|val| val.0).min() {
      return tcontext.bbp_ts_map.get(ts1).ok_or_else(|| anyhow!("find_nearest_tick: tick not found"));
    }
  Err(anyhow!("find_nearest_tick: tick not found"))
}

pub fn collect_jumps(
  tcontext: &TradingContext, 
  window: i64, 
  jump_horizon: i64,
  use_smoothing: bool,
) -> Result<Vec::<(i64, f64)>> {
  const MIN_OFFSET_ERROR: i64 = 100000;
  let smoother = |val1: f64, val2: f64| { val1 * val2 };
  let mut jump_collection = Vec::<(i64, f64)>::new();
  if use_smoothing {
    let smoothedvals = smooth_bbp_map(&tcontext.bbp_ts_map, window, smoother);
    for bbp in smoothedvals.iter() {
      if let Some(bbp_future) = find_nearest_tick(
        bbp.0 + jump_horizon, 
        tcontext, 
        MIN_OFFSET_ERROR
      ).ok() {
      let jump = 10000.0 * ((bbp_future.bap + bbp_future.bbp) / 2.0 - bbp.1) / bbp.1;
        jump_collection.push((bbp.0 + jump_horizon, jump));
      }
    }
  } else {
    for bbp in tcontext.bbp_ts_map.iter() {
      if let Some(bbp_future) = find_nearest_tick(
        bbp.0 + jump_horizon, 
        tcontext, 
        MIN_OFFSET_ERROR
      ).ok() {
      let current =  (bbp.1.bap + bbp.1.bbp) / 2.0; 
      let jump = 10000.0 * ((bbp_future.bap + bbp_future.bbp) / 2.0 - current) / current;
        jump_collection.push((bbp.0 + jump_horizon, jump));
      }
    }
  }
 
  Ok(jump_collection)
}

pub fn populate_ob_maps( 
  tcontext: &mut TradingContext,
  records: &Vec<String>,
) {
  let (start_time, end_time) = tcontext.trading_time;
  println!("time bounds: {}, {}", start_time, end_time);
  for rec in records {
    let res : serde_json::Result<Record> = serde_json::from_str(&rec[0..rec.len() - 1]);
    match res {
      Ok(v) => {
        if !v.i.eq(&tcontext.instrument) {
          continue;
        }
        if (v.gts as i64) < start_time {
          println!("dropped time : {} < {}", v.gts, start_time);
          continue;
        }
        if v.gts > end_time {
          println!("dropped time : {} > {}", v.gts, end_time);
          break;
        }
        match v.t.as_str() {
          "obss" => { 
            tcontext.book_ts_map.insert(
              serde_json::from_value(v.extra.get("ets").unwrap().to_owned()).unwrap(), 
              serde_json::from_value(v.extra.get("b").unwrap().to_owned()).unwrap());
          }
          "tobu" => {
            tcontext.bbp_ts_map.insert(v.gts, serde_json::from_value(v.extra.get("b").unwrap().to_owned()).unwrap());
          } 
          "at" => {
            tcontext.trade_ts_map.insert(v.gts, serde_json::from_value(v.extra.get("b").unwrap().to_owned()).unwrap());
          }
          "r24hpsu" => {
            continue;
          }
        _ => { 
          println!("not implemented {:?}", v);
          }
        }
      }
      Err(_) => {
        println!("bad record: {:?}, error: {:?}", &rec[0..rec.len() - 1], res);
        continue;
      }   
    }
  }
}

pub fn collect_obss_features(
  tcontext: &mut TradingContext
) {
  const TICK_OFFSET: i64 = 100000;
  const OFFSET_ERROR: i64 = 100000;
  let mut iter = tcontext.book_ts_map.iter().zip(tcontext.book_ts_map.iter().skip(1));
  while let Some((snap1, snap2)) = iter.next() {
    let tdiff = snap2.0 - snap1.0;
    if (tdiff as i64 - TICK_OFFSET).abs() < OFFSET_ERROR {
      if let Some(ts2) = (&tcontext.bbp_ts_map).range((Included(snap1.0 + tcontext.jump_horizon), 
        Included(snap1.0 + tcontext.jump_horizon + OFFSET_ERROR))).map(|val| val.0).min() {
        let bbp_future = tcontext.bbp_ts_map.get(ts2).unwrap();
        let mut fvol = FeatureVolume::new();
        fvol.ask_volume =  snap1.1.r#as.clone();
        fvol.bid_volume =  snap1.1.bs.clone();
        fvol.current_price = (snap1.1.r#as[0].0 + snap1.1.bs[0].0) /2.0;
        fvol.rel_price = ((bbp_future.bap + bbp_future.bbp) /2.0 - fvol.current_price) / fvol.current_price; 
        fvol.ts = (*snap1.0, *ts2);
        tcontext.bid_ask_volume_map.insert(fvol.ts.0, fvol);
        let (ts1, ts2) = (*snap1.0, *snap2.0);
        let mut ftrade = FeatureTrade::new();
        ftrade.ts = (ts1, ts2);
        (tcontext.trade_ts_map).range((Included(ts1),
          Included(ts2))).for_each(|(_, trade)| {
          if trade.d == "b" {
            ftrade.buy.push((trade.p, trade.s));
          } else if trade.d == "s" {
            ftrade.buy.push((trade.p, trade.s));
          }
        }); 
        tcontext.books_and_trades.insert((ts1, ts2), ftrade); 
      } 
    }
  }
}

pub fn save_trades(
  tcontext: &TradingContext
) {
  let mut f = fs::File::create("trades.json").unwrap();
  let export_trades_str: String = serde_json::to_string_pretty(&tcontext.trade_ts_map).expect("bad config format");
  write!(f, "{:?}", &export_trades_str).unwrap();
}

pub fn get_trading_context(
  datetimes: &Vec<DateTime<Utc>>, 
  path_to_dump: &str, 
  instrument: &str, 
  jump_horizon : i64, 
  trade_history_length: i64
) -> Result<TradingContext> {
  let start_tick = datetimes[0].timestamp_nanos() / 1000;
  let end_tick = datetimes[1].timestamp_nanos() / 1000;
  let records = get_instrument_records(&path_to_dump, instrument, (start_tick, end_tick))?;
  let mut tcontext = TradingContext::new(
  String::from(instrument), 
    (start_tick, end_tick), 
    jump_horizon, 
    trade_history_length,
  );
  populate_ob_maps(&mut tcontext, &records);
  collect_obss_features(&mut tcontext);
  Ok(tcontext)
}

#[derive(Serialize)]
pub struct TradingContext {
  instrument: String,
  book_ts_map: BTreeMap::<i64, Book>, 
  bbp_ts_map: BTreeMap::<i64, Bbp>, 
  trade_ts_map: BTreeMap::<i64, Trade>,
  pub books_and_trades: HashMap<(i64, i64), FeatureTrade>,
  pub bid_ask_volume_map: HashMap<i64, FeatureVolume>,
  trade_history_length: i64,
  pub trading_time: (i64, i64),
  jump_horizon: i64, 
}

impl TradingContext {
  pub fn new(
    instrument: String, 
    trading_time: (i64, i64), 
    jump_horizon: i64,
    trade_history_length: i64, 
  ) -> Self {
    Self {
      instrument: instrument,
      book_ts_map: BTreeMap::new(),
      bbp_ts_map: BTreeMap::new(),
      trade_ts_map: BTreeMap::new(),
      books_and_trades: HashMap::new(),
      bid_ask_volume_map: HashMap::new(),
      trading_time,
      trade_history_length,
      jump_horizon,
    }
  }
  pub fn save(&self, path: &str) -> Result<()> {
    let export_context = serde_json::to_string_pretty(&self)?;
    let mut f = fs::File::create(format!("{}.json", path))?;
    write!(f, "{}", &export_context)?;
    Ok(())
  }
}

#[derive(Clone, Serialize)]
pub struct FeatureVolume {
  pub instrument: super::Instrument,
  pub ts: (i64, i64),
  pub ask_volume: Vec<(f64, f64)>,
  pub bid_volume: Vec<(f64, f64)>,
  pub rel_price: f64,
  pub current_price: f64,
}

#[derive(Clone, Serialize)]
pub struct FeatureTrade {
  ts: (i64, i64),
  sell: Vec<(f64, f64)>,
  buy: Vec<(f64, f64)>,
}

impl FeatureTrade {
  pub fn new() -> Self {
    FeatureTrade {
      ts: (-1,-1),
      sell: Vec::new(),
      buy: Vec::new(),
    }
  }
}

impl FeatureVolume {
  pub fn new() -> Self {
    FeatureVolume {
      instrument: super::Instrument::BTC,
      ts: (-1,-1),
      ask_volume: Vec::new(),
      bid_volume: Vec::new(),
      rel_price: -1.0,
      current_price: -1.0,
    }
  }
}