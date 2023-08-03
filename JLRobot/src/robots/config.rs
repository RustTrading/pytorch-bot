use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Visitor, de::SeqAccess};
use chrono::{Utc, DateTime, NaiveDateTime};
use std::fmt;

#[derive(Serialize, Deserialize, Debug)]
pub enum TimeUnits {
  NanoSeconds,
  Microseconds,
  Milliseconds,
  Seconds,
  Minutes,
  Hours,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TimeValue {
  pub value: i64,
  pub units: TimeUnits,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InstrumentValue {
  pub value: i64,
  pub units: InstrumentUnits,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum InstrumentUnits {
  BNB,
  BUSD,
  USDT,
}

#[derive(Deserialize, Debug)]
pub struct Config {
  pub instrument: String,
  state: String, 
  #[serde(deserialize_with = "from_datetime")]
  pub training_time: Vec<Vec<DateTime<Utc>>>,
  #[serde(deserialize_with = "from_datetime")]
  pub trading_time: Vec<Vec<DateTime<Utc>>>,
  pub price_tick_size: f64,
  pub prediction_horizon: TimeValue,
  pub trade_history_length: TimeValue,
  pub training_rate: f64,
  pub epochs: i64,
  pub pnl_report_file: String,
  pub balance: Vec<InstrumentValue>,
  pub dump_path: String,
  export_model_file: String,
  import_model_file: String,
  pub max_ob_level: i64,
  pub quantiles: Vec<f64>,
  pub quantile_relative_margins: Vec<f64>,
}

struct DateTimeVecDeserializer;
struct DateTimeDeserializer;

#[derive(Debug)]
pub struct DateTimeOwner {
  pub dt: DateTime<Utc>,
}

#[derive(Debug)]
pub struct DateTimeVecOwner {
  dt_vec: Vec<Vec<DateTimeOwner>>,
}

impl Serialize for DateTimeOwner {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
      S: Serializer,
  {
    let s = self.dt.to_rfc3339();
    serializer.serialize_str(&s)
  }
}

impl DateTimeOwner {
  fn new(dt: DateTime<Utc>) -> Self {
    Self {
      dt
    }
  }
}

impl DateTimeVecOwner {
  fn new(dt_vec: Vec<Vec<DateTimeOwner>>) -> Self {
    Self {
      dt_vec
    }
  }
}

impl<'de> Deserialize<'de> for DateTimeOwner {
  fn deserialize<D>(deserializer: D) -> Result<DateTimeOwner, D::Error>
    where D: Deserializer<'de>,
    {
      // Deserialize from a human-readable string like "2015-05-15T17:01:00Z".
      let s = String::deserialize(deserializer)?;
      let dt_offset = DateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S %z").map_err(serde::de::Error::custom)?;
      Ok(DateTimeOwner::new(dt_offset.with_timezone(&Utc)))
    }   
}

impl<'de> Deserialize<'de> for DateTimeVecOwner {
  fn deserialize<D>(deserializer: D) -> Result<DateTimeVecOwner, D::Error>
    where D: Deserializer<'de>,
    {
      // Deserialize from a human-readable string like "2015-05-15T17:01:00Z".
      let val = deserializer.deserialize_seq(DateTimeVecDeserializer)?;
      Ok(DateTimeVecOwner::new(val))
    }   
}

impl<'de> Visitor<'de> for DateTimeDeserializer {
  type Value = Vec<DateTimeOwner>;

  fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    formatter.write_str("VecDeserializer sequence.")
  }

  fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
  where
    A: SeqAccess<'de> {
    let mut new_obj = Vec::new();
    while let Some(value) = seq.next_element()? {
      new_obj.push(value);
    } 
    Ok(new_obj)
  }
}

impl<'de> Visitor<'de> for DateTimeVecDeserializer {
  type Value = Vec<Vec<DateTimeOwner>>;

  fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    formatter.write_str("VecDeserializer sequence.")
  }

  fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
  where
    A: SeqAccess<'de> {
    let mut new_obj = Vec::new();
    while let Some(value) = seq.next_element()? {
      new_obj.push(value);
    } 
    Ok(new_obj)
  }
}

fn from_datetime<'de, D>(deserializer: D) -> Result<Vec<Vec<DateTime<Utc>>>, D::Error>
where
    D: Deserializer<'de>,
{
  Ok(
    deserializer.deserialize_seq(
      DateTimeVecDeserializer)?.iter().map(|val| {val.iter().map(|val2| {val2.dt}).collect()}).collect()
    )
}