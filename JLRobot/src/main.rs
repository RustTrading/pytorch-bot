use clap::{Arg, App};
use anyhow::{Result, anyhow};
mod robots;
mod jhistogram;
use robots::{data_collection::*, config::Config, config::TimeUnits};
use std::fs;

fn main() -> Result<()> {
  let jconfig = fs::read_to_string("config.robots.json")?;
  let matches = App::new("robot_manager")
  .version("0.0.1")
  .about("trains and run robots: --train | load")
  .arg(Arg::new("train")
  .long("train")
  .required(false)
  .takes_value(false)
  .help("Trains robot"))
  .arg(Arg::new("load")
  .long("load")
  .required(false)
  .takes_value(false)
  .help("Loads robot"))
  .arg(Arg::new("save")
  .long("save")
  .required(false)
  .takes_value(false)
  .help("Saves training samples"))
  .arg(Arg::new("visualize")
  .long("visualize")
  .required(false)
  .takes_value(false)
  .help("Builds histogram and quantile ranges"))
  .get_matches();

  let res: Config= serde_json::from_str(&jconfig)?;
  let path_to_xz =  res.dump_path;
  let instrument =  res.instrument;
  let price_tick_size = res.price_tick_size;
  let trate = res.training_rate;
  let epochs = res.epochs;
  let quantile_ratios = res.quantiles;
  let quantile_margins = res.quantile_relative_margins;
  let balance = res.balance;
  let max_ob_level= res.max_ob_level;
  let trade_history_length: i64 = convert_time(res.trade_history_length, TimeUnits::Microseconds).value;
  let jump_horizon: i64 = convert_time(res.prediction_horizon, TimeUnits::Microseconds).value;
  println!("instrument: {}, path to instrument {}", instrument,  path_to_xz);
  println!("training time time {:?}, trainig rate: {:?}", res.training_time, trate);
  println!("quantile ratios: {:?}", quantile_ratios);
  let robot_config = robots::RobotConfig {
    robot_type: robots::RobotType::SimpleLogistic,
    balance,
    training_rate: trate,
    max_ob_level,
    quantile_ratios,
    quantile_margins,
    jump_horizon,
    trade_history_length,
  };
  println!("trade_history_length: {:?}, prediction_horizon: {:?}", trade_history_length,  jump_horizon);   
  let mut robot = robots::Robot::new(robot_config);
  let smoothing_window = 100000;
  if matches.is_present("load") {
    println!("initialization pretrained robot");
    println!("checking model accuracy for trading time {:?}", &res.trading_time);
    let file_id = format!("model-{}-trained", robot.robot_config.robot_type);  
    robot.load_model(&file_id)?;
    println!("model read{:?}", robot.model.m);
    for ttime in res.trading_time {
      println!("trading time: {:?}", &ttime);
      let tcontext = get_trading_context(
        &ttime, 
        &path_to_xz, 
        &instrument, 
        jump_horizon, 
        trade_history_length
      )?;
      if matches.is_present("save") {
        let fname = format!("trading-{:?}", &ttime);
        println!("saving to {}", &fname);
        tcontext.save(&fname)?;
      }  
      println!("prediction accuracy {:?} ", robot.test_accuracy(&tcontext, price_tick_size));
      let prev = ttime[0].timestamp_nanos()/ 1000;
      let current = prev + 100_000;
      let orders = robot.create_orders((&tcontext.bid_ask_volume_map.get(&prev).unwrap(), &tcontext.bid_ask_volume_map.get(&current).unwrap(), &tcontext.books_and_trades.get(&(prev, current)).unwrap()), price_tick_size);
      println!("suggested orders: {:?}", orders);
      if  matches.is_present("visualize") {
        let time_spread = tcontext.trading_time.1 - tcontext.trading_time.0;  
        jhistogram::dataview::draw_jump_histogram(&tcontext, &path_to_xz, smoothing_window, jump_horizon, &instrument, time_spread);
      }
    }
  } else if  matches.is_present("train") { 
    let tcontext = get_trading_context(
      &res.training_time[0], 
      &path_to_xz, 
      &instrument, 
      jump_horizon, 
      trade_history_length
    )?;
    println!("max_ob_level: {:?}", max_ob_level);     
    let trained_set_sizes = robot.train_and_test(epochs, &tcontext, price_tick_size, true).ok_or(anyhow!("failed training"))?;
    robot.export_model(&format!("model-{}-trained", robot.robot_config.robot_type))?;
    println!("created model: {:?}", robot.model.m);
    println!("training size: {:?}, testing_size: {:?}", trained_set_sizes.0, trained_set_sizes.1);
    if  matches.is_present("visualize") {
      let time_spread = tcontext.trading_time.1 - tcontext.trading_time.0;  
      jhistogram::dataview::draw_jump_histogram(&tcontext, &path_to_xz, smoothing_window, jump_horizon, &instrument, time_spread);
    }
  } else {
    println!("load/train not specified, see --help");
  }
  Ok(())
}
