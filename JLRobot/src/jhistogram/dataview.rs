use plotters::prelude::*;
use plotters::coord::{Shift};
use std::cmp::{min_by, max_by};
use anyhow::Result;

use crate::robots::data_collection::{TradingContext, collect_jumps};

pub fn draw_jump_histogram(
  tcontext: &TradingContext, 
  file_name: &str, 
  smoothing_window: i64, 
  jump_horizon: i64, 
  instrument: &str,
  time_spread: i64) {
  let jump_collection = collect_jumps(&tcontext, smoothing_window, jump_horizon, false).unwrap();
  let mut jump_collection_vec: Vec<Vec<(i64, f64)>> = Vec::new();
  jump_collection_vec.push(jump_collection);
  let filename = file_name.split('/').rev().next().unwrap();
  create_gif_histogram(filename, time_spread, jump_horizon, &instrument, &jump_collection_vec).unwrap(); 
}

pub fn create_gif_histogram(
  filename: &str, 
  time_spread: i64, 
  jump_horizon: i64, 
  instrument: &str, 
  jump_collection_vec: &Vec<Vec<(i64, f64)>>) -> Result<()>{
  let output = format!("price-jump-{0}-second-{1}-{2}.gif", (jump_horizon as f64) /1e6, filename, instrument).replace('/', "-");  
  let root = BitMapBackend::gif(output.clone(), (1024, 768), 1000)?.into_drawing_area();
  let (minjump, maxjump, maxlength) = jump_collection_vec.iter().
  fold((f64::MAX, f64::MIN, 0), |prev, current| (
    min_by(prev.0, current.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().1, |a, b| a.partial_cmp(&b).unwrap()),
    max_by(prev.1, current.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().1, |a, b| a.partial_cmp(&b).unwrap()),
    usize::max(prev.2, current.len())));
  let bin_number =  (maxlength as f64).sqrt().round();
  let ymax = bin_number * 20.0;
  for frame_idx in 0..jump_collection_vec.len() {
    let histo_jumps = super::Histogram::new(
      (jump_collection_vec[frame_idx].len() as f64).sqrt().round() as i64, 
      time_spread,
      &jump_collection_vec[frame_idx], 
      &output); 
    draw_smoothed_histogram(&root, histo_jumps, (minjump, maxjump), (0.0, ymax), bin_number)?;
    root.present()?
  }
  Ok(())
}
#[cfg(test)]
pub fn draw_series(series_vect: &Vec<Vec<f64>>, 
  timestep : f64, 
  title : &str, 
  items_description : &Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
  
  let mut owned = title.to_owned();
  owned.push_str(".png");
  let root = BitMapBackend::new(&owned, (1024, 768)).into_drawing_area();
  root.fill(&WHITE)?;
  
  let minval = series_vect[0].iter().cloned().fold(0f64, f64::min);
  let maxval = series_vect[0].iter().cloned().fold(0f64, f64::max);

  let upper_bound = maxval + 2.0 * f64::abs(maxval - minval);
  let lower_bound = maxval - 2.0 * f64::abs(maxval - minval);
  let mut chart = ChartBuilder::on(&root)
  .margin(5)
  .caption(String::from(title), ("sans-serif", 30))
  .set_label_area_size(LabelAreaPosition::Left, 60)
  .set_label_area_size(LabelAreaPosition::Bottom, 60)
  .set_label_area_size(LabelAreaPosition::Right, 60)
  .build_cartesian_2d(0f64..(series_vect[0].len() as f64)  * timestep, lower_bound..upper_bound)?;

  chart.configure_mesh().draw()?; 
  let mut counter = 0;
  for s in series_vect.iter().enumerate() {   
    let color = Palette99::pick(counter);
    let s_item = LineSeries::new(
      (0..s.1.len()).map(|x| x as f64 * timestep).map(|x| {
          (
            x, s.1[(x/timestep) as usize]
          )
      }),
      &color
     );
    chart.draw_series(s_item)?
     .label(&items_description[s.0])
     .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)],  Palette99::pick(counter).filled()));
     counter += 1;
  }
  chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

   Ok(())
}

#[cfg(test)]
pub fn draw_mixture(means : Vec<f64>, 
  stdevs : Vec<f64>, weights : Vec<f64>, 
  sample_size : usize, random_points : &Vec<f64>, data_to_compare : &Vec<Vec<(f64, f64)>>) -> Result<(), Box<dyn std::error::Error>> {
  
  let frm : String = String::from("mixnormal-dist.png");
  let root = BitMapBackend::new(&frm, (1024, 768)).into_drawing_area();
  root.fill(&WHITE)?;

  let mut chart = ChartBuilder::on(&root)
  .margin(5)
  .caption("1D Gaussian Mixture Distribution", ("sans-serif", 30))
  .set_label_area_size(LabelAreaPosition::Left, 60)
  .set_label_area_size(LabelAreaPosition::Bottom, 60)
  .set_label_area_size(LabelAreaPosition::Right, 60)
  .build_cartesian_2d(-5f64..5f64, -(sample_size as f64)/2.0 .. (sample_size as f64)/2.0)?
  .set_secondary_coord(
    (-5f64..5f64).step(0.1).use_round().into_segmented(),
    0u32..(sample_size/ 4) as u32,);

  chart.configure_mesh().draw()?;
  
  let actual = Histogram::vertical(chart.borrow_secondary())
  .style(GREEN.filled())
  .margin(3)
  .data(random_points.iter().map(|x| (*x, 1)));
  
  chart.configure_secondary_axes().y_desc("Count").draw()?;
  chart.draw_secondary_series(actual)?
  .label("Observed")
  .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], GREEN.filled()));
  
  let mut counter = 0;
  for data in data_to_compare {
    let ls = LineSeries::new(data.clone(), &BLUE);
    chart.draw_series(ls)?
    .label(format!("Smoothed Price Contributions {}", counter))
    .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], BLUE.filled()));
    counter+=1;  
  }
  let pdf = LineSeries::new(
    (-500..500).map(|x| x as f64 / 100.0).map(|x| {
        (
          x,
          (0..weights.len()).map(|idx| { weights[idx] * (- (x - means[idx]) * (x - means[idx]) / 2.0 / stdevs[idx] / stdevs[idx]).exp() 
          / (2.0 * std::f64::consts::PI).sqrt()/stdevs[idx] }).sum::<f64>() * (sample_size as f64) * x/10.0
        )
    }),
    &RED,
   );
  
  chart
   .draw_series(pdf)?
   .label("Ideal Mixture Price Contribution")
   .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
  
   chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

  Ok(())
}

pub fn estimate_bandwidth (series: Vec<f64>) -> f64 {
  // Scott’s rule of thumb 
  let size = series.len() as f64;
  let mean = series.iter().sum::<f64>()/ size;
  let sigma = (series.into_iter().reduce(|a, b| a + (b - mean) * (b - mean)).unwrap() / (size - 1.0) as f64).sqrt();

  1.06 * sigma * size.powf(-0.2)
}

pub fn draw_smoothed_histogram (
  root: &DrawingArea<BitMapBackend, Shift>,
  histo: super::Histogram,
  x_range_bounds: (f64, f64),
  y_range_bounds: (f64, f64),
  bin_number: f64 
) -> Result<()> {
  root.fill(&WHITE)?;
  let data_points :Vec<f64> = histo.raw_ticks.iter().map(|val| val.1).collect();
  let points = vec![data_points.clone()];
  let bandwidth = estimate_bandwidth(data_points.clone());
  let smoothed_by_gaussian = super::gaussian_kernel_smoother_many(&points, bandwidth);
  let smoothed_by_epanchikov = super::epanechnikov_kernel_smoother_many(&points, bandwidth);
  let maxjump = smoothed_by_gaussian[0].iter()
  .map(|val| val.0.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() / 10.0;
  let tickstep = (x_range_bounds.1 - x_range_bounds.0) / bin_number as f64;
  let mut chart = ChartBuilder::on(&root)
  .margin(5u32)
  .caption(&histo.description, ("sans-serif", 30u32))
  .set_label_area_size(LabelAreaPosition::Left, 60u32)
  .set_label_area_size(LabelAreaPosition::Bottom, 60u32)
  .set_label_area_size(LabelAreaPosition::Right, 60u32)
  .build_cartesian_2d((x_range_bounds.0..x_range_bounds.1).step(tickstep), 
    y_range_bounds.0..y_range_bounds.1)?;
  
  chart.configure_mesh().draw()?;  

  let binnumber = histo.bin_number;
  chart.draw_series(histo)?
  .label(format!("observed, binnumber = {}", binnumber))
  .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], GREEN.filled())); 
 
  let price_curve : Vec::<(f64, f64)> = smoothed_by_gaussian[0].iter()
  .map(|val| (val.0, val.0.abs() * val.1 / maxjump)).collect();
  
  let s1 = LineSeries::new(smoothed_by_gaussian[0].clone(), &BLUE);
  chart.draw_series(s1)?
  .label(format!("gaussian kernel smoothed with bandwidth = {}", bandwidth))
  .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], BLUE.filled()));
  
  let s2 = LineSeries::new(smoothed_by_epanchikov[0].clone(), &RED);
  chart.draw_series(s2)?
  .label(format!("epanechnikov kernel smoothed with bandwidth = {}", bandwidth))
  .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], RED.filled()));

  let s3 = LineSeries::new(price_curve, &MAGENTA);
  chart.draw_series(s3)?
  .label(format!("price curve for gaussian smoothed with bandwidth = {}", bandwidth))
  .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], MAGENTA.filled()));

  chart.configure_series_labels()
  .background_style(&WHITE.mix(0.8))
  .border_style(&BLACK).draw()?;
  
  Ok(())
}
#[cfg(test)]
pub fn draw_histogram(histo: &super::Histogram) -> Result<(), Box<dyn std::error::Error>> {
  let form = format!("histo-{}.png", histo.description);
  let root = BitMapBackend::new(&form, (640, 480)).into_drawing_area();
  root.fill(&WHITE)?;
  let mut chart = ChartBuilder::on(&root)
    .x_label_area_size(35u32)
    .y_label_area_size(40u32)
    .margin(5u32)
    .caption(&histo.description, ("sans-serif", 50u32))
    .build_cartesian_2d(
      (-100f64..100f64).step(0.5).use_round().into_segmented(),
      0u32..10 * (histo.raw_ticks.len() as f64).sqrt() as u32)?;

  chart.configure_mesh().draw()?;    
  chart.draw_series(
    plotters::prelude::Histogram::vertical(&chart)
      .margin(1)
      .style(RED.mix(0.5).filled())
      .data(histo.raw_ticks.iter().map(|x| (((*x).1), 1))),
  )?;

  Ok(())
}
