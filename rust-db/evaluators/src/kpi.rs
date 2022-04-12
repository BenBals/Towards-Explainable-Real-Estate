//! This module contains an evaluator and output type that caputres a large number of error metrics.
//! It should be your default choice of evaluator.
//! It additionally contains a number of wrappers around both the evaluator and output that enable
//! ordering the output by different single metrics.
use std::fmt::Display;

use common::{BpResult, Immo};
use derive_more::*;
use genevo::genetic::Fitness;
use predictions::Evaluator;
use serde::Serialize;
use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::{Add, Sub};

use super::mean;
use std::fmt;

/// How many KPIs are tracked in one one KpiOutput?
// Be sure to change [KpiOutput::worst_possible] and [KpiOutput::best_possible] everytime you change
// this value.
pub const KPI_COUNT: usize = 13;

/// A struct which contains all our key performance indicators.
#[derive(Copy, Debug, Clone, Serialize, Default)]
#[allow(missing_docs)]
pub struct KpiOutput {
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub mean_absolute_error: f64,

    pub mean_percentage_error: f64,
    pub mean_absolute_percentage_error: f64,
    pub median_absolute_percentage_error: f64,

    pub mean_squared_logarithmic_error: f64,
    pub root_mean_squared_logarithmic_error: f64,

    pub mean_absolute_percentage_error_on_logs: f64,

    pub r_squared: f64,

    /// The fraction of data with percentage error at most 5%.
    pub pe_5_fraction: f64,
    /// The fraction of data with percentage error at most 10%.
    pub pe_10_fraction: f64,
    /// The fraction of data with percentage error at most 15%.
    pub pe_15_fraction: f64,
    /// The fraction of data with percentage error at most 20%.
    pub pe_20_fraction: f64,
}

fn significant(num: f64) -> f64 {
    let log10 = num.abs().log10().floor();
    let fac = 10f64.powf(-log10) * 100.0;
    (num * fac).round() / fac
}

impl KpiOutput {
    /// Returns the worst possible KpiOutput.
    pub fn worst_possible() -> Self {
        // High numbers bad for MSE, MAPE etc.
        let mut result = vec![f64::INFINITY; 8];
        // Low numbers bad for R^2, PE5 etc.
        result.append(&mut vec![0.0; 5]);
        result.push(f64::INFINITY);
        Self::from(result)
    }

    /// Returns the best possible KpiOutput.
    pub fn best_possible() -> Self {
        // Low numbers good for MSE, MAPE etc.
        let mut result = vec![0.0; 8];
        // Numbers close to 1 good for R^2, PE5 etc.
        result.append(&mut vec![1.0; 5]);
        result.push(0.0);
        Self::from(result)
    }

    /// Divides each kpi metric individually by the argument.
    /// This will produce a new [KpiOutput].
    fn divide_all_values_by(&self, divisor: f64) -> Self {
        let mut vec: Vec<f64> = self.into();
        for value in vec.iter_mut() {
            *value /= divisor;
        }
        Self::from(vec)
    }

    /// Calculate the average among a set of [KpiOutput]s
    pub fn average<'i>(values: impl IntoIterator<Item = &'i KpiOutput>) -> KpiOutput {
        let (len, sum) = values
            .into_iter()
            .copied()
            .fold((0, KpiOutput::default()), |(len, sum), curr| {
                (len + 1, sum + curr)
            });

        sum.divide_all_values_by(len as f64)
    }
}

impl Display for KpiOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "========== Key Performance Indicators ==========")?;

        // absolute metrics
        writeln!(
            f,
            "Mean Squared error: {}",
            significant(self.mean_squared_error)
        )?;
        writeln!(
            f,
            "Root mean Squared error: {}",
            significant(self.root_mean_squared_error)
        )?;
        writeln!(
            f,
            "Mean absolute error: {}",
            significant(self.mean_absolute_error)
        )?;
        writeln!(f)?;

        // relative metrics
        writeln!(
            f,
            "Mean absolute percentage error: {}",
            significant(self.mean_absolute_percentage_error)
        )?;
        writeln!(
            f,
            "Mean percentage error: {}",
            significant(self.mean_percentage_error)
        )?;
        writeln!(
            f,
            "Median absolute percentage error: {}",
            significant(self.median_absolute_percentage_error)
        )?;
        writeln!(f)?;

        // logarithm metrics
        writeln!(
            f,
            "Mean squared logarithmic error: {}",
            significant(self.mean_squared_logarithmic_error)
        )?;
        writeln!(
            f,
            "Root mean squared logarithmic error: {}",
            significant(self.root_mean_squared_logarithmic_error)
        )?;
        writeln!(
            f,
            "mean absolute percentage error on logs: {}",
            significant(self.mean_absolute_percentage_error_on_logs)
        )?;
        writeln!(f)?;

        // r squared
        writeln!(f, "R squared: {}", significant(self.r_squared))?;
        writeln!(f)?;

        // pe buckets
        for (pe, frac) in &[
            (5, self.pe_5_fraction),
            (10, self.pe_10_fraction),
            (15, self.pe_15_fraction),
            (20, self.pe_20_fraction),
        ] {
            writeln!(f, "{}% PE Bucket: {}", pe, significant(*frac))?;
        }
        writeln!(f, "================================================")?;

        Ok(())
    }
}

impl From<Vec<f64>> for KpiOutput {
    fn from(vec: Vec<f64>) -> Self {
        let result = KpiOutput {
            mean_squared_error: vec[0],
            root_mean_squared_error: vec[1],
            mean_absolute_error: vec[2],
            mean_percentage_error: vec[3],
            mean_absolute_percentage_error: vec[4],
            median_absolute_percentage_error: vec[5],
            mean_squared_logarithmic_error: vec[6],
            root_mean_squared_logarithmic_error: vec[7],
            r_squared: vec[8],
            pe_5_fraction: vec[9],
            pe_10_fraction: vec[10],
            pe_15_fraction: vec[11],
            pe_20_fraction: vec[12],
            mean_absolute_percentage_error_on_logs: vec[13],
        };
        assert!(result.r_squared <= 1.0);
        assert!(0.0 <= result.pe_5_fraction && result.pe_5_fraction <= 1.0);
        assert!(0.0 <= result.pe_10_fraction && result.pe_10_fraction <= 1.0);
        assert!(0.0 <= result.pe_15_fraction && result.pe_15_fraction <= 1.0);
        assert!(0.0 <= result.pe_20_fraction && result.pe_20_fraction <= 1.0);
        result
    }
}

impl From<KpiOutput> for Vec<f64> {
    fn from(kpi: KpiOutput) -> Self {
        (&kpi).into()
    }
}

impl From<&KpiOutput> for Vec<f64> {
    fn from(kpi: &KpiOutput) -> Self {
        vec![
            kpi.mean_squared_error,
            kpi.root_mean_squared_error,
            kpi.mean_absolute_error,
            kpi.mean_percentage_error,
            kpi.mean_absolute_percentage_error,
            kpi.median_absolute_percentage_error,
            kpi.mean_squared_logarithmic_error,
            kpi.root_mean_squared_logarithmic_error,
            kpi.r_squared,
            kpi.pe_5_fraction,
            kpi.pe_10_fraction,
            kpi.pe_15_fraction,
            kpi.pe_20_fraction,
            kpi.mean_absolute_percentage_error_on_logs,
        ]
    }
}

impl Sub for KpiOutput {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            mean_squared_error: self.mean_squared_error - other.mean_squared_error,
            root_mean_squared_error: self.root_mean_squared_error - other.root_mean_squared_error,
            mean_absolute_error: self.mean_absolute_error - other.mean_absolute_error,
            mean_percentage_error: self.mean_percentage_error - other.mean_percentage_error,
            mean_absolute_percentage_error: self.mean_absolute_percentage_error
                - other.mean_absolute_percentage_error,
            median_absolute_percentage_error: self.median_absolute_percentage_error
                - other.median_absolute_percentage_error,
            mean_squared_logarithmic_error: self.mean_squared_logarithmic_error
                - other.mean_squared_logarithmic_error,
            root_mean_squared_logarithmic_error: self.root_mean_squared_logarithmic_error
                - other.root_mean_squared_logarithmic_error,
            r_squared: self.r_squared - other.r_squared,
            pe_5_fraction: self.pe_5_fraction - other.pe_5_fraction,
            pe_10_fraction: self.pe_10_fraction - other.pe_10_fraction,
            pe_15_fraction: self.pe_15_fraction - other.pe_15_fraction,
            pe_20_fraction: self.pe_20_fraction - other.pe_20_fraction,
            mean_absolute_percentage_error_on_logs: self.mean_absolute_percentage_error_on_logs
                - other.mean_absolute_percentage_error_on_logs,
        }
    }
}

impl Add for KpiOutput {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            mean_squared_error: self.mean_squared_error + other.mean_squared_error,
            root_mean_squared_error: self.root_mean_squared_error + other.root_mean_squared_error,
            mean_absolute_error: self.mean_absolute_error + other.mean_absolute_error,
            mean_percentage_error: self.mean_percentage_error + other.mean_percentage_error,
            mean_absolute_percentage_error: self.mean_absolute_percentage_error
                + other.mean_absolute_percentage_error,
            median_absolute_percentage_error: self.median_absolute_percentage_error
                + other.median_absolute_percentage_error,
            mean_squared_logarithmic_error: self.mean_squared_logarithmic_error
                + other.mean_squared_logarithmic_error,
            root_mean_squared_logarithmic_error: self.root_mean_squared_logarithmic_error
                + other.root_mean_squared_logarithmic_error,
            r_squared: self.r_squared + other.r_squared,
            pe_5_fraction: self.pe_5_fraction + other.pe_5_fraction,
            pe_10_fraction: self.pe_10_fraction + other.pe_10_fraction,
            pe_15_fraction: self.pe_15_fraction + other.pe_15_fraction,
            pe_20_fraction: self.pe_20_fraction + other.pe_20_fraction,
            mean_absolute_percentage_error_on_logs: self.mean_absolute_percentage_error_on_logs
                + other.mean_absolute_percentage_error_on_logs,
        }
    }
}

impl Sum for KpiOutput {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, current| acc + current)
    }
}

/// An Evaluator for all our Key Performane Indicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct KpiEvaluator {}

impl KpiEvaluator {
    /// Creates a new KpiEvaluator
    pub fn new() -> Self {
        Self {}
    }
}

fn fraction_below(iter: impl Iterator<Item = f64> + Clone, threshhold: f64) -> f64 {
    iter.clone().filter(|p| *p <= threshhold).count() as f64 / iter.count() as f64
}

fn variance(iter: impl Iterator<Item = f64> + Clone) -> f64 {
    let mean: f64 = mean(iter.clone());
    iter.clone().map(|v| (v - mean) * (v - mean)).sum::<f64>() / iter.count() as f64
}

impl Evaluator for KpiEvaluator {
    type Output = KpiOutput;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        let pairs: Vec<_> = pairs.into_iter().collect();

        if pairs.is_empty() {
            return Err("No Immos were provided".into());
        }
        pairs
            .iter()
            .try_for_each(|(real, predicted)| -> BpResult<()> {
                if real.sqm_price().is_none() {
                    Err(format!("Real Immo({:?}) had no sqm price", real.id()).into())
                } else if !real.sqm_price().unwrap().is_finite() {
                    Err(format!("Real Immo({:?}) has Nan or inf sqm_price", real.id()).into())
                } else if real.sqm_price().unwrap().abs() < 1.0 + f64::EPSILON {
                    Err(format!("Real Immo({:?}) has <= 1 sqm_price", real.id()).into())
                } else if predicted.sqm_price().is_none() {
                    Err(format!("Predicted Immo({:?}) had no sqm price", predicted.id()).into())
                } else if !predicted.sqm_price().unwrap().is_finite() {
                    Err(format!("Predicted Immo({:?}) has Nan or inf sqm_price", real.id()).into())
                } else {
                    Ok(())
                }
            })?;

        let absolute_deviations = pairs
            .iter()
            .map(|(real, predicted)| real.sqm_price().unwrap() - predicted.sqm_price().unwrap());

        let relative_deviations = pairs.iter().map(|(real, predicted)| {
            (real.sqm_price().unwrap() - predicted.sqm_price().unwrap()) / real.sqm_price().unwrap()
        });
        let absolute_relative_deviations = relative_deviations.clone().map(|d| d.abs());

        let log_deviations = pairs.iter().map(|(real, predicted)| {
            real.sqm_price().unwrap().ln_1p() - predicted.sqm_price().unwrap().ln_1p()
        });

        let mean_squared_error = mean(absolute_deviations.clone().map(|d| d * d));
        let mean_squared_logarithmic_error =
            log_deviations.clone().map(|d| d * d).sum::<f64>() / pairs.len() as f64;

        let median_absolute_percentage_error = {
            let mut collected: Vec<_> = relative_deviations.clone().map(|d| d.abs()).collect();
            collected.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            // this min is needed to work on input of size 1.
            collected[((collected.len() + 1) / 2).min(collected.len() - 1)]
        };

        let pairs_log = pairs.iter().map(|(real, predicted)| {
            (
                real.sqm_price().unwrap().ln(),
                predicted.sqm_price().unwrap().ln(),
            )
        });
        let mean_absolute_percentage_error_on_logs =
            mean(pairs_log.map(|(real, predicted)| (1.0 - real / predicted).abs()));

        Ok(KpiOutput {
            mean_squared_error,
            root_mean_squared_error: mean_squared_error.sqrt(),
            mean_absolute_error: mean(absolute_deviations.clone().map(|d| d.abs())),
            mean_percentage_error: mean(relative_deviations.clone()),
            mean_absolute_percentage_error: mean(absolute_relative_deviations.clone()),
            median_absolute_percentage_error,
            mean_squared_logarithmic_error,
            root_mean_squared_logarithmic_error: mean_squared_logarithmic_error.sqrt(),
            r_squared: 1.0
                - mean_squared_error
                    / variance(pairs.iter().map(|(real, _)| real.sqm_price().unwrap())),
            pe_5_fraction: fraction_below(absolute_relative_deviations.clone(), 0.05),
            pe_10_fraction: fraction_below(absolute_relative_deviations.clone(), 0.10),
            pe_15_fraction: fraction_below(absolute_relative_deviations.clone(), 0.15),
            pe_20_fraction: fraction_below(absolute_relative_deviations, 0.20),
            mean_absolute_percentage_error_on_logs,
        })
    }
}

#[derive(Debug, Clone, Serialize, From, Into)]
/// This can be used as evaluator output. The advantage is that is it comparable. Comparison is by
/// MAPE.
pub struct MinimizeMapeKpiOutput(KpiOutput);

impl PartialEq for MinimizeMapeKpiOutput {
    fn eq(&self, other: &MinimizeMapeKpiOutput) -> bool {
        self.0.mean_absolute_percentage_error == other.0.mean_absolute_percentage_error
    }
}
impl Eq for MinimizeMapeKpiOutput {}
impl PartialOrd for MinimizeMapeKpiOutput {
    fn partial_cmp(&self, other: &MinimizeMapeKpiOutput) -> Option<Ordering> {
        self.0
            .mean_absolute_percentage_error
            .partial_cmp(&other.0.mean_absolute_percentage_error)
    }
}
impl Ord for MinimizeMapeKpiOutput {
    fn cmp(&self, other: &MinimizeMapeKpiOutput) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone, Default)]
/// This evaluator acts like [KpiEvaluator], but wraps the result in a [MinimizeMAPEKpiOutput].
pub struct MinimizeMapeKpiEvaluator(KpiEvaluator);

impl Evaluator for MinimizeMapeKpiEvaluator {
    type Output = MinimizeMapeKpiOutput;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        self.0.evaluate(pairs).map(MinimizeMapeKpiOutput)
    }
}

/// This can be used as evaluator output. The advantage is that is it comparable. Comparison is by
/// MAPE.
#[derive(Copy, Clone, Default)]
pub struct MinimizeMapeInvertedOrdKpiOutput(pub KpiOutput);

impl MinimizeMapeInvertedOrdKpiOutput {
    /// See [KpiOutput::worst_possible]
    pub fn worst_possible() -> Self {
        Self(KpiOutput::worst_possible())
    }

    /// See [KpiOutput::best_possible]
    pub fn best_possible() -> Self {
        Self(KpiOutput::best_possible())
    }

    /// See [KpiOutput::divide_all_values_by]
    pub fn divide_all_values_by(&self, divisor: f64) -> Self {
        Self(self.0.divide_all_values_by(divisor))
    }

    /// Gives the important value for the KPIOutput
    pub fn value(&self) -> f64 {
        self.0.mean_absolute_percentage_error
    }

    /// See [KpiOutput::average]
    pub fn average<'i>(
        values: impl IntoIterator<Item = &'i MinimizeMapeInvertedOrdKpiOutput>,
    ) -> MinimizeMapeInvertedOrdKpiOutput {
        MinimizeMapeInvertedOrdKpiOutput(KpiOutput::average(
            values.into_iter().map(|output| &output.0),
        ))
    }
}

impl From<MinimizeMapeInvertedOrdKpiOutput> for f64 {
    fn from(mmioko: MinimizeMapeInvertedOrdKpiOutput) -> Self {
        mmioko.value()
    }
}

impl PartialEq for MinimizeMapeInvertedOrdKpiOutput {
    fn eq(&self, other: &MinimizeMapeInvertedOrdKpiOutput) -> bool {
        self.0.mean_absolute_percentage_error == other.0.mean_absolute_percentage_error
    }
}
impl Eq for MinimizeMapeInvertedOrdKpiOutput {}

/// NOTE: This implementation is exactly the other way arround than one would expect for the MAPE.
/// That is the output is "smaller" if the MAPE is larger.
impl PartialOrd for MinimizeMapeInvertedOrdKpiOutput {
    fn partial_cmp(&self, other: &MinimizeMapeInvertedOrdKpiOutput) -> Option<Ordering> {
        other
            .0
            .mean_absolute_percentage_error
            .partial_cmp(&self.0.mean_absolute_percentage_error)
    }
}
impl Ord for MinimizeMapeInvertedOrdKpiOutput {
    fn cmp(&self, other: &MinimizeMapeInvertedOrdKpiOutput) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Sub for MinimizeMapeInvertedOrdKpiOutput {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Add for MinimizeMapeInvertedOrdKpiOutput {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sum for MinimizeMapeInvertedOrdKpiOutput {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|minimize_output| minimize_output.0).sum())
    }
}

impl fmt::Debug for MinimizeMapeInvertedOrdKpiOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_fmt(format_args!(
            "MinimizeMapeInvertedOrdKpiOutput {{ MAPE={} }}",
            self.0.mean_absolute_percentage_error
        ))
    }
}

/// This evaluator acts like [KpiEvaluator], but wraps the result in a [MinimizeMapeInvertedOrdKpiOutput].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MinimizeMapeInvertedOrdKpiEvaluator(KpiEvaluator);

impl MinimizeMapeInvertedOrdKpiEvaluator {
    /// Creates a new [MinimizeMapeInvertedOrdKpiEvaluator]
    pub fn new() -> Self {
        Self(KpiEvaluator::new())
    }
}

impl Evaluator for MinimizeMapeInvertedOrdKpiEvaluator {
    type Output = MinimizeMapeInvertedOrdKpiOutput;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        self.0.evaluate(pairs).map(MinimizeMapeInvertedOrdKpiOutput)
    }
}

impl Fitness for MinimizeMapeInvertedOrdKpiOutput {
    fn zero() -> Self {
        MinimizeMapeInvertedOrdKpiOutput(KpiOutput::default())
    }

    fn abs_diff(&self, other: &Self) -> Self {
        let self_vec: Vec<_> = self.0.into();
        let other_vec: Vec<_> = other.0.into();

        log::info!("start from");
        let res = MinimizeMapeInvertedOrdKpiOutput(KpiOutput::from(
            self_vec
                .iter()
                .zip(other_vec)
                .map(|(my_value, other_value)| (my_value - other_value).abs())
                .collect::<Vec<_>>(),
        ));
        log::info!("end from");
        res
    }
}
