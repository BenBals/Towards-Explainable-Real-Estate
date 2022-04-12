use serde::Serialize;

use super::mean;

/// A struct which accumulates many deviations.
/// It provides access to the average, minimum, maximum and some configurable number of percentiles.
#[derive(Debug, Clone, Serialize)]
pub struct DeviationOutput {
    mean: f64,
    min_deviation: f64,
    max_deviation: f64,
    percentiles: Vec<f64>,
}

impl DeviationOutput {
    /// Accumulates the given deviations, ignoring all NaN values.
    /// Like (with_percentiles)[DeviationOutput::with_percentiles] with `num_percentiles`=3.
    /// This will compute 25-th, 50-th and 75-th percentile.
    pub fn new(deviation_iter: impl Iterator<Item = f64>) -> Option<Self> {
        Self::with_percentiles(deviation_iter, 3)
    }

    /// Accumulate the given deviations, ignoring all NaN values.
    /// This will compute the minimum, maximum and average over deviations and `num_percentiles` many percentiles.
    ///
    /// # Return value
    /// If `deviation_iter` produces nothing but NaN values, this function returns None.
    pub fn with_percentiles(
        deviation_iter: impl Iterator<Item = f64>,
        num_percentiles: usize,
    ) -> Option<Self> {
        let mut deviations: Vec<_> = deviation_iter.filter(|f| !f.is_nan()).collect();
        if deviations.is_empty() {
            return None;
        }

        deviations.sort_unstable_by(|a, b| a.partial_cmp(b).expect("we have no NaNs"));
        let min_deviation = deviations.first()?;
        let max_deviation = deviations.last()?;

        let percentiles: Vec<_> = (1..=num_percentiles)
            .map(|percentile| deviations[percentile * deviations.len() / (num_percentiles + 1)])
            .collect();

        Some(Self {
            mean: mean(deviations.iter().copied()),
            min_deviation: *min_deviation,
            max_deviation: *max_deviation,
            percentiles,
        })
    }

    /// Gives the average deviation of the given data
    pub fn mean(&self) -> f64 {
        self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_deviation_output_percentiles_for_fixed_value() {
        let deviations = vec![1.0, 6.0, 3.0, 2.0, 4.0, 0.0, 5.0];

        let deviation_output = DeviationOutput::new(deviations.iter().copied()).unwrap();

        assert_eq!(deviation_output.percentiles, vec![1.0, 3.0, 5.0]);
    }

    proptest! {
        #[test]
        fn deviation_output_chooses_median_with_1_percentile(
            mut deviations in prop::collection::vec(prop::num::f64::NORMAL, 1..128)
        ) {
            let deviation_output = DeviationOutput::with_percentiles(deviations.iter().copied(), 1).unwrap();

            deviations.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let median = deviations[deviations.len() / 2];
            prop_assert!((deviation_output.percentiles[0] - median).abs() / median < f64::EPSILON);
        }
    }

    macro_rules! generate_overlapping_deviation_tests {
        ($name:ident, $gen:expr) => {
            mod $name {
                use crate::*;
                use predictions::{Evaluator};
                use test_helpers::*;
                use proptest::prelude::*;
                proptest! {
                    #[test]
                    fn deviation_is_0_for_equal_prediction(real_immos in full_immos(128)) {
                        let pairs: Vec<_> = real_immos
                            .into_iter()
                            .map(|real| {
                                let predicted = real.clone();
                                (real, predicted)
                            })
                            .collect();

                        let evaluation_output = $gen
                            .evaluate(pairs.iter().map(|(real, predicted)| (real, predicted))).unwrap();

                        prop_assert!(evaluation_output.min_deviation.abs() < f64::EPSILON);
                        prop_assert!(evaluation_output.max_deviation.abs() < f64::EPSILON);
                        prop_assert!(evaluation_output.mean.abs() < f64::EPSILON);
                    }
                }
            }
        };
    }

    generate_overlapping_deviation_tests!(
        relative_sqm_price_deviation_evaluator_deviation,
        RelativeSqmPriceDeviationEvaluator::new()
    );
    generate_overlapping_deviation_tests!(
        absolute_sqm_price_deviation_evaluator_deviation,
        AbsoluteSqmPriceDeviationEvaluator::new()
    );
}
