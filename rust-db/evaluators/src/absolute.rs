use super::DeviationOutput;
use common::{BpResult, Immo};
use predictions::Evaluator;

/// An evaluator which compares the absolute difference in square meter price.
/// As result a (DeviationOutput)[DeviationOutput] will be produced.
#[derive(Debug, Clone)]
pub struct AbsoluteSqmPriceDeviationEvaluator {
    num_percentiles: usize,
}

impl AbsoluteSqmPriceDeviationEvaluator {
    /// Creates a new AbsoluteSqmPriceDeviationEvaluator which will compute 3 percentiles (25%, 50%, 75%).
    pub fn new() -> Self {
        Self::with_percentiles(3)
    }

    /// Creates a new AbsoluteSqmPriceDeviationEvaluator which will compute `num_percentiles` percentiles.
    pub fn with_percentiles(num_percentiles: usize) -> Self {
        Self { num_percentiles }
    }
}

impl Default for AbsoluteSqmPriceDeviationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for AbsoluteSqmPriceDeviationEvaluator {
    type Output = DeviationOutput;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        let mut deviations = Vec::new();
        for tup in pairs
            .into_iter()
            .map(|(real, predicted)| (real.sqm_price(), predicted.sqm_price()))
        {
            match tup {
                (Some(real_sqm_price), Some(predicted_sqm_price)) => {
                    if real_sqm_price.is_nan() || predicted_sqm_price.is_nan() {
                        return Err("Real or predicted Immo has NaN as sqm price".into());
                    }
                    let deviation = (real_sqm_price - predicted_sqm_price).abs();
                    deviations.push(deviation);
                }
                (None, _) => return Err("Real Immo does not have sqm_price".into()),
                (Some(_), None) => return Err("Predictor did not set sqm_price".into()),
            }
        }

        Ok(
            DeviationOutput::with_percentiles(deviations.iter().copied(), self.num_percentiles)
                .ok_or("Could not construct Output")?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use test_helpers::*;

    proptest! {
        #[test]
        fn absolute_sqm_price_deviation_evaluator_gives_correct_deviation_for_0_prediction(
            real_immos in full_immos(128)
        ) {
            let pairs: Vec<_> = real_immos
                .into_iter()
                .map(|real| {
                    let mut predicted = real.clone();
                    predicted.marktwert = Some(0.0);
                    (real, predicted)
                })
                .collect();

            let evaluation_output = AbsoluteSqmPriceDeviationEvaluator::new()
                .evaluate(pairs.iter().map(|(real, predicted)| (real, predicted))).unwrap();

            let sum_sqm_price : f64 = pairs.iter().map(|(real, _)| real.sqm_price().unwrap()).sum();
            prop_assert!((evaluation_output.mean() * (pairs.len() as f64) - sum_sqm_price).abs() / sum_sqm_price < 1e-9);
        }
    }
}
