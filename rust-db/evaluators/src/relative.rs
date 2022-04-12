use common::{BpResult, Immo};
use predictions::Evaluator;

use super::DeviationOutput;

/// An evaluator which compares the relative difference in square meter price.
/// As result a (DeviationOutput)[DeviationOutput] will be produced.
#[derive(Debug, Clone)]
pub struct RelativeSqmPriceDeviationEvaluator {
    num_percentiles: usize,
}

impl RelativeSqmPriceDeviationEvaluator {
    /// Creates a new RelativeSqmPriceDeviationEvaluator which will compute 3 percentiles (25%, 50%, 75%).
    pub fn new() -> Self {
        Self::with_percentiles(3)
    }

    /// Creates a new RelativeSqmPriceDeviationEvaluator which will compute `num_percentiles` percentiles.
    pub fn with_percentiles(num_percentiles: usize) -> Self {
        Self { num_percentiles }
    }
}

impl Default for RelativeSqmPriceDeviationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for RelativeSqmPriceDeviationEvaluator {
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
                    if real_sqm_price.abs() < f64::EPSILON {
                        return Err("Real Immo has zero sqm price".into());
                    }
                    let deviation = (real_sqm_price - predicted_sqm_price).abs() / real_sqm_price;
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
        fn relative_sqm_price_deviation_evaluator_gives_err_on_real_immo_zero_sqm_price(
            mut real in full_immos(128)
        ) {
            for immo in &mut real {
                immo.marktwert = Some(0.0);
            }

            let evaluation_output = RelativeSqmPriceDeviationEvaluator::new()
                .evaluate(real.iter().zip(real.iter()));
            prop_assert!(
                evaluation_output.is_err(),
                "No error despite real immo having zero sqm price"
            );
        }
    }
}
