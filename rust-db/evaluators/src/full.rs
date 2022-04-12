use std::collections::HashMap;

use serde::Serialize;

use common::{BpResult, Immo};
use predictions::Evaluator;

/// an Output type which saves all predictions. Useful for writing to a json file.
#[derive(Serialize, Default)]
pub struct FullSqmPriceOutput {
    prediction_map: HashMap<String, f64>,
}

impl FullSqmPriceOutput {
    /// Creates a FullSqmPriceOutput with an empty prediction map
    pub fn new() -> Self {
        Self {
            prediction_map: HashMap::new(),
        }
    }
}

/// an Evaluator which creates a (FullSqmPriceOutput)[FullSqmPriceOutput]
#[derive(Debug, Clone, Default)]
pub struct FullSqmPriceEvaluator;

impl FullSqmPriceEvaluator {
    /// Creates a FullSqmPriceEvaluator
    pub fn new() -> Self {
        Self {}
    }
}

impl Evaluator for FullSqmPriceEvaluator {
    type Output = FullSqmPriceOutput;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        let mut output = FullSqmPriceOutput::new();
        for tup in pairs
            .into_iter()
            .map(|(real, predicted)| (real.id(), predicted.sqm_price()))
        {
            match tup {
                (id, Some(predicted_sqm_price)) => {
                    if predicted_sqm_price.is_nan() || predicted_sqm_price.is_infinite() {
                        return Err(
                            "Predicted Immo has invalid (i.e. Nan or inf) as sqm price".into()
                        );
                    }
                    output
                        .prediction_map
                        .insert(id.to_string(), predicted_sqm_price);
                }
                (_, None) => return Err("Predictor did not set sqm_price".into()),
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use proptest::prelude::*;
    use test_helpers::*;

    proptest! {
        #[test]
        fn full_sqm_price_evaluator_gives_correct_predictions(
            real_immos_predictions in prop::collection::vec((full_immo(), prop::num::f64::NORMAL), 1..128)
        ) {
            let pairs: Vec<_> = real_immos_predictions
                .into_iter()
                .map(|(real, prediction)| {
                    let mut predicted = real.clone();
                    predicted.marktwert = Some(prediction);

                    (real, predicted)
                })
                .collect();

            let mut evaluation_output = FullSqmPriceEvaluator::new()
                .evaluate(pairs.iter().map(|(real, predicted)| (real, predicted))).unwrap();
            for (real, predicted) in pairs.iter() {
                prop_assert!((evaluation_output.prediction_map[&real.id().to_string()] - predicted.sqm_price().unwrap()).abs() < 1e-9);
                evaluation_output.prediction_map.remove(&real.id().to_string());
            }
            prop_assert!(evaluation_output.prediction_map.is_empty());
        }
    }
}
