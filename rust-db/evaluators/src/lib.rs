#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]

//! This crate contains all our evaluators.

mod deviation_output;
pub use deviation_output::DeviationOutput;

mod absolute;
pub use absolute::AbsoluteSqmPriceDeviationEvaluator;

mod relative;
pub use relative::RelativeSqmPriceDeviationEvaluator;

mod full;
pub use full::FullSqmPriceEvaluator;

pub mod kpi;
pub use kpi::{KpiEvaluator, KpiOutput};

mod identity;
pub use identity::IdentityEvaluator;

fn mean(iter: impl Iterator<Item = f64> + Clone) -> f64 {
    iter.clone().sum::<f64>() / iter.count() as f64
}
#[cfg(test)]
mod tests {
    #[derive(Debug, Copy, Clone)]
    struct Marktwert(f64);

    macro_rules! generate_overlapping_error_tests {
        ($name:ident, $gen:expr) => {
            mod $name {
                use crate::*;
                use predictions::Evaluator;
                use proptest::prelude::*;
                use test_helpers::*;
                proptest! {
                    #[test]
                    fn err_on_missing_values_in_real_data(mut real in full_immos(128)) {
                        for immo in &mut real {
                            immo.marktwert = None;
                        }

                        let evaluation_output = $gen.evaluate(real.iter().zip(real.iter()));
                        prop_assert!(
                            evaluation_output.is_err(),
                            "No error despite real immo not having sqm price"
                        );
                    }

                    #[test]
                    fn evaluator_gives_err_on_nan(mut real in full_immos(128)) {
                        for immo in &mut real {
                            immo.wohnflaeche = Some(0.0);
                        }

                        let evaluation_output = $gen.evaluate(real.iter().zip(real.iter()));
                        prop_assert!(evaluation_output.is_err());
                    }

                    #[test]
                    fn err_on_missing_values_in_predicted_data(real in full_immos(128)) {
                        let mut predicted = real.clone();
                        for immo in &mut predicted {
                            immo.marktwert = None;
                        }

                        let evaluation_output = $gen.evaluate(real.iter().zip(predicted.iter()));
                        assert!(
                            evaluation_output.is_err(),
                            "No error despite predicted immo not having sqm price"
                        );
                    }
                }
            }
        };
    }

    generate_overlapping_error_tests!(
        absolute_sqm_price_deviation_evaluator,
        AbsoluteSqmPriceDeviationEvaluator::new()
    );
    generate_overlapping_error_tests!(
        relative_sqm_price_deviation_evaluator,
        RelativeSqmPriceDeviationEvaluator::new()
    );
    generate_overlapping_error_tests!(full_sqm_price_evaluator, FullSqmPriceEvaluator::new());
    generate_overlapping_error_tests!(kpi_evaluator, KpiEvaluator::new());
}
