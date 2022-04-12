use crate::{LearningDully, WeightedAveragePredictor};
use common::{BpResult, Immo, Trainable};
use dissimilarities::ConstantDissimilarity;
use predictions::{Evaluator, Predictor};
use proptest::prelude::*;
use std::iter::once;
use test_helpers::*;

macro_rules! dully_tests {
        ($mod : ident, $gen : expr) => {
            mod $mod {
                use super::*;

                proptest! {
                    #[test]
                    fn constant_marktwert_yields_constant_prediction(
                        mut immos in prop::collection::vec(berlin_full_immo(), 1..256),
                        mut predict in berlin_full_immo(),
                        target in 0.1..1e9f64,
                    ) {
                        for immo in &mut immos {
                            immo.marktwert = Some(target * immo.wohnflaeche.unwrap());
                        }
                        let mut dully = $gen;
                        dully.train(&immos).unwrap();

                        dully.predict(once(&mut predict)).unwrap();

                        let abs_relative_diff: f64 = (1.0 - predict.sqm_price().unwrap() / target).abs();
                        prop_assert!(abs_relative_diff < 1e-6)
                    }

                    #[test]
                    fn err_without_marktwert(
                        mut immos in prop::collection::vec(berlin_full_immo(), 1..256),
                    ) {
                        for immo in &mut immos {
                            immo.marktwert = None;
                        }
                        let mut dully = $gen;
                        prop_assert!(dully.train(&immos).is_err());
                    }

                    #[test]
                    fn err_without_wohnflaeche(
                        mut immos in prop::collection::vec(berlin_full_immo(), 1..256),
                    ) {
                        for immo in &mut immos {
                            immo.wohnflaeche = None;
                        }
                        let mut dully = $gen;
                        prop_assert!(dully.train(&immos).is_err());
                    }
                }
            }
        };
    }

#[derive(Debug, Default)]
struct IndifferentEvaluator;

impl Evaluator for IndifferentEvaluator {
    type Output = ();

    fn evaluate<'i>(
        &self,
        _pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        Ok(())
    }
}

dully_tests!(dully, WeightedAveragePredictor::dully());

dully_tests!(
    learning_dully,
    LearningDully::with(IndifferentEvaluator::default())
);

proptest! {
    #[test]
    fn constant_dissimilarity_is_average(
        immos in prop::collection::vec(berlin_full_immo(), 1..256),
        mut predict in berlin_full_immo(),
        constant in 0.1..1e3,
    ) {
        let mut predictor = WeightedAveragePredictor::with_radius(ConstantDissimilarity::with(constant), f64::INFINITY);
        predictor.train(&immos).unwrap();

        predictor.predict(once(&mut predict)).unwrap();

        let average = immos.iter().map(|immo| immo.sqm_price().unwrap()).sum::<f64>() / immos.len() as f64;

        let abs_relative_diff: f64 = (1.0 - predict.sqm_price().unwrap() / average).abs();
        prop_assert!(abs_relative_diff < 1e-6)
    }
}
