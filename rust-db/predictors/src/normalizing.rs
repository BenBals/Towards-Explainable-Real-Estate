use common::{BpResult, Immo, Normalizer, Trainable};
use predictions::Predictor;

/// A wrapper for predictors which adjusts sqm prices to the continual shift in market prices.
#[derive(Debug, Clone)]
pub struct NormalizingPredictor<N, P> {
    /// The normalizer which will be used.
    pub normalizer: N,
    /// The predictor which will do it actual prediction.
    /// It will be provided with normalized input and the output will be denormalized.
    pub inner: P,
}

impl<N, P> NormalizingPredictor<N, P> {
    /// Creates a new NormalizingPredictor which uses `normalizer` to adjust the prices and
    /// then lets `inner` do the actual prediction.
    pub fn with(normalizer: N, inner: P) -> Self {
        NormalizingPredictor { normalizer, inner }
    }

    /// Gives access to the wrapped predictor
    pub fn predictor(&self) -> &P {
        &self.inner
    }

    /// Gives access to the used normalizer
    pub fn normalizer(&self) -> &N {
        &self.normalizer
    }
}

impl<N: Normalizer, P: Trainable> Trainable for NormalizingPredictor<N, P> {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        let mut all_training_data: Vec<_> = training_data.into_iter().cloned().collect();

        self.normalizer.train(&all_training_data)?;
        self.normalizer.normalize(&mut all_training_data);

        self.inner.train(&all_training_data)
    }
}

impl<N: Normalizer, P: Predictor> Predictor for NormalizingPredictor<N, P> {
    fn predict<'j>(&self, validation_data: impl IntoIterator<Item = &'j mut Immo>) -> BpResult<()> {
        let mut to_predict: Vec<_> = validation_data.into_iter().collect();
        let res = self.inner.predict(to_predict.iter_mut().map(|i| &mut **i));

        self.normalizer
            .denormalize(to_predict.iter_mut().map(|immo| &mut **immo));

        res
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use proptest::prelude::*;
    use proptest::proptest;

    use super::*;
    use common::BpResult;
    use predictions::{Driver, Evaluator, Predictor};
    use test_helpers::*;

    #[derive(Debug, Clone, Copy)]
    struct TestNormalizer;

    #[derive(Debug)]
    struct TestPredictor<'c> {
        supplied_only_normalized: &'c mut Cell<bool>,
    }

    #[derive(Debug, Clone, Copy)]
    struct TestEvaluator;

    impl Trainable for TestNormalizer {}
    impl Normalizer for TestNormalizer {
        fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
            immos.into_iter().for_each(|immo| immo.marktwert = None)
        }

        fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
            immos
                .into_iter()
                .for_each(|immo| immo.marktwert = Some(1.0))
        }
    }

    impl<'c> Trainable for TestPredictor<'c> {
        fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
            *self.supplied_only_normalized.get_mut() &= training_data
                .into_iter()
                .all(|immo| immo.marktwert.is_none());
            Ok(())
        }
    }

    impl<'c> Predictor for TestPredictor<'c> {
        fn predict<'i>(
            &self,
            validation_data: impl IntoIterator<Item = &'i mut Immo>,
        ) -> BpResult<()> {
            for datum in validation_data {
                let old_value = self.supplied_only_normalized.get();
                self.supplied_only_normalized
                    .set(old_value && datum.marktwert.is_none());
            }
            Ok(())
        }
    }

    impl Evaluator for TestEvaluator {
        type Output = bool;

        fn evaluate<'i>(
            &self,
            pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
        ) -> BpResult<Self::Output> {
            Ok(pairs
                .into_iter()
                .all(|(real, predicted)| real.marktwert.is_some() && predicted.marktwert.is_some()))
        }
    }

    proptest! {
        #[test]
        fn normalizing_predictor_only_supplied_normalized_immos_to_predictor(immos in full_immos(128)) {
            let mut supplied_only_normalized = Cell::new(true);

            let mut predictor = NormalizingPredictor::with(TestNormalizer, TestPredictor {supplied_only_normalized : &mut supplied_only_normalized});
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            driver.drive(&mut immos.iter().collect::<Vec<_>>()).expect("this can't happen");

            prop_assert!(supplied_only_normalized.get());
        }

        #[test]
        fn normalizing_predictor_only_supplied_denormalized_immos_to_evaluator(immos in full_immos(128)) {
            let mut supplied_only_normalized = Cell::new(true);

            let mut predictor = NormalizingPredictor::with(TestNormalizer, TestPredictor {supplied_only_normalized : &mut supplied_only_normalized});
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);

            prop_assert!(driver.drive(&mut immos.iter().collect::<Vec<_>>()).unwrap());
        }
    }
}
