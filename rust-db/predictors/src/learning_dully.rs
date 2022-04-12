use crate::WeightedAveragePredictor;
use common::{BpError, BpResult, Immo, Trainable};
use dissimilarities::DistanceDissimilarity;
use indicatif::ProgressBar;
use itertools::Itertools;
use predictions::{Driver, Evaluator, Predictor};
use std::{borrow::Borrow, marker::PhantomData};

/// This struct represents a Predictor which is a wrapper around a [WeightedAveragePredictor].
/// It uses the [DistanceDissimilarity].
/// This wrapper searches for the best exponent on the training data.
/// To decide which is exponent has the best fit to the data an evaluator is supplied.
pub struct LearningDully<'e, B, E>
where
    B: Borrow<E>,
{
    evaluator: B,
    _marker: PhantomData<&'e E>,
    inner: Option<WeightedAveragePredictor<DistanceDissimilarity>>,
    learned_exponent: Option<f64>,
}

impl<'e, B, E> LearningDully<'e, B, E>
where
    B: Borrow<E>,
{
    /// Creates a new LearningDully, which takes the best dully regarding the output of `evaluator`.
    /// The exponent minimizing the output of the evaluator is chosen.
    pub fn with(evaluator: B) -> Self {
        Self {
            evaluator,
            inner: None,
            learned_exponent: None,
            _marker: PhantomData::default(),
        }
    }

    /// Might give the exponent which was found to be the best on the training data.
    /// # Returns
    /// None if this instance was never trained successfully.
    pub fn learned_exponent(&self) -> Option<f64> {
        self.learned_exponent
    }
}

impl<'e, B, E, O> Trainable for LearningDully<'e, B, E>
where
    B: Borrow<E>,
    E: Evaluator<Output = O> + Sync,
    O: Ord + Send,
{
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        use log::Level::Info;

        let progress = if log::log_enabled!(Info) {
            ProgressBar::new(50)
        } else {
            ProgressBar::hidden()
        };

        let eval_ref = self.evaluator.borrow();
        // Odd numbers will always check the middle of the range (2.0), which is the normal Dully
        // exponent.
        let samples = 51;
        let collected_data: Vec<_> = training_data.into_iter().collect();
        let (errs, oks): (Vec<_>, Vec<_>) = (0..samples).partition_map(|i| {
            let min_range = 1.0;
            let max_range = 3.0;
            let exp = min_range + (i as f64 * (max_range - min_range) / (samples as f64 - 1.0));

            let mut dully = WeightedAveragePredictor::dully_with(exp);
            let mut driver = Driver::with(&mut dully, eval_ref);

            let res = driver.drive_full(&collected_data);
            progress.inc(1);
            match res {
                Ok(evaluation_result) => Ok((evaluation_result, exp)).into(),
                Err(err) => Err(format!("{:?}", err)).into(),
            }
        });
        progress.finish_and_clear();

        if !errs.is_empty() {
            log::warn!("Training of {} dullys failed", errs.len());
            for str in errs {
                log::warn!("\t{}", str);
            }
        }

        self.learned_exponent = Some(
            oks.into_iter()
                .min_by(|a, b| a.0.cmp(&b.0))
                .ok_or_else(|| BpError::from("Training of all dullys failed"))?
                .1,
        );

        let mut inner = WeightedAveragePredictor::dully_with(self.learned_exponent.unwrap());
        inner.train(collected_data)?;
        self.inner = Some(inner);

        log::info!(
            "DONE LearningDully(exp:{})",
            self.learned_exponent().unwrap()
        );

        Ok(())
    }
}

impl<'e, B, E, O> Predictor for LearningDully<'e, B, E>
where
    B: Borrow<E>,
    E: Evaluator<Output = O> + Sync,
    O: Ord + Send,
{
    fn predict<'i>(&self, validation_data: impl IntoIterator<Item = &'i mut Immo>) -> BpResult<()> {
        self.inner
            .as_ref()
            .ok_or_else(|| {
                BpError::from("No inner dully, training might not be called or it failed")
            })
            .and_then(|dully| dully.predict(validation_data))
    }
}

#[cfg(test)]
mod tests {
    use crate::{LearningDully, WeightedAveragePredictor};
    use evaluators::{kpi::MinimizeMapeKpiEvaluator, KpiEvaluator, KpiOutput};
    use predictions::Driver;
    use proptest::prelude::*;
    use test_helpers::*;

    proptest! {
        #[test]
        fn learning_dully_is_always_atleast_as_good_as_dully(immos in berlin_full_immos(128)) {
            prop_assume!(immos.len() >= 2);
            for immo in &immos {
                prop_assume!(immo.sqm_price().unwrap() > 1.1);
            }

            let dully_eval = KpiEvaluator::new();
            let mut dully = WeightedAveragePredictor::dully();
            let mut dully_driver = Driver::with(&mut dully, &dully_eval);
            let dully_output = dully_driver.drive_full(immos.iter().collect::<Vec<_>>().as_mut_slice()).unwrap();

            println!("Dully result {:#?}", dully_output);

            let learning_dully_eval = MinimizeMapeKpiEvaluator::default();
            let mut learning_dully: LearningDully<_, MinimizeMapeKpiEvaluator> = LearningDully::with(&learning_dully_eval);
            let mut learning_dully_driver = Driver::with(&mut learning_dully, &learning_dully_eval);
            let learning_dully_output = learning_dully_driver.drive_full(immos.iter().collect::<Vec<_>>().as_mut_slice()).unwrap();

            println!("Learning dully result {:#?}", learning_dully_output);

            prop_assert!(KpiOutput::from(learning_dully_output).mean_absolute_percentage_error - dully_output.mean_absolute_percentage_error < 1e-3);
        }
    }
}
