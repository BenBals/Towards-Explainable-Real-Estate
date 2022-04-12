use core::f64;
use std::borrow::{Borrow, BorrowMut};

// should partition data to training and test data
// initiate the training and evaluate the training on test data
// should be ease to modify the underlying algorithm for training
// evaluation should be equal for all approaches
use chrono::NaiveDate;
use common::{BpError, BpResult, Immo};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use serde::Deserialize;
use std::cmp::Ordering;

use crate::{Evaluator, Predictor};

const RNG_SEED: [u8; 16] = *b"0123456789abcdef";
pub const TRAINING_RATIO: f64 = 0.8;

/// Determines which parts of the data, if any, are cleaned.
#[derive(Clone, Copy, Debug, Deserialize)]
pub enum CleaningStrategy {
    /// Clean all data, both training and testing data
    All,
    /// Only clean the testing data, not the training data
    TestingOnly,
    /// Clean none of the data, neither the training nor the testing data
    None,
}

/// This function will perturb the given slice into two parts.
///
/// # Returns
/// The first returned slice contains about 80% randomly selected elements in a random order from the original slice.
/// The second slice contains all the remaining elemnts, *without* a guaratee about there order.
fn split_data_randomly<T>(data: &mut [T]) -> (&mut [T], &mut [T]) {
    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let training_amount = (TRAINING_RATIO * (data.len() as f64)) as usize;
    data.partial_shuffle(&mut rng, training_amount)
}

/// Order the dates such that `None`s are the "earliest" and then all `Some` values in their natural
/// order.
fn order_naive_date_options(a: Option<NaiveDate>, b: Option<NaiveDate>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (Some(date_a), Some(date_b)) => date_a.cmp(&date_b),
    }
}

/// Perturb the given slice into two parts.
/// # Returns
/// - two sub-slices such that the [Immo]s in the first part are as early or earlier than those in
/// the right part. The first part is `ratio` of the whole slice long.
pub fn split_by_date_percentage<'i, 'j>(
    data: &'j mut [&'i Immo],
    ratio: f64,
) -> (&'j mut [&'i Immo], &'j mut [&'i Immo]) {
    if data.len() <= 1 {
        return (data, &mut []);
    }

    let split_index = (data.len() as f64 * ratio).floor() as usize;
    data.select_nth_unstable_by(split_index, |a, b| {
        order_naive_date_options(a.wertermittlungsstichtag, b.wertermittlungsstichtag)
    });

    data.split_at_mut(split_index)
}

pub fn split_by_hash_key<'i, 'j>(
    data: &'j mut [&'i Immo],
    ratio: f64,
    seed: &str,
) -> (&'j mut [&'i Immo], &'j mut [&'i Immo]) {
    if data.len() <= 1 {
        return (data, &mut []);
    }

    let split_index = (data.len() as f64 * ratio).floor() as usize;
    data.select_nth_unstable_by_key(split_index, |a| {
        (a.deterministic_index(seed), a.id.to_string())
    });

    data.split_at_mut(split_index)
}

/// This function will perturb the given slice into two parts.
///
/// # Returns
/// The first slice contains all [Immo]s whose `wertermittlungsstichtag` is unset or before the
/// `data`
/// The second slice contains all the remaining elements, *without* a guaratee about there order.
pub fn split_data_at_date<'i, 'j>(
    data: &'j mut [&'i Immo],
    date: NaiveDate,
) -> (&'j mut [&'i Immo], &'j mut [&'i Immo]) {
    data.sort_unstable_by(|a, b| {
        order_naive_date_options(a.wertermittlungsstichtag, b.wertermittlungsstichtag)
    });

    let mut split_index: usize = 0;
    while split_index < data.len() {
        if data[split_index].wertermittlungsstichtag.is_some()
            && data[split_index].wertermittlungsstichtag.unwrap() >= date
        {
            break;
        }
        split_index += 1;
    }
    data.split_at_mut(split_index)
}

#[derive(Copy, Clone, Debug)]
struct TwoSplitStrategy {
    training_before: NaiveDate,
    testing_after: NaiveDate,
}

enum SplitStrategy {
    Randomly,
    AtDate(NaiveDate),
    AtTwoDates(TwoSplitStrategy),
    ByHashKey(String),
}

/// This struct can be used to compose a predictor and a evaluator.
/// The main function on this struct is [drive](Driver::drive),
/// which will evaluate a predictor on a given dataset.
pub struct Driver<'p, 'e, P: Predictor, E: Evaluator> {
    predictor: &'p mut P,
    evaluator: &'e E,
    split_strategy: SplitStrategy,
    predict_all: bool,
    cleaning: CleaningStrategy,
}

impl<'p, 'e, P: Predictor, E: Evaluator> Driver<'p, 'e, P, E> {
    /// This creates a new Driver from the given `predictor` and `evaluator`.
    pub fn with(predictor: &'p mut P, evaluator: &'e E) -> Self {
        Self {
            predictor,
            evaluator,
            split_strategy: SplitStrategy::Randomly,
            predict_all: false,
            cleaning: CleaningStrategy::All,
        }
    }

    /// Set the [CleaningStrategy] used
    pub fn set_cleaning(&mut self, cleaning: CleaningStrategy) {
        self.cleaning = cleaning;
    }

    fn split_data<'i, 'j>(
        &self,
        data: &'j mut [&'i Immo],
    ) -> (&'j mut [&'i Immo], &'j mut [&'i Immo]) {
        match &self.split_strategy {
            SplitStrategy::Randomly => split_data_randomly(data),
            SplitStrategy::AtDate(date) => split_data_at_date(data, date.clone()),
            SplitStrategy::AtTwoDates(strategy) => {
                let (too_much_training, testing) = split_data_at_date(data, strategy.testing_after);
                let (training, _too_new_training) =
                    split_data_at_date(too_much_training, strategy.training_before);
                (training, testing)
            }
            SplitStrategy::ByHashKey(seed) => split_by_hash_key(data, TRAINING_RATIO, &seed),
        }
    }

    fn drive_inner(
        &mut self,
        mut training_data: Vec<Immo>,
        mut validation_data: Vec<Immo>,
    ) -> BpResult<E::Output> {
        for immo in training_data.iter_mut() {
            immo.clear_aggregates();
        }

        if self.predict_all {
            validation_data = training_data
                .iter()
                .chain(validation_data.iter())
                .cloned()
                .collect();
        }

        let mut to_predict: Vec<Immo> = validation_data.clone();

        for immo in to_predict.iter_mut() {
            immo.clear_aggregates();
            immo.clear_price_information();
        }

        log::info!("training (n={})...", training_data.len());
        self.predictor
            .borrow_mut()
            .train(&training_data)
            .map_err(BpError::rethrow_with("Training failed"))?;
        log::info!("training... DONE");

        log::info!("predicting (n={})...", to_predict.len());
        self.predictor
            .predict(to_predict.iter_mut())
            .map_err(BpError::rethrow_with("Prediction failed"))?;
        log::info!("predicting... DONE");

        log::info!("evaluating...");
        let res = self
            .evaluator
            .borrow()
            .evaluate(validation_data.iter().zip(to_predict.iter()))
            .map_err(BpError::rethrow_with("Evaluation failed"));
        log::info!("evaluating... DONE");
        res
    }

    /// This function randomly splits the `data` in test data and validation data.
    /// The predictor is then trained on the training set.
    /// All immos will be cleaned of aggregates.
    /// Then the predictor will be asked to predict each Immo in the validation dataset.
    /// It will get a immo without price information (see [Immo::clear_price_information]).
    /// The pairs of the real Immo and predicted Immo will then be provided to the evaluator,
    /// which in turn gives how "good" the prediction was.
    ///
    /// The shuffling and splitting will be done inplace.
    ///
    /// This function does not attempt to prevent any errors or panics in the underlying predictor and evaluator.
    /// **It is the callers responsibility to ensure that they don't error out.**
    ///
    /// Currently around 80% of data will be used as training data and the remaining 20% will be used as validation data.
    /// To split the data in validation and training data a fixed seed is used to ensure determinability.
    ///
    /// This function can be called multiple times.
    /// The predictor will only get provided data from the currently supplied `data`.
    /// However the predictor may hold references to Immos of previous training sets.
    /// *Make sure that is what you want.*
    /// Furthermore you should make sure that the predictor does not have direct price information about the current `data`
    /// from previous calls.
    pub fn drive(&mut self, data: &mut [&Immo]) -> BpResult<E::Output> {
        log::info!("Preparing data...DONE");
        let (training_data, validation_data) = self.split_data(data);

        let training_data: Vec<_> = training_data
            .iter()
            .filter(|immo| {
                if matches!(self.cleaning, CleaningStrategy::All) {
                    (**immo).has_realistic_values_in_default_ranges()
                } else {
                    true
                }
            })
            .map(|&immo| immo.clone())
            .collect();
        let validation_data: Vec<_> = validation_data
            .iter()
            .filter(|immo| {
                if matches!(
                    self.cleaning,
                    CleaningStrategy::All | CleaningStrategy::TestingOnly
                ) {
                    (**immo).has_realistic_values_in_default_ranges()
                } else {
                    true
                }
            })
            .map(|&immo| immo.clone())
            .collect();

        self.drive_inner(training_data, validation_data)
    }

    /// Like [drive], but all data will be used for testing *as well as* validation.
    /// This does allow cheating, but might be what you want.
    pub fn drive_full(&mut self, data: &[&Immo]) -> BpResult<E::Output> {
        log::info!("Preparing data...DONE");
        let mut training_data: Vec<_> = data.iter().map(|&immo| immo.clone()).collect();
        let mut validation_data = training_data.clone();

        let mut rng = XorShiftRng::from_seed(RNG_SEED);
        training_data.shuffle(&mut rng);
        validation_data.shuffle(&mut rng);

        self.drive_inner(training_data, validation_data)
    }

    /// Split into training and validation data at the test date. That is all [Immos] whose
    /// `wertermittlungsstichtag` is unset or before `date` are training data. All others are used
    /// for validation.
    /// If a training is given, only properties older than that date are used for training.
    /// This only affects future calles to [drive].
    pub fn split_at_dates(&mut self, testing_after: NaiveDate, training_before: Option<NaiveDate>) {
        if let Some(training_before) = training_before {
            self.split_strategy = SplitStrategy::AtTwoDates(TwoSplitStrategy {
                training_before,
                testing_after,
            });
        } else {
            self.split_strategy = SplitStrategy::AtDate(testing_after);
        }
    }

    pub fn split_by_hash_key(&mut self, seed: String) {
        self.split_strategy = SplitStrategy::ByHashKey(seed)
    }

    /// By default, the predictor is only asked to predict the validation data.
    /// If you pass `true` to this function, all **future** calls to [drive] pass all data to the
    /// predictor.
    pub fn predict_all(&mut self, should: bool) {
        self.predict_all = should;
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::cmp::Ordering;

    use proptest::prelude::*;

    use super::*;
    use common::{Immo, Trainable};
    use test_helpers::*;

    use crate::{Evaluator, Predictor};

    #[derive(Debug, Clone)]
    struct TestPredictor {
        training_size: Option<usize>,
        test_size: Cell<usize>,
        called_clear_price_information: Cell<bool>,
        called_training_before_prediction: Cell<bool>,
    }

    impl TestPredictor {
        fn new() -> Self {
            TestPredictor {
                training_size: None,
                test_size: Cell::new(0),
                called_clear_price_information: Cell::new(true),
                called_training_before_prediction: Cell::new(true),
            }
        }
    }

    impl Trainable for TestPredictor {
        fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
            self.training_size = Some(training_data.into_iter().count());
            Ok(())
        }
    }

    impl Predictor for TestPredictor {
        fn predict<'i>(
            &self,
            validation_data: impl IntoIterator<Item = &'i mut Immo>,
        ) -> BpResult<()> {
            for datum in validation_data {
                self.test_size.set(self.test_size.get() + 1);

                let mut clone = datum.clone();
                clone.clear_price_information();
                if clone != *datum {
                    self.called_clear_price_information.set(false);
                }

                if self.training_size.is_none() {
                    self.called_training_before_prediction.set(false);
                    return Err("Training was not called before predict".into());
                }
            }

            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct TestEvaluator;

    impl Evaluator for TestEvaluator {
        type Output = (usize, bool);

        fn evaluate<'i>(
            &self,
            pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
        ) -> BpResult<Self::Output> {
            let vec_pairs: Vec<_> = pairs.into_iter().collect();
            Ok((
                vec_pairs.len(),
                vec_pairs
                    .iter()
                    .all(|(real, predicted)| real.id() == predicted.id()),
            ))
        }
    }

    proptest! {
        #[test]
        fn test_size_is_evaluation_size(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            let (eval_size, _) = driver.drive_inner(immos.clone(), immos).unwrap();
            prop_assert_eq!(eval_size, driver.predictor.test_size.get());
        }

        #[test]
        fn object_ids_match(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            let (_, object_ids_match) = driver.drive_inner(immos.clone(), immos).unwrap();

            prop_assert!(object_ids_match);
        }

        #[test]
        fn predictor_get_no_marktwert(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            let _ = driver.drive_inner(immos.clone(), immos);

            prop_assert!(driver.predictor.called_clear_price_information.get());
        }

        #[test]
        fn called_train_before_predict(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            let _ = driver.drive_inner(immos.clone(), immos);

            prop_assert!(driver.predictor.called_training_before_prediction.get());
        }

        #[test]
        fn drive_splits_data(immos in prop::collection::vec(full_immo(), 5..128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            driver.drive(&mut *immos.iter().collect::<Vec<_>>()).unwrap();

            prop_assert!(driver.predictor.training_size.unwrap() < immos.len());
            prop_assert!(driver.predictor.test_size.get() < immos.len());
        }

        #[test]
        fn drive_uses_all_data(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            driver.set_cleaning(CleaningStrategy::None);
            driver.drive(&mut *immos.iter().collect::<Vec<_>>()).unwrap();

            prop_assert!(driver.predictor.training_size.unwrap() + driver.predictor.test_size.get() == immos.len());
        }

        #[test]
        fn drive_full_gives_all_data_to_training(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            driver.drive_full(&*immos.iter().collect::<Vec<_>>()).unwrap();

            prop_assert!(driver.predictor.training_size.unwrap() == immos.len());
        }

        #[test]
        fn drive_full_gives_all_data_to_prediction(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            driver.drive_full(&immos.iter().collect::<Vec<_>>()).unwrap();

            prop_assert!(driver.predictor.test_size.get() == immos.len());
        }

        #[test]
        fn drive_full_gives_all_data_to_evaluation(immos in full_immos(128)) {
            let mut predictor = TestPredictor::new();
            let mut driver = Driver::with(&mut predictor, &TestEvaluator);
            let (eval_size, _) = driver.drive_full(&immos.iter().collect::<Vec<_>>()).unwrap();

            prop_assert!(eval_size == immos.len());
        }

        #[test]
        fn split_data_randomly_retains_all_data(mut data in prop::collection::vec(prop::num::usize::ANY, 0.. 256)) {
            data.sort_unstable();
            let copy = data.clone();
            let (training, validation) = split_data_randomly(data.as_mut_slice());

            let mut all_data : Vec<_> = training.iter().chain(validation.iter()).copied().collect();
            all_data.sort_unstable();
            prop_assert_eq!(copy, all_data);
        }

        #[test]
        fn split_data_at_date_test(data in prop::collection::vec(full_immo(), 0.. 256), date in naive_date()) {
            let start_len = data.len();
            let mut refs: Vec<&Immo> = data.iter().collect();
            let (training, validation) = split_data_at_date(refs.as_mut_slice(), date);

            prop_assert_eq!(start_len, training.iter().chain(validation.iter()).count());

            prop_assert!(training.iter().all(|immo| immo.wertermittlungsstichtag.is_none() || immo.wertermittlungsstichtag.unwrap() < date));
            prop_assert!(validation.iter().all(|immo| immo.wertermittlungsstichtag.is_some() && immo.wertermittlungsstichtag.unwrap() >= date));
        }

        #[test]
        fn split_data_ratio_matches(mut data in prop::collection::vec(prop::num::usize::ANY, 0.. 256)) {
            let (training, validation) = split_data_randomly(data.as_mut_slice());
            let all = training.len() + validation.len();

            prop_assert!(training.len() as f64 - 1.0 < (all as f64) * TRAINING_RATIO);
            prop_assert!(training.len() as f64 + 1.0 > (all as f64) * TRAINING_RATIO);
        }

        #[test]
        fn split_data_by_date_percentage_ratio_matches(data in prop::collection::vec(full_immo(), 0.. 256)) {
            let mut refs: Vec<&Immo> = data.iter().collect();
            let (training, validation) = split_by_date_percentage(refs.as_mut_slice(), TRAINING_RATIO);
            let all = training.len() + validation.len();

            prop_assert!(training.len() as f64 - 1.0 < (all as f64) * TRAINING_RATIO);
            prop_assert!(training.len() as f64 + 1.0 > (all as f64) * TRAINING_RATIO);
        }

        #[test]
        #[allow(clippy::unnecessary_lazy_evaluations)]
        fn split_data_by_date_percentage_left_earlier_right(data in prop::collection::vec(full_immo(), 2..256)) {
            let mut refs: Vec<&Immo> = data.iter().collect();
            let (training, testing) = split_by_date_percentage(refs.as_mut_slice(), TRAINING_RATIO);

            let maximum_training = training.iter().max_by(|a, b| {
                order_naive_date_options(a.wertermittlungsstichtag, b.wertermittlungsstichtag)
            });
            let minimum_testing = testing.iter().min_by(|a, b| {
                order_naive_date_options(a.wertermittlungsstichtag, b.wertermittlungsstichtag)
            });

            let ordering = order_naive_date_options(
                maximum_training.and_then(|&immo| immo.wertermittlungsstichtag),
                minimum_testing.and_then(|&immo| immo.wertermittlungsstichtag)
            );

            prop_assert!(matches!(ordering, Ordering::Less | Ordering::Equal))
        }
    }
}
