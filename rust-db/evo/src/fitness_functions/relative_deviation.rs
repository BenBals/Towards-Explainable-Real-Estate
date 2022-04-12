use crate::weight::{TrivialWeightWrapper, Weight};
use common::{database, BpResult, Immo, Normalizer, Trainable};
use evaluators::kpi::{MinimizeMapeInvertedOrdKpiEvaluator, MinimizeMapeInvertedOrdKpiOutput};
use genevo::prelude::FitnessFunction;
use normalizers::HpiNormalizer;
use predictions::Evaluator;
use predictors::NormalizingPredictor;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::fmt;
use std::marker::PhantomData;

/// This is variant of [Predictor] where the predictor may accept a [Weight] to parametrize its
/// behavior by
pub trait WeightedPredictor<W: Weight>: Trainable {
    /// Given a [Weight], make predictions for the given validation_data
    fn predict_with_weight<'j>(
        &self,
        validation_data: impl IntoIterator<Item = &'j mut Immo>,
        weight: &W,
    ) -> BpResult<()>;
}

impl<N: Normalizer, W: Weight, P: WeightedPredictor<W>> WeightedPredictor<W>
    for NormalizingPredictor<N, P>
{
    fn predict_with_weight<'j>(
        &self,
        validation_data: impl IntoIterator<Item = &'j mut Immo>,
        weight: &W,
    ) -> BpResult<()> {
        let mut to_predict: Vec<_> = validation_data.into_iter().collect();
        let res = self
            .inner
            .predict_with_weight(to_predict.iter_mut().map(|i| &mut **i), weight);

        self.normalizer
            .denormalize(to_predict.iter_mut().map(|immo| &mut **immo));

        res
    }
}

/// This struct contains everything necessary to calculate the fitness of a [Weight] using a
/// [WeightedPredictor]. Fitness is the MAPE.
pub struct RelativeDeviationFitnessFunction<W: Weight, P: WeightedPredictor<W>> {
    weighted_predictor: P,
    validation_data: Vec<Immo>,
    training_data: Option<Vec<Immo>>,
    batch_size: usize,
    normalizer: HpiNormalizer,
    write_to_mongo: Option<(String, String)>,
    _weight_marker: PhantomData<W>,
}

impl<W: Weight, P: WeightedPredictor<W>> RelativeDeviationFitnessFunction<W, P> {
    /// Creates a new [SquaredDeviationFitnessFunction] with a new .
    /// Training and validation data should not be normalized.
    /// # Returns
    /// Err iff training fails on the normalizer, [ScalingLpVectorDissimilarity], or [WeightedAveragePredictor].
    pub fn new(
        mut training_data: Vec<Immo>,
        mut validation_data: Vec<Immo>,
        batch_size: usize,
        mut weighted_predictor: P,
        write_to_mongo: Option<(String, String)>,
    ) -> BpResult<Self> {
        let mut normalizer = HpiNormalizer::default();
        normalizer.train(&training_data)?;
        normalizer.normalize(&mut training_data);
        normalizer.normalize(&mut validation_data);

        weighted_predictor.train(training_data.iter())?;

        let training_data = if write_to_mongo.is_some() {
            Some(training_data)
        } else {
            None
        };

        Ok(Self {
            validation_data,
            training_data,
            batch_size,
            normalizer,
            weighted_predictor,
            write_to_mongo,
            _weight_marker: PhantomData::default(),
        })
    }

    /// Given a Weight object, this function will do a big part of the fitness function.
    /// First a [ScalingLpVectorDissimilarity] with the given Weight object is created.
    /// It uses the pretrained dissimilarity, created in [LpSquaredDeviationFitnessFunction::new].
    /// Then `self.batch_size` [Immo]s are chosen randomly from `self.validation_data` and the pretrained
    /// [WeightedAveragePredictor] (`self.predictor`) is used to predict the chosen [Immo]s.
    /// The resulting [Immo]s are then denormalized using `self.normalizer` and are returned in pairs
    /// `(real, predicted)`.
    ///
    fn prediction_pairs(&self, weight: &W) -> Vec<(Immo, Immo)> {
        self.prediction_pairs_for_immos(weight, &self.validation_data, self.batch_size)
    }
    fn prediction_pairs_for_immos(
        &self,
        weight: &W,
        validation_data: &Vec<Immo>,
        batch_size: usize,
    ) -> Vec<(Immo, Immo)> {
        let mut real_immos: Vec<_> = validation_data
            .choose_multiple(&mut thread_rng(), batch_size)
            .cloned()
            .collect();

        let mut predicted_immos: Vec<_> = real_immos.clone();

        for immo in &mut predicted_immos {
            immo.clear_price_information();
        }

        self.weighted_predictor
            .predict_with_weight(&mut predicted_immos, weight)
            .unwrap_or_else(|error| {
                panic!("prediction for fitness did not work: {}", error);
            });

        self.normalizer.denormalize(&mut real_immos);
        self.normalizer.denormalize(&mut predicted_immos);
        real_immos.iter().cloned().zip(predicted_immos).collect()
    }
}

impl<W, P> FitnessFunction<TrivialWeightWrapper<W>, MinimizeMapeInvertedOrdKpiOutput>
    for &RelativeDeviationFitnessFunction<W, P>
where
    W: Weight + Sync + Send + PartialEq,
    P: WeightedPredictor<W>,
{
    fn fitness_of(&self, weight: &TrivialWeightWrapper<W>) -> MinimizeMapeInvertedOrdKpiOutput {
        if let Some(database_params) = &self.write_to_mongo {
            let mut all_data = self
                .training_data
                .clone()
                .expect("Training data is always saved when we should write to mongo");
            all_data.append(&mut self.validation_data.clone());

            let all_immo_pairs =
                self.prediction_pairs_for_immos(&weight.0, &all_data, all_data.len());

            database::write_predictions_to_database(
                &database_params.0,
                &database_params.1,
                &all_immo_pairs,
            )
        }

        let immo_pairs = self.prediction_pairs(&weight.0);

        let evaluator = MinimizeMapeInvertedOrdKpiEvaluator::new();
        evaluator
            .evaluate(immo_pairs.iter().map(|(immo1, immo2)| (immo1, immo2)))
            .unwrap()
    }

    fn average(
        &self,
        values: &[MinimizeMapeInvertedOrdKpiOutput],
    ) -> MinimizeMapeInvertedOrdKpiOutput {
        MinimizeMapeInvertedOrdKpiOutput::average(values)
    }

    fn highest_possible_fitness(&self) -> MinimizeMapeInvertedOrdKpiOutput {
        MinimizeMapeInvertedOrdKpiOutput::best_possible()
    }

    fn lowest_possible_fitness(&self) -> MinimizeMapeInvertedOrdKpiOutput {
        MinimizeMapeInvertedOrdKpiOutput::worst_possible()
    }
}

/// Custom Debug implementation that only outputs the batch_size, because we don't want to see all [Immo]s.
impl<W: Weight, P: WeightedPredictor<W>> fmt::Debug for RelativeDeviationFitnessFunction<W, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LpSquaredDeviationFitnessFunction with batch_size {}", {
            self.batch_size
        })?;
        Ok(())
    }
}
