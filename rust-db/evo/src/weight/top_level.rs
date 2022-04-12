use crate::fitness_functions::WeightedPredictor;
use crate::weight::{
    config::{CrossoverConfig, MutationConfig},
    traits::Scalable,
    ExpertWeight, LpWapWeight, Weight,
};
use common::{BpResult, Immo, Trainable};
use dissimilarities::filtered::FilteredDissimilarity;
use dissimilarities::{ExpertDissimilarity, ScalingLpVectorDissimilarity};
use genevo::prelude::Rng;
use predictors::WeightedAveragePredictor;

/// A weight that wraps our two types of weights that are directly optimized by the evo.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TopLevelWeight {
    /// Learn an [ExpertWeight]
    Expert(ExpertWeight),
    /// Learn an [LpWapWeight]
    Lp(LpWapWeight),
}

/// An Output struct which is used for the subtraction of two TopLevelWeights
#[derive(Clone, Debug)]
pub enum TopLevelWeightSubOutput {
    /// Result of substracting two [TopLevelWeight::Expert] values.
    Expert(<ExpertWeight as Weight>::SubOutput),
    /// Result of substracting two [TopLevelWeight::Lp] values.
    Lp(<LpWapWeight as Weight>::SubOutput),
}

impl Default for TopLevelWeight {
    fn default() -> Self {
        Self::Lp(LpWapWeight::default())
    }
}

impl Scalable for TopLevelWeightSubOutput {
    fn scale(self, scalar: f64) -> Self {
        match self {
            Self::Expert(inner) => Self::Expert(inner.scale(scalar)),
            Self::Lp(inner) => Self::Lp(inner.scale(scalar)),
        }
    }
}

impl Weight for TopLevelWeight {
    type SubOutput = Option<TopLevelWeightSubOutput>;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        match self {
            Self::Expert(inner) => inner.mutate(rng, config),
            Self::Lp(inner) => inner.mutate(rng, config),
        }
    }

    fn crossover<R>(&self, other: &Self, rng: &mut R, config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        match (self, other) {
            (Self::Expert(inner), Self::Expert(other_inner)) => {
                Self::Expert(inner.crossover(other_inner, rng, config))
            }
            (Self::Lp(inner), Self::Lp(other_inner)) => {
                Self::Lp(inner.crossover(other_inner, rng, config))
            }
            _ => {
                panic!("Attempted to cross ExpertWeight with LpWapWeight");
            }
        }
    }

    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        match self {
            Self::Expert(inner) => inner.regenerate(rng),
            Self::Lp(inner) => inner.regenerate(rng),
        }
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        match (self, other) {
            (Self::Expert(inner), Self::Expert(other_inner)) => {
                inner.normalized_distance(other_inner)
            }
            (Self::Lp(inner), Self::Lp(other_inner)) => inner.normalized_distance(other_inner),
            _ => {
                panic!(
                    "Attempted calculate normalizedh distance between ExpertWeight and LpWapWeight"
                );
            }
        }
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        match (self, other) {
            (
                Self::Expert(expert_weight),
                Self::SubOutput::Some(TopLevelWeightSubOutput::Expert(expert_output)),
            ) => Self::Expert(expert_weight.add(expert_output)),
            (
                Self::Lp(lp_weight),
                Self::SubOutput::Some(TopLevelWeightSubOutput::Lp(lp_output)),
            ) => Self::Lp(lp_weight.add(lp_output)),
            (_, _) => self.clone(),
        }
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        match (self, other) {
            (Self::Expert(self_expert), Self::Expert(other_expert)) => Self::SubOutput::Some(
                TopLevelWeightSubOutput::Expert(self_expert.sub(other_expert)),
            ),
            (Self::Lp(self_lp), Self::Lp(other_lp)) => {
                Self::SubOutput::Some(TopLevelWeightSubOutput::Lp(self_lp.sub(other_lp)))
            }
            (_, _) => None,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
/// A [WeightedPredictor] that can be optimized for by evolution.
pub enum EvoPredictor {
    /// Predicts using an [ExpertDissimilarity]
    Expert(WeightedAveragePredictor<ExpertDissimilarity>),
    /// Predicts using an [FilteredDissimilarity<ScalingLpVectorDissimilarity>]
    Lp(WeightedAveragePredictor<FilteredDissimilarity<ScalingLpVectorDissimilarity>>),
}

impl Trainable for EvoPredictor {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        match self {
            Self::Expert(predictor) => predictor.train(training_data),
            Self::Lp(predictor) => predictor.train(training_data),
        }
    }
}

impl WeightedPredictor<TopLevelWeight> for EvoPredictor {
    fn predict_with_weight<'j>(
        &self,
        validation_data: impl IntoIterator<Item = &'j mut Immo>,
        weight: &TopLevelWeight,
    ) -> BpResult<()> {
        match (self, weight) {
            (Self::Expert(predictor), TopLevelWeight::Expert(weight)) => {
                predictor.predict_with_weight(validation_data, weight)
            }
            (Self::Lp(predictor), TopLevelWeight::Lp(weight)) => {
                predictor.predict_with_weight(validation_data, weight)
            }
            _ => Err(format!(
                "Tried to use predictor {:?} to predict with invalid weight {:?}",
                self, weight
            )
            .into()),
        }
    }
}
