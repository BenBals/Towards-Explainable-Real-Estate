//! This module contains weights, and associated types and helpers for an individual representing
//! an enhanced Lp quasi-norm based similarity predictor.
use common::immo::META_DATA_COUNT;
use derive_builder::Builder;
use genevo::prelude::Rng;

use super::bounded::Bounded;
use crate::fitness_functions::WeightedPredictor;
use crate::weight::{
    bounded,
    config::{CrossoverConfig, MutationConfig},
    either_weight::ActiveSide,
    filter_weight, EitherWeight, FilterWeight, OptionalWeight, Scalable, Weight,
};
use common::{BpResult, Immo};
use dissimilarities::filtered::{FilteredDissimilarity, NUM_FILTERS};
use dissimilarities::ScalingLpVectorDissimilarity;
use itertools::Either;
use predictors::weighted_average::{
    NeighborSelectionStrategy, NeighborsForAverageStrategy, WeightedAveragePredictorOptionsBuilder,
};
use predictors::WeightedAveragePredictor;
use std::fmt::{Debug, Formatter, Result};

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// A Weight which can be either a radius or a value of k for k-nearest-neighbor selection.
/// Should match with [NeighborSelectionStrategy].
pub struct RadiusOrKNearest(EitherWeight<Bounded<f64>, Bounded<i64>>);

/// An Output struct which is used for the subtraction of two RadiusOrKNearest
#[derive(Clone, Debug)]
pub struct RadiusOrKNearestSubOutput(
    <EitherWeight<bounded::Bounded<f64>, bounded::Bounded<i64>> as Weight>::SubOutput,
);

impl Scalable for RadiusOrKNearestSubOutput {
    fn scale(self, scalar: f64) -> Self {
        RadiusOrKNearestSubOutput(self.0.scale(scalar))
    }
}

impl RadiusOrKNearest {
    /// creates a new RadiusOrKNearest Weight with the given attributes
    pub fn new(radius: Bounded<f64>, k: Bounded<i64>, active: ActiveSide) -> Self {
        Self(EitherWeight::new(radius, k, active))
    }
}

impl From<&RadiusOrKNearest> for NeighborSelectionStrategy {
    fn from(weight: &RadiusOrKNearest) -> Self {
        match weight.0.active_value() {
            Either::Left(radius) => NeighborSelectionStrategy::Radius(radius.value()),
            Either::Right(k) => NeighborSelectionStrategy::KNearest(k.value() as u64),
        }
    }
}

impl Debug for RadiusOrKNearest {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        NeighborSelectionStrategy::from(self).fmt(f)
    }
}

impl Weight for RadiusOrKNearest {
    type SubOutput = RadiusOrKNearestSubOutput;

    fn mutate<Rn>(&mut self, rng: &mut Rn, config: &MutationConfig)
    where
        Rn: Rng + Sized,
    {
        self.0.mutate(rng, config)
    }

    fn crossover<Rn>(&self, other: &Self, rng: &mut Rn, config: &CrossoverConfig) -> Self
    where
        Rn: Rng + Sized,
    {
        Self(self.0.crossover(&other.0, rng, config))
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        self.0.normalized_distance(&other.0)
    }

    fn regenerate<Rn>(&mut self, rng: &mut Rn)
    where
        Rn: Rng + Sized,
    {
        self.0.regenerate(rng)
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        let mut new_self = *self;
        new_self.0 = new_self.0.add(&other.0);
        new_self
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        Self::SubOutput {
            0: self.0.sub(&other.0),
        }
    }
}

/// Our Genotype. A [LpWapWeight] object represents weights for weighting [Immo::meta_data_array]
/// The exponent is used for [ScalingLpVectorDissimilarity].
#[derive(Clone, Debug, PartialEq, Builder, serde::Serialize, serde::Deserialize)]
pub struct LpWapWeight {
    /// The \(p\) in the \(L_p\) quasi-norm
    exponent: Bounded<f64>,
    /// How should properties be preselected for calculating similarity to?
    radius_or_k_nearest: RadiusOrKNearest,
    /// Should only a limited number of very similar properties affect the weighted average?
    k_most_similar_only: OptionalWeight<Bounded<i64>>,
    /// The weights inside the weighted \(L_p\) quasi norm
    lp_weights: Vec<Bounded<f64>>,
    /// Filters to apply before calculating the quasi norm. See [FilterWeight].
    pub filters: FilterWeight,
    /// If true, no filters will be applied
    pub ignore_filters: bool,
    /// Never allow comparision objects outside this radius, even if selecting by nearest neighbors.
    pub strict_radius_limit: Option<f64>,
}

/// An Output struct which is used for the subtraction of two LpWapWeights
#[derive(Clone, Debug)]
pub struct LpWapWeightSubOutput {
    lp_weights: <Vec<bounded::Bounded<f64>> as Weight>::SubOutput,
    exponent: <bounded::Bounded<f64> as Weight>::SubOutput,
    radius_or_k_nearest: <RadiusOrKNearest as Weight>::SubOutput,
    k_most_similar_only: Option<i64>,
    filters: <filter_weight::FilterWeight as Weight>::SubOutput,
}

impl Scalable for LpWapWeightSubOutput {
    fn scale(self, scalar: f64) -> Self {
        Self {
            lp_weights: self.lp_weights.scale(scalar),
            exponent: self.exponent.scale(scalar),
            radius_or_k_nearest: self.radius_or_k_nearest.scale(scalar),
            k_most_similar_only: self.k_most_similar_only.scale(scalar),
            filters: self.filters.scale(scalar),
        }
    }
}

impl LpWapWeight {
    /// Creates a new [LpWapWeight] object from the given attributes.
    pub fn new(
        exponent: Bounded<f64>,
        radius_or_k_nearest: RadiusOrKNearest,
        k_most_similar_only: OptionalWeight<Bounded<i64>>,
        lp_weights: Vec<Bounded<f64>>,
        filters: FilterWeight,
    ) -> Self {
        assert!(
            exponent.value() >= 1e-9,
            "exponent must not be smaller or nearly equal to zero"
        );

        Self {
            exponent,
            radius_or_k_nearest,
            k_most_similar_only,
            lp_weights,
            filters,
            ignore_filters: false,
            strict_radius_limit: None,
        }
    }

    /// Never allow k_most_similar to exceed this bound
    pub fn set_k_most_similar_upper_bound(&mut self, bound: i64) {
        let seed = Bounded::new(5.min(bound), 1.min(bound), bound);
        self.k_most_similar_only = OptionalWeight::new_always_some(seed, seed);
    }

    /// Always keep the exponent at the given fixed value
    pub fn fix_exponent(&mut self, value: f64) {
        self.exponent = Bounded::new(value, value, value);
    }
}

impl Weight for LpWapWeight {
    type SubOutput = LpWapWeightSubOutput;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        self.exponent.mutate(rng, config);
        self.radius_or_k_nearest.mutate(rng, config);
        self.lp_weights.mutate(rng, config);
        self.k_most_similar_only.mutate(rng, config);
        if !self.ignore_filters {
            self.filters.mutate(rng, config);
        }
    }

    /// This will crossover the `lp_weights`.
    /// ALl other values are taken from self.
    fn crossover<R>(&self, other: &Self, rng: &mut R, config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        LpWapWeightBuilder::default()
            .exponent(self.exponent)
            .radius_or_k_nearest(self.radius_or_k_nearest.crossover(
                &other.radius_or_k_nearest,
                rng,
                config,
            ))
            .k_most_similar_only(self.k_most_similar_only.crossover(
                &other.k_most_similar_only,
                rng,
                config,
            ))
            .lp_weights(self.lp_weights.crossover(&other.lp_weights, rng, config))
            .filters(self.filters.crossover(&other.filters, rng, config))
            .ignore_filters(self.ignore_filters)
            .strict_radius_limit(self.strict_radius_limit)
            .build()
            .unwrap()
    }

    /// This will regenerate all constituent weights individually.
    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        self.exponent.regenerate(rng);
        self.radius_or_k_nearest.regenerate(rng);
        self.lp_weights.regenerate(rng);
        self.k_most_similar_only.regenerate(rng);
        if !self.ignore_filters {
            self.filters.regenerate(rng);
        }
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        let total_weight = NUM_FILTERS + META_DATA_COUNT
            + 1 // exponent
            + 1 // neighbor_selection
            + 1; // k_most_similar_only

        let individual_distances = vec![
            self.exponent.normalized_distance(&other.exponent),
            self.radius_or_k_nearest
                .normalized_distance(&other.radius_or_k_nearest),
            self.k_most_similar_only
                .normalized_distance(&other.k_most_similar_only),
            self.lp_weights.normalized_distance(&other.lp_weights) * META_DATA_COUNT as f64,
            self.filters.normalized_distance(&other.filters) * NUM_FILTERS as f64,
        ];

        individual_distances.iter().sum::<f64>() / total_weight as f64
    }
    fn add(&self, other: &Self::SubOutput) -> Self {
        Self {
            lp_weights: self.lp_weights.add(&other.lp_weights),
            exponent: self.exponent.add(&other.exponent),
            radius_or_k_nearest: self.radius_or_k_nearest.add(&other.radius_or_k_nearest),
            k_most_similar_only: self.k_most_similar_only.add(&other.k_most_similar_only),
            filters: self.filters.add(&other.filters),
            ignore_filters: self.ignore_filters,
            strict_radius_limit: self.strict_radius_limit,
        }
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        Self::SubOutput {
            lp_weights: self.lp_weights.sub(&other.lp_weights),
            exponent: self.exponent.sub(&other.exponent),
            radius_or_k_nearest: self.radius_or_k_nearest.sub(&other.radius_or_k_nearest),
            k_most_similar_only: self.k_most_similar_only.sub(&other.k_most_similar_only),
            filters: self.filters.sub(&other.filters),
        }
    }
}

impl Default for LpWapWeight {
    fn default() -> Self {
        let filters = FilterWeight::all_none();
        Self {
            exponent: Bounded::new(2.0, 0.1, 5.0),
            radius_or_k_nearest: RadiusOrKNearest::new(
                Bounded::new(3.0 * 1000.0, 10.0, 10.0 * 1000.0), // radius
                Bounded::new(100, 1, 2000),                      // k_nearest
                ActiveSide::Right,
            ),
            lp_weights: vec![Bounded::new(0.0, 0.0, 1.0); META_DATA_COUNT],
            k_most_similar_only: OptionalWeight::new(Bounded::new(5, 1, 500), None),
            filters,
            ignore_filters: false,
            strict_radius_limit: None,
        }
    }
}

impl WeightedPredictor<LpWapWeight>
    for WeightedAveragePredictor<FilteredDissimilarity<ScalingLpVectorDissimilarity>>
{
    fn predict_with_weight<'j>(
        &self,
        validation_data: impl IntoIterator<Item = &'j mut Immo>,
        weight: &LpWapWeight,
    ) -> BpResult<()> {
        let mut unfiltered_dissimilarity = self.options.dissimilarity.inner.clone();
        unfiltered_dissimilarity.weights =
            Bounded::vec_try_into::<[f64; META_DATA_COUNT]>(&weight.lp_weights).unwrap();
        unfiltered_dissimilarity.exponent = weight.exponent.value();

        let dissimilarity = weight.filters.wrap_dissimilarity(unfiltered_dissimilarity);

        let mut options_builder = WeightedAveragePredictorOptionsBuilder::default();

        match NeighborSelectionStrategy::from(&weight.radius_or_k_nearest) {
            NeighborSelectionStrategy::Radius(radius) => {
                options_builder = options_builder.radius(radius);
            }
            NeighborSelectionStrategy::KNearest(k) => {
                options_builder = options_builder.k_nearest(k);
            }
        }

        let options = options_builder
            .dissimilarity(dissimilarity)
            .strict_radius_limit(weight.strict_radius_limit)
            .neighbors_for_average_strategy(
                NeighborsForAverageStrategy::k_most_similar_only_from_option(
                    weight
                        .k_most_similar_only
                        .value()
                        .map(|bounded| bounded.value() as u64),
                ),
            )
            .median(self.options.median)
            .build()
            .unwrap();

        self.predict_with_options(validation_data, &options)
    }
}
