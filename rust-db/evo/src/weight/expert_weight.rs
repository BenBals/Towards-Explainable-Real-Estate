//! A weight that reperesents the learnable parts of an [ExpertDissimilarity]
use crate::fitness_functions::WeightedPredictor;
use crate::weight::{
    bounded,
    config::{CrossoverConfig, MutationConfig},
    Bounded, FilterWeight, Scalable, Weight,
};
use common::{BpResult, Immo};
use derive_builder::Builder;
use dissimilarities::expert::{UnfilteredExpertDissimilarityBuilder, NUM_FACTORS, NUM_WEIGHTS};
use dissimilarities::filtered::NUM_FILTERS;
use dissimilarities::ExpertDissimilarity;
use genevo::prelude::Rng;
use predictors::weighted_average::{
    KMostSimilarOnlyPerObjektunterartCategory, NeighborsForAverageStrategy,
};
use predictors::{
    weighted_average::WeightedAveragePredictorOptionsBuilder, WeightedAveragePredictor,
};

/// An Output struct which is used for the subtraction of two ExpertWeights
#[derive(Clone, Debug)]
pub struct ExpertWeightSubOutput {
    cutoff_similarity: <bounded::Bounded<f64> as Weight>::SubOutput,
    similarity_part_1_weight: <bounded::Bounded<f64> as Weight>::SubOutput,
    factors: <Vec<bounded::Bounded<f64>> as Weight>::SubOutput,
    weights: <Vec<bounded::Bounded<f64>> as Weight>::SubOutput,
    filters: <FilterWeight as Weight>::SubOutput,
    radius_km_filter: <bounded::Bounded<f64> as Weight>::SubOutput,
    k_most_similar_only: <Vec<bounded::Bounded<i64>> as Weight>::SubOutput,
}

impl Scalable for ExpertWeightSubOutput {
    fn scale(self, scale: f64) -> Self {
        Self {
            cutoff_similarity: self.cutoff_similarity.scale(scale),
            similarity_part_1_weight: self.similarity_part_1_weight.scale(scale),
            factors: self.factors.scale(scale),
            weights: self.weights.scale(scale),
            filters: self.filters.scale(scale),
            radius_km_filter: self.radius_km_filter.scale(scale),
            k_most_similar_only: self.k_most_similar_only.scale(scale),
        }
    }
}

/// This struct represents mutate-able version of an expert dissimilarity. It holds the same
/// parameters.
// Note to the implementor: You must *always* uphold that factors has the exact length of
// [NUM_FACTORS]
#[derive(Clone, Debug, PartialEq, Builder, serde::Serialize, serde::Deserialize)]
pub struct ExpertWeight {
    /// See [ExpertDissimilarity]
    cutoff_similarity: Bounded<f64>,
    /// See [ExpertDissimilarity]
    similarity_part_1_weight: Bounded<f64>,
    /// See [ExpertDissimilarity]
    factors: Vec<Bounded<f64>>,
    /// See [ExpertDissimilarity]
    weights: Vec<Bounded<f64>>,
    /// See [ExpertDissimilarity]
    filters: FilterWeight,
    /// See [ExpertDissimilarity]
    radius_km_filter: Bounded<f64>,
    /// See [ExpertDissimilarity]
    k_most_similar_only: Vec<Bounded<i64>>,
}

impl Default for ExpertWeight {
    fn default() -> Self {
        ExpertWeight {
            cutoff_similarity: Bounded::new(0.0, 0.0, 1.0),
            similarity_part_1_weight: Bounded::new(0.5, 0.0, 1.0),
            factors: vec![Bounded::new(1.0, 0.0, 100.0); NUM_FACTORS],
            weights: vec![Bounded::new(1.0, 0.0, 1.0); NUM_WEIGHTS],
            filters: FilterWeight::default(),
            radius_km_filter: Bounded::new(40.0, 0.0, 100.0),
            k_most_similar_only: vec![
                Bounded::new(5, 1, 1000),
                Bounded::new(5, 1, 1000),
                Bounded::new(1, 1, 1000),
            ],
        }
    }
}

impl From<&ExpertWeight> for ExpertDissimilarity {
    fn from(weight: &ExpertWeight) -> Self {
        assert_eq!(weight.factors.len(), NUM_FACTORS);
        let unfiltered_expert_dissimilarity = UnfilteredExpertDissimilarityBuilder::default()
            .cutoff_similarity(Some(weight.cutoff_similarity.value()))
            .similarity_part_1_weight(weight.similarity_part_1_weight.value())
            // Factors
            .wohnflaeche_factor_category_1_2(weight.factors[0].value())
            .wohnflaeche_factor_category_3(weight.factors[1].value())
            .plane_distance_factor(weight.factors[2].value())
            .baujahr_factor(weight.factors[3].value())
            .grundstuecksgroesse_factor(weight.factors[4].value())
            .anzahl_stellplaetze_factor(weight.factors[5].value())
            .anzahl_zimmer_factor(weight.factors[6].value())
            .combined_location_score_factor(weight.factors[7].value())
            // Weights
            .wohnflaeche_weight(weight.weights[0].value())
            .plane_distance_weight(weight.weights[1].value())
            .baujahr_weight(weight.weights[2].value())
            .grundstuecksgroesse_weight(weight.weights[3].value())
            .anzahl_stellplaetze_weight(weight.weights[4].value())
            .anzahl_zimmer_weight(weight.weights[5].value())
            .combined_location_score_weight(weight.weights[6].value())
            .verwendung_weight(weight.weights[7].value())
            .keller_weight(weight.weights[8].value())
            .build()
            .unwrap();

        let filtered_dissimilarity = weight
            .filters
            .wrap_dissimilarity(unfiltered_expert_dissimilarity);

        filtered_dissimilarity.into()
    }
}

impl Weight for ExpertWeight {
    type SubOutput = ExpertWeightSubOutput;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        self.cutoff_similarity.mutate(rng, config);
        self.similarity_part_1_weight.mutate(rng, config);
        self.radius_km_filter.mutate(rng, config);
        self.k_most_similar_only.mutate(rng, config);
        self.factors.mutate(rng, config);
        self.weights.mutate(rng, config);
        self.filters.mutate(rng, config);
    }

    /// This will crossover the `factors`.
    /// ALl other values are taken from self.
    fn crossover<R>(&self, other: &Self, rng: &mut R, config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        ExpertWeightBuilder::default()
            .similarity_part_1_weight(self.similarity_part_1_weight)
            .cutoff_similarity(self.cutoff_similarity)
            .k_most_similar_only(self.k_most_similar_only.crossover(
                &other.k_most_similar_only,
                rng,
                config,
            ))
            .factors(self.factors.crossover(&other.factors, rng, config))
            .weights(self.weights.crossover(&other.weights, rng, config))
            .filters(self.filters.crossover(&other.filters, rng, config))
            .radius_km_filter(
                self.radius_km_filter
                    .crossover(&other.radius_km_filter, rng, config),
            )
            .build()
            .unwrap()
    }

    /// This will regenerate all constituent weights individually.
    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        self.cutoff_similarity.regenerate(rng);
        self.similarity_part_1_weight.regenerate(rng);
        self.radius_km_filter.regenerate(rng);
        self.k_most_similar_only.regenerate(rng);
        self.factors.regenerate(rng);
        self.weights.regenerate(rng);
        self.filters.regenerate(rng);
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        let total_weight = NUM_WEIGHTS + NUM_FACTORS + NUM_FILTERS
            + 1 // cutoff
            + 1 // similarity_part_1_weight
            + 1 // radius_km_filter
            + 1; // k_most_similar_only

        let individual_distances = vec![
            self.cutoff_similarity
                .normalized_distance(&other.cutoff_similarity),
            self.similarity_part_1_weight
                .normalized_distance(&other.similarity_part_1_weight),
            self.radius_km_filter
                .normalized_distance(&other.radius_km_filter),
            self.k_most_similar_only
                .normalized_distance(&other.k_most_similar_only),
            self.factors.normalized_distance(&other.factors) * NUM_FACTORS as f64,
            self.weights.normalized_distance(&other.weights) * NUM_WEIGHTS as f64,
            self.filters.normalized_distance(&other.filters) * NUM_FILTERS as f64,
        ];

        individual_distances.iter().sum::<f64>() / total_weight as f64
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        Self {
            cutoff_similarity: self.cutoff_similarity.add(&other.cutoff_similarity),
            similarity_part_1_weight: self
                .similarity_part_1_weight
                .add(&other.similarity_part_1_weight),
            factors: self.factors.add(&other.factors),
            weights: self.weights.add(&other.weights),
            filters: self.filters.add(&other.filters),
            radius_km_filter: self.radius_km_filter.add(&other.radius_km_filter),
            k_most_similar_only: self.k_most_similar_only.add(&other.k_most_similar_only),
        }
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        Self::SubOutput {
            cutoff_similarity: self.cutoff_similarity.sub(&other.cutoff_similarity),
            similarity_part_1_weight: self
                .similarity_part_1_weight
                .sub(&other.similarity_part_1_weight),
            factors: self.factors.sub(&other.factors),
            weights: self.weights.sub(&other.weights),
            filters: self.filters.sub(&other.filters),
            radius_km_filter: self.radius_km_filter.sub(&other.radius_km_filter),
            k_most_similar_only: self.k_most_similar_only.sub(&other.k_most_similar_only),
        }
    }
}

impl WeightedPredictor<ExpertWeight> for WeightedAveragePredictor<ExpertDissimilarity> {
    fn predict_with_weight<'j>(
        &self,
        validation_data: impl IntoIterator<Item = &'j mut Immo>,
        weight: &ExpertWeight,
    ) -> BpResult<()> {
        let options = WeightedAveragePredictorOptionsBuilder::default()
            .radius(weight.radius_km_filter.value() * 1000.0) // distance
            .dissimilarity(ExpertDissimilarity::from(weight))
            .neighbors_for_average_strategy(
                NeighborsForAverageStrategy::KMostSimilarOnlyPerCategory(
                    KMostSimilarOnlyPerObjektunterartCategory {
                        category_1: weight.k_most_similar_only[0].value() as u64,
                        category_2: weight.k_most_similar_only[1].value() as u64,
                        category_3: weight.k_most_similar_only[2].value() as u64,
                    },
                ),
            )
            .build()
            .unwrap();

        self.predict_with_options(validation_data, &options)
    }
}
