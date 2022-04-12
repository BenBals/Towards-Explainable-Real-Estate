//! This module provides the [WeightedAveragePredictor] and helpers around it.
use algorithms::calculate_environment::calculate_weighted_median;
use algorithms::{
    calculate_environment::calculate_weighted_average, segment_tree::PointlikeContainer,
};
use common::{BpResult, Dissimilarity, Immo, Pointlike, Trainable};
use derive_builder::Builder;
use dissimilarities::DistanceDissimilarity;
use predictions::Predictor;
use rayon::prelude::*;
use rstar::{primitives::PointWithData, RTree};
use serde::Deserialize;

type RTreeImmoIndexPoint = PointWithData<usize, [f64; 2]>;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct ImmoIndexPoint {
    index: usize,
    x: u64,
    y: u64,
}

impl ImmoIndexPoint {
    fn query(position: [f64; 2]) -> Self {
        Self {
            index: usize::MAX,
            x: position[0] as u64,
            y: position[1] as u64,
        }
    }
    fn with(index: usize, immo: &Immo) -> Self {
        Self {
            index,
            x: immo.plane_location.unwrap().0 as u64,
            y: immo.plane_location.unwrap().1 as u64,
        }
    }
}

impl Pointlike for ImmoIndexPoint {
    fn x(&self) -> u64 {
        self.x
    }

    fn y(&self) -> u64 {
        self.y
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Deserialize)]
/// A Strategy which determines how the [WeightedAveragePredictor] chooses the neighbors for an [Immo].
pub enum NeighborSelectionStrategy {
    /// `Radius(r)` tells it to choose all [Immo]s within `r` meters.
    Radius(f64),
    /// `KNearest(k)` will give the `k` nearest neighbors.
    KNearest(u64),
}

impl Default for NeighborSelectionStrategy {
    fn default() -> Self {
        Self::Radius(10.0 * 1000.0)
    }
}

/// Configures how the neighbors are selected to the average among per objektunterart category
/// To be used in a [NeighborsForAverageStrategy]
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct KMostSimilarOnlyPerObjektunterartCategory {
    /// How many other properties to include in the average for predicting a property from category 1
    pub category_1: u64,
    /// How many other properties to include in the average for predicting a property from category 2
    pub category_2: u64,
    /// How many other properties to include in the average for predicting a property from category 3
    pub category_3: u64,
}

/// Configures how the neighbors are selected to the average among
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub enum NeighborsForAverageStrategy {
    /// Always select all available neighbors
    All,
    /// Always select only the k most similar ones
    KMostSimilarOnly(u64),
    /// Vary how many most similar ones to select by objektunterart category
    KMostSimilarOnlyPerCategory(KMostSimilarOnlyPerObjektunterartCategory),
}

impl NeighborsForAverageStrategy {
    /// # Returns
    /// - An `ALL` if the option is none,
    /// - A `KMostSimilaryOnly` with the value inside the some.
    pub fn k_most_similar_only_from_option(opt: Option<u64>) -> NeighborsForAverageStrategy {
        match opt {
            None => Self::All,
            Some(k) => Self::KMostSimilarOnly(k),
        }
    }
}

impl Default for NeighborsForAverageStrategy {
    fn default() -> Self {
        Self::All
    }
}

#[derive(Debug, Clone, Default, Builder)]
#[builder(pattern = "owned")]
/// This struct includes all configureable options for a weighted average predictor.
pub struct WeightedAveragePredictorOptions<D: Dissimilarity> {
    /// Which dissimilarity should the average be weighted by?
    pub dissimilarity: D,
    /// How should the neighbors be chosen? see [NeighborSelectionStrategy].
    neighbor_selection_strategy: NeighborSelectionStrategy,
    /// Should we limit the number of properties that effect each average to the most similar?
    /// If None, we use all properties in the radius, if Some(k) we use the k most similar ones
    /// only.
    #[builder(default)]
    neighbors_for_average_strategy: NeighborsForAverageStrategy,
    /// If an immos has less then 10 neighbors, should we use k-nearest-neighbors instead?
    /// You should probably not disable this, as otherwise a property might have no neighbours,
    /// which leads to a failed prediction.
    #[builder(default = "true")]
    ensure_minimum_number_of_neighbours: bool,
    /// Neigbors over this limit in km away are never considered, overriding other selection strategies.
    #[builder(default)]
    strict_radius_limit: Option<f64>,
    /// Use weighted median instead of weighted average prediction. Default: false
    #[builder(default = "false")]
    pub median: bool,
}

impl<D: Dissimilarity> WeightedAveragePredictorOptionsBuilder<D> {
    /// use the Radius Strategy
    pub fn radius(mut self, radius: f64) -> Self {
        self.neighbor_selection_strategy = Some(NeighborSelectionStrategy::Radius(radius));
        self
    }

    /// use the k-nearest Strategy
    pub fn k_nearest(mut self, k: u64) -> Self {
        self.neighbor_selection_strategy = Some(NeighborSelectionStrategy::KNearest(k));
        self
    }
}

/// This struct represents a simple predictor which does not use a form of clustering
/// It predicts the sqm meter price by computing a weighted average over all
/// usable immos in the training set.
/// Only immos with a sqm_price are usable.
///
/// The weight for each immo is the inverse dissimilarity
/// It is a generalized version of the Dully
#[derive(Debug, Clone, Default, Builder)]
pub struct WeightedAveragePredictor<D: Dissimilarity> {
    #[builder(setter(skip))]
    inner: Option<WeightedAveragePredictorInner>,
    /// You can use [WeightedAveragePredictorOptions] to customize the behavior of this predictor.
    pub options: WeightedAveragePredictorOptions<D>,
}

#[derive(Debug, Clone)]
struct WeightedAveragePredictorInner {
    data: Vec<Immo>,
    r_tree: RTree<RTreeImmoIndexPoint>,
    pointlike_container: PointlikeContainer<ImmoIndexPoint>,
}

impl<D: Dissimilarity> WeightedAveragePredictor<D> {
    /// Creates an untrained WeightedAveragePredictor with a given dissimilarity
    pub fn new(dissimilarity: D) -> Self {
        Self::with_radius(
            dissimilarity,
            10.0 * 1000.0, // 10km
        )
    }

    /// Create a new, untrained [WeightedAveragePredictor] with the given options.
    pub fn with_options(options: WeightedAveragePredictorOptions<D>) -> Self {
        Self {
            inner: None,
            options,
        }
    }

    /// Creates an untrained WeightedAveragePredictor with a given dissimilarity
    /// Radius must be positive and finite.
    pub fn with_radius(dissimilarity: D, radius: f64) -> Self {
        assert!(radius > 0.0);
        assert!(!radius.is_nan());

        Self::with_options(WeightedAveragePredictorOptions {
            dissimilarity,
            neighbor_selection_strategy: NeighborSelectionStrategy::Radius(radius),
            ensure_minimum_number_of_neighbours: true,
            neighbors_for_average_strategy: NeighborsForAverageStrategy::All,
            strict_radius_limit: None,
            median: false,
        })
    }

    /// Creates an untrained WeightedAveragePredictor with a given dissimilarity
    pub fn with_k_nearest(dissimilarity: D, k: u64) -> Self {
        Self::with_options(WeightedAveragePredictorOptions {
            dissimilarity,
            neighbor_selection_strategy: NeighborSelectionStrategy::KNearest(k),
            ensure_minimum_number_of_neighbours: true,
            neighbors_for_average_strategy: NeighborsForAverageStrategy::All,
            strict_radius_limit: None,
            median: false,
        })
    }
}

impl<D: Dissimilarity + Sync> WeightedAveragePredictor<D> {
    /// Predict, but use the argument dissimilarity instead of the one stored in the predictor.
    /// The caller must ensure that the dissimilarity is properly trained.
    /// # Panics
    /// - If the calculation of the average among the neighbors fails. E.g.:
    ///   - If a property to predict has no neighbor and [Self.ensure_minium_number_of_neighbours]
    ///     is set to `false`.
    ///   - If all neighbors have a dissimilarity of 0.
    pub fn predict_with_options<'i, D2: Dissimilarity + Sync>(
        &self,
        validation_data: impl IntoIterator<Item = &'i mut Immo>,
        options: &WeightedAveragePredictorOptions<D2>,
    ) -> BpResult<()> {
        if self.inner.is_none() {
            return Err("Training was not called".into());
        }

        let mut collected_data: Vec<_> = validation_data.into_iter().collect();
        collected_data.par_iter_mut().for_each(|validation_datum| {
            let query_point = [
                validation_datum.plane_location.unwrap().0,
                validation_datum.plane_location.unwrap().1,
            ];

            let mut relevant_immos: Vec<_> = match options.neighbor_selection_strategy {
                NeighborSelectionStrategy::Radius(radius) => {
                    let limited_radius = options.strict_radius_limit.map(|limit| radius.min(limit * 1000.0)).unwrap_or(radius);
                    let inner = self.inner.as_ref().unwrap();
                    let relevant_immos: Vec<_> = inner
                        .pointlike_container
                        .collect_with_distance_from_point_at_most(
                            &ImmoIndexPoint::query(query_point),
                            limited_radius,
                        )
                        .into_iter()
                        .map(|point| &inner.data[point.index])
                        .collect();


                    relevant_immos
                },
                NeighborSelectionStrategy::KNearest(k) => {
                    let inner = self.inner.as_ref().unwrap();
                    inner
                        .r_tree
                        .nearest_neighbor_iter(&query_point)
                        .take(k as usize)
                        .map(|neighbor| &inner.data[neighbor.data])
                        .filter(|neighbor| {
                            if let Some(limit) = options.strict_radius_limit {
                                log::debug!("Using strict radius limit");
                                validation_datum.plane_distance_squared(neighbor).unwrap() <= (limit * 1000.0).powf(2.0)
                            } else {
                                true
                            }
                        })
                        .collect()
                }
            };

if relevant_immos.len() < 10 && options.ensure_minimum_number_of_neighbours {
                        log::debug!(
                            "Relevant immos for {}: n = {}. Taking nearest 10 neighbors instead",
                            validation_datum.id,
                            relevant_immos.len()
                        );

    let inner = self.inner.as_ref().unwrap();

                        relevant_immos = inner
                            .r_tree
                            .nearest_neighbor_iter(&query_point)
                            .take(10)
                            .map(|neighbor| &inner.data[neighbor.data])
                            .collect()
                    }

            let selected_neighbors = &mut relevant_immos[..];

            let selected_neighbors = match &options.neighbors_for_average_strategy {
                NeighborsForAverageStrategy::All => selected_neighbors,
                NeighborsForAverageStrategy::KMostSimilarOnly(k) => {
                    k_most_similar_only(*k, selected_neighbors, validation_datum, options)
                }
                NeighborsForAverageStrategy::KMostSimilarOnlyPerCategory(config) => {
                    match validation_datum.objektunterart_category() {
                        Some(1) => k_most_similar_only(
                            config.category_1,
                            selected_neighbors,
                            validation_datum,
                            options,
                        ),
                        Some(2) => k_most_similar_only(
                            config.category_2,
                            selected_neighbors,
                            validation_datum,
                            options,
                        ),
                        Some(3) => k_most_similar_only(
                            config.category_3,
                            selected_neighbors,
                            validation_datum,
                            options,
                        ),
                        Some(_) => {
                            panic!("Invalid objektunterart category encountered in WeightedAveragePredictor")
                        }
                        None => selected_neighbors
                    }
                }
            };

            let prediction_function = if self.options.median {
                log::debug!("Predicting using median");
                calculate_weighted_median } else { calculate_weighted_average };

            let predicted_sqm_price =  prediction_function(
                selected_neighbors.iter(),
                |immo: &&&Immo| immo.sqm_price().expect("Square meter price was not set"), // train enforces this
                |immo: &&&Immo| {
                    if validation_datum.id() == immo.id() {
                        0.0
                    } else {
                        1.0 / options
                            .dissimilarity
                            .dissimilarity(validation_datum, immo)
                            // If the immo is completely similar to another, weight that one a lot,
                            // instead of infinitively much.
                            .max(1e-9)
                            // If the immos are completely dissimilary, weight it really really small
                            // instead of not at all.
                            .min(1e10)
                    }
                },
            )
            .expect("Weighted average calculation failed");
            validation_datum.marktwert = validation_datum
                .wohnflaeche
                .map(|a| a * predicted_sqm_price);
        });
        Ok(())
    }
}

fn k_most_similar_only<'i, D: Dissimilarity>(
    k: u64,
    selected_neighbors: &'i mut [&'i Immo],
    validation_datum: &Immo,
    options: &WeightedAveragePredictorOptions<D>,
) -> &'i mut [&'i Immo] {
    if selected_neighbors.len() > k as usize {
        selected_neighbors.select_nth_unstable_by(k as usize, |&neighbor1, &neighbor2| {
            let dissim1 = options
                .dissimilarity
                .dissimilarity(validation_datum, neighbor1);
            let dissim2 = options
                .dissimilarity
                .dissimilarity(validation_datum, neighbor2);
            dissim1.partial_cmp(&dissim2).unwrap()
        });

        selected_neighbors.split_at_mut(k as usize).0
    } else {
        selected_neighbors
    }
}

impl WeightedAveragePredictor<DistanceDissimilarity> {
    /// Creates an untrained Dully with exponent 2
    pub fn dully() -> Self {
        Self::dully_with(2.0)
    }

    /// Creates an untrained Dully with a given exponent
    pub fn dully_with(power: f64) -> Self {
        Self::new(DistanceDissimilarity::with(power))
    }
}

impl<D: Dissimilarity> Trainable for WeightedAveragePredictor<D> {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        let data: Vec<Immo> = training_data
            .into_iter()
            .filter(|immo| immo.sqm_price().is_some())
            .cloned()
            .collect();

        let enumerated_immos = data.iter().enumerate();

        let immo_idx_points: Vec<_> = enumerated_immos
            .clone()
            .map(|(index, immo)| ImmoIndexPoint::with(index, immo))
            .collect();
        let pointlike_container = PointlikeContainer::with(immo_idx_points)?;

        let r_tree_points: Vec<_> = enumerated_immos
            .map(|(index, immo)| {
                PointWithData::new(
                    index,
                    [
                        immo.plane_location.unwrap().0,
                        immo.plane_location.unwrap().1,
                    ],
                )
            })
            .collect();
        let r_tree = RTree::bulk_load(r_tree_points);

        self.inner = Some(WeightedAveragePredictorInner {
            data,
            r_tree,
            pointlike_container,
        });

        self.options
            .dissimilarity
            .train(self.inner.as_ref().unwrap().data.iter())?;

        if self.inner.as_ref().unwrap().data.is_empty() {
            Err("No Immos provided".into())
        } else {
            Ok(())
        }
    }
}

impl<D: Dissimilarity + Sync> Predictor for WeightedAveragePredictor<D> {
    fn predict<'i>(&self, validation_data: impl IntoIterator<Item = &'i mut Immo>) -> BpResult<()> {
        self.predict_with_options(validation_data, &self.options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use dissimilarities::DistanceDissimilarity;
    use test_helpers::*;

    #[test]
    fn predictor_respects_radius() {
        // Three immos in a row.
        // The left and middle gets the value of the middle and the middle get's the average of the
        // others
        let mut immo1 = create_new_immo_at(&[0.0, 0.0]);
        immo1.marktwert = Some(1.0);
        immo1.wohnflaeche = Some(1.0);
        let mut immo2 = create_new_immo_at(&[0.0, 1.0]);
        immo2.marktwert = Some(42.0);
        immo2.wohnflaeche = Some(1.0);
        let mut immo3 = create_new_immo_at(&[0.0, 2.0]);
        immo3.marktwert = Some(4.0);
        immo3.wohnflaeche = Some(1.0);

        let mut immos = vec![immo1, immo2, immo3];

        let dissimilarity = DistanceDissimilarity::new();
        let mut predictor = WeightedAveragePredictor::with_radius(dissimilarity, 1.0);
        predictor.options.ensure_minimum_number_of_neighbours = false;

        predictor.train(&immos).unwrap();

        predictor.predict(&mut immos).unwrap();

        // Left
        assert_approx_eq!(immos[0].marktwert.unwrap(), 42.0);
        // Right
        assert_approx_eq!(immos[2].marktwert.unwrap(), 42.0);
        // Middle
        assert_approx_eq!(immos[1].marktwert.unwrap(), 2.5);
    }
}
