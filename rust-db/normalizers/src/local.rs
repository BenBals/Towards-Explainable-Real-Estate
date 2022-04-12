use chrono::NaiveDate;

use algorithms::segment_tree::PointlikeContainer;
use common::{BpResult, Immo, Trainable};

use super::Normalizer;

use rayon::prelude::*;

/// A Bool Wrapper which captures whether a [LocalNormalizationStategy] should
/// normalize or denormalize.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub enum Mode {
    Normalize,
    Denormalize,
}

/// A Strategy on how to do the local adjustment.
pub trait LocalNormalizationStrategy: Sync {
    /// Should give the normalized or denormalized sqm_price of `immo`.
    /// This is specified in `mode`.
    /// For the refernce timeframe around the `immo.wertermittlungsstichtag` all
    /// Immos are provided in `start_immos`, around the `normalizationdate` all
    /// Immos are provided in `end_immos`.
    fn adjusted_sqm_price<'a, 'b>(
        &self,
        immo: &Immo,
        start_immos: impl Iterator<Item = &'a Immo>,
        end_immos: impl Iterator<Item = &'b Immo>,
        mode: Mode,
    ) -> Option<f64>;
}

/// A [LocalNormalizationStrategy] that adjust sqm_price by multipliying it
/// by the change in the average sqm_price.
#[derive(Debug, Clone, Copy)]
pub struct LocalAverageAdjustingStrategy;

impl LocalAverageAdjustingStrategy {
    fn count_and_sum_sqm_price<'a>(&self, immos: impl Iterator<Item = &'a Immo>) -> (usize, f64) {
        immos.fold(
            (0, 0.0),
            |(immo_neighbours_count, immo_neighbours_sum_marktwert), neighbour| {
                (
                    immo_neighbours_count + 1,
                    immo_neighbours_sum_marktwert + neighbour.sqm_price().unwrap(),
                )
            },
        )
    }
}

impl LocalNormalizationStrategy for LocalAverageAdjustingStrategy {
    fn adjusted_sqm_price<'a, 'b>(
        &self,
        immo: &Immo,
        start_immos: impl Iterator<Item = &'a Immo>,
        end_immos: impl Iterator<Item = &'b Immo>,
        mode: Mode,
    ) -> Option<f64> {
        let (immo_neighbours_count, immo_neighbours_sum_sqm_price) =
            self.count_and_sum_sqm_price(start_immos);
        let (normalize_neighbours_count, normalize_neighbours_sum_sqm_price) =
            self.count_and_sum_sqm_price(end_immos);

        if immo_neighbours_count == 0 || normalize_neighbours_count == 0 {
            immo.sqm_price()
        } else {
            let normalize_avg =
                normalize_neighbours_sum_sqm_price / (normalize_neighbours_count as f64);
            let immo_avg = immo_neighbours_sum_sqm_price / (immo_neighbours_count as f64);
            match mode {
                Mode::Normalize => immo
                    .sqm_price()
                    .map(|start| start * normalize_avg / immo_avg),
                Mode::Denormalize => immo
                    .sqm_price()
                    .map(|start| start * immo_avg / normalize_avg),
            }
        }
    }
}

/// A [LocalNormalizationStrategy] that adjust sqm_price by multipliying it
/// by the change in the median sqm_price.
#[derive(Debug, Clone, Copy)]
pub struct LocalMedianAdjusting;

impl LocalNormalizationStrategy for LocalMedianAdjusting {
    fn adjusted_sqm_price<'a, 'b>(
        &self,
        immo: &Immo,
        start_immos: impl Iterator<Item = &'a Immo>,
        end_immos: impl Iterator<Item = &'b Immo>,
        mode: Mode,
    ) -> Option<f64> {
        let mut start_immos: Vec<_> = start_immos.collect();
        let mut end_immos: Vec<_> = end_immos.collect();

        if start_immos.is_empty() || end_immos.is_empty() {
            immo.sqm_price()
        } else {
            let start_median_idx = start_immos.len() / 2;
            let end_median_idx = end_immos.len() / 2;
            let start_median = start_immos
                .select_nth_unstable_by(start_median_idx, |a, b| {
                    a.sqm_price()
                        .unwrap()
                        .partial_cmp(&b.sqm_price().unwrap())
                        .unwrap()
                })
                .1
                .sqm_price()
                .unwrap();
            let end_median = end_immos
                .select_nth_unstable_by(end_median_idx, |a, b| {
                    a.sqm_price()
                        .unwrap()
                        .partial_cmp(&b.sqm_price().unwrap())
                        .unwrap()
                })
                .1
                .sqm_price()
                .unwrap();
            let factor = end_median / start_median;

            match mode {
                Mode::Normalize => immo.sqm_price().map(|sqm_price| sqm_price * factor),
                Mode::Denormalize => immo.sqm_price().map(|sqm_price| sqm_price / factor),
            }
        }
    }
}

/// A [LocalNormalizationStrategy] that adjust sqm_price by setting it equal to
/// the sqm_price of the Immo, which is in the sorted list of Immos around the
/// normalization date at the same fraction of the length as the Immo which has
/// to be normalized in the sorted list of Immos arount its wertermittlungsstichtag.
#[derive(Debug, Clone, Copy)]
pub struct LocalRankAdjusting;

impl LocalNormalizationStrategy for LocalRankAdjusting {
    fn adjusted_sqm_price<'a, 'b>(
        &self,
        immo: &Immo,
        start_immos: impl Iterator<Item = &'a Immo>,
        end_immos: impl Iterator<Item = &'b Immo>,
        mode: Mode,
    ) -> Option<f64> {
        let start_immos: Vec<_> = start_immos.collect();
        let end_immos: Vec<_> = end_immos.collect();
        if start_immos.is_empty() || end_immos.is_empty() {
            immo.sqm_price()
        } else {
            let (current, mut other) = match mode {
                Mode::Normalize => (start_immos, end_immos),
                Mode::Denormalize => (end_immos, start_immos),
            };
            let amount_smaller = current
                .iter()
                .filter(|other| other.sqm_price().unwrap() < immo.sqm_price().unwrap())
                .count();
            let other_pos = amount_smaller * (other.len() - 1) / current.len();
            other
                .select_nth_unstable_by(other_pos, |a, b| {
                    a.sqm_price()
                        .unwrap()
                        .partial_cmp(&b.sqm_price().unwrap())
                        .unwrap()
                })
                .1
                .sqm_price()
        }
    }
}

/// A [Normalizer] which normalizes an [Immo] based on the change in the average sqm_price of the near immos.
/// The Normalization for an `immo` is parameterised by the following parameters:
/// 1. `range` the maximum distance between the `immo` and a training data point which includes
///     this training data point in the normalization (exclusive)
/// 1. `timespan` the maximum number of days which differ between the respective `wertermittlungsstichtag`
///    which allows a training data point to be included in the normalization (inclusive)
/// 1. `normalization_date` the date to which the normalization is performed.
/// 1. `strategy` how the normalization should be done.
/// For creation and default values refer to [LocalAverageAdjustingBuilder::new].
#[derive(Debug, Clone)]
pub struct LocalNormalizer<S: LocalNormalizationStrategy> {
    known_immos: Vec<Immo>,
    timespan: u64,
    normalization_date: NaiveDate,
    range: u64,
    strategy: S,
}

/// A Builder for [LocalNormalizer].
/// The default values of the settable attributes are:
/// 1. `range`: 5000
/// 1. `timespan`: 182 (i.e. half a year to include a whole year in the calculation)
/// 1. `normalization_date`: 1.1.2019
#[derive(Debug, Clone)]
pub struct LocalNormalizerBuilder<S: LocalNormalizationStrategy> {
    timespan: u64,
    normalization_date: NaiveDate,
    range: u64,
    strategy: S,
}

impl LocalNormalizerBuilder<LocalAverageAdjustingStrategy> {
    /// Creates a new [LocalNormalizerBuilder], which uses the average adjusting strategy.
    pub fn average_adjusting() -> Self {
        Self {
            timespan: 365 / 2,
            normalization_date: NaiveDate::from_yo(2019, 1),
            range: 5000,
            strategy: LocalAverageAdjustingStrategy,
        }
    }
}

impl LocalNormalizerBuilder<LocalMedianAdjusting> {
    /// Creates a new [LocalNormalizerBuilder], which uses the median adjusting strategy.
    pub fn median_adjusting() -> Self {
        Self {
            timespan: 365 / 2,
            normalization_date: NaiveDate::from_yo(2019, 1),
            range: 5000,
            strategy: LocalMedianAdjusting,
        }
    }
}

impl LocalNormalizerBuilder<LocalRankAdjusting> {
    /// Creates a new [LocalNormalizerBuilder], which uses the rank adjusting strategy.
    pub fn rank_adjusting() -> Self {
        Self {
            timespan: 365 / 2,
            normalization_date: NaiveDate::from_yo(2019, 1),
            range: 5000,
            strategy: LocalRankAdjusting,
        }
    }
}

impl<S: LocalNormalizationStrategy> LocalNormalizerBuilder<S> {
    /// Sets the `timespan`. For details refer to [LocalAverageAdjustingNormalizer].
    pub fn timespan(mut self, timespan: u64) -> Self {
        self.timespan = timespan;
        self
    }

    /// Sets the `normalization_date`. For details refer to [LocalAverageAdjustingNormalizer].
    pub fn normalization_date(mut self, normalization_date: NaiveDate) -> Self {
        self.normalization_date = normalization_date;
        self
    }

    /// Sets the `range`. For details refer to [LocalAverageAdjustingNormalizer].
    pub fn range(mut self, range: u64) -> Self {
        self.range = range;
        self
    }

    /// Builds the [LocalNormalizer].
    pub fn build(self) -> LocalNormalizer<S> {
        LocalNormalizer {
            known_immos: Vec::new(),
            timespan: self.timespan,
            range: self.range,
            normalization_date: self.normalization_date,
            strategy: self.strategy,
        }
    }
}

fn is_usable(immo: &Immo) -> bool {
    immo.marktwert
        .and(immo.plane_location)
        .and(immo.wertermittlungsstichtag)
        .is_some()
}

fn filter_around_date<'i>(
    iter: impl Iterator<Item = &'i Immo>,
    date: NaiveDate,
    timespan: u64,
) -> impl Iterator<Item = &'i Immo> {
    iter.filter(move |neighbour| {
        (neighbour.wertermittlungsstichtag.unwrap() - date)
            .num_days()
            .abs() as u64
            <= timespan
    })
}

impl<S: LocalNormalizationStrategy> LocalNormalizer<S> {
    fn normalize_or_denormalize(&self, mut immos: Vec<&'_ mut Immo>, mode: Mode) {
        immos.retain(|i| is_usable(i));
        let container = PointlikeContainer::with(self.known_immos.iter())
            .expect("Did not train or training failed");

        immos.into_par_iter().for_each(|immo| {
            let neighbours =
                container.collect_with_distance_from_point_at_most(&&*immo, self.range as f64);

            let start_immos = filter_around_date(
                neighbours.iter().copied(),
                immo.wertermittlungsstichtag.unwrap(),
                self.timespan,
            );
            let end_immos = filter_around_date(
                neighbours.iter().copied(),
                self.normalization_date,
                self.timespan,
            );

            immo.marktwert = self
                .strategy
                .adjusted_sqm_price(immo, start_immos, end_immos, mode)
                .map(|sqm_price| sqm_price * immo.wohnflaeche.unwrap());
        });
    }
}

impl<S: LocalNormalizationStrategy> Trainable for LocalNormalizer<S> {
    fn train<'j>(&mut self, immos: impl IntoIterator<Item = &'j Immo>) -> BpResult<()> {
        self.known_immos
            .extend(immos.into_iter().filter(|i| is_usable(i)).cloned());
        if self.known_immos.is_empty() {
            Err("No Immos are usable".into())
        } else {
            Ok(())
        }
    }
}

impl<S: LocalNormalizationStrategy> Normalizer for LocalNormalizer<S> {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        self.normalize_or_denormalize(immos.into_iter().collect(), Mode::Normalize);
    }

    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        self.normalize_or_denormalize(immos.into_iter().collect(), Mode::Denormalize);
    }
}

#[cfg(test)]
mod tests {
    macro_rules! local_adjusting_tests_normalize_and_denormalize {
        ($name: ident, $builder : expr) => {
            mod $name {
                use std::iter::once;

                use common::{Normalizer, Trainable};
                use proptest::prelude::*;
                use test_helpers::*;

                proptest! {
                    #[test]
                    fn normalize_works_on_singles(mut to_normalize in full_immo(), training_immo in full_immo()) {
                        prop_assume!(to_normalize.wertermittlungsstichtag != training_immo.wertermittlungsstichtag);

                        to_normalize.plane_location = training_immo.plane_location;

                        let mut normalizer = $builder
                            .timespan(0)
                            .range(1)
                            .normalization_date(training_immo.wertermittlungsstichtag.unwrap())
                            .build();

                        normalizer.train(once(&to_normalize).chain(once(&training_immo))).unwrap();
                        normalizer.normalize(once(&mut to_normalize));

                        prop_assert!(
                            (to_normalize.sqm_price().unwrap() - training_immo.sqm_price().unwrap()).abs() / training_immo.sqm_price().unwrap() < 1e-3,
                            "Expected {} got {}",
                            training_immo.sqm_price().unwrap(),
                            to_normalize.sqm_price().unwrap(),
                        );
                    }

                    #[test]
                    fn denormalize_works_on_singles(mut to_normalize in full_immo(), training_immo in full_immo()) {
                        prop_assume!(to_normalize.wertermittlungsstichtag != training_immo.wertermittlungsstichtag);

                        to_normalize.plane_location = training_immo.plane_location;

                        let mut normalizer = $builder
                            .timespan(0)
                            .range(1)
                            .normalization_date(training_immo.wertermittlungsstichtag.unwrap())
                            .build();

                        normalizer.train(once(&to_normalize).chain(once(&training_immo))).unwrap();

                        // do normalization by hand
                        let initial_sqm_price = to_normalize.sqm_price().unwrap();
                        to_normalize.marktwert = Some(to_normalize.wohnflaeche.unwrap() * training_immo.sqm_price().unwrap());

                        // then denormalize
                        normalizer.denormalize(once(&mut to_normalize));

                        prop_assert!(
                            (to_normalize.sqm_price().unwrap() - initial_sqm_price).abs() / initial_sqm_price < 1e-3,
                            "Expected {} got {}",
                            training_immo.sqm_price().unwrap(),
                            to_normalize.sqm_price().unwrap(),
                        );
                    }
                }
            }
        };
    }

    macro_rules! local_adjusting_tests_bijection {
        ($name: ident, $builder : expr) => {
            mod $name {
                use chrono::NaiveDate;

                use common::{Normalizer, Trainable};
                use proptest::prelude::*;
                use test_helpers::*;

                proptest! {
                    #[test]
                    fn defines_bijection(mut to_normalize in berlin_full_immos(128), training_immos in berlin_full_immos(128)) {
                        let mut normalizer = $builder
                            .timespan(365 * 5 / 2) // use 5 years
                            .range(1000000000) // use all immos as neighbours
                            .normalization_date(NaiveDate::from_yo(2019, 1))
                            .build();

                        normalizer.train(&training_immos).unwrap();
                        let initial = to_normalize.clone();
                        normalizer.normalize(&mut to_normalize);
                        normalizer.denormalize(&mut to_normalize);

                        prop_assert!(
                            initial
                                .iter()
                                .zip(to_normalize.iter())
                                .all(|(init, cur)| (init.sqm_price().unwrap() - cur.sqm_price().unwrap()) / init.sqm_price().unwrap() < 1e-3)
                        );
                    }
                }
            }
        };
    }

    local_adjusting_tests_normalize_and_denormalize!(
        average_adjusting,
        crate::LocalNormalizerBuilder::average_adjusting()
    );
    local_adjusting_tests_bijection!(
        average_adjusting_bijection,
        crate::LocalNormalizerBuilder::average_adjusting()
    );

    local_adjusting_tests_normalize_and_denormalize!(
        median_adjusting,
        crate::LocalNormalizerBuilder::median_adjusting()
    );
    local_adjusting_tests_bijection!(
        median_adjusting_bijection,
        crate::LocalNormalizerBuilder::median_adjusting()
    );

    local_adjusting_tests_normalize_and_denormalize!(
        rank_adjusting,
        crate::LocalNormalizerBuilder::rank_adjusting()
    );
}
