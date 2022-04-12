//! A [Weight] representing all learnable attributes of a [FilteredDissimilarity].
use crate::weight::either_weight::ActiveSide;
use crate::weight::traits::Scalable;
use crate::weight::{
    bounded,
    config::{CrossoverConfig, MutationConfig},
    Bounded, EitherWeight, OptionalWeight, Weight,
};
use dissimilarities::filtered::{
    FilterStrategy, FilteredDissimilarity, FilteredDissimilarityBuilder, NUM_FILTERS,
};
use genevo::prelude::Rng;
use itertools::Either;
use std::fmt::{Debug, Formatter, Result};

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// A Weight which can be either a category difference or a difference in the actual value.
/// Should always match with [FilterStrategy].
pub struct CategoryOrContinuousFilter(EitherWeight<Bounded<i64>, Bounded<f64>>);

/// An Output struct which is used for the subtraction of two CategoryOrContinuousFilterWeights
#[derive(Clone, Debug)]
pub struct CategoryOrContinuousFilterSubOutput(
    <EitherWeight<bounded::Bounded<i64>, bounded::Bounded<f64>> as Weight>::SubOutput,
);

impl Scalable for CategoryOrContinuousFilterSubOutput {
    fn scale(self, scalar: f64) -> Self {
        CategoryOrContinuousFilterSubOutput(self.0.scale(scalar))
    }
}

impl CategoryOrContinuousFilter {
    /// create a new CategoryOrContinuousFilter
    pub fn new(category: Bounded<i64>, continuous: Bounded<f64>, active: ActiveSide) -> Self {
        Self(EitherWeight::new(category, continuous, active))
    }
}

impl From<&CategoryOrContinuousFilter> for FilterStrategy {
    fn from(weight: &CategoryOrContinuousFilter) -> Self {
        match weight.0.active_value() {
            Either::Left(cat) => FilterStrategy::Category(cat.value() as u64),
            Either::Right(value) => FilterStrategy::Continuous(value.value()),
        }
    }
}

impl Default for CategoryOrContinuousFilter {
    fn default() -> Self {
        Self::new(Bounded::default(), Bounded::default(), ActiveSide::Left)
    }
}

impl Weight for CategoryOrContinuousFilter {
    type SubOutput = CategoryOrContinuousFilterSubOutput;

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

    fn regenerate<Rn>(&mut self, rng: &mut Rn)
    where
        Rn: Rng + Sized,
    {
        self.0.regenerate(rng)
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        self.0.normalized_distance(&other.0)
    }
    fn add(&self, other: &Self::SubOutput) -> Self {
        let mut new_self = *self;
        new_self.0 = self.0.add(&other.0);
        new_self
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        Self::SubOutput {
            0: self.0.sub(&other.0),
        }
    }
}

impl Debug for CategoryOrContinuousFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        FilterStrategy::from(self).fmt(f)
    }
}

/// A weight that filters the underlying data with the given bounds
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FilterWeight {
    difference_regiotyp: OptionalWeight<Bounded<i64>>,
    difference_baujahr: OptionalWeight<Bounded<f64>>,
    difference_grundstuecksgroesse: OptionalWeight<Bounded<f64>>,
    difference_wohnflaeche: OptionalWeight<Bounded<f64>>,
    difference_objektunterart: OptionalWeight<Bounded<i64>>,
    difference_wertermittlungsstichtag: OptionalWeight<Bounded<i64>>,
    difference_zustand: OptionalWeight<Bounded<i64>>,
    difference_ausstattungsnote: OptionalWeight<Bounded<i64>>,
    difference_balcony_area: OptionalWeight<Bounded<f64>>,
    difference_urbanity_score: OptionalWeight<Bounded<i64>>,
    difference_convenience_store_distance: OptionalWeight<Bounded<f64>>,
    difference_distance_elem_school: OptionalWeight<Bounded<f64>>,
    difference_distance_jun_highschool: OptionalWeight<Bounded<f64>>,
    difference_distance_parking: OptionalWeight<Bounded<f64>>,
    difference_walking_distance: OptionalWeight<Bounded<f64>>,
    difference_floor: OptionalWeight<Bounded<f64>>,
}

impl FilterWeight {
    /// create a new Filter Weight
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        difference_regiotyp: OptionalWeight<Bounded<i64>>,
        difference_baujahr: OptionalWeight<Bounded<f64>>,
        difference_grundstuecksgroesse: OptionalWeight<Bounded<f64>>,
        difference_wohnflaeche: OptionalWeight<Bounded<f64>>,
        difference_objektunterart: OptionalWeight<Bounded<i64>>,
        difference_wertermittlungsstichtag: OptionalWeight<Bounded<i64>>,
        difference_zustand: OptionalWeight<Bounded<i64>>,
        difference_ausstattungsnote: OptionalWeight<Bounded<i64>>,
        difference_balcony_area: OptionalWeight<Bounded<f64>>,
        difference_urbanity_score: OptionalWeight<Bounded<i64>>,
        difference_convenience_store_distance: OptionalWeight<Bounded<f64>>,
        difference_distance_elem_school: OptionalWeight<Bounded<f64>>,
        difference_distance_jun_highschool: OptionalWeight<Bounded<f64>>,
        difference_distance_parking: OptionalWeight<Bounded<f64>>,
        difference_walking_distance: OptionalWeight<Bounded<f64>>,
        difference_floor: OptionalWeight<Bounded<f64>>,
    ) -> Self {
        Self {
            difference_regiotyp,
            difference_baujahr,
            difference_grundstuecksgroesse,
            difference_wohnflaeche,
            difference_objektunterart,
            difference_wertermittlungsstichtag,
            difference_zustand,
            difference_ausstattungsnote,
            difference_balcony_area,
            difference_urbanity_score,
            difference_convenience_store_distance,
            difference_distance_elem_school,
            difference_distance_jun_highschool,
            difference_distance_parking,
            difference_walking_distance,
            difference_floor,
        }
    }
}

/// An Output struct which is used for the subtraction of two [FilterWeight]s
#[derive(Clone, Debug)]
pub struct FilterWeightSubOutput {
    difference_regiotyp: Option<i64>,
    difference_baujahr: Option<f64>,
    difference_grundstuecksgroesse: Option<f64>,
    difference_wohnflaeche: Option<f64>,
    difference_objektunterart: Option<i64>,
    difference_wertermittlungsstichtag: Option<i64>,
    difference_zustand: Option<i64>,
    difference_ausstattungsnote: Option<i64>,
    difference_balcony_area: Option<f64>,
    difference_urbanity_score: Option<i64>,
    difference_convenience_store_distance: Option<f64>,
    difference_distance_elem_school: Option<f64>,
    difference_distance_jun_highschool: Option<f64>,
    difference_distance_parking: Option<f64>,
    difference_walking_distance: Option<f64>,
    difference_floor: Option<f64>,
}

impl Scalable for FilterWeightSubOutput {
    fn scale(self, scalar: f64) -> Self {
        Self {
            difference_regiotyp: self.difference_regiotyp.scale(scalar),
            difference_baujahr: self.difference_baujahr.scale(scalar),
            difference_grundstuecksgroesse: self.difference_grundstuecksgroesse.scale(scalar),
            difference_wohnflaeche: self.difference_wohnflaeche.scale(scalar),
            difference_objektunterart: self.difference_objektunterart.scale(scalar),
            difference_wertermittlungsstichtag: self
                .difference_wertermittlungsstichtag
                .scale(scalar),
            difference_zustand: self.difference_zustand.scale(scalar),
            difference_ausstattungsnote: self.difference_ausstattungsnote.scale(scalar),
            difference_balcony_area: self.difference_balcony_area.scale(scalar),
            difference_urbanity_score: self.difference_urbanity_score.scale(scalar),
            difference_convenience_store_distance: self
                .difference_convenience_store_distance
                .scale(scalar),
            difference_distance_elem_school: self.difference_distance_elem_school.scale(scalar),
            difference_distance_jun_highschool: self
                .difference_distance_jun_highschool
                .scale(scalar),
            difference_distance_parking: self.difference_distance_parking.scale(scalar),
            difference_walking_distance: self.difference_walking_distance.scale(scalar),
            difference_floor: self.difference_floor.scale(scalar),
        }
    }
}

impl Default for FilterWeight {
    fn default() -> Self {
        Self {
            difference_regiotyp: OptionalWeight::with_value_from_seed(Bounded::new(0, 0, 50)),
            difference_baujahr: OptionalWeight::with_value_from_seed(Bounded::new(
                10.0, 0.0, 100.0,
            )),
            difference_grundstuecksgroesse: OptionalWeight::with_value_from_seed(Bounded::new(
                100.0, 0.0, 10000.0,
            )),
            difference_wohnflaeche: OptionalWeight::with_value_from_seed(Bounded::new(
                30.0, 0.0, 2000.0,
            )),
            difference_objektunterart: OptionalWeight::with_value_from_seed(Bounded::new(0, 0, 3)),
            difference_wertermittlungsstichtag: OptionalWeight::with_value_from_seed(Bounded::new(
                5 * 366,
                0,
                10 * 366,
            )),
            difference_zustand: OptionalWeight::with_value_from_seed(Bounded::new(0, 0, 4)),
            difference_ausstattungsnote: OptionalWeight::with_value_from_seed(Bounded::new(
                0, 0, 4,
            )),
            difference_balcony_area: OptionalWeight::with_value_from_seed(Bounded::new(
                20.0, 0.0, 30.0,
            )),
            difference_urbanity_score: OptionalWeight::with_value_from_seed(Bounded::new(0, 0, 4)),
            difference_convenience_store_distance: OptionalWeight::with_value_from_seed(
                Bounded::new(1000.0, 0.0, 10_000.0),
            ),
            difference_distance_elem_school: OptionalWeight::with_value_from_seed(Bounded::new(
                1000.0, 0.0, 20_000.0,
            )),
            difference_distance_jun_highschool: OptionalWeight::with_value_from_seed(Bounded::new(
                1000.0, 0.0, 20_000.0,
            )),
            difference_distance_parking: OptionalWeight::with_value_from_seed(Bounded::new(
                200.0, 0.0, 1000.0,
            )),
            difference_walking_distance: OptionalWeight::with_value_from_seed(Bounded::new(
                500.0, 0.0, 5000.0,
            )),
            difference_floor: OptionalWeight::with_value_from_seed(Bounded::new(10.0, 0.0, 30.0)),
        }
    }
}

impl FilterWeight {
    /// Use the current weights to generate a [FilteredDissimilarity] that filters the given
    /// dissimilarity
    pub fn wrap_dissimilarity<D>(&self, dissimilarity: D) -> FilteredDissimilarity<D> {
        FilteredDissimilarityBuilder::default()
            .difference_regiotyp(
                self.difference_regiotyp
                    .value()
                    .map(|bounded| bounded.value() as u64),
            )
            .difference_baujahr(
                self.difference_baujahr
                    .value()
                    .as_ref()
                    .map(|filter| FilterStrategy::Continuous(filter.value())),
            )
            .difference_grundstuecksgroesse(
                self.difference_grundstuecksgroesse
                    .value()
                    .as_ref()
                    .map(|filter| FilterStrategy::Continuous(filter.value())),
            )
            .difference_wohnflaeche(
                self.difference_wohnflaeche
                    .value()
                    .as_ref()
                    .map(|filter| FilterStrategy::Continuous(filter.value())),
            )
            .difference_objektunterart(
                self.difference_objektunterart
                    .value()
                    .map(|bounded| bounded.value() as u64),
            )
            .difference_wertermittlungsstichtag_days(
                self.difference_wertermittlungsstichtag
                    .value()
                    .map(|bounded| bounded.value() as u64),
            )
            .difference_zustand(
                self.difference_zustand
                    .value()
                    .map(|bounded| bounded.value() as u64),
            )
            .difference_ausstattungsnote(
                self.difference_ausstattungsnote
                    .value()
                    .map(|bounded| bounded.value() as u64),
            )
            .difference_balcony_area(
                self.difference_balcony_area
                    .value()
                    .map(|bounded| bounded.value()),
            )
            .difference_urbanity_score(
                self.difference_urbanity_score
                    .value()
                    .map(|bounded| bounded.value() as u64),
            )
            .difference_convenience_store_distance(
                self.difference_convenience_store_distance
                    .value()
                    .map(|bounded| bounded.value()),
            )
            .difference_distance_elem_school(
                self.difference_distance_elem_school
                    .value()
                    .map(|bounded| bounded.value()),
            )
            .difference_distance_jun_highschool(
                self.difference_distance_jun_highschool
                    .value()
                    .map(|bounded| bounded.value()),
            )
            .difference_distance_parking(
                self.difference_distance_parking
                    .value()
                    .map(|bounded| bounded.value()),
            )
            .difference_walking_distance(
                self.difference_walking_distance
                    .value()
                    .map(|bounded| bounded.value()),
            )
            .difference_floor(self.difference_floor.value().map(|bounded| bounded.value()))
            .inner(dissimilarity)
            .build()
            .unwrap()
    }

    /// Creates a [Self] where no filtering is applied.
    pub fn all_none() -> Self {
        let mut new = Self::default();
        new.difference_regiotyp.value_mut().take();
        new.difference_baujahr.value_mut().take();
        new.difference_grundstuecksgroesse.value_mut().take();
        new.difference_wohnflaeche.value_mut().take();
        new.difference_objektunterart.value_mut().take();
        new.difference_wertermittlungsstichtag.value_mut().take();
        new.difference_zustand.value_mut().take();
        new.difference_ausstattungsnote.value_mut().take();
        new.difference_balcony_area.value_mut().take();
        new.difference_urbanity_score.value_mut().take();
        new.difference_convenience_store_distance.value_mut().take();
        new.difference_distance_elem_school.value_mut().take();
        new.difference_distance_jun_highschool.value_mut().take();
        new.difference_distance_parking.value_mut().take();
        new.difference_walking_distance.value_mut().take();
        new.difference_floor.value_mut().take();

        new
    }
}

impl Weight for FilterWeight {
    type SubOutput = FilterWeightSubOutput;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        self.difference_regiotyp.mutate(rng, config);
        self.difference_baujahr.mutate(rng, config);
        self.difference_grundstuecksgroesse.mutate(rng, config);
        self.difference_wohnflaeche.mutate(rng, config);
        self.difference_objektunterart.mutate(rng, config);
        self.difference_wertermittlungsstichtag.mutate(rng, config);
        self.difference_zustand.mutate(rng, config);
        self.difference_ausstattungsnote.mutate(rng, config);
        self.difference_balcony_area.mutate(rng, config);
        self.difference_urbanity_score.mutate(rng, config);
        self.difference_convenience_store_distance
            .mutate(rng, config);
        self.difference_distance_elem_school.mutate(rng, config);
        self.difference_distance_jun_highschool.mutate(rng, config);
        self.difference_distance_parking.mutate(rng, config);
        self.difference_walking_distance.mutate(rng, config);
        self.difference_floor.mutate(rng, config);
    }

    fn crossover<R>(&self, other: &Self, rng: &mut R, config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        FilterWeight {
            difference_regiotyp: self.difference_regiotyp.crossover(
                &other.difference_regiotyp,
                rng,
                config,
            ),
            difference_baujahr: self.difference_baujahr.crossover(
                &other.difference_baujahr,
                rng,
                config,
            ),
            difference_grundstuecksgroesse: self.difference_grundstuecksgroesse.crossover(
                &other.difference_grundstuecksgroesse,
                rng,
                config,
            ),
            difference_wohnflaeche: self.difference_wohnflaeche.crossover(
                &other.difference_wohnflaeche,
                rng,
                config,
            ),
            difference_objektunterart: self.difference_objektunterart.crossover(
                &other.difference_objektunterart,
                rng,
                config,
            ),
            difference_wertermittlungsstichtag: self.difference_wertermittlungsstichtag.crossover(
                &other.difference_wertermittlungsstichtag,
                rng,
                config,
            ),
            difference_zustand: self.difference_zustand.crossover(
                &other.difference_zustand,
                rng,
                config,
            ),
            difference_ausstattungsnote: self.difference_ausstattungsnote.crossover(
                &other.difference_ausstattungsnote,
                rng,
                config,
            ),
            difference_balcony_area: self.difference_balcony_area.crossover(
                &other.difference_balcony_area,
                rng,
                config,
            ),
            difference_urbanity_score: self.difference_urbanity_score.crossover(
                &other.difference_urbanity_score,
                rng,
                config,
            ),
            difference_convenience_store_distance: self
                .difference_convenience_store_distance
                .crossover(&other.difference_convenience_store_distance, rng, config),
            difference_distance_elem_school: self.difference_distance_elem_school.crossover(
                &other.difference_distance_elem_school,
                rng,
                config,
            ),
            difference_distance_jun_highschool: self.difference_distance_jun_highschool.crossover(
                &other.difference_distance_jun_highschool,
                rng,
                config,
            ),
            difference_distance_parking: self.difference_distance_parking.crossover(
                &other.difference_distance_parking,
                rng,
                config,
            ),
            difference_walking_distance: self.difference_walking_distance.crossover(
                &other.difference_walking_distance,
                rng,
                config,
            ),
            difference_floor: self
                .difference_floor
                .crossover(&other.difference_floor, rng, config),
        }
    }

    /// This will regenerate all constituent weights individually.
    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        self.difference_regiotyp.regenerate(rng);
        self.difference_baujahr.regenerate(rng);
        self.difference_grundstuecksgroesse.regenerate(rng);
        self.difference_wohnflaeche.regenerate(rng);
        self.difference_objektunterart.regenerate(rng);
        self.difference_wertermittlungsstichtag.regenerate(rng);
        self.difference_zustand.regenerate(rng);
        self.difference_ausstattungsnote.regenerate(rng);
        self.difference_balcony_area.regenerate(rng);
        self.difference_urbanity_score.regenerate(rng);
        self.difference_convenience_store_distance.regenerate(rng);
        self.difference_distance_elem_school.regenerate(rng);
        self.difference_distance_jun_highschool.regenerate(rng);
        self.difference_distance_parking.regenerate(rng);
        self.difference_walking_distance.regenerate(rng);
        self.difference_floor.regenerate(rng);
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        let total_weight = NUM_FILTERS;

        let individual_distances = vec![
            self.difference_regiotyp
                .normalized_distance(&other.difference_regiotyp),
            self.difference_baujahr
                .normalized_distance(&other.difference_baujahr),
            self.difference_grundstuecksgroesse
                .normalized_distance(&other.difference_grundstuecksgroesse),
            self.difference_wohnflaeche
                .normalized_distance(&other.difference_wohnflaeche),
            self.difference_objektunterart
                .normalized_distance(&other.difference_objektunterart),
            self.difference_wertermittlungsstichtag
                .normalized_distance(&other.difference_wertermittlungsstichtag),
            self.difference_zustand
                .normalized_distance(&other.difference_zustand),
            self.difference_ausstattungsnote
                .normalized_distance(&other.difference_ausstattungsnote),
            self.difference_balcony_area
                .normalized_distance(&other.difference_balcony_area),
            self.difference_urbanity_score
                .normalized_distance(&other.difference_urbanity_score),
            self.difference_convenience_store_distance
                .normalized_distance(&other.difference_convenience_store_distance),
            self.difference_distance_elem_school
                .normalized_distance(&other.difference_distance_elem_school),
            self.difference_distance_jun_highschool
                .normalized_distance(&other.difference_distance_jun_highschool),
            self.difference_distance_parking
                .normalized_distance(&other.difference_distance_parking),
            self.difference_walking_distance
                .normalized_distance(&other.difference_walking_distance),
            self.difference_floor
                .normalized_distance(&other.difference_floor),
        ];

        individual_distances.iter().sum::<f64>() / total_weight as f64
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        let new_self = self.clone();
        new_self.difference_regiotyp.add(&other.difference_regiotyp);
        new_self.difference_baujahr.add(&other.difference_baujahr);
        new_self
            .difference_grundstuecksgroesse
            .add(&other.difference_grundstuecksgroesse);
        new_self
            .difference_wohnflaeche
            .add(&other.difference_wohnflaeche);
        new_self
            .difference_objektunterart
            .add(&other.difference_objektunterart);
        new_self
            .difference_wertermittlungsstichtag
            .add(&other.difference_wertermittlungsstichtag);
        new_self.difference_zustand.add(&other.difference_zustand);
        new_self
            .difference_ausstattungsnote
            .add(&other.difference_ausstattungsnote);
        new_self
            .difference_balcony_area
            .add(&other.difference_balcony_area);
        new_self
            .difference_urbanity_score
            .add(&other.difference_urbanity_score);
        new_self
            .difference_convenience_store_distance
            .add(&other.difference_convenience_store_distance);
        new_self
            .difference_distance_elem_school
            .add(&other.difference_distance_elem_school);
        new_self
            .difference_distance_jun_highschool
            .add(&other.difference_distance_jun_highschool);
        new_self
            .difference_distance_parking
            .add(&other.difference_distance_parking);
        new_self
            .difference_walking_distance
            .add(&other.difference_walking_distance);
        new_self.difference_floor.add(&other.difference_floor);
        new_self
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        Self::SubOutput {
            difference_regiotyp: self.difference_regiotyp.sub(&other.difference_regiotyp),
            difference_baujahr: self.difference_baujahr.sub(&other.difference_baujahr),
            difference_grundstuecksgroesse: self
                .difference_grundstuecksgroesse
                .sub(&other.difference_grundstuecksgroesse),
            difference_wohnflaeche: self
                .difference_wohnflaeche
                .sub(&other.difference_wohnflaeche),
            difference_objektunterart: self
                .difference_objektunterart
                .sub(&other.difference_objektunterart),
            difference_wertermittlungsstichtag: self
                .difference_wertermittlungsstichtag
                .sub(&other.difference_wertermittlungsstichtag),
            difference_zustand: self.difference_zustand.sub(&other.difference_zustand),
            difference_ausstattungsnote: self
                .difference_ausstattungsnote
                .sub(&other.difference_ausstattungsnote),
            difference_balcony_area: self
                .difference_balcony_area
                .sub(&other.difference_balcony_area),
            difference_urbanity_score: self
                .difference_urbanity_score
                .sub(&other.difference_urbanity_score),
            difference_convenience_store_distance: self
                .difference_convenience_store_distance
                .sub(&other.difference_convenience_store_distance),
            difference_distance_elem_school: self
                .difference_distance_elem_school
                .sub(&other.difference_distance_elem_school),
            difference_distance_jun_highschool: self
                .difference_distance_jun_highschool
                .sub(&other.difference_distance_jun_highschool),
            difference_distance_parking: self
                .difference_distance_parking
                .sub(&other.difference_distance_parking),
            difference_walking_distance: self
                .difference_walking_distance
                .sub(&other.difference_walking_distance),
            difference_floor: self.difference_floor.sub(&other.difference_floor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_none_is_none() {
        let filter_weight = FilterWeight::all_none();

        assert!(filter_weight.difference_regiotyp.value().is_none());
        assert!(filter_weight.difference_baujahr.value().is_none());
        assert!(filter_weight
            .difference_grundstuecksgroesse
            .value()
            .is_none());
        assert!(filter_weight.difference_objektunterart.value().is_none());
        assert!(filter_weight
            .difference_wertermittlungsstichtag
            .value()
            .is_none());
        assert!(filter_weight.difference_zustand.value().is_none());
        assert!(filter_weight.difference_ausstattungsnote.value().is_none());
        assert!(filter_weight.difference_balcony_area.value().is_none());
        assert!(filter_weight.difference_urbanity_score.value().is_none());
        assert!(filter_weight
            .difference_convenience_store_distance
            .value()
            .is_none());
        assert!(filter_weight
            .difference_distance_elem_school
            .value()
            .is_none());
        assert!(filter_weight
            .difference_distance_jun_highschool
            .value()
            .is_none());
        assert!(filter_weight.difference_distance_parking.value().is_none());
        assert!(filter_weight.difference_walking_distance.value().is_none());
        assert!(filter_weight.difference_floor.value().is_none());
    }
}
