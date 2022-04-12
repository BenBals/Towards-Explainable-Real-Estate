//! This module contains code to build a dissimilarity which filters the immos based on different values
use common::{BpResult, Dissimilarity, Immo, Trainable};
use derive_builder::Builder;
use serde::Deserialize;
use std::ops::Sub;

/// How many filters does a [FilteredDissimilarity] contain?
pub const NUM_FILTERS: usize = 9;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
/// A strategy describing how the filter should be applied.
pub enum FilterStrategy {
    /// Category will call `immo.x_category()` and compare the results
    Category(u64),
    /// Continuous will use `immo.x` and compare the results
    Continuous(f64),
}

#[derive(Clone, Debug, Deserialize)]
/// a Config struct for [FilteredDissimilarity] which is used to read all filters from a dhall file.
pub struct FilterConfig {
    difference_regiotyp: Option<u64>,
    difference_baujahr: Option<FilterStrategy>,
    difference_grundstuecksgroesse: Option<FilterStrategy>,
    difference_wohnflaeche: Option<FilterStrategy>,
    difference_objektunterart: Option<u64>,
    difference_wertermittlungsstichtag_days: Option<u64>,
    difference_spatial_distance_kilometer: Option<u64>,
    difference_zustand: Option<u64>,
    difference_ausstattungsnote: Option<u64>,
    difference_balcony_area: Option<f64>,
    difference_urbanity_score: Option<u64>,
    difference_convenience_store_distance: Option<f64>,
    difference_distance_elem_school: Option<f64>,
    difference_distance_jun_highschool: Option<f64>,
    difference_distance_parking: Option<f64>,
    difference_walking_distance: Option<f64>,
    difference_floor: Option<f64>,
}

#[derive(Clone, Debug, Builder, PartialEq)]
#[builder(pattern = "owned")]
/// A Wrapper around a Dissimilarity which filters both immos before evaluating the actual dissimilarity.
/// If both immos are not similar enough, the dissimilarity is infinite.
pub struct FilteredDissimilarity<D> {
    /// maximum difference in regiotyp for two immos, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_regiotyp: Option<u64>,
    /// maximum difference in baujahr category, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_baujahr: Option<FilterStrategy>,
    /// maximum difference in grundstuecksgroesse category, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_grundstuecksgroesse: Option<FilterStrategy>,
    /// maximum difference in wohnflaeche category, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_wohnflaeche: Option<FilterStrategy>,
    /// difference in objektunterart category, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_objektunterart: Option<u64>,
    /// max difference in wertermittlungsstichtag in years, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_wertermittlungsstichtag_days: Option<u64>,
    /// max distance in km, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_spatial_distance_kilometer: Option<u64>,
    /// maximum difference in zustand-categories, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_zustand: Option<u64>,
    /// max difference in ausstattungsnote, if difference is higher, immos are not similar at all
    #[builder(default)]
    difference_ausstattungsnote: Option<u64>,
    /// max difference in urbanity
    #[builder(default)]
    difference_urbanity_score: Option<u64>,
    /// max difference in balcony_area
    #[builder(default)]
    difference_balcony_area: Option<f64>,
    /// max difference in convenience_store_distance
    #[builder(default)]
    difference_convenience_store_distance: Option<f64>,
    /// max difference in distance_elem_school
    #[builder(default)]
    difference_distance_elem_school: Option<f64>,
    /// max difference in distance_jun_highschool
    #[builder(default)]
    difference_distance_jun_highschool: Option<f64>,
    /// max difference in distance_parking
    #[builder(default)]
    difference_distance_parking: Option<f64>,
    /// max difference in walking_distance
    #[builder(default)]
    difference_walking_distance: Option<f64>,
    /// max difference in floor
    #[builder(default)]
    difference_floor: Option<f64>,

    /// The dissimilarity to use if the immos pass all filters
    pub inner: D,
}

impl<D> FilteredDissimilarity<D> {
    /// Creates a new [FilteredDissimilarity] from a [FilterConfig] and an inner Dissimilarity.
    pub fn from_config(config: &FilterConfig, inner: D) -> Self {
        Self {
            difference_regiotyp: config.difference_regiotyp,
            difference_baujahr: config.difference_baujahr,
            difference_grundstuecksgroesse: config.difference_grundstuecksgroesse,
            difference_wohnflaeche: config.difference_wohnflaeche,
            difference_objektunterart: config.difference_objektunterart,
            difference_wertermittlungsstichtag_days: config.difference_wertermittlungsstichtag_days,
            difference_spatial_distance_kilometer: config.difference_spatial_distance_kilometer,
            difference_zustand: config.difference_zustand,
            difference_ausstattungsnote: config.difference_ausstattungsnote,
            difference_urbanity_score: config.difference_urbanity_score,
            difference_balcony_area: config.difference_balcony_area,
            difference_convenience_store_distance: config.difference_convenience_store_distance,
            difference_distance_elem_school: config.difference_distance_elem_school,
            difference_distance_jun_highschool: config.difference_distance_jun_highschool,
            difference_distance_parking: config.difference_distance_parking,
            difference_walking_distance: config.difference_walking_distance,
            difference_floor: config.difference_floor,
            inner,
        }
    }
}

impl<D: Trainable> Trainable for FilteredDissimilarity<D> {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.inner.train(training_data)
    }
}

impl<D: Dissimilarity> Dissimilarity for FilteredDissimilarity<D> {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        if !is_inside_of_allowed_integer_range(
            immo1.regiotyp,
            immo2.regiotyp,
            self.difference_regiotyp,
        ) || !is_inside_of_allowed_integer_range(
            immo1.objektunterart_category(),
            immo2.objektunterart_category(),
            self.difference_objektunterart,
        ) || !is_inside_of_allowed_integer_range(
            immo1.zustand,
            immo2.zustand,
            self.difference_zustand,
        ) || !is_inside_of_allowed_integer_range(
            immo1.urbanity_score,
            immo2.urbanity_score,
            self.difference_urbanity_score,
        ) || !is_inside_of_allowed_float_range(
            immo1.balcony_area,
            immo2.balcony_area,
            self.difference_balcony_area,
        ) || !is_inside_of_allowed_float_range(
            immo1.convenience_store_distance,
            immo2.convenience_store_distance,
            self.difference_convenience_store_distance,
        ) || !is_inside_of_allowed_float_range(
            immo1.distance_elem_school,
            immo2.distance_elem_school,
            self.difference_distance_elem_school,
        ) || !is_inside_of_allowed_float_range(
            immo1.distance_jun_highschool,
            immo2.distance_jun_highschool,
            self.difference_distance_jun_highschool,
        ) || !is_inside_of_allowed_float_range(
            immo1.distance_parking,
            immo2.distance_parking,
            self.difference_distance_parking,
        ) || !is_inside_of_allowed_float_range(
            immo1.walking_distance,
            immo2.walking_distance,
            self.difference_walking_distance,
        ) || !is_inside_of_allowed_float_range(immo1.floor, immo2.floor, self.difference_floor)
            || !passes_filter_strategy(
                (immo1.baujahr_category(), immo2.baujahr_category()),
                (
                    immo1.baujahr.map(|value| value as f64),
                    immo2.baujahr.map(|value| value as f64),
                ),
                self.difference_baujahr,
            )
        {
            return f64::INFINITY;
        }

        if immo1.objektunterart_category() == Some(3)
            && immo2.objektunterart_category() == Some(3)
            && !is_inside_of_allowed_integer_range(
                immo1.ausstattung,
                immo2.ausstattung,
                self.difference_ausstattungsnote,
            )
        {
            return f64::INFINITY;
        }

        if immo1.is_objektunterart_category_1_or_2()
            && immo2.is_objektunterart_category_1_or_2()
            && !passes_filter_strategy(
                (
                    immo1.grundstuecksgroesse_category(),
                    immo2.grundstuecksgroesse_category(),
                ),
                (immo1.grundstuecksgroesse, immo2.grundstuecksgroesse),
                self.difference_grundstuecksgroesse,
            )
        {
            return f64::INFINITY;
        }

        if immo1.objektunterart_category() == immo2.objektunterart_category()
            && !passes_filter_strategy(
                (immo1.wohnflaeche_category(), immo2.wohnflaeche_category()),
                (immo1.wohnflaeche, immo2.wohnflaeche),
                self.difference_wohnflaeche,
            )
        {
            return f64::INFINITY;
        }

        if let (Some(date1), Some(date2), Some(filter)) = (
            immo1.wertermittlungsstichtag,
            immo2.wertermittlungsstichtag,
            self.difference_wertermittlungsstichtag_days,
        ) {
            if date1.signed_duration_since(date2).num_days().abs() >= filter as i64 {
                return f64::INFINITY;
            }
        }

        if let (Some(distance), Some(filter)) = (
            immo1.plane_distance_squared(immo2),
            self.difference_spatial_distance_kilometer,
        ) {
            if distance > (filter as f64 * 1000f64).powi(2) {
                return f64::INFINITY;
            }
        }

        self.inner.dissimilarity(immo1, immo2)
    }
}

fn passes_filter_strategy<U>(
    (category1, category2): (Option<U>, Option<U>),
    (value1, value2): (Option<f64>, Option<f64>),
    filter_strategy_opt: Option<FilterStrategy>,
) -> bool
where
    U: Into<u64>,
{
    match filter_strategy_opt {
        Some(FilterStrategy::Continuous(filter)) => {
            is_inside_of_allowed_float_range(value1, value2, Some(filter))
        }
        Some(FilterStrategy::Category(filter)) => {
            is_inside_of_allowed_integer_range(category1, category2, Some(filter))
        }
        None => true,
    }
}

fn is_inside_of_allowed_integer_range<U>(
    value_opt1: Option<U>,
    value_opt2: Option<U>,
    filter_opt: Option<u64>,
) -> bool
where
    U: Into<u64>,
{
    is_inside_of_allowed_float_range(
        value_opt1.map(|value| Into::<u64>::into(value) as f64),
        value_opt2.map(|value| Into::<u64>::into(value) as f64),
        filter_opt.map(|filter| filter as f64),
    )
}

fn is_inside_of_allowed_float_range(
    value_opt1: Option<f64>,
    value_opt2: Option<f64>,
    filter_opt: Option<f64>,
) -> bool {
    if let Some(filter) = filter_opt {
        if let (Some(value1), Some(value2)) = (value_opt1, value_opt2) {
            if abs_diff(value1, value2) > filter {
                return false;
            }
        } else if value_opt1.xor(value_opt2).is_some() {
            return false;
        }
    }
    true
}

fn abs_diff<T>(a: T, b: T) -> T
where
    T: Sub<Output = T> + PartialOrd + Copy,
{
    if a > b {
        a - b
    } else {
        b - a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn none_filter_is_okay(value1 in prop::option::of(0u64..1000), value2 in prop::option::of(0u64..1000)) {
            prop_assert!(is_inside_of_allowed_integer_range(value1, value2, None));
        }

        #[test]
        fn none_filter_is_okay_strategy(
            value1 in prop::option::of(0f64..1000.0),
            value2 in prop::option::of(0f64..1000.0),
            cat1 in prop::option::of(0u64..1000),
            cat2 in prop::option::of(0u64..1000),
        ) {
            prop_assert!(passes_filter_strategy((cat1, cat2), (value1, value2), None));
        }

        #[test]
        fn both_values_none_is_okay(filter in prop::option::of(0u64..1000)) {
            prop_assert!(is_inside_of_allowed_integer_range::<u64>(None, None, filter));
        }

        #[test]
        fn both_values_none_is_okay_strategy(
            filter_cat in 0u64..1000,
            filter_val in 0f64..1000.0,
            value1 in prop::option::of(0f64..1000.0),
            value2 in prop::option::of(0f64..1000.0),
            cat1 in prop::option::of(0u64..1000),
            cat2 in prop::option::of(0u64..1000),
        ) {
            prop_assert!(passes_filter_strategy::<u64>((None, None), (value1, value2), Some(FilterStrategy::Category(filter_cat))));
            prop_assert!(passes_filter_strategy((cat1, cat2), (None, None), Some(FilterStrategy::Continuous(filter_val))));
            prop_assert!(passes_filter_strategy((cat1, cat2), (value1, value2), None));
        }

        #[test]
        fn exactly_one_value_none_is_bad(
            filter_value in 0u64..1000,
            value1 in 0u64..1000,
            value2 in 0u64..1000,
        ) {
            prop_assert!(!is_inside_of_allowed_integer_range(Some(value1), None, Some(filter_value)));
            prop_assert!(!is_inside_of_allowed_integer_range(None, Some(value2), Some(filter_value)));
        }

        #[test]
        fn exactly_one_value_none_is_bad_strategy(
            filter_val in 0f64..1000.0,
            value1 in 0f64..1000.0,
            value2 in 0f64..1000.0,
            cat1 in prop::option::of(0u64..1000),
            cat2 in prop::option::of(0u64..1000),
        ) {
            prop_assert!(!passes_filter_strategy((cat1, cat2), (Some(value1), None), Some(FilterStrategy::Continuous(filter_val))));
            prop_assert!(!passes_filter_strategy((cat1, cat2), (None, Some(value2)), Some(FilterStrategy::Continuous(filter_val))));
        }

        #[test]
        fn exactly_one_category_none_is_bad_strategy(
            filter_cat in 0u64..1000,
            cat1 in 0u64..1000,
            cat2 in 0u64..1000,
            value1 in prop::option::of(0f64..1000.0),
            value2 in prop::option::of(0f64..1000.0),
        ) {
            prop_assert!(!passes_filter_strategy((Some(cat1), None), (value1, value2), Some(FilterStrategy::Category(filter_cat))));
            prop_assert!(!passes_filter_strategy((None, Some(cat2)), (value1, value2), Some(FilterStrategy::Category(filter_cat))));
        }

        #[test]
        fn two_some_is_good_if_abs_diff_less_or_equal_to_filter(
            filter_value in 0u64..1000,
            value1 in 0u64..1000,
            value2 in 0u64..1000,
        ) {
            prop_assert!(is_inside_of_allowed_integer_range(Some(value1), Some(value2), Some(filter_value))
                == (abs_diff(value1, value2) <= filter_value));
        }

        #[test]
        fn two_some_is_good_if_abs_diff_less_or_equal_to_filter_value(
            filter_val in 0f64..1000.0,
            value1 in 0f64..1000.0,
            value2 in 0f64..1000.0,
            cat1 in prop::option::of(0u64..1000),
            cat2 in prop::option::of(0u64..1000),
        ) {
            prop_assert!(passes_filter_strategy((cat1, cat2), (Some(value1), Some(value2)), Some(FilterStrategy::Continuous(filter_val)))
                == (abs_diff(value1, value2) <= filter_val));
        }

        #[test]
        fn two_some_is_good_if_abs_diff_less_or_equal_to_filter_category(
            filter_cat in 0u64..1000,
            cat1 in 0u64..1000,
            cat2 in 0u64..1000,
            value1 in prop::option::of(0f64..1000.0),
            value2 in prop::option::of(0f64..1000.0),
        ) {
            prop_assert!(passes_filter_strategy((Some(cat1), Some(cat2)), (value1, value2), Some(FilterStrategy::Category(filter_cat)))
                == (abs_diff(cat1, cat2) <= filter_cat));
        }
    }
}
