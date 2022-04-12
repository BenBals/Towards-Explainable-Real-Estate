//! This module contains helper functions for the calculations of U
#![allow(non_snake_case)]
use crate::sweepline;
use common::{Dissimilarity, Immo};
use indicatif::ProgressBar;
use mongodb::bson::oid::ObjectId;
use std::borrow::Borrow;
use std::collections::HashMap;

/// the minimal distance two different [Immo]s should ever have (in meters)
pub const MIN_DISTANCE: f64 = 10.0;
/// the minimal squared distance two different [Immo]s should ever have (in meters)
pub const MIN_DISTANCE_SQUARED: f64 = MIN_DISTANCE * MIN_DISTANCE;

// 10km in meters
const RADIUS: u64 = 10 * 1000;

/// Calculates the squared distance between `immo1` and `immo2`.
/// If we have `immo1 == immo2` this function returns `0.0`.
/// Otherwise if the distance is less than some small threshold this threshold is returned instead.
/// This helps to avoid issues with really close immos.
pub fn capped_squared_distance(immo1: &Immo, immo2: &Immo) -> f64 {
    if immo1 == immo2 {
        return 0.0;
    }
    immo1.plane_distance_squared(immo2)
        .expect("trying to calculate U for point without plane location Hint: run plane location calculation job")
        .max(MIN_DISTANCE_SQUARED)
}

#[allow(non_snake_case)]
/// Calculates U for `immo` w.r.t. `neighbors` as defined in our formalism.
pub fn calculate_U<I: Borrow<Immo>, D: Dissimilarity>(
    immo: I,
    neighbors: &[I],
    dissimilarity: &D,
) -> Option<f64> {
    calculate_weighted_average(
        neighbors,
        |&other| dissimilarity.dissimilarity(immo.borrow(), other.borrow()),
        |&other| 1.0 / capped_squared_distance(immo.borrow(), other.borrow()),
    )
}

/// This function calculates a weighted average oversome some values.
/// Let W be the sum of all weights, then this function computes
/// \Sum_{v \in Values} \frac{weight(v)}{W} value(v).
/// If `W` is 0 then NaN might be returned.
/// # Returns
/// None if `values` produces no values.
pub fn calculate_weighted_average<'i, T: 'i, V, F>(
    values: impl IntoIterator<Item = T>,
    value_function: V,
    weight_function: F,
) -> Option<f64>
where
    V: Fn(&T) -> f64,
    F: Fn(&T) -> f64,
{
    let opt = values
        .into_iter()
        .map(|immo| (value_function(&immo), weight_function(&immo)))
        .fold(None, |value_opt, (cur_value, cur_weight)| match value_opt {
            Some((weighted_sum, weight_total)) => Some((
                weighted_sum + cur_weight * cur_value,
                weight_total + cur_weight,
            )),
            None => Some((cur_weight * cur_value, cur_weight)),
        });
    opt.map(|(weighted_sum, weight_total)| weighted_sum / weight_total)
}

/// This function calculates a weighted median over some valued and weighted objects.
/// That is if x_1, ..., x_n are the objects *sorted by their value*,
/// then the output is the value of the item x_i such that i is the smallest with
/// Sum_{j \le i} weight(x_j) >= (Sum_{j \le n} weight(x_j))/2
/// # Returns
/// - None if `values` produces no values.
pub fn calculate_weighted_median<'i, T: 'i, V, F>(
    objects: impl IntoIterator<Item = T>,
    value_function: V,
    weight_function: F,
) -> Option<f64>
where
    V: Fn(&T) -> f64,
    F: Fn(&T) -> f64,
{
    let mut values_weights: Vec<(f64, f64)> = objects
        .into_iter()
        .map(|object| (value_function(&object), weight_function(&object)))
        .collect();

    values_weights.sort_unstable_by(|value_weight1, value_weight2| {
        value_weight1.0.partial_cmp(&value_weight2.0).unwrap()
    });

    if values_weights.is_empty() {
        return None;
    }

    let total_weight_sum = values_weights
        .iter()
        .map(|value_weight| value_weight.1)
        .sum::<f64>();
    let mut current_weight_sum = 0.0;
    let mut index = 0;

    while index < values_weights.len() {
        current_weight_sum += values_weights[index].1;
        if current_weight_sum < total_weight_sum / 2.0 {
            index += 1;
        } else {
            break;
        }
    }

    Some(values_weights[index].0)
}

/// Builds a lookup table where for each of the given immos U is calculated.
/// # Returns
/// A HashMap where for each immos ObjectId the respective U value is saved.
pub fn calculate_U_for_immos<D: Dissimilarity>(
    dissimilarity: &D,
    immos: &[&Immo],
) -> HashMap<ObjectId, f64> {
    let mut id_immo_map = HashMap::new();
    for immo in immos.iter() {
        id_immo_map.insert(immo.id(), *immo);
    }

    let mut U_values: HashMap<ObjectId, f64> = HashMap::new();

    let bar = ProgressBar::new(immos.len() as u64);

    sweepline::for_every_close_point_do(
        &immos.iter().collect::<Vec<_>>(),
        RADIUS,
        |key, neighbors| {
            bar.inc(1);
            let neighbour_refs: Vec<_> = neighbors
                .iter()
                .map(|neighbor| *id_immo_map.get(&neighbor).unwrap())
                .collect::<Vec<_>>();
            U_values.insert(
                key.clone(),
                calculate_U(
                    *id_immo_map.get(&key).unwrap(),
                    &neighbour_refs,
                    dissimilarity,
                )
                .unwrap_or_default(),
            );
        },
    );

    U_values
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use common::{Immo, ImmoBuilder};
    use dissimilarities::SqmPriceDissimilarity;

    fn example_immo_1() -> Immo {
        ImmoBuilder::default()
            .id_from_string("5dde57b4c6c36b3bb4fde121")
            .marktwert(5.0)
            .wohnflaeche(100.0)
            .plane_location((1.0, 1.0))
            .build()
            .unwrap()
    }

    fn example_immo_2() -> Immo {
        ImmoBuilder::default()
            .id_from_string("5edd91d742936b07d4652e4d")
            .marktwert(420.0)
            .wohnflaeche(100.0)
            .plane_location((1.0, 11.0))
            .build()
            .unwrap()
    }

    #[test]
    fn dissimilarity_example() {
        let immo1 = example_immo_1();
        let immo2 = example_immo_2();
        assert_approx_eq!(
            SqmPriceDissimilarity.dissimilarity(&immo1, &immo2),
            4.15 * 4.15
        )
    }

    #[test]
    fn single_neighbor_U() {
        let immo = example_immo_1();
        let neighbor = example_immo_2();

        assert_approx_eq!(
            calculate_U(&immo, &[&neighbor], &SqmPriceDissimilarity).unwrap(),
            SqmPriceDissimilarity.dissimilarity(&immo, &neighbor)
        );
    }

    #[test]
    fn weighted_average_example() {
        let values = vec![1, 2, 3];

        assert_approx_eq!(
            calculate_weighted_average(
                &values,
                |value| **value as f64,
                |value| (3 - *value) as f64
            )
            .unwrap(),
            4.0 / 3.0
        );
    }

    #[test]
    fn weighted_average_None_example() {
        let values: Vec<f64> = vec![];

        assert!(
            calculate_weighted_average(&values, |value| **value, |value| (3.0 - *value)).is_none()
        );
        assert!(
            calculate_weighted_median(&values, |value| **value, |value| (3.0 - *value)).is_none()
        );
    }

    #[test]
    fn weighted_median_examples() {
        let values = vec![1, 2, 3];

        assert_approx_eq!(
            calculate_weighted_median(&values, |value| **value as f64, |value| (3 - *value) as f64)
                .unwrap(),
            1.0
        );

        assert_approx_eq!(
            calculate_weighted_median(
                &values,
                |value| **value as f64,
                |value| if **value == 3 { 1.0 } else { 0.0 }
            )
            .unwrap(),
            3.0
        );
    }
}
