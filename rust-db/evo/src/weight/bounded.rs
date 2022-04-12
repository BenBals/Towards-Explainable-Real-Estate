//! This module is a wrapper around ordered types that ensures that these types stay in a given
//! subset of their domain.
use super::traits::*;
use crate::weight::config::MutationConfig;
use common::BpResult;
use genevo::random::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;
use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::fmt::Debug;

/// This is bounded data type that is restrained to be inside given bounds.
/// Both bounds are inclusive.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Bounded<T>
where
    T: Copy + PartialOrd,
{
    value: T,
    min: T,
    max: T,
}

/// A bounded ensured the contained value is always in a fixed interval.
impl<T> Bounded<T>
where
    T: Copy + PartialOrd,
{
    /// Creates a new bounded with the given value and bounds.
    /// # Panics
    /// - Iff the value is out of bounds
    pub fn new(value: T, min: T, max: T) -> Self {
        assert!(min <= value && value <= max);

        Self { value, min, max }
    }

    /// Returns the value inside the [Bounded]
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the minimum value this [Bounded] can take
    pub fn min(&self) -> T {
        self.min
    }

    /// Returns the maximum value this [Bounded] can take
    pub fn max(&self) -> T {
        self.max
    }

    /// Try to convert a vector of [Bounded]s into any type that can be constructed from
    /// a vector of `T`s.
    /// # Returns
    /// - An error of if and only if the conversion of the `Vec<T>` into the target type fails.
    /// - Otherwise an `Ok` of the target type.
    #[allow(clippy::ptr_arg)]
    pub fn vec_try_into<V>(vec: &Vec<Self>) -> Result<V, V::Error>
    where
        V: TryFrom<Vec<T>>,
    {
        vec.iter()
            .map(|bf| bf.value())
            .collect::<Vec<_>>()
            .try_into()
    }
}

impl<T> Bounded<T>
where
    T: Copy + PartialOrd + Debug,
{
    /// Attempts to set the value of this [Bounded] to the argument.
    /// # Returns
    /// - `Ok` if the operations succeeded.
    /// - An error if the new value was out of bounds.
    pub fn set_value(&mut self, new_value: T) -> BpResult<()> {
        if self.min <= new_value && new_value <= self.max {
            self.value = new_value;
            Ok(())
        } else {
            Err(format!(
                "New value {:?} was not in bounds [{:?}, {:?}]",
                new_value,
                self.min(),
                self.max()
            )
            .into())
        }
    }

    /// Set the value of this [Bounded] to the argument, clamped to the min and max of self.
    pub fn saturating_set_value(&mut self, new_value: T) {
        if self.min > new_value {
            self.value = self.min
        } else if self.max < new_value {
            self.value = self.max
        } else {
            self.value = new_value
        }
    }
}

impl<T> Debug for Bounded<T>
where
    T: Copy + PartialOrd + Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value().fmt(formatter)
    }
}

impl<T> PartialEq<Bounded<T>> for Bounded<T>
where
    T: Copy + PartialEq + PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> PartialOrd<Bounded<T>> for Bounded<T>
where
    T: Copy + PartialOrd,
{
    fn partial_cmp(&self, other: &Bounded<T>) -> Option<Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

impl Default for Bounded<f64> {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl Weight for Bounded<f64> {
    type SubOutput = f64;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        if rng.gen_bool(config.mutation_rate) {
            let normal_dist =
                Normal::new(self.value, (self.max - self.min) * config.std_deviation).unwrap();
            self.saturating_set_value(normal_dist.sample(rng));
        }
    }

    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        // The sampling below will panic if this is the case.
        // If min=max, there can be no change anyways, no regeneration is always a  a no-op.
        #[allow(clippy::float_cmp)]
        if self.min != self.max {
            self.value = rng.gen_range(self.min, self.max);
        }
    }

    /// # Returns
    /// - Difference between the values devided by span of the bounds
    /// # Panics
    /// - If at least one bound is different
    // We really want these floats not to be touchted at all, they should be bit-identical
    #[allow(clippy::float_cmp)]
    fn normalized_distance(&self, other: &Self) -> f64 {
        assert_eq!(self.min, other.min);
        assert_eq!(self.max, other.max);
        if self.max == self.min {
            return 0.0;
        }
        (self.value - other.value).abs() / (self.max - self.min)
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        // log::info!("Add f64 {:?}, {:?}", self, other);
        let mut new_self = *self;
        new_self.saturating_set_value(self.value + other);
        new_self
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        // log::info!("Sub f64 {:?}, {:?}", self, other);
        self.value - other.value
    }
}

impl Default for Bounded<i64> {
    fn default() -> Self {
        Bounded::new(0, 0, 0)
    }
}

impl Weight for Bounded<i64> {
    type SubOutput = i64;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        if rng.gen_bool(config.mutation_rate) {
            let normal_dist = Normal::new(
                self.value as f64,
                (self.max - self.min) as f64 * config.std_deviation,
            )
            .unwrap();
            self.saturating_set_value(normal_dist.sample(rng).round() as i64)
        }
    }

    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        // Se comment in f64 implementation
        if self.min != self.max {
            self.value = rng.gen_range(self.min, self.max);
        }
    }

    /// # Returns
    /// - Difference between the values devided by span of the bounds
    /// # Panics
    /// - If at least one bound is different
    fn normalized_distance(&self, other: &Self) -> f64 {
        assert_eq!(self.min, other.min);
        assert_eq!(self.max, other.max);
        if self.max == self.min {
            return 0.0;
        }
        (self.value as f64 - other.value as f64).abs() / (self.max - self.min) as f64
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        // log::info!("Add u64 {:?}, {:?}", self, other);
        let mut new_self = *self;
        new_self.saturating_set_value(self.value + other);
        new_self
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        // log::info!("Sub u64 {:?}, {:?}", self, other);
        self.value - other.value
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::weight::{
        config::{MutationConfig, MutationConfigBuilder},
        Weight,
    };
    use proptest::prelude::*;
    use rand::thread_rng;
    use std::cmp::min;

    fn example_config() -> MutationConfig {
        MutationConfigBuilder::default()
            .std_deviation(1.0)
            .mutation_rate(0.5)
            .type_switch_probability(0.5)
            .build()
            .unwrap()
    }

    #[test]
    #[should_panic]
    fn bounded_float_asserts_min_less_eq_max() {
        Bounded::new(1.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic]
    fn bounded_float_asserts_value_less_eq_max() {
        Bounded::new(2.0, 0.0, 0.0);
    }

    #[test]
    #[should_panic]
    fn bounded_float_asserts_min_less_eq_value() {
        Bounded::new(-1.0, 0.0, 0.0);
    }

    prop_compose! {
      pub fn bounded_float(min: f64, max: f64)(value in min..max) -> Bounded<f64> {
          Bounded::new(value, min, max)
      }
    }

    prop_compose! {
      pub fn bounded_float_random_bounds()(x in 0.0..1e9, y in 0.0..1e9, z in 0.0..1e9) -> Bounded<f64> {
          let mut parameters: Vec<f64> = vec![x,y,z];
          parameters.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
          Bounded::new(parameters[1], parameters[0], parameters[2])
      }
    }

    prop_compose! {
      pub fn bounded_float_random_upper_bound()(x in 0.0..1e9, y in 0.0..1e9) -> Bounded<f64> {
          let mut parameters: Vec<f64> = vec![x,y];
          parameters.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
          Bounded::new(parameters[0], 0.0, parameters[1])
      }
    }

    prop_compose! {
      pub fn bounded_int_random_upper_bound()(x in 0.0..1e9, y in 0.0..1e9) -> Bounded<i64> {
          let mut parameters: Vec<f64> = vec![x,y];
          parameters.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
          Bounded::new(parameters[0] as i64, 0, parameters[1] as i64)
      }
    }

    proptest! {
        #[test]
        fn mutation_stays_in_range(mut float in bounded_float_random_bounds()) {
            float.mutate(&mut thread_rng(), &example_config());
            prop_assert!(float.min() <= float.value() && float.value() <= float.max());
        }

        #[test]
        #[allow(clippy::float_cmp)]
        fn mutation_does_not_change_bounds(mut float in bounded_float_random_bounds()) {
            let min_before = float.min();
            let max_before = float.max();
            float.mutate(&mut thread_rng(), &example_config());
            prop_assert_eq!(float.min(), min_before);
            prop_assert_eq!(float.max(), max_before);
        }

        #[test]
        fn test_weight_neutrality_f64(test_object in bounded_float_random_upper_bound()) {
            assert!(test_object.sub(&test_object).abs() < f64::EPSILON);
        }

        #[test]
        fn test_neutrality_after_subtraction_f64(diff in 0.0..1e4, test_object1 in bounded_float_random_upper_bound()) {
            let mut test_object2 = test_object1;
            test_object2.set_value(test_object2.max.min(test_object2.value + diff))?;
            assert_eq!(test_object1.add(&test_object2.sub(&test_object1)), test_object2);
        }

        #[test]
        fn test_weight_neutrality_u64(test_object in bounded_int_random_upper_bound()) {
            assert!(test_object.sub(&test_object) == 0);
        }

        // test doesnt work since there are problems with subtraction with u64s. They should work for their purpose I hope...
        #[test]
        fn test_neutrality_after_subtraction_u64(diff in 0i64 ..1000i64, test_object1 in bounded_int_random_upper_bound()) {
            let mut test_object2 = test_object1;
            test_object2.set_value(min(test_object2.max, test_object2.value + diff))?;
            assert_eq!(test_object1.add(&test_object2.sub(&test_object1)), test_object2);
        }
    }
}
