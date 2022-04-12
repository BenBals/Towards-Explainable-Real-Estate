use crate::weight;
use crate::weight::config::{CrossoverConfig, MutationConfig};
use genevo::prelude::{GenomeBuilder, Genotype, Rng};
use std::fmt;

/// trait that rewrites Mul<f64> for arbitrary structs. Needed for scaling of differences.
/// Should be implemented for SubOutputs
pub trait Scalable {
    /// Mul<f64> function
    fn scale(self, scalar: f64) -> Self;
}

impl Scalable for f64 {
    fn scale(self, scalar: f64) -> Self {
        self * scalar
    }
}

impl Scalable for i64 {
    fn scale(self, scalar: f64) -> Self {
        (self as f64 * scalar).round() as i64
    }
}

impl<T> Scalable for Option<T>
where
    T: Scalable,
{
    fn scale(self, scalar: f64) -> Self {
        self.map(|value| value.scale(scalar))
    }
}

/// This is the central trait for our individuals. A weight is an object on which genetic operations
/// and a few helper functions are defined.
/// In particular weights support mutation, crossover and regeneration. See their respective
/// documentation.
pub trait Weight: Clone + fmt::Debug {
    /// the output of a subtraction. This needs to be a different type since bounded and other can
    /// not display certain differences.
    type SubOutput: Scalable;

    /// Mutate `self` using the given config.
    /// Note that the implementor is free to ignore the config or use whatever part they want to.
    /// # Warning
    /// For example: [BoundedFloat] interprets the `config.std_deviation` as a factor for the
    /// difference between minimum and maximum value, so the std deviation will in general not be
    /// `config.std_deviation`.
    fn mutate<R>(&mut self, _rng: &mut R, _config: &MutationConfig)
    where
        R: Rng + Sized,
    {
    }

    /// Breed to weights to obtain a new one. The new weight should be constructed out of its
    /// parents in one way or the other. Crossover should take the [CrossoverConfig] in to account
    /// if any of its attributes apply to the operation implemented.
    fn crossover<R>(&self, _other: &Self, _rng: &mut R, _config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        self.clone()
    }

    /// Like mutate, this function changes the current weight. But instead of depending on the current
    /// state it should become a new, completely random value.
    /// Think of this as initializing a value.
    fn regenerate<R>(&mut self, _rng: &mut R)
    where
        R: Rng + Sized,
    {
    }

    /// How different is this weight to another of the same type
    /// # Returns
    /// - a float in the range `[0, 1]`
    /// - the value should be `0` if and only if the weights are identical
    /// - should be symmetric
    fn normalized_distance(&self, other: &Self) -> f64;

    /// Some mathematical operators which are associative, distributive and *not* commutative
    /// An add (+) operator. Keep in mind it is not necessarily commutative
    fn add(&self, other: &Self::SubOutput) -> Self;

    /// Some mathematical operators which are associative, distributive and *not* commutative
    /// An subtract (-) operator. Keep in mind it is not necessarily commutative
    fn sub(&self, other: &Self) -> Self::SubOutput;
}

impl<W> Scalable for Vec<W>
where
    W: Scalable + Clone,
{
    fn scale(self, scalar: f64) -> Self {
        self.iter().map(|val| val.clone().scale(scalar)).collect()
    }
}

impl<W> Weight for Vec<W>
where
    W: Weight,
    <W as weight::traits::Weight>::SubOutput: Clone,
{
    type SubOutput = Vec<W::SubOutput>;

    /// This will mutate all constituent weights individually.
    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        self.iter_mut().for_each(|float| float.mutate(rng, config));
    }

    /// This will pick each element independently at random from one of the parents.
    /// # Panics
    /// - If both parents don't have the same length.
    fn crossover<R>(&self, other: &Self, rng: &mut R, _config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        assert_eq!(self.len(), other.len());
        self.iter()
            .zip(other.iter())
            .map(|(father_gene, mother_gene)| {
                if rng.gen_bool(0.5) {
                    mother_gene
                } else {
                    father_gene
                }
            })
            .cloned()
            .collect()
    }

    /// This is only defined if both vecs have the same length.
    /// # Returns
    /// - Average normalized distance of the elements at the same index
    /// # Panics
    /// - if the vecs are of different length
    fn normalized_distance(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len());
        self.iter()
            .zip(other.iter())
            .map(|(left, right)| left.normalized_distance(right))
            .sum::<f64>()
            / self.len() as f64
    }

    /// This will regenerate all constituent weights individually.
    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        self.iter_mut().for_each(|element| element.regenerate(rng));
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(v1, v2)| (v1.add(v2)))
            .collect()
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        self.iter()
            .zip(other.iter())
            .map(|(v1, v2)| (v1.sub(v2)))
            .collect()
    }
}

#[derive(Clone, PartialEq, Eq)]
/// This is a fully transparent wrapper that can contain any weight.
/// We need it to implement various `genevo` traits for every trait.
pub struct TrivialWeightWrapper<W: Weight>(pub W);

impl<W> Weight for TrivialWeightWrapper<W>
where
    W: Weight,
{
    type SubOutput = W::SubOutput;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        self.0.mutate(rng, config)
    }

    fn crossover<R>(&self, other: &Self, rng: &mut R, config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        TrivialWeightWrapper(self.0.crossover(&other.0, rng, config))
    }

    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        self.0.regenerate(rng)
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        self.0.normalized_distance(&other.0)
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        TrivialWeightWrapper(self.0.add(other))
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        self.0.sub(&other.0)
    }
}

impl<W: fmt::Debug + Weight> fmt::Debug for TrivialWeightWrapper<W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl<W: Default + Weight> Default for TrivialWeightWrapper<W> {
    fn default() -> Self {
        Self(W::default())
    }
}

impl<W> Genotype for TrivialWeightWrapper<W>
where
    W: Weight + Sync + Send + PartialEq,
{
    type Dna = ();
}

impl<W> GenomeBuilder<TrivialWeightWrapper<W>> for TrivialWeightWrapper<W>
where
    W: Weight + Sync + Send + Default + PartialEq,
{
    /// Using the given Rng, every value is created randomly
    /// index is the index of the individual being created
    fn build_genome<R>(&self, _index: usize, rng: &mut R) -> TrivialWeightWrapper<W>
    where
        R: Rng + Sized,
    {
        let mut new = self.clone();
        new.regenerate(rng);
        new
    }
}

impl Scalable for bool {
    fn scale(self, _scalar: f64) -> Self {
        self
    }
}

impl Weight for bool {
    type SubOutput = bool;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        if rng.gen_bool(config.mutation_rate) {
            *self = !*self;
        }
    }

    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        *self = rng.gen_bool(0.5);
    }

    /// # Returns
    /// - `0.0` if both bools are identical
    /// - `1.0` otherwise
    fn normalized_distance(&self, other: &Self) -> f64 {
        (self != other) as u8 as f64
    }

    fn add(&self, other: &Self) -> Self {
        self ^ other
    }

    fn sub(&self, other: &Self) -> Self {
        self ^ other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::bounded::{tests::bounded_float, Bounded};
    use proptest::prelude::*;
    use rand::thread_rng;

    prop_compose! {
        fn bounded_float_vec()(vec in prop::collection::vec(bounded_float(0.0, 1.0), 1..100)) -> Vec<Bounded<f64>> {
            vec
        }
    }

    prop_compose! {
        /// The resulting vectors will both be the same length. Thus they will serve as parents in
        /// crossover on vectors.
        fn parents()(mut mother in bounded_float_vec(), mut father in bounded_float_vec()) -> (Vec<Bounded<f64>>, Vec<Bounded<f64>>) {
            let smaller = std::cmp::min(mother.len(), father.len());
            mother.resize_with(smaller, || panic!("We never grow a parent"));
            father.resize_with(smaller, || panic!("We never grow a parent"));
            (mother, father)
        }
    }

    proptest! {
      #[test]
      fn constant_crossover(parent in bounded_float_vec()) {
          let child = parent.crossover(&parent, &mut thread_rng(), &CrossoverConfig::default());
          prop_assert_eq!(child, parent);
      }

      #[test]
      fn crossover_needs_same_length(mother in bounded_float_vec(), father in bounded_float_vec()) {
          prop_assume!(mother.len() != father.len());
          let result = std::panic::catch_unwind(|| mother.crossover(&father, &mut thread_rng(), &CrossoverConfig::default()));
          prop_assert!(result.is_err());
      }

      #[test]
      fn crossover_one_of_parents(parents in parents()) {
          let (mother, father) = parents;
          let child = mother.crossover(&father, &mut thread_rng(), &CrossoverConfig::default());
          child.iter().enumerate().for_each(|(idx, &value)| assert!(mother[idx] == value || father[idx] == value));
      }
    }
}
