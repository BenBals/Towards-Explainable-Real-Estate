//! A [Weight] which can be one of two different types. Supports switching between those types
use crate::weight::{
    config::{CrossoverConfig, MutationConfig},
    Scalable, Weight,
};
use genevo::prelude::Rng;
use itertools::Either;
use std::fmt::{Debug, Formatter};

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// A general purpose weight sum type.
/// It maintains values for both variants so that it can switch between variants and remember bounds and values.
/// `active` indicates which value should be used
pub struct EitherWeight<L, R> {
    left: L,
    right: R,
    active: ActiveSide,
}

/// Which side the EitherWeight should use
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ActiveSide {
    /// first weight
    Left,
    /// second weight
    Right,
}

impl<L, R> EitherWeight<L, R> {
    /// Creates a new Weight given both possible variants and which side should be active.
    pub fn new(left: L, right: R, active: ActiveSide) -> Self {
        Self {
            left,
            right,
            active,
        }
    }

    /// Gives the [ActiveSide].
    pub fn active(&self) -> ActiveSide {
        self.active
    }
    /// Gives a reference to the left value.
    pub fn left(&self) -> &L {
        &self.left
    }
    /// Gives a mutable reference to the left value
    pub fn left_mut(&mut self) -> &mut L {
        &mut self.left
    }
    /// Gives a reference to the right value
    pub fn right(&self) -> &R {
        &self.right
    }
    /// Gives a mutable reference to the right value
    pub fn right_mut(&mut self) -> &mut R {
        &mut self.right
    }
    /// Converts self to an [Either]. The either contains the value of the active side of self.
    pub fn active_value(&self) -> Either<&L, &R> {
        match self.active() {
            ActiveSide::Left => Either::Left(self.left()),
            ActiveSide::Right => Either::Right(self.right()),
        }
    }
    /// Converts self to a mutable [Either]. The either contains the value of the active side of self.
    pub fn active_value_mut(&mut self) -> Either<&mut L, &mut R> {
        match self.active() {
            ActiveSide::Left => Either::Left(self.left_mut()),
            ActiveSide::Right => Either::Right(self.right_mut()),
        }
    }
}

impl<L: Debug, R: Debug> Debug for EitherWeight<L, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.active {
            ActiveSide::Left => write!(f, "Left({:?})", self.left),
            ActiveSide::Right => write!(f, "Right({:?})", self.right),
        }
    }
}

impl<L, R> Scalable for EitherWeight<L, R>
where
    L: Scalable,
    R: Scalable,
{
    fn scale(self, scale: f64) -> Self {
        Self {
            left: self.left.scale(scale),
            right: self.right.scale(scale),
            active: self.active,
        }
    }
}

impl<L: Weight, R: Weight> Weight for EitherWeight<L, R> {
    type SubOutput = EitherWeight<L::SubOutput, R::SubOutput>;

    fn mutate<Rn>(&mut self, rng: &mut Rn, config: &MutationConfig)
    where
        Rn: Rng + Sized,
    {
        if rng.gen_bool(config.type_switch_probability) {
            self.active = match self.active {
                ActiveSide::Left => ActiveSide::Right,
                ActiveSide::Right => ActiveSide::Left,
            }
        }
        match self.active_value_mut() {
            Either::Left(value) => value.mutate(rng, config),
            Either::Right(value) => value.mutate(rng, config),
        }
    }

    fn crossover<Rn>(&self, other: &Self, rng: &mut Rn, config: &CrossoverConfig) -> Self
    where
        Rn: Rng + Sized,
    {
        match (self.active_value(), other.active_value()) {
            (Either::Left(value1), Either::Left(value2)) => EitherWeight {
                left: value1.crossover(value2, rng, config),
                ..self.clone()
            },
            (Either::Right(value1), Either::Right(value2)) => EitherWeight {
                right: value1.crossover(value2, rng, config),
                ..self.clone()
            },
            (_, _) => self.clone(),
        }
    }

    fn regenerate<Rn>(&mut self, rng: &mut Rn)
    where
        Rn: Rng + Sized,
    {
        self.left.regenerate(rng);
        self.right.regenerate(rng);
        self.active = if rng.gen_bool(0.5) {
            ActiveSide::Left
        } else {
            ActiveSide::Right
        };
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        match (self.active, other.active) {
            (ActiveSide::Left, ActiveSide::Left) => self.left.normalized_distance(&other.left),
            (ActiveSide::Right, ActiveSide::Right) => self.right.normalized_distance(&other.right),
            _ => 1.0,
        }
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        let mut new_self = self.clone();
        new_self.left = new_self.left.add(&other.left);
        new_self.right = new_self.right.add(&other.right);
        // We take the other activeSide since it is commonly used in the context that an
        // old solution adds a diff (namely a new solution subtracted by the old solution).
        // Because we want to set the active side to that of the new solution (the diff),
        // we use other.active here
        new_self.active = other.active;
        new_self
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        Self::SubOutput {
            left: self.left.sub(&other.left),
            right: self.right.sub(&other.right),
            // Here we use self.active (in contrast to add-method) because it is commonly used in
            // a way that a new solution subtracts an old solution. Since we want to keep the new
            // active side, we use self.active
            active: self.active,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::config::MutationConfigBuilder;
    use crate::weight::Bounded;
    use rand::thread_rng;

    #[test]
    fn left_active_gives_left() {
        let mut either = EitherWeight::new(0, 1, ActiveSide::Left);
        assert_eq!(either.active(), ActiveSide::Left);
        assert_eq!(either.left(), &0);
        assert_eq!(either.left_mut(), &mut 0);
        assert_eq!(either.active_value(), Either::Left(&0));
        assert_eq!(either.active_value_mut(), Either::Left(&mut 0));
    }

    #[test]
    fn right_active_gives_right() {
        let mut either = EitherWeight::new(0, 1, ActiveSide::Right);
        assert_eq!(either.active(), ActiveSide::Right);
        assert_eq!(either.right(), &1);
        assert_eq!(either.right_mut(), &mut 1);
        assert_eq!(either.active_value(), Either::Right(&1));
        assert_eq!(either.active_value_mut(), Either::Right(&mut 1));
    }

    #[test]
    fn mutate_flips() {
        let mut either = EitherWeight::new(
            Bounded::new(0.0, 0.0, 1.0),
            Bounded::new(0.0, 0.0, 1.0),
            ActiveSide::Right,
        );
        let config = MutationConfigBuilder::default()
            .type_switch_probability(0.5)
            .mutation_rate(0.5)
            .std_deviation(0.5)
            .build()
            .unwrap();
        let mut mutation_cnt = 0;
        while mutation_cnt < 100 && either.active == ActiveSide::Right {
            either.mutate(&mut thread_rng(), &config);
            mutation_cnt += 1;
        }
        assert_eq!(either.active(), ActiveSide::Left);

        mutation_cnt = 0;
        while mutation_cnt < 100 && either.active == ActiveSide::Left {
            either.mutate(&mut thread_rng(), &config);
            mutation_cnt += 1;
        }
        assert_eq!(either.active(), ActiveSide::Right);
    }
}
