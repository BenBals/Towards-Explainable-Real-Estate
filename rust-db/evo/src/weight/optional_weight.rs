use crate::weight::{
    config::{CrossoverConfig, MutationConfig},
    traits::Scalable,
    Weight,
};
use genevo::prelude::Rng;
use rand::thread_rng;
use std::fmt::{Debug, Formatter};

#[derive(
    Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
/// An Option which remembers its seed. This is useful for [Bounded]s
pub struct OptionalWeight<W> {
    seed: W,
    value: Option<W>,
    always_some: bool,
    always_none: bool,
}

impl<W> OptionalWeight<W>
where
    W: Weight + Clone,
{
    /// Create a [Self] with a given seed and value
    pub fn new(seed: W, value: Option<W>) -> Self {
        Self {
            seed,
            value,
            always_some: false,
            always_none: false,
        }
    }

    /// This object will never swap to some via mutation or regeneration
    pub fn set_always_none(&mut self) {
        self.always_some = false;
        self.always_none = true;
        self.value = None;
    }

    /// This object will never swap to none via mutation or regeneration
    pub fn new_always_some(seed: W, value: W) -> Self {
        Self {
            seed,
            value: Some(value),
            always_some: true,
            always_none: false,
        }
    }

    /// Returns a reference to the value currently contained in the [OptionalWeight]
    pub fn value(&self) -> &Option<W> {
        &self.value
    }

    /// Returns a *mutable* reference to the value currently contained in the [OptionalWeight]
    pub fn value_mut(&mut self) -> &mut Option<W> {
        &mut self.value
    }

    /// *Mutates* the [OptionalWeight]
    /// If the current value is some, it becomes none.
    /// If it is none, it becomes some by regenerating from the seed
    fn invert_variant<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        if self.always_some || self.always_none {
            return;
        }
        self.value = match self.value {
            None => {
                let mut new_val = self.seed.clone();
                new_val.regenerate(rng);
                Some(new_val)
            }
            Some(_) => None,
        }
    }
}

impl<W: Clone> OptionalWeight<W> {
    /// Use the same weight as current value and seed.
    pub fn with_value_from_seed(seed: W) -> Self {
        Self {
            seed: seed.clone(),
            value: Some(seed),
            always_some: false,
            always_none: false,
        }
    }
}

impl<W: Debug> Debug for OptionalWeight<W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.always_none {
            write!(f, "{:?} (always none)", self.value)?;
        } else {
            write!(f, "{:?}", self.value)?;
        }

        Ok(())
    }
}

impl<W: Weight + Clone + Default> Weight for OptionalWeight<W> {
    type SubOutput = Option<W::SubOutput>;

    fn mutate<R>(&mut self, rng: &mut R, config: &MutationConfig)
    where
        R: Rng + Sized,
    {
        if let Some(inner) = self.value.as_mut() {
            inner.mutate(rng, config);
        }
        if rng.gen_bool(config.type_switch_probability) && !self.always_some && !self.always_none {
            self.invert_variant(rng);
        }
    }

    fn crossover<R>(&self, other: &Self, rng: &mut R, config: &CrossoverConfig) -> Self
    where
        R: Rng + Sized,
    {
        match self.value.as_ref().zip(other.value.as_ref()) {
            Some((inner1, inner2)) => {
                let mut child = Self::new(
                    self.seed.clone(),
                    Some(inner1.crossover(inner2, rng, config)),
                );
                // If only one parent is always_none or always_some,
                // take that value. If they conflict, give preference to the left.
                if self.always_some && other.always_none || self.always_none && other.always_some {
                    child.always_some = self.always_some;
                    child.always_none = self.always_none;
                } else {
                    child.always_some = self.always_some || other.always_some;
                    child.always_none = self.always_none || other.always_none;
                }
                child
            }
            None => self.clone(),
        }
    }

    fn regenerate<R>(&mut self, rng: &mut R)
    where
        R: Rng + Sized,
    {
        if (rng.gen_bool(0.5) || self.always_some) && !self.always_none {
            let mut new = self.seed.clone();
            new.regenerate(rng);
            self.value = Some(new);
        } else {
            self.value = None;
        }
    }

    fn normalized_distance(&self, other: &Self) -> f64 {
        match (self.value.as_ref(), other.value.as_ref()) {
            (None, None) => 0.0,
            (Some(left), Some(right)) => left.normalized_distance(right),
            (_, _) => 1.0,
        }
    }

    fn add(&self, other: &Self::SubOutput) -> Self {
        if self.value.is_none() && other.is_none() {
            return self.clone();
        }
        let mut new_val = self.value().clone().unwrap_or({
            let mut def = self.seed.clone();
            def.regenerate(&mut thread_rng());
            def
        });
        new_val = other
            .as_ref()
            .map_or(new_val.clone(), |other_val| new_val.add(other_val));

        Self {
            seed: self.seed.clone(),
            value: Some(new_val),
            always_some: self.always_some,
            always_none: self.always_none,
        }
    }

    fn sub(&self, other: &Self) -> Self::SubOutput {
        if self.value.is_none() && other.value.is_none() {
            return None;
        }
        match self.value.clone() {
            None => other.sub(self).scale(-1.0),
            Some(weight) => other
                .value()
                .as_ref()
                .map_or(Some(weight.sub(&W::default())), |other_weight| {
                    Some(weight.sub(other_weight))
                }),
        }
    }
}
