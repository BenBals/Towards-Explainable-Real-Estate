//! This module provides abstraction over things that can be trained.

use crate::{BpResult, Immo};

/// Something that might need training.
pub trait Trainable {
    /// This function gets used to provide new training data.
    /// Implementations may define failure conditions.
    /// # Default Implementation
    /// The default implementation ignores the input and always returns Ok(()).
    fn train<'i>(&mut self, _training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        Ok(())
    }
}

/// Represents a strategy for normalization.
pub trait Normalizer: Trainable {
    /// Adjust some attribute e.g. marktwert for all supplied immos.
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>);
    /// Given adjusted Immos this function should set their attributes like before normalization.
    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>);
}

/// A trait for a type that can be regared as a point in 2D space
pub trait Pointlike {
    /// returns the first coordinate of a Pointlike
    fn x(&self) -> u64;
    /// returns the second coordinate of a Pointlike
    fn y(&self) -> u64;
}

/// A trait for a type that can be identified by a simpler key
/// Keys are unique identifiers: Two Keyed Objects with the same key are considered to be the same Object.
pub trait Keyed {
    /// The type of key that is used.
    type Key: Clone;
    /// returns the unique key of a Pointlike
    fn key(&self) -> Self::Key;
}

/// A measure of how dissimilar two Immos are.
pub trait Dissimilarity: Trainable {
    /// Gives how dissimilar two Immos are.
    /// This should always be non-negative
    /// # Panics
    /// Implementations may define conditions for panics.
    fn dissimilarity(&self, this: &Immo, other: &Immo) -> f64;
}

/// A cost measure for Immos in a [Partition].
pub trait CostFunction {
    /// Gives the cost for including two immos in a Partition.
    /// A negative value means that its good to include the Immos in the same Block.
    /// A positive value on the other hand means that the Immos should rather stay in different Blocks.
    /// ** This function must be symmetric. **
    /// # Panics
    /// Implementations may define conditions for panics.
    fn cost(&self, a: &Immo, b: &Immo) -> f64;
}

impl<P: Pointlike> Pointlike for &'_ P {
    fn x(&self) -> u64 {
        (**self).x()
    }

    fn y(&self) -> u64 {
        (**self).y()
    }
}

impl<K: Keyed> Keyed for &'_ K {
    type Key = K::Key;

    fn key(&self) -> Self::Key {
        (**self).key()
    }
}
