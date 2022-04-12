//! Traits and structs that can be used as dissimilarity for clustering or prediciton

use algorithms::calculate_environment::MIN_DISTANCE_SQUARED;
use common::{Dissimilarity, Immo, Trainable};

/// Uses U for the cost_function.
/// Dissimilarity returns the squared difference in sqm prices.
#[derive(Debug, Clone, Copy)]
pub struct SqmPriceDissimilarity;

impl Trainable for SqmPriceDissimilarity {}

impl Dissimilarity for SqmPriceDissimilarity {
    /// Calculates the dissimilarity between `immo1` and `immo2` as defined in our formalism.
    /// # Panics
    /// If an [Immo] misses a sqm_price
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        let sqm_price1 = immo1
            .sqm_price()
            .expect("trying to calculate U for point without wohnflaeche or marktwert");
        let sqm_price2 = immo2
            .sqm_price()
            .expect("trying to calculate U for point without wohnflaeche or marktwert");
        let sqm_price_difference = sqm_price1 - sqm_price2;
        sqm_price_difference * sqm_price_difference
    }
}

/// A [Dissimilarity] which uses plane_distance^power as dissimilarity
/// Useful for dully
/// # Panics
/// When [dissimilarity] is called with any Immo that does not have a plane location it will panic.
#[derive(Debug, Clone, Copy)]
pub struct DistanceDissimilarity {
    power: f64,
}

impl DistanceDissimilarity {
    /// Creates a new DistanceDissimilarity with exponent 2
    pub fn new() -> Self {
        Self::with(2.0)
    }

    /// Creates a new DistanceDissimilarity with a custom exponent
    pub fn with(power: f64) -> Self {
        // this allows using `plane_distance_squared` without roots
        Self { power: power / 2.0 }
    }
}

impl Default for DistanceDissimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainable for DistanceDissimilarity {}

impl Dissimilarity for DistanceDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        if immo1 == immo2 {
            0.0
        } else {
            immo1
                .plane_distance_squared(immo2)
                .expect("trying to calculate plane distance for point without plane location")
                // every distance should be at least 10 meter to remove too big factors
                .max(MIN_DISTANCE_SQUARED)
                .powf(self.power)
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
/// A [Dissimilarity] which always returns a constant value except if the two [Immo]s are the same
/// Useful for testing
pub struct ConstantDissimilarity(f64);

impl ConstantDissimilarity {
    /// Creates a new ConstantDissimilarity with a given constant
    pub fn with(constant: f64) -> Self {
        Self(constant)
    }
}

impl Trainable for ConstantDissimilarity {}

impl Dissimilarity for ConstantDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        if immo1.id() == immo2.id() {
            0.0
        } else {
            self.0
        }
    }
}

/// A dissimilarity function that can be used when one has only a closure specifying the dissimilarity
pub struct ClosureDissimilarity {
    closure_dissimilarity: fn(&Immo, &Immo) -> f64,
}

impl ClosureDissimilarity {
    /// closure_dissimilarity is a closure calculating a dissimilarity
    pub fn with_closure(closure_dissimilarity: fn(&Immo, &Immo) -> f64) -> Self {
        Self {
            closure_dissimilarity,
        }
    }
}

impl Trainable for ClosureDissimilarity {}

impl Dissimilarity for ClosureDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        (self.closure_dissimilarity)(immo1, immo2)
    }
}

/// The dissimilarity is the angle between the two meta data arrays regared as vectors over R^k
#[derive(Default, Clone, Copy, Debug)]
pub struct CosineDissimilarity {}

impl Trainable for CosineDissimilarity {}

impl Dissimilarity for CosineDissimilarity {
    fn dissimilarity(&self, this: &Immo, other: &Immo) -> f64 {
        // See the [Wikipedia Article]() for the equation this implements.
        let a = this.meta_data_array();
        let b = other.meta_data_array();
        let sum_of_products = a
            .iter()
            .zip(b.iter())
            .filter_map(|(a_component, b_component)| {
                a_component.zip(*b_component).map(|(x, y)| x * y)
            })
            .sum::<f64>();
        let sum_of_squares_a = a
            .iter()
            .filter_map(|component| component.map(|inner| inner.powf(2.0)))
            .sum::<f64>();
        let sum_of_squares_b = b
            .iter()
            .filter_map(|component| component.map(|inner| inner.powf(2.0)))
            .sum::<f64>();

        if sum_of_squares_a == 0.0 || sum_of_squares_b == 0.0 {
            return f64::INFINITY;
        }

        // Note that this is exactly the reciprocal of the cosine similarity (not the missing dis-)
        (sum_of_squares_a * sum_of_squares_b).sqrt() / sum_of_products
    }
}
