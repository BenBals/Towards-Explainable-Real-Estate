#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]
//! This crate contains structs that can be used as cost function for clustering or prediciton.

use common::{CostFunction, Immo};
use std::borrow::Borrow;
use std::marker::PhantomData;

mod dissimilarity_cost_function;
pub use dissimilarity_cost_function::DissimilarityCostFunction;

/// A cost function that caps the cost if the relative sqm_price deviation is too high.
/// Immos with relative deviation in sqm_price higher than `deviation` will be infinity
/// The cost function will stay symmetrical if the original cost function is symmetrical
/// # Example
/// ```
/// # use common::*;
/// # use assert_approx_eq::assert_approx_eq;
/// # use cost_functions::*;
/// let immo1 = ImmoBuilder::default().marktwert(100.0).wohnflaeche(10.0).build().unwrap();
/// let immo2 = ImmoBuilder::default().marktwert(200.0).wohnflaeche(10.0).build().unwrap();
/// let immo3 = ImmoBuilder::default().marktwert(105.0).wohnflaeche(10.0).build().unwrap();
/// let cost_function = CappedSpmPriceCostFunction::new(ConstantCostFunction::discrete(), 0.1);
/// assert!(cost_function.cost(&immo1, &immo2).is_infinite());
/// assert!(cost_function.cost(&immo2, &immo3).is_infinite());
/// assert_approx_eq!(cost_function.cost(&immo1, &immo3), 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CappedSpmPriceCostFunction<C: CostFunction> {
    cost_function: C,
    deviation: f64,
}

impl<C: CostFunction> CappedSpmPriceCostFunction<C> {
    /// Creates a new CappedSqmPriceCostFunction with a base cost function and a deviation
    pub fn new(cost_function: C, deviation: f64) -> Self {
        Self {
            cost_function,
            deviation,
        }
    }
}

impl<C: CostFunction> CostFunction for CappedSpmPriceCostFunction<C> {
    fn cost(&self, a: &Immo, b: &Immo) -> f64 {
        let (sqm_price_a, sqm_price_b) = (a.sqm_price().unwrap(), b.sqm_price().unwrap());
        if 1.0 - sqm_price_a.min(sqm_price_b) / sqm_price_a.max(sqm_price_b) > self.deviation {
            f64::INFINITY
        } else {
            self.cost_function.cost(a, b)
        }
    }
}

/// a cost function for a partition of LORs.
/// There are negative costs iff two points are in the same LOR
#[derive(Debug, Clone, Copy)]
pub struct LorCostFunction;
impl CostFunction for LorCostFunction {
    fn cost(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        immo1
            .plr_berlin
            .as_ref()
            .zip(immo2.plr_berlin.as_ref())
            .map_or(1.0, |(plr1, plr2)| if plr1 == plr2 { -1.0 } else { 1.0 })
    }
}

/// a cost function with additional LOR-constraint
/// if two points are in the same LOR, the given cost_function is used. Else the cost is infinity.
#[derive(Debug, Clone, Copy)]
pub struct LorOrDefaultCostFunction<C: CostFunction> {
    default_cost_function: C,
}
impl<C: CostFunction> CostFunction for LorOrDefaultCostFunction<C> {
    fn cost(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        immo1
            .plr_berlin
            .as_ref()
            .zip(immo2.plr_berlin.as_ref())
            .map_or(1.0, |(plr1, plr2)| {
                if plr1 == plr2 {
                    self.default_cost_function.cost(immo1, immo2)
                } else {
                    f64::INFINITY
                }
            })
    }
}

/// a constant cost function.
/// Useful for Testing.
#[derive(Debug, Clone, Copy)]
pub struct ConstantCostFunction(f64);
impl CostFunction for ConstantCostFunction {
    fn cost(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        if *immo1 == *immo2 {
            0.0
        } else {
            self.0
        }
    }
}
impl ConstantCostFunction {
    /// returns a [ConstantCostFunction] with a given Constant
    pub const fn with(a: f64) -> Self {
        Self(a)
    }
    /// a cost function defined as the metric in a [discrete space](https://www.wikiwand.com/en/Discrete_space).
    pub const fn discrete() -> Self {
        Self::with(1.0)
    }
    /// returns a [ConstantCostFunction] with constant negative cost
    pub const fn negative() -> Self {
        Self::with(-1.0)
    }
}

/// a Dissimilarity which uses a given closure as dissimilarity function
#[derive(Debug, Clone)]
pub struct ClosureCostFunction<F>
where
    F: Fn(&Immo, &Immo) -> f64,
{
    closure: F,
}

impl<F> ClosureCostFunction<F>
where
    F: Fn(&Immo, &Immo) -> f64,
{
    /// Creates a ClosureDissimilarity from a given closure
    pub fn new(closure: F) -> Self {
        Self { closure }
    }
}

impl<F> CostFunction for ClosureCostFunction<F>
where
    F: Fn(&Immo, &Immo) -> f64,
{
    fn cost(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        (self.closure)(immo1, immo2)
    }
}

/// A [CostFunction] which wraps another CostFunction in a way that allows dynamic as well as static dispatch.
/// Most of the time you will need to specify the type of the CostFunction.
/// To use dynamic dispatch use `dyn CostFunction + OtherTraits` as type.
/// # Example
/// The following snippet fails, however it can be made to compile using this wrapper.
/// ```compile_fail
/// # use std::borrow::Borrow;
/// # use common::*;
/// # use cost_functions::*;
/// # struct Partition {}
/// # impl Partition {
/// #     fn with_immos(epsilon: f64, cost : impl CostFunction, immos : Vec<impl Borrow<Immo>>) -> Self {
/// #         Partition {}
/// #     }
/// # }
/// # static cost_function: ConstantCostFunction = ConstantCostFunction::discrete();
/// fn choose_cost_function() -> &'static dyn CostFunction {
/// # &cost_function
/// }
/// Partition::with_immos(1.0, choose_cost_function(), vec![Immo::default()]);
/// ```
/// ```
/// # use std::borrow::Borrow;
/// # use common::*;
/// # use cost_functions::*;
/// # struct Partition {}
/// # impl Partition {
/// #     fn with_immos(epsilon: f64, cost : impl CostFunction, immos : Vec<impl Borrow<Immo>>) -> Self {
/// #         Partition {}
/// #     }
/// # }
/// # static cost_function: ConstantCostFunction = ConstantCostFunction::discrete();
/// fn choose_cost_function() -> &'static dyn CostFunction {
/// # &cost_function
/// }
/// Partition::with_immos(
///     1.0,
///     BorrowCostFunction::<dyn CostFunction, _>::new(choose_cost_function()),
///     vec![Immo::default()]
/// );
/// ```
#[derive(Debug)]
pub struct BorrowCostFunction<'cost, Inner: ?Sized, BorrowInner>(
    BorrowInner,
    PhantomData<&'cost Inner>,
);

impl<Inner: ?Sized, BorrowInner: Clone> Clone for BorrowCostFunction<'_, Inner, BorrowInner> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1)
    }
}

impl<Inner: ?Sized, BorrowInner> BorrowCostFunction<'_, Inner, BorrowInner> {
    /// Creates a new BorrowCostFunction from the given `Borrow<CostFunction>`.
    pub fn new(borrow_inner: BorrowInner) -> Self {
        Self(borrow_inner, PhantomData::default())
    }
}

impl<Inner: ?Sized + CostFunction, BorrowInner: Borrow<Inner>> CostFunction
    for BorrowCostFunction<'_, Inner, BorrowInner>
{
    fn cost(&self, a: &Immo, b: &Immo) -> f64 {
        self.0.borrow().cost(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use test_helpers::*;

    proptest! {
        #[test]
        fn capped_sqm_price_cost_function_is_symmetrical(immo1 in full_immo(), immo2 in full_immo()) {
            let cost_function = CappedSpmPriceCostFunction::new(ConstantCostFunction::discrete(), 0.1);
            prop_assert!(
                cost_function.cost(&immo1, &immo2).is_finite() == cost_function.cost(&immo2, &immo1).is_finite()
            );
        }
    }
}
