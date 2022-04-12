#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]

//! This crate contains the implementation for normalization of marktwert.
mod handcrafted;
pub use handcrafted::HandcraftedBerlin;

mod regression;
pub use regression::RegressionNormalizer;

mod local;
pub use local::*;

mod plz;
pub use plz::PlzNormalizer;

mod hpi;
pub use hpi::HpiNormalizer;

mod brazil;
pub use brazil::BrazilNormalizer;

mod japan;
pub use japan::JapanNormalizer;

mod utils;

use common::{BpResult, Immo, Normalizer, Trainable};

/// A [Normalizer] which composes different Normalizers.
/// This wrapper allows you to skip certain steps in the pipeline.
/// To make the last step skippable use [IdentityNormalizer].
/// # Example
/// ```
/// # use normalizers::*;
/// # let do_normalization = true;
/// let pipeline = if do_normalization {
///     NormalizationPipeline::with_two(HandcraftedBerlin, IdentityNormalizer)
/// } else {
///     NormalizationPipeline::with_one(IdentityNormalizer)
/// };
/// ```
#[derive(Debug, Clone)]
pub enum NormalizationPipeline<CurrentNormalizer, NextNormalizer> {
    /// The current Normalizer will be skipped.
    SimpleNormalizer(NextNormalizer),
    /// The current Normalizer has to be executed.
    ChainNormalizer(CurrentNormalizer, NextNormalizer),
}

impl<CurrentNormalizer, NextNormalizer> NormalizationPipeline<CurrentNormalizer, NextNormalizer> {
    /// Creates a new [NormalizationPipeline] which just executes `next_normalizer`.
    pub fn with_one(next_normalizer: NextNormalizer) -> Self {
        NormalizationPipeline::SimpleNormalizer(next_normalizer)
    }

    /// Creates a new [NormalizationPipeline] where `cur_normalizer` is executed before `next_normalizer`.
    pub fn with_two(cur_normalizer: CurrentNormalizer, next_normalizer: NextNormalizer) -> Self {
        NormalizationPipeline::ChainNormalizer(cur_normalizer, next_normalizer)
    }
}

impl<C: Trainable, N: Trainable> Trainable for NormalizationPipeline<C, N> {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        match self {
            NormalizationPipeline::SimpleNormalizer(inner) => inner.train(training_data),
            NormalizationPipeline::ChainNormalizer(current, next) => {
                let all_data: Vec<_> = training_data.into_iter().collect();
                current.train(all_data.iter().copied())?;
                next.train(all_data)
            }
        }
    }
}

impl<C: Normalizer, N: Normalizer> Normalizer for NormalizationPipeline<C, N> {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        match self {
            NormalizationPipeline::SimpleNormalizer(inner) => inner.normalize(immos),
            NormalizationPipeline::ChainNormalizer(current, next) => {
                let mut all_data: Vec<_> = immos.into_iter().collect();
                current.normalize(all_data.iter_mut().map(|i| &mut **i));
                next.normalize(all_data)
            }
        }
    }

    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        match self {
            NormalizationPipeline::SimpleNormalizer(inner) => inner.denormalize(immos),
            NormalizationPipeline::ChainNormalizer(current, next) => {
                let mut all_data: Vec<_> = immos.into_iter().collect();
                // order must be reversed here
                next.denormalize(all_data.iter_mut().map(|i| &mut **i));
                current.denormalize(all_data)
            }
        }
    }
}

/// A [Normalizer] which does nothing.
/// Might be useful in combination with [NormalizationPipeline].
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityNormalizer;

impl Trainable for IdentityNormalizer {}

impl Normalizer for IdentityNormalizer {
    fn normalize<'i>(&self, _immos: impl IntoIterator<Item = &'i mut Immo>) {}

    fn denormalize<'i>(&self, _immos: impl IntoIterator<Item = &'i mut Immo>) {}
}
