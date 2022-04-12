#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]

//! This crate contains all our predictors.
pub use learning_dully::LearningDully;
pub use normalizing::NormalizingPredictor;
pub use single_block::SelectBlockPredictor;
pub use weighted_average::WeightedAveragePredictor;
pub use weighted_average::WeightedAveragePredictorBuilder;

mod learning_dully;
mod normalizing;
mod single_block;
pub mod weighted_average;

#[cfg(test)]
mod tests;
