#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]
#![allow(clippy::large_enum_variant)]

//! This crate contains the traits and implementation for dissimilarities.
pub use dissimilarity::*;
pub use expert::ExpertDissimilarity;
pub use lp_vector_dissimilarity::{
    LpVectorDissimilarity, NormalizingLpVectorDissimilarity, ScalingLpVectorDissimilarity,
    ScalingLpVectorDissimilarityConfig,
};

mod dissimilarity;
pub mod expert;
pub mod filtered;
mod lp_vector_dissimilarity;
