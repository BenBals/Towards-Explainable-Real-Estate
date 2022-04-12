#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]

//! This crate contains all generic algorithms for our project.

pub mod calculate_environment;

pub mod sweepline;

pub mod segment_tree;

mod distance_approximation;
pub use distance_approximation::QuadraticApproximation;
