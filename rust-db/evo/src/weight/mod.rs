//! This module contains all things that the evolutionary algorithm is supposed to learn, that is
//! the individuals, their building blocks and how to perform genetic operations on them.
mod traits;
pub use traits::{Scalable, TrivialWeightWrapper, Weight};

pub mod bounded;
pub use bounded::Bounded;

pub mod lp_wap_weight;
pub use lp_wap_weight::LpWapWeight;

pub mod expert_weight;
pub use expert_weight::ExpertWeight;

pub mod filter_weight;
pub use filter_weight::FilterWeight;

mod optional_weight;
pub use optional_weight::OptionalWeight;

pub mod either_weight;
pub use either_weight::EitherWeight;

mod top_level;

pub mod config;

pub use top_level::{EvoPredictor, TopLevelWeight};
