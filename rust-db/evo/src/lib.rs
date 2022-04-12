#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]
#![cfg_attr(feature = "strict", deny(warnings))]

//! This crate contains everything related to the use of an evo

pub mod evolutionary;
pub mod fitness_functions;
pub mod local_search;
pub mod weight;
