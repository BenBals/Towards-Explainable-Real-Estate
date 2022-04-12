#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]

//! This crate contains all prediction infrastructure.
mod driver;
pub use driver::*;

mod traits;
pub use traits::*;
