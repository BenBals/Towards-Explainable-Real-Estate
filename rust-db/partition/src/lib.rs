#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]

//! This crate contains everything related directly to partitions.

mod partition;
pub use crate::partition::*;

mod contraction;

mod make_one_stable;
pub use make_one_stable::make_one_stable;

mod are_blocks_connected;
pub use are_blocks_connected::are_blocks_connected;
