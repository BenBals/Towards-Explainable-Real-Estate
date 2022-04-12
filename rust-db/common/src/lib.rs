#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]

//! This crate contains everything which might be needed across different tasks inside our project.

mod error;

pub use error::{BpError, BpResult};

pub mod immo;
pub use immo::{Immo, ImmoBuilder};

mod traits;
pub use traits::*;

pub mod database;

pub mod logging;
pub mod util;
