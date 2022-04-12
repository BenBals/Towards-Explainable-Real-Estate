use std::{error::Error, fmt::Display};

/// This type gets used to be our catch all error.
/// We implement conversions for all Library errors to ease error management.
#[derive(Debug)]
pub enum BpError {
    /// Allows a generic Error message.
    StringBpError(String),
    /// Anticipated errors, may be rethrown with an additional error message
    RethrowBpError(String, Box<dyn Error>),
    /// All other library Errors get converted to this error.
    OtherBpError(Box<dyn Error>),
}

/// This type is our goto Result, as it allows us to convert between many different errors.
pub type BpResult<O> = Result<O, BpError>;

impl Display for BpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BpError::StringBpError(str) => str.fmt(f),
            BpError::RethrowBpError(str, err) => {
                str.fmt(f)?;
                " with: ".fmt(f)?;
                err.fmt(f)?;
                Ok(())
            }
            BpError::OtherBpError(err) => err.fmt(f),
        }
    }
}
impl Error for BpError {}

impl BpError {
    /// Allows to annotate a BPError with a to better detect the origin of errors.
    /// # Usage
    /// ```
    /// # use common::{BpError, BpResult};
    /// # fn fallible_function() -> BpResult<()> {
    /// # Err(BpError::StringBpError("".into()))
    /// # }
    /// # fn container_function() -> BpResult<()> {
    /// fallible_function().map_err(BpError::rethrow_with("function failed"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn rethrow_with(str: &'static str) -> impl Fn(BpError) -> BpError {
        move |err| BpError::RethrowBpError(str.to_string(), Box::new(err))
    }
}

macro_rules! implement_from {
    ($type:ty) => {
        impl From<$type> for BpError {
            fn from(other: $type) -> Self {
                BpError::OtherBpError(Box::from(other))
            }
        }
    };
}
implement_from!(mongodb::error::Error);
implement_from!(std::io::Error);
implement_from!(mongodb::bson::de::Error);
implement_from!(serde_json::Error);
implement_from!(mongodb::bson::oid::Error);
implement_from!(csv::Error);
implement_from!(reqwest::Error);
implement_from!(std::num::ParseFloatError);
implement_from!(linregress::Error);
implement_from!(serde_dhall::Error);

impl<'a> From<&'a str> for BpError {
    fn from(other: &'a str) -> Self {
        BpError::StringBpError(other.to_string())
    }
}
impl From<String> for BpError {
    fn from(other: String) -> Self {
        BpError::StringBpError(other)
    }
}
