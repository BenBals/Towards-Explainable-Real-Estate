use common::{BpResult, Immo, Trainable};

/// A **strategy** on how to define how *good* a predictor performs.
pub trait Evaluator {
    /// Describes how *good* should be measured.
    /// If the output should get written to json, it must derive serde::Serialize
    /// This is not enforced here since not all output need to be written to json
    type Output;

    /// This function gets pairs of real immo and predicted immo (in this order) and should determine
    /// how well they match.
    /// If no pairs are provided an Error can be returned.
    /// Implementations may define additional failure conditions.
    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output>;
}

/// A **strategy** on how to be predict missing fields on the entire validation data.
pub trait Predictor: Trainable {
    /// This function should set the missing fields on all supplied Immos.
    /// It should be ensured that `train` is called at least once called before this function is executed.
    /// Failing to do so can result in an error being returned.
    /// Implementations may define additional failure conditions.
    fn predict<'j>(&self, validation_data: impl IntoIterator<Item = &'j mut Immo>) -> BpResult<()>;
}
