use common::{BpResult, Immo};
use predictions::Evaluator;

/// This evaluator will simply collect the pairs of real and predicted immo and store them as output
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityEvaluator;

impl IdentityEvaluator {
    /// Creates an [IdentityEvaluator]
    pub fn new() -> Self {
        Self {}
    }
}

impl Evaluator for IdentityEvaluator {
    type Output = Vec<(Immo, Immo)>;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        Ok(pairs
            .into_iter()
            .map(|(immo1, immo2)| (immo1.clone(), immo2.clone()))
            .collect())
    }
}
