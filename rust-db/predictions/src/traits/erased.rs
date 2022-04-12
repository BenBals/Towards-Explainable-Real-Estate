use std::ops::Deref;

use common::{BpResult, ErasedTrainable, Immo, Trainable};

use crate::{Evaluator, Predictor};

/// This is a type erased version of [Predictor], which allows for dynamic dispatch.
/// ```
/// # use common::{Trainable, BpResult, Immo};
/// # use predictions::{Predictor, ErasedPredictor};
/// fn do_stuff(mut t : impl Predictor) {
/// }
/// fn select_predictor_at_runtime() -> Box<dyn ErasedPredictor> {
///     struct DynamicPredictor;
///     impl Trainable for DynamicPredictor { }
///     impl Predictor for DynamicPredictor {
///         fn predict<'j>(&self, validation_data: impl IntoIterator<Item = &'j mut Immo>) -> BpResult<()> {
///             Ok(())
///         }
///     }
///     Box::new(DynamicPredictor)
/// }
/// do_stuff(select_predictor_at_runtime());
/// ```
pub trait ErasedPredictor: ErasedTrainable {
    /// Like [Predictor::predict].
    fn erased_predict<'j>(
        &self,
        validation_data: &mut dyn Iterator<Item = &'j mut Immo>,
    ) -> BpResult<()>;
}

/// Allow for "Box<dyn Predictor>" and friends to be a Predictor
impl<N: Predictor + ?Sized, D: Deref<Target = N> + Trainable> Predictor for D {
    fn predict<'j>(&self, validation_data: impl IntoIterator<Item = &'j mut Immo>) -> BpResult<()> {
        (**self).erased_predict(&mut validation_data.into_iter())
    }
}

/// All Predictor can work with type erased arguments
impl<P: Predictor + ?Sized> ErasedPredictor for P {
    fn erased_predict<'j>(
        &self,
        validation_data: &mut dyn Iterator<Item = &'j mut Immo>,
    ) -> BpResult<()> {
        self.predict(validation_data)
    }
}

/// Make dyn ErasedPredictor implement Predictor
impl Trainable for dyn ErasedPredictor {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.erased_train(&mut training_data.into_iter())
    }
}

impl Predictor for dyn ErasedPredictor {
    fn predict<'j>(&self, validation_data: impl IntoIterator<Item = &'j mut Immo>) -> BpResult<()> {
        self.erased_predict(&mut validation_data.into_iter())
    }
}

/// This is a type erased version of [Evaluator], which allows for dynamic dispatch.
/// The type of Output must be known at compile time.
/// ```
/// # use common::{Trainable, BpResult, Immo};
/// # use predictions::{Evaluator, ErasedEvaluator};
/// fn do_stuff(mut t : impl Evaluator<Output = f64>) {
/// }
/// fn select_evaluator_at_runtime() -> Box<dyn ErasedEvaluator<Output = f64>> {
///     struct DynamicEvaluator;
///     impl Evaluator for DynamicEvaluator {
///         type Output = f64;
///         fn evaluate<'i>(
///             &self,
///             _pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
///         ) -> BpResult<Self::Output> {
///             Ok(0.0)
///         }
///     }
///     Box::new(DynamicEvaluator)
/// }
/// do_stuff(select_evaluator_at_runtime());
/// ```
pub trait ErasedEvaluator {
    /// Like [Evaluator::Output].
    type Output;

    /// Like [Evaluator::evaluate].
    fn erased_evaluate<'i>(
        &self,
        pairs: &mut dyn Iterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output>;
}

/// Allow for "Box<dyn Evaluator>" and friends to be a Evaluator
impl<N: Evaluator + ?Sized, D: Deref<Target = N>> Evaluator for D {
    type Output = N::Output;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        (**self).evaluate(pairs)
    }
}

/// All Evaluators can work with type erased arguments
impl<E: Evaluator + ?Sized> ErasedEvaluator for E {
    type Output = E::Output;

    fn erased_evaluate<'i>(
        &self,
        pairs: &mut dyn Iterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        self.evaluate(pairs)
    }
}

/// Make dyn ErasedEvaluator implement Evaluator
impl<Out> Evaluator for dyn ErasedEvaluator<Output = Out> {
    type Output = Out;

    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        self.erased_evaluate(&mut pairs.into_iter())
    }
}

impl<Out> Evaluator for dyn ErasedEvaluator<Output = Out> + Sync {
    type Output = Out;
    fn evaluate<'i>(
        &self,
        pairs: impl IntoIterator<Item = (&'i Immo, &'i Immo)>,
    ) -> BpResult<Self::Output> {
        self.erased_evaluate(&mut pairs.into_iter())
    }
}
