use std::ops::{Deref, DerefMut};

use crate::{traits::Normalizer, traits::Trainable, BpResult, Dissimilarity, Immo};

/// This is a type erased version of [Trainable], which allows for dynamic dispatch.
/// ```
/// # use common::{Trainable, ErasedTrainable, BpResult, Immo};
/// fn do_stuff(mut t : impl Trainable) {
/// }
/// fn select_trainable_at_runtime() -> Box<dyn ErasedTrainable> {
///     struct DynamicTrainable;
///     impl Trainable for DynamicTrainable { }
///     Box::new(DynamicTrainable)
/// }
/// do_stuff(select_trainable_at_runtime());
/// ```
pub trait ErasedTrainable {
    /// Like [Trainable::train].
    fn erased_train<'i>(&mut self, _iter: &mut dyn Iterator<Item = &'i Immo>) -> BpResult<()> {
        Ok(())
    }
}

/// Allow for "Box<dyn Trainable>" and friends to be Trainable
impl<T: Trainable + ?Sized, D: DerefMut<Target = T>> Trainable for D {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        (**self).train(training_data)
    }
}

/// All Trainable can work with type erased iterators
impl<T: Trainable + ?Sized> ErasedTrainable for T {
    fn erased_train<'i>(&mut self, iter: &mut dyn Iterator<Item = &'i Immo>) -> BpResult<()> {
        self.train(iter)
    }
}

/// Make ErasedTrainable work as Trainable
impl Trainable for dyn ErasedTrainable {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.erased_train(&mut training_data.into_iter())
    }
}

/// This is a type erased version of [Normalizer], which allows for dynamic dispatch.
/// ```
/// # use common::{Trainable, Normalizer, ErasedNormalizer, BpResult, Immo};
/// fn do_stuff(mut t : impl Normalizer) {
/// }
/// fn select_normalizer_at_runtime() -> Box<dyn ErasedNormalizer> {
///     struct DynamicNormalizer;
///     impl Trainable for DynamicNormalizer { }
///     impl Normalizer for DynamicNormalizer {
///         fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {}
///         fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {}
///     }
///     Box::new(DynamicNormalizer)
/// }
/// do_stuff(select_normalizer_at_runtime());
/// ```
pub trait ErasedNormalizer: ErasedTrainable {
    /// Like [Normalizer::normalize].
    fn erased_normalize<'i>(&self, iter: &mut dyn Iterator<Item = &'i mut Immo>);
    /// Like [Normalizer::denormalize].
    fn erased_denormalize<'i>(&self, iter: &mut dyn Iterator<Item = &'i mut Immo>);
}

/// Allow for "Box<dyn Normalizer>" and friends to be a Normalizer
impl<N: Normalizer + ?Sized, D: Deref<Target = N> + Trainable> Normalizer for D {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        (**self).normalize(immos)
    }

    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        (**self).denormalize(immos)
    }
}

/// All Normalizer can work with type erased arguments
impl<N: Normalizer + ?Sized> ErasedNormalizer for N {
    fn erased_normalize<'i>(&self, iter: &mut dyn Iterator<Item = &'i mut Immo>) {
        self.normalize(iter);
    }

    fn erased_denormalize<'i>(&self, iter: &mut dyn Iterator<Item = &'i mut Immo>) {
        self.denormalize(iter);
    }
}

/// Make dyn ErasedNormalizer implement Normalizer
impl Trainable for dyn ErasedNormalizer {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.erased_train(&mut training_data.into_iter())
    }
}

impl Normalizer for dyn ErasedNormalizer {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        self.erased_normalize(&mut immos.into_iter());
    }

    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        self.erased_denormalize(&mut immos.into_iter());
    }
}

/// This is a type erased version of [Dissimilarity], which allows for dynamic dispatch.
pub trait ErasedDissimilarity: ErasedTrainable {
    /// Like [Dissimilarity::dissimilarity].
    fn erased_dissimilarity(&self, this: &Immo, other: &Immo) -> f64;
}

/// Allow for "Box<dyn Dissimilarity>" and friends to be a Dissimilarity
impl<Dis: ErasedDissimilarity + ?Sized, D: Deref<Target = Dis> + Trainable> Dissimilarity for D {
    fn dissimilarity(&self, this: &Immo, other: &Immo) -> f64 {
        (**self).erased_dissimilarity(this, other)
    }
}

/// All Dissimilarity can work with type erased arguments
impl<D: Dissimilarity + ?Sized> ErasedDissimilarity for D {
    fn erased_dissimilarity(&self, this: &Immo, other: &Immo) -> f64 {
        self.dissimilarity(this, other)
    }
}

/// Make dyn ErasedDissimilarity implement Dissimilarity
impl Trainable for dyn ErasedDissimilarity {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.erased_train(&mut training_data.into_iter())
    }
}

impl Dissimilarity for dyn ErasedDissimilarity {
    fn dissimilarity(&self, this: &Immo, other: &Immo) -> f64 {
        self.erased_dissimilarity(this, other)
    }
}

/// Make dyn ErasedDissimilarity + Sync implement Dissimilarity
impl Trainable for dyn ErasedDissimilarity + Sync {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.erased_train(&mut training_data.into_iter())
    }
}

impl Dissimilarity for dyn ErasedDissimilarity + Sync {
    fn dissimilarity(&self, this: &Immo, other: &Immo) -> f64 {
        self.erased_dissimilarity(this, other)
    }
}
