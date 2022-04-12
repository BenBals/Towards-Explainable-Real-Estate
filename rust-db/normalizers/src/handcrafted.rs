use super::Normalizer;
use chrono::NaiveDate;
use common::{Immo, Trainable};
use std::cmp::{max, min};

/// Gives a Normalizer which is handcrafted for Berlin.
#[derive(Debug, Clone, Copy)]
pub struct HandcraftedBerlin;

impl HandcraftedBerlin {
    fn normalize_or_denormalize(&self, immo: &mut Immo, denormalize: bool) {
        let normalize_date: NaiveDate = NaiveDate::from_ymd(2019, 1, 1);

        let days_before_norm = immo
            .wertermittlungsstichtag
            .map_or(0, |wertermittlungsstichtag| {
                normalize_date
                    .signed_duration_since(wertermittlungsstichtag)
                    .num_days()
            });
        let days_before_norm = max(-365 * 2, min(days_before_norm, 365 * 3));

        let base = if days_before_norm >= 0 { 1.20 } else { 1.12 };
        let exp = days_before_norm as f64 / 365.0;
        let exp = if denormalize { -exp } else { exp };

        immo.marktwert = immo
            .marktwert
            .map(|marktwert| marktwert * f64::powf(base, exp));
    }
}

impl Trainable for HandcraftedBerlin {}
impl Normalizer for HandcraftedBerlin {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            self.normalize_or_denormalize(immo, false);
        }
    }

    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            self.normalize_or_denormalize(immo, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::proptest;

    use super::*;
    use test_helpers::*;

    proptest! {
        #[test]
        fn handcrafted_berlin_defines_bijection(mut immos in full_immos(128)) {
            let clone = immos.clone();

            let normalizer = HandcraftedBerlin;
            normalizer.normalize(&mut immos);
            normalizer.denormalize(&mut immos);

            prop_assert!(
                clone
                    .iter()
                    .zip(immos.iter())
                    .all(|(immo, clone)|
                        ((immo.marktwert.unwrap() - clone.marktwert.unwrap()) / clone.marktwert.unwrap()).abs() < 1e-9
                    )
            );
        }

        #[test]
        fn handcrafted_berlin_does_not_invent_marktpreise_on_normalization(mut immos in full_immos(128)) {
            let normalizer = HandcraftedBerlin;
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.normalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }

        #[test]
        fn handcrafted_berlin_does_not_invent_marktpreise_on_denormalization(mut immos in full_immos(128)) {
            let normalizer = HandcraftedBerlin;
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.denormalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }
    }
}
