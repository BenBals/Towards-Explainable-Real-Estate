use super::utils::*;
use common::{BpResult, Immo, Normalizer, Trainable};
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use std::collections::HashMap;

/// This normalizer will calculate a regression of `marktwert` over time for every kreis area in Germany.
/// Normalizing means projecting all valuations to a set date. Denormalizing means taking
/// valuations as if they were at the reference date and converting them to values for the
/// `werteermittungsstichtag`.
#[derive(Clone, Debug)]
pub struct RegressionNormalizer {
    regressions: Option<HashMap<String, f64>>,
    skip_outlier_normalizations: bool,
}

impl RegressionNormalizer {
    /// Create an untrained [RegressionNormalizer].
    /// To use it, first call [Trainable::train] and then [Normalizer::normalize].
    pub fn new() -> Self {
        Self {
            regressions: None,
            skip_outlier_normalizations: true,
        }
    }

    /// If this is set to true (the default), then all normalizations outside a reasonable spectrum
    /// are not performed.
    pub fn should_skip_outlier_normalizations(&mut self, should: bool) {
        self.skip_outlier_normalizations = should;
    }

    fn normalization_factor(&self, immo: &Immo) -> BpResult<f64> {
        if self.regressions.is_none() {
            return Err("was not trained".into());
        }

        if immo.wertermittlungsstichtag.is_none()
            || immo.ags0.is_none()
            || !is_reasonable_date(immo.wertermittlungsstichtag.unwrap())
        {
            return Ok(1.0);
        }

        // The regression slope gives the change in log-ed marktwert per day. We exponentiate with 2
        // to get change in marktwert. Then we take that to the power of days we want to shift the
        // price by, to get the change for that timeframe.
        let factor = f64::powf(
            self.regressions
                .as_ref()
                .unwrap()
                .get(immo.ags0.as_ref().unwrap())
                .unwrap_or(&0.0)
                .exp2(),
            -days_until_reference_date(immo.wertermittlungsstichtag.unwrap()),
        );

        if 0.05 >= factor || factor >= 10.0 {
            // log::warn!(
            //     "Outlier normalization factor {} for immo {:?}",
            //     factor,
            //     immo,
            // );

            if self.skip_outlier_normalizations {
                return Ok(1.0);
            }
        }

        Ok(factor)
    }
}

impl Default for RegressionNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainable for RegressionNormalizer {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        let mut area_map: HashMap<String, Vec<&Immo>> = HashMap::new();
        self.regressions = Some(HashMap::new());
        for immo in training_data.into_iter().filter(|immo| {
            immo.wertermittlungsstichtag.is_some() &&
         // if an immo is too old, the date is probably a typo and would ruin the regression
         is_reasonable_date(immo.wertermittlungsstichtag.unwrap())
        }) {
            if let Some(marktwert) = immo.marktwert {
                if marktwert.log2().is_infinite() {
                    return Err(
                        format!("Immo has invalid marktwert for normalization {:?}", immo).into(),
                    );
                }
            }

            if let Some(area) = immo.ags0.clone() {
                if let Some(area_vec) = area_map.get_mut(&area) {
                    area_vec.push(immo);
                } else {
                    area_map.insert(area, vec![immo]);
                }
            }
        }

        if area_map.is_empty() {
            return Err("No valid training data was supplied. Need marktwert, wertermittlungsstichtag and AGS_0.".into());
        }

        for (area, immos) in area_map {
            log::debug!("Performing regression for area {}...", area);
            let num_immos = immos.len();
            log::debug!("\tnum_immos={}", num_immos);

            if num_immos < 3 {
                log::warn!(
                    "Could not calculate regression for area {}, not enough immos",
                    area
                );
                continue;
            }

            let ages: Vec<f64> = immos
                .iter()
                .map(|immo| days_until_reference_date(immo.wertermittlungsstichtag.unwrap()))
                .collect();

            let marktwerte: Vec<_> = immos
                .iter()
                .map(|&immo| immo.marktwert.unwrap().log2())
                .collect();

            let formula = "marktwert ~ age";

            let data = match RegressionDataBuilder::new()
                .build_from(vec![("marktwert", marktwerte), ("age", ages)])
            {
                Ok(data) => data,
                Err(error @ linregress::Error::RegressionDataError(_)) => {
                    log::warn!(
                        "Could not build regression data for area {} with error {}, skipping.",
                        area,
                        error
                    );
                    continue;
                }
                Err(error) => {
                    panic!(
                        "Could not build regression data for area {} with error {}",
                        area, error
                    );
                }
            };

            let model = FormulaRegressionBuilder::new()
                .data(&data)
                .formula(formula)
                .fit()?;

            log::debug!("Performing regression for area {}...DONE", area);
            log::debug!("\tparameters: {:?}", model.parameters);

            self.regressions
                .as_mut()
                .unwrap()
                .insert(area, model.parameters.regressor_values[0]);
        }

        Ok(())
    }
}

impl Normalizer for RegressionNormalizer {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            immo.marktwert = immo
                .marktwert
                .map(|marktwert| marktwert * self.normalization_factor(immo).unwrap())
        }
    }

    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            immo.marktwert = immo
                .marktwert
                .map(|marktwert| marktwert / self.normalization_factor(immo).unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bson::doc;
    use common::database::read_reasonable_immos;
    use common::immo::ImmoBuilder;
    use common::logging::init_test_logging;
    use proptest::prelude::*;
    use proptest::proptest;
    use std::iter::once;
    use test_helpers::*;

    #[test]
    #[ignore]
    // This test reads from the database, so it's not enabled by default.
    // You can run it by passing `--ignored` to `cargo test`
    fn no_panic_on_default_usage() -> BpResult<()> {
        init_test_logging();
        let projection = Some(doc! {
            "marktwert": true,
            "wertermittlungsstichtag": true,
            "AGS_0": true
        });
        let mut immos = read_reasonable_immos(Some(1000), None, projection)?;
        let mut normalizer = RegressionNormalizer::new();
        normalizer.train(&immos)?;
        normalizer.normalize(immos.iter_mut());

        Ok(())
    }

    #[test]
    #[should_panic]
    fn regression_normalizer_must_train() {
        let normalizer = RegressionNormalizer::new();
        let mut immos = vec![ImmoBuilder::default()
            .marktwert(1.0)
            .wohnflaeche(1.0)
            .plane_location((0.0, 0.0))
            .u(1.0)
            .build()
            .unwrap()];
        normalizer.normalize(&mut immos);
    }

    proptest! {
        #[test]
        fn regression_defines_bijection(mut immos in full_immos(128)) {
            // We need more than one immos in an area to have a meaningful normalization.
            for immo in immos.iter_mut() {
                immo.ags0 = Some("FunnyMcStringFace".into());
            }

            let clone = immos.clone();

            let mut normalizer = RegressionNormalizer::new();
            normalizer.train(&immos).expect("Training failed in proptest.");
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
        fn regression_does_not_invent_marktpreise_on_normalization(mut immos in full_immos(128)) {
            let normalizer = RegressionNormalizer::new();
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.normalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }

        #[test]
        fn regression_does_not_invent_marktpreise_on_denormalization(mut immos in full_immos(128)) {
            let normalizer = RegressionNormalizer::new();
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.denormalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }

        #[test]
        fn normalize_line_between_two_immos(mut to_normalize in full_immo(), mut training_immo in full_immo()) {
            // This test takes two immos. The training immo is placed at the reference date.
            // If we now do regresison with those two, the line will go exactly bewteen those two points.
            // Thus if we now normalize the to_normalize immo to the reference date, it's marktwert will
            // exactly match that of training immo.
            // This behavior is asserted in this test.
            to_normalize.ags0 = training_immo.ags0.clone();
            training_immo.wertermittlungsstichtag = Some(reference_date());
            prop_assume!(to_normalize.wertermittlungsstichtag != Some(reference_date()));


            // We can only perform regression with at least three immos
            let mut extra_train_immo = training_immo.clone();
            extra_train_immo.marktwert = extra_train_immo.marktwert.map(|marktwert| marktwert + 1e-9);

            let mut normalizer = RegressionNormalizer::new();
            normalizer.should_skip_outlier_normalizations(false);

            normalizer.train(once(&to_normalize).chain(once(&training_immo).chain(once(&extra_train_immo)))).unwrap();
            normalizer.normalize(once(&mut to_normalize));

            prop_assert!((to_normalize.marktwert.unwrap() - training_immo.marktwert.unwrap()).abs() / training_immo.marktwert.unwrap() < 1e-3);
        }

        #[test]
        fn normalize_line_between_two_immos_recognizes_outlier(mut to_normalize in full_immo(), mut training_immo in full_immo()) {
            to_normalize.ags0 = training_immo.ags0.clone();
            training_immo.wertermittlungsstichtag = Some(reference_date());
            prop_assume!(to_normalize.wertermittlungsstichtag != Some(reference_date()));
            // Make that target ridiculously larger than the source.
            training_immo.marktwert = to_normalize.marktwert.map(|marktwert| marktwert * 1000.0);
            let old_marktwert = to_normalize.marktwert;


            // We can only perform regression with at least three immos
            let mut extra_train_immo = training_immo.clone();
            extra_train_immo.marktwert = extra_train_immo.marktwert.map(|marktwert| marktwert + 1e-9);

            let mut normalizer = RegressionNormalizer::new();

            normalizer.train(once(&to_normalize).chain(once(&training_immo).chain(once(&extra_train_immo)))).unwrap();
            normalizer.normalize(once(&mut to_normalize));

            prop_assert_eq!(to_normalize.marktwert, old_marktwert);
        }


        #[test]
        fn normaliziation_factor_is_one_for_reference_date(mut immos in full_immos(16)) {
            let mut normalizer = RegressionNormalizer::new();
            let old_marktwert = immos[0].marktwert;
            immos[0].wertermittlungsstichtag = Some(reference_date());
            normalizer.train(&immos).unwrap();

            prop_assert!((normalizer.normalization_factor(&immos[0]).unwrap() - 1.0).abs() < 1e-9);

            normalizer.normalize(&mut immos);
            prop_assert!((immos[0].marktwert.unwrap() - old_marktwert.unwrap()).abs() < 1e-9);
        }
    }
}
