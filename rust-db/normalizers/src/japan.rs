use crate::utils::Quarter;
use chrono::{Datelike, NaiveDate};
use common::util::path_or_relative_to_project_root;
use common::BpError;
use common::{BpResult, Immo, Normalizer, Trainable};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct QuarterFactor {
    quarter: Quarter,
    factor: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
struct CsvEntry {
    date: NaiveDate,
    #[serde(rename = "QBRN628BIS")]
    factor: f64,
}

impl From<CsvEntry> for QuarterFactor {
    fn from(csv: CsvEntry) -> Self {
        Self {
            quarter: Quarter::from(csv.date),
            factor: csv.factor,
        }
    }
}

#[derive(Clone, Debug)]
/// A simple Normalizer for Japan Data
pub struct JapanNormalizer {
    quarters: HashMap<Quarter, QuarterFactor>,
}

impl JapanNormalizer {
    /// Uses Data from ...?
    pub fn new() -> Self {
        let path = path_or_relative_to_project_root(None, "./data/normalization/japan.csv");

        let mut quarters_reader = csv::ReaderBuilder::new()
            .delimiter(b',')
            .from_path(path)
            .unwrap();

        let quarters: HashMap<_, _> = quarters_reader
            .deserialize()
            .map(|row: Result<CsvEntry, _>| {
                row.unwrap_or_else(|e| {
                    panic!("Could not deserialize quarter: {:?}", e);
                })
            })
            .map(QuarterFactor::from)
            .map(|quarter_factor| (quarter_factor.quarter, quarter_factor))
            .collect();

        Self { quarters }
    }

    /// Marktwert * normalization factor normalizes the price to 2010
    fn normalization_factor(&self, immo: &Immo) -> BpResult<f64> {
        let is_outside = immo
            .wertermittlungsstichtag
            .map(|date| {
                NaiveDate::from_ymd(2001, 1, 1) > date || date >= NaiveDate::from_ymd(2021, 1, 1)
            })
            .unwrap_or(true);
        if is_outside {
            return Ok(1.0);
        }

        let date = immo.wertermittlungsstichtag.unwrap();
        let quarter = date.into();
        let factor = self.quarters.get(&quarter).ok_or_else(|| {
            BpError::from(format!(
                "Did not find quarter for {:?} in month {}",
                quarter,
                date.month(),
            ))
        })?;

        Ok(100.0 / factor.factor)
    }
}

impl Default for JapanNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainable for JapanNormalizer {}

impl Normalizer for JapanNormalizer {
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
    use common::logging::init_test_logging;
    use proptest::prelude::*;
    use proptest::proptest;
    use test_helpers::*;

    #[test]
    fn brazil_normalizer_has_correct_size() {
        let normalizer = JapanNormalizer::new();
        assert_eq!(normalizer.quarters.len(), 80);
    }

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
        let mut normalizer = JapanNormalizer::new();
        normalizer.train(&immos)?;
        normalizer.normalize(immos.iter_mut());

        Ok(())
    }

    proptest! {
        #[test]
        fn regression_defines_bijection(mut immos in full_immos(128)) {
            // We need more than one immos in an area to have a meaningful normalization.
            for immo in immos.iter_mut() {
                immo.ags0 = Some("01001000".into());
            }

            let clone = immos.clone();

            let mut normalizer = JapanNormalizer::new();
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
            for immo in immos.iter_mut() {
                immo.ags0 = Some("01001000".into());
            }

            let normalizer = JapanNormalizer::new();
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.normalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }

        #[test]
        fn regression_does_not_invent_marktpreise_on_denormalization(mut immos in full_immos(128)) {
            for immo in immos.iter_mut() {
                immo.ags0 = Some("01001000".into());
            }

            let normalizer = JapanNormalizer::new();
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.denormalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }

        #[test]
        fn normaliziation_factor_is_one_before_2001(mut immos in full_immos(16), year in 1900i32..2001i32, month in 1u32..12u32, day in 1u32..28u32) {
            for immo in immos.iter_mut() {
                immo.ags0 = Some("01001000".into());
            }

            let mut normalizer = JapanNormalizer::new();
            let old_marktwert = immos[0].marktwert;
            immos[0].wertermittlungsstichtag = Some(NaiveDate::from_ymd(year, month, day));
            normalizer.train(&immos).unwrap();

            prop_assert!((normalizer.normalization_factor(&immos[0]).unwrap() - 1.0).abs() < 1e-9);

            normalizer.normalize(&mut immos);
            prop_assert!((immos[0].marktwert.unwrap() - old_marktwert.unwrap()).abs() < 1e-9);
        }
    }
}
