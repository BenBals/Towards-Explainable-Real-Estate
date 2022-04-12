use super::utils::*;
use super::Normalizer;
use chrono::NaiveDate;
use common::{BpResult, Immo, Trainable};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// This [Normalizer] will normalize immos based on their zip code and by fixed rates.
#[derive(Debug, Clone, Default)]
pub struct PlzNormalizer {
    /// The normalization factor for each zip code.
    /// The normalization factor should give by how much percent the price increases per year in a
    /// given zip code.
    normalization_per_year: HashMap<String, f64>,
}

#[derive(PartialEq, Eq)]
enum Mode {
    Normalize,
    Denormalize,
}

impl PlzNormalizer {
    /// Create a new `PlzNormalizer`, which contains no normalization factors.
    /// See [set_factor] or [with_factors] for setting those.
    pub fn new() -> Self {
        Self {
            normalization_per_year: HashMap::new(),
        }
    }

    /// Sets the **per year** change for a given zip code.
    pub fn set_factor(&mut self, plz: &str, factor: f64) {
        self.normalization_per_year.insert(plz.into(), factor);
    }

    /// Import a plz normalizer from a JSON file. This file must have the following schema:
    /// - Top Level: Object
    /// - Each key is a zip code as a string
    /// - Each value is a price increase in percentage points
    ///   - note that this is different from the argument of [set_factor]. A value of 20.0 would
    ///     correspond to `set_factor("12345", 1.2)`.
    /// # Arguments
    /// - `path` path to the JSON file
    /// - `num_years` number of years the increase was computed over
    pub fn with_file_path<P>(path: P, num_years: f64) -> BpResult<Self>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut raw_map: HashMap<String, f64> = serde_json::from_reader(reader)?;

        for (_, value) in raw_map.iter_mut() {
            *value = (1.0 + *value / 100.0).powf(num_years.recip())
        }

        Ok(Self {
            normalization_per_year: raw_map,
        })
    }

    fn normalize_immo(&self, immo: &mut Immo, mode: Mode) {
        let mut factor = match (immo.wertermittlungsstichtag.as_ref(), immo.plz.as_ref()) {
            (Some(date), Some(plz)) => self.normalization_factor(*date, plz),
            _ => 1.0,
        };

        if mode == Mode::Denormalize {
            factor = factor.recip();
        }

        immo.marktwert = immo.marktwert.map(|value| value * factor);
    }

    fn normalization_factor(&self, date: NaiveDate, plz: &str) -> f64 {
        if let Some(per_year) = self.normalization_per_year.get(plz) {
            let years = days_until_reference_date(date) / 365.0;
            per_year.powf(years)
        } else {
            1.0
        }
    }
}

impl Trainable for PlzNormalizer {}

impl Normalizer for PlzNormalizer {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            self.normalize_immo(immo, Mode::Normalize)
        }
    }
    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            self.normalize_immo(immo, Mode::Denormalize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::Duration;
    use common::immo::ImmoBuilder;
    use proptest::prelude::*;
    use std::iter::once;
    use test_helpers::*;

    #[test]
    fn dnn_generated_file_import() -> BpResult<()> {
        let normalizer: PlzNormalizer = PlzNormalizer::with_file_path(
            "../../data/normalization/normalization_example.json",
            2.0,
        )?;

        let mut immo = ImmoBuilder::default()
            .marktwert(100.0)
            .wohnflaeche(42.0)
            .build()
            .unwrap();
        immo.plz = Some("12345".into());
        immo.wertermittlungsstichtag = reference_date().checked_sub_signed(Duration::days(2 * 365));

        normalizer.normalize(once(&mut immo));

        assert_approx_eq!(immo.marktwert.unwrap(), 100.0 * 1.2 * 1.2);
        Ok(())
    }

    proptest! {
        #[test]
        fn normalize_one_year(mut immo in full_immo()) {
            let mut normalizer = PlzNormalizer::new();
            let old_value = immo.marktwert.unwrap();
            immo.wertermittlungsstichtag = reference_date().checked_sub_signed(Duration::days(365));
            normalizer.set_factor(immo.plz.as_ref().unwrap(), 2.0);

            normalizer.normalize(once(&mut immo));

            prop_assert!((immo.marktwert.unwrap() / old_value - 2.0).abs() <= 1e-9);
        }

        #[test]
        fn normalize_two_years(mut immo in full_immo()) {
            let mut normalizer = PlzNormalizer::new();
            let old_value = immo.marktwert.unwrap();
            immo.wertermittlungsstichtag = reference_date().checked_sub_signed(Duration::days(2 * 365));
            normalizer.set_factor(immo.plz.as_ref().unwrap(), 2.0);

            normalizer.normalize(once(&mut immo));

            prop_assert!((immo.marktwert.unwrap() / old_value - 4.0).abs() <= 1e-9);
        }
    }
}
