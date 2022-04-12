use chrono::{Datelike, NaiveDate};
use common::util::path_or_relative_to_project_root;
use common::{immo::Objektunterart, BpError, BpResult, Immo, Normalizer, Trainable};
use serde::Deserialize;
use std::collections::HashMap;

type Year = u16;
type QuarterNum = u8;

#[derive(Debug, Deserialize)]
struct Quarter {
    year: Year,
    quarter: QuarterNum,

    wohnung_metropolen: f64,
    wohnung_kreisfreie_grossstaedte: f64,
    wohnung_staedtische_kreise: f64,
    wohnung_laendliche_kreise_mit_verdichtungsansaetzen: f64,
    wohnung_duennbesiedelte_laendliche_kreise: f64,

    haus_metropolen: f64,
    haus_kreisfreie_grossstaedte: f64,
    haus_staedtische_kreise: f64,
    haus_laendliche_kreise_mit_verdichtungsansaetzen: f64,
    haus_duennbesiedelte_laendliche_kreise: f64,
}

impl Quarter {
    fn factor_for(&self, kreistyp: &Kreistyp, objektunterart: &Objektunterart) -> f64 {
        match (kreistyp, objektunterart) {
            (Kreistyp::Metropole, Objektunterart::Eigentumswohnung) => self.wohnung_metropolen,
            (Kreistyp::KreisfreieGrosstadt, Objektunterart::Eigentumswohnung) => {
                self.wohnung_kreisfreie_grossstaedte
            }
            (Kreistyp::StaedtischerKreis, Objektunterart::Eigentumswohnung) => {
                self.wohnung_staedtische_kreise
            }
            (Kreistyp::LaendlicherKreisMitVerdichtungsAnsatz, Objektunterart::Eigentumswohnung) => {
                self.wohnung_laendliche_kreise_mit_verdichtungsansaetzen
            }
            (Kreistyp::DuennbesiedelterLandlicherKreis, Objektunterart::Eigentumswohnung) => {
                self.wohnung_duennbesiedelte_laendliche_kreise
            }

            (Kreistyp::Metropole, _) => self.haus_metropolen,
            (Kreistyp::KreisfreieGrosstadt, _) => self.haus_kreisfreie_grossstaedte,
            (Kreistyp::StaedtischerKreis, _) => self.haus_staedtische_kreise,
            (Kreistyp::LaendlicherKreisMitVerdichtungsAnsatz, _) => {
                self.haus_laendliche_kreise_mit_verdichtungsansaetzen
            }
            (Kreistyp::DuennbesiedelterLandlicherKreis, _) => {
                self.haus_duennbesiedelte_laendliche_kreise
            }
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
enum Kreistyp {
    #[serde(rename = "Metropole")]
    Metropole,
    #[serde(rename = "kreisfreie Großstadt")]
    KreisfreieGrosstadt,
    #[serde(rename = "Städtischer Kreis")]
    StaedtischerKreis,
    #[serde(rename = "Ländlicher Kreis mit Verdichtungsansätzen")]
    LaendlicherKreisMitVerdichtungsAnsatz,
    #[serde(rename = "Dünn besiedelter ländlicher Kreis")]
    DuennbesiedelterLandlicherKreis,
}

#[derive(Debug, Deserialize)]
struct Kreis {
    #[serde(rename = "AGS0")]
    ags0: String,
    kreistyp: Kreistyp,
}

/// This [Normalizer] employs the [Häuserpreisindex](https://www.destatis.de/DE/Themen/Wirtschaft/Preise/Baupreise-Immobilienpreisindex/Tabellen/haeuserpreisindex-kreistypen.html) or HPI for short.
/// We use [BBRS data](https://www.bbsr.bund.de/BBSR/DE/forschung/raumbeobachtung/Raumabgrenzungen/deutschla[…]nen/kreis-kreisregionen-2017.xlsx?__blob=publicationFile&v=3)
/// to find the "Kreistyp" for each immo.
/// This will normalize all prices after 2015 to 2015. All prices from before 2015 will remain unchanged.2015 will remain unchanged.
pub struct HpiNormalizer {
    quarters: HashMap<(Year, QuarterNum), Quarter>,
    kreis_types: HashMap<String, Kreistyp>,
}

impl HpiNormalizer {
    /// Returns a new HpiNormalizer.
    /// # Panics
    /// - If the `csv` containing the mapping AGS_0 -> Kreistyp cannot be found.
    /// - If the `csv` containing the mapping Quarter -> growth rate connot be found.
    pub fn new() -> Self {
        let hpi_path =
            path_or_relative_to_project_root(None, "./data/normalization/price_index_quarters.csv");
        let kreistyp_path =
            path_or_relative_to_project_root(None, "./data/normalization/kreistyp.csv");

        let mut quarters_reader = csv::ReaderBuilder::new()
            .delimiter(b';')
            .from_path(hpi_path)
            .unwrap();

        let mut quarters = HashMap::new();
        quarters_reader
            .deserialize()
            .map(|row| {
                row.unwrap_or_else(|e| {
                    panic!("Could not deserialize quarter: {:?}", e);
                })
            })
            .for_each(|quarter: Quarter| {
                quarters.insert((quarter.year, quarter.quarter), quarter);
            });

        assert_eq!(quarters.len(), 20, "Not all quarters read succesfully");

        let mut kreis_types_reader = csv::ReaderBuilder::new()
            .delimiter(b';')
            .from_path(kreistyp_path)
            .unwrap();

        let mut kreis_types = HashMap::new();
        kreis_types_reader
            .deserialize()
            .map(|row| {
                row.unwrap_or_else(|e| {
                    panic!("Could not deserialize kreistyp: {:?}", e);
                })
            })
            .for_each(|kreis: Kreis| {
                kreis_types.insert(kreis.ags0, kreis.kreistyp);
            });

        assert_eq!(
            kreis_types.len(),
            401,
            "Not all Kreistypen were imported successfully"
        );

        HpiNormalizer {
            quarters,
            kreis_types,
        }
    }

    fn normalization_factor(&self, immo: &Immo) -> BpResult<f64> {
        if !Self::can_be_normalized(immo) {
            return Ok(1.0);
        }

        let date = immo
            .wertermittlungsstichtag
            .expect("A none value passed the above if, which is impossible.");

        let mut year = date.year();
        // This begins at 1 and ends at 4.
        let mut quarter_num = (date.month() + 2) / 3;

        // We can't predict into the future
        if year > 2020 {
            year = 2020;
            quarter_num = 4;
        }

        let quarter = self
            .quarters
            .get(&(year as Year, quarter_num as QuarterNum))
            .ok_or_else(|| {
                BpError::from(format!(
                    "Did not find quarter for {} {} in month {}",
                    year,
                    quarter_num,
                    date.month()
                ))
            })?;

        let kreis_type = self
            .kreis_types
            .get(
                immo.ags0
                    .as_ref()
                    .expect("A none value cant pass the `can_be_normalized check"),
            )
            .ok_or_else(|| {
                BpError::from(format!(
                    "Did not find kreistype for AGS={:?}, kreis={:?}",
                    immo.ags0.as_ref(),
                    immo.kreis
                ))
            })?;

        let result = quarter.factor_for(
            kreis_type,
            &immo
                .objektunterart
                .expect("A none value cant pass the `can_be_normalized check"),
        ) / 100.0; // Is is per cent.

        // The table should not include price increases larger than 200% or lower than 100%. We
        // previously had a bug where a typo ruined the prediction in whole types of kreises, so we
        // now have a hard check here.
        assert!((0.5..=2.0).contains(&result));

        Ok(result)
    }

    fn can_be_normalized(immo: &Immo) -> bool {
        if immo
            .wertermittlungsstichtag
            .map(|date| date <= NaiveDate::from_ymd(2016, 1, 1))
            .unwrap_or(true)
        {
            return false;
        }

        if immo.ags0.is_none() {
            return false;
        }

        if immo.objektunterart.is_none() {
            return false;
        }

        true
    }
}

impl Default for HpiNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainable for HpiNormalizer {}

impl Normalizer for HpiNormalizer {
    fn normalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            immo.marktwert = immo
                .marktwert
                .map(|marktwert| marktwert / self.normalization_factor(immo).unwrap())
        }
    }

    fn denormalize<'i>(&self, immos: impl IntoIterator<Item = &'i mut Immo>) {
        for immo in immos {
            immo.marktwert = immo
                .marktwert
                .map(|marktwert| marktwert * self.normalization_factor(immo).unwrap())
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
        let mut normalizer = HpiNormalizer::new();
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

            let mut normalizer = HpiNormalizer::new();
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

            let normalizer = HpiNormalizer::new();
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.normalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }

        #[test]
        fn regression_does_not_invent_marktpreise_on_denormalization(mut immos in full_immos(128)) {
            for immo in immos.iter_mut() {
                immo.ags0 = Some("01001000".into());
            }

            let normalizer = HpiNormalizer::new();
            immos.iter_mut().for_each(|immo| immo.marktwert = None);
            normalizer.denormalize(&mut immos);
            prop_assert!(immos.iter().all(|immo| immo.marktwert.is_none()));
        }

        #[test]
        fn normaliziation_factor_is_one_before_2016(mut immos in full_immos(16), year in 2000i32..2015i32, month in 1u32..12u32, day in 1u32..28u32) {
            for immo in immos.iter_mut() {
                immo.ags0 = Some("01001000".into());
            }

            let mut normalizer = HpiNormalizer::new();
            let old_marktwert = immos[0].marktwert;
            immos[0].wertermittlungsstichtag = Some(NaiveDate::from_ymd(year, month, day));
            normalizer.train(&immos).unwrap();

            prop_assert!((normalizer.normalization_factor(&immos[0]).unwrap() - 1.0).abs() < 1e-9);

            normalizer.normalize(&mut immos);
            prop_assert!((immos[0].marktwert.unwrap() - old_marktwert.unwrap()).abs() < 1e-9);
        }
    }
}
