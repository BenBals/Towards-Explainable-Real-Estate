//! This module provides [ExpertDissimilarity]. See its documentation.
use crate::filtered::{FilterStrategy, FilteredDissimilarity, FilteredDissimilarityBuilder};
use algorithms::calculate_environment::calculate_weighted_average;
use common::{immo::Objektunterart, BpResult, Dissimilarity, Immo, Trainable};
use derive_builder::Builder;
use serde::Deserialize;

/// How many factors are used to set the slope of the individul similarities?
pub const NUM_FACTORS: usize = 8;
/// How many weights are used to combine the individual similarities?
pub const NUM_WEIGHTS: usize = 9;

#[derive(Clone, Debug, PartialEq)]
/// The ExpertDissimilarity including filters and the rest of the calculation. This is just a wrapper around `FilteredDissimilarity<UnfilteredExpertDissimilarity>`.
pub struct ExpertDissimilarity {
    inner: FilteredDissimilarity<UnfilteredExpertDissimilarity>,
}

impl Trainable for ExpertDissimilarity {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.inner.train(training_data)
    }
}

impl Dissimilarity for ExpertDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        assert!(
            is_expert_dissimilarity_valid_for(immo1) && is_expert_dissimilarity_valid_for(immo2),
            "Passed invalid immo into ExpertDissimilarity"
        );

        self.inner.dissimilarity(immo1, immo2)
    }
}

impl Default for ExpertDissimilarity {
    fn default() -> Self {
        ExpertDissimilarity {
            inner: FilteredDissimilarityBuilder::default()
                .difference_regiotyp(Some(0))
                .difference_objektunterart(Some(0))
                .difference_wohnflaeche(Some(FilterStrategy::Category(0)))
                .difference_grundstuecksgroesse(Some(FilterStrategy::Category(0)))
                .difference_baujahr(Some(FilterStrategy::Category(0)))
                .difference_wertermittlungsstichtag_days(Some(5 * 366))
                .difference_spatial_distance_kilometer(Some(40))
                .difference_zustand(Some(1))
                .difference_ausstattungsnote(Some(1))
                .inner(UnfilteredExpertDissimilarity::default())
                .build()
                .unwrap(),
        }
    }
}

impl From<FilteredDissimilarity<UnfilteredExpertDissimilarity>> for ExpertDissimilarity {
    fn from(inner: FilteredDissimilarity<UnfilteredExpertDissimilarity>) -> Self {
        Self { inner }
    }
}

impl ExpertDissimilarity {
    /// Create a new `ExpertDissimilarity`
    pub fn new() -> Self {
        ExpertDissimilarity::default()
    }
}

/// This dissimilarity reflects interviews with real assessors.
/// See [the Notion page](https://www.notion.so/Expert-Dissimilarity-d4085ce2d7f94a95af8188abec6ed39c)
/// for background info.
/// # Panics
/// The input properties must pass [is_expert_dissimilarity_valid_for], the dissimilarity function
/// will *panic* if that's not the case.
#[derive(Debug, Clone, Copy, Deserialize, Builder, PartialEq)]
pub struct UnfilteredExpertDissimilarity {
    /// If set, similarities lower than this value will be treated as 0%
    cutoff_similarity: Option<f64>,
    /// The individual similarities are split into two parts. This sets how much the first part
    /// should be weighted. It should be between 0.0 and 1.0. The second part is weighted at 1.0
    /// minus this value.
    similarity_part_1_weight: f64,
    /// For objektunterart_category 1 or 2, how much should the similarity be reduced for each
    /// square meter difference in wohnflaeche?
    wohnflaeche_factor_category_1_2: f64,
    /// For objektunterart_category 3, how much should the similarity be reduced for each
    /// square meter difference in wohnflaeche?
    wohnflaeche_factor_category_3: f64,
    /// How much of the part 1 similarity should be determined by wohnflaeche?
    wohnflaeche_weight: f64,
    /// How much should the similarity be reduced for each km distance in location?
    plane_distance_factor: f64,
    /// How much of the part 1 similarity should be determined by plane distance?
    plane_distance_weight: f64,
    /// How much should the similarity be reduced for each year of difference in baujahr?
    baujahr_factor: f64,
    /// How much of the part 1 similarity should be determined by baujahr?
    baujahr_weight: f64,
    /// How much should the similarity be reduced for each square metre of difference in
    /// grundstuecksgroesse?
    /// This only applied to objektunterart_category 1 or 2.
    grundstuecksgroesse_factor: f64,
    /// How much of the part 1 similarity should be determined by grundstuecksgroesse?
    grundstuecksgroesse_weight: f64,
    /// How much should the similarity be reduced per 1 in difference of anzahl_stellpleatze?
    anzahl_stellplaetze_factor: f64,
    /// How much of the part 2 similarity should be determined by anzahl_stellplaetze?
    anzahl_stellplaetze_weight: f64,
    /// How much should the similarity be reduced per 1 in difference of anzahl_zimmer?
    anzahl_zimmer_factor: f64,
    /// How much of the part 2 similarity should be determined by anzahl_zimmer?
    anzahl_zimmer_weight: f64,
    /// How much should the similarity be reduced per 1 in difference of combined_location_score?
    combined_location_score_factor: f64,
    /// How much of the part 2 similarity should be determined by location_score_all?
    combined_location_score_weight: f64,
    /// How much of the part 2 similarity should be determined by verwendung?
    verwendung_weight: f64,
    /// How much of the part 2 similarity should be determined by keller?
    keller_weight: f64,
}

impl Default for UnfilteredExpertDissimilarity {
    fn default() -> Self {
        Self {
            cutoff_similarity: None,
            similarity_part_1_weight: 0.9,
            wohnflaeche_factor_category_1_2: 0.4,
            wohnflaeche_factor_category_3: 2.0,
            wohnflaeche_weight: 1.0,
            plane_distance_factor: 2.5,
            plane_distance_weight: 1.0,
            baujahr_factor: 2.0,
            baujahr_weight: 1.0,
            grundstuecksgroesse_factor: 0.2,
            grundstuecksgroesse_weight: 1.0,
            anzahl_stellplaetze_factor: 25.0,
            anzahl_stellplaetze_weight: 1.0,
            anzahl_zimmer_factor: 50.0,
            anzahl_zimmer_weight: 1.0,
            combined_location_score_factor: 10.0,
            combined_location_score_weight: 1.0,
            verwendung_weight: 1.0,
            keller_weight: 1.0,
        }
    }
}

impl UnfilteredExpertDissimilarity {
    fn similarity_part_1(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        // Note that by the check above, both immos are in the same category, so we need ony check one.
        let wohnflaeche_factor = if immo1.is_objektunterart_category_1_or_2() {
            self.wohnflaeche_factor_category_1_2
        } else {
            self.wohnflaeche_factor_category_3
        };
        let mut similarities = vec![
            optional_similarity(
                immo1
                    .fiktives_baujahr_or_baujahr()
                    .map(|baujahr| baujahr as f64),
                immo2
                    .fiktives_baujahr_or_baujahr()
                    .map(|baujahr| baujahr as f64),
                self.baujahr_factor,
            ),
            optional_similarity(immo1.wohnflaeche, immo2.wohnflaeche, wohnflaeche_factor),
            immo1
                .plane_distance(immo2)
                .map(|distance| 100.0 - (distance / 1000.0) * self.plane_distance_factor)
                .map(|sim| sim.max(0.0))
                .unwrap_or(0.0),
            optional_similarity(
                immo1.combined_location_score(),
                immo2.combined_location_score(),
                self.combined_location_score_factor,
            ),
        ];
        let mut all_weights = vec![
            self.baujahr_weight,
            self.wohnflaeche_weight,
            self.plane_distance_weight,
            self.combined_location_score_weight,
        ];

        // Note that by the check above, both immos are in the same category, so we need ony check one.
        if immo1.is_objektunterart_category_1_or_2() && immo2.is_objektunterart_category_1_or_2() {
            similarities.push(optional_similarity(
                immo1.grundstuecksgroesse,
                immo2.grundstuecksgroesse,
                self.grundstuecksgroesse_factor,
            ));
            all_weights.push(self.grundstuecksgroesse_weight)
        }

        weighted_similarity(&similarities, &all_weights).clamp(0.0, 100.0)
    }

    fn similarity_part_2(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        let mut similarities = vec![
            optional_similarity(
                immo1
                    .anzahl_stellplaetze
                    .map(|anzahl_stellplaetze| anzahl_stellplaetze as f64),
                immo2
                    .anzahl_stellplaetze
                    .map(|anzahl_stellplaetze| anzahl_stellplaetze as f64),
                self.anzahl_stellplaetze_factor,
            ),
            optional_similarity(
                immo1.anzahl_zimmer,
                immo2.anzahl_zimmer,
                self.anzahl_zimmer_factor,
            ),
        ];

        let mut all_weights = vec![
            self.anzahl_stellplaetze_weight,
            self.anzahl_zimmer_weight,
            self.verwendung_weight,
        ];

        let verwendung_similarity =
            if immo1.verwendung.is_some() && immo1.verwendung == immo2.verwendung {
                100.0
            } else {
                0.0
            };
        similarities.push(verwendung_similarity);

        // Note that by the check above, both immos are in the same category, so we need ony check one.
        if immo1.is_objektunterart_category_1_or_2() {
            let keller_similarity = immo1
                .keller
                .zip(immo2.keller)
                .map(|(keller1, keller2)| if keller1 == keller2 { 100.0 } else { 0.0 })
                .unwrap_or(0.0);
            similarities.push(keller_similarity);
            all_weights.push(self.keller_weight)
        }

        weighted_similarity(&similarities, &all_weights).clamp(0.0, 100.0)
    }
}

/// Given a set of similarities, combine them into one similarity using a weighted average.
/// If all weights are `0.0`, the average is returned.
fn weighted_similarity(similarities: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(similarities.len(), weights.len());
    let sum_weights = weights.iter().sum::<f64>();
    let should_be_average = sum_weights.abs() <= f64::EPSILON;

    calculate_weighted_average(
        similarities.iter().zip(weights.iter()),
        |(sim, _)| **sim,
        |(_, weight)| {
            if should_be_average {
                1.0
            } else {
                **weight
            }
        },
    )
    .unwrap()
}

impl Trainable for UnfilteredExpertDissimilarity {}
impl Dissimilarity for UnfilteredExpertDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        let similarity = self.similarity_part_1_weight * self.similarity_part_1(immo1, immo2)
            + (1.0 - self.similarity_part_1_weight) * self.similarity_part_2(immo1, immo2);

        if let Some(cutoff_similarity) = self.cutoff_similarity {
            if similarity < cutoff_similarity * 100.0 {
                return f64::INFINITY;
            }
        }
        similarity_to_dissimilarity(similarity)
    }
}

/// The [ExpertDissimilarity] may only be applied too immos which are "vermietbar", "verwertbar" and
/// "drittverwendbar" and DONT have "erbbaurecht.
/// Unset values are considered ok.
pub fn is_expert_dissimilarity_valid_for(immo: &Immo) -> bool {
    let vermietbar_valid = immo.vermietbarkeit.unwrap_or(true);
    let verwertbar_valid = immo.verwertbarkeit.unwrap_or(true);
    let drittverwendung_valid = immo.drittverwendungsfaehigkeit.unwrap_or(true);
    let erbaurecht_valid = !immo.erbbaurecht.unwrap_or(false);
    let objektunterart_valid = immo.objektunterart != Some(Objektunterart::Mehrfamilienhaus);
    vermietbar_valid
        && verwertbar_valid
        && drittverwendung_valid
        && erbaurecht_valid
        && objektunterart_valid
}

/// Turn a similarity valued in [0,100] into a dissimilarity valued in [0,inf].
fn similarity_to_dissimilarity(sim: f64) -> f64 {
    if (sim - 0.0).abs() < f64::EPSILON {
        f64::INFINITY
    } else if (sim - 100.0).abs() < f64::EPSILON {
        0.0
    } else {
        sim.recip()
    }
}

/// Return the weighted "similarity" in an attribute.
/// If both values (x, y) are defined, that is 100 - |x-y| * factor
/// If any is `None`, the return value is `0.0`.
fn optional_similarity(opt1: Option<f64>, opt2: Option<f64>, factor: f64) -> f64 {
    let result = if let (Some(val1), Some(val2)) = (opt1, opt2) {
        100.0 - ((val1 - val2).abs() * factor)
    } else {
        0.0
    };
    result.max(0.0)
}

#[cfg(test)]
// For some inputs, we want that the output is exactly 0.0, but clippy will not allow such asserts.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::{Duration, NaiveDate};
    use common::immo::{MacroLocationScore, MicroLocationScore, Verwendung, Zustand};
    use common::util::path_or_relative_to_project_root;
    use common::Immo;
    use mongodb::bson::oid::ObjectId;
    use proptest::prelude::*;
    use std::f64::consts::SQRT_2;
    use test_helpers::*;

    fn create_valid_immo_at(coords: &[f64; 2]) -> Immo {
        let mut immo = create_new_immo_at(coords);
        immo.wertermittlungsstichtag = Some(NaiveDate::from_ymd(2020, 1, 1));
        immo.baujahr = Some(1949);
        immo.objektunterart = Some(Objektunterart::Einfamilienhaus);
        immo.wohnflaeche = Some(100.0);
        immo.ausstattung = Some(1);
        immo.zustand = Some(Zustand::Gut);
        immo.anzahl_stellplaetze = Some(0);
        immo.anzahl_zimmer = Some(3.5);
        immo.grundstuecksgroesse = Some(0.0);
        immo.keller = Some(false);
        immo
    }
    fn create_valid_immo() -> Immo {
        create_valid_immo_at(&[0.0, 0.0])
    }

    #[test]
    fn test_helper_creates_valid_immo() {
        let immo = create_valid_immo();
        let dissim = ExpertDissimilarity::new();
        assert!(dissim.dissimilarity(&immo, &immo).is_finite());
    }

    #[test]
    #[should_panic]
    fn needs_vermietbarkeit_true_left() {
        let mut immo1 = create_new_immo();
        immo1.vermietbarkeit = Some(false);
        let mut immo2 = create_new_immo();
        immo2.vermietbarkeit = Some(true);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_vermietbarkeit_true_right() {
        let mut immo1 = create_new_immo();
        immo1.vermietbarkeit = None;
        let mut immo2 = create_new_immo();
        immo2.vermietbarkeit = Some(false);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_verwertbarkeit_true_left() {
        let mut immo1 = create_new_immo();
        immo1.verwertbarkeit = Some(false);
        let mut immo2 = create_new_immo();
        immo2.verwertbarkeit = Some(true);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_verwertbarkeit_true_right() {
        let mut immo1 = create_new_immo();
        immo1.verwertbarkeit = None;
        let mut immo2 = create_new_immo();
        immo2.verwertbarkeit = Some(false);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_drittverwendungsfaehigkeit_true_left() {
        let mut immo1 = create_new_immo();
        immo1.drittverwendungsfaehigkeit = Some(false);
        let mut immo2 = create_new_immo();
        immo2.drittverwendungsfaehigkeit = Some(true);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_drittverwendungsfaehigkeit_true_right() {
        let mut immo1 = create_new_immo();
        immo1.drittverwendungsfaehigkeit = None;
        let mut immo2 = create_new_immo();
        immo2.drittverwendungsfaehigkeit = Some(false);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_erbbaurecht_false_left() {
        let mut immo1 = create_new_immo();
        immo1.erbbaurecht = Some(true);
        let mut immo2 = create_new_immo();
        immo2.erbbaurecht = Some(false);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_erbbaurecht_false_right() {
        let mut immo1 = create_new_immo();
        immo1.erbbaurecht = None;
        let mut immo2 = create_new_immo();
        immo2.erbbaurecht = Some(true);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_objektunterart_not_mehrfamilienhaus_left() {
        let mut immo1 = create_new_immo();
        immo1.objektunterart = Some(Objektunterart::Mehrfamilienhaus);
        let immo2 = create_new_immo();

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    #[should_panic]
    fn needs_objektunterart_not_mehrfamilienhaus_right() {
        let immo1 = create_new_immo();
        let mut immo2 = create_new_immo();
        immo2.objektunterart = Some(Objektunterart::Mehrfamilienhaus);

        let dissim = ExpertDissimilarity::new();
        dissim.dissimilarity(&immo1, &immo2);
    }

    #[test]
    fn different_category_is_inf_examples() {
        let dissim = ExpertDissimilarity::new();
        let mut immo1 = create_new_immo();
        let mut immo2 = create_new_immo();

        immo1.objektunterart = Some(Objektunterart::Eigentumswohnung);
        immo2.objektunterart = Some(Objektunterart::Einfamilienhaus);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.objektunterart = Some(Objektunterart::Eigentumswohnung);
        immo2.objektunterart = Some(Objektunterart::EinfamilienhausMitEinliegerWohnung);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.objektunterart = Some(Objektunterart::Zweifamilienhaus);
        immo2.objektunterart = Some(Objektunterart::Reihenendhaus);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
    }

    #[test]
    fn category_3_ausstattung_diff_is_inf_examples() {
        let dissim = ExpertDissimilarity::new();
        let mut immo1 = create_new_immo();
        let mut immo2 = create_new_immo();

        immo1.objektunterart = Some(Objektunterart::Eigentumswohnung);
        immo2.objektunterart = Some(Objektunterart::Eigentumswohnung);

        immo1.ausstattung = Some(1);
        immo2.ausstattung = Some(3);

        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
    }

    #[test]
    fn zustand_diff_is_inf_examples() {
        let dissim = ExpertDissimilarity::new();
        let mut immo1 = create_new_immo();
        let mut immo2 = create_new_immo();

        immo1.zustand = Some(Zustand::Katastrophal);
        immo2.zustand = Some(Zustand::Maessig);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.zustand = Some(Zustand::Gut);
        immo2.zustand = Some(Zustand::Maessig);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.zustand = Some(Zustand::SehrGut);
        immo2.zustand = Some(Zustand::Mittel);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
    }

    #[test]
    fn different_baujahr_category_is_inf_examples() {
        let dissim = ExpertDissimilarity::new();
        let mut immo1 = create_new_immo();
        let mut immo2 = create_new_immo();

        immo1.baujahr = Some(1989);
        immo2.baujahr = Some(1990);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.baujahr = Some(1990);
        immo2.baujahr = Some(1989);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.baujahr = Some(1974);
        immo2.baujahr = Some(1949);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
    }

    #[test]
    fn different_wohnflaeche_category_is_inf_examples() {
        let dissim = ExpertDissimilarity::new();
        let mut immo1 = create_new_immo();
        let mut immo2 = create_new_immo();
        immo1.objektunterart = Some(Objektunterart::Einfamilienhaus);
        immo2.objektunterart = Some(Objektunterart::Einfamilienhaus);

        immo1.wohnflaeche = Some(50.0);
        immo2.wohnflaeche = Some(100.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.wohnflaeche = Some(150.0);
        immo2.wohnflaeche = Some(100.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.wohnflaeche = Some(5.0);
        immo2.wohnflaeche = Some(150.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.objektunterart = Some(Objektunterart::Eigentumswohnung);
        immo2.objektunterart = Some(Objektunterart::Eigentumswohnung);

        immo1.wohnflaeche = Some(40.0);
        immo2.wohnflaeche = Some(90.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.wohnflaeche = Some(30.0);
        immo2.wohnflaeche = Some(40.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.wohnflaeche = Some(100.0);
        immo2.wohnflaeche = Some(89.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
    }

    #[test]
    fn different_grundstuecksgroesse_category_is_inf_examples() {
        let dissim = ExpertDissimilarity::new();
        let mut immo1 = create_new_immo();
        let mut immo2 = create_new_immo();
        immo1.objektunterart = Some(Objektunterart::Einfamilienhaus);
        immo2.objektunterart = Some(Objektunterart::Einfamilienhaus);

        immo1.grundstuecksgroesse = Some(0.0);
        immo2.grundstuecksgroesse = Some(250.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.grundstuecksgroesse = Some(253.0);
        immo2.grundstuecksgroesse = Some(501.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);

        immo1.grundstuecksgroesse = Some(750.0);
        immo2.grundstuecksgroesse = Some(1500.0);
        assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
    }

    #[test]
    fn grundstuecksgroesse_is_no_filter_for_category3() {
        let dissim = ExpertDissimilarity::new();
        let mut immo1 = eigentumswohnung_example_1();
        let mut immo2 = eigentumswohnung_example_2();

        immo1.grundstuecksgroesse = Some(0.0);
        immo2.grundstuecksgroesse = Some(250.0);
        assert!(dissim.dissimilarity(&immo1, &immo2).is_finite());

        immo1.grundstuecksgroesse = None;
        immo2.grundstuecksgroesse = Some(250.0);
        assert!(dissim.dissimilarity(&immo1, &immo2).is_finite());
    }

    fn eigentumswohnung_example_1() -> Immo {
        let mut immo = create_valid_immo_at(&[0.0, 0.0]);
        immo.objektunterart = Some(Objektunterart::Eigentumswohnung);
        immo.baujahr = Some(1990);
        immo.wohnflaeche = Some(100.0);
        immo.grundstuecksgroesse = Some(0.0);
        immo.anzahl_stellplaetze = Some(1);
        immo.anzahl_zimmer = Some(3.0);
        immo.verwendung = Some(Verwendung::Eigennutzung);
        immo.keller = Some(false);
        immo.micro_location_scores = Some(MicroLocationScore {
            all: 50.0,
            education_and_work: 0.0,
            leisure: 0.0,
            public_transport: 0.0,
            shopping: 0.0,
        });
        immo.macro_location_scores = Some(MacroLocationScore {
            social_status: 50.0,
            economic_status: 50.0,
            market_dynamics: 50.0,
        });
        immo
    }

    fn eigentumswohnung_example_2() -> Immo {
        let mut immo = create_valid_immo_at(&[0.0, 1000.0]);
        immo.objektunterart = Some(Objektunterart::Eigentumswohnung);
        immo.baujahr = Some(2021);
        immo.wohnflaeche = Some(120.0);
        immo.grundstuecksgroesse = Some(17.0);
        immo.anzahl_stellplaetze = Some(2);
        immo.anzahl_zimmer = Some(4.5);
        immo.verwendung = Some(Verwendung::EigenUndFremdnutzung);
        immo.keller = Some(true);
        immo.micro_location_scores = Some(MicroLocationScore {
            all: 100.0,
            education_and_work: 0.0,
            leisure: 0.0,
            public_transport: 0.0,
            shopping: 0.0,
        });
        immo.macro_location_scores = Some(MacroLocationScore {
            social_status: 100.0,
            economic_status: 100.0,
            market_dynamics: 100.0,
        });
        immo
    }

    fn einfamilienhaus_example() -> Immo {
        let mut immo = create_valid_immo_at(&[0.0, 0.0]);
        immo.objektunterart = Some(Objektunterart::Einfamilienhaus);
        immo.baujahr = Some(1949);
        immo.wohnflaeche = Some(110.0);
        immo.grundstuecksgroesse = Some(100.0);
        immo.anzahl_stellplaetze = Some(1);
        immo.anzahl_zimmer = Some(5.0);
        immo.verwendung = Some(Verwendung::Eigennutzung);
        immo.keller = Some(true);
        immo.micro_location_scores = Some(MicroLocationScore {
            all: 100.0,
            education_and_work: 0.0,
            leisure: 0.0,
            public_transport: 0.0,
            shopping: 0.0,
        });
        immo.macro_location_scores = Some(MacroLocationScore {
            social_status: 100.0,
            economic_status: 100.0,
            market_dynamics: 100.0,
        });
        immo
    }

    fn doppelhaushaelfte_example() -> Immo {
        let mut immo = create_valid_immo_at(&[0.0, 4000.0]);
        immo.objektunterart = Some(Objektunterart::Doppelhaushaelfte);
        immo.baujahr = Some(1953);
        immo.wohnflaeche = Some(140.0);
        immo.grundstuecksgroesse = Some(230.0);
        immo.anzahl_stellplaetze = Some(3);
        immo.anzahl_zimmer = Some(6.0);
        immo.verwendung = Some(Verwendung::Eigennutzung);
        immo.keller = Some(false);
        immo.micro_location_scores = Some(MicroLocationScore {
            all: 100.0,
            education_and_work: 0.0,
            leisure: 0.0,
            public_transport: 0.0,
            shopping: 0.0,
        });
        immo.macro_location_scores = Some(MacroLocationScore {
            social_status: 90.0,
            economic_status: 90.0,
            market_dynamics: 90.0,
        });
        immo
    }

    #[test]
    fn dissimilarity_examples() {
        let dissim = UnfilteredExpertDissimilarity::default();
        let eigentumswohnung_1 = eigentumswohnung_example_1();
        let eigentumswohnung_2 = eigentumswohnung_example_2();
        let einfamilienhaus = einfamilienhaus_example();
        let doppelhaushaelfte = doppelhaushaelfte_example();

        assert_approx_eq!(
            dissim.similarity_part_1(&eigentumswohnung_1, &eigentumswohnung_2),
            48.875
        );
        assert_approx_eq!(
            dissim.similarity_part_2(&eigentumswohnung_1, &eigentumswohnung_2),
            100.0 / 3.0
        );
        assert_approx_eq!(
            dissim.dissimilarity(&eigentumswohnung_1, &eigentumswohnung_2),
            similarity_to_dissimilarity(47.32083),
            0.001
        );

        assert_approx_eq!(
            dissim.similarity_part_1(&einfamilienhaus, &doppelhaushaelfte),
            78.8
        );
        assert_approx_eq!(
            dissim.similarity_part_2(&einfamilienhaus, &doppelhaushaelfte),
            50.0
        );
        assert_approx_eq!(
            dissim.dissimilarity(&einfamilienhaus, &doppelhaushaelfte),
            similarity_to_dissimilarity(75.92),
            1e-3
        );
    }

    #[test]
    fn similarity_to_dissimilarity_examples() {
        let totally_similar = similarity_to_dissimilarity(100.0);
        assert_approx_eq!(totally_similar, 0.0);

        let totally_dissimilar = similarity_to_dissimilarity(0.0);
        assert_eq!(totally_dissimilar, f64::INFINITY);
    }

    #[test]
    fn expert_dissimilarity_dhall_default() {
        let imported = serde_dhall::from_file(path_or_relative_to_project_root(
            None,
            "config/dully/expert_parameters.dhall",
        ))
        .parse::<UnfilteredExpertDissimilarity>()
        .unwrap();

        assert_eq!(UnfilteredExpertDissimilarity::default(), imported);
    }

    prop_compose! {
        fn valid_immo()(mut immo in full_immo()) -> Immo {
            immo.vermietbarkeit = Some(true);
            immo.verwertbarkeit = Some(true);
            immo.drittverwendungsfaehigkeit = Some(true);
            immo.erbbaurecht = Some(false);
            if immo.objektunterart == Some(Objektunterart::Mehrfamilienhaus) {
                immo.objektunterart = Some(Objektunterart::Einfamilienhaus);
            }
            immo
        }
    }

    prop_compose! {
        /// Note: This does not guarantee that the dissimilarity will be finite. This will only change
        /// attributes that do not contribute to the dissimilarity average, but it will make
        /// the pair pass more filters. That is the chance of the dissimilarity being infinity
        /// decreases.
        fn valid_immo_pair()(
            mut immo1 in valid_immo(),
            mut immo2 in valid_immo(),
            location_x_immo1 in 0.0..(40000.0 / SQRT_2),
            location_y_immo1 in 0.0..(40000.0 / SQRT_2),
            location_x_immo2 in 0.0..(40000.0 / SQRT_2),
            location_y_immo2 in 0.0..(40000.0 / SQRT_2),
            baujahr_diff in -30i64..30i64,
            wohnflaeche_diff in -100.0..100.0,
            grundstuecksgroesse_diff in -500.0..500.0,
        ) -> (Immo, Immo) {
            immo1.plane_location = Some((location_x_immo1, location_y_immo1));
            immo2.plane_location = Some((location_x_immo2, location_y_immo2));

            immo2.zustand = immo1.zustand;
            immo2.objektunterart = immo1.objektunterart;
            immo2.wertermittlungsstichtag = immo1.wertermittlungsstichtag;
            immo2.ausstattung = immo1.ausstattung;
            immo2.regiotyp = immo1.regiotyp;

            immo2.baujahr = immo1.baujahr.map(|baujahr| (baujahr as i64 + baujahr_diff).max(0).min(2025) as u16);
            immo2.wohnflaeche = immo1.wohnflaeche.map(|wohnflaeche| (wohnflaeche + wohnflaeche_diff as f64).abs());
            immo2.grundstuecksgroesse = immo1.grundstuecksgroesse.map(|grundstuecksgroesse| (grundstuecksgroesse + grundstuecksgroesse_diff as f64).abs());
            (immo1, immo2)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { max_global_rejects: 10000, .. ProptestConfig::default() })]
        #[test]
        fn symmetry_finite(immos in valid_immo_pair()) {
            let (immo1, immo2) = immos;
            let dissim = ExpertDissimilarity::new();
            let left = dissim.dissimilarity(&immo1, &immo2);
            let right = dissim.dissimilarity(&immo2, &immo1);
            prop_assume!(left.is_finite() && right.is_finite());
            assert_approx_eq!(
                left,right
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { max_global_rejects: 10000, .. ProptestConfig::default() })]
        #[test]
        fn cutoff_similarity(immos in valid_immo_pair()) {
            let (immo1, immo2) = immos;
            let dissim = ExpertDissimilarity::new();
            let mut dissim_cutoff = ExpertDissimilarity::new();
            dissim_cutoff.inner.inner.cutoff_similarity = Some(0.5);
            if dissim.dissimilarity(&immo1, &immo2) > similarity_to_dissimilarity(50.0)  {
                prop_assert_eq!(dissim_cutoff.dissimilarity(&immo1, &immo2), f64::INFINITY)
            }
        }
    }

    proptest! {
        #[test]
        fn symmetry(immo1 in valid_immo(), immo2 in valid_immo()) {
            let dissim = ExpertDissimilarity::new();
            let left = dissim.dissimilarity(&immo1, &immo2);
            let right = dissim.dissimilarity(&immo2, &immo1);
            if left.is_finite() {
                assert_approx_eq!(
                    left,right
                );
            } else {
                prop_assert!(right.is_infinite());
            }
        }

        #[test]
        fn same_immo_is_zero(immo in valid_immo()) {
            let dissim = ExpertDissimilarity::new();
            prop_assert_eq!(dissim.dissimilarity(&immo, &immo), 0.0);
        }

        #[test]
        fn identical_immo_is_zero(immo in valid_immo()) {
            let mut immo2 = immo.clone();
            immo2.id = ObjectId::new();
            let dissim = ExpertDissimilarity::new();
            prop_assert_eq!(dissim.dissimilarity(&immo, &immo), 0.0);
        }

        #[test]
        fn is_positive(immo1 in valid_immo(), immo2 in valid_immo()) {
            let dissim = ExpertDissimilarity::new();
            prop_assert!(dissim.dissimilarity(&immo1, &immo2).is_sign_positive());
        }

        #[test]
        fn time_diff_5_years_is_inf(immo1 in valid_immo(), mut immo2 in valid_immo()) {
            let dissim = ExpertDissimilarity::new();
            immo2.wertermittlungsstichtag = immo1
                .wertermittlungsstichtag
                .map(|date| date.checked_add_signed(Duration::days(5 * 366)).unwrap());
            prop_assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
        }

        #[test]
        fn geo_diff_41km_is_inf(immo1 in valid_immo(), mut immo2 in valid_immo()) {
            let dissim = ExpertDissimilarity::new();
            immo2.plane_location = immo1.plane_location.map(|location| (location.0 + 41.0 * 1000.0, location.1));
            prop_assert_eq!(dissim.dissimilarity(&immo1, &immo2), f64::INFINITY);
        }

        #[test]
        fn similarity_to_dissimilarity_monotone(a in 0.0..100.0, b in 0.0..100.0) {
            if a < b {
                prop_assert!(similarity_to_dissimilarity(a) > similarity_to_dissimilarity(b)) ;
            } else if a > b {
                    prop_assert!(similarity_to_dissimilarity(a) < similarity_to_dissimilarity(b)) ;
            }
        }

        #[test]
        fn weighted_similarity_needs_same_size(similarities in proptest::collection::vec(0.0..100.0, 0..10), weights in proptest::collection::vec(0.0..100.0, 0..10)) {
            prop_assume!(similarities.len() != weights.len());
            let result = std::panic::catch_unwind(|| weighted_similarity(&similarities[..], &weights[..]));
            prop_assert!(result.is_err());
        }

        #[test]
        fn weighted_similarity_constant_weights_is_average(similarities in proptest::collection::vec(0.0..100.0, 1..10), weight in 0.0..1.0) {
            let weights = std::iter::repeat(weight).take(similarities.len()).collect::<Vec<_>>();
            let average =  similarities.iter().sum::<f64>() / similarities.len() as f64;
            prop_assert!((weighted_similarity(&similarities[..], &weights[..]) - average).abs() <= 1e-9);
        }

        #[test]
        fn weighted_similarity_is_finite(parameters in proptest::collection::vec((0.0..100.0, 0.0..100.0), 1..10)) {
            let (similarities, weights): (Vec<_>, Vec<_>) = parameters.iter().copied().unzip();
            let result = weighted_similarity(&similarities[..], &weights[..]);
            prop_assert!(result.is_finite());
            prop_assert!(result.is_sign_positive());
        }
    }
}
