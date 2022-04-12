#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![cfg_attr(feature = "strict", deny(missing_docs))]
//! This crate contains stuff that's really helpful for tests.
use chrono::NaiveDate;
use common::{
    immo::{
        set_immo_idxs, ImmoBuilder, MacroLocationScore, MicroLocationScore, Objektunterart, Zustand,
    },
    Immo,
};
use mongodb::bson::{doc, oid::ObjectId};
use proptest::prelude::*;

mod point;
use common::immo::Verwendung;
pub use point::Point;

/// This function creates a new Immo at the given location.
/// Location should be given as a two element slice.
/// All fields except location and id are missing.
pub fn create_new_immo_at(position: &[f64]) -> Immo {
    ImmoBuilder::from_document(doc! {
        "_id" : ObjectId::new(),
        "kurzgutachten": doc!{},
        "plane_location": position,
    })
    .build()
    .unwrap()
}

/// This function creates a new Immo at the origin.
/// All fields except location and id are missing.
pub fn create_new_immo() -> Immo {
    create_new_immo_at(&[0.0, 0.0])
}

/// Gives a strategy generating Zustand.
pub fn zustand() -> impl Strategy<Value = Zustand> {
    prop_oneof![
        Just(Zustand::SehrGut),
        Just(Zustand::Gut),
        Just(Zustand::Mittel),
        Just(Zustand::Maessig),
        Just(Zustand::Katastrophal),
    ]
}

/// Gives a strategy generating [Objektunterart].
pub fn objektunterart() -> impl Strategy<Value = Objektunterart> {
    prop_oneof![
        Just(Objektunterart::Eigentumswohnung),
        Just(Objektunterart::Einfamilienhaus),
        Just(Objektunterart::Doppelhaushaelfte),
        Just(Objektunterart::Zweifamilienhaus),
        Just(Objektunterart::Reihenmittelhaus),
        Just(Objektunterart::Reihenendhaus),
        Just(Objektunterart::Reihenhaus),
        Just(Objektunterart::Mehrfamilienhaus),
    ]
}

/// Gives a strategy generating [Verwendung].
pub fn verwendung() -> impl Strategy<Value = Verwendung> {
    prop_oneof![
        Just(Verwendung::Eigennutzung),
        Just(Verwendung::Fremdnutzung),
        Just(Verwendung::EigenUndFremdnutzung),
    ]
}

prop_compose! {
    /// gives a random [MicroLocationScore]
    pub fn micro_location_scores()(
        all in 0.0..100.0,
        edu in 0.0..100.0,
        leisure in 0.0..100.0,
        shopping in 0.0..100.0,
        public_transport in 0.0..100.0,
    ) -> MicroLocationScore {
        MicroLocationScore {
            all,
            education_and_work: edu,
            leisure,
            shopping,
            public_transport,
        }
    }

}

prop_compose! {
    /// gives a random [MacroLocationScore]
    pub fn macro_location_scores()(
        social_status in 0.0..100.0,
        economic_status in 0.0..100.0,
        market_dynamics in 0.0..100.0,
    ) -> MacroLocationScore {
        MacroLocationScore {
            social_status,
            economic_status,
            market_dynamics,
        }
    }
}
prop_compose! {
    /// Gives a strategy that generates a Immo, with all fields set.
    pub fn full_immo()(
        wohnflaeche in 0.1..1e6,
        marktwert in 0.1..1e9,
        plane_location in (3e6..5e6, 5e6..6.5e6),
        u in 0.1..1e9,
        wertermittlungsstichtag in naive_date(),
        baujahr in 1900u16..2021u16,
        grundstuecksgroesse in 0.0..1000.0,
        zustand in zustand(),
        ausstattung in 1u8..5u8,
        plz in "\\d{5}",
        kreis in "\\w{3,10}",
        anzahl_garagen in 0u8..50u8,
        additional_stellplaetze in 0u8..50u8,
        plr in proptest::string::string_regex("[0-9]{9}").ok().unwrap(),
        vermietbarkeit in proptest::bool::ANY,
        verwertbarkeit in proptest::bool::ANY,
        erbbaurecht in proptest::bool::ANY,
        drittverwendungsfaehigkeit in proptest::bool::ANY,
        unterkellerungsgrad in 0.0..1.0,
        keller in proptest::bool::ANY,
        objektunterart in objektunterart(),
        anzahl_zimmer in 0.5..20.0,
        verwendung in verwendung(),
        ags0 in "\\d{8}",
        micro_location_scores in micro_location_scores(),
        macro_location_scores in macro_location_scores(),
        regiotyp in 0..50u8,
        centrality in 0..5000u64,
        restnutzungsdauer in 0.0..80.0,
    ) -> Immo {
        ImmoBuilder::default()
        .marktwert(marktwert)
        .wohnflaeche(wohnflaeche)
        .plane_location(plane_location)
        .wertermittlungsstichtag(wertermittlungsstichtag)
        .u(u)
        .baujahr(baujahr)
        .grundstuecksgroesse(grundstuecksgroesse)
        .zustand(zustand)
        .ausstattung(ausstattung).plz(plz).kreis(kreis)
        .anzahl_garagen(anzahl_garagen)
        .anzahl_stellplaetze(anzahl_garagen + additional_stellplaetze)
        .plr_berlin(plr)
        .vermietbarkeit(vermietbarkeit)
        .verwertbarkeit(verwertbarkeit)
        .erbbaurecht(erbbaurecht)
        .drittverwendungsfaehigkeit(drittverwendungsfaehigkeit)
        .unterkellerungsgrad(unterkellerungsgrad)
        .keller(keller)
        .objektunterart(objektunterart)
        .anzahl_zimmer(anzahl_zimmer)
        .verwendung(verwendung)
        .ags0(ags0)
        .micro_location_scores(micro_location_scores)
        .macro_location_scores(macro_location_scores)
        .regiotyp(regiotyp)
        .centrality(centrality)
        .restnutzungsdauer(restnutzungsdauer)
        .build()
        .unwrap()
    }
}

prop_compose! {
    /// This strategy, generates a random naive date in the years 2007 to 2021.
    pub fn naive_date()(year in 2007i32..2021i32, month in 1u32..12u32, day in 1u32..31u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap_or_else(|| NaiveDate::from_ymd(year, month, 15))
    }
}

prop_compose! {
    /// Gives a strategy that generates a Immo, with coordinates in the region of Berlin.
    pub fn berlin_full_immo()(
        mut immo in full_immo(),
        location in (3.75e6..3.83e6, 5.8e6..5.85e6)
    ) -> Immo {
        immo.plane_location = Some(location);
        immo
    }
}

prop_compose! {
    /// Gives a strategy generating between one and `limit` many [full_immo]s.
    /// Calls [set_immo_idxs].
    pub fn full_immos(limit: usize)(mut immos in prop::collection::vec(full_immo(), 1..limit)) -> Vec<Immo>{
        set_immo_idxs(immos.iter_mut());
        immos
    }
}

prop_compose! {
    /// Gives a strategy generating between one and `limit` many [berlin_full_immo]s.
    /// Calls [set_immo_idxs].
    pub fn berlin_full_immos(limit: usize)(mut immos in prop::collection::vec(berlin_full_immo(), 1..limit)) -> Vec<Immo>{
        set_immo_idxs(immos.iter_mut());
        immos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn full_immo_has_full_meta_data_array(immo in full_immo()) {
            assert!(immo.meta_data_array().iter().all(|value| value.is_some()));
        }
    }
}
