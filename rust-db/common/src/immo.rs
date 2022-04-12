//! This module contains functions and constants related to [Immo].

use std::{borrow::Borrow, convert::TryInto};

use chrono::{Datelike, NaiveDate};
use derive_builder::Builder;
use derive_more::{From, Into};
use mongodb::bson::{doc, from_bson, oid::ObjectId, Bson, Document};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{Keyed, Pointlike};

/// Gives a realistic lower and upper bound on the marktwert.
pub const REALISTIC_MARKTWERT_RANGE: (f64, f64) = (20e3, 2e6);
/// Gives a realistic lower and upper bound on the wohnflaeche.
pub const REALISTIC_WOHNFLAECHE_RANGE: (f64, f64) = (20.0, 2e3);
/// Gives a realistic lower and upper bound on the year of the wertermittlungsstichtag.
pub const REALISTIC_WERTERMITTLUNGSSTICHTAG_YEAR_RANGE: (i32, i32) = (2007, 2021);
/// Gives a realistic lower and upper bound on the sqm price.
pub const REALISTIC_SQM_PRICE_RANGE: (f64, f64) = (100.0, 20e3);
/// Gives a realistic lower and upper bound on the baujahr.
pub const REALISTIC_BAUJAHR_RANGE: (u16, u16) = (1500, 2025);

/// Gives the size of [Immo::meta_data_array]
pub const META_DATA_COUNT: usize = 22;

/// an Identifier for an immo.
/// It should be unique and consecutive.
/// It's useful for a low cost lookup-table for immos and can be set via [set_immo_idxs].
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash, Default)]
pub struct ImmoIdx(usize);

/// Sets unique and consecutive [ImmoIdx]s for the given Immos, starting from 0.
/// If the Immo.idx is already set, calling this function will overwrite it.
pub fn set_immo_idxs<'i>(immos: impl Iterator<Item = &'i mut Immo>) {
    immos
        .enumerate()
        .for_each(|(idx, immo)| immo.idx = Some(ImmoIdx(idx)));
}

/// This enum describes the condition of an estate.
#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)]
pub enum Zustand {
    Katastrophal,
    Schlecht,
    #[serde(rename = "mäßig", alias = "befriedigend")]
    Maessig,
    Mittel,
    Gut,
    #[serde(rename = "sehr gut")]
    SehrGut,
}

impl From<Zustand> for u64 {
    fn from(zustand: Zustand) -> Self {
        zustand as u64
    }
}

/// This enum describes the type of property.
/// You should almost always use [parse_objektunterart] instead of the provider [Deserialize]
/// implementation, as it handles edge cases, see the documentation.
/// writing found in `cleaned_80`, `ZIMDB_joined`, and `empirica`.
#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)]
pub enum Objektunterart {
    Eigentumswohnung,
    Einfamilienhaus,
    EinfamilienhausMitEinliegerWohnung,
    Doppelhaushaelfte,
    Zweifamilienhaus,
    Reihenmittelhaus,
    Reihenendhaus,
    Reihenhaus,
    Mehrfamilienhaus,
}

/// By whom is the property currently used?
#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[allow(missing_docs)]
pub enum Verwendung {
    Eigennutzung,
    Fremdnutzung,
    #[serde(alias = "Eigen- und Fremdnutzung")]
    EigenUndFremdnutzung,
}

/// This struct represents the field scores from `cleaned_80` and `micro_scores`.
#[derive(Debug, Deserialize, PartialEq, PartialOrd, Clone, Copy, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[allow(missing_docs)]
pub struct MicroLocationScore {
    pub all: f64,
    pub education_and_work: f64,
    pub leisure: f64,
    pub public_transport: f64,
    pub shopping: f64,
}

/// This struct represents macro location scores of the zip code the immo is in
#[derive(Debug, Deserialize, PartialEq, PartialOrd, Clone, Copy, Default)]
#[allow(missing_docs)]
pub struct MacroLocationScore {
    #[serde(rename = "scoreSocialStatus")]
    pub social_status: f64,
    #[serde(rename = "scoreEconomicStatus")]
    pub economic_status: f64,
    #[serde(rename = "scoreMarketDynamics")]
    pub market_dynamics: f64,
}

impl MacroLocationScore {
    /// Gives the average of all constituent scores
    pub fn avg(&self) -> f64 {
        (self.social_status + self.economic_status + self.market_dynamics) / 3.0
    }
}

/// This struct represents an estate in our dataset.
#[derive(Clone, Debug, PartialEq, Default, Builder)]
pub struct Immo {
    /// Gives the ObjectId of this estate. equal to the one in the MongoDB
    #[builder(default = "ObjectId::new()")]
    pub id: ObjectId,
    /// Might give the [ImmoIdx], a unique identifer for the Immo.
    /// WARNING: You can only be safe that this is unique if you use [set_immo_idxs] for all immos **at once**
    #[builder(setter(strip_option), default)]
    pub idx: Option<ImmoIdx>,
    /// Might give the valuation of this estate.
    #[builder(setter(strip_option), default)]
    pub marktwert: Option<f64>,
    /// Might give the living area of this estate.
    #[builder(setter(strip_option), default)]
    pub wohnflaeche: Option<f64>,
    /// Might give the location of this estate, in the coordinate reference system EPSG:31467.
    /// This allows to assume that they lie on a flat plane instead of a sphere.
    #[builder(setter(strip_option), default)]
    pub plane_location: Option<(f64, f64)>,
    /// Might give the date when this immo was evaluated. For Empirica data this is the date when the ad ran out.
    #[builder(setter(strip_option), default)]
    pub wertermittlungsstichtag: Option<NaiveDate>,
    /// Might give the environment factor of this estate, known as U (see documentation).
    #[builder(setter(strip_option), default)]
    pub u: Option<f64>,
    /// Might give the year this estate was built
    #[builder(setter(strip_option), default)]
    pub baujahr: Option<u16>,
    /// Might give the overall area of this estate.
    #[builder(setter(strip_option), default)]
    pub grundstuecksgroesse: Option<f64>,
    /// Might give the condition of this estate.
    #[builder(setter(strip_option), default)]
    pub zustand: Option<Zustand>,
    /// Might give a grade for how well this estate is furnished.
    /// Ranges from 1 to 6, 6 is best.
    #[builder(setter(strip_option), default)]
    pub ausstattung: Option<u8>,
    /// Gives the zip code of an immo
    #[builder(setter(into, strip_option), default)]
    pub plz: Option<String>,
    /// Gives the kreis (county) an immo is located in
    #[builder(setter(into, strip_option), default)]
    pub kreis: Option<String>,
    /// Might give the number of garages
    #[builder(setter(strip_option), default)]
    pub anzahl_garagen: Option<u8>,
    /// Might give the number of "stellplätze"
    /// Anzahl Garagen ("kurzgutachten.objektangabenAnzahlGarage") + Anzahl Stellplätze Innen ("kurzgutachten.objektangabenAnzahlStellplaetzeInnen") + Anzahl Stellplätze Aussen ("kurzgutachten.objektangabenAnzahlStellplaetzeAussen") + Anzahl Carports ("kurzgutachten.objektangabenAnzahlCarport")
    #[builder(setter(strip_option), default)]
    pub anzahl_stellplaetze: Option<u8>,
    /// Gives the plr_id for Berlin
    #[builder(setter(into, strip_option), default)]
    pub plr_berlin: Option<String>,
    /// Is the property suited to be let?
    #[builder(setter(strip_option), default)]
    pub vermietbarkeit: Option<bool>,
    /// [Wikipedia](https://www.wikiwand.com/de/Verwertung) says: Vermögen ist verwertbar, wenn seine
    /// Gegenstände verbraucht, übertragen und belastet werden können.
    #[builder(setter(strip_option), default)]
    pub verwertbarkeit: Option<bool>,
    /// [Wikipedia](https://www.wikiwand.com/de/Erbbaurecht) says: Recht, meist gegen Zahlung eines
    /// regelmäßigen sogenannten Erbbauzinses, auf einem Grundstück ein Bauwerk zu errichten oder zu
    /// unterhalten.
    #[builder(setter(strip_option), default)]
    pub erbbaurecht: Option<bool>,
    /// [Logikstik und Immobilien](https://logistik-und-immobilien.de/blog/2018/09/13/drittverwendungsfaehigkeit/)
    /// says: Eigenschaft einer Immobilie, nach Beendigung eines Mietverhältnisses, ohne, oder nur
    /// mit geringen Investitionsaufwand an einen Nachnutzer vermittelt zu werden.
    #[builder(setter(strip_option), default)]
    pub drittverwendungsfaehigkeit: Option<bool>,
    /// Value between 0.0 and 1.0 representing the percentage of cellar
    #[builder(setter(strip_option), default)]
    pub unterkellerungsgrad: Option<f64>,
    /// Does the object have a cellar. Note that this often matches with an unterkellerungsgrad > 0,
    /// but not always.
    #[builder(setter(strip_option), default)]
    pub keller: Option<bool>,
    /// See documentation for [Objektunterart]
    #[builder(setter(strip_option), default)]
    pub objektunterart: Option<Objektunterart>,
    /// How many rooms does the property have?
    #[builder(setter(strip_option), default)]
    pub anzahl_zimmer: Option<f64>,
    /// See documentation for [Verwendung],
    #[builder(setter(strip_option), default)]
    pub verwendung: Option<Verwendung>,
    /// An identifier that specifies the kreis an object is in,
    #[builder(setter(strip_option), default)]
    pub ags0: Option<String>,
    /// regiotyp from acxiom-data
    #[builder(setter(strip_option), default)]
    pub regiotyp: Option<u8>,
    /// centrality score from acxiom-data
    #[builder(setter(strip_option), default)]
    pub centrality: Option<u64>,
    /// See Documentation for [MicroLocationScore]
    #[builder(setter(strip_option), default)]
    pub micro_location_scores: Option<MicroLocationScore>,
    /// See Documentation for [MacroLocationScore]
    #[builder(setter(strip_option), default)]
    pub macro_location_scores: Option<MacroLocationScore>,
    /// How many years can the immo be used in the future?
    #[builder(setter(strip_option), default)]
    pub restnutzungsdauer: Option<f64>,
    /// How many years can the immo be used in total?
    #[builder(setter(strip_option), default)]
    pub gesamtnutzungsdauer: Option<f64>,
    /// We guess the area of a balcony ...
    #[builder(setter(strip_option), default)]
    pub balcony_area: Option<f64>,
    /// Distance to station or bus stop
    #[builder(setter(strip_option), default)]
    pub walking_distance: Option<f64>,
    /// Score for urbanity - land_toshi in database
    /// 1: Urbanization area 2: Urbanization control area 3: Undrawn area 4: Outside the city planning area
    #[builder(setter(strip_option), default)]
    pub urbanity_score: Option<u8>,
    /// floor - house_kaisuu, basement if negative
    #[builder(setter(strip_option), default)]
    pub floor: Option<f64>,
    /// convenience store distance in m
    #[builder(setter(strip_option), default)]
    pub convenience_store_distance: Option<f64>,
    /// distance to junior high school
    #[builder(setter(strip_option), default)]
    pub distance_jun_highschool: Option<f64>,
    /// distance to elementary school
    #[builder(setter(strip_option), default)]
    pub distance_elem_school: Option<f64>,
    /// distance to parking space in m
    #[builder(setter(strip_option), default)]
    pub distance_parking: Option<f64>,
    /// Japanese Prefecture the property is in
    #[builder(setter(strip_option), default)]
    pub prefecture: Option<String>,
}

impl Immo {
    /// Returns the id of the current Immo.
    pub fn id(&self) -> &ObjectId {
        &self.id
    }

    /// Returns the sqaure meter price of the current Immo. As this is not present in the database,
    /// this gets calculated as marktwert / wohnflaeche.
    /// #Example
    /// ```
    /// # use common::*;
    /// # use mongodb::bson::oid::ObjectId;
    /// let immo = ImmoBuilder::default().marktwert(1e5).wohnflaeche(1e2).build().unwrap();
    /// assert_eq!(immo.sqm_price().unwrap(), 1e3);
    /// ```
    pub fn sqm_price(&self) -> Option<f64> {
        Some(self.marktwert? / self.wohnflaeche?)
    }

    /// Returns if the immo has all information neccessary to compute a sqm price
    pub fn has_sqm_price(&self) -> bool {
        self.sqm_price().is_some()
    }

    /// Returns the plane distance to another immo squared.
    /// Null is returned, if either of the given Immos does not have plane_position set.
    /// #Example
    /// ```
    /// # use common::*;
    /// # use mongodb::bson::oid::ObjectId;
    /// let immo_a = ImmoBuilder::default().plane_location((0.0,0.0)).build().unwrap();
    /// let immo_b = ImmoBuilder::default().plane_location((3.0,4.0)).build().unwrap();
    /// assert_eq!(immo_a.plane_distance_squared(&immo_b).unwrap(), 25.0);
    /// ```
    pub fn plane_distance_squared(&self, other: &Self) -> Option<f64> {
        self.plane_location
            .zip(other.plane_location)
            .map(|((x1, y1), (x2, y2))| (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    }

    /// Returns the plane distance to another immo.
    /// Null is returned, if either of the given Immos does not have plane_position set.
    /// #Example
    /// ```
    /// # use common::*;
    /// # use mongodb::bson::oid::ObjectId;
    /// let immo_a = ImmoBuilder::default().plane_location((0.0,0.0)).build().unwrap();
    /// let immo_b = ImmoBuilder::default().plane_location((3.0,4.0)).build().unwrap();
    /// assert_eq!(immo_a.plane_distance(&immo_b).unwrap(), 5.0);
    /// ```
    pub fn plane_distance(&self, other: &Self) -> Option<f64> {
        self.plane_location
            .zip(other.plane_location)
            .map(|((x1, y1), (x2, y2))| (x1 - x2).hypot(y1 - y2))
    }

    /// This function sets all fields to None, which contain direct information about the price.
    /// # Example
    /// ```
    /// # use common::*;
    /// # use mongodb::bson::oid::ObjectId;
    /// let mut immo = ImmoBuilder::default().marktwert(1.0).wohnflaeche(1.0).plane_location((0.0,0.0)).build().unwrap();
    /// immo.clear_price_information();
    /// assert!(immo.marktwert.is_none());
    /// ```
    pub fn clear_price_information(&mut self) {
        self.marktwert = None;
    }

    /// This function sets all fields to None, which contain information about other immos.
    /// # Example
    /// ```
    /// # use common::*;
    /// # use mongodb::bson::oid::ObjectId;
    /// let mut immo = ImmoBuilder::default().u(1.0).build().unwrap();
    /// assert!(immo.u.is_some());
    /// immo.clear_aggregates();
    /// assert!(immo.u.is_none());
    /// ```
    pub fn clear_aggregates(&mut self) {
        self.u = None;
    }

    /// Does the immo-object have realistic values, i.e. values that lie in a given range?
    /// Returns a boolean whether the values are realistic
    /// # Example
    /// ```
    /// # use common::immo::*;
    /// # use mongodb::bson::oid::ObjectId;
    /// # use chrono::NaiveDate;
    /// let unrealistic_immo = ImmoBuilder::default().marktwert(1.0).wohnflaeche(1.0).plane_location((0.0, 0.0)).build().unwrap();
    /// let realistic_immo = ImmoBuilder::default().marktwert(100000.0).wohnflaeche(100.0).plane_location((0.0, 0.0)).wertermittlungsstichtag(NaiveDate::from_ymd(2018,1,1)).baujahr(2001).build().unwrap();
    /// assert!(!unrealistic_immo.has_realistic_values_in_range(REALISTIC_MARKTWERT_RANGE, REALISTIC_WOHNFLAECHE_RANGE, REALISTIC_SQM_PRICE_RANGE, REALISTIC_WERTERMITTLUNGSSTICHTAG_YEAR_RANGE, REALISTIC_BAUJAHR_RANGE));
    /// assert!(realistic_immo.has_realistic_values_in_range(REALISTIC_MARKTWERT_RANGE, REALISTIC_WOHNFLAECHE_RANGE, REALISTIC_SQM_PRICE_RANGE, REALISTIC_WERTERMITTLUNGSSTICHTAG_YEAR_RANGE, REALISTIC_BAUJAHR_RANGE));
    /// ```
    pub fn has_realistic_values_in_range(
        &self,
        marktwert_range: (f64, f64),
        wohnflaeche_range: (f64, f64),
        sqm_range: (f64, f64),
        wertermittlungsstichtag_year: (i32, i32),
        baujahr_range: (u16, u16),
    ) -> bool {
        let realistic_marktwert = self.marktwert.map_or(false, |value| {
            value >= marktwert_range.0 && value <= marktwert_range.1
        });
        let realistic_wohnflaeche = self.wohnflaeche.map_or(false, |value| {
            value >= wohnflaeche_range.0 && value <= wohnflaeche_range.1
        });
        let realistic_sqm_price = self
            .sqm_price()
            .map_or(false, |value| value >= sqm_range.0 && value <= sqm_range.1);
        let realistic_wertermittlungsstichtag_year =
            self.wertermittlungsstichtag.map_or(true, |date| {
                let year = date.year();
                wertermittlungsstichtag_year.0 <= year && year <= wertermittlungsstichtag_year.1
            });
        let realistic_baujahr = self.baujahr.map_or(false, |value| {
            value >= baujahr_range.0 && value <= baujahr_range.1
        });

        realistic_marktwert
            && realistic_wohnflaeche
            && realistic_sqm_price
            && realistic_wertermittlungsstichtag_year
            && realistic_baujahr
    }

    /// Applies [has_realistic_values_in_default_ranges] while using sensible defaults for the
    /// ranges.
    pub fn has_realistic_values_in_default_ranges(&self) -> bool {
        self.has_realistic_values_in_range(
            REALISTIC_MARKTWERT_RANGE,
            REALISTIC_WOHNFLAECHE_RANGE,
            REALISTIC_SQM_PRICE_RANGE,
            REALISTIC_WERTERMITTLUNGSSTICHTAG_YEAR_RANGE,
            REALISTIC_BAUJAHR_RANGE,
        )
    }

    /// gives a category based on the objektunterart of this immo
    /// # Returns
    /// `None` iff self.objektunterart is None else Some(category)
    pub fn objektunterart_category(&self) -> Option<u8> {
        self.objektunterart
            .map(|objektunterart| match objektunterart {
                Objektunterart::Doppelhaushaelfte => 1,
                Objektunterart::Reihenmittelhaus => 1,
                Objektunterart::Reihenendhaus => 1,
                Objektunterart::Reihenhaus => 1,
                Objektunterart::Einfamilienhaus => 2,
                Objektunterart::Zweifamilienhaus => 2,
                Objektunterart::EinfamilienhausMitEinliegerWohnung => 2,
                Objektunterart::Eigentumswohnung => 3,
                Objektunterart::Mehrfamilienhaus => 4,
            })
    }

    /// returns true iff self has the [objektunterart_category] 1 or 2 else false
    pub fn is_objektunterart_category_1_or_2(&self) -> bool {
        matches!(self.objektunterart_category(), Some(1) | Some(2))
    }

    /// gives a category based on the baujahr of this immo
    /// # Returns
    /// `None` iff self.baujahr is None else Some(category)
    /// # Panics
    /// iff Baujahr is not in `[0, 2025]`
    pub fn baujahr_category(&self) -> Option<u8> {
        self.fiktives_baujahr_or_baujahr()
            .map(|baujahr| match baujahr {
                0..=1918 => 1,
                1919..=1948 => 2,
                1949..=1973 => 3,
                1974..=1989 => 4,
                1990..=2025 => 5,
                _ => {
                    log::debug!("Invalid baujahr detected: {:?}", self);
                    5
                }
            })
    }

    /// gives a category based on the wohnflaeche of this immo
    /// # Returns
    /// `None` iff self.wohnflaeche is None else Some(category)
    /// # Panics
    /// iff wohnflaeche is negative
    pub fn wohnflaeche_category(&self) -> Option<u8> {
        self.wohnflaeche.map(|wohnflaeche| {
            if wohnflaeche < 0.0 {
                panic!("Invalid wohnflaeche detected: {:?}", self);
            }
            match self.objektunterart_category() {
                Some(1) | Some(2) => {
                    if wohnflaeche < 100.0 {
                        1
                    } else if wohnflaeche < 150.0 {
                        2
                    } else {
                        3
                    }
                }
                Some(3) => {
                    if wohnflaeche < 40.0 {
                        1
                    } else if wohnflaeche < 90.0 {
                        2
                    } else {
                        3
                    }
                }
                _ => 0,
            }
        })
    }

    /// gives a category based on the grundstuecksgroesse of this immo
    /// # Returns
    /// `None` iff self.grundstuecksgroesse is None else Some(category)
    /// # Panics
    /// iff grundstuecksgroesse is negative
    pub fn grundstuecksgroesse_category(&self) -> Option<u8> {
        self.grundstuecksgroesse.map(|grundstuecksgroesse| {
            if grundstuecksgroesse < 0.0 {
                panic!("Invalid grundstuecksgroesse detected: {:?}", self);
            }
            match self.objektunterart_category() {
                Some(1) | Some(2) => {
                    if grundstuecksgroesse < 250.0 {
                        1
                    } else if grundstuecksgroesse < 500.0 {
                        2
                    } else if grundstuecksgroesse < 750.0 {
                        3
                    } else if grundstuecksgroesse < 1000.0 {
                        4
                    } else if grundstuecksgroesse < 1500.0 {
                        5
                    } else {
                        6
                    }
                }
                _ => 0,
            }
        })
    }

    /// Convert all non-price information to a handy array.
    /// This is useful for computing statistics on them, e.g. for dissimilarity.
    pub fn meta_data_array(&self) -> [Option<f64>; META_DATA_COUNT] {
        [
            self.baujahr.map(|value| value.into()),
            self.wohnflaeche,
            self.zustand.map(|value| (value as u8) as f64),
            self.grundstuecksgroesse,
            self.anzahl_stellplaetze.map(|value| value.into()),
            self.plane_location.map(|(x, _y)| x),
            self.plane_location.map(|(_x, y)| y),
            self.regiotyp.map(|value| value as f64),
            self.micro_location_scores.map(|loc| loc.all),
            self.micro_location_scores.map(|loc| loc.education_and_work),
            self.micro_location_scores.map(|loc| loc.leisure),
            self.micro_location_scores.map(|loc| loc.public_transport),
            self.micro_location_scores.map(|loc| loc.shopping),
            self.restnutzungsdauer,
            self.balcony_area,
            self.convenience_store_distance,
            self.distance_elem_school,
            self.distance_jun_highschool,
            self.distance_parking,
            self.walking_distance,
            self.urbanity_score.map(|value| value as f64),
            self.floor,
        ]
    }

    /// Takes the valuation date an subtracts the time the property has already been used.
    pub fn fiktives_baujahr(&self) -> Option<u16> {
        if let (Some(restnutzungsdauer), Some(gesamtnutzungsdauer), Some(wertermittlungsstichtag)) = (
            self.restnutzungsdauer,
            self.gesamtnutzungsdauer,
            self.wertermittlungsstichtag,
        ) {
            Some(
                (wertermittlungsstichtag.year() - (gesamtnutzungsdauer - restnutzungsdauer) as i32)
                    as u16,
            )
        } else {
            None
        }
    }

    /// Returns `self.fiktives_baujahr(` if it is `Some`, otherwise returns `self.baujahr`
    pub fn fiktives_baujahr_or_baujahr(&self) -> Option<u16> {
        self.fiktives_baujahr().or(self.baujahr)
    }

    /// Combines the micro and macro location scores
    /// # Returns
    /// - None if either micro or macro location scores are unset.
    /// - Some of the average of the micro location score ALL and the average of the three macro
    ///   location scores.
    pub fn combined_location_score(&self) -> Option<f64> {
        self.micro_location_scores
            .zip(self.macro_location_scores)
            .map(|(micro_score, macro_score)| (micro_score.all + macro_score.avg()) / 2.0)
    }

    pub fn deterministic_index(&self, seed: &str) -> u8 {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        let seed_sha = hasher.finalize()[0];

        let mut hasher = Sha256::new();
        hasher.update(self.id.to_string());
        let object_id_sha = hasher.finalize()[0];

        seed_sha ^ object_id_sha
    }
}

/// An additional implementation for the builder generated by the `derive_builder` macro
impl ImmoBuilder {
    /// creates a [[ObjectId]] from the supplied string since a simple cast from string to ObjectId is not possible.
    /// Code is taken from the documentation of `derive_builder`
    pub fn id_from_string(&mut self, id_str: &str) -> &mut Self {
        let mut new = self;
        new.id = Some(ObjectId::with_string(id_str).expect("String for id was not accepted"));
        new
    }

    /// takes a document and returns an immo - not an immo-builder!
    pub fn from_document(document: Document) -> Self {
        let kurzgutachten = document
            .get("kurzgutachten")
            .and_then(|bson| bson.as_document());
        let objektuebersicht = document
            .get("objektuebersicht")
            .and_then(|bson| bson.as_document());
        let garagen = kurzgutachten
            .and_then(|doc| doc.get("objektangabenAnzahlGaragen"))
            .and_then(bson_number_as_i64)
            .and_then(|num| num.try_into().ok());
        let stellplaetze_innen: Option<u8> = kurzgutachten
            .and_then(|doc| doc.get("objektangabenAnzahlStellplaetzeInnen"))
            .and_then(bson_number_as_i64)
            .and_then(|num| num.try_into().ok());
        let stellplaetze_aussen: Option<u8> = kurzgutachten
            .and_then(|doc| doc.get("objektangabenAnzahlStellplaetzeAussen"))
            .and_then(bson_number_as_i64)
            .and_then(|num| num.try_into().ok());
        let carports: Option<u8> = kurzgutachten
            .and_then(|doc| doc.get("objektangabenAnzahlCarport"))
            .and_then(bson_number_as_i64)
            .and_then(|num| num.try_into().ok());
        let stellplaetze = Some(
            garagen.unwrap_or(0)
                + stellplaetze_innen.unwrap_or(0)
                + stellplaetze_aussen.unwrap_or(0)
                + carports.unwrap_or(0),
        );

        let acxiom = document.get("Acxiom").and_then(|bson| bson.as_document());

        Self {
            id: document
                .get("_id")
                .and_then(|obj| obj.as_object_id().cloned()),
            idx: None,
            marktwert: document
                .get("marktwert")
                .map(|bson| bson_number_as_f64(bson)),
            wohnflaeche: kurzgutachten
                .and_then(|doc| doc.get("objektangabenWohnflaeche"))
                .map(|bson| bson_number_as_f64(bson)),
            plane_location: document.get("plane_location").and_then(|bson| {
                let array = bson.as_array()?;
                if array.len() == 2 {
                    let x = array[0].as_f64()?;
                    let y = array[1].as_f64()?;

                    Some(Some((x, y)))
                } else {
                    None
                }
            }),
            wertermittlungsstichtag: document
                .get("wertermittlungsstichtag")
                .and_then(|bson| bson.as_datetime())
                .map(|date_time| Some(date_time.naive_utc().date())),
            u: document
                .get("U_Germany")
                .map(|bson| bson_number_as_f64(bson)),
            baujahr: kurzgutachten
                .and_then(|doc| doc.get("objektangabenBaujahr"))
                .and_then(bson_number_as_i64)
                .map(|baujahr| baujahr.try_into().ok()),
            grundstuecksgroesse: document
                .get("grundstuecksgroesseInQuadratmetern")
                .map(|bson| bson_number_as_f64(bson)),
            zustand: kurzgutachten
                .and_then(|doc| doc.get("objektangabenZustand"))
                .map(|bson| from_bson(bson.clone()).ok()),
            ausstattung: kurzgutachten
                .and_then(|doc| doc.get("objektangabenAusstattungNote"))
                .and_then(|bson| bson.as_str())
                .map(|note| note.parse().ok()),
            plz: document
                .get("plz")
                .and_then(|bson| bson.as_str())
                .map(|plz_str| Some(plz_str.into())),
            kreis: document
                .get("kreis_canonic")
                .and_then(|bson| bson.as_str())
                .map(|kreis_str| Some(kreis_str.into())),
            anzahl_garagen: Some(garagen),
            anzahl_stellplaetze: Some(stellplaetze),
            plr_berlin: document
                .get("berlin_plr_id")
                .and_then(|bson| bson.as_str())
                .map(|str| Some(str.to_owned())),
            vermietbarkeit: kurzgutachten
                .and_then(|doc| doc.get("vermietbarkeit"))
                .and_then(|bson| bson.as_str())
                .map(|value| parse_ja_nein(value)),
            verwertbarkeit: kurzgutachten
                .and_then(|doc| doc.get("verwertbarkeit"))
                .and_then(|bson| bson.as_str())
                .map(|value| parse_ja_nein(value)),
            erbbaurecht: objektuebersicht
                .and_then(|doc| doc.get("erbbaurechtBesteht"))
                .map(|bson| bson.as_bool()),
            drittverwendungsfaehigkeit: kurzgutachten
                .and_then(|doc| doc.get("drittverwendungsfaehigkeit"))
                .and_then(|bson| bson.as_str())
                .map(|value| parse_ja_nein(value)),
            unterkellerungsgrad: kurzgutachten
                .and_then(|doc| doc.get("objektangabenUnterkellerungsgrad"))
                .and_then(|bson| bson.as_str())
                .map(|value| parse_percentage(value)),
            keller: kurzgutachten
                .and_then(|doc| doc.get("objektangabenKeller"))
                .and_then(|bson| bson.as_str())
                .map(|value| Some(value == "unterkellert")),
            objektunterart: document
                .get("objektunterart")
                .or_else(|| kurzgutachten.and_then(|doc| doc.get("objektunterart")))
                .and_then(|bson| bson.as_str())
                .map(|str_value| parse_objektunterart(str_value)),
            anzahl_zimmer: kurzgutachten
                .and_then(|doc| doc.get("objektangabenAnzahlZimmer"))
                .map(|bson| bson_number_as_f64(bson)),
            verwendung: kurzgutachten
                .and_then(|doc| doc.get("objektangabenVerwendung"))
                .map(|bson| from_bson(bson.clone()).ok()),
            ags0: document
                .get("AGS_0")
                .and_then(|bson| bson.as_str())
                .map(|str| Some(str.to_owned())),
            regiotyp: acxiom
                .and_then(|doc| doc.get("regioTyp"))
                .map(|bson| bson_number_as_i64(bson).map(|value| value as u8)),
            centrality: acxiom
                .and_then(|doc| doc.get("centrality"))
                .map(|bson| bson_number_as_i64(bson).map(|value| value as u64)),
            micro_location_scores: document
                .get("scores")
                .map(|bson| from_bson(bson.clone()).ok()),
            macro_location_scores: document
                .get("macro_scores")
                .map(|bson| from_bson::<MacroLocationScore>(bson.clone()).ok()),
            restnutzungsdauer: document
                .get("restnutzungsdauer")
                .map(|bson| bson_number_as_f64(bson)),
            gesamtnutzungsdauer: document
                .get("gesamtnutzungsdauer")
                .map(|bson| bson_number_as_f64(bson)),
            balcony_area: document
                .get("balcony_area")
                .map(|bson| bson_number_as_f64(bson)),
            convenience_store_distance: document
                .get("convenience_distance")
                .map(bson_number_as_f64),
            distance_elem_school: document.get("school_ele_distance").map(bson_number_as_f64),
            distance_jun_highschool: document.get("school_jun_distance").map(bson_number_as_f64),
            distance_parking: document.get("distance_parking").map(bson_number_as_f64),
            floor: document.get("house_kaisuu").map(bson_number_as_f64),
            urbanity_score: document
                .get("land_toshi")
                .map(|bson| bson_number_as_i64(bson).map(|value| value as u8)),
            walking_distance: document.get("walk_distance1").map(bson_number_as_f64),
            prefecture: document
                .get("prefecture")
                .and_then(|bson| bson.as_str())
                .map(|str| Some(str.to_owned())),
        }
    }
}

/// Returns the filter-document used for mongo to return only realistic values
/// Note that this only implements the two most important filters from
/// [Immo::has_realistic_values_in_range]. You should run the filter in Rust afterwards.
pub fn pre_filter_immos(marktwert_range: (f64, f64), wohnflaeche_range: (f64, f64)) -> Document {
    doc! {
        "plane_location": {"$exists": true},
        "marktwert": {"$gte": marktwert_range.0, "$lte": marktwert_range.1},
        "kurzgutachten.objektangabenWohnflaeche": {"$gte": wohnflaeche_range.0, "$lte": wohnflaeche_range.1},
        "erbbaurecht": {"$ne": true}
    }
}

/// Returns a query document document that only checks for the existance of absolutely neccessary
/// ttributes
pub fn necessary_filters() -> Document {
    doc! {
        "plane_location": {"$exists": true},
        "kurzgutachten.objektangabenWohnflaeche": {"$gt": 0.0},
        "marktwert": {"$gt": 0.0},
    }
}

impl Pointlike for Immo {
    fn x(&self) -> u64 {
        self.plane_location
            .expect("Immo must have a plane location to be a pointlike")
            .0 as u64
    }
    fn y(&self) -> u64 {
        self.plane_location
            .expect("Immo must have a plane location to be a pointlike")
            .1 as u64
    }
}

impl Keyed for Immo {
    type Key = ObjectId;
    fn key(&self) -> ObjectId {
        self.borrow().id().clone()
    }
}

fn parse_ja_nein(input: &str) -> Option<bool> {
    let mut cleaned: String = input.trim().into();
    cleaned.make_ascii_lowercase();
    match cleaned.as_ref() {
        "ja" => Some(true),
        "nein" => Some(false),
        _ => None,
    }
}

fn parse_percentage(input: &str) -> Option<f64> {
    let string = input.trim().replace("%", "");
    string.parse::<f64>().ok().map(|value| value / 100.0)
}

/// The function parses [Objektunterart] form string slices while handling all alternative ways of
/// writing found in `cleaned_80`, `ZIMDB_joined` and `empiricia`.
/// If you find a failure case, please add a test for it.
fn parse_objektunterart(input: &str) -> Option<Objektunterart> {
    let mut cleaned: String = input.trim().into();
    cleaned.make_ascii_lowercase();
    if cleaned.contains("etw") || cleaned.contains("eigentumswohnung") {
        return Some(Objektunterart::Eigentumswohnung);
    }
    if cleaned.contains("doppelhaus") {
        return Some(Objektunterart::Doppelhaushaelfte);
    }
    if cleaned.contains("reihenendhaus") {
        return Some(Objektunterart::Reihenendhaus);
    }
    if cleaned.contains("reihenmittelhaus") {
        return Some(Objektunterart::Reihenmittelhaus);
    }
    if cleaned.contains("zweifamilienhaus") {
        return Some(Objektunterart::Zweifamilienhaus);
    }
    if cleaned.contains("reihenhaus") {
        return Some(Objektunterart::Reihenhaus);
    }
    if cleaned.contains("mehrfamilienhaus") {
        return Some(Objektunterart::Mehrfamilienhaus);
    }
    if cleaned.contains("einlieger") {
        return Some(Objektunterart::EinfamilienhausMitEinliegerWohnung);
    }
    if cleaned.contains("efh") || cleaned.contains("einfamilienhaus") {
        return Some(Objektunterart::Einfamilienhaus);
    }

    serde_json::from_str(cleaned.as_ref()).ok()
}

/// This will return any number type supported by BSON as an f64 and None for all others.
/// In particiular, this function supports returning BSON integer type as f64s.
fn bson_number_as_f64(bson: &Bson) -> Option<f64> {
    match bson {
        Bson::Double(double) => Some(*double),
        Bson::Int32(int) => Some(*int as f64),
        Bson::Int64(int) => Some(*int as f64),
        _ => None,
    }
}

/// This will return any number type supported by BSON as an i64 and None for all others.
/// In particiular, for the double type, the value will be truncated.
fn bson_number_as_i64(bson: &Bson) -> Option<i64> {
    match bson {
        Bson::Double(double) => Some(*double as i64),
        Bson::Int32(int) => Some(*int as i64),
        Bson::Int64(int) => Some(*int),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use assert_approx_eq::assert_approx_eq;
    use chrono::{TimeZone, Utc};
    use mongodb::bson::Bson;

    use super::*;

    #[test]
    fn sqm_price_example() {
        assert_approx_eq!(
            ImmoBuilder::default()
                .id_from_string("5edd91d742936b07d4652e4d")
                .marktwert(420.0)
                .wohnflaeche(100.0)
                .plane_location((1.0, 1.0))
                .build()
                .unwrap()
                .sqm_price()
                .unwrap(),
            4.2
        );
    }

    #[test]
    fn fiktives_baujahr_example() {
        assert_eq!(
            ImmoBuilder::default()
                .id_from_string("5edd91d742936b07d4652e4d")
                .wertermittlungsstichtag(NaiveDate::from_ymd(2021, 1, 1))
                .gesamtnutzungsdauer(80.0)
                .restnutzungsdauer(44.0)
                .build()
                .unwrap()
                .fiktives_baujahr()
                .unwrap(),
            1985u16
        );
    }

    #[test]
    fn parse_ja_nein_examples() {
        assert_eq!(parse_ja_nein("ja"), Some(true));
        assert_eq!(parse_ja_nein("Ja"), Some(true));
        assert_eq!(parse_ja_nein(" Ja  \n"), Some(true));
        assert_eq!(parse_ja_nein("nein"), Some(false));
        assert_eq!(parse_ja_nein("nEiN "), Some(false));
        assert_eq!(parse_ja_nein("janein "), None);
        assert_eq!(parse_ja_nein(""), None);
    }

    #[test]
    fn parse_percentage_examples() {
        assert_eq!(parse_percentage("100%"), Some(1.0));
        assert_eq!(parse_percentage("75%"), Some(0.75));
        assert_approx_eq!(parse_percentage("7").unwrap(), 0.07);
        assert_approx_eq!(parse_percentage("10.5%").unwrap(), 0.105);
        assert_eq!(parse_percentage("gogglebod"), None);
        assert_eq!(parse_percentage(""), None);
    }

    #[test]
    fn deserialization_does_not_depend_on_number_type() {
        let doc = doc! {
            "_id" : ObjectId::with_string("5dde4c55c7c28b3bb4f5ad31").unwrap(),
            "grundstuecksgroesseInQuadratmetern" : 710,
            "marktwert" : 285000,
            "kurzgutachten" : {
                "objektangabenAnzahlGaragen" : 3.5,
                "objektangabenAnzahlZimmer" : 4.0,
                "objektangabenWohnflaeche" : 176,
                "objektangabenBaujahr" : 1976.0,
                "objektangabenZustand" : "gut",
            },
            "U_Germany" : Bson::Int32(179941684),
        };
        let immo_from_doc = ImmoBuilder::from_document(doc).build().unwrap();
        assert!(immo_from_doc.grundstuecksgroesse.is_some());
        assert!(immo_from_doc.anzahl_garagen.is_some());
        assert!(immo_from_doc.anzahl_zimmer.is_some());
        assert!(immo_from_doc.wohnflaeche.is_some());
        assert!(immo_from_doc.baujahr.is_some());
        assert!(immo_from_doc.zustand.is_some());
    }

    #[test]
    fn deserialize_immo() {
        let doc = doc! {
            "_id" : ObjectId::with_string("5dde4c55c7c28b3bb4f5ad31").unwrap(),
            "grundstuecksgroesseInQuadratmetern" : 715.0,
            "kreis_canonic" : "Berlin, Stadt",
            "plz": "18057",
            "marktwert" : 285000.0,
            "wertermittlungsstichtag" : Bson::DateTime(Utc.ymd(2014,10,20).and_hms(0,0,0)),
            "objektunterart" : "Einfamilienhaus",
            "kurzgutachten" : {
                "objektangabenAusstattung" : "mittel (Stufe 3)",
                "objektangabenAnzahlGaragen" : 3,
                "objektangabenAnzahlCarport" : 1,
                "objektangabenAnzahlStellplaetzeInnen" : 2,
                "objektangabenAnzahlZimmer" : 4,
                "objektangabenAusstattungNote" : "3",
                "objektangabenWohnflaeche" : 176.0,
                "objektangabenBaujahr" : Bson::Int32(1969),
                "objektangabenZustand" : "gut",
                "objektangabenKeller" : "unterkellert",
                "objektangabenUnterkellerungsgrad" : "75%",
                "objektangabenVerwendung": "Eigen- und Fremdnutzung",
                "vermietbarkeit": "ja",
                "verwertbarkeit": "ja",
                "drittverwendungsfaehigkeit": "nein",
            },
            "macro_scores": {
                "scoreSocialStatus": 48.15726078769166,
                "scoreEconomicStatus": 51.95639819522995,
                "scoreMarketDynamics": 43.57008136419325,
            },
            "scores": {
                "EDUCATION_AND_WORK": 94.0,
                "SHOPPING": 71.0,
                "LEISURE": 93.0,
                "PUBLIC_TRANSPORT": 37.0,
                "ALL": 74.0
            },
            "objektuebersicht": {
                "erbbaurechtBesteht": false,

            },
            "Acxiom": {
                "centrality": Bson::Int32(100),
                "regioTyp": Bson::Int32(14),
            },
            "plane_location" : [
                3333200.5007,
                5709890.8330
            ],
            "berlin_plr_id" : "04400835",
            "U_Germany" : 179941684.40,
            "glaubhaft" : true,
            "restnutzungsdauer": 44,
            "gesamtnutzungsdauer": 80,
        };
        println!("{:?}", doc);
        let immo_from_doc = ImmoBuilder::from_document(doc).build().unwrap();
        let real_immo = ImmoBuilder::default()
            .id_from_string("5dde4c55c7c28b3bb4f5ad31")
            .marktwert(285000.0)
            .wohnflaeche(176.0)
            .plane_location((3333200.5007, 5709890.8330))
            .wertermittlungsstichtag(NaiveDate::from_ymd(2014, 10, 20))
            .u(179941684.40)
            .baujahr(1969)
            .grundstuecksgroesse(715.0)
            .zustand(Zustand::Gut)
            .ausstattung(3)
            .plz("18057")
            .kreis("Berlin, Stadt")
            .anzahl_garagen(3)
            .anzahl_stellplaetze(6)
            .plr_berlin("04400835")
            .vermietbarkeit(true)
            .verwertbarkeit(true)
            .erbbaurecht(false)
            .drittverwendungsfaehigkeit(false)
            .unterkellerungsgrad(0.75)
            .keller(true)
            .objektunterart(Objektunterart::Einfamilienhaus)
            .anzahl_zimmer(4.0)
            .verwendung(Verwendung::EigenUndFremdnutzung)
            .micro_location_scores(MicroLocationScore {
                all: 74.0,
                education_and_work: 94.0,
                leisure: 93.0,
                public_transport: 37.0,
                shopping: 71.0,
            })
            .macro_location_scores(MacroLocationScore {
                social_status: 48.15726078769166,
                economic_status: 51.95639819522995,
                market_dynamics: 43.57008136419325,
            })
            .regiotyp(14)
            .centrality(100)
            .restnutzungsdauer(44.0)
            .gesamtnutzungsdauer(80.0)
            .build()
            .unwrap();

        assert_eq!(immo_from_doc, real_immo);
    }

    #[test]
    fn parse_objektunterart_example() {
        assert_eq!(
            parse_objektunterart("Eigentumswohnung"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("eIGENTUMSWOHNUNG"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Eigentumswohnung selbstbewohnt"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("vermietete Eigentumswohnung"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Eigentumswohnung vermietet"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("ETW im Mehrfamilienhaus"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Eigentumswohnung (Wohnimmobilie)"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Eigentumswohnung(en)"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Eigentumswohnungen"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Eigentumswohnung als Reihenmittelhaus"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Eigentumswohnung unvermietet"),
            Some(Objektunterart::Eigentumswohnung)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus"),
            Some(Objektunterart::Einfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus "),
            Some(Objektunterart::Einfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart(" Einfamilienhaus\n"),
            Some(Objektunterart::Einfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (freistehend)"),
            Some(Objektunterart::Einfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (freistehend) GND 80 Jahre"),
            Some(Objektunterart::Einfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (freistehend, GND 80 J.)"),
            Some(Objektunterart::Einfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (nicht freistehend)"),
            Some(Objektunterart::Einfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus mit Einliegerwohnung"),
            Some(Objektunterart::EinfamilienhausMitEinliegerWohnung)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus mit Einliegerwohnung "),
            Some(Objektunterart::EinfamilienhausMitEinliegerWohnung)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus mit Einliegerwohung"),
            Some(Objektunterart::EinfamilienhausMitEinliegerWohnung)
        );
        assert_eq!(
            parse_objektunterart("EFH mit Einliegerwohnung"),
            Some(Objektunterart::EinfamilienhausMitEinliegerWohnung)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (freistehend) mit Einliegerwohnung"),
            Some(Objektunterart::EinfamilienhausMitEinliegerWohnung)
        );
        assert_eq!(
            parse_objektunterart("Doppelhaushälfte"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Doppelhaus"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (Doppelhaushälfte)"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (Doppelhaushälfte)"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Doppelhaushälfte (Wohnimmobilie)"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Doppelhaushälfte (Ein-/Zweifamilienhaus)"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Doppelhaushälfte, Reihenendhaus"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Doppelhaushälfte GND 80 Jahre"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Doppelhaushälfte (GND 80 Jahre)"),
            Some(Objektunterart::Doppelhaushaelfte)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (freistehend)"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (freistehend) GND 80 Jahre"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (angebaut)"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (freistehend, GND 80 J.)"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (freistejend, GND 80 J.)"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (Reihenhaus)"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus freistehend"),
            Some(Objektunterart::Zweifamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenmittelhaus"),
            Some(Objektunterart::Reihenmittelhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (Reihenmittelhaus)"),
            Some(Objektunterart::Reihenmittelhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenmittelhaus (Wohnimmobilie)"),
            Some(Objektunterart::Reihenmittelhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenmittelhaus (GND 80 Jahre)"),
            Some(Objektunterart::Reihenmittelhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (Reihenmittelhaus)"),
            Some(Objektunterart::Reihenmittelhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenmittelhaus (Ein-/Zweifamilienhaus)"),
            Some(Objektunterart::Reihenmittelhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenmittelhaus GND 80 Jahre"),
            Some(Objektunterart::Reihenmittelhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenendhaus"),
            Some(Objektunterart::Reihenendhaus)
        );
        assert_eq!(
            parse_objektunterart("Zweifamilienhaus (Reihenendhaus)"),
            Some(Objektunterart::Reihenendhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenendhaus (Wohnimmobilie)"),
            Some(Objektunterart::Reihenendhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenendhaus (Ein-/Zweifamilienhaus)"),
            Some(Objektunterart::Reihenendhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenendhaus GND 80 Jahre"),
            Some(Objektunterart::Reihenendhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenendhaus (GND 80 Jahre)"),
            Some(Objektunterart::Reihenendhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (Reihenendhaus)"),
            Some(Objektunterart::Reihenendhaus)
        );
        assert_eq!(
            parse_objektunterart("Reihenhaus"),
            Some(Objektunterart::Reihenhaus)
        );
        assert_eq!(
            parse_objektunterart("Einfamilienhaus (Reihenhaus)"),
            Some(Objektunterart::Reihenhaus)
        );
        assert_eq!(
            parse_objektunterart("Mehrfamilienhaus"),
            Some(Objektunterart::Mehrfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("Mehrfamilienhaus (ab 3 Wohneinheiten)"),
            Some(Objektunterart::Mehrfamilienhaus)
        );
        assert_eq!(
            parse_objektunterart("3E15DE6B-E8BB-4027-BF72-CD4AF303413D"),
            None
        );
        assert_eq!(parse_objektunterart(""), None);
    }

    #[test]
    fn bson_number_as_f64_examples() {
        assert_eq!(bson_number_as_f64(&Bson::Double(42.0)), Some(42.0));
        assert_eq!(bson_number_as_f64(&Bson::Int32(42)), Some(42.0));
        assert_eq!(bson_number_as_f64(&Bson::Int64(42)), Some(42.0));
        assert_eq!(bson_number_as_f64(&Bson::Boolean(true)), None);
    }

    #[test]
    fn bson_number_as_i64_examples() {
        assert_eq!(bson_number_as_i64(&Bson::Double(42.7)), Some(42));
        assert_eq!(bson_number_as_i64(&Bson::Int32(42)), Some(42));
        assert_eq!(bson_number_as_i64(&Bson::Int64(42)), Some(42));
        assert_eq!(bson_number_as_i64(&Bson::Boolean(true)), None);
    }

    #[test]
    fn deterministic_hashing_example() {
        let mut immo = Immo::default();
        immo.id = ObjectId::from_str("610ce9d3d1a638258e871b95").unwrap();
        assert_eq!(immo.deterministic_index("seeed"), 227);

        let mut immo = Immo::default();
        immo.id = ObjectId::from_str("abcdef1234567890abcdef12").unwrap();
        assert_eq!(immo.deterministic_index("42"), 81);

        let mut immo = Immo::default();
        immo.id = ObjectId::from_str("103456107ab5ebc42e4ea103").unwrap();
        assert_eq!(
            immo.deterministic_index("dnn-ea-cbr-los-randomness-is-deterministic"),
            238
        );

        let mut immo = Immo::default();
        immo.id = ObjectId::from_str("1303abec1032cdf442323e20").unwrap();
        assert_eq!(immo.deterministic_index("hund sind able(n/c)kend"), 26);

        let mut immo = Immo::default();
        immo.id = ObjectId::from_str("123cde212376abc125666666").unwrap();
        assert_eq!(immo.deterministic_index("numberphile"), 155);
    }
}
