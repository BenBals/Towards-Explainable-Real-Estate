#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
//! This module provides a binary to import empirica offer data into the mongodb
//! Run `cargo run --bin import_offers -- --help` for usage.
//! Note that it's totally fine to run this in debug mode. For Berlin it takes ~2s on my device.
use chrono::{DateTime, NaiveDate, Utc};
use common::{database, logging, BpError, BpResult};
use indicatif::ParallelProgressIterator;
use mongodb::{
    bson::{doc, ser::to_document},
    sync::Collection,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use structopt::StructOpt;

const OUTPUT_COLLECTION: &str = "empirica";

#[derive(StructOpt)]
struct Cli {
    /// Where is the csv file?
    #[structopt(name = "FILE", parse(from_os_str))]
    csv_path: std::path::PathBuf,
    /// Where can we find the mongo db?
    #[structopt(long)]
    mongo_url: Option<String>,
    /// If dry run is set, no output will be written to the database
    #[structopt(long)]
    dry_run: bool,
    #[structopt(
        long,
        default_value = "cleaned_80",
        help = "All entries from this collection will be included together with the imported data."
    )]
    base_collection: String,
}

#[derive(Debug, Deserialize)]
struct EmpiricaRowRaw {
    angebot_id: String,
    nachfrageart: String,
    oadr_strasse: String,
    oadr_plz: String,
    oadr_ort: String,
    oadr_kreis: String,
    flaeche: String,
    fl_grundstueck: String,
    kosten: String,
    objekttyp_fein: String,
    baujahr: String,
    anz_zimmer: String,
    aus_klassen_empirica: String,
    zust_klassen_empirica: String,
    kstn_ausreisser: String,
    anz_einheiten: String,
    enddate: String,
}

#[derive(Debug, PartialEq, Eq, Serialize, Clone)]
enum Offertype {
    Sell,
    Rent,
}

#[derive(Debug, Serialize, Clone)]
struct Kurzgutachten {
    #[serde(rename(serialize = "objektangabenWohnflaeche"))]
    area: f64,

    #[serde(
        rename(serialize = "objektangabenAusstattung"),
        skip_serializing_if = "Option::is_none"
    )]
    ausstattung: Option<String>,

    #[serde(
        rename(serialize = "objektangabenAusstattungNote"),
        skip_serializing_if = "Option::is_none"
    )]
    ausstattung_note: Option<String>,

    #[serde(
        rename(serialize = "objektangabenBaujahr"),
        skip_serializing_if = "Option::is_none"
    )]
    baujahr: Option<i32>,

    #[serde(
        rename(serialize = "objektangabenAnzahlZimmer"),
        skip_serializing_if = "Option::is_none"
    )]
    num_rooms: Option<f64>,

    #[serde(
        rename(serialize = "objektangabenAnzahlWohneinheiten"),
        skip_serializing_if = "Option::is_none"
    )]
    num_units: Option<f64>,

    #[serde(
        rename(serialize = "objektangabenZustand"),
        skip_serializing_if = "Option::is_none"
    )]
    zustand: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct EmpiricaRow {
    #[serde(skip_serializing)]
    address: Option<String>,

    #[serde(skip_serializing)]
    angebot_id: String,

    kurzgutachten: Kurzgutachten,

    #[serde(
        rename(serialize = "grundstuecksgroesseInQuadratmetern",),
        skip_serializing_if = "Option::is_none"
    )]
    area_grundstueck: Option<f64>,

    #[serde(rename(serialize = "location"))]
    coordinates: Option<(f64, f64)>,

    #[serde(rename(serialize = "marktwert"))]
    cost: f64,
    kreis: String,

    objektunterart: Option<String>,

    #[serde(skip_serializing)]
    offertype: Offertype,

    ort: String,
    plz: String,

    #[serde(rename(serialize = "glaubhaft"))]
    realisitic_costs: bool,

    empirica: bool,

    #[serde(
        rename(serialize = "wertermittlungsstichtag"),
        skip_serializing_if = "Option::is_none"
    )]
    enddate: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize, Clone)]
struct NominatimResponse {
    lat: String,
    lon: String,
}

impl EmpiricaRow {
    fn from_raw_row(raw: EmpiricaRowRaw) -> Option<Self> {
        let offertype = match raw.nachfrageart.as_str() {
            "kauf" => Offertype::Sell,
            "miete" => Offertype::Rent,
            _ => return None,
        };

        let address = match raw.oadr_strasse.as_ref() {
            "" => None,
            _ => Some(format!(
                "{}, {} {}",
                raw.oadr_strasse, raw.oadr_plz, raw.oadr_ort
            )),
        };

        let area = parse_empirica_float(raw.flaeche.clone())?;
        let cost = parse_empirica_float(raw.kosten.clone())?;

        let enddate: Option<_> = NaiveDate::parse_from_str(&raw.enddate, "%Y-%m-%d")
            // The ZIMDB does not obey ISO 8601 and encodes (most of the time) timezone by setting
            // as time 24h - offset from UTC. We now have to do the same.
            .map(|datetime| datetime.and_hms(23, 0, 0))
            .map(|datetime| DateTime::<Utc>::from_utc(datetime, Utc))
            .ok();

        let ausstattung = parse_empirica_ausstattung(&raw.aus_klassen_empirica);

        Some(EmpiricaRow {
            address,
            angebot_id: raw.angebot_id,
            kurzgutachten: Kurzgutachten {
                area,
                ausstattung_note: ausstattung
                    .clone()
                    .and_then(|ausstattung_string| ausstattung_to_note(&ausstattung_string))
                    .map(|note| note.to_string()),
                ausstattung,
                baujahr: raw.baujahr.parse().ok(),
                num_rooms: raw.anz_zimmer.parse().ok(),
                num_units: raw.anz_einheiten.parse().ok(),
                zustand: parse_empirica_zustand(&raw.zust_klassen_empirica),
            },
            area_grundstueck: raw.fl_grundstueck.parse().ok(),
            coordinates: None,
            cost,
            enddate,
            kreis: raw.oadr_kreis,
            objektunterart: parse_empirica_objekttyp(&raw.objekttyp_fein),
            offertype,
            ort: raw.oadr_ort,
            plz: raw.oadr_plz,
            realisitic_costs: !raw.kstn_ausreisser.contains('1'),
            empirica: true,
        })
    }

    fn geocode(&self, client: &Client) -> BpResult<Self> {
        if let Some(address) = self.address.as_ref() {
            let resp: Vec<NominatimResponse> = client
                .get("http://localhost:7070/search")
                .query(&[
                    ("q", &address[..]),
                    ("countrycodes", "de"),
                    ("limit", "1"),
                    ("format", "jsonv2"),
                ])
                .send()?
                .json()?;
            if resp.is_empty() {
                return Err(BpError::StringBpError(format!(
                    "Could not resolve address {:?}",
                    self.address
                )));
            }
            let data = &resp[0];
            let lon = data.lon.parse::<f64>()?;
            let lat = data.lat.parse::<f64>()?;

            let mut new_self = (*self).clone();
            new_self.coordinates = Some((lon, lat));
            Ok(new_self)
        } else {
            Err(BpError::StringBpError(
                "Trying to geocode row without address".to_string(),
            ))
        }
    }
}

fn parse_empirica_float(string: String) -> Option<f64> {
    string.replace(",", ".").parse::<f64>().ok()
}

fn parse_empirica_objekttyp(string: &str) -> Option<String> {
    match string {
        "EFH" => Some("Einfamilienhaus".into()),
        "MFH" => Some("Mehrfamilienhaus".into()),
        "RH" => Some("Reihenhaus".into()),
        "DHH" => Some("Doppelhaushälfte".into()),
        _ => {
            if string.contains("Whg") {
                Some("Eigentumswohnung".into())
            } else {
                None
            }
        }
    }
}

fn parse_empirica_zustand(string: &str) -> Option<String> {
    match string {
        "gut" => Some("gut".into()),
        "normal" => Some("mittel".into()),
        "schlecht" => Some("schlecht".into()),
        _ => None,
    }
}

fn parse_empirica_ausstattung(string: &str) -> Option<String> {
    match string {
        "einfach" => Some("einfach (Stufe 2)".into()),
        "normal" => Some("mittel (Stufe 3)".into()),
        "gut" => Some("gehoben (Stufe 4)".into()),
        "hochwertig" => Some("sehr gehoben (Stufe 5)".into()),
        _ => None,
    }
}

fn ausstattung_to_note(string: &str) -> Option<u8> {
    match string {
        "einfach (Stufe 2)" => Some(2),
        "mittel (Stufe 3)" => Some(3),
        "gehoben (Stufe 4)" => Some(4),
        "sehr gehoben (Stufe 5)" => Some(5),
        _ => None,
    }
}

fn write_rows(collection: &Collection, rows: &[EmpiricaRow]) {
    // the bulk insert can only handle up to 16MB of data. 50'000 documents have been tried out to
    // work. This might need to be changed if we write more/different data.
    rows.par_chunks(30000).for_each(|chunk| {
        collection
            .insert_many(chunk.iter().map(|row| to_document(row).unwrap_or_else(|e| panic!("Could not serialize row {:?} with error {:?}", row, e))), None)
            .expect("Write failed. Note that this was not performed in a transaction and some chunks may have been written anyways. Consider the database as corrupt until proven otherwise.");
    });
}

fn main() -> BpResult<()> {
    logging::init_logging();
    let args = Cli::from_args();

    log::debug!("Started parsing");

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path(args.csv_path)?;

    let rows: Vec<EmpiricaRow> = rdr
        .deserialize()
        .flatten()
        .flat_map(EmpiricaRow::from_raw_row)
        .collect();

    log::info!("Parsed {} rows", rows.len());

    let relevant_rows: Vec<&EmpiricaRow> = rows
        .iter()
        .filter(|row| {
            row.offertype == Offertype::Sell && row.address.is_some() && row.plz.len() == 5
        })
        .collect();

    log::info!("Of which {} rows are relevant to us", relevant_rows.len());

    log::info!("Geocoding...");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("Could not build HTTP client");
    let geocoded_rows: Vec<EmpiricaRow> = relevant_rows
        .par_iter()
        .progress_count(relevant_rows.len() as u64)
        .filter_map(|row| match row.geocode(&client) {
            Ok(row) => Some(row),
            Err(err) => match err {
                BpError::OtherBpError(inner_err) => {
                    panic!("Failed to geocode row {:?} with error {:?}", row, inner_err);
                }
                BpError::StringBpError(string) => {
                    log::debug!(
                        "Encountered Error during geocoding. The row was ignored.\n\t{:?}",
                        string
                    );
                    None
                }
                unknown_err => unreachable!(
                    "Geocoding row {:?} failed with unexpected error {:?}",
                    row, unknown_err
                ),
            },
        })
        .collect();
    log::info!(
        "Geocoding... DONE\n\t{} rows could not be geocoded",
        relevant_rows.len() - geocoded_rows.len()
    );

    if args.dry_run {
        log::info!("Running as dry run. Not writing to database.");
        return Ok(());
    }

    log::debug!("Connecting to database");
    let db = database::get_database(args.mongo_url.as_deref())?;
    log::info!("Connection to database established");

    let input_collection = db.collection(&args.base_collection);
    let output_collection = db.collection(OUTPUT_COLLECTION);

    log::info!(
        "Copying existing entries from {} to {}...",
        args.base_collection,
        OUTPUT_COLLECTION
    );
    input_collection.aggregate(
        vec![
            doc! {"$match": doc! {
                "plane_location": doc! {
                    "$exists": true
                }
            }},
            doc! {"$project": doc! {
                "grundstucksgroesseInQuadratmetern": 1,
                "kurzgutachten.objektangabenAnzahlWohneinheiten": 1,
                "kurzgutachten.objektangabenAnzahlZimmer": 1,
                "kurzgutachten.objektangabenAusstattung": 1,
                "kurzgutachten.objektangabenAusstattungNote": 1,
                "kurzgutachten.objektangabenZustand": 1,
                "kurzgutachten.objektangabenBaujahr": 1,
                "kurzgutachten.objektangabenWohnflaeche": 1,
                "location": 1,
                "marktwert": 1,
                "objektunterart": 1,
                "ort": 1,
                "kreis_canonic": 1,
                "kreis": 1,
                "AGS_0": 1,
                "glaubhaft": 1,
                "plane_location": 1,
                "plz": 1,
                "wertermittlungsstichtag": 1,
                "Acxiom": 1,
            }},
            doc! {"$addFields": doc! {
                "empirica": false
            }},
            doc! {"$out": OUTPUT_COLLECTION},
        ],
        None,
    )?;
    log::info!(
        "Copying old entries from {} to {}... DONE",
        args.base_collection,
        OUTPUT_COLLECTION
    );

    log::info!("Started writing to database");
    write_rows(&output_collection, &geocoded_rows);
    log::info!("Finished writing to database");

    log::info!("Transforming date type...");
    output_collection.update_many(
        doc! {},
        vec![doc! {"$set": {"wertermittlungsstichtag": {"$toDate": "$wertermittlungsstichtag"}}}],
        None,
    )?;
    log::info!("Transforming date type... DONE");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_empirica_float_examples() {
        assert_eq!(parse_empirica_float("10,".into()), Some(10.0));
        assert_eq!(parse_empirica_float("10,12".into()), Some(10.12));
        assert_eq!(parse_empirica_float("10".into()), Some(10.0));
        assert_eq!(parse_empirica_float("10,,".into()), None);
        assert_eq!(parse_empirica_float("".into()), None);
    }

    #[test]
    fn parse_emiprica_objekttype_example() {
        assert_eq!(
            parse_empirica_objekttyp("EFH"),
            Some("Einfamilienhaus".into())
        );
        assert_eq!(
            parse_empirica_objekttyp("MFH"),
            Some("Mehrfamilienhaus".into())
        );
        assert_eq!(parse_empirica_objekttyp("RH"), Some("Reihenhaus".into()));
        assert_eq!(
            parse_empirica_objekttyp("DHH"),
            Some("Doppelhaushälfte".into())
        );
        assert_eq!(
            parse_empirica_objekttyp("3-Z-Whg"),
            Some("Eigentumswohnung".into())
        );
        assert_eq!(
            parse_empirica_objekttyp("4+-Z-Whg"),
            Some("Eigentumswohnung".into())
        );
    }

    #[test]
    fn parse_empirica_austattung_examples() {
        assert_eq!(
            parse_empirica_ausstattung("einfach"),
            Some("einfach (Stufe 2)".into())
        );
        assert_eq!(
            parse_empirica_ausstattung("normal"),
            Some("mittel (Stufe 3)".into())
        );
        assert_eq!(
            parse_empirica_ausstattung("gut"),
            Some("gehoben (Stufe 4)".into())
        );
        assert_eq!(
            parse_empirica_ausstattung("hochwertig"),
            Some("sehr gehoben (Stufe 5)".into())
        );
    }

    #[test]
    fn parse_empirica_zustand_examples() {
        assert_eq!(parse_empirica_zustand("gut"), Some("gut".into()));
        assert_eq!(parse_empirica_zustand("normal"), Some("mittel".into()));
        assert_eq!(parse_empirica_zustand("schlecht"), Some("schlecht".into()));
    }
}
