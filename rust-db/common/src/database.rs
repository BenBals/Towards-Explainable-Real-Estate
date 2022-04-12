//! This module captures all database related constants and functions.
use super::*;
use crate::immo::{
    pre_filter_immos, set_immo_idxs, REALISTIC_MARKTWERT_RANGE, REALISTIC_WOHNFLAECHE_RANGE,
};
use mongodb::{
    bson::{doc, Document},
    error::Result as MongoResult,
    options::FindOptions,
    sync::{Client, Collection, Database},
};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;

/// This is the URI of our database.
pub const DATABASE_URI: &str = "mongodb://localhost:27017";

/// Get our database (hpidb_bp) at a given mongo server.
/// If you give None as server url, the database is assumed to be available at your localhost on the
/// standard port.
pub fn get_database(url: Option<&str>) -> MongoResult<Database> {
    let default_uri = std::env::var("MONGO_HOST").unwrap_or_else(|_| DATABASE_URI.into());
    let client = Client::with_uri_str(url.unwrap_or(&default_uri))?;
    let database_name = std::env::var("MONGO_DATABASE").unwrap_or_else(|_| {
        if default_uri.contains("japan") {
            "japan_lifullhome".into()
        } else {
            "hpidb_bp".into()
        }
    });
    Ok(client.database(&database_name))
}

/// Returns a handle to the specified collection in our database.
/// First input is the url of the server. If you give none, a default value will be used.
pub fn get_collection(mongo_url: Option<&str>, name: &str) -> MongoResult<Collection> {
    Ok(get_database(mongo_url)?.collection(name))
}

/// Returns a handle to our default collection.
pub fn get_default_collection() -> MongoResult<Collection> {
    get_collection(None, "cleaned_80")
}

/// This function reads Immos from the database.
///
/// # Arguments
/// * limit: if this is Some(number) then at most number Immos will be queried from the database and returned
/// * filter: if this is Some(document) then only Immos for which document applies will be queried
/// * projection: if this is Some(document) then only matching fields are returned, if it's None all fields are returned
///
/// # Examples
/// ```no_run
/// # use common::database::read;
/// // Get all Immos
/// read(None, None, None);
///
/// # use mongodb::bson::doc;
/// // only get at most 100 Immos in Berlin
/// read(Some(100), Some(doc! {"kreis": "Berlin, Stadt"}), Some(doc! {"kreis": true, "marktwert": true}));
/// ```
pub fn read(
    limit: Option<i64>,
    filter: Option<Document>,
    projection: Option<Document>,
) -> MongoResult<Vec<Immo>> {
    read_from_collection(limit, filter, projection, None)
}

/// This function reads Immos from a specific collection in the database.
/// It works exactly like [read], so see the documentation there, but instead of reading from a
/// default collection, the caller may specify one themselves.
pub fn read_from_collection(
    limit: Option<i64>,
    filter: Option<Document>,
    projection: Option<Document>,
    collection: Option<&str>,
) -> MongoResult<Vec<Immo>> {
    let col = match collection {
        None => get_default_collection()?,
        Some(coll) => get_collection(None, coll)?,
    };

    let options = FindOptions::builder()
        .limit(limit)
        .projection(projection)
        .build();

    let mut output = Vec::new();
    for result in col.find(filter, options)? {
        match result {
            Ok(document) => output.push(
                ImmoBuilder::from_document(document)
                    .build()
                    .expect("could not transform document into Immo"),
            ),
            Err(e) => return Err(e),
        }
    }
    set_immo_idxs(output.iter_mut());

    Ok(output)
}

/// a simplified version of database read operation while directly filtering outliers.
/// Returns a vector of Immos which all have a marktwert and a wohnflaeche within a reasonable range (marktwert in [20.000, 5.000.000] and wohnflaeche in [20, 2000])
pub fn read_reasonable_immos(
    limit: Option<i64>,
    filter: Option<Document>,
    projection: Option<Document>,
) -> MongoResult<Vec<Immo>> {
    read_reasonable_immos_from_collection(limit, filter, projection, None)
}

/// a simplified version of database read operation while directly filtering outliers.
/// Returns a vector of Immos which all have a marktwert and a wohnflaeche within a reasonable range (marktwert in [20.000, 5.000.000] and wohnflaeche in [20, 2000])
pub fn read_reasonable_immos_from_collection(
    limit: Option<i64>,
    filter: Option<Document>,
    projection: Option<Document>,
    collection: Option<&str>,
) -> MongoResult<Vec<Immo>> {
    let reasonable_filter =
        pre_filter_immos(REALISTIC_MARKTWERT_RANGE, REALISTIC_WOHNFLAECHE_RANGE);

    let mongo_filter = match filter {
        Some(inner_filter) => doc! { "$and": [inner_filter, reasonable_filter] },
        None => reasonable_filter,
    };

    let results =
        database::read_from_collection(limit, Some(mongo_filter), projection, collection)?;
    Ok(results
        .iter()
        .filter(|immo| immo.has_realistic_values_in_default_ranges())
        .cloned()
        .collect())
}

/// Spawn n_workers threads, each of which get a chunk of the elements to process.
/// Each worker then calls the action on each element in its chunk .
/// Note that since the mongodb driver does not support transactions, some operations might fail while others land in the database.
/// Your action is responsible for handling errors.
pub fn par_op<T, F>(elements: &[T], n_workers: usize, action: F)
where
    T: Sync,
    F: Fn(&T) + Sync,
{
    // this computes ceil(elements.len() / n_workers).
    let n_operations_per_worker = (elements.len() + n_workers - 1) / n_workers;

    elements
        .par_chunks(n_operations_per_worker)
        .for_each(|chunk| {
            for element in chunk.iter() {
                action(element);
            }
        });
}

pub fn write_predictions_to_database(collection: &str, key: &str, pairs: &[(Immo, Immo)]) {
    let col: Collection =
        database::get_collection(None, collection).expect("Could not open connection to database");

    log::info!(
        "Writing values to collection {}...\n\tn={}",
        collection,
        pairs.len()
    );
    database::par_op(pairs, 50, |(original, predicted)| {
        if let Some(predicted_marktwert) = predicted.marktwert {
            col.update_one(
                doc! {"_id": original.id().clone()},
                vec![doc! {"$addFields": {"predictions": {key: predicted_marktwert}}}],
                None,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "Error {} while writing prediction pair\n\t{:#?}\n\t{:#?}",
                    error, original, predicted
                );
            });
        } else {
            log::warn!(
                "No `marktwert` on predicted immo, skipping in database write\n\t{:#?}",
                predicted
            );
        }
    });
    log::info!("Writing values to collection... DONE");
}
