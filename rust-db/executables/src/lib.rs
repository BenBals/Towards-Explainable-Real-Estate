//! This crate contains helper functions that are used exclusively in defining binaries, that is
//! main functions.
use common::BpResult;
use mongodb::bson::Document;
use std::fs;
use std::path::Path;

/// Read a JSON query from a file.
/// # Returns
/// - An `Err` if the file could not be read or is not a valid [Document]
pub fn filter_from_path<P>(path: P) -> BpResult<Document>
where
    P: AsRef<Path>,
{
    let mut file_handle = fs::File::open(path)?;
    Ok(serde_json::from_reader(&mut file_handle)?)
}

/// Read a JSON query from a file.
/// # Panics
/// - If the file could not be read or is not a valid [Document]. (That is in all cases [read_query_from_file] return an `Err`)
pub fn filter_from_path_or_panic<P>(path: P) -> Document
where
    P: AsRef<Path>,
{
    filter_from_path(&path).unwrap_or_else(|e| {
        log::error!("Could not open query file at {}", path.as_ref().display());
        log::error!("{:?}", e);
        panic!();
    })
}
