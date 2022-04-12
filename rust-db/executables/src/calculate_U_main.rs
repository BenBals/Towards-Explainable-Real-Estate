#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
#![allow(non_snake_case)]
use algorithms::calculate_environment::*;
use common::{database, logging, BpResult, Immo};
use mongodb::bson::{doc, oid::ObjectId, Document};
use mongodb::sync::Collection;

use dissimilarities::SqmPriceDissimilarity;
use std::collections::HashMap;
use structopt::StructOpt;

fn write_U_value(
    col: &Collection,
    immo: &Immo,
    u_values_map: &HashMap<ObjectId, f64>,
) -> BpResult<()> {
    let u = match u_values_map.get(immo.id()) {
        Some(u) => u,
        None => return Err("Tried to write U for immo without U".into()),
    };
    col.update_one(
        doc! {"_id": immo.id().clone()},
        doc! {"$set": {"U_Germany": u}},
        None,
    )?;
    Ok(())
}

fn write_U_values(
    immos: &[Immo],
    n_workers: usize,
    collection: &str,
    u_values_map: &HashMap<ObjectId, f64>,
) -> BpResult<()> {
    log::info!("Writing values...");
    let col = database::get_collection(None, collection)?;

    database::par_op(immos, n_workers, |immo| {
        write_U_value(&col, immo, u_values_map).unwrap_or_else(|error| {
            log::error!(
                "Could not write U value for {}, \nError: {}",
                immo.id(),
                error
            )
        });
    });

    log::info!("Writing values... DONE");
    Ok(())
}

fn necessary_input_projection() -> Document {
    doc! {
        "marktwert": true,
        "plane_location": true,
        "kurzgutachten.objektangabenWohnflaeche": true,
    }
}

#[derive(StructOpt, Debug)]
struct Cli {
    #[structopt(default_value = "cleaned_80", long)]
    collection: String,
}

fn main() -> BpResult<()> {
    logging::init_logging();

    let args = Cli::from_args();
    log::debug!("CLI Arguments {:?}", args);

    log::info!("loading data...");
    let projection = Some(necessary_input_projection());
    let immos = database::read_reasonable_immos_from_collection(
        None,
        None,
        projection,
        Some(&args.collection),
    )?;
    log::info!("loading data... DONE\n\t(n={})", immos.len());

    log::info!("Calculating U...");
    let u_values_map =
        calculate_U_for_immos(&SqmPriceDissimilarity, &immos.iter().collect::<Vec<_>>());
    log::info!("Calculating U... DONE");

    write_U_values(&immos[..], 50, &args.collection, &u_values_map)?;
    Ok(())
}
