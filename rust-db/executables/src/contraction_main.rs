#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
use chrono::Utc;
use common::util::path_or_relative_to_project_root;
use common::{database, logging, BpResult, CostFunction, Immo};
use cost_functions::{BorrowCostFunction, DissimilarityCostFunction, LorCostFunction};
use dissimilarities::SqmPriceDissimilarity;
use executables::filter_from_path_or_panic;
use log::Level::Debug;
use mongodb::bson::doc;
use partition::make_one_stable;
use partition::Partition;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Cli {
    #[structopt(default_value = "300", long)]
    epsilon: f64,
    #[structopt(long = "one-stable")]
    one_stable: bool,
    #[structopt(long = "lor")]
    lor_partition: bool,
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output_path: Option<PathBuf>,
    #[structopt(short = "q", long = "query", parse(from_os_str))]
    query_path: Option<PathBuf>,
    #[structopt(short = "c", default_value = "cleaned_80", long)]
    collection: String,
}

fn output_path(args: &Cli) -> PathBuf {
    let mut file_name: String = format!("{}-", Utc::now().to_rfc3339());
    if let Some(query_path) = &args.query_path {
        file_name.push_str(
            query_path
                .file_name()
                .expect("Query path is not a valid file")
                .to_str()
                .expect("Query path is not valid unicode"),
        )
    } else {
        file_name.push_str("germany.json")
    }
    path_or_relative_to_project_root(
        args.output_path.as_ref(),
        &format!("data/contraction/{}", &file_name),
    )
}

fn main() -> BpResult<()> {
    logging::init_logging();

    let args = Cli::from_args();

    log::info!("CLI Arguments {:?}", args);

    log::info!("start loading data");

    let epsilon = match args.lor_partition {
        false => args.epsilon,
        true => 10e3,
    };

    let mut projection = doc! {
        "marktwert": true,
        "plane_location": true,
        "kurzgutachten.objektangabenWohnflaeche": true,
        "U_Germany": true,
    };

    if args.lor_partition {
        projection.insert("berlin_plr_id", true);
    }

    let filter = if args.lor_partition {
        Some(doc! {
            "$or": [
                { "kreis": "Berlin, Stadt" },
                { "ort": "Berlin" }
            ],
            "location.1": { "$lt": 52.75, "$gt": 52.0 },
            "location.0": { "$lt": 14.0, "$gt": 12.5 }
        })
    } else {
        args.query_path.as_ref().map(filter_from_path_or_panic)
    };

    log::debug!("projection {:?}", projection);
    log::debug!("filter {:?}", filter);
    let all_immos: Vec<Immo> = database::read_reasonable_immos_from_collection(
        None,
        filter.clone(),
        Some(projection),
        Some(&args.collection),
    )?;
    let immos: Vec<&Immo> = all_immos
        .iter()
        .filter(|immo| immo.sqm_price().is_some())
        .collect();
    log::info!("loaded data (n = {})", immos.len());

    let cost_function_box: Box<dyn CostFunction + Sync + Send> = if args.lor_partition {
        Box::new(LorCostFunction)
    } else {
        Box::new(DissimilarityCostFunction::with_immos(
            SqmPriceDissimilarity,
            immos.iter().copied(),
        ))
    };
    let cost_function: BorrowCostFunction<dyn CostFunction + Sync + Send, _> =
        BorrowCostFunction::new(cost_function_box);

    if immos.is_empty() {
        panic!("Your query returned no immos! Can't cluster.")
    } else if immos.len() < 1000 {
        log::warn!(
            "Your query returned less than 1000 immos. Your results might not be informative."
        )
    }

    let mut part = Partition::with_immos(epsilon, cost_function, immos)?;

    log::info!("created partition");
    part.contraction();
    log::info!("contracted partition ... DONE");
    if args.one_stable {
        log::info!("started make_one_stable");
        make_one_stable(&mut part);
        log::info!("make_one_stable ... DONE");
    }

    if log::log_enabled!(Debug) {
        let mut costs = Vec::new();
        for block_idx in part.iter_blocks() {
            costs.push(part.cost_for_block(block_idx))
        }
        log::debug!(
            "Costs\n\tTotal Costs: {}\n\tBlock Costs: {:#?}",
            costs.iter().sum::<f64>(),
            costs
        );
    }
    let output_path = output_path(&args);
    part.create_json_file_with_metadata(&output_path, Some(args.collection), filter)?;
    log::info!("Wrote result to {:?}", &output_path);
    Ok(())
}
