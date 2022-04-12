#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
use chrono::Utc;
use common::{
    database, logging,
    util::{path_or_relative_to_project_root, write_serializable_to_json},
    BpResult, CostFunction, Dissimilarity, Immo, Normalizer, Trainable,
};
use cost_functions::{
    BorrowCostFunction, CappedSpmPriceCostFunction, DissimilarityCostFunction, LorCostFunction,
};
use dissimilarities::{ScalingLpVectorDissimilarity, SqmPriceDissimilarity};
use evaluators::*;
use executables::filter_from_path_or_panic;
use mongodb::bson::{doc, oid::ObjectId};
use normalizers::{HandcraftedBerlin, IdentityNormalizer, NormalizationPipeline};
use predictions::Driver;
use predictors::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, Clone)]
struct OmniscientSqmPriceDissimilarity {
    price_lookup: HashMap<ObjectId, f64>,
}

impl OmniscientSqmPriceDissimilarity {
    #[allow(dead_code)]
    fn with_immos<N: Normalizer>(immos: &[Immo], normalizer: &mut N) -> BpResult<Self> {
        let mut immos = immos.to_owned();
        let mut price_lookup = HashMap::new();
        normalizer.train(&immos)?;
        normalizer.normalize(immos.iter_mut());

        for immo in immos {
            price_lookup.insert(immo.id().clone(), immo.sqm_price().unwrap());
        }

        Ok(Self { price_lookup })
    }
}

impl Trainable for OmniscientSqmPriceDissimilarity {}

impl Dissimilarity for OmniscientSqmPriceDissimilarity {
    fn dissimilarity(&self, a: &Immo, b: &Immo) -> f64 {
        (self.price_lookup[a.id()] - self.price_lookup[b.id()]).abs()
    }
}
#[derive(StructOpt, Debug)]
struct Cli {
    #[structopt(long = "lor")]
    lor_prediction: bool,
    #[structopt(long = "no-time-norm")]
    no_normalized_time: bool,
    #[structopt(default_value = "300", long)]
    epsilon: f64,
    #[structopt(default_value = "0.1", long = "capped-deviation")]
    capped_deviation: f64,
    #[structopt(short = "q", long = "query", parse(from_os_str))]
    query_path: Option<PathBuf>,
    #[structopt(short = "c", default_value = "cleaned_80", long)]
    collection: String,
    #[structopt(long = "one-stable")]
    one_stable: bool,
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output_path: Option<PathBuf>,
}

impl Cli {
    fn output_path(&self) -> PathBuf {
        path_or_relative_to_project_root(
            self.output_path.as_ref(),
            &format!("data/prediction/{}.json", Utc::now().to_rfc3339()),
        )
    }
}

fn main() -> BpResult<()> {
    logging::init_logging();

    let args = Cli::from_args();

    log::info!("CLI Arguments {:?}", args);

    log::info!("start loading data");

    let epsilon = match args.lor_prediction {
        false => args.epsilon,
        true => 10e3,
    };

    let mut projection = doc! {
        "marktwert": true,
        "plane_location": true,
        "kurzgutachten.objektangabenWohnflaeche": true,
        "wertermittlungsstichtag": true,
        "kurzgutachten.objektangabenBaujahr": true,
        "kurzgutachten.objektangabenAnzahlGaragen": true,
        "kurzgutachten.objektangabenZustand": true,
        "kurzgutachten.objektangabenAusstattungNote": true,
        "grundstuecksgroesseInQuadratmetern": true,
        "scores.ALL": true,
        "Acxiom.regioTyp": true,
        "Acxiom.centrality": true,
    };

    if args.lor_prediction {
        projection.insert("berlin_plr_id", true);
    }

    let filter = match args.lor_prediction || !args.no_normalized_time {
        false => args.query_path.as_ref().map(filter_from_path_or_panic),
        true => Some(doc! {
            "$or": [
            { "kreis": "Berlin, Stadt" },
            { "ort": "Berlin" }
            ],
            "location.1": { "$lt": 52.75, "$gt": 52.0 },
            "location.0": { "$lt": 14.0, "$gt": 12.5 }
        }),
    };

    let immos = database::read_reasonable_immos(None, filter, Some(projection))?;
    log::info!("loading data... DONE\n\t(n={})", immos.len());

    let cost_function_box: Arc<dyn CostFunction + Sync + Send> = match args.lor_prediction {
        false => Arc::new(CappedSpmPriceCostFunction::new(
            DissimilarityCostFunction::with_immos(SqmPriceDissimilarity, immos.iter()),
            args.capped_deviation,
        )),
        true => Arc::new(LorCostFunction),
    };
    let cost_function: BorrowCostFunction<dyn CostFunction + Sync + Send, _> =
        BorrowCostFunction::new(cost_function_box);

    // weights are taken from DNN variance-map
    let mut dissimilarity_function = ScalingLpVectorDissimilarity::with_weights([
        62359.0, 78372.0, 14657.0, 7229.0, 166004.0, 17626.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    dissimilarity_function.train(&immos)?;
    log::info!("{:?}", dissimilarity_function);

    let inner_predictor = SelectBlockPredictor::new(
        cost_function,
        dissimilarity_function,
        epsilon,
        args.one_stable,
    );

    let normalizer = match args.no_normalized_time {
        true => NormalizationPipeline::with_one(IdentityNormalizer),
        false => NormalizationPipeline::with_two(HandcraftedBerlin, IdentityNormalizer),
    };

    let mut predictor = NormalizingPredictor::with(normalizer, inner_predictor);

    let evaluator = RelativeSqmPriceDeviationEvaluator::new();

    let mut driver = Driver::with(&mut predictor, &evaluator);
    let output = driver.drive(immos.iter().collect::<Vec<_>>().as_mut_slice())?;

    log::info!("DONE Single Block Predictor");

    let output_path = args.output_path();

    write_serializable_to_json(&output, &output_path)?;
    log::info!("Wrote evaluation to {:?}", &output_path);

    Ok(())
}
