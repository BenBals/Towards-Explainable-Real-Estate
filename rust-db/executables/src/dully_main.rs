#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]
use chrono::{NaiveDate, Utc};
use common::immo::necessary_filters;
use common::{
    database, logging,
    util::{path_or_relative_to_project_root, write_serializable_to_json},
    BpResult, ErasedDissimilarity, Immo,
};
use dissimilarities::expert::{is_expert_dissimilarity_valid_for, UnfilteredExpertDissimilarity};
use dissimilarities::filtered::{FilterConfig, FilteredDissimilarity};
use dissimilarities::{
    CosineDissimilarity, DistanceDissimilarity, ScalingLpVectorDissimilarity,
    ScalingLpVectorDissimilarityConfig,
};
use evaluators::*;
use executables::filter_from_path_or_panic;
use itertools::*;
use mongodb::bson::doc;
use normalizers::{BrazilNormalizer, HpiNormalizer, RegressionNormalizer};
use predictions::{CleaningStrategy, Driver, ErasedPredictor, Evaluator};
use predictors::weighted_average::{NeighborSelectionStrategy, NeighborsForAverageStrategy};
use predictors::{weighted_average::WeightedAveragePredictorOptionsBuilder, *};
use serde::Deserialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Cli {
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output_path: Option<PathBuf>,
    #[structopt(short = "q", long = "query", parse(from_os_str))]
    query_path: Option<PathBuf>,
    #[structopt(short = "c", default_value = "cleaned_80", long)]
    collection: String,
    #[structopt(
        short = "e",
        default_value = "kpi",
        long,
        help = "Options: kpi, kreis_kpi"
    )]
    evaluator: String,
    #[structopt(long = "config", parse(from_os_str))]
    config_path: Option<PathBuf>,
    #[structopt(
        long,
        help = "Split train/validation data at a given date. Enter in YYYY-MM-DD format."
    )]
    split_at_date: Option<String>,
    #[structopt(
        long,
        help = "Remove all training properties valued on or after the given date. Enter in YYYY-MM-DD format."
    )]
    training_before: Option<String>,
    #[structopt(default_value = "hpi", long, help = "Options: regression, hpi, brazil")]
    normalizer: String,
    #[structopt(
        long,
        help = "Write predictions to mongodb. Argument: key to use in the database."
    )]
    write_to_mongo: Option<String>,
    #[structopt(long)]
    limit: Option<i64>,
    #[structopt(
        long,
        help = "Clean \"all\", \"testing_only\", \"none\". Defaults to all."
    )]
    cleaning: Option<String>,
    #[structopt(
        long,
        help = "Use weighted median instead of weighted average prediction. Default: false"
    )]
    median: bool,
    #[structopt(
        long,
        help = "Split into training and test set deterministically based on hashing an object with following key"
    )]
    split_by_hash_key: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
enum DissimilarityConfig {
    ScalingLpVector(ScalingLpVectorDissimilarityConfig),
    // Since [ExpertDissimilarity] doe not depend on training, we can simply deserialize one right
    // away and don't need an intermediate config.
    Expert(UnfilteredExpertDissimilarity),
    UnweightedScalingLpVector(f64),
    Cosine,
    Dully,
}

/// Must always match config/dully/types.dhall
#[derive(Clone, Debug, Deserialize)]
pub struct DullyConfig {
    dissimilarity: DissimilarityConfig,
    neighbor_selection_strategy: NeighborSelectionStrategy,
    neighbors_for_average_strategy: NeighborsForAverageStrategy,
    filters: Option<FilterConfig>,
}

type PairEvaluator = fn(&Cli, Vec<(&Immo, &Immo)>) -> BpResult<()>;

impl Default for DullyConfig {
    fn default() -> Self {
        Self {
            dissimilarity: DissimilarityConfig::Dully,
            neighbor_selection_strategy: NeighborSelectionStrategy::Radius(10.0 * 1000.0),
            neighbors_for_average_strategy: NeighborsForAverageStrategy::All,
            filters: None,
        }
    }
}

impl DullyConfig {
    /// Config from dhall file
    pub fn from_file<P: AsRef<Path>>(path: P) -> BpResult<Self> {
        Ok(serde_dhall::from_file(path).parse::<DullyConfig>()?)
    }

    fn uses_expert_dissimilarity(&self) -> bool {
        matches!(self.dissimilarity, DissimilarityConfig::Expert(_))
    }

    fn construct_dissimilarity(&self) -> Box<dyn ErasedDissimilarity + Sync> {
        let unfiltered: Box<dyn ErasedDissimilarity + Sync> = match &self.dissimilarity {
            DissimilarityConfig::ScalingLpVector(dissimilarity_config) => Box::new(
                ScalingLpVectorDissimilarity::from_config(dissimilarity_config)
                    .expect("Couldn't construct ScalingLpVectorDissimilarity from Config"),
            ),
            DissimilarityConfig::UnweightedScalingLpVector(exponent) => {
                Box::new(ScalingLpVectorDissimilarity::with_exponent(*exponent))
            }
            DissimilarityConfig::Expert(expert_dissimilarity) => Box::new(*expert_dissimilarity),
            DissimilarityConfig::Cosine => Box::new(CosineDissimilarity::default()),
            DissimilarityConfig::Dully => Box::new(DistanceDissimilarity::default()),
        };

        match &self.filters {
            Some(filter_config) => Box::new(FilteredDissimilarity::from_config(
                filter_config,
                unfiltered,
            )),
            None => unfiltered,
        }
    }

    fn construct_predictor(&self, args: &Cli) -> Box<dyn ErasedPredictor> {
        let dissimilarity = self.construct_dissimilarity();
        Box::new(WeightedAveragePredictor::with_options(
            WeightedAveragePredictorOptionsBuilder::default()
                .dissimilarity(dissimilarity)
                .neighbor_selection_strategy(self.neighbor_selection_strategy)
                .neighbors_for_average_strategy(self.neighbors_for_average_strategy)
                .median(args.median)
                .build()
                .expect("Could not build WeightedAveragePredictor"),
        ))
    }
}

impl Cli {
    fn output_path(&self) -> PathBuf {
        path_or_relative_to_project_root(
            self.output_path.as_ref(),
            &format!("data/dully/{}.json", Utc::now().to_rfc3339()),
        )
    }

    fn read_config(&self) -> BpResult<DullyConfig> {
        if let Some(path) = self.config_path.as_ref() {
            DullyConfig::from_file(path)
        } else {
            Ok(DullyConfig::default())
        }
    }

    fn construct_normalized_predictor(
        &self,
        inner_predictor: Box<dyn ErasedPredictor>,
    ) -> BpResult<Box<dyn ErasedPredictor>> {
        match self.normalizer.as_ref() {
            "hpi" => Ok(Box::new(NormalizingPredictor::with(
                HpiNormalizer::new(),
                inner_predictor,
            ))),
            "regression" => Ok(Box::new(NormalizingPredictor::with(
                RegressionNormalizer::new(),
                inner_predictor,
            ))),
            "brazil" => Ok(Box::new(NormalizingPredictor::with(
                BrazilNormalizer::new(),
                inner_predictor,
            ))),
            "none" => Ok(inner_predictor),
            invalid_value => Err(format!(
                "Invalid normalizer {}. Valid options: \"hpi\", \"regression\".",
                invalid_value
            )
            .into()),
        }
    }

    fn evaluate_kpi(&self, pair_refs: Vec<(&Immo, &Immo)>) -> BpResult<()> {
        let eval = KpiEvaluator::new();
        let output = eval.evaluate(pair_refs)?;
        log::info!("\n{}", output);

        let output_path = self.output_path();

        write_serializable_to_json(&output, &output_path)?;
        log::info!("Wrote evaluation to {:?}", &output_path);

        Ok(())
    }

    fn evaluate_kreis_kpi(&self, pair_refs: Vec<(&Immo, &Immo)>) -> BpResult<()> {
        let grouped = pair_refs
            .into_iter()
            .sorted_unstable_by_key(|(real, _)| &real.prefecture)
            .group_by(|(real, _)| &real.prefecture);
        let results: Vec<_> = grouped
            .into_iter()
            .map(|(key, immos)| {
                let collected: Vec<_> = immos.into_iter().collect();
                (
                    key.clone().unwrap_or_default(),
                    collected
                        .iter()
                        .filter_map(|(real, _)| real.prefecture.as_ref())
                        .next()
                        .cloned()
                        .unwrap_or_default(),
                    KpiEvaluator::new().evaluate(collected.iter().copied()),
                )
            })
            .collect();

        for tuple in &results {
            match tuple {
                (_, prefecture, Err(err)) => {
                    log::warn!("Could not evaluate \"{}\" due to {}", prefecture, err)
                }
                (_, _, Ok(_)) => {}
            }
        }

        let map: Vec<_> = results
            .into_iter()
            .filter_map(|(key, prefecture, result_output)| {
                result_output
                    .ok()
                    .map(|output| (prefecture.clone(), prefecture, output))
            })
            .collect();

        let output_path = self.output_path();
        write_serializable_to_json(&map, &output_path)?;
        log::info!("Wrote evaluation to {:?}", &output_path);

        Ok(())
    }

    fn evaluators_map() -> HashMap<&'static str, PairEvaluator> {
        let mut map = HashMap::<&'static str, PairEvaluator>::new();
        map.insert("kpi", Cli::evaluate_kpi);
        map.insert("kreis_kpi", Cli::evaluate_kreis_kpi);
        map
    }

    fn evaluate_pairs(&self, identity_output: Vec<(Immo, Immo)>) -> BpResult<()> {
        let pair_refs = identity_output.iter().map(|(a, b)| (a, b)).collect();
        (Cli::evaluators_map()[&*self.evaluator])(self, pair_refs)?;

        if let Some(key) = self.write_to_mongo.as_ref() {
            database::write_predictions_to_database(&self.collection, key, &identity_output[..]);
        }

        Ok(())
    }

    fn should_write_to_mongo(&self) -> bool {
        self.write_to_mongo.is_some()
    }

    fn validate(&self) {
        let evaluate_map = Cli::evaluators_map();
        if evaluate_map.get(&*self.evaluator).is_none() {
            panic!(
                "Unknown evaluator. Only {:?} are supported values.",
                evaluate_map.keys().collect::<Vec<_>>()
            );
        }
    }

    fn cleaning_strategy(&self) -> CleaningStrategy {
        match self.cleaning.as_ref().map(|string| string.as_ref()) {
            None | Some("all") => CleaningStrategy::All,
            Some("testing_only") => CleaningStrategy::TestingOnly,
            Some("none") => CleaningStrategy::None,
            _ => {
                panic!(
                    "Invalid --cleaning value {}. Choose from \"all\", \"testing_only\", \"none\".",
                    self.cleaning.as_ref().unwrap()
                )
            }
        }
    }
}

fn main() -> BpResult<()> {
    logging::init_logging();

    let args = Cli::from_args();
    args.validate();

    log::info!("CLI Arguments: {:?}", args);
    let config = args.read_config()?;
    log::info!("Using config {:#?}", config);

    let inner_eval = IdentityEvaluator::new();
    let predictor = config.construct_predictor(&args);
    let mut normalizing_predictor = args.construct_normalized_predictor(predictor)?;

    let mut driver = Driver::with(&mut normalizing_predictor, &inner_eval);

    driver.set_cleaning(args.cleaning_strategy());

    if args.should_write_to_mongo() {
        driver.predict_all(true);
    }

    let training_before = args.training_before.as_ref().map(|date_str| {
        NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .expect("Could not parse date input to `--training-before`")
    });

    if let Some(date_str) = args.split_at_date.as_ref() {
        match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            Ok(date) => driver.split_at_dates(date, training_before),
            Err(err) => {
                panic!(
                    "Could not parse date input to `--split-at-date`. Error: {}",
                    err
                );
            }
        }
    }

    if let Some(seed) = args.split_by_hash_key.as_ref() {
        if args.split_at_date.is_some() {
            panic!("You cannot set --split-at-date and --split-by-hash-key at the same time.")
        }

        driver.split_by_hash_key(seed.clone());
    }

    log::info!("start loading data");
    let projection = Some(doc! {
        "Acxiom.centrality": true,
        "Acxiom.regioTyp": true,
        "grundstuecksgroesseInQuadratmetern": true,
        "kreis_canonic": true,
        "kurzgutachten.drittverwendungsfaehigkeit": true,
        "kurzgutachten.objektangabenAnzahlGaragen": true,
        "kurzgutachten.objektangabenAnzahlZimmer": true,
        "kurzgutachten.objektangabenAusstattungNote": true,
        "kurzgutachten.objektangabenBaujahr": true,
        "kurzgutachten.objektangabenUnterkellerungsgrad": true,
        "kurzgutachten.objektangabenVerwendung": true,
        "kurzgutachten.objektangabenWohnflaeche": true,
        "kurzgutachten.objektangabenZustand": true,
        "kurzgutachten.vermietbarkeit": true,
        "kurzgutachten.verwertbarkeit": true,
        "macro_scores": true,
        "marktwert": true,
        "objektuebersicht.erbbaurechtBesteht": true,
        "objektunterart": true,
        "plane_location": true,
        "prefecture": true,
        "restnutzungsdauer": true,
        "balcony_area": true,
        "walk_distance1": true,
        "land_toshi": true,
        "house_kaisuu": true,
        "convenience_distance": true,
        "school_ele_distance": true,
        "school_jun_distance": true,
        "distance_parking": true,
        "scores": true,
        "wertermittlungsstichtag": true,
    });
    let filter = args.query_path.as_ref().map(filter_from_path_or_panic);

    let immos = if matches!(
        args.cleaning_strategy(),
        CleaningStrategy::None | CleaningStrategy::TestingOnly
    ) {
        log::warn!(
            "Not performing basic outlier removal. This can lead to very inaccurate predictions."
        );
        database::read_from_collection(
            args.limit,
            Some(doc! {
                "$and": [
                    filter.unwrap_or(doc! {}),
                    necessary_filters()
                ]
            }),
            projection,
            Some(&args.collection),
        )?
    } else {
        database::read_reasonable_immos_from_collection(
            args.limit,
            filter,
            projection,
            Some(&args.collection),
        )?
    };
    log::info!("Loading data...DONE\n\tn={}", immos.len());

    log::info!("Filtering data...");
    let mut filtered_immos = immos
        .iter()
        .filter(|immo| {
            if config.uses_expert_dissimilarity() {
                is_expert_dissimilarity_valid_for(immo)
            } else {
                true
            }
        })
        .collect::<Vec<_>>();
    log::info!(
        "Filtering data... DONE\n\tFiltered out {} immos",
        immos.len() - filtered_immos.len()
    );

    let identity_output: Vec<(Immo, Immo)> = driver.drive(filtered_immos.as_mut_slice())?;

    args.evaluate_pairs(identity_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dissimilarities::ScalingLpVectorDissimilarity;

    #[test]
    #[allow(clippy::float_cmp)]
    fn import_from_dhall() {
        let imported = DullyConfig::from_file("../../config/dully/default.dhall").unwrap();
        let default_dissimilarity = ScalingLpVectorDissimilarity::default();
        match imported.dissimilarity {
            DissimilarityConfig::Expert(_) => {
                panic!("Did not expect expert dissimilarity as default")
            }
            DissimilarityConfig::Dully => {
                panic!("Did not expect dully dissimilarity as default")
            }
            DissimilarityConfig::Cosine => {
                panic!("Did not expect cosine dissimilarity as default")
            }
            DissimilarityConfig::UnweightedScalingLpVector(_) => {
                panic!("Did not expect unweighted scaling lp dissimilarity as default")
            }
            DissimilarityConfig::ScalingLpVector(inner) => {
                let imported_dissimilarity =
                    ScalingLpVectorDissimilarity::from_config(&inner).unwrap();
                assert_eq!(default_dissimilarity, imported_dissimilarity);
                assert_eq!(
                    imported.neighbor_selection_strategy,
                    NeighborSelectionStrategy::Radius(1e4)
                );
            }
        }
    }
}
