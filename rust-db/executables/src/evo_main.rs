#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]

use chrono::{NaiveDate, Utc};
use common::immo::necessary_filters;
use common::util::{path_or_relative_to_project_root, write_serializable_to_json};
use common::{database, logging, BpResult, Immo};
use dissimilarities::filtered::FilteredDissimilarityBuilder;
use dissimilarities::{ExpertDissimilarity, ScalingLpVectorDissimilarity};
use evo::evolutionary::{run_evolutionary, test_weight, EvoConfig, EvoResult};
use evo::fitness_functions::{RelativeDeviationFitnessFunction, WeightedPredictor};
use evo::local_search::{run_local_search, LocalConfig, LocalResult};
use evo::weight::{
    config::{
        Crossover, CrossoverConfig, FitnessSharingSelectionConfig, MutationConfig,
        MutationConfigBuilder, Reinsertion, ReinsertionConfig,
    },
    expert_weight::ExpertWeight,
    lp_wap_weight::LpWapWeight,
    EvoPredictor, TopLevelWeight, TrivialWeightWrapper, Weight,
};
use executables::filter_from_path_or_panic;
use genevo::prelude::{build_population, Fitness, FitnessFunction, Population};
use mongodb::bson::doc;
use predictions::{split_by_date_percentage, split_by_hash_key, split_data_at_date};
use predictors::weighted_average::{
    NeighborsForAverageStrategy, WeightedAveragePredictor, WeightedAveragePredictorOptionsBuilder,
};
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Cli {
    #[structopt(long = "config", parse(from_os_str))]
    config_path: Option<PathBuf>,
    #[structopt(short = "c", default_value = "cleaned_80", long)]
    collection: String,
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output_path: Option<PathBuf>,
    #[structopt(
        long,
        help = "Split train/validation data at a given date. Enter in YYYY-MM-DD format."
    )]
    split_at_date: Option<String>,
    #[structopt(
        long,
        help = "Split into training and test set deterministically based on hashing an object with following key"
    )]
    split_by_hash_key: Option<String>,
    #[structopt(short = "wo", long = "weight-output", parse(from_os_str))]
    weight_output_path: Option<PathBuf>,
    #[structopt(short = "q", long = "query", parse(from_os_str))]
    query_path: Option<PathBuf>,
    #[structopt(long = "expert")]
    expert: bool,
    #[structopt(short = "l")]
    limit: Option<i64>,
    #[structopt(long = "local-search")]
    local_search: bool,
    #[structopt(short = "s", long = "seed", parse(from_os_str))]
    seed_path: Option<PathBuf>,
    #[structopt(
        long,
        help = "Evaluate the best individual of each generation the test set"
    )]
    evaluate_all_generations_on_test_set: bool,
    #[structopt(
        long,
        help = "Use weighted median instead of weighted average prediction. Default: false"
    )]
    median: bool,
    #[structopt(long, help = "Don't perform data cleaning")]
    unclean: bool,
    #[structopt(
        long,
        help = "Write predictions to mongodb. Argument: key to use in the database."
    )]
    write_to_mongo: Option<String>,
}

impl Cli {
    fn output_path(&self) -> PathBuf {
        path_or_relative_to_project_root(
            self.output_path.as_ref(),
            &format!("data/evo/{}.json", Utc::now().to_rfc3339()),
        )
    }

    fn weight_output_path(&self) -> PathBuf {
        path_or_relative_to_project_root(
            self.weight_output_path.as_ref(),
            &format!("data/evo/{}_weight.json", Utc::now().to_rfc3339()),
        )
    }

    fn config_path(&self) -> PathBuf {
        path_or_relative_to_project_root(self.config_path.as_ref(), "config/evo/default.dhall")
    }

    fn predictor(&self) -> EvoPredictor {
        if self.expert {
            EvoPredictor::Expert(WeightedAveragePredictor::new(ExpertDissimilarity::new()))
        } else {
            let dissimilarity = FilteredDissimilarityBuilder::default()
                .inner(ScalingLpVectorDissimilarity::default())
                .build()
                .unwrap();

            let options = WeightedAveragePredictorOptionsBuilder::default()
                .dissimilarity(dissimilarity)
                .median(self.median) // This is copied in lp_wap_weight.rs
                .radius(1.0) // filler
                .neighbors_for_average_strategy(NeighborsForAverageStrategy::All)
                .build()
                .unwrap();

            EvoPredictor::Lp(WeightedAveragePredictor::with_options(options))
        }
    }

    fn default_seed_weight(&self, config: &Config) -> TopLevelWeight {
        if self.expert {
            if config.ignore_filters
                || config.strict_radius_limit.is_some()
                || config.k_most_similar_limit.is_some()
            {
                panic!("Set ignore_filters ore strict_radius_limit in config. These only apply with the lp wap weights");
            }
            TopLevelWeight::Expert(ExpertWeight::default())
        } else {
            TopLevelWeight::Lp(LpWapWeight::default())
        }
    }

    fn seed_weight(&self, config: &Config) -> BpResult<TopLevelWeight> {
        log::info!("seedpath {:?}", self.seed_path);
        let weight_result = match &self.seed_path {
            None => {
                log::info!("use default seed weight");
                Ok(self.default_seed_weight(config))
            }
            Some(path) => {
                log::info!("use seed weight from path");
                let file_handle = File::open(path)?;
                let tlw = serde_json::from_reader(file_handle)?;
                Ok(tlw)
            }
        };

        weight_result.map(|weight| config.modify_lp_wap_weight_using_config(weight))
    }

    fn split_date(&self) -> NaiveDate {
        self.split_at_date
            .as_ref()
            .map(|date_str| {
                NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                    .expect("Could not parse date input to `--split-at-date`.")
            })
            .unwrap_or_else(|| NaiveDate::from_ymd(2020, 1, 1))
    }
}

/// A Config which wraps a config for local search and for evo
#[derive(Clone, Debug)]
pub enum TopLevelConfig<'i, W, P, F>
where
    W: Weight + Sync + Send + PartialEq + Sized,
    P: Fitness,
    &'i F: FitnessFunction<TrivialWeightWrapper<W>, P> + Copy,
{
    LocalSearch(LocalConfig),
    Evo(EvoConfig<'i, W, P, F>),
}

/// See config/evo/types.dhall for documentation on the fields
#[derive(Clone, Debug, Deserialize)]
struct Config {
    train_test_split_ratio: f64,
    num_generations: u64,
    population_size: usize,
    batch_size: usize,
    individuals_per_parent: usize,
    mutation_rate: f64,
    type_switch_probability: f64,
    replace_ratio: f64,
    std_deviation: f64,
    selection_ratio: f64,
    multistart_threshold: u64,
    simulated_annealing: bool,
    strict_radius_limit: Option<f64>,
    ignore_filters: bool,
    disable_normalization: bool,
    k_most_similar_limit: Option<u64>,
    fix_exponent: Option<f64>,
    fitness_sharing: Option<FitnessSharingSelectionConfig>,
    local_search_factor: f64,
    best_result_as_seed: bool,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> BpResult<Self> {
        let conf: Config = serde_dhall::from_file(path).parse()?;
        assert!(conf.train_test_split_ratio > 0f64 && conf.train_test_split_ratio < 1f64);
        Ok(conf)
    }

    /// builds a mutation config from saved attributes
    fn as_mutation_config(&self) -> MutationConfig {
        MutationConfigBuilder::default()
            .std_deviation(self.std_deviation)
            .mutation_rate(self.mutation_rate)
            .type_switch_probability(self.type_switch_probability)
            .build()
            .unwrap()
    }

    /// builds a crossover config from saved attributes
    fn as_crossover_config(&self) -> Crossover {
        if self.simulated_annealing {
            Crossover::NoCrossover
        } else {
            Crossover::Crossover(CrossoverConfig {})
        }
    }

    fn modify_lp_wap_weight_using_config(&self, mut weight: TopLevelWeight) -> TopLevelWeight {
        match &mut weight {
            TopLevelWeight::Lp(inner_weight) => {
                inner_weight.ignore_filters = self.ignore_filters;
                inner_weight.strict_radius_limit = self.strict_radius_limit;
                if let Some(limit) = self.k_most_similar_limit {
                    inner_weight.set_k_most_similar_upper_bound(limit as i64)
                }
                if let Some(value) = self.fix_exponent {
                    inner_weight.fix_exponent(value);
                }
            }
            TopLevelWeight::Expert(_) => {}
        }

        weight
    }

    /// builds a reinsertion config from saved attributes
    fn as_reinsertion_config<W, P, F>(
        &self,
        fitness_function: F,
    ) -> Reinsertion<TrivialWeightWrapper<W>, P, F>
    where
        W: Weight + Sync + Send + PartialEq + Sized,
        P: Fitness,
        F: FitnessFunction<TrivialWeightWrapper<W>, P> + Copy,
    {
        if self.simulated_annealing {
            Reinsertion::NoReinsertion
        } else {
            Reinsertion::Reinsertion(ReinsertionConfig::new(fitness_function, self.replace_ratio))
        }
    }

    /// creates an evo config from the saved data and the given fitness function
    fn as_evo_config<'i, W, P, F>(
        &self,
        fitness_function: &'i F,
        args: &Cli,
    ) -> TopLevelConfig<'i, W, P, F>
    where
        W: Weight + Sync + Send + PartialEq + Sized,
        P: Fitness,
        &'i F: FitnessFunction<TrivialWeightWrapper<W>, P> + Copy,
    {
        let crossover_conf = match self.as_crossover_config() {
            Crossover::Crossover(crossover_conf) => crossover_conf,
            _ => panic!("Crossover config cannot be set for evo."),
        };
        let reinsertion_conf = match self.as_reinsertion_config(fitness_function) {
            Reinsertion::Reinsertion(reinserion_conf) => reinserion_conf,
            _ => panic!("Reinsertion config cannot be set for evo."),
        };
        TopLevelConfig::Evo(EvoConfig::new(
            self.selection_ratio,
            self.individuals_per_parent as u64,
            self.multistart_threshold,
            self.simulated_annealing,
            self.as_mutation_config(),
            crossover_conf,
            reinsertion_conf,
            self.fitness_sharing.clone(),
            args.evaluate_all_generations_on_test_set,
        ))
    }

    /// creates a local config from the saved data. It should be given a correct fitness function
    /// for type references but it is not used
    fn as_local_config<'i, W, P, F>(&self, _fitness_function: &'i F) -> TopLevelConfig<'i, W, P, F>
    where
        W: Weight + Sync + Send + PartialEq + Sized,
        P: Fitness,
        &'i F: FitnessFunction<TrivialWeightWrapper<W>, P> + Copy,
    {
        TopLevelConfig::LocalSearch(LocalConfig::new(
            self.multistart_threshold,
            self.local_search_factor,
            self.as_mutation_config(),
            self.simulated_annealing,
        ))
    }
}

/// Builds a population completely determined by the weights given as parameter
pub fn population_by_weights<W>(weights: Vec<W>) -> Population<TrivialWeightWrapper<W>>
where
    W: Weight + PartialEq + Send + Sync,
{
    Population::with_individuals(
        weights
            .iter()
            .map(|weight| TrivialWeightWrapper(weight.clone()))
            .collect(),
    )
}

/// Build an initial population from a given "seed" weight.
/// This is done by treating the seed as a [GenomeBuilder] that can generate weights of its own type.
pub fn random_initial_population<W>(
    population_size: usize,
    seed: W,
) -> Population<TrivialWeightWrapper<W>>
where
    W: Weight + Sync + Send + PartialEq + Default,
{
    build_population()
        .with_genome_builder(TrivialWeightWrapper(seed))
        .of_size(population_size)
        .uniform_at_random()
}

/// runs a local search calling the run_local_search method
/// the methods handles restarts itself, i.e. run_local_search returns if the algo
/// is stuck and needs a restart with all necessary parameter
fn run_local<W, P>(
    training_data: Vec<Immo>,
    validation_data: Vec<Immo>,
    config: &Config,
    predictor: P,
    seed_weight: W,
) -> W
where
    W: Weight + Sync + Send + PartialEq + Default,
    P: WeightedPredictor<W> + Sync,
{
    let generations = config.num_generations;
    let mut current_generations = 0;
    let population_size = 1;
    let mut population: TrivialWeightWrapper<W> = if config.best_result_as_seed {
        population_by_weights(vec![seed_weight.clone()]).individuals()[0].clone()
    } else {
        random_initial_population(population_size, seed_weight.clone()).individuals()[0].clone()
    };

    log::info!("creating fitness function...");
    let fitness_function = RelativeDeviationFitnessFunction::new(
        training_data,
        validation_data,
        config.batch_size,
        predictor,
        None,
    )
    .unwrap_or_else(|error| {
        panic!("Creation of Fitness Function failed: {}", error);
    });

    let mut local_config = match config.as_local_config(&fitness_function) {
        TopLevelConfig::LocalSearch(local_conf) => local_conf,
        _ => panic!("Local config is no local config"),
    };

    // dummy initialization
    let mut best_local_weight = LocalResult {
        weight: population.clone(),
        generations_run: 0,
    };
    let mut best_local_fitness = (&fitness_function).fitness_of(&best_local_weight.weight);

    loop {
        let local_result: LocalResult<W> = run_local_search(
            &fitness_function,
            population,
            &local_config,
            generations - current_generations,
        );
        let local_fitness = (&fitness_function).fitness_of(&local_result.weight);
        current_generations += local_result.generations_run;

        if local_fitness > best_local_fitness {
            best_local_weight = local_result.clone();
            best_local_fitness = local_fitness;
        }

        if current_generations >= generations {
            log::info!(
                "Local Search done after {} generations with weight {:?}",
                current_generations,
                local_result.weight.0
            );
            break;
        } else {
            log::info!(
                "Restart Local Search after {} generations. In total: {}/{} generations done.",
                local_result.generations_run,
                current_generations,
                generations
            );
            local_config.mutation_operator.reset_iteration();
            // we can think about changing this so the new population is chosen from last state (i.e. local_result.weight.0)
            population = random_initial_population(population_size, seed_weight.clone())
                .individuals()[0]
                .clone();
        }
    }

    best_local_weight.weight.0
}

/// runs the evo calling the run_evolutionary function
/// Restarts are handled here since run_evo returns in case of getting stuck
/// Please note that the local search has a different method but not the simulated annealing
fn run_evo<W, P>(
    training_data: Vec<Immo>,
    validation_data: Vec<Immo>,
    all_data: Vec<Immo>,
    config: &Config,
    predictor: P,
    seed_weight: W,
    args: &Cli,
) -> W
where
    W: Weight + Sync + Send + PartialEq + Default,
    P: WeightedPredictor<W> + Sync + Clone,
{
    log::info!("creating fitness function...");
    let generations = config.num_generations;
    let mut current_generations = 0;
    let population_size = config.population_size;
    let mut population = if config.best_result_as_seed {
        population_by_weights(vec![seed_weight; population_size])
    } else {
        random_initial_population(population_size, seed_weight)
    };

    let fitness_function = RelativeDeviationFitnessFunction::new(
        training_data,
        validation_data,
        config.batch_size,
        predictor.clone(),
        None,
    )
    .unwrap_or_else(|error| {
        panic!("Creation of Fitness Function failed: {}", error);
    });

    let mut evo_config = match config.as_evo_config(&fitness_function, args) {
        TopLevelConfig::Evo(evo_conf) => evo_conf,
        _ => panic!("Evo config is no evo config"),
    };

    // dummy initialization
    let mut best_evo_weight = EvoResult {
        weight: population.individuals()[0].clone(),
        generations_run: 0,
    };
    let mut best_evo_fitness = (&fitness_function).fitness_of(&best_evo_weight.weight);

    loop {
        let evo_result: EvoResult<W> = run_evolutionary(
            &fitness_function,
            population,
            &evo_config,
            generations - current_generations,
            all_data.clone(),
            predictor.clone(),
            args.split_date(),
        );
        let evo_fitness = (&fitness_function).fitness_of(&evo_result.weight);
        current_generations += evo_result.generations_run;

        if evo_fitness > best_evo_fitness {
            best_evo_weight = evo_result.clone();
            best_evo_fitness = evo_fitness;
        }

        if current_generations >= generations {
            log::info!(
                "Evo done after {} generations with weight {:?}",
                current_generations,
                evo_result.weight.0
            );
            break;
        } else {
            log::info!(
                "Restart Evo after {} generations. In total: {}/{} generations done.",
                evo_result.generations_run,
                current_generations,
                generations
            );
            evo_config.mutation_config.reset_iteration();
            // we can think about changing this so the new population is chosen from the inital seed
            population = random_initial_population(population_size, evo_result.weight.0);
        }
    }

    best_evo_weight.weight.0
}

fn main() -> BpResult<()> {
    logging::init_logging();

    let args = Cli::from_args();
    log::info!("{:?}", args);
    let config = Config::from_file(args.config_path())?;
    log::info!("{:?}", config);
    log::info!("start loading data");

    let mut projection = doc! {
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
    };

    if !config.disable_normalization {
        projection.insert("AGS_0", true);
    }

    let filter = args.query_path.as_ref().map(filter_from_path_or_panic);
    let immos = if args.unclean {
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
            Some(projection),
            Some(&args.collection),
        )?
    } else {
        database::read_reasonable_immos_from_collection(
            args.limit,
            filter,
            Some(projection),
            Some(&args.collection),
        )?
    };

    log::info!("loaded data (n={})", immos.len());

    let mut immo_refs: Vec<_> = immos.iter().collect();

    let (training_data, validation_data, test_data) =
        if let Some(seed) = args.split_by_hash_key.as_ref() {
            let (trainable_data, test_data) =
                split_by_hash_key(&mut immo_refs, config.train_test_split_ratio, seed);

            let (training_data, validation_data) =
                split_by_hash_key(trainable_data, config.train_test_split_ratio, seed);

            (training_data, validation_data, test_data)
        } else {
            let (trainable_data, test_data) = split_data_at_date(&mut immo_refs, args.split_date());

            let (training_data, validation_data) =
                split_by_date_percentage(trainable_data, config.train_test_split_ratio);

            (training_data, validation_data, test_data)
        };

    log::info!(
        "Splitting stats:\n\ttraining = {}\n\tvalidation={}\n\ttest={}",
        training_data.len(),
        validation_data.len(),
        test_data.len()
    );

    let training_vec: Vec<_> = training_data.iter().copied().cloned().collect();
    let validation_vec: Vec<_> = validation_data.iter().copied().cloned().collect();

    let best_weight = match args.local_search {
        false => run_evo(
            training_vec,
            validation_vec,
            immos.clone(),
            &config,
            args.predictor(),
            args.seed_weight(&config)?,
            &args,
        ),
        true => run_local(
            training_vec,
            validation_vec,
            &config,
            args.predictor(),
            args.seed_weight(&config)?,
        ),
    };

    log::info!("==========BEST WEIGHTS==========\n{:#?}", best_weight);

    log::info!("evaluating evo results...");
    log::info!("Immo len before start of test_weight {}", immos.len());
    let output = test_weight(
        best_weight.clone(),
        args.predictor(),
        immos,
        args.split_date(),
        args.split_by_hash_key.clone(),
        Some(args.collection.clone()).zip(args.write_to_mongo.clone()),
    );
    let output_weight = best_weight;
    log::info!("evaluating evo results... DONE");
    log::info!("{}", output);

    let output_path = args.output_path();
    let weight_output_path = args.weight_output_path();

    write_serializable_to_json(&output, &output_path)?;
    write_serializable_to_json(&output_weight, &weight_output_path)?;
    log::info!("Wrote evaluation to {:?}", &output_path);

    Ok(())
}
