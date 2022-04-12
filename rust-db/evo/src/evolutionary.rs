//! the main for the evolutionary algorithm

use crate::fitness_functions::{RelativeDeviationFitnessFunction, WeightedPredictor};
use crate::weight::{
    config::{
        CrossoverConfig, FitnessSharingSelectionConfig, MutationConfig, ReinsertionConfig,
        Selection, SharedMutationConfig,
    },
    TrivialWeightWrapper, Weight,
};
use chrono::NaiveDate;
use common::Immo;
use evaluators::kpi::MinimizeMapeInvertedOrdKpiOutput;
use evaluators::KpiOutput;
use genevo::algorithm::{BestSolution, EvaluatedPopulation};
use genevo::operator::prelude::MaximizeSelector;
use genevo::prelude::{
    genetic_algorithm, simulate, Fitness, FitnessFunction, GenerationLimit, Population, SimResult,
    Simulation, SimulationBuilder,
};
use predictions::{split_by_hash_key, split_data_at_date, TRAINING_RATIO};
use rand::{thread_rng, Rng};
use std::convert::TryInto;
use std::fmt::Debug;

/// the result of the evo algorithm
#[derive(Clone, Debug)]
pub struct EvoResult<W>
where
    W: Weight,
{
    /// The resulting weight
    pub weight: TrivialWeightWrapper<W>,
    /// How many generations did the evo run
    pub generations_run: u64,
}

/// a struct for lookup of parameters for the run_evo method
#[derive(Clone, Debug)]
pub struct EvoConfig<'i, W, P, F: 'i>
where
    W: Weight + Sync + Send + PartialEq + Sized,
    P: Fitness,
    &'i F: FitnessFunction<TrivialWeightWrapper<W>, P>,
{
    /// see Config in evo_main.
    pub selection_ratio: f64,
    /// see Config in evo_main.
    pub individuals_per_parent: u64,
    /// see Config in evo_main.
    pub restart_threshold: u64,
    /// see Config in evo_main.
    pub simulated_annealing: bool,
    /// see Config in evo_main.
    pub mutation_config: MutationConfig,
    /// see Config in evo_main.
    pub crossover_config: CrossoverConfig,
    /// see Config in evo_main.
    pub reinsertion_config: ReinsertionConfig<TrivialWeightWrapper<W>, P, &'i F>,
    /// see Config in evo_main.
    pub fitness_sharing: Option<FitnessSharingSelectionConfig>,
    /// evaluate the best individual of each generation the test set
    pub evaluate_all_generations_on_test_set: bool,
}

impl<'i, W, P, F> EvoConfig<'i, W, P, F>
where
    W: Weight + Sync + Send + PartialEq + Sized,
    P: Fitness,
    &'i F: FitnessFunction<TrivialWeightWrapper<W>, P>,
{
    /// creates a new EvoConfig
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        selection_ratio: f64,
        individuals_per_parent: u64,
        restart_threshold: u64,
        simulated_annealing: bool,
        mutation_config: MutationConfig,
        crossover_config: CrossoverConfig,
        reinsertion_config: ReinsertionConfig<TrivialWeightWrapper<W>, P, &'i F>,
        fitness_sharing: Option<FitnessSharingSelectionConfig>,
        evaluate_all_generations_on_test_set: bool,
    ) -> Self {
        Self {
            selection_ratio,
            individuals_per_parent,
            restart_threshold,
            simulated_annealing,
            mutation_config,
            crossover_config,
            reinsertion_config,
            fitness_sharing,
            evaluate_all_generations_on_test_set,
        }
    }

    fn as_selection_config(&self) -> Selection {
        Selection {
            fitness_sharing: self.fitness_sharing.clone(),
            inner: MaximizeSelector::new(
                self.selection_ratio,
                self.individuals_per_parent.try_into().unwrap(),
            ),
        }
    }
}

fn total_diversity<W, F>(
    evaluated_population: &EvaluatedPopulation<TrivialWeightWrapper<W>, F>,
) -> f64
where
    W: Weight + PartialEq + Sync + Send + PartialEq,
    F: Fitness,
{
    let indis = evaluated_population.individuals().to_vec();

    let mut sum = 0.0;
    for i1 in indis.iter() {
        for i2 in indis.iter() {
            let dist = i1.normalized_distance(i2);
            if dist.is_nan() {
                panic!("Diversity is NaN between\n{:#?}\nand\n{:#?}", i1, i2);
            }
            sum += dist;
        }
    }
    sum
}

/// runs an evo until it is stuck a given number of times or it has run sufficiently many generations
pub fn run_evolutionary<'i, W, F: 'i, P>(
    fitness_function: &'i F,
    initial_population: Population<TrivialWeightWrapper<W>>,
    config: &EvoConfig<'i, W, MinimizeMapeInvertedOrdKpiOutput, F>,
    generations: u64,
    all_data: Vec<Immo>,
    test_predictor: P,
    split_date: NaiveDate,
) -> EvoResult<W>
where
    W: Weight + Sync + Send + PartialEq + Default,
    P: WeightedPredictor<W> + Clone,
    F: Debug,
    &'i F: FitnessFunction<TrivialWeightWrapper<W>, MinimizeMapeInvertedOrdKpiOutput> + Sync + Send,
{
    log::info!("building genetic algorithm and initial population...");

    let mutation_arc = SharedMutationConfig::from(config.mutation_config.clone());

    let builder = genetic_algorithm()
        .with_evaluation(fitness_function)
        .with_selection(config.as_selection_config())
        .with_crossover(config.crossover_config.clone())
        .with_mutation(mutation_arc.clone())
        .with_reinsertion(config.reinsertion_config.clone());

    log::info!(
        "\n\nusing genetic algorithm structure: \n\t{:#?}\n",
        builder
    );

    let mut simulation = simulate(
        builder
            .clone()
            .with_initial_population(initial_population)
            .build(),
    )
    .until(GenerationLimit::new(generations))
    .build();

    log::info!("building genetic algorithm and initial population... DONE");
    log::info!("running evo... ");

    let mut best_solution: Option<
        BestSolution<TrivialWeightWrapper<W>, MinimizeMapeInvertedOrdKpiOutput>,
    > = None;
    let mut num_stuck = 0;

    let result = loop {
        mutation_arc.0.write().unwrap().increase_iteration();
        let (step, stop_reason_option) = match simulation.step() {
            Ok(SimResult::Intermediate(step)) => (step, None),
            Ok(SimResult::Final(step, _, _, stop_reason)) => (step, Some(stop_reason)),
            Err(error) => panic!("Simulation failed: {}", error),
        };

        let evaluated_population = step.result.evaluated_population;

        log::info!("----> Diversity {}", total_diversity(&evaluated_population));

        let current_solution = step.result.best_solution;

        log::info!(
            "step: generation: {:?} / {}, average_fitness: {:?}, best fitness: {:?},\
            duration: {:?}\n\tbest Weight: {:?}",
            step.iteration,
            generations,
            evaluated_population.average_fitness(),
            current_solution.solution.fitness,
            step.duration,
            current_solution.solution.genome,
        );

        if config.evaluate_all_generations_on_test_set {
            let test_result = test_weight(
                current_solution.solution.genome.0.clone(),
                test_predictor.clone(),
                all_data.clone(),
                split_date,
                None,
                None,
            );
            log::info!(
                "\tResults on test set: {}",
                test_result.mean_absolute_percentage_error
            );
        }

        match &best_solution {
            None => best_solution = Some(current_solution),
            Some(solution) => {
                let fitness_diff: f64 =
                    (solution.solution.fitness - current_solution.solution.fitness).into();
                if fitness_diff > 0.0
                    || (config.simulated_annealing
                        && thread_rng().gen_bool((fitness_diff * step.iteration as f64).exp()))
                {
                    // improvement
                    best_solution = Some(current_solution);
                    num_stuck = 0;
                } else {
                    // no improvement
                    num_stuck += 1;
                    log::info!("Evo stuck {} times", num_stuck);
                    if num_stuck >= config.restart_threshold {
                        // restart simulation (rebuild simulation population)
                        break EvoResult {
                            weight: solution.solution.genome.clone(),
                            generations_run: step.iteration,
                        };
                    }
                }
            }
        }

        if let Some(stop_reason) = stop_reason_option {
            log::info!("Stop reason: {}", stop_reason);
            break EvoResult {
                weight: best_solution.unwrap().solution.genome,
                generations_run: step.iteration,
            };
        }
    };
    result
}

/// Calculate the [KpiOutput] for a given weight using given data and predictor
pub fn test_weight<W, P>(
    weight: W,
    predictor: P,
    data: Vec<Immo>,
    split_date: NaiveDate,
    split_by_hash_key_seed: Option<String>,
    write_to_mongo: Option<(String, String)>,
) -> KpiOutput
where
    W: Weight + Sync + Send + PartialEq,
    P: WeightedPredictor<W>,
{
    let mut immo_refs: Vec<_> = data.iter().collect();
    let (training_data, test_data) = if let Some(seed) = split_by_hash_key_seed {
        split_by_hash_key(&mut immo_refs, TRAINING_RATIO, &seed)
    } else {
        split_data_at_date(&mut immo_refs, split_date)
    };
    let training_vec: Vec<_> = training_data.iter().copied().cloned().collect();
    let test_vec: Vec<_> = test_data.iter().copied().cloned().collect();
    let test_vec_len = test_vec.len();

    log::info!("test vec len in test_weight {}", test_vec_len);

    let fitness_function = RelativeDeviationFitnessFunction::new(
        training_vec,
        test_vec,
        test_vec_len,
        predictor,
        write_to_mongo,
    )
    .unwrap();

    (&fitness_function)
        .fitness_of(&TrivialWeightWrapper(weight))
        .0
}
