//! the main for the local search

#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "strict", deny(clippy::all))]

use crate::weight::{
    config::{MutationConfig, SharedMutationConfig},
    Scalable, TrivialWeightWrapper, Weight,
};
use genevo::genetic::Fitness;
use genevo::operator::MutationOp;
use genevo::prelude::FitnessFunction;
use rand::{thread_rng, Rng};
use std::ops::Sub;

/// a config used by the local search for looking up parameters
#[derive(Clone, Debug)]
pub struct LocalConfig {
    /// How many iterations does it take until the algo stops since it is stuck
    pub restart_threshold: u64,
    /// how great are the steps of the local search. In range [0,1]
    pub factor: f64,
    /// the mutation operator
    pub mutation_operator: MutationConfig,
    /// should simulated annealing be used?
    pub simulated_annealing: bool,
}

impl LocalConfig {
    /// creates a new LocalConfig
    pub fn new(
        restart_threshold: u64,
        factor: f64,
        mutation_operator: MutationConfig,
        simulated_annealing: bool,
    ) -> Self {
        Self {
            restart_threshold,
            factor,
            mutation_operator,
            simulated_annealing,
        }
    }
}

/// the output struct of the local search
#[derive(Clone, Debug)]
pub struct LocalResult<W>
where
    W: Weight,
{
    /// the final weight
    pub weight: TrivialWeightWrapper<W>,
    /// after how many iterations was the weight returned
    pub generations_run: u64,
}

/// runs a local search until it is stuck a given number of times or it has run sufficiently many generations
pub fn run_local_search<W, F, P>(
    fitness_function: F,
    initial_population: TrivialWeightWrapper<W>,
    config: &LocalConfig,
    generations: u64,
) -> LocalResult<W>
where
    W: Weight + Sync + Send + PartialEq + Default,
    P: Fitness + Sub<P, Output = P> + Into<f64>,
    F: FitnessFunction<TrivialWeightWrapper<W>, P>,
{
    log::info!("initializing local search");

    // let mutation = config.mutation_operator.clone();
    let mutation_arc = SharedMutationConfig::from(config.mutation_operator.clone());

    let mut current_solution = initial_population;
    let mut current_fitness = fitness_function.fitness_of(&current_solution);
    let mut old_solution;
    let mut old_fitness;
    let mut difference_solution;
    let mut fitness_diff: f64;

    let mut num_stuck = 0;
    let mut iteration = 0;

    log::info!("initialization done");

    let final_weight = loop {
        old_solution = current_solution;
        old_fitness = current_fitness;
        iteration += 1;
        mutation_arc.0.write().unwrap().increase_iteration();

        if iteration > generations {
            log::info!("Local search stops since all generations are done");
            break old_solution;
        }
        current_solution = mutation_arc.mutate(old_solution.clone(), &mut thread_rng());
        current_fitness = fitness_function.fitness_of(&current_solution);

        log::info!(
            "step: generation: {:?} / {}, current fitness: {:?}, best fitness: {:?},\
            best Weight: {:?}",
            iteration,
            generations,
            current_fitness,
            old_fitness,
            old_solution,
        );

        fitness_diff = (old_fitness.clone() - current_fitness.clone()).into();
        if fitness_diff > 0.0
            || (config.simulated_annealing
                && thread_rng().gen_bool((fitness_diff * iteration as f64).exp().clamp(0.0, 1.0)))
        {
            // new is better
            log::info!("current weight is BETTER than old one");
            difference_solution = current_solution.sub(&old_solution).scale(config.factor);
        } else {
            // doesnt get better
            log::info!("current weight is WORSE than old one");
            difference_solution = current_solution.sub(&old_solution).scale(-config.factor);
        }
        current_solution = old_solution.add(&difference_solution);
        current_fitness = fitness_function.fitness_of(&current_solution);
        fitness_diff = (old_fitness.clone() - current_fitness.clone()).into();

        // check if we did something stupid
        if fitness_diff > 0.0
            || (config.simulated_annealing
                && thread_rng().gen_bool((fitness_diff * iteration as f64).exp().clamp(0.0, 1.0)))
        {
            // did nothing stupid
            log::info!("Improved fitness");
            num_stuck = 0;
        } else {
            // did something stupid
            log::info!("Could not improve fitness");
            num_stuck += 1;
            if num_stuck > config.restart_threshold {
                log::info!("Local search stops since it is stuck");
                break current_solution;
            }
            current_solution = old_solution;
            current_fitness = old_fitness;
        }
    };
    log::info!("running local search ... DONE");
    LocalResult {
        weight: final_weight,
        generations_run: iteration - 1,
    }
}
