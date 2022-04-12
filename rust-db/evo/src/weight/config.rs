//! This module contains configurations for controlling the behavior of genetic operations.
use crate::weight::{TrivialWeightWrapper, Weight};
use derive_builder::Builder;
use evaluators::kpi::MinimizeMapeInvertedOrdKpiOutput;
use genevo::algorithm::EvaluatedPopulation;
use genevo::genetic::{Children, Offspring, Parents};
use genevo::operator::prelude::{ElitistReinserter, MaximizeSelector};
use genevo::operator::{CrossoverOp, GeneticOperator, MutationOp, ReinsertionOp, SelectionOp};
use genevo::prelude::{Fitness, FitnessFunction, Genotype, Rng};
use serde::Deserialize;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

/// Gives parameters that may influence mutation. **See [Weight::mutate]**.
#[derive(Clone, Debug, Builder)]
pub struct MutationConfig {
    /// At which current iteration is the evo?
    #[builder(default = "0")]
    pub iteration: u64,
    /// How likely should an individual mutation be to occur?
    pub mutation_rate: f64,
    /// What should the std deviation of a numeric mutation be?
    pub std_deviation: f64,
    /// How likely should an enum switch its variant?
    pub type_switch_probability: f64,
}

impl MutationConfig {
    /// increase the current time for the mutation operator by one
    pub fn increase_iteration(&mut self) {
        self.iteration += 1;
    }

    /// reset the current time to 0
    pub fn reset_iteration(&mut self) {
        self.iteration = 0;
    }
}

impl GeneticOperator for MutationConfig {
    fn name() -> String {
        "Mutation with config".into()
    }
}

/// This mutation operation will simply call the [Weight::mutate] function with the config
impl<W> MutationOp<TrivialWeightWrapper<W>> for MutationConfig
where
    W: Weight + Sync + Send + PartialEq,
{
    fn mutate<R>(&self, mut genome: TrivialWeightWrapper<W>, rng: &mut R) -> TrivialWeightWrapper<W>
    where
        R: Rng + Sized,
    {
        genome.mutate(rng, self);
        genome
    }
}

/// a mutation operator as a wrapper around an arc.
/// Usable to set the current time while not having ownership of the instance
#[derive(Clone, Debug)]
pub struct SharedMutationConfig(pub Arc<RwLock<MutationConfig>>);

impl From<MutationConfig> for SharedMutationConfig {
    fn from(config: MutationConfig) -> Self {
        Self(Arc::new(RwLock::new(config)))
    }
}

impl GeneticOperator for SharedMutationConfig {
    fn name() -> String {
        MutationConfig::name()
    }
}

impl<G> MutationOp<G> for SharedMutationConfig
where
    G: Genotype,
    MutationConfig: MutationOp<G>,
{
    fn mutate<R>(&self, genome: G, rng: &mut R) -> G
    where
        R: Rng + Sized,
    {
        self.0.read().unwrap().mutate(genome, rng)
    }
}

/// An enum which wraps a CrossoverConfig in case there should be no crossover
#[derive(Clone, Debug)]
pub enum Crossover {
    /// Dont do crossover (always return calling parent)
    NoCrossover,
    /// Do crossover according to CrossoverConfig
    Crossover(CrossoverConfig),
}

impl GeneticOperator for Crossover {
    fn name() -> String {
        "Crossover Wrapper".into()
    }
}

impl<W> CrossoverOp<TrivialWeightWrapper<W>> for Crossover
where
    W: Weight + Sync + Send + PartialEq,
{
    fn crossover<R>(
        &self,
        parents: Parents<TrivialWeightWrapper<W>>,
        rng: &mut R,
    ) -> Children<TrivialWeightWrapper<W>>
    where
        R: Rng + Sized,
    {
        match self {
            Self::NoCrossover => parents,
            Self::Crossover(crossover_config) => crossover_config.crossover(parents, rng),
        }
    }
}

/// Gives parameters that might influence crossover.
#[derive(Clone, Debug, Default)]
pub struct CrossoverConfig {}

impl GeneticOperator for CrossoverConfig {
    fn name() -> String {
        "Crossover with config".into()
    }
}

impl<W> CrossoverOp<TrivialWeightWrapper<W>> for CrossoverConfig
where
    W: Weight + Sync + Send + PartialEq,
{
    fn crossover<R>(
        &self,
        parents: Parents<TrivialWeightWrapper<W>>,
        rng: &mut R,
    ) -> Children<TrivialWeightWrapper<W>>
    where
        R: Rng + Sized,
    {
        parents
            .iter()
            .map(|mother| {
                let father_idx = rng.gen_range(0, parents.len());
                mother.crossover(&parents[father_idx], rng, self)
            })
            .collect()
    }
}

/// An enum which wraps a ReinsertionConfig in case there should be no reinsertion
#[derive(Clone, Debug)]
pub enum Reinsertion<W, P, F>
where
    W: Genotype,
    P: Fitness,
    F: FitnessFunction<W, P>,
{
    /// Dont do reinsertion (always all offspring are chosen)
    NoReinsertion,
    /// Do reinsertion according to ReinsertionConfig
    Reinsertion(ReinsertionConfig<W, P, F>),
}

impl<W, P, F> GeneticOperator for Reinsertion<W, P, F>
where
    W: Genotype,
    P: Fitness,
    F: FitnessFunction<W, P>,
{
    fn name() -> String {
        "Reinsertion Wrapper".into()
    }
}

impl<W, P, F> ReinsertionOp<W, P> for Reinsertion<W, P, F>
where
    W: Weight + Sync + Send + PartialEq + Genotype + Sized,
    P: Fitness,
    F: FitnessFunction<W, P> + Copy,
{
    fn combine<R>(
        &self,
        offspring: &mut Offspring<W>,
        population: &EvaluatedPopulation<W, P>,
        rng: &mut R,
    ) -> Vec<W>
    where
        R: Rng + Sized,
    {
        match self {
            Self::NoReinsertion => offspring.to_vec(),
            Self::Reinsertion(reinsertion_config) => {
                reinsertion_config.combine(offspring, population, rng)
            }
        }
    }
}
/// orchestrates the reinsertion for the evo
#[derive(Clone, Debug)]
pub struct ReinsertionConfig<W, P, F>
where
    W: Genotype,
    P: Fitness,
    F: FitnessFunction<W, P>,
{
    /// Which reinserter should perform the actual reinsertion?
    pub reinserter: ElitistReinserter<W, P, F>,
    _w: PhantomData<W>,
    _p: PhantomData<P>,
}

impl<W, P, F> ReinsertionConfig<W, P, F>
where
    W: Genotype,
    P: Fitness,
    F: FitnessFunction<W, P>,
{
    /// Creates a new reinsertion config. Wraps [ElitistReinserted::new].
    pub fn new(fitness_function: F, replace_ratio: f64) -> Self {
        Self {
            reinserter: ElitistReinserter::new(fitness_function, false, replace_ratio),
            _w: Default::default(),
            _p: Default::default(),
        }
    }
}

impl<W, P, F> GeneticOperator for ReinsertionConfig<W, P, F>
where
    W: Genotype,
    P: Fitness,
    F: FitnessFunction<W, P>,
{
    fn name() -> String {
        "Reinsertion Config".into()
    }
}

impl<W, P, F> ReinsertionOp<W, P> for ReinsertionConfig<W, P, F>
where
    W: Weight + Sync + Send + PartialEq + Genotype + Sized,
    P: Fitness,
    F: FitnessFunction<W, P> + Copy,
{
    fn combine<R>(
        &self,
        offspring: &mut Offspring<W>,
        population: &EvaluatedPopulation<W, P>,
        rng: &mut R,
    ) -> Vec<W>
    where
        R: Rng + Sized,
    {
        self.reinserter.combine(offspring, population, rng)
    }
}

/// Implements fitness sharing as detailed in the [Paper by Martin](https://hpi.de/friedrich/people/martin-krejca.html?tx_extbibsonomycsl_publicationlist%5Bcontroller%5D=Document&tx_extbibsonomycsl_publicationlist%5BfileName%5D=EscapingLocalOptimaWithDiversityMechanismsAndCrossover.pdf&tx_extbibsonomycsl_publicationlist%5BintraHash%5D=39c4a62b79798d197d8e5e49eb7ec75b&tx_extbibsonomycsl_publicationlist%5BuserName%5D=puma-friedrich&cHash=fab77311e17b49d328bf2b486c531cf8)
#[derive(Clone, Debug, Builder, Deserialize)]
pub struct FitnessSharingSelectionConfig {
    /// If two indivudal's distance is below this threshold they will be punished in fitness sharing (sigma in Martin's paper)
    distance_threshold: f64,
    /// Configures how strong the effect of fitness sharing should be (alpha in Martin's paper)
    exponent: f64,
}

impl FitnessSharingSelectionConfig {
    /// The unadjusted fitness should be divided by that value.
    /// A small quotient is good (the individual is similar to few others)
    /// A large quotient is bad (the individual is similar to many others)
    fn fitness_sharing_quotient<W>(&self, individual: &W, population: &[W]) -> f64
    where
        W: Weight,
    {
        population
            .iter()
            .map(|other| {
                let goodness = (individual.normalized_distance(other) / self.distance_threshold)
                    .powf(self.exponent);
                (1.0 - goodness).max(0.0)
            })
            .sum::<f64>()
    }

    fn apply_fitness_sharing_quotient(
        fitness: &mut MinimizeMapeInvertedOrdKpiOutput,
        quotient: f64,
    ) {
        // We multiply because a small quotient is good.
        fitness.0.mean_absolute_percentage_error *= quotient
    }

    fn adjust_fitness<W>(
        &self,
        population: &EvaluatedPopulation<W, MinimizeMapeInvertedOrdKpiOutput>,
    ) -> EvaluatedPopulation<W, MinimizeMapeInvertedOrdKpiOutput>
    where
        W: Weight + Sync + Send + PartialEq + Genotype + Sized,
    {
        let individuals: Vec<_> = population.individuals().to_vec();
        let mut fitness_values: Vec<_> = population.fitness_values().to_vec();

        for idx in 0..individuals.len() {
            let quotient = self.fitness_sharing_quotient(&individuals[idx], &individuals[..]);
            Self::apply_fitness_sharing_quotient(&mut fitness_values[idx], quotient)
        }

        let best_fitness = *fitness_values.iter().max().unwrap();
        let worst_fitness = *fitness_values.iter().min().unwrap();
        let average_fitness = MinimizeMapeInvertedOrdKpiOutput::average(&fitness_values);

        EvaluatedPopulation::new(
            Rc::new(individuals),
            fitness_values,
            best_fitness,
            worst_fitness,
            average_fitness,
        )
    }
}

#[derive(Clone, Debug)]
/// A genetic operator performing selection. Could do fitness sharing via a [FitnessSharingConfig].
/// The [Self::inner] selection operations will be applied after fitness sharing
/// if fitness sharing is set. Otherwise [Self::inner] is used.
pub struct Selection {
    /// Which [SelectionOp] to use for the actual selection
    pub inner: MaximizeSelector,
    /// If set, the [FitnessSharingConfig] will be used to adjust the fitness of the population
    /// before the actual selection.
    pub fitness_sharing: Option<FitnessSharingSelectionConfig>,
}

impl GeneticOperator for Selection {
    fn name() -> String {
        "selection (maximze or fitness sharing)".into()
    }
}

impl<W> SelectionOp<W, MinimizeMapeInvertedOrdKpiOutput> for Selection
where
    W: Weight + Sync + Send + PartialEq + Genotype + Sized,
{
    fn select_from<R>(
        &self,
        population: &EvaluatedPopulation<W, MinimizeMapeInvertedOrdKpiOutput>,
        rng: &mut R,
    ) -> Vec<Parents<W>>
    where
        R: Rng + Sized,
    {
        let modified_fitness_population = if let Some(fitness_sharing) = &self.fitness_sharing {
            fitness_sharing.adjust_fitness(population)
        } else {
            population.clone()
        };

        self.inner.select_from(&modified_fitness_population, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use itertools::Itertools;
    use proptest::prelude::*;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    struct ConstantDistanceWeight(f64);

    impl Weight for ConstantDistanceWeight {
        type SubOutput = f64;

        fn normalized_distance(&self, other: &Self) -> f64 {
            if std::ptr::eq(self, other) {
                // the same object should give 0.0 as distance
                0.0
            } else {
                self.0
            }
        }

        fn add(&self, _other: &Self::SubOutput) -> Self {
            unimplemented!()
        }

        fn sub(&self, _other: &Self) -> Self::SubOutput {
            unimplemented!()
        }
    }

    #[test]
    fn fitness_sharing_handcrafted_test() {
        let config = FitnessSharingSelectionConfig {
            distance_threshold: 0.5,
            exponent: 2.0,
        };
        let individuals = [ConstantDistanceWeight(0.1); 3];
        assert_approx_eq!(
            config.fitness_sharing_quotient(&individuals[0], &individuals),
            1.0 + 2.0 * (1.0 - (0.1f64 / 0.5).powf(2.0))
        );
    }

    proptest! {
        #[test]
        fn smaller_distance_yields_higher_quotient(
            distance_a in 0.0f64..1.0f64,
            distance_b in 0.0f64..1.0f64,
            exponent in 0.01f64..100f64,
            distance_threshold in 0.0f64..1.0f64,
            num_individuals in 1usize..256usize,
        ) {

            let (smaller_dist, larger_dist) = vec![distance_a, distance_b]
                .into_iter()
                .minmax()
                .into_option()
                .unwrap();

            let smaller_dist_individuals = vec![ConstantDistanceWeight(smaller_dist); num_individuals];
            let larger_dist_individuals = vec![ConstantDistanceWeight(larger_dist); num_individuals];

            let config = FitnessSharingSelectionConfig {
                distance_threshold, exponent
            };

            prop_assert!(
                config.fitness_sharing_quotient(&smaller_dist_individuals[0], &smaller_dist_individuals) >=
                config.fitness_sharing_quotient(&larger_dist_individuals[0], &larger_dist_individuals)
            );
        }

        #[test]
        fn far_individuals_do_not_contribute(
            exponent in 0.01f64..100f64,
            distance_threshold in 0.0f64..0.99f64,
            num_individuals in 1usize..256usize,
        ) {
            let config = FitnessSharingSelectionConfig {
                distance_threshold, exponent
            };

            // distance_threshold < distance < 1.0
            let distance = (distance_threshold + 1.0) / 2.0;
            let individuals = vec![ConstantDistanceWeight(distance); num_individuals];

            prop_assert!((config.fitness_sharing_quotient(&individuals[0], &individuals) - 1.0).abs() < 1e-6);
        }
    }
}
