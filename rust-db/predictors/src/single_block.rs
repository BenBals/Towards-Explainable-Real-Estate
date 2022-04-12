use std::collections::HashSet;

use algorithms::sweepline::for_every_close_point_do;
use common::{BpResult, CostFunction, Dissimilarity, Immo, Keyed, Pointlike, Trainable};
use partition::{make_one_stable, BlockEntryIdx, BlockIdx, Partition};
use predictions::Predictor;
use typed_index_collections::TiVec;

const EPSILON: f64 = 300f64;

/// This predictor will cluster all training data into a [Partition] and then try to assign every
/// testing immo to one of the blocks. It will then predict the avg sqm price of that block for that
/// immo.
pub struct SelectBlockPredictor<C: CostFunction, D: Dissimilarity>(SelectBlockPredictorInner<C, D>);

enum SelectBlockPredictorInner<C: CostFunction, D: Dissimilarity> {
    Untrained {
        partition_cost_function: C,
        selection_dissimilarity: D,
        epsilon: f64,
        one_stable: bool,
    },
    Trained {
        /// The training data has been partitioned using the first dissimilarity.
        partition: Partition<Immo, C>,
        /// We store the avg sqm price for each block in the [Partition], so it has to be
        /// calculated less often.
        avg_sqm_prices: TiVec<BlockIdx, f64>,
        /// a dissimilarity used to select the most suitable Block for a given Immo
        selection_dissimilarity: D,
        /// epsilon from the partition creation, used for the near points
        epsilon: f64,
    },
}

#[derive(Debug, Clone, Copy)]
enum SweeplinePoint<'c, 'i, C: CostFunction> {
    PredictionPoint(usize, &'c Vec<&'i mut Immo>),
    PartitionPoint(BlockEntryIdx, &'c Partition<Immo, C>),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum SweeplinePointKey {
    PredictionPointKey(usize),
    PartitionPointKey(BlockEntryIdx),
}

impl<'c, 'i, C: CostFunction> SweeplinePoint<'c, 'i, C> {
    fn immo(&self) -> &Immo {
        match self {
            SweeplinePoint::PredictionPoint(idx, vec) => vec[*idx],
            SweeplinePoint::PartitionPoint(idx, part) => part[*idx].immo(),
        }
    }
}

impl<'c, 'i, C: CostFunction> Pointlike for SweeplinePoint<'c, 'i, C> {
    fn x(&self) -> u64 {
        self.immo().plane_location.unwrap().0 as u64
    }
    fn y(&self) -> u64 {
        self.immo().plane_location.unwrap().1 as u64
    }
}

impl<'c, 'i, C: CostFunction> Keyed for SweeplinePoint<'c, 'i, C> {
    type Key = SweeplinePointKey;
    fn key(&self) -> SweeplinePointKey {
        match self {
            SweeplinePoint::PredictionPoint(idx, _) => SweeplinePointKey::PredictionPointKey(*idx),
            SweeplinePoint::PartitionPoint(entry_idx, _) => {
                SweeplinePointKey::PartitionPointKey(*entry_idx)
            }
        }
    }
}

impl<C: CostFunction, D: Dissimilarity> SelectBlockPredictor<C, D> {
    /// Create a new, untrained predictor with the given dissimilarity metrics.
    /// The first is used for clustering training data, the second is used
    /// for selecting a block for each testing immo.
    pub fn new(
        partition_cost_function: C,
        selection_dissimilarity: D,
        epsilon: f64,
        one_stable: bool,
    ) -> Self {
        Self(SelectBlockPredictorInner::Untrained {
            partition_cost_function,
            selection_dissimilarity,
            epsilon,
            one_stable,
        })
    }

    /// Creates a new SelectBlockPredictor which uses the supplied function as cost function and the supplied epsilon as epsilon.
    pub fn with_functions(partition_cost_function: C, selection_dissimilarity: D) -> Self {
        Self(SelectBlockPredictorInner::Untrained {
            partition_cost_function,
            selection_dissimilarity,
            epsilon: EPSILON,
            one_stable: false,
        })
    }
}

impl<'i, C: CostFunction + Sync + Clone + Send, D: Dissimilarity + Clone> Trainable
    for SelectBlockPredictor<C, D>
{
    fn train<'j>(&mut self, training_data: impl IntoIterator<Item = &'j Immo>) -> BpResult<()> {
        let training_data: Vec<_> = training_data
            .into_iter()
            .filter(|immo| immo.sqm_price().is_some())
            .cloned()
            .collect();

        if training_data.is_empty() {
            return Err("No immo has sqm price set".into());
        }

        match &self.0 {
            SelectBlockPredictorInner::Trained { .. } => {
                Err("Tried to train already trained predictor.".into())
            }
            SelectBlockPredictorInner::Untrained {
                partition_cost_function,
                selection_dissimilarity,
                epsilon,
                one_stable,
            } => {
                log::info!("Training with {} immos...", training_data.len());
                let mut partition = Partition::with_immos(
                    *epsilon,
                    partition_cost_function.clone(),
                    training_data,
                )?;
                partition.contraction();

                if *one_stable {
                    log::info!("started make_one_stable");
                    make_one_stable(&mut partition);
                    log::info!("make_one_stable ... DONE");
                }

                let mut avg_sqm_prices = partition.create_block_data_vec(&0f64);
                for block_idx in partition.iter_blocks() {
                    let sum_sqm_prices: f64 = partition[block_idx]
                        .iter_entries()
                        .map(|entry_idx| partition[entry_idx].immo().sqm_price().unwrap())
                        .sum();
                    let count = partition[block_idx].iter_entries().count() as f64;
                    avg_sqm_prices[block_idx] = sum_sqm_prices / count;
                }

                *self = Self(SelectBlockPredictorInner::Trained {
                    partition,
                    avg_sqm_prices,
                    selection_dissimilarity: selection_dissimilarity.clone(),
                    epsilon: *epsilon,
                });
                log::info!("Training... DONE");

                Ok(())
            }
        }
    }
}

impl<'i, C, D> Predictor for SelectBlockPredictor<C, D>
where
    C: CostFunction + Sync + Clone + Send,
    D: Dissimilarity + Clone,
{
    fn predict<'j>(&self, validation_data: impl IntoIterator<Item = &'j mut Immo>) -> BpResult<()> {
        match &self.0 {
            SelectBlockPredictorInner::Untrained { .. } => Err("Was not trained".into()),
            SelectBlockPredictorInner::Trained {
                partition,
                avg_sqm_prices,
                selection_dissimilarity,
                epsilon,
                ..
            } => {
                let mut to_predict: Vec<_> = validation_data.into_iter().collect();

                if !to_predict
                    .iter()
                    .all(|immo| immo.plane_location.is_some() && immo.wohnflaeche.is_some())
                {
                    return Err("Tried to predict immo without plane_location".into());
                }

                let mut predictions: Vec<_> = (0..to_predict.len()).map(|_| None).collect();
                let all_points: Vec<_> = (0..to_predict.len())
                    .map(|idx| SweeplinePoint::PredictionPoint(idx, &to_predict))
                    .chain(
                        partition
                            .iter_entries()
                            .map(|entry_idx| SweeplinePoint::PartitionPoint(entry_idx, partition)),
                    )
                    .collect();

                let mut neighboring_blocks_count = vec![];

                for_every_close_point_do(&all_points, *epsilon as u64, |cur_point, neighbors| {
                    if let SweeplinePointKey::PredictionPointKey(idx) = cur_point {
                        let immo = &to_predict[idx];
                        let neighboring_blocks: HashSet<_> = neighbors
                            .iter()
                            .filter_map(|key| match key {
                                SweeplinePointKey::PartitionPointKey(entry_idx) => {
                                    Some(partition[*entry_idx].block())
                                }
                                SweeplinePointKey::PredictionPointKey(_) => None,
                            })
                            .collect();

                        let best_block = neighboring_blocks.iter().min_by_key(|&&block_idx| {
                            // small deviations are irrelevant
                            partition[block_idx]
                                .iter_entries()
                                .map(|entry_idx| {
                                    selection_dissimilarity
                                        .dissimilarity(immo, partition[entry_idx].immo())
                                        as u64
                                })
                                .min()
                                .expect("Tried to compute dissimilarity with empty block.")
                                as u64
                        });

                        neighboring_blocks_count.push(neighboring_blocks.len());
                        predictions[idx] = best_block.map(|&block_idx| avg_sqm_prices[block_idx]);
                    }
                });

                let mut without_prediction = 0;
                for (predicted_sqm_price_opt, immo) in predictions.iter().zip(to_predict.iter_mut())
                {
                    immo.marktwert = Some(match predicted_sqm_price_opt {
                        Some(predicted_sqm_price) => {
                            immo.wohnflaeche.unwrap() * predicted_sqm_price
                        }
                        None => {
                            without_prediction += 1;
                            0.0
                        }
                    });
                }

                if without_prediction > 0 {
                    log::warn!(
                        "{} of {} immos without prediction",
                        without_prediction,
                        to_predict.len()
                    );
                }

                neighboring_blocks_count.sort_unstable();
                log::info!(
                    "Minimum neighboring blocks: {}",
                    neighboring_blocks_count[0]
                );
                log::info!(
                    "Maximum neighboring blocks: {}",
                    neighboring_blocks_count[neighboring_blocks_count.len() - 1]
                );
                log::info!(
                    "Average neighboring blocks: {}",
                    neighboring_blocks_count.iter().sum::<usize>() / neighboring_blocks_count.len()
                );
                log::info!(
                    "Median neighboring blocks: {}",
                    neighboring_blocks_count[neighboring_blocks_count.len() / 2]
                );

                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::once;

    use super::*;
    use common::immo::ImmoBuilder;
    use common::Trainable;
    use cost_functions::ConstantCostFunction;
    use dissimilarities::LpVectorDissimilarity;
    use proptest::prelude::*;
    use test_helpers::*;

    #[test]
    fn new_predictor_is_untrained() {
        matches!(
            SelectBlockPredictor::with_functions(
                ConstantCostFunction::with(1.0),
                LpVectorDissimilarity::manhattan(),
            ),
            SelectBlockPredictor(SelectBlockPredictorInner::Untrained { .. })
        );
    }

    #[test]
    fn cant_train_without_immos() {
        let mut predictor = SelectBlockPredictor::with_functions(
            ConstantCostFunction::with(1.0),
            LpVectorDissimilarity::manhattan(),
        );

        assert!(predictor.train(vec![]).is_err());
    }

    #[test]
    fn unreachable_immos_gets_zero() {
        let train_immos = vec![ImmoBuilder::default()
            .marktwert(0.0)
            .wohnflaeche(0.0)
            .plane_location((0.0, 0.0))
            .build()
            .unwrap()];

        let mut test_immos = vec![ImmoBuilder::default()
            .marktwert(0.0)
            .wohnflaeche(0.0)
            .plane_location((0.0, EPSILON * 2.0))
            .build()
            .unwrap()];

        let mut predictor = SelectBlockPredictor::with_functions(
            ConstantCostFunction::with(1.0),
            LpVectorDissimilarity::manhattan(),
        );
        let res = predictor.train(&train_immos);
        assert!(!res.is_err());
        let res = predictor.predict(test_immos.iter_mut());
        assert!(!res.is_err());
        assert_eq!(test_immos[0].marktwert, Some(0.0));
    }

    proptest! {
        #[test]
        fn cant_train_twice(immos in full_immos(128)) {
            let mut predictor = SelectBlockPredictor::with_functions(ConstantCostFunction::with(1.0), LpVectorDissimilarity::manhattan());

            prop_assert!(predictor.train(&immos).is_ok());
            prop_assert!(predictor.train(&immos).is_err());
        }

        #[test]
        fn one_training_immo_predicts_const(train_immo in full_immo(), mut test_immo in full_immo()) {
            let mut predictor = SelectBlockPredictor::with_functions(ConstantCostFunction::with(1.0), LpVectorDissimilarity::manhattan());
            test_immo.plane_location = train_immo.plane_location;

            predictor.train(once(&train_immo))?;
            let mut to_predict = [test_immo];
            predictor.predict(to_predict.iter_mut())?;

            println!("Predicted: {}, Input: {}", to_predict[0].sqm_price().unwrap(), train_immo.sqm_price().unwrap());

            prop_assert!((train_immo.sqm_price().unwrap() - to_predict[0].sqm_price().unwrap()).abs() < 1e-9);
        }
    }
}
