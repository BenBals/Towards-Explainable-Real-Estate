//! This module contains the contraction algorithm for a Partition. See docs for information
use crate::partition::{BlockIdx, Partition};
use common::CostFunction;
use common::Immo;
use derive_more::{Add, From, Into, Sub};
use priority_queue::PriorityQueue;
use std::collections::{HashMap, HashSet};
use std::{
    borrow::Borrow,
    cmp::{Ord, Ordering},
};
use typed_index_collections::TiVec;

#[derive(Debug, Clone, Copy, From, Into, Add, Sub)]
struct Weight {
    value: f64,
}

impl PartialEq for Weight {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() < f64::EPSILON
    }
}

impl Eq for Weight {}

// Notice, that Less and Greater are swapped so the max-priority queue does work
impl PartialOrd for Weight {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(if self.value - other.value > f64::EPSILON {
            Ordering::Less
        } else if other.value - self.value > f64::EPSILON {
            Ordering::Greater
        } else {
            Ordering::Equal
        })
    }
}

impl Ord for Weight {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Weight {
    fn value(&self) -> f64 {
        self.value
    }
}

struct BlockGraph {
    edge_map: HashMap<(BlockIdx, BlockIdx), Weight>,
    adj_list: TiVec<BlockIdx, HashSet<BlockIdx>>,
    edge_queue: PriorityQueue<(BlockIdx, BlockIdx), Weight>,
    all_time_block_degrees: Vec<usize>,
}

impl BlockGraph {
    fn new() -> Self {
        Self {
            edge_map: HashMap::new(),
            adj_list: TiVec::new(),
            edge_queue: PriorityQueue::new(),
            all_time_block_degrees: vec![],
        }
    }

    fn with_partition<I, C>(part: &Partition<I, C>) -> BlockGraph
    where
        I: Borrow<Immo> + Sync,
        C: CostFunction + Sync,
    {
        let mut block_graph = Self::new();
        block_graph.adj_list = part.create_block_data_vec::<HashSet<BlockIdx>>(&HashSet::new());

        for block_idx in part.iter_blocks() {
            for entry_idx in part[block_idx].iter_entries() {
                for reachable_entry_idx in part[entry_idx].iter_reachable() {
                    let block_reachable_idx = part[reachable_entry_idx].block();
                    if block_idx == block_reachable_idx
                        || block_graph
                            .get_edge_weight(block_idx, block_reachable_idx)
                            .is_some()
                    {
                        continue;
                    }

                    block_graph.insert_or_update_edge(
                        block_idx,
                        block_reachable_idx,
                        part.cost_for_merging_blocks(block_idx, block_reachable_idx)
                            .into(),
                    );
                }
            }
        }

        for neighbor_set in &block_graph.adj_list {
            block_graph.all_time_block_degrees.push(neighbor_set.len());
        }

        block_graph
    }

    fn get_edge_weight(&self, idx1: BlockIdx, idx2: BlockIdx) -> Option<&Weight> {
        self.edge_map.get(&self.sort_indices(idx1, idx2))
    }

    fn insert_or_update_edge(&mut self, idx1: BlockIdx, idx2: BlockIdx, weight: Weight) {
        let (idx1, idx2) = self.sort_indices(idx1, idx2);
        self.adj_list[idx1].insert(idx2);
        self.adj_list[idx2].insert(idx1);
        self.edge_map.insert((idx1, idx2), weight);

        self.edge_queue.push((idx1, idx2), weight);
    }

    fn remove_edge(&mut self, idx1: BlockIdx, idx2: BlockIdx) {
        let (idx1, idx2) = self.sort_indices(idx1, idx2);
        self.adj_list[idx1].remove(&idx2);
        self.adj_list[idx2].remove(&idx1);
        self.edge_map.remove(&(idx1, idx2));

        self.edge_queue.remove(&(idx1, idx2));
    }

    fn sort_indices(&self, idx1: BlockIdx, idx2: BlockIdx) -> (BlockIdx, BlockIdx) {
        if idx1 > idx2 {
            (idx2, idx1)
        } else {
            (idx1, idx2)
        }
    }

    fn iter_neighbors(&self, block_idx: BlockIdx) -> impl Iterator<Item = &BlockIdx> {
        self.adj_list[block_idx].iter()
    }

    /// Merge two blocks in the graph
    /// Weights are recalculated via edge weights
    fn merge_blocks<I, C>(
        &mut self,
        block_idx1: BlockIdx,
        block_idx2: BlockIdx,
        part: &mut Partition<I, C>,
    ) where
        I: Borrow<Immo> + Sync,
        C: CostFunction + Sync,
    {
        let (new_block_idx, old_block_idx) = self.sort_indices(block_idx1, block_idx2);

        self.remove_edge(new_block_idx, old_block_idx);
        self.adj_list[new_block_idx] = self.adj_list[old_block_idx]
            .union(&self.adj_list[new_block_idx])
            .copied()
            .collect();

        self.adj_list[old_block_idx] = HashSet::new();

        self.all_time_block_degrees
            .push(self.adj_list[new_block_idx].len());

        // edit edges according whether neighbour was connected to any of the blocks before merging
        let neighboring_blocks: Vec<_> = self.iter_neighbors(new_block_idx).copied().collect();
        for neighbor in neighboring_blocks {
            // set weight accordingly to formula. status updates handle everything else
            let weight = self
                .get_edge_weight(neighbor, new_block_idx)
                .copied()
                .unwrap_or_else(|| part.cost_for_merging_blocks(neighbor, new_block_idx).into())
                + self
                    .get_edge_weight(neighbor, old_block_idx)
                    .copied()
                    .unwrap_or_else(|| {
                        part.cost_for_merging_blocks(neighbor, old_block_idx).into()
                    });

            self.remove_edge(old_block_idx, neighbor);
            self.insert_or_update_edge(new_block_idx, neighbor, weight);
        }

        part.merge_blocks(new_block_idx, old_block_idx);
    }
}

impl<I, C> Partition<I, C>
where
    I: Borrow<Immo> + Sync,
    C: CostFunction + Sync,
{
    /// runs the contraction algorithm and modifies the current partition
    /// This algorithm also works if the partition already consists of blocks that contain more than one immo.
    /// Afterwards there are no two epsilon-reachable blocks with negative cost for merging blocks
    pub fn contraction(&mut self) {
        let mut block_graph = BlockGraph::with_partition(self);

        log::info!("built graph");

        let mut contraction_count = 0;
        while let Some(((block_idx1, block_idx2), weight)) = block_graph.edge_queue.pop() {
            if weight.value() > -f64::EPSILON {
                break;
            }
            block_graph.merge_blocks(block_idx1, block_idx2, self);

            contraction_count += 1;
            if contraction_count % 1000 == 0 {
                log::info!("Total contractions so far: {}", contraction_count);
            }
        }

        self.remove_empty_blocks();

        log::info!("Number of contractions: {}", contraction_count);
        log::info!(
            "Maximum degree during contraction was {}",
            block_graph
                .all_time_block_degrees
                .iter()
                .max()
                .unwrap_or(&0)
        );
        log::info!(
            "Sum of degrees during contraction was {}",
            block_graph.all_time_block_degrees.iter().sum::<usize>()
        );
    }
}

#[cfg(test)]
mod tests {
    use test_helpers::*;

    use super::*;
    use crate::{are_blocks_connected, partition::Partition};
    use cost_functions::{ConstantCostFunction, DissimilarityCostFunction};
    use dissimilarities::SqmPriceDissimilarity;
    use proptest::prelude::*;

    #[test]
    fn test_weight_cmp() {
        // Note that 0.0 is supposed to be greater than 1 for priority_queue reasons
        assert!(Weight::from(0.0) > Weight::from(1.0));
        assert_eq!(Weight::from(0.0), Weight::from(0.0));
        assert!(Weight::from(2.0) < Weight::from(-1.0));
    }

    #[test]
    fn test_weight_value() {
        assert!((Weight::from(1.1).value() - 1.1).abs() <= f64::EPSILON);
    }

    #[test]
    fn test_part_cost_for_merging_blocks() {
        let immo1 = create_new_immo();
        let immo2 = create_new_immo();
        let immo3 = create_new_immo();
        let immos = vec![&immo1, &immo2, &immo3];
        let mut part = Partition::with_immos(3.0, ConstantCostFunction::negative(), immos).unwrap();

        let blocks: Vec<_> = part.iter_blocks().collect();

        assert!((part.cost_for_merging_blocks(blocks[0], blocks[1]) + 2.0).abs() < 1e-9);

        part.merge_blocks(blocks[0], blocks[1]);
        assert!((part.cost_for_merging_blocks(blocks[0], blocks[2]) + 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_block_graph_merge_block_entries() {
        let immo0 = create_new_immo_at(&[1.0, 0.0]);
        let immo1 = create_new_immo_at(&[-1.0, 0.0]);
        let immo2 = create_new_immo_at(&[0.0, 0.0]);
        let immo3 = create_new_immo_at(&[3.0, 0.0]);
        let immo4 = create_new_immo_at(&[-3.0, 0.0]);
        let immo5 = create_new_immo_at(&[0.0, 10.0]);
        let immos = vec![&immo0, &immo1, &immo2, &immo3, &immo4, &immo5];
        let mut part = Partition::with_immos(3.0, ConstantCostFunction::negative(), immos).unwrap();
        let blocks: Vec<_> = part.iter_blocks().collect();

        let mut block_graph = BlockGraph::with_partition(&part);
        block_graph.merge_blocks(blocks[0], blocks[1], &mut part);

        let (new_block_idx, old_block_idx) = block_graph.sort_indices(blocks[0], blocks[1]);

        assert_eq!(
            block_graph.adj_list[new_block_idx],
            vec![blocks[2], blocks[3], blocks[4]]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
        );
        assert_eq!(block_graph.adj_list[old_block_idx].len(), 0);
        assert!(block_graph
            .get_edge_weight(old_block_idx, new_block_idx)
            .is_none());
    }

    fn contracted_partition_for<C>(
        epsilon: f64,
        cost_function: C,
        immos: &[Immo],
    ) -> Partition<&Immo, C>
    where
        C: CostFunction + Sync,
    {
        let immo_refs: Vec<_> = immos.iter().collect();
        let mut part = Partition::with_immos(epsilon, cost_function, immo_refs).expect(
            "Could not create partition. This is probably an error in the calling test code.",
        );
        part.contraction();
        part
    }

    proptest! {
        #[test]
        fn test_clusters_cant_be_merged_after_contraction(immos in full_immos(128), epsilon in 0.0..2e6) {
            let part = contracted_partition_for(
                epsilon,
                DissimilarityCostFunction::with_immos(SqmPriceDissimilarity, immos.iter()),
                &immos,
            );

            prop_assert!(are_blocks_connected(&part));

            for block_idx1 in part.iter_blocks() {
                prop_assert_ne!(part[block_idx1].iter_entries().count(), 0);
                prop_assert!(part.cost_for_block(block_idx1) < 1e-3);

                for block_idx2 in part.iter_blocks() {
                    if block_idx1 != block_idx2 && !part[block_idx1].is_empty() && !part[block_idx2].is_empty() {

                        'outer_entry_loop: for entry_idx1 in part[block_idx1].iter_entries() {
                            for entry_idx2 in part[block_idx2].iter_entries() {
                                if part[entry_idx1].distance(&part[entry_idx2]) <= epsilon {
                                    prop_assert!(part.cost_for_merging_blocks(block_idx1, block_idx2) > -1e-9, "{:?}\n\n{:?} should be reachable from {:?}\n\n", part, entry_idx1, entry_idx2);
                                    break 'outer_entry_loop
                                }
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn contraction_constant_negative_cost_function(immos in full_immos(128), epsilon in 0.0..2e6) {
            let part = contracted_partition_for(
                epsilon,
                ConstantCostFunction::negative(),
                &immos,
            );

            for block_idx in part.iter_blocks() {
                for entry_idx in part[block_idx].iter_entries() {
                    for reachable_entry_idx in part[entry_idx].iter_reachable() {
                        prop_assert_eq!(part[entry_idx].block(), part[reachable_entry_idx].block());
                    }
                }
            }
        }

        #[test]
        fn contraction_constant_positive_cost_function(immos in full_immos(128), epsilon in 0.0..2e6) {
            let part = contracted_partition_for(
                epsilon,
                ConstantCostFunction::discrete(),
                &immos,
            );

            for block_idx in part.iter_blocks() {
                for entry_idx in part[block_idx].iter_entries() {
                    for reachable_entry_idx in part[entry_idx].iter_reachable() {
                        prop_assert_ne!(part[entry_idx].block(), part[reachable_entry_idx].block());
                    }
                }
            }
        }

        #[test]
        fn contraction_cost_non_positive(immos in full_immos(128), epsilon in 0.0..2e6) {
            let part = contracted_partition_for(
                epsilon,
                DissimilarityCostFunction::with_immos(SqmPriceDissimilarity, immos.iter()),
                &immos,
            );

            for block_idx in part.iter_blocks() {
                prop_assert!(part.cost_for_block(block_idx) < 1e-9);
            }
        }
    }
}
