use std::{borrow::Borrow, cmp::min};

use crate::partition::{BlockEntryIdx, BlockIdx, Partition};
use common::CostFunction;
use common::Immo;
use itertools::Itertools;
use rayon::prelude::*;
use typed_index_collections::TiVec;

const MIN_DIFFERENCE: f64 = -1e-4;

/// This enum is used to encode a change to the partition, which decreases the cost.
#[derive(Clone, Debug, Copy, Eq, PartialEq)]
enum Move {
    IntoEmptyBlock,
    Into(BlockIdx),
}

/// This struct gets used to determine the cut vertices of a block.
/// # Fields
/// * `global_time` : is the last time stamp where this was updated
#[derive(Clone, Debug, Default)]
struct CutVertexData {
    global_time: usize,
    dfs_low: u32,
    dfs_time: u32,
}

impl CutVertexData {
    fn with(global_time: usize, dfs_time: u32) -> Self {
        Self {
            global_time,
            dfs_time,
            dfs_low: dfs_time,
        }
    }
}

fn mark_cut_vertices_dfs<I: Borrow<Immo>, C: CostFunction + Sync>(
    part: &Partition<I, C>,
    data: &mut TiVec<BlockEntryIdx, CutVertexData>,
    entry_can_leave: &mut TiVec<BlockEntryIdx, bool>,
    cur_entry: BlockEntryIdx,
    global_time: usize,
    mut dfs_time: u32,
) -> u32 {
    dfs_time += 1;
    data[cur_entry] = CutVertexData::with(global_time, dfs_time);
    entry_can_leave[cur_entry] = true;

    for neighbour in part[cur_entry]
        .iter_reachable()
        .filter(|&o| part[o].block() == part[cur_entry].block())
    {
        if data[neighbour].global_time < global_time {
            dfs_time = mark_cut_vertices_dfs(
                part,
                data,
                entry_can_leave,
                neighbour,
                global_time,
                dfs_time,
            );

            entry_can_leave[cur_entry] &= data[neighbour].dfs_low < data[cur_entry].dfs_time;
        }
    }
    for neighbour in part[cur_entry]
        .iter_reachable()
        .filter(|&o| part[o].block() == part[cur_entry].block())
    {
        data[cur_entry].dfs_low = min(data[cur_entry].dfs_low, data[neighbour].dfs_low);
    }

    dfs_time
}

/// This function sets for all BlockEntries in `block_idx`, whether they are a cut vertex or not.
/// If a vertex is a cut vertex, then entry_can_leave of this entry is set to false.
/// Otherwise its set to true.
/// For all CutVertexData in the current block, `gloabl_time` of CutVertexData has to be *strictly*
/// less than the argument `global_time`.
fn mark_cut_vertices<I: Borrow<Immo>, C: CostFunction + Sync>(
    part: &Partition<I, C>,
    data: &mut TiVec<BlockEntryIdx, CutVertexData>,
    entry_can_leave: &mut TiVec<BlockEntryIdx, bool>,
    block_idx: BlockIdx,
    global_time: usize,
) {
    if let Some(start) = part[block_idx].iter_entries().next() {
        let mut dfs_time = 0;
        data[start] = CutVertexData::with(global_time, dfs_time);

        let mut started_dfs = 0;
        for neighbour in part[start]
            .iter_reachable()
            .filter(|&o| part[o].block() == block_idx)
        {
            if data[neighbour].global_time < global_time {
                started_dfs += 1;
                dfs_time = mark_cut_vertices_dfs(
                    part,
                    data,
                    entry_can_leave,
                    neighbour,
                    global_time,
                    dfs_time,
                );
            }
        }

        entry_can_leave[start] = started_dfs <= 1;
    }
}

fn find_better_block<I, C>(part: &Partition<I, C>, entry_idx: BlockEntryIdx) -> Option<Move>
where
    I: Borrow<Immo> + Sync,
    C: CostFunction + Sync,
{
    let remove_cost = part.cost_for_removing_entry_from_current_block(entry_idx);
    if remove_cost < MIN_DIFFERENCE {
        Some(Move::IntoEmptyBlock)
    } else {
        let current_block = part[entry_idx].block();
        part[entry_idx]
            .iter_reachable()
            .map(|r| part[r].block())
            .filter(|&b| b != current_block)
            .unique()
            .find(|&other_idx| {
                remove_cost + part.cost_for_adding_entry_to_block(entry_idx, other_idx)
                    < MIN_DIFFERENCE
            })
            .map(Move::Into)
    }
}

/// Checks for all entries in this block, whether there is a way to move this entry to decrease the costs.
/// Every suggested move leaves all clusters valid
/// # Returns
/// * None if this block is 1-stable.
/// * Some((entry_idx, Move::IntoEmptyBlock)) if creating a new block for `entry_idx` would decrease overall cost.
/// * Some((entry_idx, Move::Into(block_idx))) if moving `entry_idx` into `block_idx` decreases costs.
fn find_unstable_entry_in_block<I, C>(
    part: &Partition<I, C>,
    block_idx: BlockIdx,
    entry_can_leave: &TiVec<BlockEntryIdx, bool>,
) -> Option<(BlockEntryIdx, Move)>
where
    I: Borrow<Immo> + Sync,
    C: CostFunction + Sync + Send,
{
    // check that entry can leave
    part[block_idx]
        .iter_par_entries()
        .filter(|&e| entry_can_leave[e])
        .map(|e| find_better_block(part, e).map(|b| (e, b)))
        .find_any(|o| o.is_some())
        .flatten()
}

/// This function changes `part` to be 1-stable.
/// It only does changes to `part` which decrease the cost.
/// # Runtime
/// We don't have a good theoretical bound, but its quite fast.
pub fn make_one_stable<I, C>(part: &mut Partition<I, C>)
where
    I: Borrow<Immo> + Sync,
    C: CostFunction + Send + Sync,
{
    let mut blocks_stable = part.create_block_data_vec(&false);
    let mut unstable_blocks = Vec::new();

    let mut entry_can_leave = part.create_entry_data_vec(&false);
    let mut cut_vertex_data = part.create_entry_data_vec(&CutVertexData::default());
    let mut global_time = 1;
    for (block_idx, _) in blocks_stable.iter_mut_enumerated() {
        unstable_blocks.push(block_idx);

        mark_cut_vertices(
            part,
            &mut cut_vertex_data,
            &mut entry_can_leave,
            block_idx,
            global_time,
        )
    }

    let mut moved_empty = 0;
    let mut moved_other = 0;
    while let Some(&block) = unstable_blocks.last() {
        match find_unstable_entry_in_block(part, block, &entry_can_leave) {
            None => {
                blocks_stable[block] = true;
                unstable_blocks.pop();
            }
            Some((entry, instruction)) => {
                let new_block = match instruction {
                    Move::IntoEmptyBlock => {
                        moved_empty += 1;
                        if moved_empty % 1000 == 0 {
                            log::info!("Moved {} points to a new singleton cluster", moved_empty);
                        }
                        blocks_stable.push(false);
                        let new = part.create_new_block();
                        unstable_blocks.push(new);
                        new
                    }
                    Move::Into(block) => {
                        moved_other += 1;
                        if moved_other % 1000 == 0 {
                            log::info!("Moved {} points to a differnt cluster", moved_other);
                        }
                        block
                    }
                };

                let old_block = part[entry].block();
                part.move_entry_to(entry, new_block);

                // update cut vertices
                global_time += 1;
                mark_cut_vertices(
                    part,
                    &mut cut_vertex_data,
                    &mut entry_can_leave,
                    new_block,
                    global_time,
                );
                mark_cut_vertices(
                    part,
                    &mut cut_vertex_data,
                    &mut entry_can_leave,
                    old_block,
                    global_time,
                );

                // mark all neighboring blocks as instable
                for neighbour_block in part[new_block]
                    .iter_entries()
                    .flat_map(|e| part[e].iter_reachable().map(|r| part[r].block()))
                    .chain(
                        part[new_block]
                            .iter_entries()
                            .flat_map(|e| part[e].iter_reachable().map(|r| part[r].block())),
                    )
                {
                    if blocks_stable[neighbour_block] {
                        blocks_stable[neighbour_block] = false;
                        unstable_blocks.push(neighbour_block);
                    }
                }
            }
        }
    }

    log::info!("Moved {} points to a different cluster", moved_other);
    log::info!("Moved {} points to a new singleton cluster", moved_empty);
}

#[cfg(test)]
mod tests {
    use crate::partition::Partition;
    use proptest::prelude::*;
    use test_helpers::*;

    use super::*;
    use crate::are_blocks_connected;
    use cost_functions::{ClosureCostFunction, ConstantCostFunction};

    #[test]
    fn mark_cut_vertices_marks_no_vertices_for_1_entries() {
        let mut part = Partition::new(1.0, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);

        let block = part.create_new_block();
        let a_idx = part.add_new_entry_to_block(&immo_a, block).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());

        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
        assert!(entry_can_leave[a_idx]);
    }
    #[test]
    fn mark_cut_vertices_marks_no_vertices_for_2_entries() {
        let mut part = Partition::new(1.0, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[0.0, 1.0]);

        let block = part.create_new_block();
        let a_idx = part.add_new_entry_to_block(&immo_a, block).unwrap();
        let b_idx = part.add_new_entry_to_block(&immo_b, block).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());

        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
        assert!(entry_can_leave[a_idx]);
        assert!(entry_can_leave[b_idx]);
    }

    #[test]
    fn mark_cut_vertices_marks_correct_vertex_for_3_path() {
        let mut part = Partition::new(1.0, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[0.0, 1.0]);
        let immo_c = create_new_immo_at(&[0.0, 2.0]);

        let block = part.create_new_block();
        let a_idx = part.add_new_entry_to_block(&immo_a, block).unwrap();
        let b_idx = part.add_new_entry_to_block(&immo_b, block).unwrap();
        let c_idx = part.add_new_entry_to_block(&immo_c, block).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());

        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
        assert!(entry_can_leave[a_idx]);
        assert!(!entry_can_leave[b_idx]);
        assert!(entry_can_leave[c_idx]);
    }

    #[test]
    fn mark_cut_vertices_marks_correct_vertex_for_3_cycle() {
        let mut part = Partition::new(1.0, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[0.0, 1.0]);
        let immo_c = create_new_immo_at(&[0.5, 0.5]);

        let block = part.create_new_block();
        let a_idx = part.add_new_entry_to_block(&immo_a, block).unwrap();
        let b_idx = part.add_new_entry_to_block(&immo_b, block).unwrap();
        let c_idx = part.add_new_entry_to_block(&immo_c, block).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());

        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
        assert!(entry_can_leave[a_idx]);
        assert!(entry_can_leave[b_idx]);
        assert!(entry_can_leave[c_idx]);
    }

    // todo: add quickcheck for mark_cut_vertices

    #[test]
    fn find_unstable_entry_in_block_for_stable_block_is_none() {
        let mut part = Partition::new(1.0, ConstantCostFunction::negative());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[0.0, 1.0]);
        let immo_c = create_new_immo_at(&[0.5, 0.5]);

        let block = part.create_new_block();
        part.add_new_entry_to_block(&immo_a, block).unwrap();
        part.add_new_entry_to_block(&immo_b, block).unwrap();
        part.add_new_entry_to_block(&immo_c, block).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());
        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
        assert_eq!(
            find_unstable_entry_in_block(&part, block, &entry_can_leave),
            None
        );
    }

    #[test]
    fn find_unstable_entry_in_block_finds_entry_with_positive_contribution() {
        // b and c want to be in one block and a wants to be alone
        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[0.0, 1.0]);
        let immo_c = create_new_immo_at(&[0.5, 0.5]);

        let mut part = Partition::new(
            1.0,
            ClosureCostFunction::new(|a, b| {
                if a == b {
                    0f64
                } else if a != &immo_a && b != &immo_a {
                    -1f64
                } else {
                    1f64
                }
            }),
        );

        let block = part.create_new_block();
        let a_idx = part.add_new_entry_to_block(&immo_a, block).unwrap();
        part.add_new_entry_to_block(&immo_b, block).unwrap();
        part.add_new_entry_to_block(&immo_c, block).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());
        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
        assert_eq!(
            find_unstable_entry_in_block(&part, block, &entry_can_leave),
            Some((a_idx, Move::IntoEmptyBlock))
        );
    }

    #[test]
    fn find_unstable_entry_in_block_finds_entry_which_wants_to_be_in_another_block() {
        // initially a and b are in one block and c in a singleton
        // but a rather wants to be in a block with c
        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[0.0, 1.0]);
        let immo_c = create_new_immo_at(&[0.5, 0.5]);

        let mut part = Partition::new(
            1.0,
            ClosureCostFunction::new(|a, b| {
                if a == b {
                    0f64
                } else if (a == &immo_a && b == &immo_b) || (a == &immo_b && b == &immo_a) {
                    -1f64
                } else if (a == &immo_a && b == &immo_c) || (a == &immo_c && b == &immo_a) {
                    -10f64
                } else if (a == &immo_b && b == &immo_c) || (a == &immo_c && b == &immo_b) {
                    1f64
                } else {
                    unreachable!()
                }
            }),
        );

        let block_ab = part.create_new_block();
        let block_c = part.create_new_block();
        let a_idx = part.add_new_entry_to_block(&immo_a, block_ab).unwrap();
        part.add_new_entry_to_block(&immo_b, block_ab).unwrap();
        part.add_new_entry_to_block(&immo_c, block_c).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());
        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block_ab, 1);
        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block_c, 1);
        assert_eq!(
            find_unstable_entry_in_block(&part, block_ab, &entry_can_leave),
            Some((a_idx, Move::Into(block_c)))
        );
    }

    #[test]
    fn find_unstable_entry_in_block_does_not_remove_cut_vertices() {
        // lie on path and b wants to be alone, but connects a and c,
        // which really want to be in one cluster, preventing b from leaving
        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[0.0, 1.0]);
        let immo_c = create_new_immo_at(&[0.0, 2.0]);

        let mut part = Partition::new(
            1.0,
            ClosureCostFunction::new(|a, b| {
                if a == b {
                    0f64
                } else if a == &immo_b || b == &immo_b {
                    1f64
                } else {
                    -10f64
                }
            }),
        );

        let block = part.create_new_block();
        part.add_new_entry_to_block(&immo_a, block).unwrap();
        part.add_new_entry_to_block(&immo_b, block).unwrap();
        part.add_new_entry_to_block(&immo_c, block).unwrap();

        let mut entry_can_leave = part.create_entry_data_vec(&false);
        let mut cut_data = part.create_entry_data_vec(&CutVertexData::default());
        mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
        assert_eq!(
            find_unstable_entry_in_block(&part, block, &entry_can_leave),
            None
        );
    }

    proptest! {
        // creates a clustering and checks that gets one_stable
        #[test]
        fn test_random_instance(points in prop::collection::vec((0..255, 0..255), 0..128)) {
            let immos: Vec<_> = points
                .into_iter()
                .map(|a| create_new_immo_at(&[a.0 as f64, a.1 as f64]))
                .collect();

            let mut part = Partition::new(1000f64, ClosureCostFunction::new(|a, b| {
                if a == b {
                    0.0
                } else {
                    let dist = a.plane_distance(b).unwrap();
                    -10.0 + 30.0 / (1.0 + (-dist / 10.0 + 10.0).exp())
                }

            }));

            for immo in &immos {
                let new_block = part.create_new_block();
                part.add_new_entry_to_block(immo, new_block).unwrap();
            }

            make_one_stable(&mut part);

            let mut entry_can_leave = part.create_entry_data_vec(&false);
            let mut cut_data = part.create_entry_data_vec(& CutVertexData::default());

            let result = part.iter_blocks().all(|block| {
                mark_cut_vertices(&part, &mut cut_data, &mut entry_can_leave, block, 1);
                find_unstable_entry_in_block(&part, block, &entry_can_leave).is_none()
            }) && are_blocks_connected(&part);
            prop_assert!(result);
        }
    }
}
