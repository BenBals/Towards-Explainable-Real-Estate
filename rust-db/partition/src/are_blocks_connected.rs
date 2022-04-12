use std::borrow::Borrow;

use common::{CostFunction, Immo};

use crate::Partition;

/// This function checks whether all blocks in this partition are connected.
/// Connected here means, that each entries are pairwise reachable
/// # Runtime
/// O(n) where n is the number of entries in the partition.
pub fn are_blocks_connected<I: Borrow<Immo>, C: CostFunction>(part: &Partition<I, C>) -> bool {
    let mut visited = part.create_entry_data_vec(&false);

    part.iter_blocks().all(|b_idx| {
        // check connectedness using dfs
        if let Some(start) = part[b_idx].iter_entries().next() {
            visited[start] = true;
            let mut stack = vec![start];

            while let Some(cur) = stack.pop() {
                for next in part[cur].iter_reachable() {
                    if !visited[next] && part[next].block() == b_idx {
                        visited[next] = true;
                        stack.push(next);
                    }
                }
            }
        }

        // check all were reached
        part[b_idx].iter_entries().all(|e| visited[e])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cost_functions::ConstantCostFunction;
    use test_helpers::*;

    #[test]
    fn are_blocks_connected_test_block_and_partition_connected() {
        let mut part = Partition::new(1f64, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[1.0, 0.0]);
        let immo_c = create_new_immo_at(&[1.0, 1.0]);

        let block = part.create_new_block();
        part.add_new_entry_to_block(&immo_a, block).unwrap();
        part.add_new_entry_to_block(&immo_b, block).unwrap();
        part.add_new_entry_to_block(&immo_c, block).unwrap();
        assert!(are_blocks_connected(&part));
    }
    #[test]
    fn are_blocks_connected_test_blocks_and_partition_not_connected() {
        let mut part = Partition::new(1f64, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_c = create_new_immo_at(&[1.0, 1.0]);

        let block = part.create_new_block();
        part.add_new_entry_to_block(&immo_a, block).unwrap();
        part.add_new_entry_to_block(&immo_c, block).unwrap();
        assert!(!are_blocks_connected(&part));
    }

    #[test]
    fn are_blocks_connected_test_block_connected_partition_not() {
        let mut part = Partition::new(1f64, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[1.0, 0.0]);
        let immo_c = create_new_immo_at(&[3.0, 0.0]);

        let block = part.create_new_block();
        let block_2 = part.create_new_block();
        part.add_new_entry_to_block(&immo_a, block).unwrap();
        part.add_new_entry_to_block(&immo_b, block).unwrap();
        part.add_new_entry_to_block(&immo_c, block_2).unwrap();
        assert!(are_blocks_connected(&part));
    }
    #[test]
    fn are_blocks_connected_test_partition_connected_blocks_not() {
        let mut part = Partition::new(1f64, ConstantCostFunction::discrete());

        let immo_a = create_new_immo_at(&[0.0, 0.0]);
        let immo_b = create_new_immo_at(&[1.0, 0.0]);
        let immo_c = create_new_immo_at(&[1.0, 1.0]);

        let block = part.create_new_block();
        let block_2 = part.create_new_block();
        part.add_new_entry_to_block(&immo_a, block).unwrap();
        part.add_new_entry_to_block(&immo_b, block_2).unwrap();
        part.add_new_entry_to_block(&immo_c, block).unwrap();
        assert!(!are_blocks_connected(&part));
    }
}
